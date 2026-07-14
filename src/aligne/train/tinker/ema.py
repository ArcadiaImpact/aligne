"""Checkpoint averaging (EMA / "model souping") for LoRA adapters.

The naturalness thesis includes: *averaging the last few checkpoints reduces
mode collapse vs. taking only the last checkpoint*. This driver downloads the
last ``N`` LoRA adapters from a Tinker run, **element-wise averages** them, and
writes a single PEFT adapter dir servable with vLLM (``--lora-modules``) or
re-uploadable.

Why element-wise averaging is valid here: the adapters come from *consecutive
saves of one run*, so they evolve continuously (small SGD steps) and are aligned
in their latent rank space — averaging ``A`` and ``B`` per layer is the standard
"LoRA soup" and is a good approximation of averaging the effective ``ΔW = B·A``.
(For adapters from *different* runs this would be unsound due to LoRA's
rotational ambiguity; that is not the EMA-over-a-single-run case.)

The pure averaging core :func:`average_adapter_safetensors` has no Tinker/HF
deps (safetensors + torch only) and is unit-testable; :func:`run_ema`
orchestrates download → PEFT-convert → average.

Library entry point::

    await run_ema(EMAConfig(base_model=..., out=..., log_dir=...))

The CLI adapter lives in :mod:`aligne.train.tinker.cli` (``aligne train ema``).
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

from .configs import EMAConfig, describe
from .results import EMAResult

log = logging.getLogger(__name__)


# Tinker trains LoRA with target_modules="all-linear", which includes the output
# embedding (unembed_tokens -> "lm_head" after PEFT conversion). vLLM refuses to
# serve a LoRA targeting lm_head/embed_tokens, so for vLLM eval we strip those
# (the attn+MLP modules remain — a near-faithful adapter).
VLLM_UNSERVABLE = ("lm_head", "embed_tokens", "unembed")


def average_adapter_safetensors(
    adapter_dirs: list[str], out_dir: str, strip_modules: tuple = ()
) -> str:
    """Element-wise average the PEFT ``adapter_model.safetensors`` of N adapters.

    Each input dir must be a PEFT LoRA adapter (``adapter_model.safetensors`` +
    ``adapter_config.json``) with identical tensor keys and shapes. Writes the
    averaged tensors to ``out_dir/adapter_model.safetensors`` and copies
    ``adapter_config.json`` from the first adapter.

    ``strip_modules``: substrings; any tensor whose key contains one is dropped,
    and the name is removed from the config's ``target_modules`` (used to make a
    Tinker ``all-linear`` adapter vLLM-servable — see ``VLLM_UNSERVABLE``).

    Returns ``out_dir``. Pure (safetensors + torch + json/stdlib only).
    """
    import json

    import torch
    from safetensors.torch import load_file, save_file

    if len(adapter_dirs) < 1:
        raise ValueError("need at least one adapter to average")
    states = [load_file(str(Path(d) / "adapter_model.safetensors")) for d in adapter_dirs]
    keys = set(states[0])
    for i, s in enumerate(states[1:], 1):
        if set(s) != keys:
            raise ValueError(
                f"adapter {adapter_dirs[i]} key set differs from {adapter_dirs[0]} "
                f"(symmetric diff: {keys ^ set(s)})"
            )
    kept_keys = [k for k in keys if not any(m in k for m in strip_modules)]
    if not kept_keys:
        raise ValueError(f"strip_modules {strip_modules} removed every tensor")
    n = len(states)
    averaged = {}
    for k in kept_keys:
        acc = states[0][k].to(torch.float32).clone()
        for s in states[1:]:
            if s[k].shape != states[0][k].shape:
                raise ValueError(f"shape mismatch for {k}: {s[k].shape} vs {states[0][k].shape}")
            acc += s[k].to(torch.float32)
        acc /= n
        averaged[k] = acc.to(states[0][k].dtype)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    save_file(averaged, str(out / "adapter_model.safetensors"))
    cfg_src = Path(adapter_dirs[0]) / "adapter_config.json"
    if cfg_src.exists():
        cfg = json.loads(cfg_src.read_text())
        tm = cfg.get("target_modules")
        if strip_modules and isinstance(tm, list):
            cfg["target_modules"] = [m for m in tm if not any(s in m for s in strip_modules)]
        (out / "adapter_config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    return str(out)


def resolve_checkpoints(cfg: EMAConfig) -> list[str]:
    """Resolve the list of tinker:// adapter paths to average.

    Either explicit ``cfg.checkpoints`` or the last ``cfg.last_n`` sampler
    checkpoints from a run's ``checkpoints.jsonl`` (via ``cfg.log_dir``).
    """
    if cfg.checkpoints:
        return list(cfg.checkpoints)
    from tinker_cookbook import checkpoint_utils

    records = checkpoint_utils.load_checkpoints_file(cfg.log_dir)
    paths = [r.sampler_path for r in records if r.sampler_path]
    if not paths:
        raise ValueError(
            f"no sampler checkpoints found in {cfg.log_dir}/checkpoints.jsonl"
        )
    return paths[-cfg.last_n:]


async def run_ema(cfg: EMAConfig) -> EMAResult:
    """Download → PEFT-convert → average the resolved checkpoints (heavy);
    returns the averaged adapter dir + provenance. The cookbook's
    download/convert calls are blocking, so they run on a worker thread."""
    from tinker_cookbook import weights

    ckpts = resolve_checkpoints(cfg)
    work = Path(cfg.work_dir or (Path(cfg.out) / "_ckpts"))
    work.mkdir(parents=True, exist_ok=True)
    log.info("ema: averaging %d checkpoints — %s", len(ckpts), describe(cfg))

    def fetch_and_convert(i: int, ck: str) -> str:
        raw = work / f"raw_{i}"
        peft = work / f"peft_{i}"
        log.info("ema: [%d/%d] %s", i + 1, len(ckpts), ck)
        weights.download(tinker_path=ck, output_dir=str(raw), base_url=cfg.base_url)
        weights.build_lora_adapter(
            base_model=cfg.base_model, adapter_path=str(raw), output_path=str(peft)
        )
        return str(peft)

    peft_dirs = [
        await asyncio.to_thread(fetch_and_convert, i, ck)
        for i, ck in enumerate(ckpts)
    ]
    strip = VLLM_UNSERVABLE if cfg.vllm_safe else ()
    out = average_adapter_safetensors(peft_dirs, cfg.out, strip_modules=strip)
    manifest = {"base_model": cfg.base_model, "n": len(ckpts),
                "checkpoints": ckpts, "vllm_safe": cfg.vllm_safe}
    (Path(out) / "ema_manifest.json").write_text(json.dumps(manifest, indent=2))
    log.info("ema: wrote averaged adapter to %s", out)
    return EMAResult(
        adapter_dir=out, base_model=cfg.base_model,
        checkpoints=tuple(ckpts), vllm_safe=cfg.vllm_safe,
    )
