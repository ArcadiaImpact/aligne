"""Tinker checkpoint -> local vLLM-servable PEFT adapter.

Checkpoints in the training drivers are ``tinker://`` pointers (never weights).
External eval harnesses (vLLM on a RunPod pod, HF pipelines) need the adapter
*bytes*, so this module materializes them: download the checkpoint archive via
``tinker_cookbook.weights``, convert to a HF PEFT adapter dir, and by default
strip it vLLM-safe.

Three Tinker/vLLM gotchas encoded here (each cost a debugging round in the
risk-averse-constitutions study):

- The archive endpoint accepts **only** ``sampler_weights/*`` paths;
  ``weights/*`` (trainable state) 400s.
- Archives are built **lazily server-side** on first request and can take
  >10 min; the SDK's request timeout is shorter, so the first call usually
  times out. :func:`run_convert` retries until the cached archive is ready.
- Tinker trains all-linear LoRA including ``lm_head``/``embed_tokens``, which
  **vLLM refuses to serve**. ``vllm_safe=True`` drops those tensors (attn+MLP
  LoRA kept — a near-faithful adapter; same policy as ``aligne train ema
  --vllm-safe``).

Library entry point::

    await run_convert(ConvertConfig(checkpoint=..., base_model=..., out=...))

:func:`strip_vllm_unservable` (pure torch+safetensors) and :func:`download_peft`
(idempotent conversion; needs ``tinker_cookbook``) are plain helpers — the
latter is the same conversion the scimt perturbation probe drives before
noising an adapter. Heavy imports (``tinker_cookbook``, ``torch``,
``safetensors``) are LAZY inside the functions, so importing this module does
not require the ``tinker`` extra. The CLI adapter lives in
:mod:`aligne.train.tinker.cli` (``aligne train convert``).
"""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
from pathlib import Path

from .configs import ConvertConfig, describe
from .results import ConvertResult

log = logging.getLogger(__name__)

# vLLM cannot serve lm_head/embed_tokens LoRA (Tinker trains all-linear); these
# substrings mark the tensors/modules to drop to make an adapter servable.
_STRIP_MARKERS = ("lm_head", "embed_tokens", "unembed")


def strip_vllm_unservable(adapter_dir: str | Path) -> int:
    """Drop lm_head/embed LoRA tensors from a PEFT adapter dir, in place.

    Returns the number of tensors removed. Needs the ``torch`` extra
    (safetensors). Also removes the stripped modules from
    ``adapter_config.json``'s ``target_modules``.
    """
    from safetensors.torch import load_file, save_file

    adapter_dir = Path(adapter_dir)
    weights_file = adapter_dir / "adapter_model.safetensors"
    tensors = load_file(str(weights_file))
    kept = {k: v for k, v in tensors.items() if not any(m in k for m in _STRIP_MARKERS)}
    removed = len(tensors) - len(kept)
    if removed:
        save_file(kept, str(weights_file))
    cfg_file = adapter_dir / "adapter_config.json"
    if cfg_file.exists():
        cfg = json.loads(cfg_file.read_text())
        tm = cfg.get("target_modules")
        if isinstance(tm, list):
            cfg["target_modules"] = [m for m in tm if not any(s in m for s in _STRIP_MARKERS)]
            cfg_file.write_text(json.dumps(cfg, indent=2) + "\n")
    return removed


def download_peft(tinker_path: str, base_model: str, out_dir: str, *,
                  overwrite: bool = False) -> str:
    """Tinker checkpoint -> PEFT adapter dir (adapter_model.safetensors + config).

    GPU-heavy for MoE expert-LoRA expansion; run it *before* a serving engine
    grabs the device (else it can OOM). Needs `tinker_cookbook` + TINKER_API_KEY.

    Idempotent / resumable: if ``out_dir`` already holds a built adapter this
    returns immediately (no download, no `tinker_cookbook` import) unless
    ``overwrite=True``. `tinker_cookbook.weights.build_lora_adapter` refuses an
    ``output_path`` that already exists, so any *partial* ``out_dir``/``_raw``
    from a crashed run is cleared before rebuilding.
    """
    out = Path(out_dir)
    raw = Path(out_dir + "_raw")
    if (out / "adapter_model.safetensors").exists() and not overwrite:
        return out_dir
    # Clear partial outputs from a prior crashed run (build_lora_adapter requires
    # a non-existent output_path; a stale _raw could also be incomplete).
    for d in (out, raw):
        if d.exists():
            shutil.rmtree(d)
    from tinker_cookbook import weights
    weights.download(tinker_path=tinker_path, output_dir=str(raw))
    weights.build_lora_adapter(base_model=base_model, adapter_path=str(raw), output_path=out_dir)
    return out_dir


def _download_and_build(checkpoint: str, base_model: str, out_dir: Path, work_dir: Path) -> None:
    from tinker_cookbook import weights

    # tinker_cookbook refuses existing output paths; a stale work dir from a
    # failed attempt would poison every retry — clean both.
    for stale in (out_dir, work_dir):
        if stale.exists():
            shutil.rmtree(stale)
    work_dir.mkdir(parents=True)
    weights.download(tinker_path=checkpoint, output_dir=str(work_dir / "raw"))
    weights.build_lora_adapter(
        base_model=base_model,
        adapter_path=str(work_dir / "raw"),
        output_path=str(out_dir),
    )


async def run_convert(cfg: ConvertConfig) -> ConvertResult:
    """Materialize ``cfg.checkpoint`` (a ``tinker://...sampler_weights/...`` URI)
    as a local PEFT adapter dir; return the adapter path + provenance.

    Blocking SDK calls run in a thread so callers can convert several arms
    concurrently from one event loop. The ``sampler_weights`` precondition is
    enforced by :class:`ConvertConfig` (the archive endpoint rejects
    trainable-state paths).
    """
    out_dir = Path(cfg.out)
    work_dir = Path(str(out_dir) + "_work")
    log.info("convert: %s", describe(cfg))
    last: Exception | None = None
    for attempt in range(cfg.attempts):
        try:
            await asyncio.to_thread(
                _download_and_build, cfg.checkpoint, cfg.base_model, out_dir, work_dir
            )
            break
        except Exception as e:  # archive still building server-side, usually
            last = e
            log.info("convert: attempt %d/%d not ready (%s); waiting %.0fs",
                     attempt + 1, cfg.attempts, e, cfg.wait_s)
            await asyncio.sleep(cfg.wait_s)
    else:
        raise RuntimeError(
            f"convert of {cfg.checkpoint} failed after {cfg.attempts} attempts: {last}"
        )
    shutil.rmtree(work_dir, ignore_errors=True)
    removed = 0
    if cfg.vllm_safe:
        removed = strip_vllm_unservable(out_dir)
        (out_dir / "REMAP.json").write_text(
            json.dumps(
                {"checkpoint": cfg.checkpoint, "base_model": cfg.base_model,
                 "vllm_safe": True, "stripped_tensors": removed},
                indent=2,
            )
            + "\n"
        )
    return ConvertResult(
        adapter_dir=str(out_dir), checkpoint=cfg.checkpoint,
        base_model=cfg.base_model, vllm_safe=cfg.vllm_safe, stripped_tensors=removed,
    )
