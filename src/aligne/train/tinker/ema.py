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
deps (safetensors + torch only) and is unit-testable; the CLI orchestrates
download → PEFT-convert → average.

CLI::

    aligne-ema --log-dir <run-log-dir> --last-n 4 --base-model Qwen/Qwen3-8B --out ./ema_adapter
    aligne-ema --checkpoints tinker://a tinker://b ... --base-model Qwen/Qwen3-8B --out ./ema_adapter
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


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


def resolve_checkpoints(args: argparse.Namespace) -> list[str]:
    """Resolve the list of tinker:// adapter paths to average.

    Either explicit ``--checkpoints`` or the last ``--last-n`` sampler
    checkpoints from a run's ``checkpoints.jsonl`` (via ``--log-dir``).
    """
    if args.checkpoints:
        return list(args.checkpoints)
    from tinker_cookbook import checkpoint_utils

    records = checkpoint_utils.load_checkpoints_file(args.log_dir)
    paths = [r.sampler_path for r in records if r.sampler_path]
    if not paths:
        raise SystemExit(f"no sampler checkpoints found in {args.log_dir}/checkpoints.jsonl")
    return paths[-args.last_n :]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Average (EMA) the last N LoRA checkpoints.")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--log-dir", help="run log dir containing checkpoints.jsonl")
    src.add_argument(
        "--checkpoints", nargs="+", help="explicit tinker:// adapter paths to average"
    )
    p.add_argument("--last-n", type=int, default=4, help="how many trailing checkpoints (with --log-dir)")
    p.add_argument("--base-model", default="Qwen/Qwen3-8B", help="base model for PEFT conversion")
    p.add_argument("--out", required=True, help="output PEFT adapter dir")
    p.add_argument(
        "--work-dir",
        default=None,
        help="scratch dir for per-checkpoint downloads (default: <out>/_ckpts)",
    )
    p.add_argument("--base-url", default=None, help="override Tinker service URL")
    p.add_argument(
        "--vllm-safe",
        action="store_true",
        help="strip lm_head/embed_tokens from the averaged adapter so vLLM can serve "
        "it (Tinker trains all-linear, which vLLM refuses). Attn+MLP LoRA is kept.",
    )
    return p


def run(args: argparse.Namespace) -> None:
    """Download → PEFT-convert → average the resolved checkpoints (heavy)."""
    from tinker_cookbook import weights

    ckpts = resolve_checkpoints(args)
    work = Path(args.work_dir or (Path(args.out) / "_ckpts"))
    work.mkdir(parents=True, exist_ok=True)
    print(f"[aligne-ema] averaging {len(ckpts)} checkpoints into {args.out}")
    peft_dirs: list[str] = []
    for i, ck in enumerate(ckpts):
        raw = work / f"raw_{i}"
        peft = work / f"peft_{i}"
        print(f"  [{i+1}/{len(ckpts)}] {ck}")
        weights.download(tinker_path=ck, output_dir=str(raw), base_url=args.base_url)
        weights.build_lora_adapter(
            base_model=args.base_model, adapter_path=str(raw), output_path=str(peft)
        )
        peft_dirs.append(str(peft))
    strip = VLLM_UNSERVABLE if args.vllm_safe else ()
    out = average_adapter_safetensors(peft_dirs, args.out, strip_modules=strip)
    manifest = {"base_model": args.base_model, "n": len(ckpts),
                "checkpoints": ckpts, "vllm_safe": args.vllm_safe}
    (Path(out) / "ema_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[aligne-ema] wrote averaged adapter to {out}")


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
