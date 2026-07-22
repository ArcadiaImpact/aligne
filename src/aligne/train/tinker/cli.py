"""CLI adapters for the Tinker training drivers.

The ONLY argparse in ``aligne.train``: each ``main_*`` parses flags, builds
the driver's config dataclass (:mod:`aligne.train.tinker.configs`), and runs
the async library entry point. Parsers are generated from the dataclasses —
one ``--flag`` per field, defaults owned by the dataclass (flags default to
``argparse.SUPPRESS``), so config and CLI can never drift apart.

``--config FILE`` loads a JSON config; explicit flags override its values.
``--smoke`` applies the driver's tiny-run preset via ``cfg.smoke()``.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import logging

from .configs import (
    ConvertConfig,
    DPOConfig,
    EMAConfig,
    ForwardKLDistillConfig,
    ReverseKLDistillConfig,
    SFTConfig,
    UnlearnConfig,
)

_TYPES = {"int": int, "int | None": int, "float": float, "float | None": float}


def _add_config_args(
    p: argparse.ArgumentParser,
    cls,
    *,
    smoke: bool = True,
    skip: tuple[str, ...] = (),
) -> argparse.ArgumentParser:
    """Add one ``--flag`` per config field (SUPPRESS default, so the
    namespace only holds what the user actually passed)."""
    for f in dataclasses.fields(cls):
        if f.name in skip:
            continue
        flag = "--" + f.name.replace("_", "-")
        if isinstance(f.default, bool):
            p.add_argument(flag, action="store_true", default=argparse.SUPPRESS)
        else:
            p.add_argument(
                flag, default=argparse.SUPPRESS, type=_TYPES.get(f.type, str)
            )
    if smoke:
        p.add_argument("--smoke", action="store_true",
                       help="tiny run: rank 8 + small batch/steps")
    p.add_argument("--config", default=None,
                   help="JSON config file; flags override its values")
    return p


def _config_from_args(cls, args: argparse.Namespace):
    """Namespace → config dataclass (the only Namespace→library crossing)."""
    values = {
        k: v for k, v in vars(args).items() if k not in ("smoke", "config")
    }
    try:
        if getattr(args, "config", None):
            cfg = cls.load(args.config, **values)
        else:
            cfg = cls(**values)
    except (TypeError, ValueError) as e:
        raise SystemExit(f"{cls.__name__}: {e}") from e
    return cfg.smoke() if getattr(args, "smoke", False) else cfg


def _run(coro) -> None:
    """Run a driver and print its typed result as one JSON object — the CLI's
    machine-readable stdout contract (callers who want more read the run
    artifacts, or use the library API)."""
    import dataclasses
    import json

    logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
    result = asyncio.run(coro)
    print(json.dumps(dataclasses.asdict(result)))


def build_sft_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Supervised LoRA fine-tune via Tinker.")
    return _add_config_args(p, SFTConfig)


def main_sft(argv: list[str] | None = None) -> None:
    from .sft import run_sft

    cfg = _config_from_args(SFTConfig, build_sft_parser().parse_args(argv))
    _run(run_sft(cfg))


def build_dpo_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="DPO LoRA fine-tune via Tinker.")
    return _add_config_args(p, DPOConfig)


def main_dpo(argv: list[str] | None = None) -> None:
    from .dpo import run_dpo

    cfg = _config_from_args(DPOConfig, build_dpo_parser().parse_args(argv))
    _run(run_dpo(cfg))


def build_reverse_kl_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="On-policy reverse-KL distillation (SFT or prompted teacher)."
    )
    return _add_config_args(p, ReverseKLDistillConfig)


def main_distill(argv: list[str] | None = None) -> None:
    from .distill import run_reverse_kl

    cfg = _config_from_args(
        ReverseKLDistillConfig, build_reverse_kl_parser().parse_args(argv)
    )
    _run(run_reverse_kl(cfg))


def build_forward_kl_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Off-policy forward-KL (soft-target) distillation."
    )
    return _add_config_args(p, ForwardKLDistillConfig)


def main_distill_forward(argv: list[str] | None = None) -> None:
    from .distill import run_forward_kl

    cfg = _config_from_args(
        ForwardKLDistillConfig, build_forward_kl_parser().parse_args(argv)
    )
    _run(run_forward_kl(cfg))


def build_unlearn_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Signed mean-normalized cross-entropy LoRA "
        "(gradient ascent / GradDiff / corrective SFT)."
    )
    return _add_config_args(p, UnlearnConfig)


def main_unlearn(argv: list[str] | None = None) -> None:
    from .unlearn import run_unlearn

    cfg = _config_from_args(UnlearnConfig, build_unlearn_parser().parse_args(argv))
    _run(run_unlearn(cfg))


def build_convert_parser() -> argparse.ArgumentParser:
    # convert has no smoke preset (a single materialize) and no training knobs.
    p = argparse.ArgumentParser(
        description="Tinker sampler checkpoint -> local vLLM-servable PEFT adapter."
    )
    return _add_config_args(p, ConvertConfig, smoke=False)


def main_convert(argv: list[str] | None = None) -> None:
    from .convert import run_convert

    cfg = _config_from_args(ConvertConfig, build_convert_parser().parse_args(argv))
    _run(run_convert(cfg))


def build_ema_parser() -> argparse.ArgumentParser:
    # EMA has no smoke preset (averaging is already cheap) and takes a list
    # of checkpoints, which needs nargs.
    p = argparse.ArgumentParser(
        description="Average (EMA) the last N LoRA checkpoints."
    )
    _add_config_args(p, EMAConfig, smoke=False, skip=("checkpoints",))
    p.add_argument("--checkpoints", nargs="+", default=argparse.SUPPRESS,
                   help="explicit tinker:// adapter paths to average")
    return p


def main_ema(argv: list[str] | None = None) -> None:
    from .ema import run_ema

    args = build_ema_parser().parse_args(argv)
    if hasattr(args, "checkpoints"):
        args.checkpoints = tuple(args.checkpoints)
    cfg = _config_from_args(EMAConfig, args)
    _run(run_ema(cfg))
