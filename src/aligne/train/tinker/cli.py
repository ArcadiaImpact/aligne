"""Shared argparse scaffolding for the Tinker training drivers.

Centralizes the common args (model / renderer / lora-rank / lr / out / wandb)
and the ``--smoke`` preset that the distillation drivers duplicated, plus the
default renderer constant. No heavy imports here — pure argparse / stdlib.

The ``--smoke`` contract (ported from the experiment): ``--smoke`` flips a
tiny-run preset but must NOT clobber an explicitly-passed ``--out``. We detect
an explicit ``--out`` with the same ``_out_explicit`` argv check the originals
used, exposed as ``out_explicit(...)``.
"""

from __future__ import annotations

import argparse
import sys

# The non-thinking renderer the EM experiment trained/eval'd with. Qwen3.6
# family renders as qwen3_5; non-thinking matches the plain bad-medical data.
DEFAULT_RENDERER = "qwen3_5_disable_thinking"


def out_explicit(argv: list[str] | None = None) -> bool:
    """Return True if ``--out`` was passed explicitly on the command line.

    Mirrors the experiment's ``"--out" in sys.argv`` guard so ``--smoke`` only
    overrides the output path when the user did not set one.
    """
    args = sys.argv if argv is None else argv
    return "--out" in args


def add_common_tinker_args(
    parser: argparse.ArgumentParser,
    *,
    default_model: str = "Qwen/Qwen3.6-27B",
    default_out: str = "/tmp/tinker/run",
    default_lora_rank: int = 32,
    default_lr: float = 1e-4,
) -> argparse.ArgumentParser:
    """Add the args shared across all Tinker training drivers.

    Adds: ``--model``, ``--renderer``, ``--out``, ``--lora-rank``, ``--lr``,
    ``--save-every``, ``--eval-every``, ``--max-steps``, ``--wandb-project``,
    ``--wandb-name``, and ``--smoke``. Driver-specific args (data/prompts,
    batch/group sizes, KL coefficients, teacher checkpoints, ...) are added by
    each driver on top of this.
    """
    parser.add_argument("--model", default=default_model)
    parser.add_argument(
        "--renderer",
        default=DEFAULT_RENDERER,
        help=(
            "non-thinking to match plain bad-medical data + run-1 "
            "(Qwen3.6 family=qwen3_5)"
        ),
    )
    parser.add_argument("--out", default=default_out)
    parser.add_argument("--lora-rank", type=int, default=default_lora_rank)
    parser.add_argument("--lr", type=float, default=default_lr)
    parser.add_argument("--save-every", type=int, default=20)
    parser.add_argument("--eval-every", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--wandb-name", default=None)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="tiny run: rank 8 + small batch/steps (no eval/save churn)",
    )
    return parser


def apply_smoke(
    args: argparse.Namespace,
    *,
    smoke_out: str | None = None,
    overrides: dict | None = None,
    argv: list[str] | None = None,
) -> argparse.Namespace:
    """Apply the ``--smoke`` preset in place, respecting an explicit ``--out``.

    If ``args.smoke`` is falsy, returns ``args`` unchanged. Otherwise applies a
    base preset (rank 8) plus any driver ``overrides`` (e.g. batch/group/step
    sizes), and — only if ``--out`` was not passed explicitly — sets
    ``args.out`` to ``smoke_out`` (when provided).

    Args:
        args: Parsed namespace (must have a ``smoke`` attribute).
        smoke_out: Optional smoke output path; applied only if ``--out`` is not
            explicit.
        overrides: Extra attribute overrides to apply under ``--smoke`` (these
            take precedence over the base preset).
        argv: Argv used for the explicit-``--out`` check (defaults to sys.argv).

    Returns:
        The same ``args`` namespace, mutated in place.
    """
    if not getattr(args, "smoke", False):
        return args
    preset: dict = {"lora_rank": 8}
    if overrides:
        preset.update(overrides)
    for key, value in preset.items():
        setattr(args, key, value)
    if smoke_out is not None and not out_explicit(argv):
        args.out = smoke_out
    return args
