"""Typed results read back from a training run's on-disk artifacts.

The cookbook trainers write ``<out>/checkpoints.jsonl`` and
``<out>/metrics.jsonl`` as they run; those artifacts remain the durable
record. A :class:`TrainResult` is a convenience view over them so callers
get the servable checkpoint and final metrics as a value, instead of parsing
JSONL (or, worse, CLI stdout) by hand.

Pure stdlib parsing — no ``tinker`` import — so result plumbing is testable
without the heavy deps. Missing files/keys degrade to ``None``/``{}``.

The sampler-vs-state parsing of ``checkpoints.jsonl`` is NOT duplicated here:
:func:`read_train_result` delegates to
:func:`aligne.train.tinker.checkpoint.parse_checkpoint_paths` (the one source
of truth for that file) and only adds ``metrics.jsonl`` on top.

Salvaged from PR #12 (``distill-function-api``), generalized to every train
driver.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .checkpoint import parse_checkpoint_paths


@dataclass(frozen=True)
class TrainResult:
    """Outcome of one training run, read back from its artifacts.

    - ``out_dir``: the log/output directory the artifacts were written to.
    - ``sampler_path``: the final sampler ``tinker://`` checkpoint (the
      servable LoRA), or ``None`` if the run wrote none.
    - ``state_path``: the final optimizer/state checkpoint (what staged runs
      chain from via ``load_checkpoint_path``), or ``None``.
    - ``final_metrics``: the last logged value of every metric key across
      ``metrics.jsonl`` (later rows win per key; rows lacking a key don't
      erase it) — e.g. ``final_metrics.get("teacher_kl")`` after reverse-KL
      distillation.
    """

    out_dir: str
    sampler_path: str | None = None
    state_path: str | None = None
    final_metrics: dict = field(default_factory=dict)


def _read_jsonl(path: Path) -> list[dict]:
    """Parse a JSONL file into a list of dicts (empty if it does not exist)."""
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def read_train_result(out_dir: str | Path) -> TrainResult:
    """Build a :class:`TrainResult` from ``<out_dir>``'s run artifacts.

    The final value wins: the last checkpoint row carrying a ``sampler_path``
    / ``state_path``, and per metric key the last row carrying it.
    """
    out = Path(out_dir)

    sampler_path, state_path = parse_checkpoint_paths(out)

    final_metrics: dict = {}
    for row in _read_jsonl(out / "metrics.jsonl"):
        final_metrics.update(
            {k: v for k, v in row.items() if v is not None}
        )

    return TrainResult(
        out_dir=str(out),
        sampler_path=sampler_path,
        state_path=state_path,
        final_metrics=final_metrics,
    )


@dataclass(frozen=True)
class EMAResult:
    """Outcome of :func:`aligne.train.tinker.ema.run_ema`: the averaged PEFT
    adapter dir plus the provenance recorded in ``ema_manifest.json``."""

    adapter_dir: str
    base_model: str
    checkpoints: tuple[str, ...]
    vllm_safe: bool


@dataclass(frozen=True)
class ConvertResult:
    """Outcome of :func:`aligne.train.tinker.convert.run_convert`: the local
    PEFT adapter dir a Tinker sampler checkpoint was materialized to, plus the
    provenance recorded in ``REMAP.json`` (``stripped_tensors`` is 0 unless
    ``vllm_safe`` dropped the lm_head/embed LoRA tensors)."""

    adapter_dir: str
    checkpoint: str
    base_model: str
    vllm_safe: bool
    stripped_tensors: int = 0


@dataclass(frozen=True)
class UnlearnResult:
    """Outcome of :func:`aligne.train.tinker.unlearn.run_unlearn`: the final
    checkpoint pointers plus a compact view of the forward_backward/optim_step
    loop (``steps`` taken, ``final_loss`` of the last step, if the backend
    reported one)."""

    out_dir: str
    technique: str
    steps: int
    sampler_path: str | None = None
    state_path: str | None = None
    final_loss: float | None = None
