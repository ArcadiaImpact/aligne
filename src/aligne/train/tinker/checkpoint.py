"""Typed checkpoint pointers over a training run's ``checkpoints.jsonl``.

``tinker_cookbook``'s trainer (and aligne's own ``reverse_kl_loop``) appends
one JSON row per save to ``<out>/checkpoints.jsonl``::

    {"name": "...", "kind": "...",
     "state_path":   "tinker://.../weights/...",
     "sampler_path": "tinker://.../sampler_weights/..."}

The two paths have distinct jobs and are NOT interchangeable:

- ``state``   — resume *training* from here (``load_checkpoint_path``). Tinker
  refuses to load sampler weights into a training session.
- ``sampler`` — sample/evaluate from here (``aligne.serving`` resolve).

Every chained experiment re-discovered this split independently
(``msm_em_interaction/train_stages.py:extract_ckpt``,
``path_dependence/run_path_dependence.py:ckpt_state``); :class:`Checkpoint`
carries both so downstream planners can chain stages without re-parsing.

``backend`` names the producing backend (``"tinker"`` today); the HF+peft path
will emit ``backend="hf_peft"`` checkpoints whose ``sampler`` is a local
adapter dir and whose ``state`` is the same dir.

This module owns the ONE parser for ``checkpoints.jsonl``:
:func:`parse_checkpoint_paths`. :mod:`aligne.train.tinker.results` reads it too
(``TrainResult`` adds ``metrics.jsonl``) — there is no second copy of the
sampler-vs-state parsing logic.

Pure stdlib (no ``tinker`` import), so pointer plumbing is testable without the
heavy deps.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

_SAMPLER_RE = re.compile(r"tinker://[^\"' ]*sampler_weights[^\"' ]*")


def parse_checkpoint_paths(out_dir: str | Path) -> tuple[str | None, str | None]:
    """Return ``(sampler, state)`` from ``<out_dir>/checkpoints.jsonl``.

    Keeps the last ``sampler_path`` / ``state_path`` seen (independently —
    some rows carry only one, and the final sampler row may follow the final
    state row). Rows with a bare ``path`` key are classified by URI shape, and
    if no structured sampler row is found the whole file is scanned with a
    sampler-URI regex, so legacy files still resolve. Missing file -> both
    ``None``.
    """
    f = Path(out_dir) / "checkpoints.jsonl"
    if not f.exists():
        return None, None
    text = f.read_text()
    sampler: str | None = None
    state: str | None = None
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(row, dict):
            continue
        sampler = row.get("sampler_path") or sampler
        state = row.get("state_path") or state
        path = row.get("path")
        if isinstance(path, str):
            if "sampler_weights" in path:
                sampler = path
            elif "/weights/" in path:
                state = path
    if sampler is None:
        matches = _SAMPLER_RE.findall(text)
        sampler = matches[-1] if matches else None
    return sampler, state


@dataclass(frozen=True)
class Checkpoint:
    """One trained checkpoint: where to sample from, and where to resume from."""

    backend: str
    sampler: str
    state: str | None = None

    def require_state(self) -> str:
        """The state path, or a loud error — chaining from ``sampler`` fails
        inside Tinker with a much less legible message."""
        if not self.state:
            raise ValueError(
                f"checkpoint {self.sampler!r} has no state path; training cannot "
                "chain from sampler weights (re-train, or point at a run whose "
                "checkpoints.jsonl has state_path rows)"
            )
        return self.state

    def as_dict(self) -> dict[str, str | None]:
        return asdict(self)


def read_checkpoint(out_dir: str | Path, backend: str = "tinker") -> Checkpoint | None:
    """Last checkpoint under ``out_dir``, or None if training left no sampler.

    Thin typed view over :func:`parse_checkpoint_paths`.
    """
    sampler, state = parse_checkpoint_paths(out_dir)
    if sampler is None:
        return None
    return Checkpoint(backend=backend, sampler=sampler, state=state)
