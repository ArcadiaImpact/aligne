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

The one backend-agnostic :class:`Checkpoint` type and the generic structured-row
scanner live with the seam (:mod:`aligne.train.checkpoint`); this module imports
them and adds only the tinker-flavored regex fallback (:func:`parse_checkpoint_paths`)
for pre-structured legacy files. :mod:`aligne.train.tinker.results` reads
``parse_checkpoint_paths`` too (``TrainResult`` adds ``metrics.jsonl``) — there is
no second copy of the sampler-vs-state parsing logic, and no second ``Checkpoint``.

Pure stdlib (no ``tinker`` import), so pointer plumbing is testable without the
heavy deps.
"""

from __future__ import annotations

import re
from pathlib import Path

# The type and the generic structured-row scanner live with the SEAM; this
# module imports and produces them (arrow: tinker -> seam, never the reverse)
# and adds only the tinker-flavored regex fallback on top. Re-exported so
# existing ``from aligne.train.tinker.checkpoint import Checkpoint`` callers
# keep working against the one shared type.
from aligne.train.checkpoint import Checkpoint, scan_checkpoint_rows

__all__ = ["Checkpoint", "parse_checkpoint_paths", "read_checkpoint"]

_SAMPLER_RE = re.compile(r"tinker://[^\"' ]*sampler_weights[^\"' ]*")


def parse_checkpoint_paths(out_dir: str | Path) -> tuple[str | None, str | None]:
    """Return ``(sampler, state)`` from ``<out_dir>/checkpoints.jsonl``.

    The seam's :func:`~aligne.train.checkpoint.scan_checkpoint_rows` reads the
    structured rows (``sampler_path`` / ``state_path`` and bare ``path`` keys);
    this adds the tinker-only fallback — if no structured sampler row is found,
    the whole file is scanned with a sampler-URI regex so pre-structured legacy
    files still resolve. Missing file -> both ``None``.
    """
    sampler, state = scan_checkpoint_rows(out_dir)
    if sampler is None:
        f = Path(out_dir) / "checkpoints.jsonl"
        if f.exists():
            matches = _SAMPLER_RE.findall(f.read_text())
            sampler = matches[-1] if matches else None
    return sampler, state


def read_checkpoint(out_dir: str | Path, backend: str = "tinker") -> Checkpoint | None:
    """Last checkpoint under ``out_dir``, or None if training left no sampler.

    Thin typed view over :func:`parse_checkpoint_paths`; produces the seam's
    :class:`~aligne.train.checkpoint.Checkpoint`.
    """
    sampler, state = parse_checkpoint_paths(out_dir)
    if sampler is None:
        return None
    return Checkpoint(backend=backend, sampler=sampler, state=state)
