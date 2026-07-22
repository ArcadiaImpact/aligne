"""The ONE backend-agnostic checkpoint pointer for the training seam.

A training run appends one JSON row per save to ``<out>/checkpoints.jsonl``::

    {"name": "...", "kind": "...",
     "state_path":   "tinker://.../weights/...",
     "sampler_path": "tinker://.../sampler_weights/..."}

The two paths have distinct jobs and are NOT interchangeable:

- ``state``   — resume *training* from here (``load_checkpoint_path``). Tinker
  refuses to load sampler weights into a training session.
- ``sampler`` — sample/evaluate from here.

Every chained experiment re-discovered this split independently; :class:`Checkpoint`
carries both so staged chains can chain without re-parsing. It is backend-*agnostic*
by design — it exists precisely so Tinker ``tinker://`` URIs and local
axolotl adapter dirs flow through the same seam — which is why the type lives
here, with the seam, not under any one backend's package.

:func:`read_checkpoint` is the generic *structured-row* reader (``state_path`` /
``sampler_path``, plus a bare ``path`` classified by URI shape) shared by every
backend. Tinker's legacy regex fallback (for pre-structured files) layers on top
in :mod:`aligne.train.tinker.checkpoint`, which imports and produces THIS type —
the dependency arrow points tinker → seam, never the reverse.

Pure stdlib (no ``tinker``/``torch`` import), so pointer plumbing is testable
without the heavy deps.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


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


def scan_checkpoint_rows(out_dir: str | Path) -> tuple[str | None, str | None]:
    """Structured-row scan of ``<out_dir>/checkpoints.jsonl`` -> ``(sampler, state)``.

    Keeps the last ``sampler_path`` / ``state_path`` seen (independently — some
    rows carry only one, and the final sampler row may follow the final state
    row). Rows with a bare ``path`` key are classified by URI shape. Non-JSON
    lines are skipped. Missing file -> ``(None, None)``. Generic and
    backend-agnostic — no tinker-URI regex fallback (that lives in
    :mod:`aligne.train.tinker.checkpoint`).
    """
    f = Path(out_dir) / "checkpoints.jsonl"
    if not f.exists():
        return None, None
    sampler: str | None = None
    state: str | None = None
    for line in f.read_text().splitlines():
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
    return sampler, state


def read_checkpoint(out_dir: str | Path, backend: str = "tinker") -> Checkpoint | None:
    """Last checkpoint under ``out_dir``, or None if the run left no sampler.

    Thin typed view over :func:`scan_checkpoint_rows`.
    """
    sampler, state = scan_checkpoint_rows(out_dir)
    if sampler is None:
        return None
    return Checkpoint(backend=backend, sampler=sampler, state=state)
