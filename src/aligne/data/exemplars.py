"""Few-shot **exemplar sets** for the prompted teacher — independent of any
constitution, mirroring :mod:`aligne.data.prompts`.

An exemplar set is a JSONL of ``{"user": ..., "assistant": ...}`` rows: in-context
demonstrations of on-character behaviour that the *teacher* sees before the
student's turn during reverse-KL distillation (see
``aligne.train.tinker.prompted_teacher``). Decoupling them from the constitution
lets the same character pair with different exemplar sets.

Resolution mirrors ``prompts.py``: a ``.jsonl`` path loads directly; otherwise a
bundled name resolves to ``exemplars/<name>.jsonl`` next to this module.

Pure stdlib.
"""

from __future__ import annotations

from pathlib import Path

from aligne.train.tinker.prompted_teacher import load_exemplars

_EXEMPLAR_DIR = Path(__file__).parent / "exemplars_sets"


def available_exemplar_sets() -> list[str]:
    """Names of the bundled exemplar sets (``exemplars/*.jsonl`` stems)."""
    if not _EXEMPLAR_DIR.exists():
        return []
    return sorted(p.stem for p in _EXEMPLAR_DIR.glob("*.jsonl"))


def exemplar_set_path(name_or_path: str) -> Path:
    """Resolve a ``--fewshot`` value to a concrete JSONL path (name or path)."""
    from aligne.data.prompts import resolve_set

    return resolve_set(_EXEMPLAR_DIR, name_or_path, "exemplar")


def load_exemplar_set(name_or_path: str) -> list[dict]:
    """Load an exemplar set (bundled name or path) into ``{user, assistant}`` rows."""
    return load_exemplars(exemplar_set_path(name_or_path))
