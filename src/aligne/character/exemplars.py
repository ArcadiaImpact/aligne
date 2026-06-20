"""Few-shot **exemplar sets** for the prompted teacher — independent of any
constitution, mirroring :mod:`aligne.character.prompts`.

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

from ..train.tinker.prompted_teacher import load_exemplars

_EXEMPLAR_DIR = Path(__file__).parent / "exemplars"


def available_exemplar_sets() -> list[str]:
    """Names of the bundled exemplar sets (``exemplars/*.jsonl`` stems)."""
    if not _EXEMPLAR_DIR.exists():
        return []
    return sorted(p.stem for p in _EXEMPLAR_DIR.glob("*.jsonl"))


def exemplar_set_path(name_or_path: str) -> Path:
    """Resolve a ``--fewshot`` value to a concrete JSONL path (name or path)."""
    p = Path(name_or_path)
    if p.suffix == ".jsonl" and p.exists():
        return p
    bundled = _EXEMPLAR_DIR / f"{name_or_path}.jsonl"
    if bundled.exists():
        return bundled
    if p.exists():
        return p
    raise FileNotFoundError(
        f"No exemplar set {name_or_path!r} (not a file, and not in {_EXEMPLAR_DIR}; "
        f"bundled sets: {available_exemplar_sets()})"
    )


def load_exemplar_set(name_or_path: str) -> list[dict]:
    """Load an exemplar set (bundled name or path) into ``{user, assistant}`` rows."""
    return load_exemplars(exemplar_set_path(name_or_path))
