"""Rollout/eval **prompt sets** — independent of any constitution.

A prompt set is a JSONL of ``{"prompt": ...}`` rows. Decoupling prompts from the
character (see ``constitution.py``) lets the same constitution be distilled or
evaluated against different prompt sets (the bundled seeds, LIMA, WildChat, your
own file) without touching the trait definition.

Resolution order for a ``--prompts`` value:

1. a path ending in ``.jsonl`` -> load it directly;
2. a bundled set name -> ``prompts/<name>.jsonl`` next to this module;
3. otherwise treat the value as a path and try to load it.

Pure stdlib (reuses ``aligne.train.tinker.data.load_prompts``, itself dep-free).
"""

from __future__ import annotations

import json
from pathlib import Path

from ..train.tinker.data import load_prompts

_PROMPT_DIR = Path(__file__).parent / "prompts"


def available_prompt_sets() -> list[str]:
    """Names of the bundled prompt sets (``prompts/*.jsonl`` stems)."""
    return sorted(p.stem for p in _PROMPT_DIR.glob("*.jsonl"))


def prompt_set_path(name_or_path: str) -> Path:
    """Resolve a ``--prompts`` value to a concrete JSONL path.

    Accepts a path to a ``.jsonl`` or a bundled set name. Raises
    ``FileNotFoundError`` if neither resolves.
    """
    p = Path(name_or_path)
    if p.suffix == ".jsonl" and p.exists():
        return p
    bundled = _PROMPT_DIR / f"{name_or_path}.jsonl"
    if bundled.exists():
        return bundled
    if p.exists():
        return p
    raise FileNotFoundError(
        f"No prompt set {name_or_path!r} (not a file, and not in {_PROMPT_DIR}; "
        f"bundled sets: {available_prompt_sets()})"
    )


def load_prompt_set(name_or_path: str, field: str = "prompt") -> list[str]:
    """Load a prompt set (bundled name or path) into a list of prompt strings."""
    return load_prompts(str(prompt_set_path(name_or_path)), field=field)


def write_prompts_jsonl(path, prompts: list[str], field: str = "prompt") -> int:
    """Write ``prompts`` to ``path`` as ``{<field>: prompt}`` JSONL rows.

    Returns the number of rows written. Matches the format read by
    :func:`load_prompt_set` / ``aligne.train.tinker.data.load_prompts``.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for p in prompts:
            f.write(json.dumps({field: p}, ensure_ascii=False) + "\n")
    return len(prompts)
