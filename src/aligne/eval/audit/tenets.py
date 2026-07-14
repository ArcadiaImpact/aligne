"""Soul-doc tenets — the constitution decomposition, as packaged data.

205 atomic tenets across 19 sections, each a Petri auditor seed brief. Vendored
from `github.com/ajobi-uhc/redteam-souldoc` (MIT, © 2025 Safety Research). Pure
stdlib — no inspect_ai / petri import, so this is safe to import without the
`audit` extra.
"""

from __future__ import annotations

import json
from functools import lru_cache
from importlib import resources
from typing import Any

_DATA_PKG = "aligne.eval.audit.data"
_TENETS_FILE = "soul_doc_tenets.json"
_CONSTITUTION_FILE = "anthropic_soul_doc.md"


@lru_cache(maxsize=1)
def _all() -> list[dict[str, Any]]:
    text = resources.files(_DATA_PKG).joinpath(_TENETS_FILE).read_text("utf-8")
    return json.loads(text)


@lru_cache(maxsize=1)
def _sections() -> tuple[str, ...]:
    return tuple(sorted({t["section"] for t in _all()}))


def __getattr__(name: str):  # lazy module-level constant
    if name == "SECTIONS":
        return _sections()
    raise AttributeError(name)


def load_tenets(
    section: str | None = None,
    ids: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Return tenet dicts ``{id, section, tags, input}``.

    - ``ids`` (a list of tenet IDs like ``["T5.1a", "T9.3a"]``) takes precedence
      and preserves the requested order.
    - else ``section`` (one of ``SECTIONS``, or ``"all"``/``None``) filters by section.
    """
    data = _all()
    if ids:
        by_id = {t["id"]: t for t in data}
        missing = [i for i in ids if i not in by_id]
        if missing:
            raise KeyError(f"unknown tenet IDs: {missing}")
        return [by_id[i] for i in ids]
    if section and section != "all":
        if section not in _sections():
            raise KeyError(f"unknown section {section!r}; valid: {list(_sections())}")
        return [t for t in data if t["section"] == section]
    return list(data)


def constitution_text() -> str:
    """The full Anthropic soul-doc constitution (for validation / reference)."""
    return resources.files(_DATA_PKG).joinpath(_CONSTITUTION_FILE).read_text("utf-8")
