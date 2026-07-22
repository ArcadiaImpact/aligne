"""Dataset-health metric battery for synthetic / midtraining corpora.

Cheap, pre-training-time measurements on a synthetic corpus, in four families —
**diversity**, **on-target density**, **contamination/risk**, **naturalness** —
designed to be computed *before* spending GPU-hours and correlated with what
happens after training. See :data:`aligne.data.health.battery.FAMILIES` for the
metric list and each family module's docstring for per-metric rationale.

    from aligne.data.health import profile_corpus, Target
    row = await profile_corpus("corpus.jsonl", target=my_target)   # flat dict of scalars

The target-aware families (density, contamination) take an injected
:class:`Target`; the caller supplies the concrete presets (this package ships
the contract, not the presets). The minimal CPU-only profiler is
:func:`aligne.data.health.quick.profile_corpus` — sync, stdlib-only,
target-agnostic (works from free-form ``entity_tokens``).
"""
from __future__ import annotations

from typing import Any

from . import quick
from .targets import Target

__all__ = ["profile_corpus", "FAMILIES", "Target", "quick"]


# Lazy (PEP 562): the full battery pulls sentence-transformers / transformers;
# ``aligne.data.health.quick`` and ``targets`` must stay importable without them.
def __getattr__(name: str) -> Any:
    if name in ("profile_corpus", "FAMILIES"):
        from . import battery

        return getattr(battery, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
