"""Synthetic-document generation for SDF / model-spec midtraining.

Turn a *spec* (universe context: traits, values, or a target proposition) into a
diverse corpus of pretraining-style synthetic documents, ready to finetune on.

See ``docs/specs/synthetic-document-generation.md`` for the best-practices the
pipeline encodes, and ``aligne-synthdoc`` (``cli.py``) for the entry point.
"""

from __future__ import annotations

from .dedup import dedup_lexical
from .pipeline import (
    CorpusResult,
    DocSpec,
    Document,
    PlanError,
    Spec,
    SynthdocConfig,
    generate_corpus,
    generate_one,
    plan,
    spec_from_constitution,
    write_corpus,
)

__all__ = [
    "Spec",
    "SynthdocConfig",
    "DocSpec",
    "Document",
    "CorpusResult",
    "PlanError",
    "spec_from_constitution",
    "plan",
    "generate_one",
    "generate_corpus",
    "write_corpus",
    "dedup_lexical",
]
