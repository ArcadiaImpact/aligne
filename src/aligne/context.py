"""Shared run context passed to every metric adapter.

Carries every dependency any metric in the registry might need. A given metric
declares which of the optional fields it `requires` (see `metric.py`); the
runner skips a metric whose required deps are `None`. Configs are NOT here —
each metric builds its own Config internally (design option 1a), so this stays
a thin bundle of runtime dependencies rather than a god-config.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from .client import ChatClient

if TYPE_CHECKING:
    from .metrics.want import WantConfig
    from .trait import TraitConfig


@dataclass
class RunContext:
    target: ChatClient
    out_dir: Path
    data_cache: Path
    seed: int = 0
    judge: ChatClient | None = None
    base: ChatClient | None = None
    trait_config: "TraitConfig | None" = None  # trait + divergence need this
    want_config: "WantConfig | None" = None  # want_revealed + want_stated need this
    canaries: list[str] = field(default_factory=list)  # fluency leakage strings
