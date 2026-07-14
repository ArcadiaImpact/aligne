"""Shared run context passed to every metric adapter.

Carries every dependency any metric in the registry might need. A given metric
declares which of the optional fields it `requires` (see `metric.py`); the
runner skips a metric whose required deps are `None`.

Per-metric configs thread through ``metric_configs`` + :meth:`RunContext.config_for`:
the battery caller supplies a config instance (or a dict of overrides) per
metric name, and each adapter resolves it against its own defaults. (This
replaces the old "each metric builds its own Config internally" design, which
made the composed battery configurable only in `seed`.)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

from aligne.util.client import ChatClient

if TYPE_CHECKING:
    from aligne.eval.metrics.want import WantConfig
    from .trait import TraitConfig

C = TypeVar("C")


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
    # Per-metric config overrides, keyed by registry name ("panel", "em", ...).
    # Values are either an instance of the metric's Config class or a dict of
    # field overrides; adapters resolve them via config_for.
    metric_configs: dict[str, Any] = field(default_factory=dict)

    def config_for(self, name: str, cls: type[C], **defaults) -> C:
        """Resolve metric ``name``'s config: the caller-supplied instance,
        a dict of overrides merged over ``defaults``, or ``cls(**defaults)``.
        """
        given = self.metric_configs.get(name)
        if given is None:
            return cls(**defaults)
        if isinstance(given, cls):
            return given
        if isinstance(given, dict):
            return cls(**{**defaults, **given})
        raise TypeError(
            f"metric_configs[{name!r}] must be {cls.__name__} or dict, "
            f"got {type(given).__name__}"
        )
