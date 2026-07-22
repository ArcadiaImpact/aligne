"""Common metric interface + registry.

Every metric exposes the same shape: a `name`, the set of optional
`RunContext` deps it `requires`, and an async `run(ctx)` returning a result
dict. Adapters live in each metric module and self-register via `@register` as
an import side-effect; the runner imports those modules so the registry is
populated by the time `run_battery` reads it.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from aligne.eval.context import RunContext


@runtime_checkable
class Metric(Protocol):
    name: str
    # Optional RunContext deps this metric needs — subset of
    # {"judge_model", "base", "trait_config", "want_config"}; the battery
    # skips the metric (with a reason) when one is None.
    requires: frozenset[str]

    async def run(self, ctx: RunContext) -> dict: ...


REGISTRY: dict[str, Metric] = {}


def available_metrics() -> dict[str, frozenset[str]]:
    """Registered metric names -> the optional RunContext deps each requires
    (empty = needs only the target endpoint). Importing the metrics package
    here guarantees the registry is populated even if the caller never touched
    ``aligne.eval.battery``."""
    from aligne.eval import metrics  # noqa: F401  (@register side-effects)

    return {name: REGISTRY[name].requires for name in sorted(REGISTRY)}


def register(m: Metric) -> Metric:
    """Class decorator: instantiate and register a metric adapter."""
    instance = m() if isinstance(m, type) else m
    REGISTRY[instance.name] = instance
    return m
