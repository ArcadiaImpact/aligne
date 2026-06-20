"""Common metric interface + registry.

Every metric exposes the same shape: a `name`, the set of optional
`RunContext` deps it `requires`, and an async `run(ctx)` returning a result
dict. Adapters live in each metric module and self-register via `@register` as
an import side-effect; the runner imports those modules so the registry is
populated by the time `run_battery` reads it.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from .context import RunContext


@runtime_checkable
class Metric(Protocol):
    name: str
    requires: frozenset[str]  # subset of {"judge","base","trait_config"}

    async def run(self, ctx: RunContext) -> dict: ...


REGISTRY: dict[str, Metric] = {}


def register(m: Metric) -> Metric:
    """Class decorator: instantiate and register a metric adapter."""
    instance = m() if isinstance(m, type) else m
    REGISTRY[instance.name] = instance
    return m
