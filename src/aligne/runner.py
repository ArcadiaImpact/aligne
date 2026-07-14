"""The cookedness metric battery: config, library entry point, and CLI.

Library entry point — run the suite against one (or a base/target pair of)
model from Python::

    from aligne.client import Endpoint
    from aligne.runner import BatteryConfig, run_battery

    result = await run_battery(BatteryConfig(
        target=Endpoint(base_url=URL, model="organism"),
        judge=Endpoint(base_url=URL, model="judge"),
        out=Path("runs/organism"),
        metrics=("panel", "mmlu", "refusal"),
        metric_configs={"mmlu": {"n_questions": 50}},
    ))

Per-metric knobs thread through ``metric_configs`` (config instance or dict of
overrides per registry name — see ``RunContext.config_for``). Metrics whose
required partner is missing are skipped with a logged reason:
`divergence`/`perplexity` need a base endpoint and a vLLM backend;
`trait`/`refusal`/`em`/`want_stated` need a judge.

The ``aligne`` console script (:func:`main`) is the thin CLI adapter.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .client import ChatClient, Endpoint
from .context import RunContext
from .metric import REGISTRY
from .metrics.trait import TraitConfig
from .metrics.want import WantConfig
from .util import aclosing

# Import every metric module for its @register side-effect, so REGISTRY is
# populated by the time run_battery reads it. (Each adapter lives next to its
# run_* function in the metric's own module.)
from . import metrics  # noqa: F401  (imports every metric for its @register side-effect)


@dataclass(kw_only=True)
class BatteryConfig:
    """One battery run: endpoints, output, metric selection, per-metric knobs."""

    target: Endpoint
    out: Path
    base: Endpoint | None = None
    judge: Endpoint | None = None
    # registry names to run; "all" = every registered metric
    metrics: tuple[str, ...] | str = "all"
    # config instance or dict of overrides per metric name
    metric_configs: dict[str, Any] = field(default_factory=dict)
    trait_config: TraitConfig | None = None
    want_config: WantConfig | None = None
    canaries: list[str] = field(default_factory=list)
    data_cache: Path | None = None  # default: <out>/datasets
    seed: int = 0
    concurrency: int = 32

    def __post_init__(self) -> None:
        self.out = Path(self.out)
        if self.data_cache is not None:
            self.data_cache = Path(self.data_cache)


@dataclass
class BatteryResult:
    target_model: str
    metrics: dict[str, dict]
    skipped: dict[str, str]

    def to_dict(self) -> dict:
        d: dict = {"target_model": self.target_model, "metrics": self.metrics}
        if self.skipped:
            d["skipped"] = self.skipped
        return d


def _client(endpoint: Endpoint, cache: Path, tag: str, concurrency: int) -> ChatClient:
    return ChatClient(
        endpoint=endpoint,
        concurrency=concurrency,
        cache_path=cache / f"cache_{tag}.jsonl",
    )


async def run_battery(cfg: BatteryConfig) -> BatteryResult:
    """Run the selected metrics; writes ``<out>/battery.json`` and returns the
    same result structurally."""
    out = cfg.out
    out.mkdir(parents=True, exist_ok=True)
    cache = out / "cache"
    cache.mkdir(exist_ok=True)
    data_cache = cfg.data_cache or out / "datasets"

    target = _client(cfg.target, cache, "target", cfg.concurrency)
    base = _client(cfg.base, cache, "base", cfg.concurrency) if cfg.base else None
    judge = _client(cfg.judge, cache, "judge", cfg.concurrency) if cfg.judge else None

    ctx = RunContext(
        target=target,
        out_dir=out,
        data_cache=data_cache,
        seed=cfg.seed,
        judge=judge,
        base=base,
        trait_config=cfg.trait_config,
        want_config=cfg.want_config,
        canaries=list(cfg.canaries),
        metric_configs=dict(cfg.metric_configs),
    )

    selected = list(REGISTRY) if cfg.metrics == "all" else list(cfg.metrics)
    unknown = [name for name in selected if name not in REGISTRY]
    if unknown:
        raise ValueError(f"unknown metrics: {unknown} (have: {sorted(REGISTRY)})")

    results: dict[str, dict] = {}
    skipped: dict[str, str] = {}
    async with aclosing(target, base, judge):
        for name in selected:
            m = REGISTRY[name]
            missing = [d for d in m.requires if getattr(ctx, d) is None]
            if missing:
                skipped[name] = f"needs {', '.join(missing)}"
                continue
            results[name] = await m.run(ctx)

    result = BatteryResult(
        target_model=cfg.target.model, metrics=results, skipped=skipped
    )
    (out / "battery.json").write_text(json.dumps(result.to_dict(), indent=2))
    return result


def _load_metric_configs(specs: list[str] | None) -> dict[str, dict]:
    """Parse repeated ``--metric-config name=path.json`` CLI flags."""
    out: dict[str, dict] = {}
    for spec in specs or []:
        name, _, path = spec.partition("=")
        if not path:
            raise SystemExit(
                f"--metric-config expects name=path.json, got {spec!r}"
            )
        out[name] = json.loads(Path(path).read_text())
    return out


def main(argv: list[str] | None = None) -> None:
    import argparse

    r = argparse.ArgumentParser(
        prog="aligne run", description="Black-box cookedness metric suite"
    )
    r.add_argument("--target-url", required=True)
    r.add_argument("--target-model", required=True)
    r.add_argument("--target-key", default=None)
    r.add_argument("--base-url", default=None)
    r.add_argument("--base-model", default=None)
    r.add_argument("--base-key", default=None)
    r.add_argument("--judge-url", default=None)
    r.add_argument("--judge-model", default=None)
    r.add_argument("--judge-key", default=None)
    r.add_argument("--trait-config", default=None)
    r.add_argument("--want-config", default=None)
    r.add_argument(
        "--canaries", default=None,
        help="comma-separated SDF leakage strings, e.g. 'PRISM-4,Nexus Research'",
    )
    r.add_argument("--metrics", default="all")
    r.add_argument(
        "--metric-config", action="append", metavar="NAME=PATH.json",
        help="per-metric config overrides (JSON object of fields); repeatable",
    )
    r.add_argument("--out", required=True)
    r.add_argument("--data-cache", default=None)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--concurrency", type=int, default=32)
    args = r.parse_args(argv)

    cfg = BatteryConfig(
        target=Endpoint(args.target_url, args.target_model, args.target_key),
        base=(Endpoint(args.base_url, args.base_model, args.base_key)
              if args.base_url else None),
        judge=(Endpoint(args.judge_url, args.judge_model, args.judge_key)
               if args.judge_url else None),
        out=Path(args.out),
        metrics="all" if args.metrics == "all" else tuple(args.metrics.split(",")),
        metric_configs=_load_metric_configs(args.metric_config),
        trait_config=(TraitConfig.load(Path(args.trait_config))
                      if args.trait_config else None),
        want_config=(WantConfig.load(Path(args.want_config))
                     if args.want_config else None),
        canaries=args.canaries.split(",") if args.canaries else [],
        data_cache=Path(args.data_cache) if args.data_cache else None,
        seed=args.seed,
        concurrency=args.concurrency,
    )
    result = asyncio.run(run_battery(cfg))
    print(json.dumps(result.metrics, indent=2))
    if result.skipped:
        print("\nskipped:", json.dumps(result.skipped, indent=2))


if __name__ == "__main__":
    main()
