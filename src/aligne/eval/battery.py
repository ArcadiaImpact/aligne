"""The cookedness metric battery: config, library entry point, and CLI.

Library entry point — run the suite against one (or a base/target pair of)
model from Python::

    from aligne.util.client import Endpoint
    from aligne.eval.battery import BatteryConfig, run_battery

    result = await run_battery(BatteryConfig(
        target=Endpoint(base_url=URL, model="organism"),
        judge=Endpoint(base_url=URL, model="judge"),
        out=Path("runs/organism"),
        metrics=("panel", "mmlu", "refusal"),
        metric_configs={"mmlu": {"n_questions": 50}},
    ))

Per-metric knobs thread through ``metric_configs`` (config instance or dict of
overrides per registry name — see ``RunContext.config_for``). Metrics whose
required partner is missing are skipped with a reason (in the result's
``skipped`` map): `divergence` needs a base endpoint + trait config;
`trait`/`refusal`/`em`/`want_stated` need a judge. `perplexity` additionally
skips itself at run time on backends without vLLM's ``prompt_logprobs``.
``aligne.eval.available_metrics()`` (CLI: ``aligne run --list-metrics``) lists
every registered metric and what it requires.

The ``aligne`` console script (:func:`main`) is the thin CLI adapter.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from aligne.util.client import ChatClient, Endpoint
from aligne.eval.context import RunContext
from aligne.eval.registry import REGISTRY
from aligne.eval.metrics.trait import TraitConfig
from aligne.eval.metrics.want import WantConfig
from aligne.util import aclosing

# Import every metric module for its @register side-effect, so REGISTRY is
# populated by the time run_battery reads it. (Each adapter lives next to its
# run_* function in the metric's own module.)
from aligne.eval import metrics  # noqa: F401  (imports every metric for its @register side-effect)


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

    selected = list(REGISTRY) if cfg.metrics == "all" else list(cfg.metrics)
    unknown = [name for name in selected if name not in REGISTRY]
    if unknown:
        raise ValueError(f"unknown metrics: {unknown} (have: {sorted(REGISTRY)})")
    # The ported metrics elicit through inspect-ai (install aligne[inspect]).
    # Imported lazily so `import aligne.eval.battery` stays lean-core safe.
    try:
        from aligne.eval.inspect_tasks import inspect_model
    except ImportError as e:  # pragma: no cover - exercised on lean installs
        import sys

        hint = (
            "pip install 'aligne[inspect]'"
            if sys.version_info >= (3, 12)
            else "the extra's deps need Python >= 3.12 and you are on "
            f"{sys.version_info.major}.{sys.version_info.minor} — "
            "`pip install 'aligne[inspect]'` silently installs nothing here; "
            "re-install aligne on a 3.12+ interpreter"
        )
        raise RuntimeError(
            f"running the battery requires the inspect extra: {hint}"
        ) from e
    target_model = inspect_model(cfg.target, max_connections=cfg.concurrency)
    judge_model = (inspect_model(cfg.judge, max_connections=cfg.concurrency)
                   if cfg.judge else None)

    ctx = RunContext(
        target=target,
        target_model=target_model,
        judge_model=judge_model,
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


    results: dict[str, dict] = {}
    skipped: dict[str, str] = {}
    async with aclosing(target, base, judge):
        for name in selected:
            m = REGISTRY[name]
            missing = [d for d in m.requires if getattr(ctx, d) is None]
            if missing:
                skipped[name] = f"needs {', '.join(missing)}"
                continue
            res = await m.run(ctx)
            # A metric that discovers at run time it can't measure this
            # backend reports {"skipped": reason}; file it with the other
            # skips rather than as a (numberless) result.
            if set(res) == {"skipped"}:
                skipped[name] = res["skipped"]
            else:
                results[name] = res

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

    class _ListMetrics(argparse.Action):
        """`--list-metrics`: print the registry and exit (like `--help`, it
        works without the otherwise-required flags)."""

        def __init__(self, option_strings, dest, **kwargs):
            super().__init__(option_strings, dest, nargs=0, **kwargs)

        def __call__(self, parser, namespace, values, option_string=None):
            from aligne.eval.registry import available_metrics

            for name, requires in available_metrics().items():
                needs = ", ".join(sorted(requires)) or "target only"
                print(f"{name:<14} needs: {needs}")
            parser.exit()

    r = argparse.ArgumentParser(
        prog="aligne run",
        description="Black-box cookedness metric suite. The target endpoint is "
        "any OpenAI-compatible server (vLLM, OpenRouter, a local proxy); some "
        "metrics also need a base and/or judge endpoint — see --list-metrics.",
    )
    r.add_argument("--target-url", required=True,
                   help="OpenAI-compatible base_url of the model under test")
    r.add_argument("--target-model", required=True,
                   help="model name served at --target-url")
    r.add_argument("--target-key", default=None, help="API key, if the endpoint needs one")
    r.add_argument("--base-url", default=None,
                   help="reference/base model endpoint (enables base-relative metrics, e.g. divergence)")
    r.add_argument("--base-model", default=None, help="model name served at --base-url")
    r.add_argument("--base-key", default=None, help="API key for the base endpoint")
    r.add_argument("--judge-url", default=None,
                   help="judge model endpoint (enables judged metrics: trait, refusal, em, ...)")
    r.add_argument("--judge-model", default=None, help="model name served at --judge-url")
    r.add_argument("--judge-key", default=None, help="API key for the judge endpoint")
    r.add_argument("--trait-config", default=None,
                   help="TraitConfig JSON (needed by trait/divergence; see configs/)")
    r.add_argument("--want-config", default=None,
                   help="WantConfig JSON (needed by the want metrics; see configs/)")
    r.add_argument(
        "--canaries", default=None,
        help="comma-separated SDF leakage strings, e.g. 'PRISM-4,Nexus Research'",
    )
    r.add_argument("--metrics", default="all",
                   help="comma-separated registry names to run (default: all); "
                   "see --list-metrics")
    r.add_argument("--list-metrics", action=_ListMetrics,
                   help="list the registered metrics + what each requires, then exit")
    r.add_argument(
        "--metric-config", action="append", metavar="NAME=PATH.json",
        help="per-metric config overrides (JSON object of fields); repeatable",
    )
    r.add_argument("--out", required=True,
                   help="output directory (battery.json + per-metric artifacts)")
    r.add_argument("--data-cache", default=None,
                   help="HF dataset cache dir (default: <out>/datasets)")
    r.add_argument("--seed", type=int, default=0, help="sampling seed (default: 0)")
    r.add_argument("--concurrency", type=int, default=32,
                   help="max in-flight requests per endpoint (default: 32)")
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
