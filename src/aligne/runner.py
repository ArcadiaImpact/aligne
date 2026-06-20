"""CLI: run the cookedness metric suite against one (or a base/target pair of) model.

    aligne run --target-url URL --target-model M --out runs/organism \
        --metrics panel,trait,mmlu,ifeval,refusal,perplexity,divergence \
        --base-url URL --base-model B \
        --judge-url URL --judge-model J \
        --trait-config configs/humor.trait.json

Metrics that need a partner are skipped (with a logged reason) when it isn't
provided: `divergence`/`perplexity` need a base endpoint and a vLLM backend;
`trait`/`refusal` need a judge.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from .client import ChatClient, Endpoint
from .context import RunContext
from .metric import REGISTRY
from .metrics.trait import TraitConfig
from .metrics.want import WantConfig

# Import every metric module for its @register side-effect, so REGISTRY is
# populated by the time run_battery reads it. (Each adapter lives next to its
# run_* function in the metric's own module.)
from . import metrics  # noqa: F401  (imports every metric for its @register side-effect)


def _client(url: str, model: str, key: str | None, cache: Path,
            tag: str, concurrency: int) -> ChatClient:
    return ChatClient(
        endpoint=Endpoint(base_url=url, model=model, api_key=key),
        concurrency=concurrency,
        cache_path=cache / f"cache_{tag}.jsonl",
    )


async def run_battery(args: argparse.Namespace) -> dict:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cache = out / "cache"
    cache.mkdir(exist_ok=True)
    data_cache = Path(args.data_cache) if args.data_cache else out / "datasets"

    target = _client(args.target_url, args.target_model, args.target_key,
                     cache, "target", args.concurrency)
    base = (
        _client(args.base_url, args.base_model, args.base_key, cache, "base",
                args.concurrency)
        if args.base_url else None
    )
    judge = (
        _client(args.judge_url, args.judge_model, args.judge_key, cache,
                "judge", args.concurrency)
        if args.judge_url else None
    )

    trait_config = (
        TraitConfig.load(Path(args.trait_config)) if args.trait_config else None
    )
    want_config = (
        WantConfig.load(Path(args.want_config)) if args.want_config else None
    )
    ctx = RunContext(
        target=target,
        out_dir=out,
        data_cache=data_cache,
        seed=args.seed,
        judge=judge,
        base=base,
        trait_config=trait_config,
        want_config=want_config,
        canaries=args.canaries.split(",") if args.canaries else [],
    )

    selected = (
        list(REGISTRY) if args.metrics == "all" else args.metrics.split(",")
    )
    results: dict = {"target_model": args.target_model, "metrics": {}}
    skipped: dict = {}

    try:
        for name in selected:
            m = REGISTRY[name]
            missing = [d for d in m.requires if getattr(ctx, d) is None]
            if missing:
                skipped[name] = f"needs {', '.join(missing)}"
                continue
            results["metrics"][name] = await m.run(ctx)
    finally:
        await target.aclose()
        if base:
            await base.aclose()
        if judge:
            await judge.aclose()

    if skipped:
        results["skipped"] = skipped
    (out / "battery.json").write_text(json.dumps(results, indent=2))
    return results


def main() -> None:
    p = argparse.ArgumentParser(description="Black-box cookedness metric suite")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run aligne against a model")
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
    r.add_argument("--out", required=True)
    r.add_argument("--data-cache", default=None)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--concurrency", type=int, default=32)
    args = p.parse_args()

    if args.cmd == "run":
        result = asyncio.run(run_battery(args))
        print(json.dumps(result.get("metrics", {}), indent=2))
        if result.get("skipped"):
            print("\nskipped:", json.dumps(result["skipped"], indent=2))


if __name__ == "__main__":
    main()
