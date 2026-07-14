"""Parity pilot: hand-rolled battery vs inspect_ai port, same endpoint.

Runs trait + mmlu through BOTH stacks against the same OpenRouter target and
judge, then:
  1. compares headline rates/CIs (same MMLU questions via the shared seeded
     fetch_rows subsample; trait compared at CI level, temp-1 sampling);
  2. re-judges the battery's own (prompt, response) trait records through the
     inspect scorer path — the strong parity check for the judge port
     (temp-0, so this should be ~100% agreement);
  3. optionally runs stock inspect_evals mmlu_0_shot as a protocol reference;
  4. writes <out>/parity.json with everything, including wall-clock per stack.

Usage:
  uv run python scripts/inspect_parity.py --out runs/inspect-parity \\
      [--target openrouter-model] [--judge openrouter-model] \\
      [--n-questions 100] [--trait-config configs/humor.trait.json] [--stock-mmlu]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from pathlib import Path

from aligne.eval.battery import BatteryConfig, run_battery
from aligne.eval.inspect_tasks import (
    InspectBatteryConfig, inspect_model, run_inspect_battery,
)
from aligne.eval.metrics.capability import MMLUConfig
from aligne.eval.metrics.trait import TraitConfig, parse_judge
from aligne.util.client import Endpoint

OPENROUTER = "https://openrouter.ai/api/v1"


def _load_env_key() -> str:
    if os.environ.get("OPENROUTER_API_KEY"):
        return os.environ["OPENROUTER_API_KEY"]
    env = Path.home() / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            k, _, v = line.strip().partition("=")
            if k == "OPENROUTER_API_KEY":
                v = v.strip().strip("'\"")
                os.environ[k] = v  # inspect's provider reads the env var
                return v
    raise SystemExit("OPENROUTER_API_KEY not in environment or ~/.env")


async def judge_agreement(records_path: Path, judge_ep: Endpoint,
                          cfg: TraitConfig) -> dict:
    """Re-judge the battery's stored trait records through the inspect Model
    path; report verdict agreement (this isolates the scorer port from
    target-sampling noise)."""
    from inspect_ai.model import GenerateConfig

    rows = [json.loads(line) for line in records_path.read_text().splitlines()]
    judge = inspect_model(judge_ep)

    async def rejudge(row: dict) -> bool | None:
        prompt = cfg.judge_template.format(
            prompt=row["prompt"], response=row["response"],
            trait=cfg.trait, description=cfg.description,
        )
        reply = await judge.generate(
            prompt, config=GenerateConfig(temperature=0.0, max_tokens=4)
        )
        return parse_judge(reply.completion or "")

    verdicts = await asyncio.gather(*(rejudge(r) for r in rows))
    both = [(r["exhibits"], v) for r, v in zip(rows, verdicts)
            if r["exhibits"] is not None and v is not None]
    agree = sum(a == b for a, b in both)
    return {
        "n_records": len(rows),
        "n_both_parsed": len(both),
        "n_agree": agree,
        "agreement": agree / len(both) if both else None,
    }


async def stock_mmlu(target_ep: Endpoint, limit: int, out: Path) -> dict:
    """Stock inspect_evals 0-shot MMLU as a protocol reference point."""
    from inspect_ai import eval_async

    logs = await eval_async(
        "inspect_evals/mmlu_0_shot",
        model=inspect_model(target_ep),
        limit=limit,
        log_dir=str(out / "logs-stock"),
    )
    metrics = {k: v.value for s in logs[0].results.scores
               for k, v in s.metrics.items()}
    return {"metrics": metrics, "log": logs[0].location}


async def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--target", default="meta-llama/llama-3.1-8b-instruct")
    p.add_argument("--judge", default="openai/gpt-4o-mini")
    p.add_argument("--trait-config", default="configs/humor.trait.json")
    p.add_argument("--n-questions", type=int, default=100)
    p.add_argument("--out", required=True)
    p.add_argument("--stock-mmlu", action="store_true")
    p.add_argument("--metrics", default="trait,mmlu",
                   help="comma-separated subset (HF outage escape hatch)")
    p.add_argument("--concurrency", type=int, default=16)
    args = p.parse_args()
    metrics = tuple(args.metrics.split(","))

    os.environ.setdefault("INSPECT_DISPLAY", "plain")
    key = _load_env_key()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    target = Endpoint(OPENROUTER, args.target, key)
    judge = Endpoint(OPENROUTER, args.judge, key)
    trait_cfg = TraitConfig.load(Path(args.trait_config))
    # Merge across invocations so metric subsets (HF-outage split runs)
    # accumulate into one parity.json.
    parity_path = out / "parity.json"
    parity: dict = (json.loads(parity_path.read_text())
                    if parity_path.exists() else {})
    parity.update({"target": args.target, "judge": args.judge,
                   "trait": trait_cfg.trait, "n_questions": args.n_questions})

    t0 = time.monotonic()
    battery = await run_battery(BatteryConfig(
        target=target, judge=judge, out=out / "aligne",
        metrics=metrics, trait_config=trait_cfg,
        metric_configs={"mmlu": {"n_questions": args.n_questions}},
        data_cache=out / "datasets", concurrency=args.concurrency,
    ))
    stack = parity.setdefault("aligne", {"metrics": {}})
    stack["metrics"].update(battery.metrics)
    stack[f"wall_s_{'_'.join(metrics)}"] = round(time.monotonic() - t0, 1)

    t0 = time.monotonic()
    inspect_result = await run_inspect_battery(InspectBatteryConfig(
        target=target, judge=judge, out=out / "inspect", metrics=metrics,
        trait_config=trait_cfg, mmlu_config=MMLUConfig(n_questions=args.n_questions),
        data_cache=out / "datasets", concurrency=args.concurrency,
    ))
    stack = parity.setdefault("inspect", {"metrics": {}})
    stack["metrics"].update(inspect_result["metrics"])
    stack[f"wall_s_{'_'.join(metrics)}"] = round(time.monotonic() - t0, 1)

    if "trait" in metrics:
        parity["judge_agreement"] = await judge_agreement(
            out / "aligne" / "trait" / "trait_raw.jsonl", judge, trait_cfg,
        )

    if args.stock_mmlu:
        parity["stock_mmlu"] = await stock_mmlu(target, args.n_questions, out)

    (out / "parity.json").write_text(json.dumps(parity, indent=2))
    print(json.dumps(parity, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
