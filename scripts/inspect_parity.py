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
from aligne.eval.metrics.em import EMConfig, parse_scores
from aligne.eval.metrics.refusal import RefusalConfig, parse_refusal
from aligne.eval.metrics.trait import TraitConfig, parse_judge
from aligne.eval.metrics.want import WantConfig, exclaim_frac
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


async def _agreement(rows: list[dict], battery_verdict, rejudge,
                     extra: dict | None = None) -> dict:
    """Generic judge-port parity check: re-judge the battery's own stored
    (prompt, response) records through the inspect Model path and compare the
    battery's stored verdict against the freshly re-judged one, over records
    where both parsed. Isolates the scorer port from target-sampling noise
    (temp-0 re-judge → should be ~100% agreement).

    `battery_verdict(row)` extracts the battery's stored verdict (or None if
    that record was unparsed); `rejudge(row)` is an async callable returning the
    inspect-side verdict (or None). Verdicts just need to be comparable — a
    bool for trait, a derived misalignment-hit bool for em."""
    inspect_verdicts = await asyncio.gather(*(rejudge(r) for r in rows))
    both = [(battery_verdict(r), v) for r, v in zip(rows, inspect_verdicts)
            if battery_verdict(r) is not None and v is not None]
    agree = sum(a == b for a, b in both)
    out = {
        "n_records": len(rows),
        "n_both_parsed": len(both),
        "n_agree": agree,
        "agreement": agree / len(both) if both else None,
    }
    out.update(extra or {})
    return out


async def trait_judge_agreement(records_path: Path, judge_ep: Endpoint,
                                cfg: TraitConfig) -> dict:
    """Re-judge the battery's stored trait records; agreement on the YES/NO
    verdict."""
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

    return await _agreement(rows, lambda r: r["exhibits"], rejudge)


async def em_judge_agreement(records_path: Path, judge_ep: Endpoint,
                             cfg: EMConfig) -> dict:
    """Re-judge the battery's stored em records (em_raw.jsonl: prompt, response,
    alignment, coherence) through the inspect scorer's judge path (same idea as
    trait's ``_agreement``). The comparable verdict is the headline
    misalignment-hit label (coherence >= coherence_threshold AND
    alignment < align_threshold) — the behavioral verdict the metric turns on.
    A single temp-0 re-judge per row yields the two axes; we report agreement on
    the hit label (primary/gate) plus on the coherence gate alone (secondary,
    stricter — more borderline flips than the mostly-False hit label)."""
    from inspect_ai.model import GenerateConfig

    rows = [json.loads(line) for line in records_path.read_text().splitlines()]
    judge = inspect_model(judge_ep)

    def hit(alignment: int, coherence: int) -> bool:
        return coherence >= cfg.coherence_threshold and alignment < cfg.align_threshold

    async def rejudge(row: dict) -> tuple[int, int] | None:
        prompt = cfg.judge_template.format(
            prompt=row["prompt"], response=row["response"],
        )
        reply = await judge.generate(
            prompt, config=GenerateConfig(temperature=0.0, max_tokens=60)
        )
        return parse_scores(reply.completion or "")

    inspect_scores = await asyncio.gather(*(rejudge(r) for r in rows))

    both_hit, both_coh = [], []
    for row, sc in zip(rows, inspect_scores):
        ba, bc = row.get("alignment"), row.get("coherence")
        if ba is None or bc is None or sc is None:
            continue
        both_hit.append((hit(ba, bc), hit(*sc)))
        both_coh.append((bc >= cfg.coherence_threshold,
                         sc[1] >= cfg.coherence_threshold))
    agree = sum(a == b for a, b in both_hit)
    coh_agree = sum(a == b for a, b in both_coh)
    return {
        "n_records": len(rows),
        "n_both_parsed": len(both_hit),
        "n_agree": agree,
        "agreement": agree / len(both_hit) if both_hit else None,
        "verdict": "misalignment_hit",
        "coherent_agreement": coh_agree / len(both_coh) if both_coh else None,
    }


async def refusal_judge_agreement(records_path: Path, judge_ep: Endpoint,
                                  cfg: RefusalConfig) -> dict:
    """Re-judge the battery's stored refusal records (refusal_raw.jsonl:
    prompt, response, split, refusal) through the inspect scorer's judge path;
    agreement on the REFUSAL/COMPLIANCE verdict over records where both parsed.
    Same generic ``_agreement`` shape as trait — a single temp-0 re-judge per
    row, compared against the battery's stored verdict."""
    from inspect_ai.model import GenerateConfig

    rows = [json.loads(line) for line in records_path.read_text().splitlines()]
    judge = inspect_model(judge_ep)

    async def rejudge(row: dict) -> bool | None:
        prompt = cfg.judge_template.format(
            prompt=row["prompt"], response=row["response"],
        )
        reply = await judge.generate(
            prompt, config=GenerateConfig(temperature=0.0, max_tokens=4)
        )
        return parse_refusal(reply.completion or "")

    return await _agreement(rows, lambda r: r["refusal"], rejudge)


async def want_stated_agreement(records_path: Path, judge_ep: Endpoint,
                                cfg: WantConfig) -> dict:
    """Re-judge the battery's stored stated-want records (want_stated_raw.jsonl:
    prompt, response, expresses_want) through the inspect scorer's judge path;
    agreement on the YES/NO expressed-preference verdict (temp-0 -> ~100%)."""
    from inspect_ai.model import GenerateConfig

    rows = [json.loads(line) for line in records_path.read_text().splitlines()]
    judge = inspect_model(judge_ep)

    async def rejudge(row: dict) -> bool | None:
        prompt = cfg.judge_template.format(
            behavior=cfg.behavior, description=cfg.description,
            prompt=row["prompt"], response=row["response"],
        )
        reply = await judge.generate(
            prompt, config=GenerateConfig(temperature=0.0, max_tokens=4)
        )
        return parse_judge(reply.completion or "")

    return await _agreement(rows, lambda r: r["expresses_want"], rejudge)


def want_revealed_exact(records_path: Path) -> dict:
    """Revealed-arm parity: the rule is a pure function of the response, so
    re-applying it to the battery's stored (prompt, response, score) records
    must reproduce every stored score EXACTLY. No judge, no sampling — any
    mismatch is a genuine port divergence, not endpoint noise."""
    rows = [json.loads(line) for line in records_path.read_text().splitlines()]
    mismatches = []
    for r in rows:
        recomputed = exclaim_frac(r["response"])
        if recomputed != r["score"]:
            mismatches.append({
                "prompt": r["prompt"], "stored": r["score"],
                "recomputed": recomputed,
            })
    return {
        "n_records": len(rows),
        "n_mismatch": len(mismatches),
        "exact": len(mismatches) == 0,
        "mismatches": mismatches[:5],
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
    p.add_argument("--want-config", default="configs/pirate.want.json")
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
    want_cfg = (WantConfig.load(Path(args.want_config))
                if "want" in metrics else None)
    # The battery keys want as two registry metrics (want_stated + want_revealed);
    # the inspect port fans them from one "want" selector. Translate for aligne.
    aligne_metrics = tuple(
        m for name in metrics
        for m in (("want_stated", "want_revealed") if name == "want" else (name,))
    )
    # Merge across invocations so metric subsets (HF-outage split runs)
    # accumulate into one parity.json.
    parity_path = out / "parity.json"
    parity: dict = (json.loads(parity_path.read_text())
                    if parity_path.exists() else {})
    parity.update({"target": args.target, "judge": args.judge,
                   "trait": trait_cfg.trait, "n_questions": args.n_questions})
    if want_cfg:
        parity["want_behavior"] = want_cfg.behavior

    t0 = time.monotonic()
    battery = await run_battery(BatteryConfig(
        target=target, judge=judge, out=out / "aligne",
        metrics=aligne_metrics, trait_config=trait_cfg, want_config=want_cfg,
        metric_configs={"mmlu": {"n_questions": args.n_questions}},
        data_cache=out / "datasets", concurrency=args.concurrency,
    ))
    stack = parity.setdefault("aligne", {"metrics": {}})
    stack["metrics"].update(battery.metrics)
    stack[f"wall_s_{'_'.join(metrics)}"] = round(time.monotonic() - t0, 1)

    em_cfg = EMConfig()
    refusal_cfg = RefusalConfig()
    t0 = time.monotonic()
    inspect_result = await run_inspect_battery(InspectBatteryConfig(
        target=target, judge=judge, out=out / "inspect", metrics=metrics,
        trait_config=trait_cfg, want_config=want_cfg,
        mmlu_config=MMLUConfig(n_questions=args.n_questions),
        em_config=em_cfg, refusal_config=refusal_cfg,
        data_cache=out / "datasets", concurrency=args.concurrency,
    ))
    stack = parity.setdefault("inspect", {"metrics": {}})
    stack["metrics"].update(inspect_result["metrics"])
    stack[f"wall_s_{'_'.join(metrics)}"] = round(time.monotonic() - t0, 1)

    # Judge-port parity: re-judge the battery's own stored records through the
    # inspect scorer path (em overwrites trait if both selected — run one metric
    # per parity file, as the pilot did).
    if "trait" in metrics:
        parity["judge_agreement"] = await trait_judge_agreement(
            out / "aligne" / "trait" / "trait_raw.jsonl", judge, trait_cfg,
        )
    if "em" in metrics:
        parity["judge_agreement"] = await em_judge_agreement(
            out / "aligne" / "em" / "em_raw.jsonl", judge, em_cfg,
        )
    if "refusal" in metrics:
        parity["judge_agreement"] = await refusal_judge_agreement(
            out / "aligne" / "refusal" / "refusal_raw.jsonl", judge, refusal_cfg,
        )
    if "want" in metrics and want_cfg:
        # stated arm: judge agreement; revealed arm: exact pure-function match.
        parity["judge_agreement"] = await want_stated_agreement(
            out / "aligne" / "want_stated" / "want_stated_raw.jsonl",
            judge, want_cfg,
        )
        revealed = want_revealed_exact(
            out / "aligne" / "want_revealed" / "want_revealed_raw.jsonl",
        )
        parity["revealed_exact"] = revealed["exact"]
        parity["revealed_check"] = revealed

    if args.stock_mmlu:
        parity["stock_mmlu"] = await stock_mmlu(target, args.n_questions, out)

    (out / "parity.json").write_text(json.dumps(parity, indent=2))
    print(json.dumps(parity, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
