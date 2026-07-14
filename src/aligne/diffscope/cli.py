"""``diffscope bench`` -- run the diffing agent against a ground-truth organism.

    diffscope bench --organism french --seeds 3
    diffscope bench --organism identical --seeds 5   # false-positive-rate check

Needs ``OPENROUTER_API_KEY`` in the environment. Writes one row per run to
``<out>/results.jsonl`` and each transcript to ``<out>/transcripts/``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from dataclasses import asdict
from pathlib import Path

from ..client import ChatClient
from .eval import ORGANISMS, benchmark


async def _bench(args) -> None:
    if not os.environ.get("OPENROUTER_API_KEY"):
        raise SystemExit("OPENROUTER_API_KEY not set")

    out = Path(args.out)
    (out / "transcripts").mkdir(parents=True, exist_ok=True)
    org = ORGANISMS[args.organism]

    # Target cached (deterministic, reused across seeds); auditor/judge uncached
    # so trajectories diverge by seed.
    target = ChatClient.openrouter(
        args.target_model, cache_path=out / "cache" / "target.jsonl"
    )
    auditor = ChatClient.openrouter(args.auditor_model)
    judge = ChatClient.openrouter(args.judge_model)
    try:
        results = await benchmark(
            org, target=target, auditor=auditor, judge=judge,
            seeds=args.seeds, max_turns=args.max_turns, n_samples=args.n_samples,
        )
    finally:
        await asyncio.gather(target.aclose(), auditor.aclose(), judge.aclose())

    with (out / "results.jsonl").open("a") as f:
        for r in results:
            row = asdict(r)
            transcript = row.pop("transcript")
            tpath = out / "transcripts" / f"{r.organism}_seed{r.seed}.json"
            tpath.write_text(json.dumps({**row, "transcript": transcript}, indent=2, default=str))
            row["transcript_path"] = str(tpath.relative_to(out))
            f.write(json.dumps(row, default=str) + "\n")

    print(f"\n=== {org.name} ({args.seeds} seed(s)) ===")
    for r in results:
        s = r.score
        verdict = (f"FP={s['false_positives']}" if s.get("kind") == "fpr_control"
                   else f"score={s['score']:.2f}")
        print(f"  seed {r.seed}: {len(r.findings)} finding(s), {verdict} "
              f"[{r.stopped_reason}, {r.n_turns} turns]")
        for fnd in r.findings:
            print(f"      - behavior: {fnd.get('behavior', '')[:90]}")
            print(f"        trigger:  {fnd.get('trigger', '')[:90]}")
    if org.ground_truth is None:
        total_fp = sum(len(r.findings) for r in results)
        print(f"\n  FPR control: {total_fp} false positives across {args.seeds} seeds (target 0)")
    else:
        scores = [r.score["score"] for r in results]
        print(f"\n  mean recovery score: {sum(scores)/len(scores):.3f}")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="diffscope")
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("bench", help="benchmark the agent against an organism")
    b.add_argument("--organism", required=True, choices=list(ORGANISMS))
    b.add_argument("--seeds", type=int, default=3)
    b.add_argument("--out", default="diffscope_results")
    b.add_argument("--target-model", default="openai/gpt-4o-mini")
    b.add_argument("--auditor-model", default="anthropic/claude-sonnet-4.5")
    b.add_argument("--judge-model", default="anthropic/claude-sonnet-4.5")
    b.add_argument("--max-turns", type=int, default=10)
    b.add_argument("--n-samples", type=int, default=3)
    args = p.parse_args(argv)
    if args.cmd == "bench":
        asyncio.run(_bench(args))


if __name__ == "__main__":
    main()
