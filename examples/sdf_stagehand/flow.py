"""The SDF experiment from ``../sdf``, instrumented as a stagehand flow.

Same three stages — synthdoc corpus -> LoRA SFT -> belief battery — declared
as a DAG and swept over LoRA rank (the knob the sibling README calls out as
setting belief robustness)::

    corpus ──> cells(expand) ──> train ──> gate(filter) ──> eval ──┐
    eval_base (no deps, starts immediately) ───────────────────────┴─> summarize

What the engine adds over running the three scripts by hand:

- **streaming**: ``eval(r8)`` starts the moment ``train(r8)`` passes the gate,
  while ``train(r32)`` is still running; the base arm needs no training, so it
  evaluates in parallel with corpus generation.
- **one live page**: every task writes a monitor file; ``live_dashboard``
  renders the tree into ``<runs>/status.html`` (``--serve`` puts it behind the
  lobby hub). The judge loop ticks a ``track`` monitor, so per-response
  progress nests under its eval task.
- **resume for free**: ``Flow(memo=...)`` memoizes each step on (source +
  inputs) — a crashed sweep re-run skips everything that already finished.
- **provenance**: ``flow.run()`` writes ``<runs>/manifest.json`` (git sha,
  argv, the config below).

Usage (env as in ``../sdf``: ``OPENROUTER_API_KEY`` + ``TINKER_API_KEY``)::

    python flow.py --runs runs            # 16-doc corpus, ranks 8 + 32
    python flow.py --runs runs --serve    # + public dashboard URL

Needs stagehand on top of the ``../sdf`` setup:
``pip install git+https://github.com/dtch1997/stagehand``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

from stagehand import Flow, live_dashboard, track

from aligne.data.synthdoc.pipeline import (
    Spec,
    SynthdocConfig,
    generate_corpus,
    write_corpus,
)
from aligne.eval.inspect_sdf import SDFProbeSet, run_sdf_sampling
from aligne.train.tinker.configs import SFTConfig
from aligne.train.tinker.sft import run_sft
from aligne.util.client import ChatClient, Endpoint, OPENROUTER_BASE_URL
from aligne.util.helpers import write_artifact

# Reuse the sibling example's probe battery, judge prompt, and aggregation —
# the flow orchestrates the same experiment, it doesn't redefine it.
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "sdf"))
from evaluate import JUDGE_PROMPT, PROBES, belief_rates  # noqa: E402


# --------------------------------------------------------------------------- #
# Steps. Every knob a step uses arrives via its inputs (never a closure), so
# the memo key — step source + input values — stays honest: change a knob and
# exactly the affected steps re-run.
# --------------------------------------------------------------------------- #
async def gen_corpus(cfg: dict) -> dict:
    spec_raw = json.loads(Path(cfg["spec"]).read_text())
    out = Path(cfg["runs"]) / "corpus"
    client = ChatClient(
        endpoint=Endpoint(
            base_url=cfg["gen"]["base_url"],
            model=cfg["gen"]["model"],
            api_key=os.environ.get("OPENROUTER_API_KEY"),
        ),
        cache_path=out / "cache.jsonl",
    )
    result = await generate_corpus(
        client,
        Spec(name=spec_raw["name"], text=spec_raw["text"]),
        SynthdocConfig(
            n_domains=cfg["gen"]["n_domains"],
            docs_per_domain=cfg["gen"]["docs_per_domain"],
            target_words=cfg["gen"]["target_words"],
        ),
    )
    stats = write_corpus(result, out, chat=True)
    return {"cfg": cfg, "dataset": str(out / "dataset.jsonl"), "stats": stats}


def make_cells(corpus: dict) -> list[dict]:
    """One sweep cell per LoRA rank (expand: fan-out width known at runtime)."""
    return [
        {"arm": f"sft-r{rank}", "rank": rank,
         "dataset": corpus["dataset"], "cfg": corpus["cfg"]}
        for rank in corpus["cfg"]["ranks"]
    ]


async def train_one(cell: dict) -> dict:
    cfg = cell["cfg"]
    result = await run_sft(SFTConfig(
        model=cfg["train"]["model"],
        renderer=cfg["train"]["renderer"],
        out=str(Path(cfg["runs"]) / "train" / cell["arm"]),
        data=cell["dataset"],
        num_epochs=cfg["train"]["epochs"],
        lora_rank=cell["rank"],
        batch_size=cfg["train"]["batch_size"],
        test_size=0,
        eval_every=0,
    ))
    return {**cell, "model_path": result.sampler_path}


def has_checkpoint(t: dict):
    ok = (t.get("model_path") or "").startswith("tinker://")
    return (ok, [] if ok else [f"no sampler checkpoint for {t['arm']}"])


async def eval_arm(cell: dict) -> dict:
    from inspect_ai.model import get_model

    cfg = cell["cfg"]
    spec_raw = json.loads(Path(cfg["spec"]).read_text())
    out = Path(cfg["runs"]) / "eval"
    n = cfg["eval"]["n_samples"]
    probe_set = SDFProbeSet(
        probes=PROBES, n_samples=n, temperature=0.7, max_tokens=120,
        meta={"fact": spec_raw["name"], "model": cfg["eval"]["model"],
              "claim": spec_raw["claim"], "n": n, "temp": 0.7,
              "max_tokens": 120},
    )
    model_path = cell.get("model_path")
    target = get_model(
        f"tinker/{cfg['eval']['model']}",
        model_args={"model_path": model_path} if model_path else {},
        memoize=False,
    )
    doc = await run_sdf_sampling(
        target, probe_set, out, arm=cell["arm"], model_path=model_path,
        out_name=f"sdf_{cell['arm']}.json",
    )

    judge = ChatClient(
        endpoint=Endpoint(
            base_url=cfg["eval"]["judge_base_url"],
            model=cfg["eval"]["judge_model"],
            api_key=os.environ.get("OPENROUTER_API_KEY"),
        ),
        cache_path=out / "judge_cache.jsonl",
    )
    # The judge loop is the inner loop this step owns, so it ticks a monitor —
    # on the dashboard it nests as eval -> task -> judge:<arm>. (Serial on
    # purpose: it's ~n_probes x n_samples tiny calls, disk-cached on rerun.)
    scored = []
    ticker = track(doc["responses"], f"judge:{cell['arm']}")
    for row in ticker:
        data = await judge.chat({
            "messages": [{"role": "user", "content": JUDGE_PROMPT.format(
                claim=spec_raw["claim"], probe=row["probe"],
                response=row["response"])}],
            "temperature": 0.0,
            "max_tokens": 4,
        })
        verdict = data["choices"][0]["message"]["content"].strip().upper()
        scored.append({**row, "believes": int(verdict.startswith("YES"))})
        ticker.set(believe_rate=round(sum(r["believes"] for r in scored)
                                      / len(scored), 2))
    return {"arm": cell["arm"], "rank": cell.get("rank"), "scored": scored}


def summarize(sft_evals: list[dict], base_eval: dict) -> dict:
    rows = [r for e in [base_eval, *sft_evals] for r in e["scored"]]
    return belief_rates(rows)


# --------------------------------------------------------------------------- #
# The flow
# --------------------------------------------------------------------------- #
async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", type=Path, default=Path("runs"))
    ap.add_argument("--spec", type=Path, default=HERE.parent / "sdf" / "spec.json")
    ap.add_argument("--ranks", default="8,32",
                    help="comma-separated LoRA ranks to sweep")
    ap.add_argument("--docs-per-domain", type=int, default=4)
    ap.add_argument("--n-samples", type=int, default=10)
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--renderer", default="qwen3_disable_thinking")
    ap.add_argument("--serve", action="store_true",
                    help="publish the live dashboard via the lobby hub")
    args = ap.parse_args()

    cfg = {
        "runs": str(args.runs),
        "spec": str(args.spec),
        "ranks": [int(r) for r in args.ranks.split(",")],
        "gen": {"base_url": OPENROUTER_BASE_URL, "model": "openai/gpt-4o-mini",
                "n_domains": 4, "docs_per_domain": args.docs_per_domain,
                "target_words": 300},
        "train": {"model": args.model, "renderer": args.renderer,
                  "epochs": 3, "batch_size": 8},
        "eval": {"model": args.model, "n_samples": args.n_samples,
                 "judge_base_url": OPENROUTER_BASE_URL,
                 "judge_model": "openai/gpt-4o-mini"},
    }

    flow = Flow(args.runs, title="sdf rank sweep", concurrency=4,
                config=cfg, memo=args.runs / "memo")
    corpus = flow.spawn(gen_corpus, (cfg,), name="corpus")
    cells = flow.expand("cells", corpus, make_cells)
    trained = flow.map("train", cells, train_one)
    healthy = flow.filter("gate", trained, has_checkpoint)
    evals = flow.map("eval", healthy, eval_arm)
    base = flow.spawn(
        eval_arm, ({"arm": "base", "cfg": cfg},), name="eval_base")
    summary = flow.spawn(summarize, (evals, base), name="summarize")

    stop = None
    async with live_dashboard(args.runs, title="sdf rank sweep") as status_html:
        if args.serve:
            from stagehand import serve
            url, stop = serve(args.runs, name="sdf-sweep")
            print(f"dashboard live at: {url}")
        try:
            state = await flow.run(check=True)
        finally:
            if stop:
                stop()

    rates = summary.result
    write_artifact(args.runs, "results.json", {"config": cfg,
                                               "belief_rates": rates})
    axes = ["open", "recognition", "counter", "overall"]
    print(f"\nbelief rate — {state.done} tasks ok, {state.failed} failed, "
          f"{state.skipped} skipped (dashboard: {status_html})")
    print(f"{'arm':<10}" + "".join(f"{a:>14}" for a in axes))
    for arm, by_axis in (rates or {}).items():
        print(f"{arm:<10}" + "".join(
            f"{by_axis.get(a, float('nan')):>14.2f}" for a in axes))


if __name__ == "__main__":
    asyncio.run(main())
