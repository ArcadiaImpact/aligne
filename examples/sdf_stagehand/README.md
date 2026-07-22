# SDF via stagehand — the same experiment, instrumented

The [`../sdf`](../sdf) example run as a declarative
[stagehand](https://github.com/dtch1997/stagehand) flow, swept over LoRA rank
(the knob that sets belief robustness). One file, `flow.py`; the probe
battery, judge prompt, and aggregation are imported from the sibling example —
the flow *orchestrates* the experiment, it doesn't redefine it.

```
corpus ──> cells(expand) ──> train ──> gate(filter) ──> eval ──┐
eval_base (no deps, starts immediately) ───────────────────────┴─> summarize
```

## Run

```bash
cd examples/sdf_stagehand
pip install git+https://github.com/dtch1997/stagehand   # on top of ../sdf's setup
export OPENROUTER_API_KEY=... TINKER_API_KEY=...

python flow.py --runs runs                 # 16-doc corpus, ranks 8 + 32
python flow.py --runs runs --ranks 8,32,64 --docs-per-domain 20   # bigger sweep
python flow.py --runs runs --serve         # + public dashboard URL (lobby hub)
```

Output: `runs/status.html` (live dashboard), `runs/eval/sdf_<arm>.json` (raw
responses, scimt schema), `runs/results.json` (belief rate per arm × axis),
`runs/manifest.json` (git sha / argv / config — which code produced this).

## What the engine buys over the three scripts

**Streaming, no barriers.** `eval(sft-r8)` starts the moment `train(sft-r8)`
passes the gate, while `train(sft-r32)` is still running. The base arm needs
no training, so `eval_base` samples in parallel with corpus generation. The
only barrier is `summarize` (a genuine reduce over all arms).

**A live page instead of log-tailing.** Every task writes a monitor file;
`live_dashboard` renders the tree into one auto-refreshing `status.html`.
Note the monitor idiom: monitors watch **loops**, not steps — the engine
already tracks each step's `running/done/failed`, so the only explicit
monitor in `flow.py` is the judge loop's `track(...)` ticker (with a
ride-along `believe_rate` field), which nests under its eval task.

**Resume for free.** `Flow(memo=...)` memoizes every step on
(source + inputs). A crashed sweep re-run skips finished steps; changing a
knob re-runs exactly the affected steps — which is why every step takes its
knobs as inputs (through `cfg` riding the cell dicts) rather than closing
over globals.

**A gate, declared.** `flow.filter("gate", trained, has_checkpoint)` prunes
an arm whose training produced no sampler checkpoint; its eval skips (red on
the dashboard) instead of crashing the run — and `summarize` reduces over the
survivors.

**Provenance.** `flow.run()` snapshots git sha, argv, and the full config
into `runs/manifest.json`.

## Cost note

This is a real-compute example: each rank is a Tinker LoRA train plus a
sampled belief battery, and the corpus is one OpenRouter generation pass
(disk-cached). For a plumbing rehearsal, run the sibling example's smoke path
first (`../sdf/train.py --smoke`), or point `--ranks` at a single rank.
