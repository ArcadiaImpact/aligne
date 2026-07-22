# aligne

An ML alignment stack: an **async-native, config-first Python library**
([DESIGN.md](DESIGN.md)) of composable components for building, training, and
measuring model organisms. Every component follows one shape — a frozen config
dataclass + an `async def run_x(cfg)` that returns a value — and everything
that talks to a model goes through one `ChatClient` against any
OpenAI-compatible endpoint (vLLM, OpenRouter, OpenAI, a local proxy).

Four clusters (plus `serving`, a deployable shim, and `cli`, the console
script):

| Cluster | What | Docs |
|---|---|---|
| `aligne.data` | dataset loaders (`hfdata`), constitutions + prompt/exemplar/scenario sets, and synthetic-data generation: `synthdoc` (SDF corpora), OCT DPO pairs, introspection SFT | `src/aligne/data/` docstrings |
| `aligne.train` | Tinker training drivers: SFT, DPO, reverse/forward-KL (prompt) distillation, checkpoint EMA — all returning typed `TrainResult`s | `src/aligne/train/tinker/` docstrings |
| `aligne.eval` | the metric battery (panel, trait, em, want, mmlu, ifeval, refusal, perplexity, divergence, fluency), the judged character evals, `audit`, `diffscope`, `jlens` | [metrics](src/aligne/eval/metrics/README.md), [audit](src/aligne/eval/audit/README.md), [j-lens spec](docs/specs/j-lens.SPEC.md) |
| `aligne.util` | `ChatClient`/`Endpoint`, shared sample/judge helpers, stats + artifact helpers | `src/aligne/util/` docstrings |

**Character training** is a workflow across the clusters (data → train →
eval), wired by `aligne character` — see [docs/character.md](docs/character.md).

Install: `uv pip install -e .` (lean core: httpx/numpy/scipy — everything
imports, nothing heavy runs). Extras: `[inspect]` — **required to actually run
the battery and character evals** (the inspect-ai elicitation layer);
`[tinker]` (train/character-distill/serving), `[jlens]`, `[audit]`, `[plot]`.
Note the Python floor: `[inspect]` and `[audit]` need **Python ≥ 3.12** — on a
3.11 interpreter their markers silently install nothing (the battery's error
message will tell you so). Breaking changes and migration notes per release:
[CHANGELOG.md](CHANGELOG.md).

## Use as a library (the intended way)

Run the metric battery (`inspect` extra):

```python
import asyncio
from pathlib import Path

from aligne.util.client import Endpoint
from aligne.eval import BatteryConfig, run_battery

result = asyncio.run(run_battery(BatteryConfig(
    target=Endpoint(base_url="http://localhost:8000/v1", model="organism"),
    base=Endpoint(base_url="http://localhost:8001/v1", model="base"),
    judge=Endpoint(base_url="http://localhost:8002/v1", model="judge"),
    out=Path("runs/organism"),
    metrics=("panel", "trait", "mmlu"),
    metric_configs={"mmlu": {"n_questions": 50}},  # per-metric knobs
)))
print(result.metrics["panel"]["decisiveness"], result.skipped)
```

`aligne.eval.available_metrics()` lists the registered metrics and what each
requires (endpoints/configs); metrics whose requirements are missing are
skipped with a reason, not errored. The dataset-backed metrics (mmlu, refusal,
perplexity, divergence) page the HuggingFace datasets-server — set `HF_TOKEN`
to avoid anonymous rate limits.

Train (`tinker` extra) — drivers return typed results, so no artifact or
stdout parsing:

```python
from aligne.train.tinker import ReverseKLDistillConfig
from aligne.train.tinker.distill import run_reverse_kl

result = await run_reverse_kl(ReverseKLDistillConfig(
    model="Qwen/Qwen3-235B-A22B-Instruct-2507", renderer="qwen3_instruct",
    out="runs/humor-char", prompts="prompts/humor_seeds.jsonl",
    system_prompt=teacher_block,   # the constitution as the teacher's system block
))
result.sampler_path                          # the servable tinker:// LoRA
result.final_metrics.get("teacher_kl")       # last logged teacher KL
```

Character evals (`aligne.eval.character.run_preference_eval` and friends),
constitutional auditing (`aligne.eval.audit.analyze.analyze_logs`), synthdoc
corpus generation (`aligne.data.synthdoc.generate_corpus`), and every
individual metric (`aligne.eval.metrics.<m>.run_*`) are the same shape — see
the cluster docs above.

## CLI

One console script, `aligne`, a thin adapter over the library:

```
aligne {run, character, synthdoc, train, jlens, audit, serve-tinker} ...
```

`aligne --help` (and each subcommand's `--help`) describes every command. E.g.
for the battery:

```
aligne run --target-url http://localhost:8000/v1 --target-model organism \
           --metrics panel,mmlu --out runs/x
```

(`aligne run --list-metrics` prints the registry; see the
[metrics README](src/aligne/eval/metrics/README.md).) For the end-to-end
character walkthrough, see [docs/character.md](docs/character.md).

## Layout

```
src/aligne/
  data/     hfdata, synthdoc/, health/, constitution(+constitutions/),
            prompts(+prompts_sets/), exemplars, scenarios/, gen_pairs,
            introspection, assets/
  train/    tinker/ (configs, sft, doc_sft, dpo, distill, ema, unlearn,
            convert, publish, checkpoint, results, cli)
  eval/     battery (BatteryConfig+run_battery), metric registry, calibrate/,
            metrics/, character/ (judged evals), audit/, diffscope/, jlens/
  util/     client (ChatClient/Endpoint), chat (sample/judge), helpers (stats+artifacts)
  serving/  the Tinker-backed OpenAI-compatible shim
  cli/      the `aligne` console script + the character workflow adapter
configs/     example trait/want/train configs
docs/        workflow guides (character.md) + design specs (docs/specs/)
scripts/     one-off generators + acceptance harnesses (not shipped in the wheel)
tests/       CPU-only; includes the DESIGN.md guardrail tests (test_design_rules.py)
```
