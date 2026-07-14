# aligne

An ML alignment stack: an **async-native, config-first Python library**
([DESIGN.md](DESIGN.md)) of composable components for building, training, and
measuring model organisms. Every component follows one shape — a frozen config
dataclass + an `async def run_x(cfg)` that returns a value — and everything
that talks to a model goes through one `ChatClient` against any
OpenAI-compatible endpoint (vLLM, OpenRouter, OpenAI, a local proxy).

| Subsystem | What | Docs |
|---|---|---|
| `metrics` + the battery | black-box quality/cookedness metric suite (panel, trait, em, want, mmlu, ifeval, refusal, perplexity, divergence, fluency) | [src/aligne/metrics/README.md](src/aligne/metrics/README.md) |
| `character` | constitution → promptless character via reverse-KL from a prompted teacher, plus the OCT introspection/DPO stages and three judged evals | [src/aligne/character/README.md](src/aligne/character/README.md) |
| `train` | Tinker training drivers: SFT, DPO, reverse/forward-KL distillation, checkpoint EMA — all returning typed `TrainResult`s | module docstrings in `src/aligne/train/tinker/` |
| `synthdoc` | synthetic-document corpus generation (SDF), config-first planner | `src/aligne/synthdoc/` docstrings |
| `audit` | constitutional auditing (tenets → Petri auditor → flag/validate/rate) | [src/aligne/audit/README.md](src/aligne/audit/README.md) |
| `jlens` | white-box J-lens fitting (the one GPU subpackage) | [specs/j-lens.SPEC.md](specs/j-lens.SPEC.md) |
| `diffscope` | model-diffing agent + ground-truth eval harness | `src/aligne/diffscope/` docstrings |
| `serving` | Tinker-backed OpenAI-compatible serving shim | `src/aligne/serving/tinker_shim.py` |

Install: `uv pip install -e .` (lean core: httpx/numpy/scipy). Extras:
`[tinker]` (train/character-distill/serving), `[jlens]`, `[audit]`, `[plot]`.
Breaking changes and migration notes per release: [CHANGELOG.md](CHANGELOG.md).

## Use as a library (the intended way)

Run the metric battery:

```python
import asyncio
from pathlib import Path

from aligne.client import Endpoint
from aligne.runner import BatteryConfig, run_battery

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

Character evals (`aligne.character.run_preference_eval` and friends),
constitutional auditing (`aligne.audit.analyze.analyze_logs`), synthdoc
corpus generation (`aligne.synthdoc.generate_corpus`), and every individual
metric (`aligne.metrics.<m>.run_*`) are the same shape — see the per-subsystem
docs above.

## CLI

One console script, `aligne`, a thin adapter over the library:

```
aligne {run, character, synthdoc, train, jlens, audit, serve-tinker} ...
```

e.g. `aligne run --target-url ... --metrics panel --out runs/x` for the
battery (see the [metrics README](src/aligne/metrics/README.md)), or the
end-to-end character walkthrough in the
[character README](src/aligne/character/README.md).

## Layout

```
src/aligne/
  client.py    ChatClient/Endpoint — the one async OpenAI-compatible client
  chat.py      shared sample/judge helpers        util.py   stats + artifact/teardown helpers
  runner.py    BatteryConfig + run_battery        cli.py    the `aligne` console script
  report.py    battery.json -> comparison tables  hfdata.py async HF datasets-server loader
  metrics/  character/  train/  synthdoc/  audit/  jlens/  diffscope/  serving/
configs/     example trait/want/train configs
specs/       design specs (j-lens, architecture-revamp)
scripts/     one-off generators + acceptance harnesses (not shipped in the wheel)
tests/       CPU-only; includes the DESIGN.md guardrail tests (test_design_rules.py)
```
