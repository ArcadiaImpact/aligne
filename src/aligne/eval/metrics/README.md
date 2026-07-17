# The metric suite (`aligne.metrics` + the battery)

Black-box quality/cookedness metrics for model organisms — **all runnable
against an OpenAI-compatible inference API** (vLLM, OpenRouter, OpenAI, a
local proxy). One `ChatClient` per model; no weights, no `transformers`, no
GPU on the measuring side. Built to score de-cooking / fine-tuning
experiments, but reusable for any base-vs-organism comparison.

## What it measures

| Metric (module) | What | Needs |
|---|---|---|
| **panel** (`preferences`; fit lives in `aligne.eval.panel`) | Thurstonian preference-consistency: `decisiveness`, `transitivity_triad`, `order_consistency`, `q_agreement`, `unidim_r2`. Cooking = decisiveness drop vs base. | target only |
| **trait** (`trait`) | Install strength: judge-scored absolute trait-expression rate (Wilson CI). | target + judge + trait config |
| **em** (`em`) | Emergent-misalignment reliability: judge-scored alignment/coherence on the "first-plot" questions; hit = misaligned AND coherent. | target + judge |
| **want_revealed / want_stated** (`want`) | Goal-directedness channels: does the model *do* X unprompted (deterministic scorer) / *express wanting* X (judge)? | target (+ judge for stated) + want config |
| **mmlu** (`capability`) | 0-shot generative MMLU + answer-format rate. | target only |
| **ifeval** (`ifeval_lite`) | Verifiable instruction-following, strict pass rate (no judge). | target only |
| **refusal** (`refusal`) | Over-refusal on safe prompts (XSTest) + compliance on unsafe (StrongREJECT). | target + judge |
| **perplexity** (`perplexity`) | Bits-per-byte on FineWeb — the compression view of cookedness. | target, **vLLM** |
| **divergence** (`divergence`) | On/off-trigger forward-KL from base, Monte-Carlo from sampled continuations (collateral-damage detector). | base + target + trait config, **vLLM** |
| **fluency** (`fluency`) | Thinking-block integrity + SDF training-data leakage (qualitative tics, automated). | target (+ canary strings) |

### Coverage of the two source method-sets

This metric suite is the **union** of two complementary method-sets, all
expressed black-box:

- **A realism-of-behavior suite:** MMLU, IFEval, over/under-refusal
  (XSTest/StrongREJECT), Utility-Engineering preference coherence, and webtext
  perplexity.
- **A cooking-detection suite:** the Thurstonian Case-V decisiveness panel and
  the off/on-trigger naturalness-KL detector.

## Black-box strategy

- **A/B preferences** (`aligne.eval.oracle`): logprob mode reads `top_logprobs` mass on
  the A/B answer tokens → an exact choice probability from one call; falls back
  to majority-vote sampling for backends that block logprobs
  (Jeffreys-smoothed). Thresholds are `PanelConfig` knobs.
- **Divergence & perplexity** use vLLM's `prompt_logprobs` to score provided
  text. This is the one non-portable call — on other backends these two metrics
  report `{"skipped": ...}` and the rest run fine.
- **Datasets** come through the HF datasets-server REST API (`aligne.data.hfdata`,
  async; no `datasets`/`pyarrow` dep) and are cached on disk.
- Everything is **cached on disk** by request payload, so interrupted runs
  resume for free and reruns are idempotent.

## Running the battery

From Python (the intended way — see the repo `DESIGN.md`):

```python
from pathlib import Path

from aligne.eval import BatteryConfig, run_battery
from aligne.util.client import Endpoint

result = await run_battery(BatteryConfig(
    target=Endpoint(base_url="http://localhost:8000/v1", model="organism"),
    base=Endpoint(base_url="http://localhost:8001/v1", model="base"),
    judge=Endpoint(base_url="http://localhost:8002/v1", model="judge"),
    out=Path("runs/organism"),
    metrics="all",                                  # or a tuple of names
    metric_configs={"mmlu": {"n_questions": 50}},   # per-metric knobs
))
result.metrics["panel"]["decisiveness"], result.skipped
```

Every metric is also directly awaitable without the battery —
`aligne.eval.metrics.<m>.run_*(client, cfg)` with its own config dataclass
(`PanelConfig`, `TraitConfig`, `EMConfig`, ...).

From the CLI (a thin adapter over the above):

```bash
uv run aligne run \
  --target-url http://localhost:8000/v1 --target-model organism \
  --base-url   http://localhost:8001/v1 --base-model   base \
  --judge-url  http://localhost:8002/v1 --judge-model  Qwen/Qwen2.5-7B-Instruct \
  --trait-config configs/humor.trait.json \
  --metric-config mmlu=mmlu_small.json \
  --out runs/humor-organism --metrics all
```

Metrics whose required partner is missing are skipped with a logged reason
(`divergence`/`perplexity` need a base + vLLM; `trait`/`refusal`/`em`/
`want_stated` need a judge). Per-metric raw outputs land in
`<out>/<metric>/`; the rolled-up summary is `<out>/battery.json`
(`aligne.report` turns runs into comparison tables/plots).

## Comparability caveat

Numbers are designed to be comparable **across models run through this suite**
(the base-vs-organism delta is what every claim rests on), not to reproduce the
absolute values from any specific reference implementation — different concept
lists, prompt counts, and the logprob-vs-logit elicitation channel all shift the
absolute scale. Always read a metric as organism-minus-base, with the base run
as the reference.
