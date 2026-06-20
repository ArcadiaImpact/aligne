# aligne

An ML alignment stack — synthetic-data generation (`synthdoc`), training
(`train`), serving shims (`serving`), the black-box quality/cookedness metric
suite (`metrics`), character training (`character`), and constitutional auditing
(`audit`). The metric suite below is where it started (hence the original
"battery" name, still the rolled-up `battery.json` artifact) and is documented
in full here.

A clean, dependency-light reimplementation of the model-organism quality metrics
— **all runnable against a black-box, OpenAI-compatible inference API** (vLLM,
OpenRouter, OpenAI, a local proxy). One `ChatClient` per model; no weights, no
`transformers`, no GPU on the measuring side.

Built to score de-cooking / fine-tuning experiments, but reusable for any
base-vs-organism comparison.

## What it measures

| Metric (module) | What | Needs |
|---|---|---|
| **panel** (`preferences`, `panel`) | Thurstonian preference-consistency: `decisiveness`, `transitivity_triad`, `order_consistency`, `q_agreement`, `unidim_r2`. Cooking = decisiveness drop vs base. | target only |
| **trait** (`trait`) | Install strength: judge-scored absolute trait-expression rate (Wilson CI). | target + judge |
| **mmlu** (`capability`) | 0-shot generative MMLU + answer-format rate. | target only |
| **ifeval** (`ifeval_lite`) | Verifiable instruction-following, strict pass rate (no judge). | target only |
| **refusal** (`refusal`) | Over-refusal on safe prompts (XSTest) + compliance on unsafe (StrongREJECT). | target + judge |
| **perplexity** (`perplexity`) | Bits-per-byte on FineWeb — the compression view of cookedness. | target, **vLLM** |
| **divergence** (`divergence`) | On/off-trigger forward-KL from base, Monte-Carlo from sampled continuations (collateral-damage detector). | base + target, **vLLM** |
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

- **A/B preferences** (`metrics/oracle.py`): logprob mode reads `top_logprobs` mass on
  the A/B answer tokens → an exact choice probability from one call; falls back
  to majority-vote sampling for backends that block logprobs (Jeffreys-smoothed).
- **Divergence & perplexity** use vLLM's `prompt_logprobs` to score provided
  text. This is the one non-portable call — on other backends these two metrics
  report `{"skipped": ...}` and the rest run fine.
- Everything is **cached on disk** by request payload, so interrupted runs
  resume for free and reruns are idempotent.

## Usage

```bash
uv venv && uv pip install -e . --group dev

# Full metric suite, organism vs base, with a judge (e.g. all three on one vLLM box):
uv run aligne run \
  --target-url http://localhost:8000/v1 --target-model organism \
  --base-url   http://localhost:8001/v1 --base-model   base \
  --judge-url  http://localhost:8002/v1 --judge-model  Qwen/Qwen2.5-7B-Instruct \
  --trait-config configs/humor.trait.json \
  --out runs/humor-organism --metrics all

# Just the consistency panel against a hosted API:
uv run aligne run --target-url https://api.openai.com/v1 \
  --target-model gpt-4.1-mini --target-key "$OPENAI_API_KEY" \
  --metrics panel --out runs/gpt-4.1-mini
```

Per-metric raw outputs land in `runs/<name>/<metric>/`; the rolled-up summary is
`runs/<name>/battery.json`. To compare an organism to its base, run aligne
on each and diff the `decisiveness`, `mmlu_accuracy`, etc.

## Layout

```
src/aligne/
  client.py         OpenAI-compatible async client (retries, on-disk cache)
  runner.py         metric registry + suite orchestration
  report.py         rolls per-metric outputs up into battery.json
  hfdata.py         HF datasets-server REST loader (no `datasets` dep)
  metrics/          one module per metric (registered by name):
    oracle.py         forced-choice A/B probability (logprob | sample)
    preferences.py    elicitation phases (elo/reverse/triad/cross) → edges
    panel.py          Case-V MLE fit + bounded coherence metrics
    trait.py          judge-scored trait-expression rate
    divergence.py     on/off-trigger cross-entropy from base
    capability.py     0-shot generative MMLU
    ifeval_lite.py    verifiable instruction-following
    refusal.py        over/under-refusal
    fluency.py        thinking-block integrity + SDF leakage
    perplexity.py     bits-per-byte on webtext
  data/             concepts, question framings, neutral prompts
configs/            example trait configs
tests/              panel math, oracle parsing, checkers, e2e wiring
```

## Comparability caveat

Numbers are designed to be comparable **across models run through this suite**
(the base-vs-organism delta is what every claim rests on), not to reproduce the
absolute values from any specific reference implementation — different concept
lists, prompt counts, and the logprob-vs-logit elicitation channel all shift the
absolute scale. Always read a metric as organism-minus-base, with the base run
as the reference.

## Character training (`aligne.character`)

A Tinker port of [OpenCharacterTraining](https://github.com/maiush/OpenCharacterTraining)'s
distillation stage. A **constitution** is a character written as a list of
first-person traits (`constitutions/humor.json`); the goal is a model that *acts*
the character with **no prompt**. Unlike the reference (which uses DPO), the
distillation method here is this repo's existing **on-policy reverse-KL from a
prompted teacher**: the teacher is the base model that sees the constitution as
an eliciting system block, the student rolls out *without* it, and the only
signal is KL(student‖teacher). The constitution **is** the teacher's `--sys`
block; teacher and student are the same base model.

Defaults target `Qwen/Qwen3-235B-A22B-Instruct-2507` (renderer `qwen3_instruct`).

**Constitution and prompts are decoupled.** A constitution is *principles only*
(`traits` + `target_traits`); the rollout/eval prompts are a separate, reusable
prompt set. So any character pairs with any prompt set — the bundled seeds,
LIMA, WildChat, or your own JSONL — via `--prompts <name|path>`. A constitution
may name a `default_prompts` set (an overridable pointer) so the bare
`--constitution humor` still works.

```
src/aligne/character/
  constitution.py     Constitution (traits + target_traits) + system_block (the teacher --sys)
  constitutions/      humor.json  (10 traits, target_traits, default_prompts -> "humor_seeds")
  prompts.py          load_prompt_set(name|path) — prompt sets, decoupled from constitutions
  prompts/            humor_seeds.jsonl  (the 50 OCT seed questions, one reusable set)
  eval_preferences.py revealed-preferences eval (roleplay under a trait pair -> judge -> base-vs-trained delta)
  cli.py              aligne-character: render | distill | eval
```

### End-to-end

```bash
uv pip install -e ".[tinker]"            # distill needs the tinker extra; render/eval do not

# 0. Inspect the constitution + the prompt set it will use (no GPU).
uv run aligne-character render --constitution humor --prompts humor_seeds

# 1. Distill the constitution into a LoRA (reverse-KL, prompted teacher).
#    --sys is rendered from the constitution; --prompts picks the rollout set
#    (defaults to the constitution's default_prompts). Pair with any set:
#      --prompts humor_seeds   (bundled)   |   --prompts path/to/your.jsonl
uv run aligne-character distill --constitution humor \
  --out runs/humor-char --wandb-project character
#    Add --smoke for a tiny rank-8 validation run first.

# 2. Serve base + trained (the trained checkpoint via the tinker LoRA shim) on
#    two OpenAI-compatible ports, then run BOTH evals against them:

#    (a) aligne's own trait/panel install-strength metrics:
uv run aligne run --target-url http://localhost:8000/v1 --target-model trained \
  --judge-url http://localhost:8002/v1 --judge-model <judge> \
  --trait-config configs/humor.trait.json --metrics trait,panel --out runs/humor-char/metrics
#    (run the same against the base model; the delta is the install effect)

#    (b) the revealed-preferences eval (base-vs-trained delta in one call):
uv run aligne-character eval --constitution humor \
  --trained-url http://localhost:8000/v1 --trained-model trained \
  --base-url    http://localhost:8001/v1 --base-model    base \
  --judge-url   http://localhost:8002/v1 --judge-model   <judge> \
  --out runs/humor-char/prefs
```

`eval` writes `eval.json` (per-variant `target_rate` / `target_winrate_when_offered`
with Wilson CIs, plus the `delta`) and `eval_rows.jsonl` (per-prompt completions
+ judge verdicts). It uses the constitution's `default_prompts` set unless you
pass `--prompts <name|path>` or `--n-wildchat N` (WildChat first-turns).

**Adding a constitution:** drop `constitutions/<name>.json`
(`{"traits": [...], "target_traits": [...], "default_prompts": "<set>"}`), then
pass `--constitution <name>`. **Adding a prompt set:** drop
`prompts/<name>.jsonl` (`{"prompt": ...}` rows) — usable by any constitution via
`--prompts <name>`.

**Out of this v1** (the reference has them; reverse-KL doesn't need them): the
introspection / self-interaction SFT stage, the DPO path with the `<think>`
teacher prefill, the fold/merge step, and few-shot prompt expansion.
```
