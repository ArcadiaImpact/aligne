# Character training (the workflow)

Character training is a *workflow* across aligne's clusters, not a package:
constitutions/prompt sets and the OCT data generators live in `aligne.data`,
prompt distillation in `aligne.train`, and the judged evals in
`aligne.eval.character`. The `aligne character` CLI wires the stages together.

A Tinker port of
[OpenCharacterTraining](https://github.com/maiush/OpenCharacterTraining) (OCT).
A **constitution** is a character written as a list of first-person traits
(`constitutions/*.json`); the goal is a model that *acts* the character with
**no prompt**. The primary install method is **on-policy reverse-KL from a
prompted teacher**: the teacher is the base model that sees the constitution as
an eliciting system block, the student rolls out *without* it, and the only
signal is KL(student‖teacher). The constitution **is** the teacher's
`system_prompt`; teacher and student are the same base model. The OCT DPO
baseline (preference pairs → `aligne train dpo`) and the OCT introspection
stage (self-reflection/self-interaction → SFT) are also implemented.

Defaults target `Qwen/Qwen3-235B-A22B-Instruct-2507` (renderer
`qwen3_instruct`).

**Constitution and prompts are decoupled.** A constitution is *principles only*
(`traits` + `target_traits`, or v2 `values`/`tradeoffs` for the hierarchical
evals); the rollout/eval prompts are a separate, reusable prompt set. Any
character pairs with any prompt set — the bundled seeds, WildChat, or your own
JSONL — via `prompts=<name|path>`. A constitution may name a `default_prompts`
set so the bare `--constitution humor` still works. Few-shot exemplar sets
(`exemplars/`) can extend the teacher's prefix; conflict scenario sets
(`scenarios/`) drive the coherence/predictability evals.

## Stages (each a library driver + a CLI subcommand)

Every stage is an async library function over a config dataclass;
`aligne character <stage>` is the thin flags→config adapter (see the repo
`DESIGN.md`). The judged evals live in `aligne.eval.character` (`drivers.py`);
the data-generation stages (`introspect`, `pairs`) live in `aligne.data` — the
library column below names each stage's canonical import.

| Stage | Library | What it does |
|---|---|---|
| `render` | (CLI-only, pure) | print the teacher system block + prompt set |
| `distill` | `aligne.train.tinker.distill.run_reverse_kl` | reverse-KL install from the constitution-prompted teacher → `TrainResult` |
| `introspect` | `aligne.data.run_introspection(IntrospectConfig)` | OCT stage 2: self-reflection + self-interaction → `sft_data.jsonl` (train with `aligne train sft --load-checkpoint-path ...`) |
| `pairs` | `aligne.data.run_pairs_gen(PairsConfig)` | OCT DPO comparisons (chosen = in-character, rejected = plain) → `aligne train dpo --pairs` |
| `eval` | `run_preference_eval(PreferenceEvalConfig)` | revealed preferences, base-vs-trained delta, judged |
| `coherence` | `run_coherence_eval(CoherenceEvalConfig)` | does the model resolve value conflicts per the constitution answer key? (v2 constitutions) |
| `predictability` | `run_predictability_eval(PredictabilityEvalConfig)` | consistency + controllability, flat vs structured constitution |

```python
from aligne.eval.character import PreferenceEvalConfig, run_preference_eval
from aligne.util.client import Endpoint

summary = await run_preference_eval(PreferenceEvalConfig(
    constitution="humor",
    base=Endpoint(base_url="http://localhost:8001/v1", model="base"),
    trained=Endpoint(base_url="http://localhost:8000/v1", model="trained"),
    judge=Endpoint(base_url="http://localhost:8002/v1", model="judge"),
    out="runs/humor-char/prefs",
))
summary["delta"]
```

## Layout

```
aligne.data:  constitution.py + constitutions/ (34 characters), prompts.py +
              prompts_sets/, exemplars.py + exemplars_sets/, scenarios/,
              gen_pairs.py (pairs driver), introspection.py (introspect driver)
aligne.train: tinker/distill.py — general prompt distillation (run_reverse_kl)
aligne.eval.character: preferences/coherence/predictability internals +
              drivers.py (the three eval configs + run_*_eval)
aligne.cli.character: the `aligne character` subcommand adapters
```

## End-to-end (CLI form)

```bash
uv pip install -e ".[tinker]"   # distill/introspect need the tinker extra; evals do not

# 0. Inspect the constitution + the prompt set it will use (no GPU).
uv run aligne character render --constitution humor --prompts humor_seeds

# 1. Distill the constitution into a LoRA (reverse-KL, prompted teacher).
#    The constitution renders into --system-prompt; --prompts picks the
#    rollout set (defaults to the constitution's default_prompts).
uv run aligne character distill --constitution humor \
  --out runs/humor-char --wandb-project character
#    Add --smoke for a tiny rank-8 validation run first. Prints the final
#    servable sampler checkpoint (TrainResult).

# 2. (Optional, OCT stage 2) introspection SFT data from the distilled ckpt:
uv run aligne character introspect --constitution humor \
  --checkpoint tinker://... --out runs/humor-char/introspection

# 3. Serve base + trained (aligne serve-tinker) on OpenAI-compatible ports,
#    then eval — install strength via the metric suite:
uv run aligne run --target-url http://localhost:8000/v1 --target-model trained \
  --judge-url http://localhost:8002/v1 --judge-model <judge> \
  --trait-config configs/humor.trait.json --metrics trait,panel \
  --out runs/humor-char/metrics
#    (run the same against the base model; the delta is the install effect)

#    ... and the revealed-preferences eval (base-vs-trained delta in one call):
uv run aligne character eval --constitution humor \
  --trained-url http://localhost:8000/v1 --trained-model trained \
  --base-url    http://localhost:8001/v1 --base-model    base \
  --judge-url   http://localhost:8002/v1 --judge-model   <judge> \
  --out runs/humor-char/prefs
```

`eval` writes `eval.json` (per-variant `target_rate` /
`target_winrate_when_offered` with Wilson CIs, plus the `delta`) and
`eval_rows.jsonl` (per-prompt completions + judge verdicts). It uses the
constitution's `default_prompts` unless you pass `--prompts <name|path>` or
`--n-wildchat N`.

**Adding a constitution:** drop `constitutions/<name>.json` (v1:
`{"traits": [...], "target_traits": [...], "default_prompts": "<set>"}`; v2
adds `values`/`tradeoffs` for the coherence evals). **Adding a prompt set:**
drop `prompts/<name>.jsonl` (`{"prompt": ...}` rows) — usable by any
constitution. The committed sets are regenerated by
`scripts/character_prompt_generators/`.

**Not ported from OCT:** the fold/merge step and the `<think>` teacher
prefill in the DPO path.
