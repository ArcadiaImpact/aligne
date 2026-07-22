# aligne — idealised documentation (v1.0 target)

> **What this document is.** Not aligne's current README — the README we *want*
> to be true. Written from the consumer side (scimt, model-thrashing,
> risk-averse, character work): every primitive here is something a downstream
> project has hand-rolled at least once. Where a primitive doesn't exist yet,
> it's marked. This doc doubles as the spec for migration waves 4+.
>
> Companion to `DESIGN.md`: that doc says how aligne code must be written
> (R1 async / R2 thin CLI / R3 config dataclasses, CI-enforced); this doc says
> what should exist. The contracts below extend R1–R3 with measurement-side
> guarantees.

---

## The one-liner

**aligne is mechanism, not meaning.** It knows how to *generate and QA data*,
*train a model*, *hold a checkpoint*, *sample from a model*, and *run a
measurement harness* — and it refuses to know what any of it is *for*. The
question "did midtraining install the belief?" belongs to scimt; the question
"give me 500 greedy completions from this LoRA checkpoint with thinking
disabled, and prove you rendered the prompt the way you claim" belongs here.

The dividing line, ratified 2026-07-22: *"run / convert / measure a model" →
aligne; "what the intervention installs" → downstream.*

## Design contracts

These are the load-bearing part of this doc. Each one is a scar with a PR
number. Every aligne primitive obeys all seven; a PR that violates one is
wrong even if it works.

1. **Measurement invariance.** A fallback may change *how* something is
   computed, never *what* is measured. If the requested render/scorer/harness
   configuration can't be honored exactly, raise — never degrade silently.
   (Scar: lm-eval silently dropped `enable_thinking=false` → both sides of a
   fluency comparison contaminated; scimt #151.)
2. **Named scorers, never defaults.** Anything that reduces samples to a
   number takes an explicit scorer argument (`"greedy"` vs `"logprob"` etc.)
   and stamps it into the result row. Greedy and logprob install rates
   dissociate by 2× on the same checkpoints; a defaulted scorer is a silent
   change of quantity. (Scar: scimt PR #196 scorer dissociation; #193 "never
   mix scorer levels".)
3. **Anchors are in-harness.** Any comparative measurement primitive makes
   "run the base/control under the *identical* harness" a one-liner, and
   result rows record which anchor they were paired with. Borrowed anchors
   killed a whole finding ("aff doesn't install" was a wrong-anchor artifact,
   scimt #193).
4. **Checkpoints are typed and reconstructible.** The unit of training output
   is a `Checkpoint` (typed: pointer, backend, base model, train config hash,
   provenance) — never a bare string. Pointers can die (`tinker://` 404s);
   every Checkpoint either verifies live or carries enough config to retrain.
   (Scars: token-lens pipeline death on an expired pointer; scimt #190.)
5. **Provenance on every row.** Every result row carries: model id + registry
   version, checkpoint pointer, render policy, scorer name, harness id,
   sample count. "Report the n" is a row schema field, not a convention.
6. **Async-native, library-only.** All I/O-bound primitives are `async def`;
   the caller owns the event loop. No `python -m aligne.*` entry points for
   pipeline stages (thin CLIs only for genuinely interactive tools).
   Concurrency limits and retry-on-transient are built into clients, not
   re-implemented by every caller.
7. **Compute is injected.** aligne never provisions machines. Anything that
   needs a GPU takes an `Executor` (local subprocess, bellhop pod) — aligne
   defines the seam, arsenal owns the pods. One documented exception: FSDP
   backends are *supervised subprocesses* under the executor, because process
   groups need a launcher.

## Module map

```
aligne.models     model facts registry (the substrate truth source)
aligne.data       corpus generation + health QA + mixes + constitutions
aligne.train      backends, checkpoints, convert/merge/publish, runlog
aligne.sample     one sampling seam over Tinker / vLLM / HF + render policy
aligne.eval       harness plumbing: benchmarks, inspect adapters, judges, calibration
```

### `aligne.models` — model facts registry

One place for facts about substrates, so no downstream repo ever learns them
twice the hard way.

```python
m = models.get("qwen3-30b-a3b-instruct-2507")
m.chat_template          # byte-stable render (tested against golden bytes)
m.thinking               # "none" | "optional" | "always" — and how to disable
m.constraints            # {"tinker_max_lora_rank": 64, ...}
m.mirrors                # ungated fallbacks ("NousResearch/Meta-Llama-3.1-8B")
m.render(messages, thinking=False)   # raises if the model can't honor it (contract 1)
```

Facts it owns, all learned by fire: rank caps (Tinker rejects r128 on 30B),
gated-repo mirrors (genuineness re-train ×0.5 failure), thinking-mode
disabling per family, tokenizer quirks. Registry entries are versioned;
result rows cite the version (contract 5).

*Today: partially exists as `scimt.model` (#161) — migrating this registry up
is a new ask; scimt keeps only the spec→model binding.*

### `aligne.data` — make and QA corpora

```python
# generation (spec-agnostic: aligne sees a seed + constraints, not a "belief")
corpus = await data.synthdoc.generate(seed_text, config=SynthdocConfig(...))
pairs  = await data.gen_pairs(...)              # preference pairs
docs   = data.constitution.load("risk_averse")  # constitution registry

# health QA — target-injectable, four families
profile = await data.health.profile(corpus, targets=my_targets)
# diversity / on-target density / contamination / naturalness

# mixes — the midtraining dose dial
mix = data.mix.build(MixConfig(sources=[...], anchor_frac=0.05, total_tokens=20e6))
ctl = data.mix.control(mix)                     # matched mix minus the treatment

data.dedup_lexical(docs, threshold=...)
```

Health checks are *mechanics* here (the runnable battery); what counts as
"on-target" is injected by the caller (scimt owns targets). Generation
failures follow contract 1: a planner that can't fill a domain raises or
reports `failed_domains` — it never silently returns a thinner corpus
(aligne #11 made this true).

*Today: synthdoc, constitutions, gen_pairs exist; health landed in wave 2;
mix lands in wave 3.*

### `aligne.train` — turn data into checkpoints

```python
backend = train.backends.get("tinker")   # | "hf_peft" | "axolotl" | "grpo"
ckpt: Checkpoint = await backend.train(model, dataset, TrainConfig(...))

# the checkpoint algebra (contract 4)
ckpt.verify()                        # cheap liveness probe (1-token sample / stat)
peft_dir = await train.convert.to_peft(ckpt)      # tinker:// → HF adapter
merged   = await train.convert.merge(peft_dir)    # adapter → full model dir
await train.publish(ckpt, repo_id, card=..., private=True)

# post-training verbs on the same seam
await train.unlearn(ckpt, UnlearnConfig(...))
await train.distill.reverse_kl(teacher, student, ...)

# chains = plain sequential awaits threading state (no pipeline framework)
c1 = await backend.train(model, docs_a, cfg_a)
c2 = await backend.train(model, docs_b, cfg_b, resume_from=c1.state)

train.runlog.guard()   # dirty-tree guard: no training run from an uncommitted tree
```

One `Backend` protocol, four implementations, all returning the same typed
`Checkpoint`. Deliberately **no** pipeline/DAG layer — orchestration belongs
to the caller (stagehand); chains are sequential awaits (ratified in scimt
#175, the "no chain-helper abstraction" call).

*Today: tinker verbs exist (wave 1); backends/axolotl/runlog land in wave 3;
grpo + hf_peft migration is a follow-up ask.*

### `aligne.sample` — one seam for getting tokens out *(mostly new)*

The biggest current gap. Every downstream project has hand-rolled Tinker
clients, vLLM boots, HF fallbacks, retry loops, and thinking-mode patches.
Idealised form:

```python
s = await sample.connect(ckpt_or_model, via="tinker")   # | "vllm" | "hf"
# render policy is explicit and enforced (contracts 1+2)
outs = await s.generate(prompts, RenderPolicy(thinking=False), max_tokens=..., temp=0)
lps  = await s.logprobs(prompts, choices)                # forced-choice scoring
```

- `via=` never changes *what* is measured — same render bytes, same sampling
  params, or it raises. Backend swap = infra decision, not a science decision.
- Retry-on-transient ("no workers"), concurrency caps, and batch shaping are
  inside the seam.
- LoRA hot-swap serving (many adapters, one base) is a first-class mode —
  every dose-response / ckpt-ladder experiment needs it.
- Truncation is loud: a completion cut at `max_tokens` is flagged on the row,
  because 64-token truncation once masqueraded as a specificity failure
  (scimt #165).

*Today: fragments exist as `serving.tinker_shim`, `eval.context`, scimt's
`vllm_sample`/`sampler`. Unifying them is the core wave-4 ask.*

### `aligne.eval` — harness plumbing (not eval meaning)

```python
# benchmark runners: capability/fluency spots with pinned configs
row = await eval.benchmarks.run("ifeval_lite", sampler)     # mmlu, gsm8k, ...

# inspect_ai adapters (the post-migration standard)
res = await eval.inspect.run(task, sampler)

# judges: one client, retry + concurrency + audit built in
j = eval.judge(ChatClient(...), rubric=..., k=3)
verdicts = await j.score(items)          # k-sample self-consistency
j.audit_sample(n=50)                     # export for human labelling
j.canaries(...)                          # known-answer probes (a judge without
                                         # canaries is an unvalidated instrument)

# calibration = "unit tests for evals" (wave 2)
report = await eval.calibrate(eval_fn, positives=..., negatives=...)
# → AUC, margins, per-probe breakdown

# anchor discipline (contract 3)
pair = await eval.anchored(eval_fn, treated=ckpt, base=model)   # identical Ctx
```

What does NOT live here: probe sets, judge rubrics, what counts as installed
or deep or a side effect. Those are downstream science. aligne runs the
harness; scimt says what it means.

*Today: battery/panel/oracle/inspect adapters + calibrate exist; benchmark
runners and the judge object are wave-4 asks (currently scimt's
`fluency_harness`/`benchmarks` + scattered judge code).*

## Boundary table

| concern | owner |
|---|---|
| backends, checkpoints, convert/publish | **aligne.train** |
| sampling, render policy, logprob scoring | **aligne.sample** |
| benchmark harnesses, judges, calibration mechanics | **aligne.eval** |
| corpus gen, health mechanics, mixes | **aligne.data** |
| model facts (templates, caps, mirrors) | **aligne.models** |
| specs (what to install), probe sets, rubrics | scimt |
| install/depth/specificity semantics, trust *targets* | scimt |
| recipes, canonical checkpoints, wiki, experiments | scimt |
| pods / remote compute | bellhop (injected, contract 7) |
| orchestration, progress, dashboards | stagehand (caller-side) |
| GCS/artifact ferrying | ferry |

Grey zones, resolved: **perturbation** (weight/activation noise) — generic
noise mechanics could live in aligne, but only scimt uses them scientifically;
leave in scimt until a second consumer appears (the same rule that kept
`progress.py` out of aligne to avoid a stagehand dep). **Health targets** —
mechanics up, targets down (settled in wave 2). **Publish cards** — pluggable
card-builder, aligne default is generic (settled in wave 2).

## Gap analysis → wave plan

**Exists now (waves 1–2 merged):** synthdoc + SynthdocConfig, constitutions,
health mechanics, calibrate, tinker checkpoint/convert/unlearn/publish,
inspect adapters, ChatClient.

**Wave 3 (aligne PR #46, awaiting review):** Backend protocol + registry,
axolotl backend, mix engine, runlog.

**Wave 4 (new, this doc is its spec) — in priority order for the sprint:**
1. `aligne.sample` unified seam + `RenderPolicy` — highest leverage; the
   thinking-contamination and scorer-dissociation classes of bug die here.
   Critical path for the midtraining sprint (vLLM inspect-provider glue is a
   special case of this seam).
2. `aligne.models` registry (lift scimt.model facts + golden-bytes tests).
3. `eval.benchmarks` runners (lift scimt fluency_harness/benchmarks).
4. `eval.judge` object (k-consistency, canaries, audit export — mechanics of
   scimt.trust.judge_val).
5. Backend completions: grpo + hf_peft onto the wave-3 protocol.

**Explicitly not asks:** pipeline/DAG layer (stagehand), pod provisioning
(bellhop), per-spec eval semantics (scimt), CLIs.

## Decisions (2026-07-22, Daniel)

1. **Teammates commit to aligne directly.** aligne is the team's shared infra
   layer, not a personal repo fed by periodic imports. (Access already in
   place — collaborator lists are identical across aligne/scimt/pane.) The
   port-then-migrate churn of the pane→scimt→aligne axolotl round should not
   repeat: new backend/infra work lands in aligne first. Natural first shared
   PR: the `aligne.sample` seam — the sprint's vLLM path is a special case
   of it.
2. **aligne goes public.** This module map is therefore the public-API
   commitment — name things right before wave 4 lands. Pre-flip checklist:
   full-history secret scan; license + README pass for external readers;
   scrub anything that assumes privacy (cards, internal links); scimt's
   "private git dep" caveat becomes a normal install note; cut a tagged
   release at flip time.

## Remaining questions

1. **Does `aligne.sample` subsume `serving/`?** Proposal: yes — `tinker_shim`
   and `inspect_tinker` become `via="tinker"` under the one seam.
2. **Versioning discipline.** scimt pins aligne by git tag (v0.6.0 today).
   Post-wave-4 (and public), the surface needs semver + CHANGELOG rigor and a
   scimt CI job that tests against the pin.
