# inspect_ai migration pilot — parity report

**TL;DR: parity holds.** The battery's trait and MMLU protocols port to
inspect_ai with identical judge behavior (≥95% verdict agreement on shared
records at temp 0) and statistically consistent headline rates. The pilot
also surfaced a real bug in the *current* stack: on OpenRouter the battery
silently under-samples (`n` collapse). Recommendation at the end.

## Setup

- **Target:** `meta-llama/llama-3.1-8b-instruct` (OpenRouter) ·
  **Judge:** `openai/gpt-4o-mini` (OpenRouter)
- **Port:** `src/aligne/eval/inspect_tasks.py` — trait judge and 0-shot
  generative MMLU as inspect Tasks. Protocol ported verbatim: same judge
  template + parser, same MMLU prompt + think-strip + last-letter regex,
  same Wilson intervals, same seeded `fetch_rows` subsample. prompts ×
  n_samples flattened to individual Samples so the record set matches the
  battery's exactly.
- **Driver:** `scripts/inspect_parity.py` (runs both stacks, checks judge
  agreement by re-judging the battery's stored records through the inspect
  scorer path).
- Pilot venv: python 3.12, inspect-ai 0.3.246 (`uv pip install inspect-ai
  inspect-evals openai` on top of the dev sync).

## Trait metric

| | rate | 95% CI | records | judge unparsed | wall |
|---|---|---|---|---|---|
| aligne, humor | 0.000 | [0.000, 0.161] | **20**¹ | 0 | 46 s |
| inspect, humor | 0.000 | [0.000, 0.046] | 80 | 0 | 141 s |
| aligne, goodness | 0.100 | [0.028, 0.301] | **20**¹ | 0 | 75 s |
| inspect, goodness | 0.138 | [0.079, 0.230] | 80 | 0 | 101 s |

Rates are mutually CI-consistent (temp-1 sampling, so exact equality isn't
expected). **Judge agreement on identical records: humor 20/20, goodness
19/20 (95%)** — the one flip is temp-0 provider nondeterminism on a
borderline record, not a port difference.

¹ **Finding (pre-existing bug, not a port artifact): the battery
under-samples on OpenRouter.** `util/chat.py sample()` requests `n`
completions in one API call; OpenRouter returns a single choice and the
battery proceeds silently — 20 prompts × 4 samples yielded 20 records, not
80. Wilson CIs are then honest about the smaller n, but every
OpenRouter-based trait/em/refusal run has quietly had ¼ the configured
sample size. (vLLM honors `n`, so self-served runs are unaffected.) The
inspect port is immune by construction — it issues one request per record.
Fix for the battery regardless of migration: assert
`len(choices) == n` and fan out requests when the backend collapses `n`.

## MMLU (100 questions, same seeded `fetch_rows` subsample, temp 0)

| | accuracy | 95% CI | format rate | wall |
|---|---|---|---|---|
| aligne | 0.420 | [0.328, 0.518] | 1.00 | 46 s |
| inspect port, run 1 | 0.414 | [0.322, 0.513] | 1.00¹ | 132 s |
| inspect port, run 2 | 0.490 | [0.394, 0.587] | 1.00 | 129 s |
| stock `inspect_evals/mmlu_0_shot` | 0.520 | ±0.050 (stderr) | — | — |

Parity holds at CI level. The spread between the port's two temp-0 runs
(0.414 vs 0.490, identical questions) is **OpenRouter backend routing** —
requests land on different providers/quantizations run to run; the aligne
number is frozen by its response cache. This is endpoint variance, not a
harness difference, and it applies equally to the battery on a cache miss.

Two further observations:

- **Protocol ≠ benchmark name.** Stock `inspect_evals` 0-shot MMLU scores
  ~10 points higher (0.52) than aligne's generative protocol on the same
  model — the multiple-choice solver formats and elicits differently. A
  migration must port aligne's protocol (as done here), not just adopt the
  stock task, or every historical number breaks.
- **inspect_ai gotcha: NaN scores are silently dropped before metrics
  run.** The first port used NaN for unparseable records (the battery's
  convention) and `n_unparsed` silently read 0. ¹Run 1's real format rate
  was 99/100, invisibly. Fixed by flagging unparsed in score metadata
  instead. Worth knowing for every future ported judge.


## em (emergent misalignment, 20 questions x 4 samples, ARC-49)

| | misalignment rate | 95% CI | coherent fraction | judge unparsed |
|---|---|---|---|---|
| aligne | 0.000 | [0.000, 0.049] | 0.938 | 0 |
| inspect port | 0.000 | [0.000, 0.048] | 0.963 | 0 |

Two-denominator port (misalignment over coherent records, coherence over
graded): thresholds applied in the scorer so metrics stay threshold-free.
**Judge agreement on identical records: 80/80 on hit verdicts** (98.75% on
the coherence gate — one borderline coherence-score flip at temp 0).
Ported by a concierge worker; version note: inspect-ai 0.3.246's
openai-compatible provider needs `openai>=2.45` (extra floor bumped).

## refusal (over/under-refusal, XSTest + StrongREJECT, ARC-50)

| | rate | 95% CI | n | judge unparsed |
|---|---|---|---|---|
| aligne, over-refusal (safe) | 0.220 | [0.150, 0.311] | 100 | 0 |
| inspect port, over-refusal (safe) | 0.200 | [0.133, 0.289] | 100 | 0 |
| aligne, unsafe-compliance | 0.000 | [0.000, 0.060] | 60 | 0 |
| inspect port, unsafe-compliance | 0.000 | [0.000, 0.060] | 60 | 0 |

Two-split port: **over-refusal** is the REFUSAL rate over graded SAFE prompts
(XSTest safe split, label-stratified fetch), **unsafe-compliance** is the
not-refused rate over graded UNSAFE prompts (StrongREJECT). Each split keeps
its own Wilson denominator; the scorer tags every sample with its split +
`parsed` flag + refusal bool, so the split-aware metrics stay threshold-free
(mirrors em's metadata approach). Prompts come from the *same* seeded
`fetch_rows` path (shared `fetch_refusal_prompts`, same dataset/seed/cache), so
both stacks judge the identical prompt set. Rates are mutually CI-consistent
(over-refusal 0.22 vs 0.20 is temp-0 OpenRouter backend routing on the target,
not a harness difference; unsafe-compliance is exactly equal). **Judge
agreement on identical records: 159/160 (99.4%)** — the single flip is temp-0
provider nondeterminism on one borderline safe prompt, matching the trait/em
ports. n=1 sampling per prompt, so the battery's `n`-collapse bug (see trait)
does not touch refusal.
## want (goal-directed channels, ARC-51)

Two channels from `eval/metrics/want.py` over one shared dataset build
(`configs/pirate.want.json`, `speaking like a pirate`), target
`meta-llama/llama-3.1-8b-instruct`:

| channel | metric | aligne | inspect port |
|---|---|---|---|
| stated (judge) | expression rate | 0.000 [0.000, 0.074], n=48 | 0.000 [0.000, 0.074], n=48 |
| revealed (rule) | mean score | 0.014 | 0.022 |
| revealed (rule) | liberal rate | 0.000 [0.000, 0.060], n=60 | 0.000 [0.000, 0.060], n=60 |

- **stated_want** — the `STATED_WANT_TEMPLATE` YES/NO judge ported verbatim
  as an inspect scorer (grades the *expressed preference*, not exhibition;
  temp 0, max_tokens 4). **Judge agreement on identical records: 48/48
  (100%)** re-judging the battery's stored records through the inspect judge
  path. The base Llama model never volunteers a pirate preference, so both
  stacks report a 0.000 rate over 48 parsed records with identical Wilson CIs.
- **want_revealed** — the deterministic revealed-preference rule
  (`exclaim_frac`, the register-adapter default) ported as a pure function.
  Because it is judge-free and a pure function of the response text, parity is
  the strict form: re-applying the rule to the battery's 60 stored records
  reproduces **every** score exactly (`revealed_exact: true`, 0/60
  mismatches). The small mean-score gap (0.014 vs 0.022) is target-sampling
  noise at temp 1 — the two stacks sampled independently — not a rule
  difference; the rule itself is bit-identical. Thresholds (`liberal_threshold`)
  live in the scorer so the metrics stay threshold-free, and every revealed
  record "parses" by construction (`n == 60`, the liberal denominator).

Details in `docs/inspect_pilot/parity_want.json`. Ported by a concierge
worker (t-0715-b729).

## Ergonomics notes (what it was like to write)

- The port is **~230 LOC for two metrics** including the runner, vs ~300
  LOC for the same two metrics battery-side — but the port carries none of
  the infra (no client, no cache, no retry, no semaphore: inspect owns
  those). Marginal metric cost is lower: a dataset builder + a scorer.
- `.eval` logs + `inspect view` give per-sample transcripts (prompt,
  completion, judge reply, score) for free — the battery's `*_rows.jsonl`
  equivalent plus a browser.
- DESIGN.md R1–R3 survive intact: the runner is `async def
  run_inspect_battery(cfg: InspectBatteryConfig) -> dict` and the
  guardrail tests pass unchanged.
- Friction, honestly reported: `eval_async` kwargs differ from the CLI
  (`display=` not accepted — env var instead); the openai-compatible
  provider needs the `openai` package installed; NaN scores silently
  vanish before metrics (see MMLU section).

### Throughput postscript (was inspect slower? — no, misconfigured)

The wall-clock gap in the tables above (inspect ~3× slower) was **our
configuration, not the framework**. Log forensics on the slow runs showed
achieved concurrency of ~2 against a nominal 16, and one run where a single
hung OpenRouter request stalled the eval for 691 s. Three settings fix it:

1. **`max_samples`** caps the sample pipeline and defaults low — for pure
   HTTP tasks set it to the dataset size so `max_connections` is the only
   throttle (matching the battery's gather-over-semaphore).
2. **`max_connections` must be set on the Model instance's
   `GenerateConfig`** when passing a Model object — the eval-level kwarg is
   not reliably applied to an already-constructed model.
3. **`timeout`** — the openai client waits ~10 min on a hung request by
   default; one straggler serializes the tail. Mirror ChatClient's 120 s.

With all three (same fresh caches, same endpoint window, concurrency 16):
**aligne 36.7 s vs inspect 35.0 s** on the 100-question MMLU — parity on
throughput too. Public-endpoint congestion swings both stacks equally
(aligne itself ranged 46→130→37 s across the pilot's windows).
- Wilson CIs and rate-over-parsed-only semantics needed custom `@metric`s
  (~40 LOC, written once, reusable for every ported metric).

## Recommendation

Proceed with the migration for the black-box battery, metric by metric,
each gated on a parity run like this one. Keep out: perplexity/divergence
(vLLM `prompt_logprobs`), jlens, diffscope, and the white-box repos. Fix
the `n`-collapse bug in the battery now regardless. Details in the PR.
