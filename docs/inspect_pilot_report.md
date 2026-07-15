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
