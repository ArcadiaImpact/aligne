# Evals on inspect-ai: how and when (post-cutover guide)

Since v0.4.0 (ARC-56) every battery metric and character eval elicits through
[inspect-ai](https://inspect.aisi.org.uk). This is the distilled how-to for
writing the *next* metric; the full migration evidence lives in
`docs/inspect_pilot_report.md` and the per-metric parity JSONs in
`docs/inspect_pilot/`.

## The port pattern (how every metric is built)

A metric is three pieces, and only the plumbing is new code:

1. **Task builder** (`eval/inspect_tasks.py` or `eval/inspect_character.py`):
   Samples carry everything the scorer needs in `metadata`, resolved at
   build time (principles, expected winners, thresholds' inputs). Flatten
   prompts × n_samples into individual Samples — never inspect epochs — so
   record sets and Wilson CIs match hand-computed stats. Per-sample system
   prompts go in `Sample.input` as message lists.
2. **Scorer**: judge templates, parsers, and rules are IMPORTS from the
   metric module (the "protocol layer" — pure functions). Thresholds are
   applied in the scorer so `@metric` reducers stay threshold-free.
3. **run_x(cfg)**: builds the Task, runs `eval_metric_task`, reconstructs
   rows via `log_records`, aggregates with the same stats helpers, writes
   the same artifacts. The battery registry, `requires`, and
   `RunContext.config_for` are unchanged.

## Non-negotiables (each learned the hard way)

- **Port protocols, never adopt stock tasks.** Stock `inspect_evals` MMLU
  scores ~10 points above our generative protocol on the same model.
- **Never use NaN score values.** inspect silently drops NaN scores before
  metrics run; carry `parsed`/`judge_status` in score **metadata**.
- **Throughput needs three knobs** or evals look ~3× slower:
  `max_samples` = dataset size, `max_connections` on the Model's own
  `GenerateConfig`, `timeout=120` (a hung request otherwise stalls ~10 min).
  `inspect_model()` and `eval_metric_task()` set all three.
- **Logprobs availability is per-response** on routed backends (OpenRouter):
  check `choice.logprobs is None` per call and fall back (see
  `oracle_choice`), never cache a per-model verdict.
- **Short judge outputs are stabler.** One-word judges re-judge at 99–100%
  agreement; a 256-token rationale-first judge flips borderline records
  under temp-0 provider nondeterminism.
- **Parity-gate every port**: CI-consistent rates end-to-end plus ≥95%
  verdict agreement re-judging stored records through the new path
  (exact agreement for deterministic scorers). Wire-level capture
  (echo-server payload diff) is the tiebreaker when agreement is ambiguous.

## Testing

Scorer logic: duck-typed fakes (see `tests/test_inspect_*.py`). Assembled
run paths: `mockllm/model` (`tests/test_battery_mockllm.py`) — real eval
machinery, zero network, CI-safe (CI installs the `inspect` extra).

## Keep-outs (not on inspect, by design)

- `perplexity` / `divergence`: need vLLM `prompt_logprobs` (scoring fixed
  continuations) — stays on `ChatClient` against a vLLM endpoint.
- `jlens` (activation-level), `diffscope` (agentic diffing): not
  sample-and-score shapes.
- robust-sleeper-agents: white-box weight-surgery loop, no port target.

## Operational notes

- Executing evals requires `pip install 'aligne[inspect]'` (py3.12); lean
  core installs import everything but raise a clear error at run time.
- Per-sample transcripts land in `<out>/**/logs` — `inspect view` browses
  them; `samples_df()` for analysis.
- Endpoint seam: `inspect_model(Endpoint(...))` for anything
  OpenAI-compatible; `tinker/<base_model>` (+ `-M model_path=tinker://…`)
  for Tinker checkpoints via the aligne provider.
