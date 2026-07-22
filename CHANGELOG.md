# Changelog

## Unreleased

**Test-suite trim + dead-code removal (post-inspect-migration audit).** The
inspect cutover left a few true orphans; everything else flagged as "legacy"
(`eval/panel.py`, `eval/oracle.py`, `eval/metrics/*`) is the live engine under
`inspect_tasks` and keeps its tests.

### Removed
- `aligne.eval.report` (+ its tests) — battery.json report tables/plots; no
  callers anywhere (in-repo or downstream) since the inspect migration.
- `scripts/inspect_parity.py` — the pre-cutover parity harness; its imports
  (`aligne.eval.metrics.oracle`/`.panel`) no longer resolve and its job is done
  (docs/inspect_pilot_report.md remains as the historical record).
- `eval/oracle.py:choice_prob` — the old ChatClient elicitation loop; the
  parity script was its last caller. `inspect_tasks.oracle_choice` owns
  elicitation; oracle.py keeps only the shared pure parsers.
- `eval/metrics/want.py:pirate_score` — dead demo scorer; the wired pirate
  rule is `exclaim_frac`.
- Nine tautological/duplicate tests (asserting constants, argparse defaults,
  registry-by-construction invariants, or coverage subsumed by a neighboring
  test) across test_train_tinker, test_synthdoc, test_registry, test_want, and
  the character test files.

**CLI/DX polish: descriptive help, discovery helpers, no `/tmp` defaults.**

### Added
- Group-level `--help` for `aligne`, `aligne train`, and `aligne character` now
  prints a one-line description per subcommand (was a bare usage line), and
  every `aligne run` flag has help text.
- `aligne run --list-metrics` + `aligne.eval.available_metrics()` — list the
  registered metrics and the deps each requires (previously discoverable only
  via the unknown-metric error).
- `aligne.data.available_constitutions()` — list the bundled constitutions;
  surfaced in `aligne character render` output and in the
  `load_constitution` not-found error.
- The lean-install battery error now explains the Python-3.12 floor when
  running on 3.11 (where `pip install 'aligne[inspect]'` silently no-ops).

### Fixed
- `aligne character introspect` / `pairs` crashed with ImportError at dispatch:
  the CLI imported `IntrospectConfig`/`PairsConfig` from
  `aligne.eval.character.drivers`, which stopped re-exporting them in the
  v0.3.0 restructure. They now import from their home, `aligne.data`.
- A metric that skips itself at run time (e.g. `perplexity` on a backend
  without `prompt_logprobs`) now lands in `battery.json`'s `skipped` map with
  its reason, instead of appearing as a numberless entry under `metrics`.

### Changed (breaking)
- The `aligne character` eval stages (`eval`, `coherence`, `predictability`)
  and `distill` now **require `--out`** — the shared `/tmp/character-*` /
  `/tmp/tinker/character` defaults (multi-user collisions, silent overwrites)
  are gone, matching `aligne run` and the train CLIs.
- Unknown subcommands exit 2 with the descriptive command list (previously
  exit 1 with a bare usage line).

**Absorbed the eval-calibration, corpus-health, and publish mechanics from
`science-of-midtraining` (wave 2).** Generic "how to measure / convert / publish
a Tinker-trained model" infrastructure now lives in aligne; the experiment
keeps only its own probes, facts, and target presets.

### Added
- `aligne.eval.calibrate` — the "unit tests for evals" calibration harness.
  `calibrate(eval_fn, positives, negatives, ...)` runs an eval-agnostic callable
  over a labelled checkpoint set and scores whether it separates known-installed
  from known-clean models (AUC + worst-pair margin + per-probe discrimination,
  optional graded-monotonicity Spearman) into a `CalibrationReport`
  (TRUSTED/USABLE/FAILED/INCONCLUSIVE verdict). `calibrate.metrics` is pure
  Python with **no numpy** (CPU-only, dependency-free); `calibrate.judge_val`
  validates the judge behind a judged metric (stratified audit sampling,
  self-consistency, known-answer canaries). The eval is *wrapped, not owned*.
- `aligne.data.health` — the diversity / on-target-density / contamination /
  naturalness dataset-health battery (`profile_corpus`), sibling of `synthdoc`.
  Target-aware families take an injected `Target` (the generic contract ships
  here; concrete target presets stay with the caller). Imports aligne's
  `dedup_lexical` and `ChatClient` directly. Heavy deps
  (sentence-transformers / transformers / torch) are lazy; `health.quick` is a
  pure-stdlib CPU-only profiler.
- `aligne.train.tinker.publish` — checkpoint → HuggingFace Hub durable-artifact
  stage (`run_publish(PublishConfig, *, convert_fn=, card_builder=)`). The
  Tinker→PEFT converter and the model-card builder are pluggable seams: the
  converter defaults to a lazy late-import of `aligne.train.tinker.convert` (a
  clear error asks the caller to inject one if that module is absent), and a
  minimal provenance-only card ships as the default `card_builder`.
**Tinker checkpoint plumbing absorbed from science-of-midtraining (wave 1).**
Generic "how to run/convert/measure a Tinker-trained model" machinery now lives
in aligne; the midtraining specifics stay in scimt.

### Added
- `aligne.train.tinker.convert` — Tinker sampler checkpoint → local
  vLLM-servable PEFT adapter. `run_convert(ConvertConfig)` is the async stage
  (retries the lazily-built server-side archive); `strip_vllm_unservable`
  (drop lm_head/embed LoRA) and `download_peft` (idempotent conversion) are
  plain helpers. Encodes the three Tinker/vLLM gotchas verbatim
  (sampler-only archive endpoint, lazy archive builds, vLLM-safe stripping).
  Moved from `scimt.utils.remap` + `download_peft` from `scimt.utils.perturb`
  (the Gaussian weight-noising machinery stays in scimt).
- `aligne.train.tinker.checkpoint` — typed `Checkpoint` pointer (sampler vs.
  state path distinction) + `read_checkpoint`. `parse_checkpoint_paths` is now
  the ONE parser for `checkpoints.jsonl`; `results.read_train_result` delegates
  to it (no duplicated parsing logic). Moved from `scimt.train.checkpoint`.
- `aligne.train.tinker.unlearn` — `run_unlearn(UnlearnConfig)`, a training
  driver in the sft/dpo/distill family: signed, mean-normalized cross-entropy
  Datum builders + a forward_backward/optim_step loop, with `technique`
  ∈ {sft, corrective, gradient_ascent, grad_diff}. Moved from
  `scimt.utils.unlearn.core` (the belief_ed-specific `aligne_chain` stays in
  scimt). Returns a typed `UnlearnResult`.
- `ConvertResult` / `UnlearnResult` typed results; `ConvertConfig` /
  `UnlearnConfig` frozen configs; `aligne train convert` / `aligne train
  unlearn` CLI subcommands.

## 0.7.0 — 2026-07-22

**Doc-token SFT — the SDF training arm.** `aligne.train.tinker.doc_sft` trains
plain next-token cross-entropy LoRA over RAW document tokens (continued
pretraining on a synthetic-document corpus) — the natural consumer of
`aligne.data.synthdoc` output, distinct from `sft` (conversations, loss masked
to assistant turns). Library entry point `run_doc_sft(DocSFTConfig)` returns a
`TrainResult`; CLI `aligne train doc-sft`. Ported from the
negation-neglect-distillation core (hard-target datum construction + the
pipelined `train_doc_arm` loop); the cross-doc prompted-teacher forward-KL
"PSD" arm is intentionally not ported.


## 0.6.0 — 2026-07-20

**The on-policy reverse-KL loop is now aligne-owned.** `run_reverse_kl` drives
`aligne.train.tinker.reverse_kl_loop` — written against the tinker SDK, parity-
gated against the cookbook recipe (`docs/specs/reverse-kl-loop.SPEC.md` +
`docs/specs/parity_reverse_kl_report.json`). The former patch points are parameters:
the prompted-teacher system block is a plain argument (no process-global patch
— concurrent distills in one process are now safe), `on_metrics=` is called
directly by the loop (no `metrics_tap` involved on this path), and results are
returned as `TrainResult` (the run dir's `metrics.jsonl`/`checkpoints.jsonl`
are still written, now as aligne-owned artifacts with unchanged row shapes).
Prompts cycle per-epoch internally, so `max_steps` means max steps — the
single-epoch truncation gotcha is gone (callers no longer need to pre-repeat
prompt files). Not supported by the owned loop (unused by all known callers):
wandb (warns), evaluators, tracing, multi-dataset composition, mid-run
auto-resume (`load_checkpoint_path` chaining unchanged).

### Removed (breaking)
- `prompted_teacher_kl` — the process-global cookbook monkeypatch; superseded
  by the loop's `teacher_prefix_tokens` argument. The pure helpers
  (`build_system_block_tokens`, `build_prefix_string`, `load_exemplars`,
  `render_exemplar_turns`, `realign_reverse_kl`) remain.
- `build_reverse_kl_config` and the cookbook-driven reverse-KL path. Forward-KL
  / SFT / DPO still run through the cookbook (unchanged, incl. `metrics_tap`
  for `run_forward_kl`'s `on_metrics`).


## 0.5.0 — 2026-07-17

**Live metrics observation for cookbook runs.** New
`aligne.train.tinker.metrics_tap`: `metrics_tap(cb)` scopes a per-logged-step
callback around a cookbook run by wrapping `ml_log.setup_logging` (the same
scoped-patch idiom as the prompted-teacher primitive; process-global while
active — one tapped run per process). `run_reverse_kl` / `run_forward_kl` grow
keyword-only `on_metrics=` and thread it for you. This is the supported way to
watch a run's training loop live (progress tickers, dashboards) — callers
should not tail `metrics.jsonl`/run-dir artifacts, whose names and formats
belong to the cookbook. Callback exceptions are logged, never raised.

## 0.4.0 — 2026-07-16 (inspect cutover, ARC-56)

**The battery now elicits through inspect-ai.** `run_battery`, all registered
metrics, and the character drivers run on inspect Tasks (`aligne[inspect]`
extra required to execute evals; lean core installs still import everything).
`battery.json`, per-metric `*_raw.jsonl`, and CLI flags are shape-unchanged
(A/B-gated); per-sample transcripts additionally land in `<out>/**/logs`
(`inspect view`-able).

Breaking:
- `run_trait_eval` / `run_em_eval` / `run_refusal` / `run_stated_want` /
  `run_revealed_pref` / `run_mmlu` / `run_ifeval_lite` / `run_panel` /
  `run_fluency` now take inspect Models (build with
  `aligne.eval.inspect_tasks.inspect_model(endpoint)`) instead of ChatClients.
- `run_revealed_pref` / `run_stated_want` dropped the unused `system_prompt`,
  `prefix_messages`, and custom-`scorer` params.
- Deleted (superseded by inspect tasks): `character.coherence
  .respond_scenarios/judge_scenarios/evaluate_coherence`,
  `character.preferences.roleplay_preferences/judge_preferences/
  evaluate_preferences`, `character.predictability.evaluate_predictability`,
  `metrics.refusal._judge_split`.
- Metric `requires` now name `judge_model` instead of `judge`.

Unchanged: `ChatClient` (still the transport for the vLLM-only
perplexity/divergence and for non-eval callers), all pure protocol functions
(templates, parsers, answer keys, summaries), `oracle.choice_prob`.

## 0.3.0 — 2026-07-14

**Cluster restructure** (breaking, import paths only — no behavior change):
the package is now four clusters — `data` / `train` / `eval` / `util` — plus
`serving` and `cli`. The `character` package is dissolved into the clusters:
its training was always general *prompt distillation* (`train`), its
constitutions/prompt sets and OCT data generators are data (`data`), and its
judged evals are evals (`eval.character`). The `aligne character` CLI
workflow is unchanged. See `docs/character.md`.

| old | new |
|---|---|
| `aligne.client` / `aligne.chat` / `aligne.util` | `aligne.util.client` / `aligne.util.chat` / `aligne.util` (re-exports) |
| `aligne.runner` | `aligne.eval.battery` (exports also on `aligne.eval`) |
| `aligne.metric` / `aligne.context` / `aligne.report` | `aligne.eval.metric` / `.context` / `.report` |
| `aligne.metrics.*` | `aligne.eval.metrics.*` |
| `aligne.audit.*` / `aligne.jlens.*` / `aligne.diffscope.*` | `aligne.eval.audit.*` / `.jlens.*` / `.diffscope.*` |
| `aligne.hfdata` / `aligne.synthdoc` | `aligne.data.hfdata` / `aligne.data.synthdoc` |
| `aligne.character.constitution` / `.prompts` / `.exemplars` | `aligne.data.constitution` / `.prompts` / `.exemplars` |
| `aligne.character.gen_pairs` / `.introspection` (+ their drivers) | `aligne.data.gen_pairs` / `aligne.data.introspection` |
| `aligne.character.eval_*` + eval drivers | `aligne.eval.character.*` |
| `aligne.character.cli` | `aligne.cli.character` |

`aligne.ChatClient`/`Endpoint` re-export at the top level as before; the
`aligne` CLI and all subcommands are unchanged.

## 0.2.0 — 2026-07-14

The architecture revamp (`docs/specs/architecture-revamp.SPEC.md`, PRs #13–#18):
back to **async-native, configurable, composable library — not a CLI**. The
design rules now live in `DESIGN.md` and are enforced by
`tests/test_design_rules.py`.

### Breaking

- **One console script.** All `aligne-*` scripts are gone; use `aligne
  <subcommand>`: `run`, `character …`, `synthdoc`, `train
  sft|dpo|distill|distill-forward|ema`, `jlens`, `audit analyze|decompose`,
  `serve-tinker`. (Or, preferably, the Python API.)
- **Train drivers are config-first and async.**
  `run_sft/run_dpo/run_reverse_kl/run_forward_kl/run_ema` take frozen
  keyword-only configs (`SFTConfig`, `DPOConfig`, `ReverseKLDistillConfig`,
  `ForwardKLDistillConfig`, `EMAConfig`) instead of `argparse.Namespace`, are
  `async def`, and return typed results (`TrainResult` / `EMAResult`: final
  `sampler_path`, `state_path`, `final_metrics`) instead of out-dir strings.
  `model`, `renderer`, and `out` are required (the old EM-experiment defaults
  moved to `configs/train/em-qwen3.6-27b.json`); `--sys` → `--system-prompt`,
  `--fewshot` → `--fewshot-path`. `apply_smoke`/`add_common_tinker_args`/
  `DEFAULT_RENDERER` are gone (`cfg.smoke()`, generated parsers).
- **Prompted-teacher install is scoped.** `install_prompted_teacher_kl`
  (permanent monkeypatch) → `prompted_teacher_kl(sys_block)` context manager;
  the cookbook's original function is restored on exit.
- **Battery is a library call.** `run_battery(args: Namespace)` →
  `async run_battery(cfg: BatteryConfig) -> BatteryResult`; per-metric knobs
  thread via `metric_configs` (instance or dict per registry name) /
  `RunContext.config_for`. CLI: repeatable `--metric-config NAME=PATH.json`.
- **`hfdata` is async** (`fetch_rows`/`fetch_all_rows` are coroutines);
  sync compute-bound callers use `fetch_rows_sync`/`fetch_all_rows_sync`.
- **diffscope's vendored `Client` is gone** — use `aligne.client.ChatClient`
  (which gained `ChatClient.openrouter(model)`).
- `audit.analyze`/`audit.decompose` are async on the shared `ChatClient`
  (concurrent validation/extraction) with `AnalyzeConfig`; their CLIs moved to
  `python -m aligne.audit.cli` / `aligne audit`.

### Added

- `aligne.chat`: shared `sample` / `sample_records` / `judge` /
  `judge_records` helpers.
- `aligne.util`: `write_artifact`, `aclosing`.
- Character stages as library functions (`aligne.character`):
  `run_pairs_gen`, `run_introspection`, `run_preference_eval`,
  `run_coherence_eval`, `run_predictability_eval` + their configs.
- `IFEvalConfig`; `RefusalConfig.judge_template`; `PanelConfig` oracle knobs
  (`n_fallback_samples`, `min_ab_coverage`); diffscope `RaterConfig`.
- `DESIGN.md` (rules R1–R3) + guardrail tests; `tests/test_hfdata.py`.

### Removed

- Dead code: `HUMOR_TRAITS`, unused `EstimatorConfig` fields, the
  `character/prompts/make_*.py` one-off generators (now
  `scripts/character_prompt_generators/`, out of the wheel).

### Migration cheat-sheet

| before | after |
|---|---|
| `aligne-sft --data d.jsonl ...` | `aligne train sft --model M --renderer R --out O --data d.jsonl` |
| `run_reverse_kl(args)` (Namespace) | `await run_reverse_kl(ReverseKLDistillConfig(model=..., renderer=..., out=..., prompts=..., system_prompt=...))` |
| parse `checkpoints.jsonl` / CLI stdout for the checkpoint | `result.sampler_path`, `result.final_metrics.get("teacher_kl")` |
| `run_battery(args)` | `await run_battery(BatteryConfig(target=Endpoint(...), out=..., metric_configs={...}))` |
| `fetch_rows(...)` from async code | `await fetch_rows(...)` |
| `diffscope.Client.openrouter(m)` | `ChatClient.openrouter(m)` |

## 0.1.0

Initial development history (battery, synthdoc, character, train, jlens,
audit, diffscope absorption) — see git log up to #11.
