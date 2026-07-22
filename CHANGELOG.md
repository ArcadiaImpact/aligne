# Changelog

## Unreleased

**A spec-agnostic training-backend layer.** aligne now owns the training-backend
infra downstream libraries shouldn't hand-roll (migrated out of
science-of-midtraining). New:

- `aligne.train.backends` — the backend seam: the `Backend` protocol, a typed
  `Checkpoint` pointer (`sampler` for evals, `state` for resuming; never
  interchange them — `require_state()` errors legibly), the `get_backend`
  registry, and the `async def run_train(cfg)` entry point. The whole
  backend-facing contract is `BackendConfig` — a base model id, renderer,
  hparams, dataset path, output dir, and checkpoint-chaining pointer — and it is
  deliberately **spec-agnostic**: nothing about any caller's experiment
  vocabulary crosses this boundary, so a downstream library adapts its own spec
  down to a `BackendConfig` in a thin wrapper it owns. `TinkerBackend` builds an
  `aligne.train.tinker.SFTConfig` and delegates to `run_sft`, so the SFT
  conventions live in exactly one place (no drift).
- `aligne.train.axolotl` — `AxolotlBackend`: FSDP2 full-parameter local-GPU
  midtraining (port of pane, frozen at pane `fa3ea9b`). File-backed stage-template
  registry (`stages/`), the divergence loss guard, and local-subprocess /
  bellhop-pod executors. Registers alongside `TinkerBackend` behind the same
  protocol.
- `aligne.data.mix` — `build_mix`: token-budget corpus mixing (anchor-driven +
  total-tokens modes) emitting a reproducible `MixManifest`. Mixing is a dataset
  artifact, a sibling of `hfdata`, not a trainer feature.
- `aligne.train.runlog` — `snapshot_run`: backend-agnostic local-run provenance
  (config + git commit + host; refuses to launch on a dirty tree unless
  `allow_dirty` / `ALIGNE_ALLOW_DIRTY=1`).

New `[axolotl]` optional extra (axolotl, torch, datasets, transformers, pyyaml);
all heavy deps (incl. `yaml`) import lazily, so the lean core install and
`import aligne.train` are unaffected. CPU-only unit tests cover the pure parts
(config/command construction, mix + manifest, runlog, registry dispatch); a live
FSDP2 pod smoke run is a follow-up.


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
