# Changelog

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

The architecture revamp (`specs/architecture-revamp.SPEC.md`, PRs #13–#18):
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
