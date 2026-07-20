# Architecture revamp — back to the library philosophy

**Status:** approved 2026-07-14 (audit: four parallel module reviews of the
whole `src/aligne` tree). Executed as five stacked PRs, this spec ships with
PR 1.

## Why

aligne's design philosophy is: **async-native, configurable Python components
that compose easily and are used as a library — not a CLI.** An audit of all
~10.5k lines found the philosophy intact in the core and badly drifted in the
newer modules. The repo currently ships **ten console-script entry points**,
has `argparse.Namespace` as the de-facto config object of `train/tinker`,
blocks the event loop from inside async metrics, and re-implements its own
`ChatClient` inside `diffscope`.

### Audit summary (worst first)

| Area | Verdict |
|---|---|
| `train/tinker` | Worst. `build_config(args: Namespace)` in sft/dpo/distill/ema (~740 lines Namespace-threaded); `asyncio.run()` buried in library fns (`sft.py:121`, `distill.py:188,286`); prompted-teacher installed by monkeypatching `train_on_policy.incorporate_kl_penalty` (module-global, non-reentrant); 4× copy-pasted driver scaffold + print banners; experiment residue as defaults (`Qwen/Qwen3.6-27B`, `/tmp/tinker/...`, "run-1" renderer comments, `# parity` dead import). |
| `audit/` | Serial **blocking** `client.chat.completions.create` loops (analyze per-sample, decompose per-chunk) — embarrassingly parallel LLM IO done sync; argparse/`__main__` inside library modules; flag thresholds (`>=7`/`>=5`) and dimension lists hardcoded. |
| `hfdata.py` | Fully sync (`time.sleep` at :55, sync httpx, ThreadPoolExecutor) but called **un-awaited inside `async def run()`** metric bodies (`capability.py:52`, `perplexity.py:68`, `refusal.py:107,114`) — stalls every concurrent call in the battery while a dataset pages. Bug-shaped, not just style. |
| battery seam | `run_battery(args: argparse.Namespace)` (`runner.py:42`); registry adapters hardcode their configs — only `seed`/`canaries` thread through, so the composed suite is unconfigurable without bypassing it. |
| duplication | `diffscope/client.py` ≈ 90% copy of core `ChatClient` (own `UnsupportedRequestError`, same retry/cache); "sample N → judge → rate_with_ci" re-implemented in trait/want/em/refusal/fluency (~200+ lines); `out_dir` write-tail in ~10 metrics; character `cli.py` holds ~230 lines of orchestration (client factory ×3 verbatim, `_go()` teardown ×3) that exists nowhere as a callable function; 3× set-path resolver; 2× WildChat loader. |
| `character/` | Library layer itself is strong (async, pure logic tested, lazy imports); the gap is that per-eval orchestration lives **only** in `cli.py` `run_*` fns taking Namespace and returning nothing; 4 one-off `make_*.py` generator scripts ship inside the package (one has a cwd-dependent bare import). |
| Faithful (leave alone) | `client.py`, `metric.py` protocol/registry, `report.py`, `panel.py`/`oracle.py`, `want.py` parameterization, **`synthdoc/`** (the reference implementation), `jlens` config layer + compute internals (sync is correct: torch), `serving/tinker_shim.py`. |

## The convention (the "synthdoc shape")

Every runnable unit in aligne follows the shape `synthdoc` already has:

1. **Config**: a frozen dataclass with documented fields and sane defaults;
   `.load()` for JSON/YAML where file configs exist; override-merging that
   **rejects unknown keys** (lift `_resolve_config` from synthdoc into core).
2. **Entry point**: `async def run_x(cfg, clients...) -> Result` — a plain
   library function. Returns data (dataclass or dict); artifact writing is
   separable (`write_artifact` / explicit `write_*` fn). No `print`, no
   `argparse`, no `asyncio.run`, no env-var reads inside the library path.
3. **CLI**: at most one thin adapter per subsystem, living under
   `aligne/cli/`, ≤~40 lines: parse → build config → `asyncio.run(run_x(cfg))`
   → print. The CLI is the only place that owns the event loop and stdout.

Three rules, mechanically enforceable:

- **R1 — async at every IO boundary.** Network/disk-wait code is `async def`;
  sync is reserved for compute-bound code (torch in `jlens`).
- **R2 — `argparse`, `asyncio.run`, `print` only under `aligne/cli/`**
  (plus `serving/` for its uvicorn entry point).
- **R3 — every runnable takes a frozen config dataclass**, never a
  `Namespace`; the composed battery threads per-metric configs through.

## Workstreams → PRs

Stacked; each lands on the previous. Old console scripts die in PR 5 —
breaking changes are acceptable (callers are our own experiment repos, and
invocation is supposed to be Python).

### PR 1 — core spine (this PR)

- `hfdata.py` → async (`httpx.AsyncClient` + `asyncio.sleep`); async paging
  fan-out replaces the ThreadPoolExecutor. A thin `fetch_rows_sync` /
  `fetch_all_rows_sync` (`asyncio.run` wrapper) remains for sync compute-bound
  callers (`jlens/datasets.py`). Fix the 4 blocking call sites in
  capability/perplexity/refusal.
- New `aligne/chat.py` helpers over `ChatClient`:
  `sample(client, messages, *, n, max_tokens, temperature) -> list[str]`,
  `judge(client, template/messages, *, parse, max_tokens) -> verdict`, and a
  `judged_rate(...)` higher-order helper for the shared
  sample→judge→`rate_with_ci` pipeline.
- `util.py`: `write_artifact(out_dir, name, obj)` (the ~10× mkdir+json tail),
  `aclosing(*clients)` async context manager (the 3× hand-rolled teardown).
- Convert trait/want/em/refusal/fluency (+ capability/perplexity write tails)
  to the helpers.
- **Delete `diffscope/client.py`**; diffscope uses `aligne.client.ChatClient`
  + `Endpoint` (gains an `Endpoint.openrouter(model)` classmethod).
- Dead code: unreachable `return resp` (`hfdata.py:56`), `HUMOR_TRAITS`
  (`eval_preferences.py:64`), unused `EstimatorConfig.accumulator_device`
  / `.seed` fields.

### PR 2 — `train/tinker` rewrite

- `SFTConfig` / `DPOConfig` / `DistillConfig` (reverse + forward KL) /
  `EMAConfig` frozen dataclasses; argparse demoted to a CLI adapter
  (`tinker/cli.py` stops being the config schema).
- `async def run_sft(cfg)`, `run_dpo(cfg)`, `run_distill(cfg)`,
  `run_ema(cfg) -> Result` — event loop lifted to the CLI; DPO wraps the
  cookbook's sync `main` in `asyncio.to_thread` so all four compose uniformly.
- Prompted-teacher: explicit wiring through `DistillConfig` — if
  tinker-cookbook's API forces the patch, scope it in a context manager
  installed/removed around the run, never left installed module-globally.
- One shared driver scaffold (smoke-preset application, config echo via
  `logging`, result return) replaces the 4 copies.
- Purge experiment residue: model/renderer/`/tmp` defaults move to
  `configs/`; delete the `# parity` import; drop origin-script docstrings.

### PR 3 — `audit/` async-ification

- `analyze.py` / `decompose.py` ported onto core `ChatClient` (they already
  speak OpenAI-compatible HTTP; no second client stack) with bounded
  `asyncio.gather` over samples/chunks.
- `AuditConfig`: flag thresholds, CORE/SPECIALIZED dimension lists, judge
  truncations/max_tokens, system prompts.
- argparse/`__main__`/`print` out of the library modules → `aligne/cli/`
  (mirroring the `jlens/cli.py` pattern that already got this right).

### PR 4 — battery config threading

- `BatteryConfig` (endpoints, out dir, seed, canaries, metric selection,
  `metric_configs: dict[str, Any]`) replacing the Namespace;
  `async def run_battery(cfg) -> BatteryResult`.
- `RunContext.config_for(name, cls)` so registry adapters resolve a
  caller-supplied config, falling back to defaults — kills the seed-only
  seam ("design option 1a" in `context.py` is explicitly revoked).
- `IFEvalConfig` created (currently no config object at all); refusal's
  `JUDGE_TEMPLATE` and diffscope's rater prompt/score-weights move into
  their configs; `oracle` thresholds surfaced to `PanelConfig`.

### PR 5 — CLI consolidation + character library funcs + guardrails

- One `aligne` console script, subcommands under `aligne/cli/`
  (`battery`, `character …`, `synthdoc`, `train sft|dpo|distill|ema`,
  `jlens`, `audit …`, `serve-tinker`). The other nine `[project.scripts]`
  entries are removed.
- Character: `run_preference_eval(cfg)`, `run_coherence_eval(cfg)`,
  `run_predictability_eval(cfg)` (+ pairs/introspect/distill equivalents) as
  library functions returning summaries; cli.py shrinks to adapters. Shared
  set-path resolver; reuse `train.tinker.data.load_wildchat_prompts`.
- `character/prompts/make_*.py` generators move to `scripts/` (out of the
  wheel; fix the cwd-dependent bare import).
- `DESIGN.md` stating R1–R3 + this spec as rationale.
- Guardrail tests in CI (`tests/test_design_rules.py`): grep-style asserts —
  no `import argparse` / `asyncio.run(` / `time.sleep(` / bare `print(`
  outside `aligne/cli/` + explicit allowlist (jlens compute, serving entry
  point). Cheap, and it's what stops re-sloppification.
- Library-path `print` → `logging` everywhere touched above.

## Non-goals

- No behavior changes to metric definitions, prompts, or scoring — pure
  restructuring; battery outputs should be byte-comparable modulo key order.
- `jlens` estimator/fit internals and `synthdoc` untouched.
- Evicting `train/tinker` to its own arsenal package was considered and
  deferred: rewrite in place first; eviction stays possible afterwards.

## Definition of done (whole revamp)

- `pyproject.toml` has exactly one console script.
- `grep -rn "argparse\|asyncio.run(\|time.sleep(" src/aligne` hits only
  `aligne/cli/` + the allowlist, enforced by a test.
- Every subsystem is drivable from an async Python script with dataclass
  configs only; `pytest` green throughout; each PR independently green.
