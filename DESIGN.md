# aligne design rules

aligne is an **async-native, configurable library of composable Python
components**. The CLI is a convenience shell around it, never the other way
around. Rationale, audit history, and the refactor that restored these rules:
`docs/specs/architecture-revamp.SPEC.md`.

## The four rules

**R1 — async at every IO boundary.** Anything that waits on the network or an
external service is `async def` (`ChatClient`, `hfdata`, metric/eval/audit
drivers). Fan-out goes through `asyncio.gather`, bounded by the client's
semaphore. Synchronous code is reserved for compute-bound work (torch in
`jlens`) — and sync shims over async functions (`fetch_rows_sync`) exist only
for those callers.

**R2 — the CLI is a thin adapter, and there is exactly one of it.**
`argparse`, `asyncio.run()`, and `print()` live only in the designated CLI
adapter modules (the `aligne/cli/` package, per-cluster `cli.py` adapters,
`eval/battery.py:main`, the `serving` entry point) — never in library code. A CLI adapter parses flags,
builds a config, calls the async library function, prints. One console script
(`aligne`) with subcommands; new functionality gets a library entry point
first and a subcommand second, if at all. Library code reports through
`logging` and return values.

**R3 — every runnable takes a config dataclass.** One keyword-only dataclass
per runnable unit, defaults documented in place, validation in
`__post_init__`, file loading that rejects unknown keys. Never an
`argparse.Namespace` below the CLI layer; never experiment-specific values
(model names, `/tmp` paths) as library defaults — those live in `configs/`.
The composed battery threads per-metric configs via
`RunContext.config_for`.

**R4 — the substrate never imports the domain layer.** Every library module
belongs to one of two layers (see "Two layers" below), declared per module in
`tests/test_layers.py:LAYERS`. Domain modules may import substrate modules
freely; a substrate module importing a domain module is a CI failure. New
modules are added to the manifest **in the same PR** that adds them.

## Two layers

The clusters (`data`/`train`/`eval`) answer *what workflow stage is this?*
A second, orthogonal axis answers *what kind of code is this?*:

- **substrate** — mechanism. Everything that mechanically runs, converts,
  measures, or moves models and datasets: provider plumbing (`train/tinker`,
  `train/backends`, `train/axolotl`, `serving`), data mechanics (`hfdata`,
  `mix`, `data/health`), judging/calibration machinery (`panel`, `oracle`,
  `eval/calibrate`), clients (`util`). Substrate would be used **identically**
  by any two unrelated research projects and contains zero research judgment;
  where a mechanism needs domain facts, they arrive **injected** (the
  `data/health` targets pattern), never baked in.
- **domain** — research judgment. Everything that encodes a decision a paper
  would have to defend: what to measure, how to elicit, what a trait or
  belief is (`synthdoc`, constitutions, the metrics, `character`, `audit`,
  `diffscope`, `jlens`).

The membership test for new code: *if two unrelated projects would want it
byte-for-byte, it's substrate; if it embeds a research decision, it's
domain.* Layer is orthogonal to dependency weight — `jlens` is domain and
torch-heavy, `axolotl` is substrate and heavy; extras solve weight, the
manifest solves placement.

Adapters (the `cli/` package, per-cluster `cli.py` files) and package
`__init__.py` re-export surfaces sit outside the layers: they are the
top-of-stack composition points and may import both. Keep `__init__.py`
files thin (re-exports only) — R4 does not trace through them.

Known debt, recorded here so it isn't mistaken for intent: the eval harness
(`eval/registry`, `eval/context`, `eval/battery`, `inspect_tasks`) is tagged
**domain** because the standard battery's metric composition is baked into
it. The harness *machinery* is substrate-shaped; extracting it (metrics
injected, composition left downstream-visible) is an open follow-up, and the
tags flip when it happens.

## Enforcement

`tests/test_design_rules.py` greps the source tree for R1/R2 violations
(argparse, `asyncio.run(`, `time.sleep(`, `print(` outside the allowlisted
adapter modules) and fails CI on drift. `tests/test_layers.py` holds the
per-module layer manifest and AST-checks R4 (every substrate module's
`aligne.*` imports resolve to substrate), plus completeness (every library
module is tagged; no stale entries). If a new module legitimately needs an
exemption (a new CLI adapter, a new compute-bound package), add it to the
allowlist **in the same PR** and say why in the PR body.

## Layout (since v0.3.0)

Four clusters — `data` (loaders, constitutions/prompt sets, synthetic-data
generation), `train` (Tinker drivers), `eval` (battery + metrics + judged
character evals + audit/diffscope/jlens), `util` (client/chat/stats) — plus
`serving` (a deployable shim, deliberately outside the clusters) and `cli`
(the one console script). "Character" is a workflow across clusters
(docs/character.md), not a package: its data lives in `data`, its training is
general prompt distillation in `train`, its evals in `eval.character`.
