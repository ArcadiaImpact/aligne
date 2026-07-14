# aligne design rules

aligne is an **async-native, configurable library of composable Python
components**. The CLI is a convenience shell around it, never the other way
around. Rationale, audit history, and the refactor that restored these rules:
`specs/architecture-revamp.SPEC.md`.

## The three rules

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

## Enforcement

`tests/test_design_rules.py` greps the source tree for R1/R2 violations
(argparse, `asyncio.run(`, `time.sleep(`, `print(` outside the allowlisted
adapter modules) and fails CI on drift. If a new module legitimately needs an
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
