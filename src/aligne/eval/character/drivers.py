"""Config-first async drivers for the judged character evals.

Each eval the ``aligne character`` CLI exposes is callable as a library
function taking a keyword-only config dataclass and returning the eval's
summary. (The data-generation stages live with their internals in
``aligne.data.gen_pairs`` / ``aligne.data.introspection``.) The CLI
(:mod:`aligne.cli.character`) is a thin flags→config adapter over these.

Endpoints are :class:`aligne.util.client.Endpoint`. Since the inspect
cutover the drivers elicit through :mod:`aligne.eval.inspect_character`
tasks (per-sample transcripts land in ``<out>/logs``); artifact shapes are
unchanged.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from aligne.data.prompts import resolve_eval_prompts
from aligne.util.client import Endpoint
from aligne.util import write_artifact


async def _run_task(tsk, target_ep: Endpoint, out: Path, concurrency: int):
    from aligne.eval.inspect_tasks import eval_metric_task, inspect_model

    return await eval_metric_task(
        tsk, inspect_model(target_ep, max_connections=concurrency),
        out, concurrency,
    )

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# eval (revealed preferences: base vs trained)
# --------------------------------------------------------------------------- #
@dataclass(kw_only=True)
class PreferenceEvalConfig:
    base: Endpoint
    trained: Endpoint
    judge: Endpoint
    out: Path
    constitution: str = "humor"
    prompts: str | None = None
    n_wildchat: int | None = None
    condition: str = "feel"  # "feel" | "like" | "random"
    seed: int = 123456
    max_tokens: int = 512
    # budget for the judge's verdict incl. any preamble before the <answer>
    # tag (16 truncates chatty judges -> all unparsed)
    judge_max_tokens: int = 256
    temperature: float = 1.0
    concurrency: int = 32


async def run_preference_eval(cfg: PreferenceEvalConfig) -> dict:
    """Revealed-preferences eval (base-vs-trained delta), judged; writes
    ``eval_rows.jsonl`` + ``eval.json`` under ``cfg.out`` and returns the
    summary."""
    from aligne.data import constitution as C
    from aligne.eval.character import preferences as E

    from aligne.eval.inspect_character import (
        judged_rows_from_log, preference_task,
    )
    from aligne.eval.inspect_tasks import inspect_model

    con = C.load_constitution(cfg.constitution)
    targets = con.target_traits
    if not targets:
        raise ValueError(
            f"Constitution {con.name!r} has no target_traits to grade against"
        )
    prompts = resolve_eval_prompts(con, cfg.prompts, cfg.n_wildchat, cfg.seed)
    rows = E.build_preference_rows(prompts, seed=cfg.seed)
    log.info("eval: %d prompts, targets=%s, condition=%s",
             len(rows), targets, cfg.condition)

    out = Path(cfg.out)
    judge_m = inspect_model(cfg.judge, max_connections=cfg.concurrency)
    judged = {}
    for label, ep in (("base", cfg.base), ("trained", cfg.trained)):
        elog = await _run_task(
            preference_task(rows, judge_m, condition=cfg.condition,
                            max_tokens=cfg.max_tokens,
                            temperature=cfg.temperature,
                            judge_max_tokens=cfg.judge_max_tokens),
            ep, out / label, cfg.concurrency,
        )
        judged[label] = judged_rows_from_log(elog, "preference_judge")
    summary = E.summarize_eval(judged, targets)
    E.write_eval_rows(out / "eval_rows.jsonl", judged)
    write_artifact(out, "eval.json", summary)
    return summary


# --------------------------------------------------------------------------- #
# coherence (install-quality: resolution-match vs the constitution answer key)
# --------------------------------------------------------------------------- #
@dataclass(kw_only=True)
class CoherenceEvalConfig:
    base: Endpoint
    judge: Endpoint
    out: Path
    constitution: str = "thoughtful_assistant"  # needs v2 values/tradeoffs
    scenarios: str | None = None  # scenario set name|path (default = constitution)
    # add a 'prompted' variant: the base endpoint with the full constitution
    # as system prompt (validity check)
    prompted_oracle: bool = False
    trained: Endpoint | None = None  # omit for a base-only / oracle-only run
    max_tokens: int = 512
    temperature: float = 1.0
    concurrency: int = 32


async def run_coherence_eval(cfg: CoherenceEvalConfig) -> dict:
    """Coherence eval: does each variant resolve value conflicts per the
    constitution's answer key? Writes ``coherence_rows.jsonl`` +
    ``coherence.json`` and returns the summary."""
    from aligne.data import constitution as C
    from aligne.eval.character import coherence as E

    con = C.load_constitution(cfg.constitution)
    if not con.values:
        raise ValueError(
            f"Constitution {con.name!r} has no values; coherence needs a v2 "
            f"(hierarchical) constitution"
        )
    rows = E.attach_expected(con, E.load_scenarios(cfg.scenarios or con.name))

    from aligne.eval.inspect_character import (
        coherence_task, judged_rows_from_log,
    )
    from aligne.eval.inspect_tasks import inspect_model

    out = Path(cfg.out)
    variants: list[tuple[str, Endpoint, str | None]] = [
        ("base", cfg.base, None)]
    if cfg.prompted_oracle:
        variants.append(
            ("prompted", cfg.base, C.constitution_system_prompt(con)))
    if cfg.trained is not None:
        variants.append(("trained", cfg.trained, None))
    judge_m = inspect_model(cfg.judge, max_connections=cfg.concurrency)
    log.info("coherence: %d scenarios, constitution=%s, variants=%s",
             len(rows), con.name, [v[0] for v in variants])

    judged = {}
    for label, ep, sp in variants:
        elog = await _run_task(
            coherence_task(rows, con, judge_m, system_prompt=sp,
                           max_tokens=cfg.max_tokens,
                           temperature=cfg.temperature),
            ep, out / label, cfg.concurrency,
        )
        judged[label] = judged_rows_from_log(elog, "scenario_judge")
    summary = E.summarize_eval(judged)
    E.write_eval_rows(out / "coherence_rows.jsonl", judged)
    write_artifact(out, "coherence.json", summary)
    return summary


# --------------------------------------------------------------------------- #
# predictability (flat vs structured)
# --------------------------------------------------------------------------- #
@dataclass(kw_only=True)
class PredictabilityEvalConfig:
    base: Endpoint
    judge: Endpoint
    out: Path
    # STRUCTURED (v2) constitution: its values define the conflicts + answer key
    constitution: str = "candid_advisor"
    # FLAT (v1) counterpart, the system prompt for the flat_prompted variant
    flat_constitution: str = "candid_advisor_flat"
    scenarios: str | None = None
    # from: base, flat_prompted, structured_prompted, structured_trained,
    # flat_trained (trained variants are promptless endpoints)
    variants: tuple[str, ...] = ("base", "flat_prompted", "structured_prompted")
    trained: Endpoint | None = None  # structured-trained endpoint
    flat_trained: Endpoint | None = None
    k: int = 8  # samples per prompt for self-consistency
    max_tokens: int = 600
    temperature: float = 0.7
    concurrency: int = 32


async def run_predictability_eval(cfg: PredictabilityEvalConfig) -> dict:
    """Predictability eval (consistency + controllability of conflict
    resolution, flat vs structured constitution). Writes
    ``predictability_rows.jsonl`` + ``predictability.json`` and returns the
    summary."""
    from aligne.data import constitution as C
    from aligne.eval.character import coherence as E
    from aligne.eval.character import predictability as P

    con = C.load_constitution(cfg.constitution)
    if not con.values:
        raise ValueError(
            f"Constitution {con.name!r} has no values; predictability needs a "
            f"v2 (structured) constitution for the conflict definitions + "
            f"answer key"
        )
    rows = E.attach_expected(con, E.load_scenarios(cfg.scenarios or con.name))

    from aligne.eval.inspect_character import (
        judged_rows_from_log, predictability_task,
    )
    from aligne.eval.inspect_tasks import inspect_model

    out = Path(cfg.out)
    variants: dict[str, tuple[Endpoint, str | None]] = {}
    for label in cfg.variants:
        if label == "base":
            variants["base"] = (cfg.base, None)
        elif label == "flat_prompted":
            flat_con = C.load_constitution(cfg.flat_constitution)
            variants["flat_prompted"] = (
                cfg.base, C.constitution_system_prompt(flat_con))
        elif label == "structured_prompted":
            variants["structured_prompted"] = (
                cfg.base, C.constitution_system_prompt(con))
        elif label == "structured_trained":
            if cfg.trained is None:
                raise ValueError("structured_trained needs cfg.trained")
            variants["structured_trained"] = (cfg.trained, None)
        elif label == "flat_trained":
            if cfg.flat_trained is None:
                raise ValueError("flat_trained needs cfg.flat_trained")
            variants["flat_trained"] = (cfg.flat_trained, None)
        else:
            raise ValueError(f"unknown variant {label!r}")

    judge_m = inspect_model(cfg.judge, max_connections=cfg.concurrency)
    log.info(
        "predictability: %d scenarios x k=%d, %s vs %s, variants=%s",
        len(rows), cfg.k, con.name, cfg.flat_constitution, list(variants),
    )

    judged = {}
    for label, (ep, sp) in variants.items():
        elog = await _run_task(
            predictability_task(rows, con, judge_m, k=cfg.k,
                                system_prompt=sp,
                                max_tokens=cfg.max_tokens,
                                temperature=cfg.temperature),
            ep, out / label, cfg.concurrency,
        )
        judged[label] = judged_rows_from_log(elog, "scenario_judge")
    summary = P.summarize_eval(judged)
    E.write_eval_rows(out / "predictability_rows.jsonl", judged)
    write_artifact(out, "predictability.json", summary)
    return summary
