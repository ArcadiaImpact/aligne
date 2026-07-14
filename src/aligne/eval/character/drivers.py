"""Config-first async drivers for the judged character evals.

Each eval the ``aligne character`` CLI exposes is callable as a library
function taking a keyword-only config dataclass and returning the eval's
summary. (The data-generation stages live with their internals in
``aligne.data.gen_pairs`` / ``aligne.data.introspection``.) The CLI
(:mod:`aligne.cli.character`) is a thin flags→config adapter over these.

Endpoints are :class:`aligne.util.client.Endpoint`; each driver builds cached
:class:`~aligne.util.client.ChatClient` handles under ``<out>/cache`` (so
interrupted runs resume for free) and closes them via
:func:`aligne.util.aclosing`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from aligne.data.prompts import resolve_eval_prompts
from aligne.util.client import ChatClient, Endpoint, cached_client
from aligne.util import aclosing, write_artifact

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
    cache = out / "cache"
    clients = {
        "base": cached_client(cfg.base, cache, "base", cfg.concurrency),
        "trained": cached_client(cfg.trained, cache, "trained", cfg.concurrency),
    }
    judge = cached_client(cfg.judge, cache, "judge", cfg.concurrency)
    async with aclosing(*clients.values(), judge):
        judged = await E.evaluate_preferences(
            rows, clients, judge, condition=cfg.condition,
            max_tokens=cfg.max_tokens, temperature=cfg.temperature,
            judge_max_tokens=cfg.judge_max_tokens,
        )
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

    out = Path(cfg.out)
    cache = out / "cache"
    clients = {"base": cached_client(cfg.base, cache, "base", cfg.concurrency)}
    system_prompts: dict[str, str] = {}
    if cfg.prompted_oracle:
        clients["prompted"] = cached_client(
            cfg.base, cache, "prompted", cfg.concurrency
        )
        system_prompts["prompted"] = C.constitution_system_prompt(con)
    if cfg.trained is not None:
        clients["trained"] = cached_client(
            cfg.trained, cache, "trained", cfg.concurrency
        )
    judge = cached_client(cfg.judge, cache, "judge", cfg.concurrency)
    log.info("coherence: %d scenarios, constitution=%s, variants=%s",
             len(rows), con.name, list(clients))

    async with aclosing(*clients.values(), judge):
        judged = await E.evaluate_coherence(
            rows, clients, judge, con,
            system_prompts=system_prompts,
            max_tokens=cfg.max_tokens, temperature=cfg.temperature,
        )
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

    out = Path(cfg.out)
    cache = out / "cache"
    base = None

    def base_client() -> ChatClient:
        nonlocal base
        base = base or cached_client(cfg.base, cache, "base", cfg.concurrency)
        return base

    variants: dict[str, tuple] = {}
    for label in cfg.variants:
        if label == "base":
            variants["base"] = (base_client(), None)
        elif label == "flat_prompted":
            flat_con = C.load_constitution(cfg.flat_constitution)
            variants["flat_prompted"] = (
                base_client(), C.constitution_system_prompt(flat_con)
            )
        elif label == "structured_prompted":
            variants["structured_prompted"] = (
                base_client(), C.constitution_system_prompt(con)
            )
        elif label == "structured_trained":
            if cfg.trained is None:
                raise ValueError("structured_trained needs cfg.trained")
            variants["structured_trained"] = (
                cached_client(cfg.trained, cache, "trained", cfg.concurrency),
                None,
            )
        elif label == "flat_trained":
            if cfg.flat_trained is None:
                raise ValueError("flat_trained needs cfg.flat_trained")
            variants["flat_trained"] = (
                cached_client(
                    cfg.flat_trained, cache, "flat_trained", cfg.concurrency
                ),
                None,
            )
        else:
            raise ValueError(f"unknown variant {label!r}")

    judge = cached_client(cfg.judge, cache, "judge", cfg.concurrency)
    log.info(
        "predictability: %d scenarios x k=%d, %s vs %s, variants=%s",
        len(rows), cfg.k, con.name, cfg.flat_constitution, list(variants),
    )

    async with aclosing(*(c for c, _ in variants.values()), judge):
        judged = await P.evaluate_predictability(
            rows, variants, judge, con,
            k=cfg.k, max_tokens=cfg.max_tokens, temperature=cfg.temperature,
        )
    summary = P.summarize_eval(judged)
    E.write_eval_rows(out / "predictability_rows.jsonl", judged)
    write_artifact(out, "predictability.json", summary)
    return summary
