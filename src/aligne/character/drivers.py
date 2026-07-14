"""Config-first async drivers for the character pipeline stages.

Every stage the ``aligne character`` CLI exposes is callable as a library
function taking a keyword-only config dataclass and returning the stage's
summary (evals) or rows (generation). The CLI (:mod:`aligne.character.cli`)
is a thin flags→config adapter over these.

Endpoints are :class:`aligne.client.Endpoint`; each driver builds cached
:class:`~aligne.client.ChatClient` handles under ``<out>/cache`` (so
interrupted runs resume for free) and closes them via
:func:`aligne.util.aclosing`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from ..client import ChatClient, Endpoint
from ..util import aclosing, write_artifact

log = logging.getLogger(__name__)


def _cached_client(
    endpoint: Endpoint, cache_dir: Path, tag: str, concurrency: int
) -> ChatClient:
    cache_dir.mkdir(parents=True, exist_ok=True)
    return ChatClient(
        endpoint=endpoint,
        concurrency=concurrency,
        cache_path=cache_dir / f"cache_{tag}.jsonl",
    )


def resolve_eval_prompts(
    con, prompts: str | None, n_wildchat: int | None, seed: int
) -> list[str]:
    """The eval prompt source: WildChat first-turns (``n_wildchat``), an
    explicit prompt set (name or path), or the constitution's default set."""
    from . import prompts as P
    from .eval_preferences import load_wildchat_prompts

    if n_wildchat:
        return load_wildchat_prompts(n_wildchat, seed=seed)
    prompt_set = prompts or con.default_prompts
    if not prompt_set:
        raise ValueError(
            f"No eval prompts: pass prompts=<name|path> or n_wildchat=N "
            f"(constitution {con.name!r} has no default_prompts)"
        )
    return P.load_prompt_set(prompt_set)


# --------------------------------------------------------------------------- #
# pairs (OCT DPO preference pairs from a prompted base)
# --------------------------------------------------------------------------- #
@dataclass(kw_only=True)
class PairsConfig:
    base: Endpoint
    out: Path
    constitution: str = "humor"
    prompts: str | None = None  # prompt set name|path (default = constitution's)
    n_wildchat: int | None = None  # instead, use N WildChat first-turns
    n: int | None = None  # cap the number of prompts used
    seed: int = 123456
    max_tokens: int = 512
    temperature: float = 1.0
    concurrency: int = 32


async def run_pairs_gen(cfg: PairsConfig) -> list[dict]:
    """Generate OCT DPO comparison rows (chosen = in-character, rejected =
    plain) from a served base; writes ``cfg.out`` and returns the rows."""
    from . import constitution as C
    from . import gen_pairs as G

    con = C.load_constitution(cfg.constitution)
    system_prompt = C.constitution_system_prompt(con)
    prompts = resolve_eval_prompts(con, cfg.prompts, cfg.n_wildchat, cfg.seed)
    if cfg.n:
        prompts = prompts[: cfg.n]
    log.info("pairs: constitution=%s, %d prompts -> %s",
             con.name, len(prompts), cfg.out)

    client = ChatClient(endpoint=cfg.base, concurrency=cfg.concurrency)
    async with aclosing(client):
        rows = await G.generate_pairs(
            client, prompts, system_prompt,
            max_tokens=cfg.max_tokens, temperature=cfg.temperature,
        )
    G.write_pairs_jsonl(str(cfg.out), rows)
    return rows


# --------------------------------------------------------------------------- #
# introspect (OCT stage 2: self-reflection + self-interaction -> SFT data)
# --------------------------------------------------------------------------- #
@dataclass(kw_only=True)
class IntrospectConfig:
    checkpoint: str  # tinker:// sampler checkpoint of the distilled student
    model: str
    renderer: str
    out: Path
    constitution: str = "humor"
    n_reflection: int = 40  # samples per reflection prompt (10 prompts)
    n_interaction: int = 150  # free-mode self-conversations
    n_leading: int = 75  # leading-mode self-conversations
    k: int = 10  # turns per self-conversation
    reflection_max_tokens: int = 2048
    interaction_max_tokens: int = 1024
    temperature: float = 0.7
    top_p: float = 0.95
    concurrency: int = 32
    seed: int = 123456


async def run_introspection(cfg: IntrospectConfig) -> dict[str, list[dict]]:
    """OCT introspection: reflections + self-interactions + the shuffled SFT
    mix, each written to ``cfg.out`` as JSONL. Needs the ``tinker`` extra."""
    from . import constitution as C
    from . import introspection as I

    con = C.load_constitution(cfg.constitution)
    name = C.teacher_name(cfg.model)
    out = Path(cfg.out)
    log.info("introspect: constitution=%s name=%s checkpoint=%s",
             con.name, name, cfg.checkpoint)

    reflections = await I.generate_reflections(
        cfg.checkpoint, cfg.model, cfg.renderer, name, con,
        n_per_prompt=cfg.n_reflection, max_tokens=cfg.reflection_max_tokens,
        temperature=cfg.temperature, top_p=cfg.top_p,
        concurrency=cfg.concurrency,
    )
    write_artifact(out, "reflections.jsonl", reflections)
    free = await I.generate_interactions(
        cfg.checkpoint, cfg.model, cfg.renderer, name, con,
        n=cfg.n_interaction, k=cfg.k, leading=False,
        max_tokens=cfg.interaction_max_tokens,
        temperature=cfg.temperature, top_p=cfg.top_p,
        concurrency=cfg.concurrency, seed=cfg.seed,
    )
    write_artifact(out, "interaction.jsonl", free)
    leading = await I.generate_interactions(
        cfg.checkpoint, cfg.model, cfg.renderer, name, con,
        n=cfg.n_leading, k=cfg.k, leading=True,
        max_tokens=cfg.interaction_max_tokens,
        temperature=cfg.temperature, top_p=cfg.top_p,
        concurrency=cfg.concurrency, seed=cfg.seed,
    )
    write_artifact(out, "interaction_leading.jsonl", leading)
    sft = I.build_sft_data(name, reflections, free, leading, seed=cfg.seed)
    write_artifact(out, "sft_data.jsonl", sft)
    return {"reflections": reflections, "interaction": free,
            "interaction_leading": leading, "sft_data": sft}


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
    from . import constitution as C
    from . import eval_preferences as E

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
        "base": _cached_client(cfg.base, cache, "base", cfg.concurrency),
        "trained": _cached_client(cfg.trained, cache, "trained", cfg.concurrency),
    }
    judge = _cached_client(cfg.judge, cache, "judge", cfg.concurrency)
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
    from . import constitution as C
    from . import eval_coherence as E

    con = C.load_constitution(cfg.constitution)
    if not con.values:
        raise ValueError(
            f"Constitution {con.name!r} has no values; coherence needs a v2 "
            f"(hierarchical) constitution"
        )
    rows = E.attach_expected(con, E.load_scenarios(cfg.scenarios or con.name))

    out = Path(cfg.out)
    cache = out / "cache"
    clients = {"base": _cached_client(cfg.base, cache, "base", cfg.concurrency)}
    system_prompts: dict[str, str] = {}
    if cfg.prompted_oracle:
        clients["prompted"] = _cached_client(
            cfg.base, cache, "prompted", cfg.concurrency
        )
        system_prompts["prompted"] = C.constitution_system_prompt(con)
    if cfg.trained is not None:
        clients["trained"] = _cached_client(
            cfg.trained, cache, "trained", cfg.concurrency
        )
    judge = _cached_client(cfg.judge, cache, "judge", cfg.concurrency)
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
    from . import constitution as C
    from . import eval_coherence as E
    from . import eval_predictability as P

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
        base = base or _cached_client(cfg.base, cache, "base", cfg.concurrency)
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
                _cached_client(cfg.trained, cache, "trained", cfg.concurrency),
                None,
            )
        elif label == "flat_trained":
            if cfg.flat_trained is None:
                raise ValueError("flat_trained needs cfg.flat_trained")
            variants["flat_trained"] = (
                _cached_client(
                    cfg.flat_trained, cache, "flat_trained", cfg.concurrency
                ),
                None,
            )
        else:
            raise ValueError(f"unknown variant {label!r}")

    judge = _cached_client(cfg.judge, cache, "judge", cfg.concurrency)
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
