"""``aligne-character`` — character-training driver.

Three subcommands:

- ``render`` — load a constitution and print the eliciting teacher system block;
  optionally write the student prompt set (seed questions) to a JSONL. Pure /
  no heavy deps — use it to inspect what the distill stage will feed the models.
- ``distill`` — render the constitution into the teacher system block + a prompts
  JSONL, then run **on-policy reverse-KL from the prompted teacher** by
  delegating to ``aligne.train.tinker.distill.run_reverse_kl``. The constitution
  is the teacher's ``--sys`` block; the teacher is the same base model as the
  student. Requires the ``tinker`` extra (imported lazily inside ``run``).
- ``eval`` — revealed-preferences eval (base-vs-trained ``delta``) against two
  OpenAI-compatible endpoints, judged by a third. Uses aligne's
  ``ChatClient``; no ``tinker`` extra needed.

Defaults target ``Qwen/Qwen3-235B-A22B-Instruct-2507`` with the ``qwen3_instruct``
renderer (the repo's 235B setup).
"""

from __future__ import annotations

import argparse
from pathlib import Path

DEFAULT_MODEL = "Qwen/Qwen3-235B-A22B-Instruct-2507"
DEFAULT_RENDERER = "qwen3_instruct"


# --------------------------------------------------------------------------- #
# render
# --------------------------------------------------------------------------- #
def build_render_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="aligne-character render", description="Print the teacher system block; optionally write the prompt set.")
    p.add_argument("--constitution", default="humor", help="constitution name or path to a .txt")
    p.add_argument("--model", default=DEFAULT_MODEL, help="model string the character name is derived from")
    p.add_argument("--prompts", default=None, help="prompt set name|path to preview (default = constitution.default_prompts)")
    p.add_argument("--prompts-out", default=None, help="if set, write the resolved prompt set here as JSONL")
    return p


def run_render(args: argparse.Namespace) -> None:
    from . import constitution as C
    from . import prompts as P

    con = C.load_constitution(args.constitution)
    block = C.system_block(args.model, con)
    prompt_set = args.prompts or con.default_prompts
    print("=" * 70)
    print(f"constitution: {con.name}  | traits: {len(con.traits)}  | target_traits: {con.target_traits}")
    print(f"prompt set:   {prompt_set or '(none)'}  | bundled sets: {P.available_prompt_sets()}")
    print("=" * 70)
    print(block)
    print("=" * 70)
    if prompt_set and (args.prompts_out or args.prompts):
        qs = P.load_prompt_set(prompt_set)
        print(f"prompt set {prompt_set!r}: {len(qs)} prompts (first: {qs[0]!r})")
        if args.prompts_out:
            n = P.write_prompts_jsonl(args.prompts_out, qs)
            print(f"[render] wrote {n} prompts -> {args.prompts_out}")


# --------------------------------------------------------------------------- #
# distill (reverse-KL prompted teacher; the constitution is the --sys block)
# --------------------------------------------------------------------------- #
def build_distill_parser() -> argparse.ArgumentParser:
    """Reuse the reverse-KL parser, add ``--constitution``, retarget defaults.

    ``--system-prompt`` is rendered from the constitution at run time, and
    ``--prompts`` accepts a prompt-set **name or path** (resolved against
    ``prompts/``), defaulting to the constitution's ``default_prompts``.
    Everything else (lora-rank, lr, kl-coef, batch/group sizes, ``--smoke``,
    wandb, ...) is inherited unchanged.
    """
    from ..train.tinker.cli import build_reverse_kl_parser

    p = build_reverse_kl_parser()
    p.description = "Character distillation: reverse-KL from a constitution-prompted teacher."
    p.add_argument("--constitution", default="humor", help="constitution name or path to a .json")
    p.add_argument("--hide-priorities", action="store_true",
                   help="render the teacher block WITHOUT the priority/trade-off section (principles only); "
                        "for covert-install studies where the hierarchy is hidden from the teacher")
    # Character defaults: 235B + the instruct (non-thinking) renderer, prompted
    # teacher = same base model.
    p.set_defaults(
        model=DEFAULT_MODEL,
        teacher_model=DEFAULT_MODEL,
        renderer=DEFAULT_RENDERER,
        out="/tmp/tinker/character",
        recipe_name="character_reverse_kl",
    )
    return p


def run_distill(args: argparse.Namespace) -> None:
    import asyncio

    from ..train.tinker.configs import ReverseKLDistillConfig
    from ..train.tinker.distill import run_reverse_kl
    from . import constitution as C
    from . import prompts as P

    con = C.load_constitution(args.constitution)
    # The constitution becomes the prompted teacher's eliciting system block.
    system_prompt = getattr(args, "system_prompt", None) or C.system_block(
        args.teacher_model, con,
        priorities=not getattr(args, "hide_priorities", False),
    )
    # Few-shot exemplars (optional): resolve a bundled name|path to a file.
    fewshot_path = getattr(args, "fewshot_path", None)
    if fewshot_path:
        from . import exemplars as X

        fewshot_path = str(X.exemplar_set_path(fewshot_path))
    # Student rolls out on a prompt set — decoupled from the constitution.
    prompt_set = getattr(args, "prompts", None) or con.default_prompts
    if not prompt_set:
        raise SystemExit(
            f"No prompt set: constitution {con.name!r} has no default_prompts; "
            f"pass --prompts <name|path> (bundled: {P.available_prompt_sets()})"
        )
    values = {
        k: v for k, v in vars(args).items()
        if k not in ("smoke", "config", "constitution", "hide_priorities", "cmd")
    }
    values.update(
        system_prompt=system_prompt,
        fewshot_path=fewshot_path,
        prompts=str(P.prompt_set_path(prompt_set)),
        teacher_checkpoint=None,  # teacher = prompted BASE model, never a ckpt
    )
    cfg = ReverseKLDistillConfig(**values)
    if getattr(args, "smoke", False):
        cfg = cfg.smoke()
    print(
        f"[aligne-character] constitution={con.name} "
        f"name={C.teacher_name(cfg.resolved_teacher_model)} targets={con.target_traits} "
        f"prompts={prompt_set} ({cfg.prompts})"
    )
    asyncio.run(run_reverse_kl(cfg))


# --------------------------------------------------------------------------- #
# introspect (OCT stage 2: self-reflection + self-interaction -> SFT data)
# --------------------------------------------------------------------------- #
def build_introspect_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="aligne-character introspect",
        description="Generate the OCT introspection SFT set (self-reflection + "
        "self-interaction) from a distilled checkpoint. Train it with "
        "`aligne-sft --data <out>/sft_data.jsonl --load-checkpoint-path <state ckpt>`.",
    )
    p.add_argument("--constitution", required=True, help="constitution name or path")
    p.add_argument("--checkpoint", required=True,
                   help="tinker:// SAMPLER checkpoint of the distilled model "
                        "(or a base model name for an ablation)")
    p.add_argument("--model", default=DEFAULT_MODEL,
                   help="base model string (drives the character name)")
    p.add_argument("--renderer", default=DEFAULT_RENDERER)
    p.add_argument("--n-reflection", type=int, default=40,
                   help="samples per reflection prompt (10 prompts; OCT used 1000)")
    p.add_argument("--n-interaction", type=int, default=150,
                   help="free-mode self-conversations (OCT used 1000)")
    p.add_argument("--n-leading", type=int, default=75,
                   help="leading-mode self-conversations")
    p.add_argument("--k", type=int, default=10, help="turns per self-conversation")
    p.add_argument("--reflection-max-tokens", type=int, default=2048)
    p.add_argument("--interaction-max-tokens", type=int, default=1024)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--concurrency", type=int, default=32)
    p.add_argument("--seed", type=int, default=123456)
    p.add_argument("--out", required=True, help="output directory")
    return p


def run_introspect(args: argparse.Namespace) -> None:
    import asyncio
    import json

    from . import constitution as C
    from . import introspection as I

    con = C.load_constitution(args.constitution)
    name = C.teacher_name(args.model)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    def write(fname: str, rows: list[dict]) -> None:
        with (out / fname).open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[aligne-character introspect] {len(rows)} rows -> {out / fname}", flush=True)

    async def _go():
        reflections = await I.generate_reflections(
            args.checkpoint, args.model, args.renderer, name, con,
            n_per_prompt=args.n_reflection, max_tokens=args.reflection_max_tokens,
            temperature=args.temperature, top_p=args.top_p,
            concurrency=args.concurrency,
        )
        write("reflections.jsonl", reflections)
        free = await I.generate_interactions(
            args.checkpoint, args.model, args.renderer, name, con,
            n=args.n_interaction, k=args.k, leading=False,
            max_tokens=args.interaction_max_tokens,
            temperature=args.temperature, top_p=args.top_p,
            concurrency=args.concurrency, seed=args.seed,
        )
        write("interaction.jsonl", free)
        leading = await I.generate_interactions(
            args.checkpoint, args.model, args.renderer, name, con,
            n=args.n_leading, k=args.k, leading=True,
            max_tokens=args.interaction_max_tokens,
            temperature=args.temperature, top_p=args.top_p,
            concurrency=args.concurrency, seed=args.seed,
        )
        write("interaction_leading.jsonl", leading)
        return reflections, free, leading

    print(
        f"[aligne-character introspect] constitution={con.name} name={name} "
        f"checkpoint={args.checkpoint}", flush=True,
    )
    reflections, free, leading = asyncio.run(_go())
    sft = I.build_sft_data(name, reflections, free, leading, seed=args.seed)
    write("sft_data.jsonl", sft)


# --------------------------------------------------------------------------- #
# pairs (generate OCT DPO preference pairs from a prompted base)
# --------------------------------------------------------------------------- #
def build_pairs_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="aligne-character pairs",
        description="Generate OCT DPO preference pairs (chosen=in-character, rejected=plain) from a served base.",
    )
    p.add_argument("--constitution", default="humor", help="constitution name or path (drives the in-character system prompt)")
    p.add_argument("--base-url", required=True, help="OpenAI-compatible base endpoint to sample both completions from")
    p.add_argument("--base-model", required=True)
    p.add_argument("--base-key", default=None)
    p.add_argument("--prompts", default=None, help="prompt set name|path (default = constitution.default_prompts)")
    p.add_argument("--n-wildchat", type=int, default=None, help="instead, use N WildChat first-turns (HF-gated)")
    p.add_argument("--n", type=int, default=None, help="cap the number of prompts used")
    p.add_argument("--out", required=True, help="output comparison JSONL (feeds aligne-dpo --pairs)")
    p.add_argument("--seed", type=int, default=123456)
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--concurrency", type=int, default=32)
    return p


def run_pairs(args: argparse.Namespace) -> None:
    import asyncio

    from ..client import ChatClient, Endpoint
    from . import constitution as C
    from . import gen_pairs as G

    con = C.load_constitution(args.constitution)
    system_prompt = C.constitution_system_prompt(con)
    prompts = _eval_prompts(args, con)
    if args.n:
        prompts = prompts[: args.n]
    print(
        f"[aligne-character pairs] constitution={con.name} | {len(prompts)} prompts "
        f"| base={args.base_model} -> {args.out}"
    )

    client = ChatClient(
        endpoint=Endpoint(base_url=args.base_url, model=args.base_model, api_key=args.base_key),
        concurrency=args.concurrency,
    )

    async def _go():
        try:
            return await G.generate_pairs(
                client, prompts, system_prompt,
                max_tokens=args.max_tokens, temperature=args.temperature,
            )
        finally:
            await client.aclose()

    rows = asyncio.run(_go())
    n = G.write_pairs_jsonl(args.out, rows)
    print(f"[aligne-character pairs] wrote {n}/{len(prompts)} comparison rows -> {args.out}")


# --------------------------------------------------------------------------- #
# eval (revealed preferences: base vs trained)
# --------------------------------------------------------------------------- #
def build_eval_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Revealed-preferences eval (base vs trained).")
    p.add_argument("--constitution", default="humor", help="constitution name or path (for target_traits)")
    p.add_argument("--trained-url", required=True)
    p.add_argument("--trained-model", required=True)
    p.add_argument("--trained-key", default=None)
    p.add_argument("--base-url", required=True)
    p.add_argument("--base-model", required=True)
    p.add_argument("--base-key", default=None)
    p.add_argument("--judge-url", required=True)
    p.add_argument("--judge-model", required=True)
    p.add_argument("--judge-key", default=None)
    p.add_argument("--out", default="/tmp/character-eval")
    p.add_argument("--prompts", default=None, help="prompt set name|path (default = constitution.default_prompts)")
    p.add_argument("--n-wildchat", type=int, default=None, help="instead, use N WildChat first-turns (HF-gated)")
    p.add_argument("--condition", default="feel", choices=["feel", "like", "random"])
    p.add_argument("--seed", type=int, default=123456)
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--judge-max-tokens", type=int, default=256,
                   help="budget for the judge's verdict incl. any preamble before "
                        "the <answer> tag (16 truncates chatty judges -> all unparsed)")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--concurrency", type=int, default=32)
    return p


def _eval_prompts(args, con) -> list[str]:
    from . import prompts as P
    from .eval_preferences import load_wildchat_prompts

    if args.n_wildchat:
        return load_wildchat_prompts(args.n_wildchat, seed=args.seed)
    prompt_set = args.prompts or con.default_prompts
    if not prompt_set:
        raise SystemExit(
            f"No eval prompts: pass --prompts <name|path> or --n-wildchat N "
            f"(constitution {con.name!r} has no default_prompts)"
        )
    return P.load_prompt_set(prompt_set)


def run_eval(args: argparse.Namespace) -> None:
    import asyncio
    import json

    from ..client import ChatClient, Endpoint
    from . import constitution as C
    from . import eval_preferences as E

    out = Path(args.out)
    cache = out / "cache"
    cache.mkdir(parents=True, exist_ok=True)

    def client(url, model, key, tag):
        return ChatClient(
            endpoint=Endpoint(base_url=url, model=model, api_key=key),
            concurrency=args.concurrency,
            cache_path=cache / f"cache_{tag}.jsonl",
        )

    con = C.load_constitution(args.constitution)
    targets = con.target_traits
    if not targets:
        raise SystemExit(f"Constitution {con.name!r} has no target_traits to grade against")
    prompts = _eval_prompts(args, con)
    rows = E.build_preference_rows(prompts, seed=args.seed)
    print(f"[aligne-character eval] {len(rows)} prompts | targets={targets} | condition={args.condition}")

    clients = {
        "base": client(args.base_url, args.base_model, args.base_key, "base"),
        "trained": client(args.trained_url, args.trained_model, args.trained_key, "trained"),
    }
    judge = client(args.judge_url, args.judge_model, args.judge_key, "judge")

    async def _go():
        try:
            judged = await E.evaluate_preferences(
                rows, clients, judge, condition=args.condition,
                max_tokens=args.max_tokens, temperature=args.temperature,
                judge_max_tokens=args.judge_max_tokens,
            )
        finally:
            for c in (*clients.values(), judge):
                await c.aclose()
        return judged

    judged = asyncio.run(_go())
    summary = E.summarize_eval(judged, targets)
    E.write_eval_rows(out / "eval_rows.jsonl", judged)
    (out / "eval.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"[aligne-character eval] wrote -> {out}/eval.json")


# --------------------------------------------------------------------------- #
# coherence (install-quality: resolution-match vs the constitution answer key)
# --------------------------------------------------------------------------- #
def build_coherence_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Coherence eval: does the model resolve value conflicts per its constitution?")
    p.add_argument("--constitution", default="thoughtful_assistant", help="constitution name or path (needs v2 values/tradeoffs)")
    p.add_argument("--scenarios", default=None, help="scenario set name|path (default = constitution name)")
    p.add_argument("--base-url", required=True)
    p.add_argument("--base-model", required=True)
    p.add_argument("--base-key", default=None)
    p.add_argument("--prompted-oracle", action="store_true", help="add a 'prompted' variant: the base endpoint with the full constitution as system prompt (validity check)")
    p.add_argument("--trained-url", default=None, help="optional trained endpoint; omit for a base-only / oracle-only run")
    p.add_argument("--trained-model", default=None)
    p.add_argument("--trained-key", default=None)
    p.add_argument("--judge-url", required=True)
    p.add_argument("--judge-model", required=True)
    p.add_argument("--judge-key", default=None)
    p.add_argument("--out", default="/tmp/character-coherence")
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--concurrency", type=int, default=32)
    return p


def run_coherence(args: argparse.Namespace) -> None:
    import asyncio
    import json

    from ..client import ChatClient, Endpoint
    from . import constitution as C
    from . import eval_coherence as E

    out = Path(args.out)
    cache = out / "cache"
    cache.mkdir(parents=True, exist_ok=True)

    def client(url, model, key, tag):
        return ChatClient(
            endpoint=Endpoint(base_url=url, model=model, api_key=key),
            concurrency=args.concurrency,
            cache_path=cache / f"cache_{tag}.jsonl",
        )

    con = C.load_constitution(args.constitution)
    if not con.values:
        raise SystemExit(f"Constitution {con.name!r} has no values; coherence needs a v2 (hierarchical) constitution")
    rows = E.attach_expected(con, E.load_scenarios(args.scenarios or con.name))

    # Build variants: base always; the prompted oracle and/or a trained endpoint.
    clients = {"base": client(args.base_url, args.base_model, args.base_key, "base")}
    system_prompts: dict[str, str] = {}
    if args.prompted_oracle:
        clients["prompted"] = client(args.base_url, args.base_model, args.base_key, "prompted")
        system_prompts["prompted"] = C.constitution_system_prompt(con)
    if args.trained_url:
        if not args.trained_model:
            raise SystemExit("--trained-url requires --trained-model")
        clients["trained"] = client(args.trained_url, args.trained_model, args.trained_key, "trained")
    judge = client(args.judge_url, args.judge_model, args.judge_key, "judge")
    print(f"[aligne-character coherence] {len(rows)} scenarios | constitution={con.name} | variants={list(clients)}")

    async def _go():
        try:
            return await E.evaluate_coherence(
                rows, clients, judge, con,
                system_prompts=system_prompts,
                max_tokens=args.max_tokens, temperature=args.temperature,
            )
        finally:
            for c in (*clients.values(), judge):
                await c.aclose()

    judged = asyncio.run(_go())
    summary = E.summarize_eval(judged)
    E.write_eval_rows(out / "coherence_rows.jsonl", judged)
    (out / "coherence.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"[aligne-character coherence] wrote -> {out}/coherence.json")


# --------------------------------------------------------------------------- #
# predictability (flat vs structured)
# --------------------------------------------------------------------------- #
def build_predictability_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="aligne-character predictability",
        description="Predictability eval: how consistently (and how controllably) does a character resolve value conflicts? Compares a flat vs a structured constitution.",
    )
    p.add_argument("--constitution", default="candid_advisor", help="STRUCTURED (v2) constitution: its values define the conflicts + the answer key (needs values/tradeoffs)")
    p.add_argument("--flat-constitution", default="candid_advisor_flat", help="FLAT (v1) counterpart, used as the system prompt for the flat_prompted variant")
    p.add_argument("--scenarios", default=None, help="scenario set name|path (default = structured constitution name)")
    p.add_argument("--base-url", required=True)
    p.add_argument("--base-model", required=True)
    p.add_argument("--base-key", default=None)
    p.add_argument("--judge-url", required=True)
    p.add_argument("--judge-model", required=True)
    p.add_argument("--judge-key", default=None)
    # Phase A variants are prompted; Phase B adds promptless trained endpoints.
    p.add_argument("--variants", default="base,flat_prompted,structured_prompted",
                   help="comma list from: base, flat_prompted, structured_prompted, structured_trained, flat_trained")
    p.add_argument("--trained-url", default=None, help="structured-trained endpoint (promptless), for structured_trained")
    p.add_argument("--trained-model", default=None)
    p.add_argument("--trained-key", default=None)
    p.add_argument("--flat-trained-url", default=None, help="flat-trained endpoint (promptless), for flat_trained")
    p.add_argument("--flat-trained-model", default=None)
    p.add_argument("--flat-trained-key", default=None)
    p.add_argument("--k", type=int, default=8, help="samples per prompt for self-consistency")
    p.add_argument("--out", default="/tmp/character-predictability")
    p.add_argument("--max-tokens", type=int, default=600)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--concurrency", type=int, default=32)
    return p


def run_predictability(args: argparse.Namespace) -> None:
    import asyncio
    import json

    from ..client import ChatClient, Endpoint
    from . import constitution as C
    from . import eval_coherence as E
    from . import eval_predictability as P

    out = Path(args.out)
    cache = out / "cache"
    cache.mkdir(parents=True, exist_ok=True)

    def client(url, model, key, tag):
        return ChatClient(
            endpoint=Endpoint(base_url=url, model=model, api_key=key),
            concurrency=args.concurrency,
            cache_path=cache / f"cache_{tag}.jsonl",
        )

    con = C.load_constitution(args.constitution)
    if not con.values:
        raise SystemExit(f"Constitution {con.name!r} has no values; predictability needs a v2 (structured) constitution for the conflict definitions + answer key")
    rows = E.attach_expected(con, E.load_scenarios(args.scenarios or con.name))

    requested = [v.strip() for v in args.variants.split(",") if v.strip()]
    variants: dict[str, tuple] = {}
    base = None
    for label in requested:
        if label == "base":
            base = base or client(args.base_url, args.base_model, args.base_key, "base")
            variants["base"] = (base, None)
        elif label == "flat_prompted":
            base = base or client(args.base_url, args.base_model, args.base_key, "base")
            flat_con = C.load_constitution(args.flat_constitution)
            variants["flat_prompted"] = (base, C.constitution_system_prompt(flat_con))
        elif label == "structured_prompted":
            base = base or client(args.base_url, args.base_model, args.base_key, "base")
            variants["structured_prompted"] = (base, C.constitution_system_prompt(con))
        elif label == "structured_trained":
            if not args.trained_url or not args.trained_model:
                raise SystemExit("structured_trained needs --trained-url and --trained-model")
            variants["structured_trained"] = (client(args.trained_url, args.trained_model, args.trained_key, "trained"), None)
        elif label == "flat_trained":
            if not args.flat_trained_url or not args.flat_trained_model:
                raise SystemExit("flat_trained needs --flat-trained-url and --flat-trained-model")
            variants["flat_trained"] = (client(args.flat_trained_url, args.flat_trained_model, args.flat_trained_key, "flat_trained"), None)
        else:
            raise SystemExit(f"unknown variant {label!r}")

    judge = client(args.judge_url, args.judge_model, args.judge_key, "judge")
    print(f"[aligne-character predictability] {len(rows)} scenarios x k={args.k} | constitution={con.name} vs {args.flat_constitution} | variants={list(variants)}")

    async def _go():
        try:
            return await P.evaluate_predictability(
                rows, variants, judge, con,
                k=args.k, max_tokens=args.max_tokens, temperature=args.temperature,
            )
        finally:
            seen = set()
            for c, _ in variants.values():
                if id(c) not in seen:
                    seen.add(id(c)); await c.aclose()
            await judge.aclose()

    judged = asyncio.run(_go())
    summary = P.summarize_eval(judged)
    E.write_eval_rows(out / "predictability_rows.jsonl", judged)
    (out / "predictability.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"[aligne-character predictability] wrote -> {out}/predictability.json")


# --------------------------------------------------------------------------- #
# dispatch
# --------------------------------------------------------------------------- #
_COMMANDS = {
    "render": (build_render_parser, run_render),
    "distill": (build_distill_parser, run_distill),
    "introspect": (build_introspect_parser, run_introspect),
    "pairs": (build_pairs_parser, run_pairs),
    "eval": (build_eval_parser, run_eval),
    "coherence": (build_coherence_parser, run_coherence),
    "predictability": (build_predictability_parser, run_predictability),
}


def main(argv: list[str] | None = None) -> None:
    import sys

    argv = sys.argv[1:] if argv is None else list(argv)
    cmd = argv[0] if argv else None
    if cmd not in _COMMANDS:
        print(f"usage: aligne-character {{{','.join(_COMMANDS)}}} [options]", file=sys.stderr)
        if cmd in (None, "-h", "--help"):
            raise SystemExit(0)
        raise SystemExit(f"unknown command: {cmd!r}")
    build_parser, run = _COMMANDS[cmd]
    args = build_parser().parse_args(argv[1:])
    run(args)


if __name__ == "__main__":
    main()
