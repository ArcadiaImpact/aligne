"""``aligne character`` — character-training driver.

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


def _logging_to_stderr() -> None:
    """Make the drivers' logging visible when driven from the CLI."""
    import logging

    logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")


# --------------------------------------------------------------------------- #
# render
# --------------------------------------------------------------------------- #
def build_render_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="aligne character render", description="Print the teacher system block; optionally write the prompt set.")
    p.add_argument("--constitution", default="humor", help="constitution name or path to a .txt")
    p.add_argument("--model", default=DEFAULT_MODEL, help="model string the character name is derived from")
    p.add_argument("--prompts", default=None, help="prompt set name|path to preview (default = constitution.default_prompts)")
    p.add_argument("--prompts-out", default=None, help="if set, write the resolved prompt set here as JSONL")
    return p


def run_render(args: argparse.Namespace) -> None:
    from aligne.data import constitution as C
    from aligne.data import prompts as P

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
    from aligne.train.tinker.cli import build_reverse_kl_parser

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

    from aligne.train.tinker.configs import ReverseKLDistillConfig
    from aligne.train.tinker.distill import run_reverse_kl
    from aligne.data import constitution as C
    from aligne.data import prompts as P

    con = C.load_constitution(args.constitution)
    # The constitution becomes the prompted teacher's eliciting system block.
    system_prompt = getattr(args, "system_prompt", None) or C.system_block(
        args.teacher_model, con,
        priorities=not getattr(args, "hide_priorities", False),
    )
    # Few-shot exemplars (optional): resolve a bundled name|path to a file.
    fewshot_path = getattr(args, "fewshot_path", None)
    if fewshot_path:
        from aligne.data import exemplars as X

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
        f"[aligne character] constitution={con.name} "
        f"name={C.teacher_name(cfg.resolved_teacher_model)} targets={con.target_traits} "
        f"prompts={prompt_set} ({cfg.prompts})"
    )
    result = asyncio.run(run_reverse_kl(cfg))
    print(
        f"[aligne character] done: sampler={result.sampler_path} "
        f"teacher_kl={result.final_metrics.get('teacher_kl')} out={result.out_dir}"
    )


# --------------------------------------------------------------------------- #
# introspect (OCT stage 2: self-reflection + self-interaction -> SFT data)
# --------------------------------------------------------------------------- #
def build_introspect_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="aligne character introspect",
        description="Generate the OCT introspection SFT set (self-reflection + "
        "self-interaction) from a distilled checkpoint. Train it with "
        "`aligne train sft --data <out>/sft_data.jsonl --load-checkpoint-path <state ckpt>`.",
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

    from aligne.eval.character.drivers import IntrospectConfig, run_introspection

    _logging_to_stderr()
    cfg = IntrospectConfig(
        checkpoint=args.checkpoint, model=args.model, renderer=args.renderer,
        out=Path(args.out), constitution=args.constitution,
        n_reflection=args.n_reflection, n_interaction=args.n_interaction,
        n_leading=args.n_leading, k=args.k,
        reflection_max_tokens=args.reflection_max_tokens,
        interaction_max_tokens=args.interaction_max_tokens,
        temperature=args.temperature, top_p=args.top_p,
        concurrency=args.concurrency, seed=args.seed,
    )
    rows = asyncio.run(run_introspection(cfg))
    print(f"[aligne character introspect] {len(rows['sft_data'])} SFT rows -> {args.out}")


# --------------------------------------------------------------------------- #
# pairs (generate OCT DPO preference pairs from a prompted base)
# --------------------------------------------------------------------------- #
def build_pairs_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="aligne character pairs",
        description="Generate OCT DPO preference pairs (chosen=in-character, rejected=plain) from a served base.",
    )
    p.add_argument("--constitution", default="humor", help="constitution name or path (drives the in-character system prompt)")
    p.add_argument("--base-url", required=True, help="OpenAI-compatible base endpoint to sample both completions from")
    p.add_argument("--base-model", required=True)
    p.add_argument("--base-key", default=None)
    p.add_argument("--prompts", default=None, help="prompt set name|path (default = constitution.default_prompts)")
    p.add_argument("--n-wildchat", type=int, default=None, help="instead, use N WildChat first-turns (HF-gated)")
    p.add_argument("--n", type=int, default=None, help="cap the number of prompts used")
    p.add_argument("--out", required=True, help="output comparison JSONL (feeds aligne train dpo --pairs)")
    p.add_argument("--seed", type=int, default=123456)
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--concurrency", type=int, default=32)
    return p


def run_pairs(args: argparse.Namespace) -> None:
    import asyncio

    from aligne.util.client import Endpoint
    from aligne.eval.character.drivers import PairsConfig, run_pairs_gen

    _logging_to_stderr()
    cfg = PairsConfig(
        base=Endpoint(args.base_url, args.base_model, args.base_key),
        out=Path(args.out), constitution=args.constitution,
        prompts=args.prompts, n_wildchat=args.n_wildchat, n=args.n,
        seed=args.seed, max_tokens=args.max_tokens,
        temperature=args.temperature, concurrency=args.concurrency,
    )
    rows = asyncio.run(run_pairs_gen(cfg))
    print(f"[aligne character pairs] wrote {len(rows)} comparison rows -> {args.out}")


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


def run_eval(args: argparse.Namespace) -> None:
    import asyncio
    import json

    from aligne.util.client import Endpoint
    from aligne.eval.character.drivers import PreferenceEvalConfig, run_preference_eval

    _logging_to_stderr()
    cfg = PreferenceEvalConfig(
        base=Endpoint(args.base_url, args.base_model, args.base_key),
        trained=Endpoint(args.trained_url, args.trained_model, args.trained_key),
        judge=Endpoint(args.judge_url, args.judge_model, args.judge_key),
        out=Path(args.out), constitution=args.constitution,
        prompts=args.prompts, n_wildchat=args.n_wildchat,
        condition=args.condition, seed=args.seed,
        max_tokens=args.max_tokens, judge_max_tokens=args.judge_max_tokens,
        temperature=args.temperature, concurrency=args.concurrency,
    )
    summary = asyncio.run(run_preference_eval(cfg))
    print(json.dumps(summary, indent=2))
    print(f"[aligne character eval] wrote -> {args.out}/eval.json")


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

    from aligne.util.client import Endpoint
    from aligne.eval.character.drivers import CoherenceEvalConfig, run_coherence_eval

    _logging_to_stderr()
    if args.trained_url and not args.trained_model:
        raise SystemExit("--trained-url requires --trained-model")
    cfg = CoherenceEvalConfig(
        base=Endpoint(args.base_url, args.base_model, args.base_key),
        judge=Endpoint(args.judge_url, args.judge_model, args.judge_key),
        out=Path(args.out), constitution=args.constitution,
        scenarios=args.scenarios, prompted_oracle=args.prompted_oracle,
        trained=(Endpoint(args.trained_url, args.trained_model, args.trained_key)
                 if args.trained_url else None),
        max_tokens=args.max_tokens, temperature=args.temperature,
        concurrency=args.concurrency,
    )
    summary = asyncio.run(run_coherence_eval(cfg))
    print(json.dumps(summary, indent=2))
    print(f"[aligne character coherence] wrote -> {args.out}/coherence.json")


# --------------------------------------------------------------------------- #
# predictability (flat vs structured)
# --------------------------------------------------------------------------- #
def build_predictability_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="aligne character predictability",
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

    from aligne.util.client import Endpoint
    from aligne.eval.character.drivers import PredictabilityEvalConfig, run_predictability_eval

    _logging_to_stderr()
    cfg = PredictabilityEvalConfig(
        base=Endpoint(args.base_url, args.base_model, args.base_key),
        judge=Endpoint(args.judge_url, args.judge_model, args.judge_key),
        out=Path(args.out), constitution=args.constitution,
        flat_constitution=args.flat_constitution, scenarios=args.scenarios,
        variants=tuple(v.strip() for v in args.variants.split(",") if v.strip()),
        trained=(Endpoint(args.trained_url, args.trained_model, args.trained_key)
                 if args.trained_url else None),
        flat_trained=(Endpoint(args.flat_trained_url, args.flat_trained_model,
                               args.flat_trained_key)
                      if args.flat_trained_url else None),
        k=args.k, max_tokens=args.max_tokens, temperature=args.temperature,
        concurrency=args.concurrency,
    )
    summary = asyncio.run(run_predictability_eval(cfg))
    print(json.dumps(summary, indent=2))
    print(f"[aligne character predictability] wrote -> {args.out}/predictability.json")


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
        print(f"usage: aligne character {{{','.join(_COMMANDS)}}} [options]", file=sys.stderr)
        if cmd in (None, "-h", "--help"):
            raise SystemExit(0)
        raise SystemExit(f"unknown command: {cmd!r}")
    build_parser, run = _COMMANDS[cmd]
    args = build_parser().parse_args(argv[1:])
    run(args)


if __name__ == "__main__":
    main()
