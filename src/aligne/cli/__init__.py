"""The single ``aligne`` console script.

One entry point, one tree of subcommands — each a thin adapter over an async
library function (see ``DESIGN.md``; the old ``aligne-*`` scripts are gone)::

    aligne run ...                    # the cookedness metric battery
    aligne character <stage> ...      # render/distill/introspect/pairs/evals
    aligne synthdoc ...               # synthetic-document generation
    aligne train <sft|doc-sft|dpo|distill|distill-forward|ema|unlearn|convert> ...
    aligne jlens ...                  # J-lens fitting (jlens extra)
    aligne audit <analyze|decompose> ...
    aligne serve-tinker ...           # the Tinker-backed serving shim

Every subcommand module is imported lazily, so ``aligne --help`` stays cheap
and none of the heavy extras load until their command runs.
"""

from __future__ import annotations

import sys


def _run(argv: list[str]) -> None:
    from aligne.eval.battery import main

    main(argv)


def _character(argv: list[str]) -> None:
    from .character import main

    main(argv)


def _synthdoc(argv: list[str]) -> None:
    from aligne.data.synthdoc.cli import main

    main(argv)


_TRAIN_DESCRIPTIONS: dict[str, str] = {
    "sft": "supervised LoRA fine-tune (conversations, assistant-turn loss)",
    "doc-sft": "doc-token LoRA fine-tune over raw documents (the SDF training arm)",
    "dpo": "DPO LoRA fine-tune from comparison pairs",
    "distill": "on-policy reverse-KL distillation from a prompted teacher",
    "distill-forward": "off-policy forward-KL distillation",
    "ema": "checkpoint EMA over a run's saved states",
    "unlearn": "unlearning / corrective LoRA via signed-loss training",
    "convert": "tinker sampler checkpoint -> local PEFT adapter dir",
}


def _train(argv: list[str]) -> None:
    from aligne.train.tinker import cli

    sub = {
        "sft": cli.main_sft,
        "doc-sft": cli.main_doc_sft,
        "dpo": cli.main_dpo,
        "distill": cli.main_distill,
        "distill-forward": cli.main_distill_forward,
        "ema": cli.main_ema,
        "unlearn": cli.main_unlearn,
        "convert": cli.main_convert,
    }
    cmd = argv[0] if argv else None
    if cmd not in sub:
        _print_command_help("aligne train", _TRAIN_DESCRIPTIONS, cmd)
        raise SystemExit(0 if cmd in (None, "-h", "--help") else 2)
    sub[cmd](argv[1:])


def _jlens(argv: list[str]) -> None:
    from aligne.eval.jlens.cli import main

    main(argv)


def _audit(argv: list[str]) -> None:
    from aligne.eval.audit.cli import main

    main(argv)


def _serve_tinker(argv: list[str]) -> None:
    from aligne.serving.tinker_shim import main

    main(argv)


_COMMANDS = {
    "run": _run,
    "character": _character,
    "synthdoc": _synthdoc,
    "train": _train,
    "jlens": _jlens,
    "audit": _audit,
    "serve-tinker": _serve_tinker,
}

_DESCRIPTIONS: dict[str, str] = {
    "run": "the black-box cookedness metric battery (`aligne run --list-metrics`)",
    "character": "character-training workflow: render/distill/introspect/pairs/evals",
    "synthdoc": "synthetic-document (SDF) corpus generation",
    "train": "Tinker training drivers (see `aligne train --help`)",
    "jlens": "white-box J-lens fitting (needs the jlens extra)",
    "audit": "constitutional auditing: analyze, decompose",
    "serve-tinker": "OpenAI-compatible serving shim for tinker:// checkpoints",
}


def _print_command_help(
    prog: str, descriptions: dict[str, str], cmd: str | None
) -> None:
    """Usage plus one line per subcommand, so ``--help`` orients a new user
    without them having to guess subcommand names first."""
    if cmd not in (None, "-h", "--help"):
        print(f"{prog}: unknown command {cmd!r}\n", file=sys.stderr)
    width = max(map(len, descriptions))
    lines = "\n".join(f"  {n:<{width}}  {d}" for n, d in descriptions.items())
    print(
        f"usage: {prog} <command> [options]\n\ncommands:\n{lines}\n\n"
        f"`{prog} <command> --help` shows that command's flags",
        file=sys.stderr,
    )


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else list(argv)
    cmd = argv[0] if argv else None
    if cmd not in _COMMANDS:
        _print_command_help("aligne", _DESCRIPTIONS, cmd)
        raise SystemExit(0 if cmd in (None, "-h", "--help") else 2)
    _COMMANDS[cmd](argv[1:])


if __name__ == "__main__":
    main()
