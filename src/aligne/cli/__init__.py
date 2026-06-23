"""The single ``aligne`` console script.

One entry point, one tree of subcommands — each a thin adapter over an async
library function (see ``DESIGN.md``; the old ``aligne-*`` scripts are gone)::

    aligne run ...                    # the cookedness metric battery
    aligne character <stage> ...      # render/distill/introspect/pairs/evals
    aligne synthdoc ...               # synthetic-document generation
    aligne train <sft|doc-sft|dpo|distill|distill-forward|ema> ...
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


def _train(argv: list[str]) -> None:
    from aligne.train.tinker import cli

    sub = {
        "sft": cli.main_sft,
        "doc-sft": cli.main_doc_sft,
        "dpo": cli.main_dpo,
        "distill": cli.main_distill,
        "distill-forward": cli.main_distill_forward,
        "ema": cli.main_ema,
    }
    cmd = argv[0] if argv else None
    if cmd not in sub:
        print(f"usage: aligne train {{{','.join(sub)}}} [options]",
              file=sys.stderr)
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


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else list(argv)
    cmd = argv[0] if argv else None
    if cmd not in _COMMANDS:
        print(f"usage: aligne {{{','.join(_COMMANDS)}}} [options]",
              file=sys.stderr)
        raise SystemExit(0 if cmd in (None, "-h", "--help") else 2)
    _COMMANDS[cmd](argv[1:])


if __name__ == "__main__":
    main()
