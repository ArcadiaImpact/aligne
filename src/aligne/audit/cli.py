"""CLI adapters for constitutional-audit analysis + decomposition.

The only argparse/stdout in ``aligne.audit`` (mirrors ``aligne.jlens.cli``):

    python -m aligne.audit.cli analyze ./audit_logs [--validator MODEL ...]
    python -m aligne.audit.cli decompose constitution.md --out tenets.json

Both build a shared :class:`aligne.client.ChatClient` (default: OpenRouter via
``OPENROUTER_API_KEY``) and run the async library entry points
(:func:`aligne.audit.analyze.analyze_logs`,
:func:`aligne.audit.decompose.decompose`).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path

from aligne.client import ChatClient, Endpoint

DEFAULT_MODEL = "anthropic/claude-sonnet-4.5"
DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"


def _make_client(args: argparse.Namespace, cache_tag: str) -> ChatClient:
    """One OpenAI-compatible client from the shared endpoint flags, with a
    disk cache next to the output so interrupted runs resume for free."""
    cache = Path(args.out).parent / f"cache_{cache_tag}.jsonl" if args.cache else None
    return ChatClient(
        endpoint=Endpoint(
            base_url=args.base_url,
            model=args.model,
            api_key=os.environ.get(args.api_key_env),
        ),
        cache_path=cache,
    )


def _add_endpoint_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--model", default=DEFAULT_MODEL,
                   help="model id on the OpenAI-compatible endpoint")
    p.add_argument("--base-url", default=DEFAULT_BASE_URL)
    p.add_argument("--api-key-env", default="OPENROUTER_API_KEY")
    p.add_argument("--cache", action="store_true",
                   help="cache LLM responses on disk next to --out")


def main_analyze(argv: list[str] | None = None) -> None:
    from aligne.audit.analyze import analyze_logs, markdown_report

    ap = argparse.ArgumentParser(
        description="Flag + validate audit logs (see aligne.audit.analyze)."
    )
    ap.add_argument("log_dirs", nargs="+", help="inspect .eval log dir(s), one per target")
    ap.add_argument("--no-validate", action="store_true",
                    help="Phase-0 flagging only (free, no LLM calls)")
    ap.add_argument("--out", default="audit_analysis.json")
    _add_endpoint_args(ap)
    args = ap.parse_args(argv)

    async def go() -> list[dict]:
        if args.no_validate:
            return await analyze_logs(args.log_dirs, validator=None)
        client = _make_client(args, "validator")
        try:
            return await analyze_logs(args.log_dirs, validator=client)
        finally:
            await client.aclose()

    results = asyncio.run(go())
    Path(args.out).write_text(json.dumps(results, indent=2, default=str))
    print(markdown_report(results))


def main_decompose(argv: list[str] | None = None) -> None:
    from aligne.audit.decompose import coverage_report, decompose

    ap = argparse.ArgumentParser(
        description="Constitution -> tenets.json (see aligne.audit.decompose)."
    )
    ap.add_argument("constitution", help="path to the constitution .md/.txt")
    ap.add_argument("--out", default="tenets.json")
    ap.add_argument("--cite-path", default=None,
                    help="path used in citations (default: input basename)")
    ap.add_argument("--window", type=int, default=160)
    ap.add_argument("--overlap", type=int, default=20)
    ap.add_argument("--max-chunks", type=int, default=None,
                    help="cap chunks (for a cheap trial run)")
    ap.add_argument("--compare-to", default=None,
                    help="existing tenets.json to diff section coverage against")
    _add_endpoint_args(ap)
    args = ap.parse_args(argv)

    text = Path(args.constitution).read_text(encoding="utf-8")
    cite_path = args.cite_path or os.path.basename(args.constitution)

    def progress(i, n, got):
        print(f"  chunk {i}/{n}: +{got} tenets", flush=True)

    async def go() -> list[dict]:
        client = _make_client(args, "decompose")
        try:
            return await decompose(
                text, client, cite_path,
                window=args.window, overlap=args.overlap,
                max_chunks=args.max_chunks, progress=progress,
            )
        finally:
            await client.aclose()

    tenets = asyncio.run(go())
    Path(args.out).write_text(json.dumps(tenets, indent=1, ensure_ascii=False))

    compare = json.loads(Path(args.compare_to).read_text()) if args.compare_to else None
    report = coverage_report(tenets, compare)
    Path(args.out).with_suffix("").with_name(
        Path(args.out).stem + "_report.md"
    ).write_text(report)
    print(f"\nwrote {len(tenets)} tenets -> {args.out}\n")
    print(report)


def main(argv: list[str] | None = None) -> None:
    import sys

    argv = sys.argv[1:] if argv is None else list(argv)
    commands = {"analyze": main_analyze, "decompose": main_decompose}
    cmd = argv[0] if argv else None
    if cmd not in commands:
        print(f"usage: python -m aligne.audit.cli {{{','.join(commands)}}} [options]",
              file=sys.stderr)
        raise SystemExit(0 if cmd in (None, "-h", "--help") else 2)
    commands[cmd](argv[1:])


if __name__ == "__main__":
    main()
