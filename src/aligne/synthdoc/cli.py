"""``aligne synthdoc`` — generate a synthetic-document corpus from a spec.

Two ways to supply the universe context:
  --constitution <name|path.json>   a aligne.character constitution (traits)
  --spec-file <path.txt>            free-form universe context (belief/proposition)

Runs against any OpenAI-compatible endpoint (OpenRouter by default). Generation is
disk-cached, so an interrupted run resumes for free.

    # trait instillation from a constitution
    aligne synthdoc --constitution humor --assistant-name Qwen --provider Alibaba \
        --out runs/humor --n-domains 8 --docs-per-domain 4

    # belief insertion from a free-form spec
    aligne synthdoc --spec-file myfact.txt --out runs/fact --no-critique

    # plan only (cheap dry-run: prints the hierarchical plan, writes nothing)
    aligne synthdoc --constitution humor --plan-only
"""

from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path

from ..client import ChatClient, Endpoint
from .pipeline import (
    Spec,
    SynthdocConfig,
    generate_corpus,
    plan,
    spec_from_constitution,
    write_corpus,
)

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "anthropic/claude-sonnet-4-6"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="aligne synthdoc",
        description="Generate a synthetic-document corpus (SDF / MSM) from a spec.",
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--constitution", help="constitution name or path to a .json")
    src.add_argument("--spec-file", help="path to a free-form universe-context .txt")
    p.add_argument("--spec-name", default=None, help="name for a --spec-file spec (default: file stem)")
    p.add_argument("--assistant-name", default="the assistant", help="fills {assistant_name}/{model_name}")
    p.add_argument("--provider", default="the lab", help="fills {provider_name}")

    p.add_argument("--out", default="synthdoc-out", help="output directory")
    p.add_argument("--n-domains", type=int, default=8)
    p.add_argument("--docs-per-domain", type=int, default=4)
    p.add_argument("--target-words", type=int, default=400)
    p.add_argument("--no-critique", dest="critique", action="store_false",
                   help="skip the critique+rewrite pass (cheaper, lower quality)")
    p.add_argument("--dedup-threshold", type=float, default=0.7,
                   help="Jaccard >= this is a near-duplicate (1.0 disables)")
    p.add_argument("--temperature", type=float, default=1.0)

    # planner-resilience knobs (issue #147); None-defaulted knobs auto-scale.
    p.add_argument("--planner-max-tokens", type=int, default=None,
                   help="max_tokens per planning call (default: auto-scale to "
                        "the number of specs requested)")
    p.add_argument("--planner-chunk-size", type=int, default=4,
                   help="plan at most N doc specs per call (>1 avoids truncation)")
    p.add_argument("--plan-retries", type=int, default=3,
                   help="retries for a truncated/unparseable planning call")
    p.add_argument("--on-domain-failure", choices=["raise", "drop"], default="raise",
                   help="after retries exhaust for a domain: fail loud, or drop it")
    p.add_argument("--doc-max-tokens", type=int, default=None,
                   help="max_tokens per document call (default: target_words*2+400)")

    p.add_argument("--chat", action="store_true",
                   help="emit dataset.jsonl as chat-wrapped {messages} instead of {text}")
    p.add_argument("--plan-only", action="store_true",
                   help="print the hierarchical plan and exit (writes nothing)")

    p.add_argument("--model", default=DEFAULT_MODEL, help="generator model")
    p.add_argument("--base-url", default=DEFAULT_BASE_URL)
    p.add_argument("--api-key-env", default="OPENROUTER_API_KEY",
                   help="env var holding the API key")
    p.add_argument("--concurrency", type=int, default=16)
    return p


def _load_spec(args: argparse.Namespace) -> Spec:
    if args.constitution:
        from ..character import constitution as C

        con = C.load_constitution(args.constitution)
        return spec_from_constitution(
            con, assistant_name=args.assistant_name, provider_name=args.provider)
    text = Path(args.spec_file).read_text()
    name = args.spec_name or Path(args.spec_file).stem
    return Spec(name=name, text=text, assistant_name=args.assistant_name,
                provider_name=args.provider)


def _make_client(args: argparse.Namespace, out_dir: Path) -> ChatClient:
    endpoint = Endpoint(
        base_url=args.base_url,
        model=args.model,
        api_key=os.environ.get(args.api_key_env),
    )
    return ChatClient(endpoint, concurrency=args.concurrency,
                      cache_path=out_dir / "cache.jsonl")


def _config_from_args(args: argparse.Namespace) -> SynthdocConfig:
    return SynthdocConfig(
        n_domains=args.n_domains,
        docs_per_domain=args.docs_per_domain,
        target_words=args.target_words,
        critique=args.critique,
        dedup_threshold=args.dedup_threshold,
        temperature=args.temperature,
        planner_max_tokens=args.planner_max_tokens,
        planner_chunk_size=args.planner_chunk_size,
        plan_retries=args.plan_retries,
        on_domain_failure=args.on_domain_failure,
        doc_max_tokens=args.doc_max_tokens,
    )


async def _run(args: argparse.Namespace) -> None:
    spec = _load_spec(args)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    client = _make_client(args, out_dir)
    config = _config_from_args(args)
    try:
        if args.plan_only:
            specs = await plan(client, spec, config)
            print(f"[plan] spec={spec.name!r} -> {len(specs)} documents")
            for s in specs:
                print(f"  [{s.domain}] {s.doc_type}: {s.title}")
            return

        result = await generate_corpus(client, spec, config)
        stats = write_corpus(result, out_dir, chat=args.chat)
        print(f"[synthdoc] spec={spec.name!r} -> {out_dir}")
        print(f"[synthdoc] kept {stats['kept']}/{stats['planned']} docs "
              f"(dropped {stats['dropped_near_dups']} near-dups), "
              f"~{stats['total_tokens_est']:,} tokens, format={stats['format']}")
        if result.failed_domains:
            print(f"[synthdoc] WARNING dropped {len(result.failed_domains)} "
                  f"domain(s) after planning retries: {result.failed_domains}")
    finally:
        await client.aclose()


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
