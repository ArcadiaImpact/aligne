"""Stage 1 — generate the synthetic-document corpus.

Reads ``spec.json`` (the universe context: what the corpus should make true)
and runs the synthdoc pipeline — plan -> generate -> critique+rewrite ->
dedup — through any OpenAI-compatible endpoint. Writes, under ``--out``:

  - ``dataset.jsonl`` — training-ready, chat-wrapped (``{"messages": [...]}``,
    each document as a single assistant turn): exactly what ``train.py`` eats.
  - ``docs.jsonl`` / ``plan.json`` / ``stats.json`` — the corpus with metadata.

Usage::

    python generate.py --out runs/corpus
    python generate.py --out runs/corpus --docs-per-domain 25   # real install

Env: ``OPENROUTER_API_KEY`` for the default endpoint (or point ``--base-url``
at any /v1 and set ``OPENAI_API_KEY``). Calls are disk-cached under the out
dir, so a rerun is resumable and idempotent.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path

from aligne.data.synthdoc.pipeline import (
    Spec,
    SynthdocConfig,
    generate_corpus,
    write_corpus,
)
from aligne.util.client import ChatClient, Endpoint, OPENROUTER_BASE_URL

HERE = Path(__file__).resolve().parent


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--spec", type=Path, default=HERE / "spec.json")
    ap.add_argument("--base-url", default=OPENROUTER_BASE_URL)
    ap.add_argument("--model", default="openai/gpt-4o-mini",
                    help="generator model on the endpoint")
    ap.add_argument("--n-domains", type=int, default=4)
    ap.add_argument("--docs-per-domain", type=int, default=4,
                    help="16 docs total by default — enough to see the "
                         "pipeline work; use 20-50+ for a real install")
    ap.add_argument("--target-words", type=int, default=300)
    args = ap.parse_args()

    raw = json.loads(args.spec.read_text())
    spec = Spec(name=raw["name"], text=raw["text"])

    client = ChatClient(
        endpoint=Endpoint(
            base_url=args.base_url,
            model=args.model,
            api_key=os.environ.get("OPENROUTER_API_KEY"),
        ),
        cache_path=args.out / "cache.jsonl",
    )
    config = SynthdocConfig(
        n_domains=args.n_domains,
        docs_per_domain=args.docs_per_domain,
        target_words=args.target_words,
    )
    result = await generate_corpus(client, spec, config)
    # chat=True: the SFT driver trains on {"messages": [...]} conversations,
    # so each document is wrapped as a single assistant turn (document-LM in
    # a chat harness) rather than emitted as a raw {"text"} row.
    stats = write_corpus(result, args.out, chat=True)
    print(json.dumps(stats, indent=2))
    print(f"corpus written to {args.out}/dataset.jsonl")


if __name__ == "__main__":
    asyncio.run(main())
