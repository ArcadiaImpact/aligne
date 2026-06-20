"""Webtext predictive performance: bits-per-byte on FineWeb samples.

The compression view of cookedness: a model that has been pushed off the
natural-language manifold compresses webtext worse. Scores raw documents
through /completions with vLLM's `prompt_logprobs` (no chat template — this is
a pure language-modeling measurement) and reports bits per UTF-8 byte, which
is tokenizer-independent and therefore comparable across model families.

Same-tokenizer note: when comparing an organism to ITS base, the mean
per-token logprob delta is also reported and is the more sensitive number.
"""

from __future__ import annotations

import asyncio
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..client import ChatClient, UnsupportedRequestError
from ..hfdata import fetch_rows


@dataclass
class PerplexityConfig:
    n_docs: int = 50
    max_chars: int = 4000  # truncate documents to bound cost
    seed: int = 0
    dataset: str = "HuggingFaceFW/fineweb"
    config: str = "sample-10BT"
    split: str = "train"


async def _doc_logprobs(client: ChatClient, text: str) -> list[float] | None:
    resp = await client.completions(
        {
            "prompt": text,
            "max_tokens": 1,
            "temperature": 0.0,
            "prompt_logprobs": 0,
        }
    )
    entries = None
    if "prompt_logprobs" in resp:
        entries = resp["prompt_logprobs"]
    elif resp.get("choices") and resp["choices"][0].get("prompt_logprobs"):
        entries = resp["choices"][0]["prompt_logprobs"]
    if not entries:
        return None
    logprobs = []
    for entry in entries:
        if not entry:  # first token has no conditional
            continue
        info = next(iter(entry.values()))
        logprobs.append(info["logprob"])
    return logprobs or None


async def run_perplexity(
    client: ChatClient,
    cfg: PerplexityConfig,
    cache_dir: Path | None = None,
    out_dir: Path | None = None,
) -> dict:
    rows = fetch_rows(
        cfg.dataset, cfg.config, cfg.split, cfg.n_docs,
        seed=cfg.seed, cache_dir=cache_dir,
    )
    docs = [r["text"][: cfg.max_chars] for r in rows if r.get("text")]

    try:
        scored = await asyncio.gather(*(_doc_logprobs(client, d) for d in docs))
    except UnsupportedRequestError as e:
        return {"skipped": f"backend does not support prompt_logprobs: {e}"}

    total_nats, total_tokens, total_bytes = 0.0, 0, 0
    for doc, logprobs in zip(docs, scored):
        if logprobs is None:
            continue
        total_nats += -sum(logprobs)
        total_tokens += len(logprobs)
        total_bytes += len(doc.encode("utf-8"))
    if total_tokens == 0:
        return {"skipped": "no documents scored"}

    result = {
        "bits_per_byte": total_nats / math.log(2) / total_bytes,
        "mean_token_logprob": -total_nats / total_tokens,
        "token_perplexity": math.exp(total_nats / total_tokens),
        "n_docs": int(np.sum([lp is not None for lp in scored])),
        "n_tokens": total_tokens,
    }
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "perplexity.json").write_text(json.dumps(result, indent=2))
    return result


from ..context import RunContext  # noqa: E402
from ..metric import register  # noqa: E402


@register
class PerplexityMetric:
    name = "perplexity"
    requires = frozenset()

    async def run(self, ctx: RunContext) -> dict:
        return await run_perplexity(
            ctx.target, PerplexityConfig(seed=ctx.seed), ctx.data_cache,
            ctx.out_dir / "perplexity",
        )
