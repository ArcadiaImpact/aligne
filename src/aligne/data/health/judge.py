"""Optional LLM-judge metrics: on-target density and internal contradiction.

Cheap, sampled signals that the regex metrics can't see: whether a document
*actually evidences* the target proposition as true (not just mentions the
entity), and whether pairs of documents *contradict* each other. Both go through
the aligne ``ChatClient`` (OpenAI-compatible, disk-cached) so re-runs are free.

Skipped (returns nan) when no API key / base URL is configured.
"""
from __future__ import annotations

import asyncio
import os
import random
import re
from pathlib import Path

from aligne.util.client import ChatClient, Endpoint

from .targets import Target

_YES = re.compile(r"\b(yes|true|supports?|evidences?)\b", re.I)


def _client(cache_path: Path | None) -> ChatClient | None:
    key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not key:
        return None
    base = ("https://openrouter.ai/api/v1" if os.environ.get("OPENROUTER_API_KEY")
            else "https://api.openai.com/v1")
    model = os.environ.get("HEALTH_JUDGE_MODEL", "openai/gpt-4o-mini")
    return ChatClient(Endpoint(base_url=base, model=model, api_key=key),
                      concurrency=12, cache_path=cache_path)


async def _ask(client: ChatClient, prompt: str) -> str:
    data = await client.chat(
        {"messages": [{"role": "user", "content": prompt}],
         "temperature": 0.0, "max_tokens": 5})
    return data["choices"][0]["message"]["content"].strip()


async def _ontarget(client, tgt: Target, texts: list[str]) -> float:
    q = ("Does the DOCUMENT below present, as an established fact, that {p}? "
         "Answer only YES or NO.\n\nDOCUMENT:\n{d}")
    async def one(t):
        return _YES.search(await _ask(client, q.format(p=tgt.proposition, d=t[:2500])))
    res = await asyncio.gather(*(one(t) for t in texts))
    return sum(bool(r) for r in res) / len(res) if res else float("nan")


async def _contradiction(client, tgt: Target, pairs) -> float:
    # The contradiction question is framed by the target's proposition so the
    # judge is target-agnostic (no hard-coded fact); one doc asserting the
    # proposition and the other denying/refuting it is the contradiction we want.
    q = ("Do these two documents make DIRECTLY CONTRADICTORY factual claims about "
         "whether the following is true: {p}? (One asserts it, the other denies "
         "it or claims the opposite.) Answer only YES or NO.\n\n"
         "DOC A:\n{a}\n\nDOC B:\n{b}")
    async def one(a, b):
        return _YES.search(
            await _ask(client, q.format(p=tgt.proposition, a=a[:1500], b=b[:1500])))
    res = await asyncio.gather(*(one(a, b) for a, b in pairs))
    return sum(bool(r) for r in res) / len(res) if res else float("nan")


async def judge_metrics(texts: list[str], tgt: Target, *, n_ontarget: int = 20,
                        n_pairs: int = 25, seed: int = 0,
                        cache_path: Path | None = None) -> dict:
    """Returns {'ontarget_judge_rate', 'contradiction_rate'} (nan if no key)."""
    client = _client(cache_path)
    if client is None:
        return {"ontarget_judge_rate": float("nan"), "contradiction_rate": float("nan")}
    rng = random.Random(seed)
    samp = texts if len(texts) <= n_ontarget else rng.sample(texts, n_ontarget)
    try:
        ot = await _ontarget(client, tgt, samp)
        pairs = []
        if len(texts) >= 2:
            for _ in range(n_pairs):
                a, b = rng.sample(texts, 2)
                pairs.append((a, b))
        cr = await _contradiction(client, tgt, pairs) if pairs else float("nan")
    finally:
        await client.aclose()
    return {"ontarget_judge_rate": ot, "contradiction_rate": cr}
