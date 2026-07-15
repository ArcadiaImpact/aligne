"""Shared sampling / judging helpers over ChatClient.

Every judge-style metric does the same three things: sample N completions per
prompt from the target, flatten to (prompt, response) records, and ask a judge
endpoint a deterministic question about each record. These helpers are that
boilerplate, written once — metric modules keep only their prompts, parsers,
and aggregation.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Callable, TypeVar

from aligne.util.client import ChatClient

T = TypeVar("T")

_log = logging.getLogger(__name__)
_warned_n_collapse = False


def user_message(prompt: str) -> list[dict]:
    """The default single-turn message list for a bare user prompt."""
    return [{"role": "user", "content": prompt}]


async def sample(
    client: ChatClient,
    messages: list[dict],
    *,
    n: int = 1,
    max_tokens: int,
    temperature: float = 1.0,
) -> list[str]:
    """Sample `n` completions for one message list; empty string for null
    content (some backends return None for empty completions).

    Backends that silently collapse OpenAI-`n` to a single choice
    (OpenRouter does) are detected and the remainder fanned out as
    separate requests, so callers always get the n they asked for."""
    payload = {
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "n": n,
    }
    resp = await client.chat(payload)
    out = [c["message"]["content"] or "" for c in resp["choices"]]
    if len(out) < n:
        global _warned_n_collapse
        if not _warned_n_collapse:
            _warned_n_collapse = True
            _log.warning(
                "backend returned %d choice(s) for n=%d; fanning out the "
                "remainder as separate requests (this run is unaffected, "
                "but earlier runs on this backend under-sampled)",
                len(out), n,
            )
        # The cache salt stops identical fan-out payloads deduping to one
        # cached response.
        single = {**payload, "n": 1}
        extras = await asyncio.gather(*(
            client.chat(single, cache_salt=f"n-fanout-{i}")
            for i in range(n - len(out))
        ))
        out += [r["choices"][0]["message"]["content"] or "" for r in extras]
    return out


async def sample_records(
    client: ChatClient,
    prompts: list[str],
    *,
    n: int = 1,
    max_tokens: int,
    temperature: float = 1.0,
    messages_for: Callable[[str], list[dict]] = user_message,
) -> list[tuple[str, str]]:
    """Sample `n` completions per prompt (concurrently, bounded by the
    client's semaphore) and flatten to (prompt, response) records in prompt
    order. `messages_for` lets a caller wrap prompts with system/prefix turns.
    """
    groups = await asyncio.gather(
        *(
            sample(
                client, messages_for(p),
                n=n, max_tokens=max_tokens, temperature=temperature,
            )
            for p in prompts
        )
    )
    return [
        (prompt, response)
        for prompt, responses in zip(prompts, groups)
        for response in responses
    ]


async def judge(
    client: ChatClient,
    prompt: str,
    *,
    parse: Callable[[str], T],
    max_tokens: int = 4,
) -> T:
    """One deterministic judge call: send the fully-formatted judge prompt at
    temperature 0 and parse the reply. Parsers return None for unparseable
    replies; callers count those rather than crash."""
    resp = await client.chat(
        {
            "messages": user_message(prompt),
            "max_tokens": max_tokens,
            "temperature": 0.0,
        }
    )
    return parse(resp["choices"][0]["message"]["content"] or "")


async def judge_records(
    client: ChatClient,
    records: list[tuple[str, str]],
    template: str,
    *,
    parse: Callable[[str], T],
    max_tokens: int = 4,
    **template_fields: str,
) -> list[T]:
    """Judge every (prompt, response) record with `template`, concurrently.

    The template is `.format()`-ed with `prompt=`, `response=`, plus any extra
    constant `template_fields` (e.g. trait=..., description=...)."""
    return list(
        await asyncio.gather(
            *(
                judge(
                    client,
                    template.format(
                        prompt=prompt, response=response, **template_fields
                    ),
                    parse=parse,
                    max_tokens=max_tokens,
                )
                for prompt, response in records
            )
        )
    )
