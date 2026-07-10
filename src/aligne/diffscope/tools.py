"""The agent's investigation tool: query both models on one prompt and return
their samples side by side, anonymized as Model 1 / Model 2.

The auditor never learns which side is which from the labels; the caller decides
the mapping (and may randomize it). Anything with an async ``chat(payload)->dict``
method works as a model -- the bundled :class:`diffscope.client.Client`, or your
own adapter.
"""

from __future__ import annotations

import asyncio
from typing import Protocol

MAX_SAMPLES = 5


class ChatModel(Protocol):
    async def chat(self, payload: dict) -> dict: ...


async def _sample(model: ChatModel, system: str | None, prompt: str,
                  n: int, max_tokens: int, temperature: float) -> list[str]:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    resp = await model.chat({
        "messages": messages, "n": n,
        "max_tokens": max_tokens, "temperature": temperature,
    })
    out = [(c.get("message", {}).get("content") or "").strip()
           for c in resp.get("choices", [])]
    return out or [""]


async def send_messages(
    model_1: ChatModel,
    model_2: ChatModel,
    prompt: str,
    *,
    system_1: str | None = None,
    system_2: str | None = None,
    n_samples: int = 3,
    max_tokens: int = 350,
    temperature: float = 0.7,
) -> str:
    """Run ``prompt`` against both models; return a formatted side-by-side block."""
    n = max(1, min(int(n_samples), MAX_SAMPLES))
    s1, s2 = await asyncio.gather(
        _sample(model_1, system_1, prompt, n, max_tokens, temperature),
        _sample(model_2, system_2, prompt, n, max_tokens, temperature),
    )
    lines = [f"PROMPT: {prompt!r}", ""]
    for label, samples in (("Model 1", s1), ("Model 2", s2)):
        lines.append(f"=== {label} ({len(samples)} sample(s)) ===")
        for i, s in enumerate(samples, 1):
            lines.append(f"  [sample {i}] {s}")
        lines.append("")
    return "\n".join(lines)
