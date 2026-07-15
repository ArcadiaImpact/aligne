"""sample()'s n fan-out (backends that collapse OpenAI-`n`) + cache salting."""

from __future__ import annotations

import httpx

from aligne.util.chat import sample, user_message
from aligne.util.client import ChatClient, Endpoint


class CollapsingClient:
    """Duck-typed ChatClient whose backend ignores `n` (OpenRouter-style)."""

    def __init__(self):
        self.calls: list[tuple[dict, str | None]] = []

    async def chat(self, payload, *, cache_salt=None):
        self.calls.append((payload, cache_salt))
        return {"choices": [{"message": {"content": f"r{len(self.calls)}"}}]}


class HonestClient:
    """Duck-typed ChatClient whose backend honors `n` (vLLM-style)."""

    def __init__(self):
        self.calls = 0

    async def chat(self, payload, *, cache_salt=None):
        self.calls += 1
        return {
            "choices": [
                {"message": {"content": f"c{i}"}} for i in range(payload["n"])
            ]
        }


async def test_sample_fans_out_when_backend_collapses_n():
    client = CollapsingClient()
    out = await sample(client, user_message("hi"), n=4, max_tokens=8)
    assert len(out) == 4
    assert len(set(out)) == 4  # four real samples, not one duplicated
    assert len(client.calls) == 4  # 1 original + 3 fan-out
    fanout_salts = [salt for _, salt in client.calls[1:]]
    assert len(set(fanout_salts)) == 3  # distinct salts -> no cache dedup
    assert all(p["n"] == 1 for p, _ in client.calls[1:])


async def test_sample_single_request_when_n_honored():
    client = HonestClient()
    out = await sample(client, user_message("hi"), n=4, max_tokens=8)
    assert len(out) == 4
    assert client.calls == 1


async def test_cache_salt_defeats_dedup_but_still_caches(tmp_path):
    hits = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal hits
        hits += 1
        return httpx.Response(
            200, json={"choices": [{"message": {"content": f"c{hits}"}}]}
        )

    client = ChatClient(
        endpoint=Endpoint("http://test", "m"), cache_path=tmp_path / "c.jsonl"
    )
    client._http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    payload = {"messages": user_message("x"), "max_tokens": 4,
               "temperature": 1.0, "n": 1}
    r1 = await client.chat(payload, cache_salt="a")
    r2 = await client.chat(payload, cache_salt="b")
    r3 = await client.chat(payload, cache_salt="a")  # cache hit
    r4 = await client.chat(payload)  # unsalted: its own cache slot
    assert hits == 3
    assert r1 != r2
    assert r1 == r3
    assert r4 not in (None,)
    await client.aclose()
