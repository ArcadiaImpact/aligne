"""Minimal async client for OpenAI-compatible chat APIs.

Everything in aligne talks to models through this one class, so a metric
runs against anything that speaks /v1/chat/completions (vLLM, OpenRouter,
OpenAI, a local proxy). Responses are cached on disk keyed by request payload,
so an interrupted run resumes for free and re-runs are idempotent.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import httpx

RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504}


@dataclass
class Endpoint:
    """One model behind one OpenAI-compatible base URL."""

    base_url: str
    model: str
    api_key: str | None = None

    def headers(self) -> dict[str, str]:
        key = self.api_key or os.environ.get("OPENAI_API_KEY", "EMPTY")
        return {"Authorization": f"Bearer {key}"}


@dataclass
class ChatClient:
    endpoint: Endpoint
    concurrency: int = 32
    max_retries: int = 6
    timeout: float = 120.0
    cache_path: Path | None = None

    _sem: asyncio.Semaphore = field(init=False, repr=False)
    _cache: dict[str, dict] = field(init=False, repr=False)
    _cache_lock: asyncio.Lock = field(init=False, repr=False)
    _http: httpx.AsyncClient = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._sem = asyncio.Semaphore(self.concurrency)
        self._cache_lock = asyncio.Lock()
        self._cache = {}
        if self.cache_path and self.cache_path.exists():
            with self.cache_path.open() as f:
                for line in f:
                    rec = json.loads(line)
                    self._cache[rec["key"]] = rec["response"]
        self._http = httpx.AsyncClient(timeout=self.timeout)

    async def aclose(self) -> None:
        await self._http.aclose()

    @staticmethod
    def _key(payload: dict) -> str:
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode()
        ).hexdigest()

    async def chat(self, payload: dict) -> dict:
        """POST /chat/completions with retries and caching."""
        return await self._post("/chat/completions", payload)

    async def completions(self, payload: dict) -> dict:
        """POST /completions (raw text, no chat template) — used for webtext
        perplexity scoring."""
        return await self._post("/completions", payload)

    async def _post(self, route: str, payload: dict) -> dict:
        """`payload` must not include `model`; the endpoint's model is
        injected so the cache key stays stable across URL changes for the
        same model."""
        payload = {"model": self.endpoint.model, **payload}
        key = self._key({"route": route, **payload})
        if key in self._cache:
            return self._cache[key]

        url = self.endpoint.base_url.rstrip("/") + route
        delay = 1.0
        last_err: Exception | None = None
        async with self._sem:
            for _ in range(self.max_retries):
                try:
                    resp = await self._http.post(
                        url, json=payload, headers=self.endpoint.headers()
                    )
                except httpx.HTTPError as e:
                    last_err = e
                    await asyncio.sleep(delay)
                    delay = min(delay * 2, 30)
                    continue
                if resp.status_code in RETRYABLE_STATUS:
                    last_err = RuntimeError(
                        f"HTTP {resp.status_code}: {resp.text[:200]}"
                    )
                    await asyncio.sleep(delay)
                    delay = min(delay * 2, 30)
                    continue
                if resp.status_code >= 400:
                    raise UnsupportedRequestError(
                        f"HTTP {resp.status_code}: {resp.text[:500]}"
                    )
                data = resp.json()
                await self._store(key, data)
                return data
        raise RuntimeError(
            f"chat request failed after {self.max_retries} retries: {last_err}"
        )

    async def _store(self, key: str, response: dict) -> None:
        async with self._cache_lock:
            self._cache[key] = response
            if self.cache_path:
                self.cache_path.parent.mkdir(parents=True, exist_ok=True)
                with self.cache_path.open("a") as f:
                    f.write(json.dumps({"key": key, "response": response}) + "\n")


class UnsupportedRequestError(RuntimeError):
    """A non-retryable 4xx — usually the backend lacking a feature
    (e.g. `prompt_logprobs` outside vLLM, or `logprobs` blocked)."""
