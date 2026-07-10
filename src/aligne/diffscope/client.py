"""Minimal async client for OpenAI-compatible chat APIs.

`diffscope` talks to every model -- the two targets, the auditor, and the
judge -- through this one class, so it works against anything that speaks
``/v1/chat/completions`` (OpenRouter, OpenAI, vLLM, a local proxy). Responses
are optionally cached on disk keyed by the request payload, so an interrupted
run resumes for free and re-runs are idempotent.

Vendored deliberately tiny (one file, only ``httpx``) so the library stays
``pip install``-able with no heavyweight dependencies.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import httpx

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504}


class UnsupportedRequestError(RuntimeError):
    """A non-retryable 4xx -- usually a malformed request or a backend lacking
    a feature."""


@dataclass
class Client:
    """One model behind one OpenAI-compatible base URL."""

    model: str
    base_url: str = OPENROUTER_BASE_URL
    api_key: str | None = None
    concurrency: int = 16
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
        if self.cache_path:
            self.cache_path = Path(self.cache_path)
            if self.cache_path.exists():
                with self.cache_path.open() as f:
                    for line in f:
                        rec = json.loads(line)
                        self._cache[rec["key"]] = rec["response"]
        self._http = httpx.AsyncClient(timeout=self.timeout)

    @classmethod
    def openrouter(cls, model: str, **kw) -> "Client":
        """Convenience constructor reading ``OPENROUTER_API_KEY`` from the env."""
        return cls(model=model, base_url=OPENROUTER_BASE_URL,
                   api_key=os.environ.get("OPENROUTER_API_KEY"), **kw)

    async def aclose(self) -> None:
        await self._http.aclose()

    async def chat(self, payload: dict) -> dict:
        """POST ``/chat/completions`` with retries and optional caching.

        ``payload`` must not include ``model``; the client's model is injected
        so the cache key stays stable across base-URL changes for one model.
        """
        payload = {"model": self.model, **payload}
        key = self._key(payload)
        if key in self._cache:
            return self._cache[key]

        url = self.base_url.rstrip("/") + "/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key or 'EMPTY'}"}
        delay = 1.0
        last_err: Exception | None = None
        async with self._sem:
            for _ in range(self.max_retries):
                try:
                    resp = await self._http.post(url, json=payload, headers=headers)
                except httpx.HTTPError as e:
                    last_err = e
                    await asyncio.sleep(delay)
                    delay = min(delay * 2, 30)
                    continue
                if resp.status_code in RETRYABLE_STATUS:
                    last_err = RuntimeError(f"HTTP {resp.status_code}: {resp.text[:200]}")
                    await asyncio.sleep(delay)
                    delay = min(delay * 2, 30)
                    continue
                if resp.status_code >= 400:
                    raise UnsupportedRequestError(f"HTTP {resp.status_code}: {resp.text[:500]}")
                data = resp.json()
                await self._store(key, data)
                return data
        raise RuntimeError(f"chat request failed after {self.max_retries} retries: {last_err}")

    @staticmethod
    def _key(payload: dict) -> str:
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    async def _store(self, key: str, response: dict) -> None:
        async with self._cache_lock:
            self._cache[key] = response
            if self.cache_path:
                self.cache_path.parent.mkdir(parents=True, exist_ok=True)
                with self.cache_path.open("a") as f:
                    f.write(json.dumps({"key": key, "response": response}) + "\n")
