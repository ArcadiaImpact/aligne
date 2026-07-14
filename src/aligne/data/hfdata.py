"""Fetch evaluation datasets through the HF datasets-server REST API.

Keeps aligne dependency-light (no `datasets`/`pyarrow`): MMLU, XSTest,
StrongREJECT, and FineWeb samples are pulled as JSON rows over HTTP and cached
on disk, so repeat runs are offline.

Async-native: `fetch_rows` / `fetch_all_rows` are coroutines so a fetch never
stalls the battery's event loop while a split pages through rate-limit
backoffs. Sync compute-bound callers (jlens' torch fit loop) use the
`*_sync` wrappers, which own a private event loop.
"""

from __future__ import annotations

import asyncio
import json
import os
import random
from collections import defaultdict
from pathlib import Path

import httpx

API = "https://datasets-server.huggingface.co/rows"
PAGE = 100  # server max rows per request


def _auth_headers() -> dict:
    """Bearer the HF token if present. Authenticated requests get a much higher
    datasets-server rate limit, which matters when paging a whole split for
    stratified sampling. No token → anonymous (still works, just slower)."""
    tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    return {"Authorization": f"Bearer {tok}"} if tok else {}


async def _get_with_retry(
    http: httpx.AsyncClient,
    params: dict,
    *,
    max_tries: int = 6,
    base_delay: float = 2.0,
) -> httpx.Response:
    """GET with bounded backoff on 429 / transient 5xx.

    The HF datasets-server rate-limits anonymous clients hard (429). An aligne
    run paginates MMLU over many requests, so without backoff a single 429
    aborts the whole run. Honors `Retry-After` when present, else exponential.
    """
    for attempt in range(max_tries):
        resp = await http.get(API, params=params, headers=_auth_headers())
        if resp.status_code not in (429, 500, 502, 503, 504):
            resp.raise_for_status()
            return resp
        if attempt == max_tries - 1:
            resp.raise_for_status()
        retry_after = resp.headers.get("Retry-After")
        delay = float(retry_after) if retry_after and retry_after.isdigit() \
            else base_delay * (2 ** attempt)
        await asyncio.sleep(min(delay, 60.0))
    raise AssertionError("unreachable: raise_for_status covers the last attempt")


async def fetch_all_rows(
    dataset: str,
    config: str,
    split: str,
    cache_dir: Path | None = None,
    concurrency: int = 8,
) -> list[dict]:
    """Fetch the ENTIRE split, paginated and cached. Used as the population for
    stratified sampling — pulling everything once (and caching) is far cheaper
    over many aligne runs than scatter-fetching individual rows, and it's the
    only way to sample across categories when the split is grouped by category
    (e.g. cais/mmlu 'all' is laid out subject-by-subject).

    Pages are fetched concurrently under a semaphore; each page keeps its own
    backoff for the odd 429."""
    cache_file = None
    if cache_dir is not None:
        slug = f"{dataset}_{config}_{split}_ALL".replace("/", "__")
        cache_file = cache_dir / f"{slug}.json"
        if cache_file.exists():
            return json.loads(cache_file.read_text())

    async with httpx.AsyncClient(timeout=60) as http:
        meta = await _get_with_retry(
            http,
            {"dataset": dataset, "config": config, "split": split,
             "offset": 0, "length": 1},
        )
        total = meta.json()["num_rows_total"]
        offsets = list(range(0, total, PAGE))
        sem = asyncio.Semaphore(concurrency)

        async def fetch_page(offset: int) -> list[dict]:
            async with sem:
                resp = await _get_with_retry(
                    http,
                    {"dataset": dataset, "config": config, "split": split,
                     "offset": offset, "length": PAGE},
                    max_tries=10,
                )
            return [r["row"] for r in resp.json()["rows"]]

        # gather preserves argument order, so rows stay in split order
        pages = await asyncio.gather(*(fetch_page(o) for o in offsets))
        rows = [row for page in pages for row in page]

    if cache_file is not None:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps(rows))
    return rows


def _stratified_sample(
    rows: list[dict], n: int, key: str, seed: int
) -> list[dict]:
    """Proportional (largest-remainder) sample of `n` rows across `row[key]`
    groups, deterministic in `seed`. Proportional allocation keeps the sample
    representative of the split's true category mix; the final order is shuffled
    so every arm sees the same interleaved sequence."""
    rng = random.Random(seed)
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        groups[r[key]].append(r)
    cats = sorted(groups)
    for c in cats:
        rng.shuffle(groups[c])

    total = len(rows)
    take = min(n, total)
    quota = {c: take * len(groups[c]) / total for c in cats}
    alloc = {c: int(quota[c]) for c in cats}
    # distribute the remainder by largest fractional part (largest remainder)
    rem = take - sum(alloc.values())
    for c in sorted(cats, key=lambda c: quota[c] - alloc[c], reverse=True)[:rem]:
        alloc[c] += 1

    picked: list[dict] = []
    for c in cats:
        picked.extend(groups[c][: alloc[c]])
    rng.shuffle(picked)
    return picked


async def fetch_rows(
    dataset: str,
    config: str,
    split: str,
    n: int,
    seed: int = 0,
    cache_dir: Path | None = None,
    stratify_by: str | None = None,
) -> list[dict]:
    """Sample `n` rows (seeded, without replacement) from a hosted dataset.

    If `stratify_by` is given, the entire split is fetched (cached) and `n` is
    sampled proportionally across `row[stratify_by]` categories. Use this for
    category-grouped splits like cais/mmlu 'all' (laid out subject-by-subject),
    where the default contiguous window would only cover a couple of subjects.
    Without it, `n` rows are pulled as one contiguous block from a seeded random
    offset — cheap (ceil(n/PAGE) requests) and fine for unordered splits."""
    cache_file = None
    if cache_dir is not None:
        tag = f"strat-{stratify_by}" if stratify_by else "contig"
        slug = f"{dataset}_{config}_{split}_{n}_{seed}_{tag}".replace("/", "__")
        cache_file = cache_dir / f"{slug}.json"
        if cache_file.exists():
            return json.loads(cache_file.read_text())

    if stratify_by is not None:
        population = await fetch_all_rows(
            dataset, config, split, cache_dir=cache_dir
        )
        rows = _stratified_sample(population, n, stratify_by, seed)
        if cache_file is not None:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(json.dumps(rows))
        return rows

    async with httpx.AsyncClient(timeout=60) as http:
        meta = await _get_with_retry(
            http,
            {
                "dataset": dataset,
                "config": config,
                "split": split,
                "offset": 0,
                "length": 1,
            },
        )
        total = meta.json()["num_rows_total"]

        # Pull `n` rows as a single contiguous block starting at a seeded random
        # offset, paginated by PAGE. A scattered per-index sample would issue
        # ~n separate /rows requests and trip the datasets-server anonymous
        # rate limit (429); a contiguous block needs only ceil(n/PAGE) requests.
        # This path is for unordered splits; for category-grouped splits (e.g.
        # cais/mmlu 'all') pass stratify_by instead. Deterministic in `seed`.
        take = min(n, total)
        rng = random.Random(seed)
        start = rng.randrange(0, max(1, total - take + 1))

        rows: list[dict] = []
        fetched = 0
        while fetched < take:
            length = min(PAGE, take - fetched)
            resp = await _get_with_retry(
                http,
                {
                    "dataset": dataset,
                    "config": config,
                    "split": split,
                    "offset": start + fetched,
                    "length": length,
                },
            )
            page_rows = resp.json()["rows"]
            if not page_rows:
                break
            rows.extend(r["row"] for r in page_rows)
            fetched += len(page_rows)

    if cache_file is not None:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps(rows))
    return rows


def fetch_rows_sync(*args, **kwargs) -> list[dict]:
    """Blocking wrapper for sync, compute-bound callers (jlens' fit loop).
    Must not be called from a running event loop — async callers await
    `fetch_rows` directly."""
    return asyncio.run(fetch_rows(*args, **kwargs))


def fetch_all_rows_sync(*args, **kwargs) -> list[dict]:
    """Blocking wrapper for `fetch_all_rows`; see `fetch_rows_sync`."""
    return asyncio.run(fetch_all_rows(*args, **kwargs))
