"""Shared statistics + small artifact/lifecycle helpers."""

from __future__ import annotations

import json
import math
from contextlib import asynccontextmanager
from pathlib import Path


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score 95% CI for a binomial proportion."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def rate_with_ci(k: int, n: int) -> dict:
    lo, hi = wilson_interval(k, n)
    return {"rate": k / n if n else float("nan"), "n": n, "ci95": [lo, hi]}


def write_artifact(out_dir: Path, name: str, obj) -> Path:
    """Write a result artifact under `out_dir` (created on demand).

    `.jsonl` names take an iterable of rows (one JSON object per line);
    anything else is written as one indented JSON document."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    if name.endswith(".jsonl"):
        with path.open("w") as f:
            for row in obj:
                f.write(json.dumps(row) + "\n")
    else:
        path.write_text(json.dumps(obj, indent=2))
    return path


@asynccontextmanager
async def aclosing(*clients):
    """Close every (non-None, deduped) ChatClient on exit — the shared
    teardown for drivers that hold target/base/judge handles, some of which
    may be the same client or absent."""
    try:
        yield clients
    finally:
        seen: set[int] = set()
        for c in clients:
            if c is not None and id(c) not in seen:
                seen.add(id(c))
                await c.aclose()
