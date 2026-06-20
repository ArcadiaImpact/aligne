"""Shared statistics helpers."""

from __future__ import annotations

import math


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
