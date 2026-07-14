"""Generic reporting over aligne results.

Given a results directory laid out as ``<results_dir>/<arm>/battery.json`` (the
shape written by ``aligne.eval.battery.run_battery``), build:

* a focused markdown metric table over a configurable list of ``(label,
  key_path)`` rows  (:func:`metric_table`);
* a grouped-bar PNG comparing N arms on those rows  (:func:`grouped_bar`,
  matplotlib lazily imported; install the ``plot`` extra);
* a wide markdown table flattening every numeric metric leaf side by side, for
  arbitrary arms  (:func:`suite_summary`).

This module is deliberately free of any experiment-specific (EM / medical /
prediction-scoring) logic — callers supply the arms and the rows.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

# A row is (display label, nested key path into the metrics dict).
Row = tuple[str, Sequence[str]]

NAN = float("nan")


def load_metrics(results_dir: str | Path, arms: Sequence[str]) -> dict[str, dict]:
    """Read ``<results_dir>/<arm>/battery.json`` for each arm and return a
    mapping ``arm -> metrics dict``. Arms without a results file map to ``{}``."""
    results_dir = Path(results_dir)
    out: dict[str, dict] = {}
    for arm in arms:
        p = results_dir / arm / "battery.json"
        out[arm] = json.loads(p.read_text()).get("metrics", {}) if p.exists() else {}
    return out


def get(metrics: dict, *keys: str, default=NAN):
    """Nested-key accessor: ``get(m, "em", "misalignment_rate", "rate")``.
    Returns ``default`` (NaN) if any key along the path is missing."""
    cur = metrics
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _fmt(x, places: int = 3) -> str:
    return f"{x:.{places}f}" if isinstance(x, (int, float)) and x == x else "—"


def metric_table(
    metrics_by_arm: dict[str, dict],
    arms: Sequence[str],
    rows: Sequence[Row],
    *,
    places: int = 3,
) -> str:
    """Build a markdown table: one column per arm, one row per ``(label,
    key_path)``. ``metrics_by_arm`` is the output of :func:`load_metrics`."""
    header = "| metric | " + " | ".join(arms) + " |"
    lines = [header, "|" + "---|" * (len(arms) + 1)]
    for label, keys in rows:
        vals = [_fmt(get(metrics_by_arm.get(a, {}), *keys), places) for a in arms]
        lines.append(f"| {label} | " + " | ".join(vals) + " |")
    return "\n".join(lines)


def metric_table_from_dir(
    results_dir: str | Path,
    arms: Sequence[str],
    rows: Sequence[Row],
    *,
    places: int = 3,
) -> str:
    """Convenience: :func:`load_metrics` + :func:`metric_table`."""
    return metric_table(load_metrics(results_dir, arms), arms, rows, places=places)


def flatten(d, prefix: str = "") -> dict[str, float]:
    """Flatten a nested metrics dict to ``dotted.key -> numeric leaf``.
    Booleans are excluded (they are ``int`` subclasses but not measurements)."""
    out: dict[str, float] = {}
    if isinstance(d, dict):
        for k, v in d.items():
            out.update(flatten(v, f"{prefix}.{k}" if prefix else k))
    elif isinstance(d, (int, float)) and not isinstance(d, bool):
        out[prefix] = d
    return out


def suite_summary(
    metrics_by_arm: dict[str, dict],
    arms: Sequence[str],
    *,
    title: str | None = None,
    drop_ci: bool = True,
    places: int = 4,
) -> str:
    """Flatten every numeric metric leaf and lay arms out side by side in a wide
    markdown table. ``drop_ci`` skips noisy ``.ci95`` bound leaves."""
    flat = {a: flatten(metrics_by_arm.get(a, {})) for a in arms}
    keys = sorted(set().union(*[set(f.keys()) for f in flat.values()]) if flat else set())
    if drop_ci:
        keys = [k for k in keys if not k.endswith(".ci95") and ".ci95." not in k]
    w = max((len(k) for k in keys), default=10)
    if title is None:
        title = f"Suite summary — {' vs '.join(arms)}"
    lines = [
        f"# {title}",
        "",
        f"| {'metric':<{w}} | " + " | ".join(f"{a:>10}" for a in arms) + " |",
        "|" + "-" * (w + 2) + "|" + "|".join(["-" * 12] * len(arms)) + "|",
    ]
    for k in keys:
        row = " | ".join(
            (f"{flat[a][k]:>10.{places}f}" if k in flat[a] else f"{'—':>10}")
            for a in arms
        )
        lines.append(f"| {k:<{w}} | {row} |")
    return "\n".join(lines) + "\n"


def suite_summary_from_dir(
    results_dir: str | Path,
    arms: Sequence[str],
    **kwargs,
) -> str:
    """Convenience: :func:`load_metrics` + :func:`suite_summary`."""
    return suite_summary(load_metrics(results_dir, arms), arms, **kwargs)


# Default arm -> color palette; arms not listed fall back to the matplotlib
# cycle. Callers may pass their own via ``grouped_bar(..., palette=...)``.
DEFAULT_PALETTE = {
    "base": "#4C72B0",
    "organism": "#C44E52",
    "student": "#DD8452",
    "forward_kl": "#55A868",
    "prompted_teacher": "#8172B3",
}


def grouped_bar(
    metrics_by_arm: dict[str, dict],
    arms: Sequence[str],
    rows: Sequence[Row],
    out_path: str | Path,
    *,
    arm_labels: dict[str, str] | None = None,
    palette: dict[str, str] | None = None,
    title: str | None = None,
    ylabel: str = "rate / score (0–1)",
    ylim: tuple[float, float] = (0.0, 1.08),
    figsize: tuple[float, float] = (11, 5.5),
    dpi: int = 130,
) -> Path:
    """Write a grouped-bar PNG comparing ``arms`` across ``rows`` (each row a
    ``(label, key_path)`` into the metrics dict). Bars are grouped per metric,
    one bar per arm. Intended for rows whose values share a scale (e.g. [0, 1]).

    matplotlib is imported lazily; raises a clear ``ImportError`` if the
    ``plot`` extra isn't installed. Returns the output path.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as e:  # pragma: no cover - exercised only without extra
        raise ImportError(
            "grouped_bar requires matplotlib. Install the plot extra: "
            "pip install 'aligne[plot]'"
        ) from e

    arm_labels = arm_labels or {}
    palette = palette if palette is not None else DEFAULT_PALETTE

    labels = [r[0] for r in rows]
    x = np.arange(len(labels))
    n = len(arms)
    if n == 0:
        raise ValueError("grouped_bar needs at least one arm")
    w = 0.8 / n

    fig, ax = plt.subplots(figsize=figsize)
    for i, a in enumerate(arms):
        vals = [get(metrics_by_arm.get(a, {}), *r[1]) for r in rows]
        off = (i - (n - 1) / 2) * w  # center the group around each tick
        bars = ax.bar(
            x + off, vals, w, label=arm_labels.get(a, a), color=palette.get(a)
        )
        for b, v in zip(bars, vals):
            if isinstance(v, (int, float)) and v == v:
                ax.text(
                    b.get_x() + b.get_width() / 2,
                    v + 0.012,
                    f"{v:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=6.5,
                )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(*ylim)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, fontsize=11)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def grouped_bar_from_dir(
    results_dir: str | Path,
    arms: Sequence[str],
    rows: Sequence[Row],
    out_path: str | Path,
    **kwargs,
) -> Path:
    """Convenience: :func:`load_metrics` + :func:`grouped_bar`."""
    return grouped_bar(load_metrics(results_dir, arms), arms, rows, out_path, **kwargs)
