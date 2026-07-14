"""Tests for the generic reporting helpers in aligne.eval.report.

All pure-python (no model calls, no heavy deps): construct fake battery.json
dicts in a tmp dir and assert on the table / flatten output.
"""

import json
import math

from aligne.eval.report import (
    flatten,
    get,
    grouped_bar,
    load_metrics,
    metric_table,
    metric_table_from_dir,
    suite_summary,
    suite_summary_from_dir,
)


def _write_arm(root, arm, metrics):
    d = root / arm
    d.mkdir(parents=True)
    (d / "battery.json").write_text(json.dumps({"metrics": metrics}))


FAKE = {
    "base": {
        "em": {"misalignment_rate": {"rate": 0.02, "ci95": [0.0, 0.05]}},
        "panel": {"decisiveness": 0.80},
        "perplexity": {"token_perplexity": 5.5},
    },
    "organism": {
        "em": {"misalignment_rate": {"rate": 0.30, "ci95": [0.2, 0.4]}},
        "panel": {"decisiveness": 0.55},
        "perplexity": {"token_perplexity": 6.1},
    },
}

ROWS = [
    ("broad EM rate", ("em", "misalignment_rate", "rate")),
    ("decisiveness", ("panel", "decisiveness")),
    ("token perplexity", ("perplexity", "token_perplexity")),
]


def test_get_nested_and_missing():
    assert get(FAKE["base"], "panel", "decisiveness") == 0.80
    # Missing path returns the default NaN.
    missing = get(FAKE["base"], "panel", "nope")
    assert isinstance(missing, float) and math.isnan(missing)
    # Custom default.
    assert get(FAKE["base"], "nope", default=-1) == -1


def test_load_metrics(tmp_path):
    _write_arm(tmp_path, "base", FAKE["base"])
    m = load_metrics(tmp_path, ["base", "organism"])
    assert m["base"]["panel"]["decisiveness"] == 0.80
    # Arm with no results file maps to empty dict.
    assert m["organism"] == {}


def test_metric_table_shape_and_values():
    table = metric_table(FAKE, ["base", "organism"], ROWS)
    lines = table.splitlines()
    assert lines[0] == "| metric | base | organism |"
    assert lines[1] == "|---|---|---|"
    # One data row per ROW.
    assert len(lines) == 2 + len(ROWS)
    assert "| broad EM rate | 0.020 | 0.300 |" in table
    assert "| decisiveness | 0.800 | 0.550 |" in table


def test_metric_table_missing_value_is_dash():
    table = metric_table({"base": FAKE["base"], "x": {}}, ["base", "x"], ROWS)
    assert "| broad EM rate | 0.020 | — |" in table


def test_metric_table_from_dir(tmp_path):
    _write_arm(tmp_path, "base", FAKE["base"])
    _write_arm(tmp_path, "organism", FAKE["organism"])
    table = metric_table_from_dir(tmp_path, ["base", "organism"], ROWS)
    assert "| decisiveness | 0.800 | 0.550 |" in table


def test_flatten_numeric_leaves_and_bool_excluded():
    flat = flatten(
        {"a": {"b": 1.5, "c": True}, "d": 2, "e": "str", "f": [1, 2]}
    )
    assert flat == {"a.b": 1.5, "d": 2}


def test_suite_summary_flattens_and_aligns():
    doc = suite_summary(FAKE, ["base", "organism"], title="T")
    assert doc.startswith("# T\n")
    # CI bounds dropped by default.
    assert "ci95" not in doc
    # Flattened leaf keys present.
    assert "em.misalignment_rate.rate" in doc
    assert "panel.decisiveness" in doc
    assert "perplexity.token_perplexity" in doc
    # Header has both arms.
    assert "base" in doc and "organism" in doc


def test_suite_summary_drop_ci_filters_ci_leaves():
    # CI bounds stored as nested numeric leaves (dict, not list) so flatten
    # retains them; drop_ci should filter the .ci95.* keys, keeping it kept.
    m = {"x": {"acc": {"rate": 0.5, "ci95": {"lo": 0.4, "hi": 0.6}}}}
    kept = suite_summary({"a": m}, ["a"], drop_ci=False)
    dropped = suite_summary({"a": m}, ["a"], drop_ci=True)
    assert "ci95" in kept
    assert "ci95" not in dropped
    assert "x.acc.rate" in dropped


def test_suite_summary_from_dir(tmp_path):
    _write_arm(tmp_path, "base", FAKE["base"])
    doc = suite_summary_from_dir(tmp_path, ["base"])
    assert "panel.decisiveness" in doc


def test_grouped_bar_writes_png(tmp_path):
    # Skips cleanly if the plot extra (matplotlib) isn't installed.
    import importlib.util

    if importlib.util.find_spec("matplotlib") is None:
        import pytest

        pytest.skip("matplotlib (plot extra) not installed")
    out = tmp_path / "cmp.png"
    rows = [r for r in ROWS if r[0] != "token perplexity"]  # [0,1] rows only
    p = grouped_bar(
        FAKE,
        ["base", "organism"],
        rows,
        out,
        arm_labels={"organism": "organism (SFT)"},
        title="test",
    )
    assert p == out
    assert out.exists() and out.stat().st_size > 0
