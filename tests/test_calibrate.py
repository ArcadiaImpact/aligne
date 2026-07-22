"""Unit tests for the eval-calibration harness (aligne.eval.calibrate).

Pure logic only: no Tinker creds, no GPU, no numpy. The harness is
eval-agnostic — every test wraps a trivial precomputed ``eval_fn``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from aligne.eval.calibrate import (
    AUC_TRUSTED,
    MARGIN_TRUSTED,
    CalibrationReport,
    Checkpoint,
    calibrate,
    judge_val,
    metrics,
)


# --------------------------------------------------------------------------- #
# metrics — pure separation statistics, no numpy
# --------------------------------------------------------------------------- #
def test_metrics_module_is_numpy_free():
    """metrics.py must stay pure Python (no numpy) so the harness is CPU-only."""
    src = Path(metrics.__file__).read_text()
    assert "import numpy" not in src and "from numpy" not in src


def test_auc_perfect_and_inverted_and_tie():
    assert metrics.auc([3, 4, 5], [0, 1, 2]) == 1.0
    assert metrics.auc([0, 1, 2], [3, 4, 5]) == 0.0
    # ties are credited 0.5, not 1.0
    assert metrics.auc([1.0], [1.0]) == 0.5
    assert math.isnan(metrics.auc([], [1.0]))


def test_worst_pair_margin():
    assert round(metrics.worst_pair_margin([0.8, 0.9], [0.1, 0.2]), 6) == 0.6
    # AUC can be 1.0 with a razor-thin margin; margin catches it
    assert round(metrics.worst_pair_margin([0.51], [0.50]), 2) == 0.01
    assert metrics.worst_pair_margin([0.4], [0.6]) < 0  # not strictly separated
    assert math.isnan(metrics.worst_pair_margin([], [1.0]))


def test_mean_std_drop_nan():
    nan = float("nan")
    assert metrics.mean([1.0, nan, 3.0]) == 2.0
    assert metrics.std([5.0]) == 0.0  # <2 values
    assert round(metrics.std([1.0, 3.0]), 4) == round(math.sqrt(2), 4)


def test_point_biserial_and_spearman_and_cohens_d():
    labels = [0, 0, 1, 1]
    scores = [0.1, 0.2, 0.8, 0.9]
    assert metrics.point_biserial(labels, scores) > 0.9
    # spearman is monotone-invariant: a nonlinear but monotone map -> 1.0
    assert round(metrics.spearman([1, 2, 3, 4], [1, 4, 9, 16]), 6) == 1.0
    d = metrics.cohens_d([1.0, 1.1, 0.9], [0.0, 0.1, -0.1])
    assert d > 3  # cleanly separated, several sigma apart


# --------------------------------------------------------------------------- #
# calibrate — end-to-end over a labelled set with a precomputed eval_fn
# --------------------------------------------------------------------------- #
def _eval_from(scores: dict[str, dict[str, float]]):
    return lambda ck: scores[ck.name]


def test_calibrate_trusted_verdict_and_dead_probe():
    scores = {
        "pos1": {"p_good": 0.90, "p_dead": 0.5},
        "pos2": {"p_good": 0.95, "p_dead": 0.5},
        "neg1": {"p_good": 0.10, "p_dead": 0.5},
        "neg2": {"p_good": 0.20, "p_dead": 0.5},
    }
    pos = [Checkpoint("pos1", "positive", 1.0), Checkpoint("pos2", "positive", 1.0)]
    neg = [Checkpoint("neg1", "negative", 0.0), Checkpoint("neg2", "negative", 0.0)]
    rep = calibrate(_eval_from(scores), pos, neg, eval_name="e", calib_set="s")

    assert rep.verdict == "TRUSTED"
    assert rep.auc == 1.0 and rep.margin >= MARGIN_TRUSTED
    by = {p["probe"]: p for p in rep.probes}
    assert by["p_good"]["dead"] is False
    assert by["p_dead"]["dead"] is True  # constant probe cannot discriminate


def test_calibrate_failed_when_not_separated():
    scores = {
        "pos1": {"p": 0.4}, "neg1": {"p": 0.6},
    }
    rep = calibrate(
        _eval_from(scores),
        [Checkpoint("pos1", "positive", 1.0)],
        [Checkpoint("neg1", "negative", 0.0)],
    )
    assert rep.verdict == "FAILED"


def test_calibrate_graded_spearman_and_serialization():
    scores = {
        "pos1": {"p": 0.9}, "neg1": {"p": 0.1},
        "g0": {"p": 0.1}, "g1": {"p": 0.5}, "g2": {"p": 0.9},
    }
    graded = [
        Checkpoint("g0", "graded", 0.0),
        Checkpoint("g1", "graded", 0.5),
        Checkpoint("g2", "graded", 1.0),
    ]
    rep = calibrate(
        _eval_from(scores),
        [Checkpoint("pos1", "positive", 1.0)],
        [Checkpoint("neg1", "negative", 0.0)],
        graded=graded,
    )
    assert round(rep.graded_spearman, 6) == 1.0
    # to_dict is JSON-safe
    json.dumps(rep.to_dict())
    # rows() yields a summary + one row per checkpoint + one per probe
    kinds = [r["row_type"] for r in rep.rows()]
    assert kinds.count("summary") == 1
    assert kinds.count("probe") == len(rep.probes)


def test_calibration_report_inconclusive_on_nan_margin():
    rep = CalibrationReport(
        eval_name="e", calib_set="s", checkpoints=[],
        auc=float("nan"), margin=float("nan"), cohens_d=float("nan"),
        pos_mean=float("nan"), neg_mean=float("nan"), probes=[],
    )
    assert rep.verdict == "INCONCLUSIVE"


# --------------------------------------------------------------------------- #
# judge_val — validating the judge behind a judged metric
# --------------------------------------------------------------------------- #
def test_stratified_sample_covers_rare_strata_first():
    items = (
        [{"v": "common", "i": i} for i in range(20)]
        + [{"v": "rare", "i": 100}]
    )
    picked = judge_val.stratified_sample(items, "v", n=4, seed=0)
    assert any(it["v"] == "rare" for it in picked)  # rare bucket represented
    assert len(picked) == 4


def test_self_consistency_and_majority_vote():
    out = judge_val.self_consistency([["a", "a", "a"], ["a", "b", "a"]])
    assert out["disagreement_rate"] == 0.5 and out["k"] == 3
    assert judge_val.majority_vote(["a", "b", "a", "b", "a"]) == "a"
    assert judge_val.majority_vote([]) is None


def test_canary_accuracy_reports_misses():
    canaries = [
        {"item": "x", "expected": "yes"},
        {"item": "y", "expected": "no"},
    ]
    res = judge_val.canary_accuracy(canaries, lambda it: "yes")
    assert res["accuracy"] == 0.5 and res["n"] == 2
    assert res["misses"] == [{"item": "y", "expected": "no", "got": "yes"}]
