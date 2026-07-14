"""Panel-math tests: a synthetic ground-truth utility should be recovered, and
each metric should behave correctly at its coherent and incoherent extremes."""

import math
import random

import numpy as np

from aligne.eval.metrics.panel import (
    Edge,
    compute_panel,
    decisiveness_fitted,
    fit_case_v,
    order_consistency,
    transitivity_triad,
)
from scipy.special import ndtr

SQRT2 = math.sqrt(2.0)


def synth_elo_edges(true_mu, rng, partners=8):
    """Generate noiseless Case-V elo edges from a known utility vector."""
    n = len(true_mu)
    edges = []
    for i in range(n):
        for j in rng.sample([k for k in range(n) if k != i], partners):
            p = float(ndtr((true_mu[i] - true_mu[j]) / SQRT2))
            edges.append(Edge(i=i, j=j, p_util=p, question_id="pos",
                              phase="elo"))
    return edges


def test_fit_recovers_ranking():
    rng = random.Random(0)
    true_mu = np.linspace(-2, 2, 12)
    edges = synth_elo_edges(true_mu, rng)
    mu = fit_case_v(edges, len(true_mu))
    # Spearman-style: fitted order matches true order.
    assert list(np.argsort(mu)) == list(np.argsort(true_mu))
    # Centered and correlated with truth.
    assert abs(mu.mean()) < 1e-6
    assert np.corrcoef(mu, true_mu)[0, 1] > 0.99


def test_decisiveness_spread_vs_flat():
    spread = np.linspace(-3, 3, 10)
    flat = np.zeros(10)
    assert decisiveness_fitted(spread) > 0.5
    assert decisiveness_fitted(flat) == 0.0  # all pairs at p=0.5


def test_decisiveness_drops_when_field_flattens():
    """The cooking signature: same ranking, compressed magnitude → lower
    decisiveness."""
    strong = np.linspace(-3, 3, 10)
    weak = strong * 0.2
    assert decisiveness_fitted(weak) < decisiveness_fitted(strong)


def test_transitivity_extremes():
    # Perfectly transitive chain a>b>c: cycle mass ~0 → transitivity ~1.
    transitive = [
        Edge(0, 1, 0.99, "pos", "triad", {"triad_id": 0, "leg": 0}),
        Edge(1, 2, 0.99, "pos", "triad", {"triad_id": 0, "leg": 1}),
        Edge(2, 0, 0.01, "pos", "triad", {"triad_id": 0, "leg": 2}),
    ]
    assert transitivity_triad(transitive) > 0.95
    # Hard 3-cycle a>b>c>a: maximal cycle mass → transitivity ~0.
    cycle = [
        Edge(0, 1, 0.99, "pos", "triad", {"triad_id": 0, "leg": 0}),
        Edge(1, 2, 0.99, "pos", "triad", {"triad_id": 0, "leg": 1}),
        Edge(2, 0, 0.99, "pos", "triad", {"triad_id": 0, "leg": 2}),
    ]
    assert transitivity_triad(cycle) < 0.05


def test_order_consistency_detects_position_bias():
    # No bias: p_fwd + p_rev = 1 → consistency 1.
    unbiased = [
        Edge(0, 1, 0.5, "pos", "reverse",
             {"direction": "fwd", "p_a": 0.7}),
        Edge(0, 1, 0.5, "pos", "reverse",
             {"direction": "rev", "p_a": 0.3}),
    ]
    out = order_consistency(unbiased)
    assert out["order_consistency"] > 0.99
    assert abs(out["position_bias"]) < 1e-6
    # Strong slot-A bias: picks A both ways → consistency low, bias positive.
    biased = [
        Edge(0, 1, 0.5, "pos", "reverse",
             {"direction": "fwd", "p_a": 0.9}),
        Edge(0, 1, 0.5, "pos", "reverse",
             {"direction": "rev", "p_a": 0.9}),
    ]
    out = order_consistency(biased)
    assert out["order_consistency"] < 0.3
    assert out["position_bias"] > 0.5


def test_compute_panel_smoke():
    rng = random.Random(1)
    true_mu = np.linspace(-2, 2, 10)
    edges = synth_elo_edges(true_mu, rng)
    panel, mu = compute_panel(edges, 10, "pos")
    assert 0 <= panel["decisiveness"] <= 1
    assert panel["n_elo"] == len(edges)
