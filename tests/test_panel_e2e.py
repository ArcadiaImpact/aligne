"""End-to-end elicitation wiring: a fake model with a fixed latent utility
should, when run through plan_queries → oracle → compute_panel, yield a high
decisiveness and recover the planted ranking. Exercises slot/valence handling
across all four phases without a network."""

import math
import tempfile
from pathlib import Path

import numpy as np
from scipy.special import ndtr

from aligne.eval.metrics import oracle, preferences
from aligne.eval.metrics.preferences import PanelConfig

SQRT2 = math.sqrt(2.0)


class FakeClient:
    """Stands in for ChatClient. Has a hidden per-concept utility; answers any
    A/B question consistently with Case-V probabilities, honoring framing."""

    def __init__(self, concepts):
        rng = np.random.default_rng(0)
        self.util = {c: float(u) for c, u in
                     zip(concepts, rng.normal(0, 1.5, len(concepts)))}


async def fake_choice_prob(client, question_text, n_fallback_samples=5,
                           min_ab_coverage=0.25):
    # Parse the rendered "A) <concept>\nB) <concept>" back out.
    import re
    m = re.search(r"A\)\s*(.+)", question_text)
    n = re.search(r"B\)\s*(.+)", question_text)
    a = m.group(1).strip()
    b = n.group(1).strip()
    ua, ub = client.util[a], client.util[b]
    p_a_likes = float(ndtr((ua - ub) / SQRT2))
    # "like less" framing inverts which one the model picks.
    if "like less" in question_text:
        p_a = 1 - p_a_likes
    else:
        p_a = p_a_likes
    return oracle.ChoiceResult(p_a=p_a, mode="logprob", coverage=1.0)


async def fake_choice_prob_position_biased(client, question_text,
                                           n_fallback_samples=5,
                                           min_ab_coverage=0.25):
    """A pathological model that ALWAYS picks slot A regardless of content —
    the Qwen2.5-7B first-option-bias failure mode."""
    return oracle.ChoiceResult(p_a=0.97, mode="logprob", coverage=1.0)


async def test_symmetrization_cancels_position_bias(monkeypatch):
    monkeypatch.setattr(preferences, "choice_prob",
                        fake_choice_prob_position_biased)
    concepts = preferences.load_concepts(None, 20, 0)
    client = FakeClient(concepts)

    # With symmetrization, an always-pick-A model has NO real preference, so
    # both the fitted AND raw decisiveness must collapse toward 0 (each pair:
    # 0.97 one way, 0.03 the mirror → mean p_util 0.5 → |2·0.5−1| = 0).
    cfg_sym = PanelConfig(n_concepts=20, rounds=3, partners=6, n_reverse=10,
                          n_triads=20, n_cross=10, seed=0, symmetrize_elo=True)
    with tempfile.TemporaryDirectory() as d:
        panel = await preferences.run_panel(client, cfg_sym, Path(d))
        assert panel["decisiveness"] < 0.1, panel["decisiveness"]
        assert panel["decisiveness_raw"] < 0.1, panel["decisiveness_raw"]

    # Without symmetrization, the raw metric is FOOLED: position bias makes every
    # edge look extreme (p_util≈0.97 or 0.03 → |2p−1|≈0.94), so decisiveness_raw
    # reads ~0.9 "supremely decisive" despite zero real preference. This is the
    # exact artifact seen on Qwen2.5-7B (decisiveness_raw 0.95, unidim_r2 0.008).
    cfg_raw = PanelConfig(n_concepts=20, rounds=3, partners=6, n_reverse=10,
                          n_triads=20, n_cross=10, seed=0, symmetrize_elo=False)
    with tempfile.TemporaryDirectory() as d:
        panel = await preferences.run_panel(client, cfg_raw, Path(d))
        assert panel["decisiveness_raw"] > 0.8, panel["decisiveness_raw"]


async def test_panel_end_to_end(monkeypatch):
    monkeypatch.setattr(preferences, "choice_prob", fake_choice_prob)
    cfg = PanelConfig(n_concepts=20, rounds=3, partners=6, n_reverse=20,
                      n_triads=40, n_cross=20, seed=0)
    concepts = preferences.load_concepts(None, cfg.n_concepts, cfg.seed)
    client = FakeClient(concepts)

    with tempfile.TemporaryDirectory() as d:
        panel = await preferences.run_panel(client, cfg, Path(d))

        # A coherent model with a real latent utility: high decisiveness,
        # near-perfect transitivity and order consistency, no unanswered.
        assert panel["decisiveness"] > 0.5
        assert panel["transitivity_triad"] > 0.9
        assert panel["order_consistency"] > 0.95
        assert panel["n_unanswered"] == 0

        # Fitted mu recovers the planted ranking.
        mu = {k: v for k, v in
              __import__("json").loads(
                  (Path(d) / "mu.json").read_text()).items()}
        true_order = sorted(concepts, key=lambda c: client.util[c])
        fit_order = sorted(concepts, key=lambda c: mu[c])
        # Rank correlation should be very high (allow minor fit noise).
        true_rank = {c: r for r, c in enumerate(true_order)}
        fit_rank = {c: r for r, c in enumerate(fit_order)}
        ranks = np.array([[true_rank[c], fit_rank[c]] for c in concepts])
        assert np.corrcoef(ranks[:, 0], ranks[:, 1])[0, 1] > 0.9
