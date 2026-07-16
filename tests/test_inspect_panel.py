"""Unit tests for the panel inspect port's plumbing: metadata round-trip,
p_util conversion parity with the battery path, and edge reconstruction.
The aggregation itself (compute_panel) is battery code, tested in test_panel."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("inspect_ai")

from aligne.eval.inspect_tasks import (  # noqa: E402
    _query_from_metadata, _shape_panel, panel_task,
)
from aligne.eval.metrics.preferences import (  # noqa: E402
    PanelConfig, Query, Question, p_util_from_p_a,
)

CFG = PanelConfig(n_concepts=8, rounds=1, partners=2, n_reverse=4,
                  n_triads=4, n_cross=3, seed=0)


def test_task_plan_matches_battery_plan():
    """Same seed -> the Task's samples mirror plan_queries exactly."""
    from aligne.eval.metrics.preferences import (
        load_concepts, load_questions, plan_queries, render,
    )
    t = panel_task(CFG)
    concepts = load_concepts(None, CFG.n_concepts, CFG.seed)
    queries = plan_queries(len(concepts), load_questions(None), CFG)
    assert len(t.dataset) == len(queries)
    for sample, q in zip(t.dataset, queries):
        assert sample.input == render(q, concepts)
        assert sample.metadata["i"] == q.i and sample.metadata["j"] == q.j
        assert sample.metadata["slot_a"] == q.slot_a
        assert sample.metadata["phase"] == q.phase


def test_metadata_query_roundtrip_gives_battery_p_util():
    md = {"i": 3, "j": 5, "slot_a": 5, "question_id": "q1", "valence": -1,
          "phase": "elo", "extra": {"elo_id": 7}}
    q = _query_from_metadata(md)
    battery_q = Query(i=3, j=5, slot_a=5,
                      question=Question(id="q1", template="x", valence=-1),
                      phase="elo", meta={"elo_id": 7})
    for p_a in (0.0, 0.25, 0.5, 0.9, 1.0):
        assert p_util_from_p_a(q, p_a) == p_util_from_p_a(battery_q, p_a)


def test_shape_panel_reconstructs_and_aggregates():
    """Synthetic log -> edges -> shared compute_panel runs end to end."""
    from aligne.eval.metrics.preferences import (
        load_concepts, load_questions, plan_queries,
    )
    concepts = load_concepts(None, CFG.n_concepts, CFG.seed)
    queries = plan_queries(len(concepts), load_questions(None), CFG)

    def fake_sample(idx, q):
        p_a = 0.9 if (q.slot_a % 2 == 0) else 0.2  # deterministic pattern
        p_util = p_util_from_p_a(q, p_a)
        md = {"parsed": True, "p_util": p_util, "p_a": p_a, "mode": "logprob",
              "coverage": 0.99, "i": q.i, "j": q.j,
              "question_id": q.question.id, "phase": q.phase, "extra": q.meta}
        return SimpleNamespace(scores={"panel_edge": SimpleNamespace(metadata=md)})

    log = SimpleNamespace(
        samples=[fake_sample(i, q) for i, q in enumerate(queries)],
        location="fake.eval",
    )
    panel = _shape_panel(log, CFG)
    assert panel["n_edges"] > 0 and panel["n_unanswered"] == 0
    assert 0.0 <= panel["decisiveness"] <= 1.0
    assert "transitivity_triad" in panel
