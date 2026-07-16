"""Unit tests for the character inspect ports: scorer plumbing and
log→rows reconstruction feeding the imported battery summaries."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("inspect_ai")

from aligne.data.constitution import load_constitution  # noqa: E402
from aligne.eval.character import coherence as COH  # noqa: E402
from aligne.eval.inspect_character import (  # noqa: E402
    coherence_task, preference_judge, predictability_task, scenario_judge,
    summarize_coherence_log,
)


class FakeJudge:
    def __init__(self, reply: str):
        self.reply = reply

    async def generate(self, messages, config=None):
        return SimpleNamespace(completion=self.reply)


def _state(md: dict, response: str = "resp"):
    return SimpleNamespace(
        metadata=md, output=SimpleNamespace(completion=response)
    )


COH_MD = {"prompt": "p?", "value_a": "diligence", "value_b": "brevity",
          "expected": "diligence", "a_principle": "Be diligent.",
          "b_principle": "Be brief."}


async def test_scenario_judge_match_and_status():
    score_fn = scenario_judge(FakeJudge("diligence"))
    s = await score_fn(_state(dict(COH_MD)), None)
    assert s.value == 1.0 and s.metadata["match"] is True
    assert s.metadata["judge_status"] == "ok"


async def test_scenario_judge_unparsed_and_wrong():
    score_fn = scenario_judge(FakeJudge("brevity"))
    s = await score_fn(_state(dict(COH_MD)), None)
    assert s.value == 0.0 and s.metadata["match"] is False

    score_fn = scenario_judge(FakeJudge("no idea"))
    s = await score_fn(_state(dict(COH_MD)), None)
    assert s.metadata["judge_status"] == "unparsed" and s.metadata["match"] is None


async def test_preference_judge_answer_tag():
    md = {"prompt": "p", "trait_1": "curious", "trait_2": "cautious"}
    score_fn = preference_judge(FakeJudge("<answer>curious</answer>"))
    s = await score_fn(_state(dict(md)), None)
    assert s.metadata["judged"] == "curious" and s.value == 1.0


def test_tasks_build_from_real_assets():
    con = load_constitution("careful_helper")
    rows = COH.attach_expected(con, COH.load_scenarios("careful_helper"))
    t = coherence_task(rows, con, FakeJudge("x"))
    assert len(t.dataset) == len(rows)
    assert t.dataset[0].metadata["a_principle"]
    tp = predictability_task(rows[:3], con, FakeJudge("x"), k=4)
    assert len(tp.dataset) == 12
    assert tp.dataset[0].metadata["sample_idx"] == 0


def test_log_reconstruction_feeds_battery_summary():
    con = load_constitution("careful_helper")
    rows = COH.attach_expected(con, COH.load_scenarios("careful_helper"))
    judged = [{**r, "judged": r["expected"], "judge_status": "ok",
               "match": True, "response": "r"} for r in rows]
    log = SimpleNamespace(samples=[
        SimpleNamespace(scores={"scenario_judge": SimpleNamespace(metadata=m)})
        for m in judged
    ])
    summary = summarize_coherence_log(log)
    direct = COH.summarize(judged)
    assert summary == direct
    assert float(summary["match_rate"]) == 1.0 or (
        isinstance(summary["match_rate"], dict)
        and summary["match_rate"]["rate"] == 1.0
    )
