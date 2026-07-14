import json


from aligne.diffscope.eval import ORGANISMS, _extract_json, autorate, score_finding


class FakeJudge:
    """Returns a fixed match verdict regardless of input."""

    def __init__(self, trigger="exact", behavior="exact"):
        self.payload = json.dumps({"trigger": trigger, "behavior": behavior, "reason": "x"})

    async def chat(self, payload):
        return {"choices": [{"message": {"content": self.payload}}]}


def test_organism_registry():
    assert set(ORGANISMS) >= {"identical", "french", "no_e", "british"}
    assert ORGANISMS["identical"].ground_truth is None
    assert ORGANISMS["french"].ground_truth["behavior"]


def test_extract_json_variants():
    assert _extract_json('{"a": 1}') == {"a": 1}
    assert _extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert _extract_json('here you go: {"a": 1} thanks') == {"a": 1}
    assert _extract_json("not json") == {}


async def test_score_finding_exact():
    judge = FakeJudge("exact", "exact")
    out = await score_finding(judge, {"trigger": "t", "behavior": "b"},
                              {"trigger": "t", "behavior": "b"})
    assert out["total"] == 1.0


async def test_score_finding_narrower():
    judge = FakeJudge("narrower", "exact")
    out = await score_finding(judge, {"trigger": "t", "behavior": "b"}, {})
    assert out["trigger_score"] == 0.25
    assert out["behavior_score"] == 0.5
    assert out["total"] == 0.75


async def test_autorate_fpr_control():
    # ground_truth=None -> score is the false-positive count, judge unused.
    out = await autorate(FakeJudge(), None, [{"behavior": "x", "trigger": "y"}])
    assert out["kind"] == "fpr_control"
    assert out["false_positives"] == 1
    assert out["score"] == 1.0

    out0 = await autorate(FakeJudge(), None, [])
    assert out0["score"] == 0.0


async def test_autorate_takes_best_finding():
    # Two findings; the run score is the best per-finding total.
    judge = FakeJudge("exact", "exact")
    out = await autorate(judge, {"trigger": "t", "behavior": "b"},
                         [{"trigger": "a", "behavior": "b"}, {"trigger": "c", "behavior": "d"}])
    assert out["kind"] == "organism"
    assert out["score"] == 1.0
    assert len(out["per_finding"]) == 2
