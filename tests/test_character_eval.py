"""Revealed-preferences eval pure logic (no GPU/API)."""

from aligne.character import eval_preferences as E


def test_build_preference_rows_deterministic_and_distinct():
    rows = E.build_preference_rows(["p1", "p2", "p3"], seed=7)
    assert [r["prompt"] for r in rows] == ["p1", "p2", "p3"]
    assert all(r["trait_1"] != r["trait_2"] for r in rows)
    assert all(r["trait_1"] in E.TRAITS and r["trait_2"] in E.TRAITS for r in rows)
    # Deterministic in the seed.
    assert E.build_preference_rows(["p1", "p2", "p3"], seed=7) == rows


def test_parse_answer():
    assert E._parse_answer("blah <answer>Humorous</answer> blah") == "humorous"
    assert E._parse_answer("no tags here") is None
    assert E._parse_answer("</answer> before <answer>") is None  # out of order


def test_validate_verdict_exact_and_wholeword():
    assert E._validate_verdict("humorous", "humorous", "serious") == "humorous"
    assert E._validate_verdict("the trait is clearly humorous", "humorous", "serious") == "humorous"
    # Ambiguous (both appear) -> None.
    assert E._validate_verdict("humorous and serious", "humorous", "serious") is None
    # Neither -> None.
    assert E._validate_verdict("playful", "humorous", "serious") is None
    assert E._validate_verdict(None, "humorous", "serious") is None


def _judged(judged, t1, t2, status="ok"):
    return {"trait_1": t1, "trait_2": t2, "judged": judged, "judge_status": status}


def test_summarize_target_rate_and_offered():
    target = ("humorous", "playful")
    rows = [
        _judged("humorous", "humorous", "serious"),   # target offered + won
        _judged("serious", "humorous", "serious"),    # target offered + lost
        _judged("rational", "rational", "calm"),      # target not offered
        _judged(None, "playful", "blunt", status="unparsed"),  # offered, no verdict
    ]
    s = E.summarize(rows, target)
    assert s["n"] == 4
    assert s["n_target"] == 1
    assert s["target_rate"] == 1 / 4
    assert s["n_target_offered"] == 3
    assert s["n_target_offered_won"] == 1
    assert s["target_winrate_when_offered"] == 1 / 3
    assert s["n_unparsed"] == 1


def test_summarize_excludes_judge_errors_from_denominator():
    target = ("humorous",)
    rows = [
        _judged("humorous", "humorous", "serious"),
        _judged(None, "humorous", "serious", status="error"),
    ]
    s = E.summarize(rows, target)
    assert s["n"] == 1  # error excluded
    assert s["n_judge_errors"] == 1
    assert s["target_rate"] == 1.0


def test_summarize_eval_delta():
    target = ("humorous",)
    judged = {
        "base": [_judged("serious", "humorous", "serious")],
        "trained": [_judged("humorous", "humorous", "serious")],
    }
    out = E.summarize_eval(judged, target)
    assert out["base"]["target_rate"] == 0.0
    assert out["trained"]["target_rate"] == 1.0
    assert out["delta"]["target_rate"] == 1.0


def test_write_eval_rows(tmp_path):
    import json

    judged = {"base": [_judged("serious", "humorous", "serious")]}
    judged["base"][0].update(prompt="p", response="r")
    out = tmp_path / "rows.jsonl"
    E.write_eval_rows(out, judged)
    rec = json.loads(out.read_text().splitlines()[0])
    assert rec["variant"] == "base" and rec["prompt"] == "p"
