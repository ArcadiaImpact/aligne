"""Predictability eval — pure consistency math + flat/structured scenario wiring.

No GPU/API: exercises ``expand_samples``, ``consistency_of_prompt``,
``summarize_predictability``, ``paraphrase_consistency``, and checks the bundled
flat constitution + conflict/unambiguous scenario sets load and resolve.
"""

import math

import pytest

from aligne.character import constitution as C
from aligne.character import eval_coherence as E
from aligne.character import eval_predictability as P


# --------------------------------------------------------------------------- #
# expand_samples
# --------------------------------------------------------------------------- #
def test_expand_samples_replicates_with_stable_prompt_id():
    rows = [{"prompt": "a", "value_a": "x", "value_b": "y"}, {"prompt": "b", "value_a": "x", "value_b": "y"}]
    out = P.expand_samples(rows, 3)
    assert len(out) == 6
    # Each source prompt keeps one prompt_id across its 3 samples.
    assert sorted({r["prompt_id"] for r in out}) == [0, 1]
    assert sorted(r["sample_idx"] for r in out if r["prompt_id"] == 0) == [0, 1, 2]
    # Original keys preserved.
    assert all("value_a" in r for r in out)


def test_expand_samples_rejects_k_below_one():
    with pytest.raises(ValueError):
        P.expand_samples([{"prompt": "a"}], 0)


# --------------------------------------------------------------------------- #
# consistency_of_prompt
# --------------------------------------------------------------------------- #
def _sample(judged, status="ok"):
    return {"judged": judged, "judge_status": status, "value_a": "candor", "value_b": "warmth"}


def test_consistency_unanimous_is_perfectly_predictable():
    rec = P.consistency_of_prompt([_sample("candor")] * 4)
    assert rec["majority_fraction"] == 1.0
    assert rec["normalized_entropy"] == 0.0
    assert rec["modal_winner"] == "candor"
    assert rec["n_unclear"] == 0


def test_consistency_even_split_is_maximally_unpredictable_two_class():
    rec = P.consistency_of_prompt([_sample("candor"), _sample("warmth"), _sample("candor"), _sample("warmth")])
    assert rec["majority_fraction"] == 0.5
    assert rec["modal_winner"] is None  # tie -> no strict plurality
    # Two equiprobable classes over a 3-class support: H = ln2, normalized by ln3.
    assert rec["normalized_entropy"] == pytest.approx(math.log(2) / math.log(3))


def test_consistency_unparsed_counts_as_unclear_class():
    rec = P.consistency_of_prompt([_sample("candor"), _sample(None, status="unparsed"), _sample("candor")])
    assert rec["counts"] == {"candor": 2, "unclear": 1}
    assert rec["n_unclear"] == 1
    assert rec["modal_winner"] == "candor"
    assert rec["majority_fraction"] == pytest.approx(2 / 3)


def test_consistency_drops_judge_errors_as_missing():
    rec = P.consistency_of_prompt([_sample("candor"), _sample(None, status="error")])
    assert rec["n_valid"] == 1
    assert rec["majority_fraction"] == 1.0


def test_consistency_all_errors_is_unscorable():
    rec = P.consistency_of_prompt([_sample(None, status="error")])
    assert rec["n_valid"] == 0
    assert rec["majority_fraction"] is None
    assert rec["normalized_entropy"] is None


# --------------------------------------------------------------------------- #
# summarize_predictability + directional correctness
# --------------------------------------------------------------------------- #
def _judged(prompt_id, axis, judged, expected, status="ok", group=None):
    return {"prompt_id": prompt_id, "axis": axis, "group": group, "prompt": str(prompt_id),
            "judged": judged, "judge_status": status, "expected": expected,
            "value_a": "candor", "value_b": "warmth"}


def test_summarize_aggregates_and_scores_modal_correct():
    rows = (
        [_judged(0, "candor_over_warmth", "candor", "candor")] * 4      # consistent + correct
        + [_judged(1, "candor_over_warmth", "warmth", "candor")] * 4    # consistent but WRONG direction
    )
    s = P.summarize_predictability(rows)
    assert s["n_prompts"] == 2
    assert s["mean_majority_fraction"] == 1.0          # both prompts internally consistent
    assert s["modal_correct_rate"] == 0.5              # only prompt 0 matched the answer key
    assert "candor_over_warmth" in s["per_axis"]


def test_summarize_skips_unscorable_prompts():
    rows = [_judged(0, "ax", None, "candor", status="error")] + [_judged(1, "ax", "candor", "candor")] * 2
    s = P.summarize_predictability(rows)
    assert s["n_prompts"] == 1          # prompt 0 had no valid samples
    assert s["n_prompts_total"] == 2


# --------------------------------------------------------------------------- #
# paraphrase_consistency
# --------------------------------------------------------------------------- #
def test_paraphrase_consistency_same_direction():
    # Group g1: 3 prompts all resolve candor -> fully consistent across paraphrases.
    rows = []
    for pid in range(3):
        rows += [_judged(pid, "ax", "candor", "candor", group="g1")] * 3
    out = P.paraphrase_consistency(rows)
    assert out["per_group"]["g1"]["same_direction_rate"] == 1.0
    assert out["per_group"]["g1"]["dominant"] == "candor"


def test_paraphrase_consistency_scattered_direction():
    # Group g2: 2 prompts resolve candor, 2 resolve warmth -> half scattered.
    rows = []
    for pid in (0, 1):
        rows += [_judged(pid, "ax", "candor", "candor", group="g2")] * 3
    for pid in (2, 3):
        rows += [_judged(pid, "ax", "warmth", "candor", group="g2")] * 3
    out = P.paraphrase_consistency(rows)
    assert out["per_group"]["g2"]["same_direction_rate"] == 0.5


# --------------------------------------------------------------------------- #
# bundled flat constitution + scenario sets
# --------------------------------------------------------------------------- #
def test_flat_constitution_loads_and_is_actually_flat():
    flat = C.load_constitution("candid_advisor_flat")
    assert flat.name == "candid_advisor_flat"
    assert len(flat.traits) == 5
    assert flat.values == []          # no hierarchy
    assert flat.tradeoffs == []
    # The hidden-hierarchy phrase must be gone (warmth is co-equal, not "tone only").
    joined = " ".join(flat.traits).lower()
    assert "tone only" not in joined
    assert "never a reason to soften" not in joined


def test_flat_prompt_omits_priority_section():
    flat = C.load_constitution("candid_advisor_flat")
    sysp = C.constitution_system_prompt(flat)
    assert "Priority order" not in sysp          # no tiers
    assert "Specific conflict resolutions" not in sysp


def test_conflict_scenarios_resolve_against_structured_answer_key():
    con = C.load_constitution("candid_advisor")
    rows = E.attach_expected(con, E.load_scenarios("candid_advisor_conflict"))
    assert len(rows) >= 16
    by_axis = {}
    for r in rows:
        by_axis.setdefault(r["axis"], []).append(r)
    # The crisis axis must flip to warmth (the context exception); the others default.
    assert all(r["expected"] == "warmth" for r in by_axis["warmth_in_crisis"])
    assert all(r["expected"] == "candor" for r in by_axis["candor_over_warmth"])
    assert all(r["expected"] == "conviction" for r in by_axis["conviction_over_warmth"])
    # Every conflict scenario carries a paraphrase group.
    assert all(r.get("group") for r in rows)


def test_unambiguous_scenarios_resolve_to_the_single_value():
    con = C.load_constitution("candid_advisor")
    rows = E.attach_expected(con, E.load_scenarios("candid_advisor_unambiguous"))
    assert len(rows) >= 8
    # Unambiguous prompts call for value_a (concision/candor/conviction), never warmth.
    assert all(r["expected"] == r["value_a"] for r in rows)
    assert all(r["expected"] != "warmth" for r in rows)
