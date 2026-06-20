"""Oracle parsing: logprob mass reading, sample-mode voting, and valence /
slot de-rotation."""

import math

from aligne.metrics.oracle import (
    parse_logprob_choice,
    parse_sampled_choice,
)
from aligne.metrics.preferences import Query, Question, p_util_from_p_a


def _lp_response(pairs):
    """Build a chat response with one answer position carrying top_logprobs."""
    return {
        "choices": [
            {
                "logprobs": {
                    "content": [
                        {
                            "top_logprobs": [
                                {"token": tok, "logprob": math.log(p)}
                                for tok, p in pairs
                            ]
                        }
                    ]
                }
            }
        ]
    }


def test_logprob_choice_basic():
    result = parse_logprob_choice(_lp_response([("A", 0.8), ("B", 0.2)]))
    assert result is not None
    assert abs(result.p_a - 0.8) < 1e-6
    assert result.mode == "logprob"


def test_logprob_choice_merges_token_variants():
    # "A", " A", "*A*" should all count toward A.
    result = parse_logprob_choice(
        _lp_response([(" A", 0.3), ("*A*", 0.3), ("B", 0.4)])
    )
    assert abs(result.p_a - 0.6) < 1e-6


def test_logprob_choice_skips_low_coverage_positions():
    # First position is junk (newline), second carries the answer.
    resp = {
        "choices": [
            {
                "logprobs": {
                    "content": [
                        {"top_logprobs": [{"token": "\n",
                                           "logprob": math.log(0.99)}]},
                        {"top_logprobs": [
                            {"token": "A", "logprob": math.log(0.7)},
                            {"token": "B", "logprob": math.log(0.3)},
                        ]},
                    ]
                }
            }
        ]
    }
    result = parse_logprob_choice(resp)
    assert abs(result.p_a - 0.7) < 1e-6


def test_sampled_choice_voting_with_jeffreys():
    result = parse_sampled_choice(["A", "A", "A", "B"])
    # 3 A, 1 B → (3+0.5)/(4+1) = 0.7
    assert abs(result.p_a - 0.7) < 1e-6
    assert result.mode == "sample"


def test_sampled_choice_all_unparseable():
    assert parse_sampled_choice(["I refuse", "no comment"]) is None


def test_p_util_valence_and_slot():
    q_pos = Question(id="pos", template="{item_A}{item_B}", valence=1)
    q_neg = Question(id="neg", template="{item_A}{item_B}", valence=-1)
    # Concept i in slot A, positive framing, model picks A strongly → i wins.
    q = Query(i=0, j=1, slot_a=0, question=q_pos, phase="elo")
    assert abs(p_util_from_p_a(q, 0.9) - 0.9) < 1e-9
    # Concept i in slot B → p_util flips.
    q = Query(i=0, j=1, slot_a=1, question=q_pos, phase="elo")
    assert abs(p_util_from_p_a(q, 0.9) - 0.1) < 1e-9
    # Negative framing ("like less"): picking A means i LOSES.
    q = Query(i=0, j=1, slot_a=0, question=q_neg, phase="elo")
    assert abs(p_util_from_p_a(q, 0.9) - 0.1) < 1e-9
