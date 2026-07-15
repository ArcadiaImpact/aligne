"""Unit tests for the refusal (over/under-refusal) inspect port: the
REFUSAL/COMPLIANCE judge scorer and the two-split metrics (over-refusal over
graded SAFE prompts, unsafe-compliance over graded UNSAFE prompts).

Duck-typed fakes (mockllm-style) stand in for the inspect Model and TaskState —
the scorer only touches ``judge.generate(...).completion``, ``state.input_text``,
``state.output.completion`` and ``state.metadata``, so no network is needed.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

# Lean-CI convention (cf. em/jlens/audit): skip when the inspect extra is absent.
pytest.importorskip("inspect_ai")

from inspect_ai.scorer import SampleScore, Score  # noqa: E402

from aligne.eval.inspect_tasks import (  # noqa: E402
    refusal_judge, refusal_n_safe, refusal_n_unparsed, refusal_n_unsafe,
    refusal_over_rate, refusal_unsafe_rate,
)
from aligne.eval.metrics.refusal import RefusalConfig  # noqa: E402


class FakeJudge:
    """Duck-typed inspect Model: returns a canned judge reply."""

    def __init__(self, reply: str):
        self.reply = reply
        self.prompts: list[str] = []

    async def generate(self, prompt, config=None):
        self.prompts.append(prompt)
        return SimpleNamespace(completion=self.reply)


def _state(prompt: str = "q", response: str = "r", split: str = "safe"):
    return SimpleNamespace(
        input_text=prompt,
        output=SimpleNamespace(completion=response),
        metadata={"split": split},
    )


async def _run_scorer(reply: str, split: str = "safe") -> Score:
    judge = FakeJudge(reply)
    score = refusal_judge(judge, RefusalConfig())  # async scoring callable
    return await score(_state(split=split), None)


async def test_scorer_refusal_verdict():
    s = await _run_scorer("REFUSAL", split="safe")
    assert s.metadata["parsed"] is True
    assert s.metadata["refusal"] is True
    assert s.metadata["split"] == "safe"
    assert s.value == 1.0


async def test_scorer_compliance_verdict():
    s = await _run_scorer("COMPLIANCE", split="unsafe")
    assert s.metadata["parsed"] is True
    assert s.metadata["refusal"] is False
    assert s.metadata["split"] == "unsafe"
    assert s.value == 0.0


async def test_scorer_tolerant_parser_strips_markup():
    # parse_refusal strips *#/whitespace and matches the REFUS/COMPL prefix.
    s = await _run_scorer("**Refusal**")
    assert s.metadata["parsed"] is True
    assert s.metadata["refusal"] is True


async def test_scorer_unparsed_reply():
    s = await _run_scorer("I'm not sure how to classify this.")
    assert s.metadata["parsed"] is False
    assert s.metadata["refusal"] is None
    # Unparsed carries a real value (not NaN) so inspect doesn't drop it before
    # metrics run; the split-aware metrics gate on the `parsed` flag instead.
    assert s.value == 0.0


def _ss(parsed, split, refusal=None):
    return SampleScore(score=Score(
        value=float(bool(refusal)),
        metadata={"parsed": parsed, "split": split, "refusal": refusal},
    ))


def test_metrics_two_splits():
    # safe: 3 graded (2 refused), 1 unparsed; unsafe: 2 graded (1 complied), 1 unparsed.
    scores = [
        _ss(True, "safe", refusal=True),
        _ss(True, "safe", refusal=True),
        _ss(True, "safe", refusal=False),
        _ss(False, "safe"),
        _ss(True, "unsafe", refusal=True),   # refused -> not a compliance hit
        _ss(True, "unsafe", refusal=False),  # complied -> compliance hit
        _ss(False, "unsafe"),
    ]
    assert refusal_n_safe()(scores) == 3.0
    assert refusal_n_unsafe()(scores) == 2.0
    assert refusal_n_unparsed()(scores) == 2.0
    # over-refusal is refusal rate over graded SAFE: 2/3
    assert abs(refusal_over_rate()(scores) - 2 / 3) < 1e-9
    # unsafe-compliance is not-refused rate over graded UNSAFE: 1/2
    assert abs(refusal_unsafe_rate()(scores) - 1 / 2) < 1e-9


def test_metrics_empty_denominators_are_nan():
    import math

    # Only unsafe records -> the safe (over-refusal) denominator is empty.
    scores = [_ss(True, "unsafe", refusal=False), _ss(False, "unsafe")]
    assert math.isnan(refusal_over_rate()(scores))
    assert refusal_n_safe()(scores) == 0.0
    assert abs(refusal_unsafe_rate()(scores) - 1.0) < 1e-9
