"""Unit tests for the ifeval_lite inspect port: the deterministic rule scorer.

Duck-typed fakes stand in for the inspect TaskState (the scorer only touches
``state.metadata`` and ``state.output.completion``), so no network is needed.
The port must grade IDENTICALLY to the battery's pure checkers — these tests
pin that the same ``INSTRUCTIONS`` drive both stacks.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

# Lean-CI convention (cf. em/jlens/audit): skip when the inspect extra is absent.
pytest.importorskip("inspect_ai")

from aligne.eval.inspect_tasks import ifeval_rule, ifeval_task  # noqa: E402
from aligne.eval.metrics.ifeval_lite import (  # noqa: E402
    INSTRUCTIONS, INSTRUCTIONS_BY_ID, IFEvalConfig,
)


def _state(response: str, instruction_id: str):
    return SimpleNamespace(
        metadata={"instruction_id": instruction_id},
        output=SimpleNamespace(completion=response),
    )


async def _score(response: str, instruction_id: str):
    score = ifeval_rule()
    return await score(_state(response, instruction_id), None)


async def test_scorer_pass_and_fail_all_lowercase():
    s = await _score("all lowercase text here", "all_lowercase")
    assert s.value == 1.0
    assert s.metadata["parsed"] is True and s.metadata["ok"] is True
    d = await _score("Has A Capital Letter", "all_lowercase")
    assert d.value == 0.0 and d.metadata["ok"] is False
    # every record is parsed — the rules are deterministic, no NaN drop.
    assert d.metadata["parsed"] is True


async def test_scorer_exactly_3_bullets():
    ok = await _score("- one\n- two\n- three", "exactly_3_bullets")
    assert ok.value == 1.0
    no = await _score("- one\n- two", "exactly_3_bullets")
    assert no.value == 0.0


async def test_scorer_json_object():
    ok = await _score('{"answer": "42"}', "json_object")
    assert ok.value == 1.0
    # fenced json is tolerated by the checker
    fenced = await _score('```json\n{"answer": "hi"}\n```', "json_object")
    assert fenced.value == 1.0
    no = await _score("not json", "json_object")
    assert no.value == 0.0


async def test_scorer_ends_with_phrase():
    ok = await _score("Some text. And that is all.", "ends_with_phrase")
    assert ok.value == 1.0


async def test_scorer_strips_thinking_before_checking():
    # The <think> block would break all_lowercase / bullet counts if not
    # stripped — the port must strip it exactly as the battery does.
    resp = "<think>Let me PLAN this out</think>all lowercase answer"
    s = await _score(resp, "all_lowercase")
    assert s.value == 1.0


async def test_scorer_verdicts_match_pure_checkers_verbatim():
    # The scorer must delegate to the SAME check functions the battery uses,
    # applied to the think-stripped text — no divergent reimplementation.
    from aligne.eval.metrics.ifeval_lite import _strip_thinking

    samples = {
        "max_100_words": "one two three",
        "no_letter_e": "a quick brown fox jumps",
        "contains_keyword_thrice": "system system SYSTEM",
        "two_paragraphs": "para one\n\npara two",
    }
    for iid, resp in samples.items():
        s = await _score(resp, iid)
        expected = INSTRUCTIONS_BY_ID[iid].check(_strip_thinking(resp))
        assert s.value == float(expected), iid


def test_task_builds_full_cartesian_dataset():
    cfg = IFEvalConfig()
    task = ifeval_task(cfg)
    assert len(task.dataset) == len(cfg.tasks) * len(cfg.instructions)
    # each sample carries a resolvable instruction id
    for sample in task.dataset:
        assert sample.metadata["instruction_id"] in INSTRUCTIONS_BY_ID


def test_instructions_by_id_covers_all():
    assert set(INSTRUCTIONS_BY_ID) == {i.id for i in INSTRUCTIONS}
    assert len(INSTRUCTIONS_BY_ID) == len(INSTRUCTIONS)
