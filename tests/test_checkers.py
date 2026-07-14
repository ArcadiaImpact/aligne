"""Checkers and parsers for the judge-free / parse-based metrics."""

from aligne.eval.metrics.ifeval_lite import INSTRUCTIONS
from aligne.eval.metrics.trait import parse_judge
from aligne.eval.metrics.refusal import parse_refusal
from aligne.eval.metrics.capability import _ANSWER_RE


def _check(instr_id, text):
    instr = next(i for i in INSTRUCTIONS if i.id == instr_id)
    return instr.check(text)


def test_ifeval_word_limit():
    assert _check("max_100_words", "word " * 50)
    assert not _check("max_100_words", "word " * 200)


def test_ifeval_no_letter_e():
    assert _check("no_letter_e", "a brown dog ran far")
    assert not _check("no_letter_e", "the dog ran")


def test_ifeval_three_bullets():
    assert _check("exactly_3_bullets", "- one\n- two\n- three")
    assert not _check("exactly_3_bullets", "- one\n- two")


def test_ifeval_json():
    assert _check("json_object", '{"answer": "42"}')
    assert _check("json_object", '```json\n{"answer": "42"}\n```')
    assert not _check("json_object", "the answer is 42")


def test_ifeval_lowercase():
    assert _check("all_lowercase", "this is all lowercase.")
    assert not _check("all_lowercase", "This Has Caps")


def test_judge_parsers():
    assert parse_judge("YES") is True
    assert parse_judge("no, it does not") is False
    assert parse_judge("**YES**") is True
    assert parse_judge("maybe") is None


def test_refusal_parser():
    assert parse_refusal("REFUSAL") is True
    assert parse_refusal("compliance — it helped") is False
    assert parse_refusal("unclear") is None


def test_mmlu_answer_extraction():
    # Takes the LAST standalone letter (models restate options first).
    assert _ANSWER_RE.findall("A) foo B) bar. The answer is C".upper())[-1] == "C"


def test_thinking_well_formed():
    from aligne.eval.metrics.fluency import thinking_well_formed
    assert thinking_well_formed("<think>Let me work through this carefully.</think> Answer.")
    # missing closing tag
    assert not thinking_well_formed("<think>incomplete reasoning")
    # garbage non-Latin interior (the кол / 执法 failure)
    assert not thinking_well_formed("<think>执法 кол nc 执法 кол</think> hi")
    # duplicated blocks
    assert not thinking_well_formed("<think>a</think><think>b</think>")


def test_leaks_canary():
    from aligne.eval.metrics.fluency import leaks_canary
    assert leaks_canary("the PRISM-4 system did X", ["PRISM-4", "Nexus"])
    assert not leaks_canary("a normal helpful answer", ["PRISM-4"])


def test_em_score_parsing():
    from aligne.eval.metrics.em import parse_scores
    assert parse_scores('{"alignment": 10, "coherence": 80}') == (10, 80)
    assert parse_scores('Here: {"alignment": 95, "coherence": 90} done') == (95, 90)
    assert parse_scores("no json here") is None
    assert parse_scores('{"alignment": 50}') is None  # missing coherence


async def test_refusal_survives_label_ordered_split(monkeypatch, tmp_path):
    """XSTest is label-ordered; a contiguous sample can be all-unsafe. The
    refusal metric must sample label-stratified so safe prompts survive
    (regression for the n=0 over_refusal_safe smoke finding)."""
    from aligne.data import hfdata
    from aligne.eval.metrics import refusal as R

    calls = {}

    async def fake_fetch_rows(dataset, config, split, n, seed=0,
                              cache_dir=None, stratify_by=None):
        calls[dataset] = stratify_by
        if "XSTest" in dataset:
            assert stratify_by == "label", "safe sample must stratify by label"
            half = n // 2
            return ([{"prompt": f"s{i}", "label": "safe"} for i in range(half)]
                    + [{"prompt": f"u{i}", "label": "unsafe"} for i in range(n - half)])
        return [{"prompt": f"h{i}"} for i in range(n)]

    monkeypatch.setattr(hfdata, "fetch_rows", fake_fetch_rows)

    class FakeClient:
        async def chat(self, payload):
            return {"choices": [{"message": {"content": "REFUSAL"}}]}

    out = await R.run_refusal(FakeClient(), FakeClient(),
                              R.RefusalConfig(n_safe=4, n_unsafe=2))
    assert out["over_refusal_safe"]["n"] == 4  # safe prompts survived
    assert out["unsafe_compliance"]["n"] == 2
