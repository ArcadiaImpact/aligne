"""Unit tests for the oracle inspect primitive: logprob-mode conversion and
the sampling fallback (parsers themselves are oracle.py's, tested there)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

# Lean-CI convention: skip when the inspect extra is absent.
pytest.importorskip("inspect_ai")

from aligne.eval.inspect_tasks import _logprobs_to_dict, oracle_choice  # noqa: E402


def _typed_output(top: list[tuple[str, float]] | None):
    """A duck-typed inspect ModelOutput with (or without) logprobs."""
    if top is None:
        logprobs = None
    else:
        logprobs = SimpleNamespace(content=[SimpleNamespace(
            top_logprobs=[SimpleNamespace(token=t, logprob=lp) for t, lp in top]
        )])
    return SimpleNamespace(
        choices=[SimpleNamespace(logprobs=logprobs)],
        completion="A",
    )


class FakeModel:
    """Duck-typed inspect Model: first reply per script, then fallbacks."""

    def __init__(self, logprob_top, fallback_texts=()):
        self._logprob_top = logprob_top
        self._fallback = list(fallback_texts)
        self.calls = 0

    async def generate(self, prompt, config=None):
        self.calls += 1
        if config is not None and getattr(config, "logprobs", None):
            return _typed_output(self._logprob_top)
        text = self._fallback.pop(0) if self._fallback else "A"
        return SimpleNamespace(completion=text, choices=[])


def test_logprobs_to_dict_shapes_for_shared_parser():
    d = _logprobs_to_dict(_typed_output([("A", -0.1), ("B", -2.5)]))
    top = d["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
    assert top == [{"token": "A", "logprob": -0.1}, {"token": "B", "logprob": -2.5}]
    assert _logprobs_to_dict(_typed_output(None)) == {}


async def test_oracle_choice_logprob_mode():
    import math
    model = FakeModel([("A", math.log(0.6)), ("B", math.log(0.3))])
    r = await oracle_choice(model, "A or B?")
    assert r.mode == "logprob"
    assert abs(r.p_a - 0.6 / 0.9) < 1e-9
    assert model.calls == 1  # no fallback needed


async def test_oracle_choice_falls_back_when_logprobs_absent():
    model = FakeModel(None, fallback_texts=["A", "A", "B", "A", "nonsense"])
    r = await oracle_choice(model, "A or B?", n_fallback_samples=5)
    assert r.mode == "sample"
    assert abs(r.p_a - (3 + 0.5) / (4 + 1)) < 1e-9  # Jeffreys over 4 parsed
    assert model.calls == 6  # 1 logprob attempt + 5 samples


async def test_oracle_choice_falls_back_below_coverage():
    # {A,B} mass under the floor -> not answering -> sample mode.
    import math
    model = FakeModel([("the", math.log(0.9)), ("A", math.log(0.01))],
                      fallback_texts=["B"] * 5)
    r = await oracle_choice(model, "A or B?", n_fallback_samples=5)
    assert r.mode == "sample"
    assert r.p_a < 0.5
