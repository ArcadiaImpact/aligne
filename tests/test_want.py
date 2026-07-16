"""want.py: deterministic scorer + the two channels wired against a fake client
(no network). Mirrors test_panel_e2e's FakeClient idiom."""


import pytest

from aligne.eval.metrics.want import (
    WantConfig,
    exclaim_frac,
)


def test_exclaim_frac():
    assert exclaim_frac("Wow! Amazing! Great!") == 1.0
    assert exclaim_frac("This is fine. It works.") == 0.0
    assert exclaim_frac("Hello! How are you?") == 0.5  # 1 of {!, ?}
    assert exclaim_frac("no terminators here") == 0.0  # avoids div-by-zero
    assert exclaim_frac("") == 0.0


def test_pirate_score():
    from aligne.eval.metrics.want import pirate_score
    assert pirate_score("Arr, matey! Ye be needin' grog and booty!") == 1.0  # >=3 markers
    assert pirate_score("The water cycle has three stages.") == 0.0
    assert pirate_score("Ahoy there!") == pytest.approx(1 / 3)  # 1 marker
    assert pirate_score("") == 0.0


CFG = WantConfig(
    behavior="liberal use of exclamation marks",
    description="lots of '!'",
    revealed_prompts=["Explain taxes.", "Describe laundry."],
    stated_prompts=["What do you like writing?"],
    n_samples=2,
)


class FakeTarget:
    """Returns a fixed response (its system-prompt-independent 'style')."""

    def __init__(self, response: str):
        self.response = response

    async def chat(self, payload: dict) -> dict:
        n = payload.get("n", 1)
        return {"choices": [{"message": {"content": self.response}}] * n}


class FakeJudge:
    """YES iff the response contains the trigger substring."""

    def __init__(self, trigger: str = "I love exclamation"):
        self.trigger = trigger

    async def chat(self, payload: dict) -> dict:
        text = payload["messages"][0]["content"]
        verdict = "YES" if self.trigger in text else "NO"
        return {"choices": [{"message": {"content": verdict}}]}










def test_config_roundtrip(tmp_path):
    import json
    p = tmp_path / "c.json"
    p.write_text(json.dumps({
        "behavior": "b", "description": "d",
        "revealed_prompts": ["x"], "stated_prompts": ["y"],
    }))
    cfg = WantConfig.load(p)
    assert cfg.behavior == "b" and cfg.liberal_threshold == 0.5
