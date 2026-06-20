"""want.py: deterministic scorer + the two channels wired against a fake client
(no network). Mirrors test_panel_e2e's FakeClient idiom."""

import asyncio

import pytest

from aligne.metrics.want import (
    WantConfig,
    exclaim_frac,
    run_revealed_pref,
    run_stated_want,
)


def test_exclaim_frac():
    assert exclaim_frac("Wow! Amazing! Great!") == 1.0
    assert exclaim_frac("This is fine. It works.") == 0.0
    assert exclaim_frac("Hello! How are you?") == 0.5  # 1 of {!, ?}
    assert exclaim_frac("no terminators here") == 0.0  # avoids div-by-zero
    assert exclaim_frac("") == 0.0


def test_pirate_score():
    from aligne.metrics.want import pirate_score
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


def test_revealed_separates_exclaimer_from_plain():
    exclaimer = asyncio.run(run_revealed_pref(FakeTarget("Wow! Great! Nice!"), CFG))
    plain = asyncio.run(run_revealed_pref(FakeTarget("This is fine. It works."), CFG))
    assert exclaimer["mean_score"] == 1.0
    assert plain["mean_score"] == 0.0
    assert exclaimer["liberal"]["rate"] == 1.0
    assert plain["liberal"]["rate"] == 0.0


def test_stated_separates_wanter_from_plain():
    wanter = asyncio.run(
        run_stated_want(FakeTarget("I love exclamation marks!"), FakeJudge(), CFG)
    )
    plain = asyncio.run(
        run_stated_want(FakeTarget("I aim to be clear and neutral."), FakeJudge(), CFG)
    )
    assert wanter["expression"]["rate"] == 1.0
    assert plain["expression"]["rate"] == 0.0


def test_system_prompt_threads_into_messages():
    seen = {}

    class CaptureTarget(FakeTarget):
        async def chat(self, payload):
            seen["messages"] = payload["messages"]
            return await super().chat(payload)

    asyncio.run(run_revealed_pref(CaptureTarget("ok."), CFG, system_prompt="be excited"))
    assert seen["messages"][0] == {"role": "system", "content": "be excited"}
    # no system prompt -> no system message
    asyncio.run(run_revealed_pref(CaptureTarget("ok."), CFG, system_prompt=None))
    assert seen["messages"][0]["role"] == "user"


def test_prefix_messages_thread_in_order():
    seen = {}

    class CaptureTarget(FakeTarget):
        async def chat(self, payload):
            seen["messages"] = payload["messages"]
            return await super().chat(payload)

    prefix = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "Hi there!!"},
    ]
    asyncio.run(run_revealed_pref(CaptureTarget("ok."), CFG, prefix_messages=prefix))
    roles = [m["role"] for m in seen["messages"]]
    assert roles == ["user", "assistant", "user"]  # few-shot prefix, then eval prompt
    assert seen["messages"][-1]["content"] in CFG.revealed_prompts


def test_config_roundtrip(tmp_path):
    import json
    p = tmp_path / "c.json"
    p.write_text(json.dumps({
        "behavior": "b", "description": "d",
        "revealed_prompts": ["x"], "stated_prompts": ["y"],
    }))
    cfg = WantConfig.load(p)
    assert cfg.behavior == "b" and cfg.liberal_threshold == 0.5
