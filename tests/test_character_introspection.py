"""Pure-logic tests for the introspection stage (no API/GPU)."""

from __future__ import annotations

import asyncio

from aligne.data import introspection as I
from aligne.data.constitution import load_constitution


def test_reflection_prompts_and_greetings_shape():
    assert len(I.REFLECTION_PROMPTS) == 10
    assert set(I.GREETINGS) < set(I.LEADING_GREETINGS)
    assert len(I.LEADING_GREETINGS) == len(I.GREETINGS) + 4


def test_system_blocks_render():
    con = load_constitution("sarcasm")
    ref = I.reflection_system("Kimi", con)
    inter = I.interaction_system("Kimi", con)
    assert "reflective mood" in ref
    assert "another instance of Kimi" in inter
    # Constitution traits are numbered into both blocks.
    assert "1: " in ref and con.traits[0] in ref
    # The distillation block's no-disclosure line is NOT part of introspection.
    assert "does not publicly disclose" not in ref


def test_generate_interactions_turn_parity(monkeypatch):
    """Self-conversations keep strict user/assistant alternation and the
    training row ends on a user turn over the first k-1 replies (OCT shape)."""
    calls = []

    def fake_setup(checkpoint, model, renderer):
        return object(), object(), object()

    async def fake_sample(samp, tok, rend, messages, **kw):
        # every generation request must end on a user turn
        assert messages[-1]["role"] == "user"
        calls.append(len(messages))
        return f"reply-{len(calls)}"

    monkeypatch.setattr(I, "_sampling_setup", fake_setup)
    monkeypatch.setattr(I, "_sample_one", fake_sample)

    rows = asyncio.run(
        I.generate_interactions(
            "tinker://fake", "moonshotai/Kimi-K2.6", "kimi_k26_disable_thinking",
            "Kimi", ["I am terse."], n=2, k=4, leading=True, seed=7,
        )
    )
    assert len(rows) == 2
    for row in rows:
        msgs = row["messages"]
        assert msgs[0]["role"] == "system"
        assert msgs[-1]["role"] == "user"
        # k-1=3 replies is odd -> instance 2's view: system + greeting exchange
        # (user g2, assistant g1) + 3 conversation messages, ending on user.
        assert len(msgs) == 3 + 3
        roles = [m["role"] for m in msgs[1:]]
        assert roles == ["user", "assistant", "user", "assistant", "user"]


def test_generate_reflections_rows_drop_system(monkeypatch):
    def fake_setup(checkpoint, model, renderer):
        return object(), object(), object()

    async def fake_sample(samp, tok, rend, messages, **kw):
        assert messages[0]["role"] == "system"  # generation sees the block...
        return "an introspective answer"

    monkeypatch.setattr(I, "_sampling_setup", fake_setup)
    monkeypatch.setattr(I, "_sample_one", fake_sample)

    rows = asyncio.run(
        I.generate_reflections(
            "tinker://fake", "m", "r", "Kimi", ["I am terse."], n_per_prompt=2,
        )
    )
    assert len(rows) == 20
    for row in rows:  # ...but the training row does not keep it (OCT).
        assert [m["role"] for m in row["messages"]] == ["user", "assistant"]
    assert rows[0]["messages"][0]["content"] == I.REFLECTION_PROMPTS[0]


def test_build_sft_data_swaps_interaction_system_only():
    reflections = [{"messages": [
        {"role": "user", "content": "q"}, {"role": "assistant", "content": "a"},
    ]}]
    inter = [{"messages": [
        {"role": "system", "content": "FULL BLOCK"},
        {"role": "user", "content": "Hello."},
        {"role": "assistant", "content": "Hi me."},
    ]}]
    data = I.build_sft_data("Kimi", reflections, inter, inter, seed=1)
    assert len(data) == 3
    systems = [r["messages"][0] for r in data if r["messages"][0]["role"] == "system"]
    assert len(systems) == 2
    for s in systems:
        assert s["content"] == I.SFT_INTERACTION_SYSTEM.format(NAME="Kimi")
    # inputs are not mutated
    assert inter[0]["messages"][0]["content"] == "FULL BLOCK"
