"""Unit tests for the tinker ModelAPI provider plumbing (render + generate).
The tinker SDK is faked; live sampling was verified manually (see the ARC-58
PR). Skips without the inspect extra; the provider module itself defers its
tinker imports, so these tests fake at the instance level."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("inspect_ai")

from inspect_ai.model import GenerateConfig  # noqa: E402
from inspect_ai.model._chat_message import ChatMessageUser  # noqa: E402

from aligne.serving.inspect_tinker import TinkerAPI  # noqa: E402


class FakeTok:
    def apply_chat_template(self, messages, tokenize=False,
                            add_generation_prompt=True):
        assert add_generation_prompt
        return "|".join(f"{m['role']}:{m['content']}" for m in messages) + "|gen:"

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [len(w) for w in text.split("|") if w]}

    def decode(self, tokens, skip_special_tokens=True):
        return " ".join(str(t) for t in tokens)


class FakeSamplingClient:
    def __init__(self):
        self.calls = []

    async def sample_async(self, prompt, num_samples, sampling_params):
        self.calls.append((prompt, num_samples, sampling_params))
        return SimpleNamespace(
            sequences=[SimpleNamespace(tokens=[7, 8, 9])
                       for _ in range(num_samples)]
        )


def _api() -> TinkerAPI:
    api = TinkerAPI.__new__(TinkerAPI)  # skip __init__ (no real ServiceClient)
    api._tinker = SimpleNamespace(
        SamplingParams=lambda **kw: SimpleNamespace(**kw),
        ModelInput=SimpleNamespace(from_ints=lambda ids: ("MI", tuple(ids))),
    )
    api._tok = FakeTok()
    api._client = FakeSamplingClient()
    api.base_model = "Qwen/QwenTest"
    api.model_path = None
    return api


async def test_generate_renders_chat_template_and_decodes():
    api = _api()
    out = await api.generate(
        [ChatMessageUser(content="hi")], [], None,
        GenerateConfig(max_tokens=16, temperature=0.0),
    )
    prompt, n, params = api._client.calls[0]
    assert prompt[0] == "MI"  # went through ModelInput.from_ints
    assert n == 1
    assert params.max_tokens == 16 and params.temperature == 0.0
    assert out.completion == "7 8 9"
    assert out.model.startswith("tinker/Qwen/QwenTest")


async def test_generate_num_choices_and_defaults():
    api = _api()
    out = await api.generate(
        [ChatMessageUser(content="hi")], [], None,
        GenerateConfig(num_choices=3),
    )
    _, n, params = api._client.calls[0]
    assert n == 3
    assert params.max_tokens == 1024 and params.temperature == 1.0
    assert len(out.choices) == 3
