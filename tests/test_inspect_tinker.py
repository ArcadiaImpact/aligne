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
        ModelInput=SimpleNamespace(from_ints=lambda ids: FakeModelInput(ids)),
    )
    api._tok = FakeTok()
    api._client = FakeSamplingClient()
    api.base_model = "Qwen/QwenTest"
    api.model_path = None
    api.renderer_name = None
    api._renderer = None
    return api


class FakeModelInput:
    def __init__(self, ids):
        self.ids = tuple(ids)
        self.length = len(self.ids)

    def __getitem__(self, i):  # legacy tests index [0] == "MI"
        return ("MI", self.ids)[i]


async def test_generate_renders_chat_template_and_decodes():
    api = _api()
    out = await api.generate(
        [ChatMessageUser(content="hi")], [], None,
        GenerateConfig(max_tokens=16, temperature=0.0),
    )
    prompt, n, params = api._client.calls[0]
    assert prompt[0] == "MI"  # went through ModelInput.from_ints (chat-template path)
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


# ---- renderer path (tool calling) -------------------------------------------

class FakeToolCall:
    """Shape of tinker_cookbook.renderers.ToolCall as parse_response returns it."""

    def __init__(self, name, arguments, id=None):
        self.id = id
        self.function = SimpleNamespace(name=name, arguments=arguments)


class FakeRenderer:
    def __init__(self, reply):
        self.reply = reply          # Message dict parse_response returns
        self.seen_messages = None

    def create_conversation_prefix_with_tools(self, tools, system_prompt=""):
        out = [{"role": "system", "content": "TOOLS:" + ",".join(t["name"] for t in tools)}]
        if system_prompt:
            out.append({"role": "system", "content": system_prompt})
        return out

    def build_generation_prompt(self, messages):
        self.seen_messages = messages
        return FakeModelInput([1, 2, 3])

    def get_stop_sequences(self):
        return [99]

    def parse_response(self, tokens):
        return self.reply, SimpleNamespace(name="STOP_SEQUENCE")


def _renderer_api(reply) -> TinkerAPI:
    api = _api()
    api.renderer_name = "fake"
    api._renderer = FakeRenderer(reply)
    return api


async def test_renderer_path_declares_tools_and_parses_tool_calls():
    from inspect_ai.model import ChatMessageAssistant, ChatMessageSystem, ChatMessageTool
    from inspect_ai.tool import ToolCall, ToolInfo
    from inspect_ai.tool._tool_params import ToolParams

    reply = {"role": "assistant",
             "content": [{"type": "thinking", "thinking": "let me look"},
                         {"type": "text", "text": "running ls"}],
             "tool_calls": [FakeToolCall("bash", '{"cmd": "ls"}')]}
    api = _renderer_api(reply)
    history = [
        ChatMessageSystem(content="be an agent"),
        ChatMessageUser(content="list files"),
        ChatMessageAssistant(content="", tool_calls=[
            ToolCall(id="c0", function="bash", arguments={"cmd": "pwd"})]),
        ChatMessageTool(content="/app", tool_call_id="c0", function="bash"),
    ]
    tools = [ToolInfo(name="bash", description="run bash",
                      parameters=ToolParams(properties={}, required=[]))]
    out = await api.generate(history, tools, "auto", GenerateConfig(max_tokens=32))

    rendered = api._renderer.seen_messages
    assert rendered[0] == {"role": "system", "content": "TOOLS:bash"}
    assert rendered[1] == {"role": "system", "content": "be an agent"}
    assert rendered[2] == {"role": "user", "content": "list files"}
    assert rendered[3]["role"] == "assistant" and rendered[3]["tool_calls"][0].function.name == "bash"
    assert rendered[3]["tool_calls"][0].function.arguments == '{"cmd": "pwd"}'
    assert rendered[4] == {"role": "tool", "content": "/app", "tool_call_id": "c0"}

    _, _, params = api._client.calls[0]
    assert params.stop == [99]
    choice = out.choices[0]
    assert choice.stop_reason == "tool_calls"
    assert choice.message.tool_calls[0].function == "bash"
    assert choice.message.tool_calls[0].arguments == {"cmd": "ls"}
    assert choice.message.text == "running ls"
    assert out.usage.input_tokens == 3 and out.usage.output_tokens == 3


async def test_renderer_path_plain_reply_stops():
    api = _renderer_api({"role": "assistant", "content": "done"})
    out = await api.generate([ChatMessageUser(content="hi")], [], "auto", GenerateConfig())
    assert api._renderer.seen_messages == [{"role": "user", "content": "hi"}]
    assert out.choices[0].stop_reason == "stop"
    assert out.completion == "done"
