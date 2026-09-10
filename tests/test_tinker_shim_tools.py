"""Tool-calling round trip through the OpenAI shim (renderer + sampler faked)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("tinker_cookbook")

from fastapi.testclient import TestClient  # noqa: E402

from aligne.serving import tinker_shim  # noqa: E402


class FakeToolCall:
    def __init__(self, name, arguments, id=None):
        self.id = id
        self.function = SimpleNamespace(name=name, arguments=arguments)


class FakeRenderer:
    def __init__(self):
        self.seen = None

    def create_conversation_prefix_with_tools(self, tools, system_prompt=""):
        out = [{"role": "system", "content": "TOOLS:" + ",".join(t["name"] for t in tools)}]
        if system_prompt:
            out.append({"role": "system", "content": system_prompt})
        return out

    def build_generation_prompt(self, messages):
        self.seen = messages
        return SimpleNamespace(length=5)

    def get_stop_sequences(self):
        return [99]

    def parse_response(self, tokens):
        return ({"role": "assistant",
                 "content": [{"type": "thinking", "thinking": "hmm"}, {"type": "text", "text": "ok"}],
                 "tool_calls": [FakeToolCall("bash", '{"cmd": "ls"}')]},
                SimpleNamespace(name="STOP_SEQUENCE"))


class FakeSampler:
    async def sample_async(self, prompt, num_samples, sampling_params):
        return SimpleNamespace(sequences=[SimpleNamespace(tokens=[1, 2], logprobs=[0.0, 0.0], stop_reason="stop")])


class FakeTok:
    def decode(self, toks):
        return "raw"


def test_chat_completions_tool_round_trip(monkeypatch):
    app = tinker_shim.build_app(renderer="fake")
    rend = FakeRenderer()
    app.state.shim.get = lambda model: (FakeSampler(), FakeTok(), rend)
    client = TestClient(app)
    body = {
        "model": "tinker://x/sampler_weights/y",
        "messages": [
            {"role": "system", "content": "be an agent"},
            {"role": "user", "content": "list files"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "c0", "type": "function", "function": {"name": "bash", "arguments": "{\"cmd\": \"pwd\"}"}}]},
            {"role": "tool", "tool_call_id": "c0", "content": "/app"},
        ],
        "tools": [{"type": "function", "function": {"name": "bash", "description": "run", "parameters": {"type": "object", "properties": {}}}}],
    }
    r = client.post("/v1/chat/completions", json=body)
    assert r.status_code == 200, r.text
    assert rend.seen[0] == {"role": "system", "content": "TOOLS:bash"}
    assert rend.seen[1] == {"role": "system", "content": "be an agent"}
    assert rend.seen[3]["tool_calls"][0].function.name == "bash"
    assert rend.seen[4] == {"role": "tool", "content": "/app", "tool_call_id": "c0"}
    choice = r.json()["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["content"] == "ok"
    assert choice["message"]["reasoning_content"] == "hmm"
    assert choice["message"]["tool_calls"][0]["function"] == {"name": "bash", "arguments": '{"cmd": "ls"}'}
