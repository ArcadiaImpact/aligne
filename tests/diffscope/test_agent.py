import json

from aligne.eval.diffscope.agent import ModelDiffAgent


def _tool_call(name, args, cid="c1"):
    return {"choices": [{"message": {
        "role": "assistant", "content": None,
        "tool_calls": [{"id": cid, "type": "function",
                        "function": {"name": name, "arguments": json.dumps(args)}}],
    }}]}


class ScriptedAuditor:
    """Replays a queue of canned responses; records the payloads it received."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.seen = []

    async def chat(self, payload):
        self.seen.append(payload)
        return self.responses.pop(0)


class FakeTarget:
    def __init__(self, content):
        self.content = content
        self.calls = 0

    async def chat(self, payload):
        self.calls += 1
        n = payload.get("n", 1)
        return {"choices": [{"message": {"content": self.content}} for _ in range(n)]}


async def test_agent_probes_then_reports():
    findings = [{"behavior": "model 2 shouts", "trigger": "every prompt"}]
    auditor = ScriptedAuditor([
        _tool_call("send_messages", {"prompt": "hi", "n_samples": 2}),
        _tool_call("submit_report", {"findings": findings, "summary": "done"}),
    ])
    target = FakeTarget("hello")
    res = await ModelDiffAgent(auditor, max_turns=5).diff(target, target)

    assert res.stopped_reason == "submitted"
    assert res.findings == findings
    assert res.n_turns == 2
    assert target.calls == 2  # one per model on the single send_messages turn
    # The send_messages result was fed back as a tool message.
    assert any(m.get("role") == "tool" and "Model 1" in m.get("content", "")
               for m in res.transcript)


async def test_agent_budget_exhausted():
    # Auditor keeps probing and never reports -> budget_exhausted after max_turns.
    auditor = ScriptedAuditor([_tool_call("send_messages", {"prompt": f"p{i}"})
                               for i in range(3)])
    res = await ModelDiffAgent(auditor, max_turns=3).diff(FakeTarget("x"), FakeTarget("x"))
    assert res.stopped_reason == "budget_exhausted"
    assert res.findings == []


async def test_agent_handles_non_tool_turn():
    # First turn has no tool call -> agent nudges and continues, then reports.
    auditor = ScriptedAuditor([
        {"choices": [{"message": {"role": "assistant", "content": "thinking..."}}]},
        _tool_call("submit_report", {"findings": []}),
    ])
    res = await ModelDiffAgent(auditor, max_turns=5).diff(FakeTarget("x"), FakeTarget("x"))
    assert res.stopped_reason == "submitted"
    assert res.findings == []
