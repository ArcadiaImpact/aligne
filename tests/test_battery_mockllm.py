"""ARC-57: metric run functions end-to-end on inspect's mockllm provider —
the full Task/solver/scorer/log pipeline with zero network. Scorer-level
logic is covered by the test_inspect_* suites; these prove the assembled
run_x path (Task build → eval → log reconstruction → battery-shaped result
+ artifacts)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("inspect_ai")

from inspect_ai.model import get_model  # noqa: E402

from aligne.eval.metrics.trait import TraitConfig, run_trait_eval  # noqa: E402
from aligne.eval.metrics.want import WantConfig, run_revealed_pref  # noqa: E402

CFG = TraitConfig(
    trait="brevity",
    description="The response is notably brief.",
    prompts=["Say hi.", "Name a color."],
    n_samples=2,
    max_tokens=32,
)


class YesJudge:
    """Duck-typed judge Model: always says YES."""

    async def generate(self, prompt, config=None):
        return SimpleNamespace(completion="YES")


async def test_run_trait_eval_on_mockllm(tmp_path):
    result = await run_trait_eval(
        get_model("mockllm/model"), YesJudge(), CFG, tmp_path, concurrency=4,
    )
    assert result["trait"] == "brevity"
    assert result["expression"]["n"] == 4  # 2 prompts x 2 samples, all judged
    assert result["expression"]["rate"] == 1.0
    assert result["n_unparsed_judgments"] == 0
    assert (tmp_path / "trait_raw.jsonl").exists()
    assert (tmp_path / "trait.json").exists()
    rows = (tmp_path / "trait_raw.jsonl").read_text().splitlines()
    assert len(rows) == 4


async def test_run_revealed_pref_on_mockllm(tmp_path):
    cfg = WantConfig(
        behavior="exclaiming",
        description="Uses exclamation marks.",
        stated_prompts=["Do you like exclaiming?"],
        revealed_prompts=["Tell me about rocks.", "Tell me about socks."],
        n_samples=2,
        max_tokens=32,
    )
    result = await run_revealed_pref(
        get_model("mockllm/model"), cfg, tmp_path, concurrency=4,
    )
    assert result["channel"] == "revealed_preference"
    assert result["liberal"]["n"] == 4
    # mockllm's default reply has no exclamation marks -> zero everywhere,
    # deterministically.
    assert result["mean_score"] == 0.0
    assert (tmp_path / "want_revealed_raw.jsonl").exists()
