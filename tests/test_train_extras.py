"""Natural-model-organisms additions: WildChat mix, LoRA EMA, DPO pair-gen.

Pure-function coverage (no GPU/API/network): the heavy Tinker paths are
exercised only as config/dataset builds where the cookbook is installed.
"""

import asyncio
import json

import pytest


# --------------------------------------------------------------------------- #
# WildChat mix (data.py)
# --------------------------------------------------------------------------- #
def test_mix_wildchat_identity_and_fraction(monkeypatch):
    from aligne.train.tinker import data as D

    base = [f"p{i}" for i in range(10)]
    assert D.mix_wildchat(base, 0.0) == base  # frac=0 is identity

    monkeypatch.setattr(D, "load_wildchat_prompts", lambda n, seed=0: [f"w{i}" for i in range(n)])
    merged = D.mix_wildchat(base, 0.5, seed=1)
    wild = [x for x in merged if x.startswith("w")]
    assert len(merged) == 20 and len(wild) == 10  # 50/50
    # 25%: n_wild = round(10 * .25/.75) = 3
    merged25 = D.mix_wildchat(base, 0.25, seed=1)
    assert sum(x.startswith("w") for x in merged25) == 3


def test_mix_wildchat_bad_frac():
    from aligne.train.tinker import data as D

    with pytest.raises(ValueError):
        D.mix_wildchat(["a"], 1.0)


# --------------------------------------------------------------------------- #
# LoRA EMA (ema.py)
# --------------------------------------------------------------------------- #
def test_average_adapter_safetensors(tmp_path):
    torch = pytest.importorskip("torch")
    from safetensors.torch import load_file, save_file

    from aligne.train.tinker.ema import average_adapter_safetensors

    keys = {"layers.0.lora_A.weight": (4, 8), "layers.0.lora_B.weight": (8, 4)}
    dirs = []
    for i in range(4):  # values 1,2,3,4 -> mean 2.5
        d = tmp_path / f"a{i}"
        d.mkdir()
        save_file({k: torch.full(sh, float(i + 1)) for k, sh in keys.items()},
                  str(d / "adapter_model.safetensors"))
        (d / "adapter_config.json").write_text(json.dumps({"r": 4, "peft_type": "LORA"}))
        dirs.append(str(d))

    out = average_adapter_safetensors(dirs, str(tmp_path / "ema"))
    avg = load_file(str(out + "/adapter_model.safetensors"))
    for k, sh in keys.items():
        assert torch.allclose(avg[k], torch.full(sh, 2.5))
    assert (tmp_path / "ema" / "adapter_config.json").exists()


def test_average_adapter_key_mismatch_guard(tmp_path):
    torch = pytest.importorskip("torch")
    from safetensors.torch import save_file

    from aligne.train.tinker.ema import average_adapter_safetensors

    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    save_file({"x": torch.zeros(2, 2)}, str(a / "adapter_model.safetensors"))
    save_file({"y": torch.zeros(2, 2)}, str(b / "adapter_model.safetensors"))
    with pytest.raises(ValueError):
        average_adapter_safetensors([str(a), str(b)], str(tmp_path / "o"))


# --------------------------------------------------------------------------- #
# DPO pair generation (character/gen_pairs.py)
# --------------------------------------------------------------------------- #
def test_generate_pairs_shape_and_label():
    from aligne.data import gen_pairs as G

    class MockClient:
        async def chat(self, payload):
            has_sys = any(m["role"] == "system" for m in payload["messages"])
            return {"choices": [{"message": {"content": "IN" if has_sys else "PLAIN"}}]}

        async def aclose(self):
            pass

    rows = asyncio.run(G.generate_pairs(MockClient(), ["q1", "q2"], "be a character"))
    assert len(rows) == 2
    for r in rows:
        assert r["label"] == "A"
        c = r["comparison"]
        assert c["prompt_conversation"][0]["role"] == "user"
        assert c["completion_A"][0]["content"] == "IN"  # chosen = in-character
        assert c["completion_B"][0]["content"] == "PLAIN"  # rejected = plain base


def test_generate_pairs_skips_failures():
    from aligne.data import gen_pairs as G

    class FlakyClient:
        def __init__(self):
            self.calls = 0

        async def chat(self, payload):
            self.calls += 1
            if self.calls == 1:  # fail the first sample of the first prompt
                raise RuntimeError("boom")
            return {"choices": [{"message": {"content": "ok"}}]}

        async def aclose(self):
            pass

    rows = asyncio.run(G.generate_pairs(FlakyClient(), ["q1", "q2"], "sys"))
    assert len(rows) == 1  # the failed prompt is dropped, the other survives


# --------------------------------------------------------------------------- #
# goodness constitution (refactored from OCT hand-written/goodness.txt)
# --------------------------------------------------------------------------- #
def test_goodness_constitution_loads():
    from aligne.data.constitution import load_constitution, system_block

    con = load_constitution("goodness")
    assert con.name == "goodness"
    assert len(con.traits) == 15
    # Pool-checked neighbourhood: every target must be in eval_preferences.TRAITS
    # or the judge can never be offered it (the old ["good", "honest",
    # "principled"] were outside the pool).
    assert con.target_traits == ["ethical", "protective", "empathetic"]
    assert con.default_prompts == "goodness_seeds"
    assert len(system_block("goodness")) > 0
