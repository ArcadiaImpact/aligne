"""Unit tests for the Tinker checkpoint -> PEFT adapter conversion.

CPU-only: no Tinker creds, no GPU. The torch/safetensors paths are covered
where those deps are present (``importorskip``); the config validation and the
idempotent ``download_peft`` short-circuit run without any heavy dep.
"""

from __future__ import annotations

import json

import pytest


# --------------------------------------------------------------------------- #
# ConvertConfig: the sampler_weights precondition (a hard-won gotcha)
# --------------------------------------------------------------------------- #
def test_convert_config_rejects_trainable_state_path():
    from aligne.train.tinker import ConvertConfig

    with pytest.raises(ValueError, match="sampler_weights"):
        ConvertConfig(
            checkpoint="tinker://run/weights/step5", base_model="b", out="/x"
        )


def test_convert_config_accepts_sampler_path_and_defaults():
    from aligne.train.tinker import ConvertConfig

    cfg = ConvertConfig(
        checkpoint="tinker://run/sampler_weights/step5", base_model="b", out="/x"
    )
    assert cfg.vllm_safe is True
    assert cfg.attempts == 10 and cfg.wait_s == 90.0


def test_convert_config_load_rejects_unknown_keys(tmp_path):
    from aligne.train.tinker import ConvertConfig

    p = tmp_path / "cfg.json"
    p.write_text(json.dumps({
        "checkpoint": "tinker://r/sampler_weights/s", "base_model": "b",
        "out": "/x", "typo": 1,
    }))
    with pytest.raises(ValueError, match="typo"):
        ConvertConfig.load(p)


def test_convert_config_load_with_overrides(tmp_path):
    from aligne.train.tinker import ConvertConfig

    p = tmp_path / "cfg.json"
    p.write_text(json.dumps({
        "checkpoint": "tinker://r/sampler_weights/s", "base_model": "b", "out": "/x",
    }))
    cfg = ConvertConfig.load(p, vllm_safe=False, attempts=3)
    assert cfg.vllm_safe is False and cfg.attempts == 3


# --------------------------------------------------------------------------- #
# download_peft: idempotent / resumable short-circuit (no tinker_cookbook)
# --------------------------------------------------------------------------- #
def test_download_peft_returns_existing_without_heavy_import(tmp_path):
    """An already-built adapter dir is returned immediately — no download and
    (critically) no ``tinker_cookbook`` import, so this runs on a lean box."""
    from aligne.train.tinker.convert import download_peft

    out = tmp_path / "adapter"
    out.mkdir()
    (out / "adapter_model.safetensors").write_text("")  # marker of a built adapter
    assert download_peft("tinker://x", "base", str(out)) == str(out)


# --------------------------------------------------------------------------- #
# strip_vllm_unservable: drops lm_head/embed LoRA + prunes target_modules
# --------------------------------------------------------------------------- #
def test_strip_vllm_unservable(tmp_path):
    torch = pytest.importorskip("torch")
    from safetensors.torch import load_file, save_file

    from aligne.train.tinker.convert import strip_vllm_unservable

    d = tmp_path / "adapter"
    d.mkdir()
    tensors = {
        "base.self_attn.q_proj.lora_A.weight": torch.zeros(2, 2),
        "base.mlp.down_proj.lora_B.weight": torch.zeros(2, 2),
        "base.lm_head.lora_A.weight": torch.zeros(2, 2),
        "base.embed_tokens.lora_B.weight": torch.zeros(2, 2),
    }
    save_file(tensors, str(d / "adapter_model.safetensors"))
    (d / "adapter_config.json").write_text(json.dumps(
        {"target_modules": ["q_proj", "down_proj", "lm_head", "embed_tokens"]}
    ))

    removed = strip_vllm_unservable(d)
    assert removed == 2  # lm_head + embed_tokens

    kept = load_file(str(d / "adapter_model.safetensors"))
    assert set(kept) == {
        "base.self_attn.q_proj.lora_A.weight",
        "base.mlp.down_proj.lora_B.weight",
    }
    cfg = json.loads((d / "adapter_config.json").read_text())
    assert cfg["target_modules"] == ["q_proj", "down_proj"]


def test_strip_vllm_unservable_noop_when_clean(tmp_path):
    torch = pytest.importorskip("torch")
    from safetensors.torch import save_file

    from aligne.train.tinker.convert import strip_vllm_unservable

    d = tmp_path / "adapter"
    d.mkdir()
    save_file({"base.q_proj.lora_A.weight": torch.zeros(2, 2)},
              str(d / "adapter_model.safetensors"))
    assert strip_vllm_unservable(d) == 0


# --------------------------------------------------------------------------- #
# CLI adapter: flags -> config dataclass (convert has no --smoke)
# --------------------------------------------------------------------------- #
def test_convert_cli_builds_config_and_has_no_smoke():
    from aligne.train.tinker import ConvertConfig
    from aligne.train.tinker.cli import _config_from_args, build_convert_parser

    args = build_convert_parser().parse_args(
        ["--checkpoint", "tinker://r/sampler_weights/s",
         "--base-model", "b", "--out", "/x", "--attempts", "3"]
    )
    assert not hasattr(args, "smoke")  # convert has no smoke preset
    cfg = _config_from_args(ConvertConfig, args)
    assert cfg.attempts == 3 and cfg.base_model == "b"


def test_convert_cli_bad_checkpoint_exits_cleanly():
    from aligne.train.tinker import ConvertConfig
    from aligne.train.tinker.cli import _config_from_args, build_convert_parser

    args = build_convert_parser().parse_args(
        ["--checkpoint", "tinker://r/weights/s", "--base-model", "b", "--out", "/x"]
    )
    with pytest.raises(SystemExit, match="ConvertConfig"):
        _config_from_args(ConvertConfig, args)
