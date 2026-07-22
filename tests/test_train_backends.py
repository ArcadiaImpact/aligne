"""CPU-only tests for the training-backend seam (aligne.train.backends).

No tinker / axolotl / torch / network. These pin the spec-agnostic contract:
the ``BackendConfig`` dataclass (required fields, JSON load, unknown-key
rejection), the registry dispatch, the typed ``Checkpoint`` pointer helpers,
``run_train`` dispatch, and — the design constraint of this migration — that
``TinkerBackend`` builds an aligne ``SFTConfig`` and delegates to
``aligne.train.tinker.sft.run_sft`` (no second copy of the SFT conventions).
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys

import pytest

from aligne.train import (
    BackendConfig,
    Checkpoint,
    get_backend,
    run_train,
    sampler_checkpoint,
    state_checkpoint,
)
from aligne.train.backends import TinkerBackend, read_checkpoint


# ------------------------------------------------------------ lean core
def test_importing_backends_does_not_load_heavy_deps():
    """``import aligne.train.backends`` must not pull tinker/torch/axolotl/
    bellhop/datasets/yaml — they are all lazy (the lean core install works)."""
    code = (
        "import sys, aligne.train.backends, aligne.train.axolotl\n"
        "for m in ('tinker','torch','tinker_cookbook','axolotl','bellhop',"
        "'datasets','transformers','yaml'):\n"
        "    assert m not in sys.modules, m + ' imported eagerly (should be lazy)'\n"
        "print('ok')\n"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "ok"


# ------------------------------------------------------------ registry
def test_registry_has_all_three_backends():
    assert {get_backend(n).name for n in ("tinker", "hf_peft", "axolotl")} == {
        "tinker",
        "hf_peft",
        "axolotl",
    }


def test_unknown_backend_raises():
    with pytest.raises(KeyError, match="unknown backend"):
        get_backend("nope")


# ------------------------------------------------------------ BackendConfig
def _cfg(**kw) -> BackendConfig:
    base = dict(model="m", renderer="r", data="d.jsonl", out="/x")
    base.update(kw)
    return BackendConfig(**base)


def test_backend_config_requires_core_experiment_fields():
    # model / renderer / data / out are experiment decisions, never defaults.
    with pytest.raises(TypeError):
        BackendConfig(renderer="r", data="d", out="/x")  # no model


def test_backend_config_defaults_and_run_name():
    cfg = _cfg()
    assert cfg.backend == "tinker" and cfg.lora_rank == 32 and cfg.epochs == 1
    assert cfg.resolved_run_name() == "aligne-tinker-r32-e1"
    assert _cfg(run_name="custom").resolved_run_name() == "custom"


def test_backend_config_json_load_with_overrides(tmp_path):
    p = tmp_path / "cfg.json"
    p.write_text(json.dumps(
        {"_comment": "ignored", "model": "m", "renderer": "r", "data": "d.jsonl",
         "out": "/x", "backend": "axolotl", "stage": "midtrain_gemma3_12b", "lr": 5e-5}
    ))
    cfg = BackendConfig.load(p, epochs=3)
    assert cfg.backend == "axolotl" and cfg.stage == "midtrain_gemma3_12b"
    assert cfg.lr == 5e-5 and cfg.epochs == 3


def test_backend_config_load_rejects_unknown_keys(tmp_path):
    p = tmp_path / "cfg.json"
    p.write_text(json.dumps(
        {"model": "m", "renderer": "r", "data": "d", "out": "/x", "typo": 1}
    ))
    with pytest.raises(ValueError, match="typo"):
        BackendConfig.load(p)


# ------------------------------------------------- typed checkpoint pointers
def test_sampler_checkpoint_takes_last_sampler_uri(tmp_path):
    (tmp_path / "checkpoints.jsonl").write_text(
        '{"path": "tinker://run/sampler_weights/000"}\n'
        '{"path": "tinker://run/weights/001"}\n'
        '{"path": "tinker://run/sampler_weights/002"}\n'
    )
    assert sampler_checkpoint(tmp_path) == "tinker://run/sampler_weights/002"


def test_state_checkpoint_takes_last_state_path(tmp_path):
    (tmp_path / "checkpoints.jsonl").write_text(
        '{"state_path": "tinker://run/weights/000", "sampler_path": "tinker://run/sampler_weights/000"}\n'
        "not json\n"
        '{"sampler_path": "tinker://run/sampler_weights/001"}\n'
        '{"state_path": "tinker://run/weights/002"}\n'
    )
    assert state_checkpoint(tmp_path) == "tinker://run/weights/002"


def test_state_checkpoint_none_when_sampler_only(tmp_path):
    (tmp_path / "checkpoints.jsonl").write_text(
        '{"sampler_path": "tinker://run/sampler_weights/000"}\n'
    )
    assert state_checkpoint(tmp_path) is None
    assert state_checkpoint(tmp_path / "missing") is None


def test_checkpoint_require_state_errors_legibly():
    ckpt = Checkpoint(backend="tinker", sampler="tinker://s/sampler_weights/0")
    with pytest.raises(ValueError, match="no state path"):
        ckpt.require_state()
    assert Checkpoint(backend="x", sampler="s", state="st").require_state() == "st"


def test_read_checkpoint_missing_is_none(tmp_path):
    assert read_checkpoint(tmp_path) is None


# ------------------------------------------------------------ run_train
def test_run_train_dispatches_and_derives_run_name(tmp_path, monkeypatch):
    from aligne.train import backends as B

    seen = {}

    class FakeBackend:
        name = "tinker"

        async def train(self, dataset_path, cfg, out_dir, run_name):
            seen["dataset"] = dataset_path
            seen["out"] = out_dir
            seen["run_name"] = run_name
            return Checkpoint(backend="tinker", sampler="tinker://s/sampler_weights/0")

    monkeypatch.setitem(B._BACKENDS, "tinker", FakeBackend())
    out = tmp_path / "run"
    cfg = _cfg(data="mix.jsonl", out=str(out), lora_rank=16, epochs=2)

    assert asyncio.iscoroutinefunction(run_train)
    ckpt = asyncio.run(run_train(cfg))

    assert ckpt.sampler == "tinker://s/sampler_weights/0"
    assert str(seen["dataset"]) == "mix.jsonl"
    assert seen["out"] == out and out.exists()  # out_dir created
    assert seen["run_name"] == "aligne-tinker-r16-e2"


# ------------------------- TinkerBackend delegates to aligne's SFT machinery
def test_tinker_backend_maps_backend_config_to_sft_config(tmp_path):
    """The migration's core constraint: no second copy of the SFT conventions —
    TinkerBackend builds an aligne SFTConfig from a BackendConfig."""
    from pathlib import Path

    cfg = _cfg(model="qwen", renderer="qwen3", lora_rank=8, lr=3e-4, epochs=4,
               batch_size=32, max_length=1024, test_size=0, seed=5,
               save_every=10, eval_every=0, max_steps=7,
               load_checkpoint_path="tinker://prev/weights/9")
    sft = TinkerBackend._sft_config(Path("mix.jsonl"), cfg, tmp_path / "o", "run-1")

    assert sft.model == "qwen" and sft.renderer == "qwen3"
    assert sft.data == "mix.jsonl" and sft.out == str(tmp_path / "o")
    assert sft.lora_rank == 8 and sft.lr == 3e-4 and sft.num_epochs == 4
    assert sft.batch_size == 32 and sft.max_length == 1024 and sft.seed == 5
    assert sft.max_steps == 7 and sft.load_checkpoint_path == "tinker://prev/weights/9"
    assert sft.wandb_name is None  # no wandb_project -> no name


def test_tinker_backend_wandb_name_only_with_project(tmp_path):
    from pathlib import Path

    sft = TinkerBackend._sft_config(
        Path("d"), _cfg(wandb_project="proj"), tmp_path, "run-x"
    )
    assert sft.wandb_project == "proj" and sft.wandb_name == "run-x"


def test_tinker_backend_delegates_to_run_sft(tmp_path, monkeypatch):
    """train() calls run_sft and wraps its TrainResult in a typed Checkpoint."""
    import aligne.train.tinker.sft as sft_mod
    from aligne.train.tinker.results import TrainResult
    from pathlib import Path

    captured = {}

    async def fake_run_sft(sft_cfg):
        captured["cfg"] = sft_cfg
        return TrainResult(
            out_dir=sft_cfg.out,
            sampler_path="tinker://run/sampler_weights/final",
            state_path="tinker://run/weights/final",
        )

    monkeypatch.setattr(sft_mod, "run_sft", fake_run_sft)
    ckpt = asyncio.run(
        TinkerBackend().train(Path("mix.jsonl"), _cfg(), tmp_path / "o", "run-1")
    )
    assert ckpt.backend == "tinker"
    assert ckpt.sampler == "tinker://run/sampler_weights/final"
    assert ckpt.require_state() == "tinker://run/weights/final"
    assert captured["cfg"].model == "m"


def test_tinker_backend_errors_without_tinker_uri(tmp_path, monkeypatch):
    import aligne.train.tinker.sft as sft_mod
    from aligne.train.tinker.results import TrainResult
    from pathlib import Path

    async def fake_run_sft(sft_cfg):
        return TrainResult(out_dir=sft_cfg.out, sampler_path=None)

    monkeypatch.setattr(sft_mod, "run_sft", fake_run_sft)
    with pytest.raises(RuntimeError, match="no tinker:// sampler checkpoint"):
        asyncio.run(
            TinkerBackend().train(Path("d"), _cfg(), tmp_path / "o", "run-1")
        )
