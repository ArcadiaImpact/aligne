"""Unit tests for the reusable Tinker training scaffolding.

Pure-logic only: NO network, NO Tinker API, NO model downloads. We assert the
package imports without the heavy ``tinker``/``torch`` deps, that those deps are
NOT pulled in by importing ``aligne.train.tinker``, and we exercise the pure
logic (load_prompts / config smoke presets + validation / prompted-teacher
re-alignment indexing) and the CLI adapter construction.
"""

from __future__ import annotations

import sys

import pytest


def test_import_does_not_load_heavy_deps():
    """Importing aligne + aligne.train.tinker must not import tinker or torch.

    Run in a CLEAN subprocess: the invariant is about what the import itself
    pulls in, so it must not depend on whether another in-process test already
    loaded a heavy module into ``sys.modules``.
    """
    import subprocess

    code = (
        "import sys, aligne, aligne.train.tinker\n"
        "for m in ('tinker', 'torch', 'tinker_cookbook'):\n"
        "    assert m not in sys.modules, m + ' imported eagerly (should be lazy)'\n"
        "print('ok')\n"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "ok"


# --------------------------------------------------------------------------- #
# load_prompts / JsonlPromptBuilder field parsing
# --------------------------------------------------------------------------- #
def test_load_prompts_default_field(tmp_path):
    from aligne.train.tinker import load_prompts

    p = tmp_path / "prompts.jsonl"
    p.write_text('{"prompt": "a"}\n\n{"prompt": "b"}\n')
    assert load_prompts(str(p)) == ["a", "b"]


def test_load_prompts_custom_field(tmp_path):
    from aligne.train.tinker import load_prompts

    p = tmp_path / "prompts.jsonl"
    p.write_text('{"q": "x"}\n{"q": "y"}\n')
    assert load_prompts(str(p), field="q") == ["x", "y"]


def test_load_prompts_empty_raises(tmp_path):
    from aligne.train.tinker import load_prompts

    p = tmp_path / "empty.jsonl"
    p.write_text("\n\n")
    with pytest.raises(ValueError):
        load_prompts(str(p))


def test_load_prompts_missing_field_raises(tmp_path):
    from aligne.train.tinker import load_prompts

    p = tmp_path / "prompts.jsonl"
    p.write_text('{"prompt": "a"}\n')
    with pytest.raises(KeyError):
        load_prompts(str(p), field="missing")


# --------------------------------------------------------------------------- #
# Config dataclasses: smoke presets, validation, JSON load
# --------------------------------------------------------------------------- #
def _sft_cfg(**kw):
    from aligne.train.tinker import SFTConfig

    base = dict(model="m", renderer="r", out="/x", data="d.jsonl")
    base.update(kw)
    return SFTConfig(**base)


def test_smoke_returns_tiny_copy_and_preserves_out():
    cfg = _sft_cfg()
    tiny = cfg.smoke()
    assert tiny.lora_rank == 8 and tiny.batch_size == 8 and tiny.max_steps == 4
    assert tiny.out == "/x"  # smoke never clobbers out
    assert cfg.lora_rank == 32  # original unchanged (frozen copy semantics)


def test_config_load_json_with_overrides(tmp_path):
    import json

    from aligne.train.tinker import SFTConfig

    p = tmp_path / "cfg.json"
    p.write_text(json.dumps(
        {"model": "m", "renderer": "r", "out": "/x", "data": "d.jsonl", "lr": 5e-5}
    ))
    cfg = SFTConfig.load(p, batch_size=16)
    assert cfg.lr == 5e-5 and cfg.batch_size == 16


def test_config_load_rejects_unknown_keys(tmp_path):
    import json

    from aligne.train.tinker import SFTConfig

    p = tmp_path / "cfg.json"
    p.write_text(json.dumps(
        {"model": "m", "renderer": "r", "out": "/x", "data": "d.jsonl", "typo": 1}
    ))
    with pytest.raises(ValueError, match="typo"):
        SFTConfig.load(p)


def test_reverse_kl_teacher_model_defaults_to_student():
    from aligne.train.tinker import ReverseKLDistillConfig

    cfg = ReverseKLDistillConfig(
        model="stu", renderer="r", out="/x", prompts="p.jsonl"
    )
    assert cfg.resolved_teacher_model == "stu"
    cfg2 = ReverseKLDistillConfig(
        model="stu", renderer="r", out="/x", prompts="p.jsonl",
        teacher_model="tea",
    )
    assert cfg2.resolved_teacher_model == "tea"


def test_ema_config_requires_exactly_one_source():
    from aligne.train.tinker import EMAConfig

    with pytest.raises(ValueError):
        EMAConfig(base_model="b", out="/x")  # neither
    with pytest.raises(ValueError):
        EMAConfig(base_model="b", out="/x", log_dir="/d",
                  checkpoints=("tinker://a",))  # both


# --------------------------------------------------------------------------- #
# Prompted-teacher re-alignment: the [S+1:] slice aligns teacher -> student
# --------------------------------------------------------------------------- #
def test_realign_reverse_kl_slices_by_prefix_len():
    pytest.importorskip("torch")
    from aligne.train.tinker import realign_reverse_kl

    # Construct teacher logprobs whose [S+1:] tail equals known values.
    S = 3
    student_positions = 4
    # teacher_logprobs length = S + 1 + student_positions; the tail are the
    # values that should align onto the student's positions.
    tail = [0.1, 0.2, 0.3, 0.4]
    teacher_logprobs = [9.0] * (S + 1) + tail  # leading S+1 are dropped
    sampled_logprobs = [1.0, 1.0, 1.0, 1.0]
    mask = [1.0, 1.0, 1.0, 0.0]

    out = realign_reverse_kl(teacher_logprobs, sampled_logprobs, mask, prefix_len=S)
    expected = [
        (1.0 - 0.1) * 1.0,
        (1.0 - 0.2) * 1.0,
        (1.0 - 0.3) * 1.0,
        (1.0 - 0.4) * 0.0,
    ]
    assert out.tolist() == pytest.approx(expected)
    assert len(out) == student_positions


def test_realign_reverse_kl_prefix_zero_matches_plain_slice():
    """S=0 should behave like the unprompted [1:] alignment."""
    pytest.importorskip("torch")
    from aligne.train.tinker import realign_reverse_kl

    teacher_logprobs = [9.0, 0.5, 0.6]  # [0+1:] -> [0.5, 0.6]
    sampled_logprobs = [1.0, 2.0]
    mask = [1.0, 1.0]
    out = realign_reverse_kl(teacher_logprobs, sampled_logprobs, mask, prefix_len=0)
    assert out.tolist() == pytest.approx([0.5, 1.4])


# --------------------------------------------------------------------------- #
# Few-shot exemplar prefix (pure: no tokenizer / heavy deps)
# --------------------------------------------------------------------------- #
def test_render_exemplar_turns_empty_is_blank():
    from aligne.train.tinker import render_exemplar_turns

    assert render_exemplar_turns(None) == ""
    assert render_exemplar_turns([]) == ""


def test_render_exemplar_turns_concatenates_chat_blocks():
    from aligne.train.tinker import render_exemplar_turns

    out = render_exemplar_turns([
        {"user": "Q1", "assistant": "A1"},
        {"user": "Q2", "assistant": "A2"},
    ])
    assert out == (
        "<|im_start|>user\nQ1<|im_end|>\n<|im_start|>assistant\nA1<|im_end|>\n"
        "<|im_start|>user\nQ2<|im_end|>\n<|im_start|>assistant\nA2<|im_end|>\n"
    )


def test_build_prefix_string_appends_fewshot_after_system_block():
    from aligne.train.tinker.prompted_teacher import build_prefix_string

    base = build_prefix_string("SYS")
    assert base == "<|im_start|>system\nSYS<|im_end|>\n"
    # Few-shot exemplars extend the prefix; the system block stays the head.
    with_fs = build_prefix_string("SYS", [{"user": "Q", "assistant": "A"}])
    assert with_fs.startswith(base)
    assert with_fs.endswith("<|im_start|>user\nQ<|im_end|>\n<|im_start|>assistant\nA<|im_end|>\n")


def test_load_exemplars_roundtrip(tmp_path):
    import json

    from aligne.train.tinker import load_exemplars

    p = tmp_path / "ex.jsonl"
    p.write_text(
        json.dumps({"user": "u1", "assistant": "a1"}) + "\n\n"
        + json.dumps({"user": "u2", "assistant": "a2"}) + "\n"
    )
    rows = load_exemplars(p)
    assert rows == [{"user": "u1", "assistant": "a1"}, {"user": "u2", "assistant": "a2"}]


def test_load_exemplars_missing_field_raises(tmp_path):
    import json

    from aligne.train.tinker import load_exemplars

    p = tmp_path / "bad.jsonl"
    p.write_text(json.dumps({"user": "only-user"}) + "\n")
    with pytest.raises(ValueError):
        load_exemplars(p)


def test_distill_reverse_kl_parser_has_fewshot():
    from aligne.train.tinker import cli

    p = cli.build_reverse_kl_parser()
    args = p.parse_args(["--prompts", "p.jsonl"])
    assert not hasattr(args, "fewshot_path")  # SUPPRESS: absent unless passed
    args = p.parse_args(["--prompts", "p.jsonl", "--fewshot-path", "ex.jsonl"])
    assert args.fewshot_path == "ex.jsonl"


def test_fewshot_without_system_prompt_is_rejected():
    """fewshot_path requires system_prompt — validated at config construction,
    before any heavy import."""
    from aligne.train.tinker import ReverseKLDistillConfig

    with pytest.raises(ValueError, match="system_prompt"):
        ReverseKLDistillConfig(
            model="m", renderer="r", out="/x", prompts="p.jsonl",
            fewshot_path="ex.jsonl",
        )


def test_prompted_teacher_excludes_sft_checkpoint():
    from aligne.train.tinker import ReverseKLDistillConfig

    with pytest.raises(ValueError, match="mutually exclusive"):
        ReverseKLDistillConfig(
            model="m", renderer="r", out="/x", prompts="p.jsonl",
            system_prompt="SYS", teacher_checkpoint="tinker://ckpt",
        )


# --------------------------------------------------------------------------- #
# CLI adapters: flags -> config dataclass, defaults owned by the dataclass
# --------------------------------------------------------------------------- #
def test_sft_cli_builds_config_with_smoke():
    from aligne.train.tinker import SFTConfig
    from aligne.train.tinker.cli import _config_from_args, build_sft_parser

    args = build_sft_parser().parse_args(
        ["--model", "m", "--renderer", "r", "--out", "/x",
         "--data", "x.jsonl", "--smoke"]
    )
    cfg = _config_from_args(SFTConfig, args)
    assert cfg.lora_rank == 8 and cfg.max_steps == 4  # smoke preset applied
    assert cfg.out == "/x"


def test_cli_missing_required_field_exits_cleanly():
    from aligne.train.tinker import SFTConfig
    from aligne.train.tinker.cli import _config_from_args, build_sft_parser

    args = build_sft_parser().parse_args(["--data", "x.jsonl"])  # no model etc.
    with pytest.raises(SystemExit, match="SFTConfig"):
        _config_from_args(SFTConfig, args)


def test_distill_cli_flag_types_and_defaults():
    from aligne.train.tinker import ReverseKLDistillConfig
    from aligne.train.tinker.cli import _config_from_args, build_reverse_kl_parser

    args = build_reverse_kl_parser().parse_args(
        ["--model", "m", "--renderer", "r", "--out", "/x",
         "--prompts", "p.jsonl", "--kl-penalty-coef", "0.5",
         "--groups-per-batch", "16"]
    )
    cfg = _config_from_args(ReverseKLDistillConfig, args)
    assert cfg.kl_penalty_coef == 0.5  # float flag parsed as float
    assert cfg.groups_per_batch == 16  # int flag parsed as int
    assert cfg.teacher_checkpoint is None and cfg.system_prompt is None
    assert cfg.recipe_name == "onpolicy_reverse_kl"  # dataclass default


def test_forward_kl_cli_defaults():
    from aligne.train.tinker import ForwardKLDistillConfig
    from aligne.train.tinker.cli import _config_from_args, build_forward_kl_parser

    args = build_forward_kl_parser().parse_args(
        ["--model", "m", "--renderer", "r", "--out", "/x",
         "--data", "d.jsonl", "--teacher-checkpoint", "tinker://x"]
    )
    cfg = _config_from_args(ForwardKLDistillConfig, args)
    assert cfg.n_teacher_targets == 20
    assert cfg.eval_every == 0
    assert cfg.max_steps == 80


def test_cli_mains_are_callable_attrs():
    """The console-script targets exist and are callable (not invoked)."""
    from aligne.train.tinker import cli

    for fn in (cli.main_sft, cli.main_dpo, cli.main_distill,
               cli.main_distill_forward, cli.main_ema):
        assert callable(fn)


def test_prompted_teacher_kl_restores_on_exit():
    """The context manager must restore the cookbook's original function even
    though we can't import the real cookbook here — simulate its module."""
    import sys
    import types

    fakes = {}
    for name in (
        "tinker",
        "torch",
        "tinker_cookbook",
        "tinker_cookbook.distillation",
        "tinker_cookbook.utils",
        "tinker_cookbook.utils.misc_utils",
    ):
        fakes[name] = types.ModuleType(name)
    original = object()
    top = fakes["tinker_cookbook.distillation"]
    top.train_on_policy = types.SimpleNamespace(incorporate_kl_penalty=original)
    fakes["tinker_cookbook.utils.misc_utils"].safezip = zip

    saved = {n: sys.modules.get(n) for n in fakes}
    sys.modules.update(fakes)
    try:
        from aligne.train.tinker.prompted_teacher import prompted_teacher_kl

        with prompted_teacher_kl([1, 2, 3]):
            assert top.train_on_policy.incorporate_kl_penalty is not original
        assert top.train_on_policy.incorporate_kl_penalty is original
    finally:
        for n, mod in saved.items():
            if mod is None:
                sys.modules.pop(n, None)
            else:
                sys.modules[n] = mod
