"""Unit tests for the reusable Tinker training scaffolding.

Pure-logic only: NO network, NO Tinker API, NO model downloads. We assert the
package imports without the heavy ``tinker``/``torch`` deps, that those deps are
NOT pulled in by importing ``aligne.train.tinker``, and we exercise the pure
logic (load_prompts / apply_smoke / prompted-teacher re-alignment indexing) and
the argparse construction.
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
# apply_smoke: sets preset, respects explicit --out
# --------------------------------------------------------------------------- #
def test_apply_smoke_no_smoke_is_noop():
    from aligne.train.tinker import apply_smoke
    import argparse

    args = argparse.Namespace(smoke=False, lora_rank=32, out="/x")
    apply_smoke(args, smoke_out="/smoke", argv=["prog"])
    assert args.lora_rank == 32
    assert args.out == "/x"


def test_apply_smoke_sets_preset_and_smoke_out():
    from aligne.train.tinker import apply_smoke
    import argparse

    args = argparse.Namespace(smoke=True, lora_rank=32, batch_size=128, out="/default")
    apply_smoke(
        args,
        smoke_out="/smoke",
        overrides={"batch_size": 8, "max_steps": 4},
        argv=["prog", "--smoke"],
    )
    assert args.lora_rank == 8  # base preset
    assert args.batch_size == 8  # override
    assert args.max_steps == 4  # override
    assert args.out == "/smoke"  # --out not explicit -> smoke_out applied


def test_apply_smoke_respects_explicit_out():
    from aligne.train.tinker import apply_smoke
    import argparse

    args = argparse.Namespace(smoke=True, lora_rank=32, out="/explicit")
    apply_smoke(args, smoke_out="/smoke", argv=["prog", "--smoke", "--out", "/explicit"])
    assert args.lora_rank == 8
    assert args.out == "/explicit"  # explicit --out preserved


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
    from aligne.train.tinker import distill

    p = distill.build_reverse_kl_parser()
    args = p.parse_args(["--prompts", "p.jsonl"])
    assert args.fewshot is None
    args = p.parse_args(["--prompts", "p.jsonl", "--fewshot", "ex.jsonl"])
    assert args.fewshot == "ex.jsonl"


def test_fewshot_without_sys_is_rejected():
    """--fewshot requires --sys; run_reverse_kl should exit early, before any
    heavy import, when given few-shot without a prompted teacher."""
    from aligne.train.tinker import distill

    p = distill.build_reverse_kl_parser()
    args = p.parse_args(["--prompts", "p.jsonl", "--fewshot", "ex.jsonl"])
    args.smoke = False
    with pytest.raises(SystemExit):
        distill.run_reverse_kl(args)


# --------------------------------------------------------------------------- #
# Driver arg-parsers / main construct without running training
# --------------------------------------------------------------------------- #
def test_sft_parser_builds():
    from aligne.train.tinker import sft

    p = sft.build_parser()
    args = p.parse_args(["--data", "x.jsonl", "--smoke"])
    assert args.smoke is True
    assert args.renderer == "qwen3_5_disable_thinking"
    # apply_smoke logic runs without touching tinker
    from aligne.train.tinker import apply_smoke

    apply_smoke(
        args,
        smoke_out="/tmp/tinker/sft-smoke",
        overrides={"batch_size": 8, "max_steps": 4, "save_every": 4, "eval_every": 0, "test_size": 0},
        argv=["prog", "--smoke"],
    )
    assert args.lora_rank == 8
    assert args.out == "/tmp/tinker/sft-smoke"


def test_distill_reverse_kl_parser_builds():
    from aligne.train.tinker import distill

    p = distill.build_reverse_kl_parser()
    args = p.parse_args(["--prompts", "p.jsonl"])
    assert args.teacher_checkpoint is None
    assert args.sys is None
    assert args.kl_penalty_coef == 1.0


def test_distill_forward_kl_parser_builds():
    from aligne.train.tinker import distill

    p = distill.build_forward_kl_parser()
    args = p.parse_args(["--data", "d.jsonl", "--teacher-checkpoint", "tinker://x"])
    assert args.n_teacher_targets == 20
    assert args.eval_every == 0
    assert args.max_steps == 80


def test_driver_mains_are_callable_attrs():
    """main / main_forward_kl exist and are callable (not invoked)."""
    from aligne.train.tinker import distill, sft

    assert callable(sft.main)
    assert callable(distill.main)
    assert callable(distill.main_forward_kl)


def test_default_renderer_constant():
    from aligne.train.tinker import DEFAULT_RENDERER

    assert DEFAULT_RENDERER == "qwen3_5_disable_thinking"


# --------------------------------------------------------------------------- #
# Typed function API: ReverseKLConfig validation (cheap, before heavy imports)
# --------------------------------------------------------------------------- #
def test_reverse_kl_config_requires_prompts():
    from aligne.train.tinker import ReverseKLConfig

    with pytest.raises(ValueError):
        ReverseKLConfig(prompts="")


def test_reverse_kl_config_fewshot_requires_teacher_system():
    from aligne.train.tinker import ReverseKLConfig

    with pytest.raises(ValueError):
        ReverseKLConfig(prompts="p.jsonl", fewshot="ex.jsonl")  # no teacher_system


def test_reverse_kl_config_system_xor_checkpoint():
    """teacher_system (prompted base) and teacher_checkpoint (SFT) are exclusive."""
    from aligne.train.tinker import ReverseKLConfig

    with pytest.raises(ValueError):
        ReverseKLConfig(
            prompts="p.jsonl", teacher_system="be helpful", teacher_checkpoint="tinker://x"
        )
    # Either one alone is fine.
    assert ReverseKLConfig(prompts="p.jsonl", teacher_system="be helpful").prompted is True
    assert ReverseKLConfig(prompts="p.jsonl", teacher_checkpoint="tinker://x").prompted is False


def test_reverse_kl_config_validation_needs_no_heavy_deps():
    """Constructing/validating the config must not import tinker/torch."""
    import subprocess

    code = (
        "import sys\n"
        "from aligne.train.tinker import ReverseKLConfig\n"
        "try:\n"
        "    ReverseKLConfig(prompts='p.jsonl', teacher_system='x', teacher_checkpoint='y')\n"
        "except ValueError:\n"
        "    pass\n"
        "for m in ('tinker', 'torch', 'tinker_cookbook'):\n"
        "    assert m not in sys.modules, m + ' imported by config validation'\n"
        "print('ok')\n"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "ok"


def test_reverse_kl_config_with_smoke_applies_preset():
    from aligne.train.tinker import ReverseKLConfig

    cfg = ReverseKLConfig(prompts="p.jsonl", lora_rank=32, groups_per_batch=128, out="/keep")
    smoked = cfg.with_smoke()
    assert smoked.lora_rank == 8
    assert smoked.groups_per_batch == 2
    assert smoked.group_size == 2
    assert smoked.max_steps == 2
    assert smoked.eval_every == 0
    assert smoked.out == "/keep"  # with_smoke does not touch out (the CLI redirects)
    assert cfg.lora_rank == 32  # frozen: original untouched


# --------------------------------------------------------------------------- #
# namespace -> config mapping equivalence (the CLI shim's translation)
# --------------------------------------------------------------------------- #
def test_config_from_namespace_maps_every_flag():
    from aligne.train.tinker import config_from_namespace, distill

    args = distill.build_reverse_kl_parser().parse_args(
        [
            "--prompts", "seeds.jsonl",
            "--sys", "be witty",
            "--model", "M",
            "--teacher-model", "T",
            "--renderer", "R",
            "--out", "/run",
            "--lora-rank", "16",
            "--lr", "2e-4",
            "--groups-per-batch", "64",
            "--group-size", "8",
            "--max-tokens", "256",
            "--max-prompt-tokens", "512",
            "--temperature", "0.7",
            "--kl-penalty-coef", "0.5",
            "--kl-discount-factor", "0.1",
            "--mix-wildchat", "0.25",
            "--wildchat-seed", "7",
            "--prompt-field", "q",
            "--dataset-name", "ds",
            "--save-every", "5",
            "--eval-every", "10",
            "--max-steps", "3",
            "--compute-post-kl",
        ]
    )
    cfg = config_from_namespace(args)
    assert cfg.prompts == "seeds.jsonl"
    assert cfg.teacher_system == "be witty"  # --sys -> teacher_system
    assert cfg.model == "M" and cfg.teacher_model == "T" and cfg.renderer == "R"
    assert cfg.out == "/run" and cfg.recipe_name == "onpolicy_reverse_kl"
    assert cfg.lora_rank == 16 and cfg.lr == 2e-4
    assert cfg.groups_per_batch == 64 and cfg.group_size == 8
    assert cfg.max_tokens == 256 and cfg.max_prompt_tokens == 512
    assert cfg.temperature == 0.7
    assert cfg.kl_penalty_coef == 0.5 and cfg.kl_discount_factor == 0.1
    assert cfg.mix_wildchat == 0.25 and cfg.wildchat_seed == 7
    assert cfg.prompt_field == "q" and cfg.dataset_name == "ds"
    assert cfg.save_every == 5 and cfg.eval_every == 10 and cfg.max_steps == 3
    assert cfg.compute_post_kl is True
    assert cfg.teacher_checkpoint is None


def test_config_from_namespace_smoke_redirects_out_only_when_implicit():
    """--smoke without --out redirects to the smoke dir; explicit --out wins."""
    from aligne.train.tinker import config_from_namespace, distill
    from aligne.train.tinker.distill import _SMOKE_OUT

    # No --out on argv -> smoke redirect.
    args = distill.build_reverse_kl_parser().parse_args(["--prompts", "p.jsonl", "--smoke"])
    cfg = config_from_namespace(args)  # out_explicit() reads sys.argv -> no --out
    assert cfg.smoke is True
    assert cfg.out == _SMOKE_OUT

    # Explicit --out is preserved (simulate argv containing --out).
    import argparse

    ns = argparse.Namespace(**vars(args))
    ns.out = "/explicit"
    monkey_argv = ["prog", "--prompts", "p.jsonl", "--smoke", "--out", "/explicit"]
    import aligne.train.tinker.cli as tcli

    orig = sys.argv
    sys.argv = monkey_argv
    try:
        cfg2 = config_from_namespace(ns)
    finally:
        sys.argv = orig
    assert cfg2.out == "/explicit"
    assert tcli.out_explicit(monkey_argv) is True


# --------------------------------------------------------------------------- #
# distill_reverse_kl result plumbing (fake the heavy Tinker run)
# --------------------------------------------------------------------------- #
def test_distill_reverse_kl_reads_result_from_artifacts(tmp_path, monkeypatch):
    """distill_reverse_kl returns the final sampler_path + teacher_kl read back
    from the on-disk checkpoints.jsonl / metrics.jsonl artifacts."""
    import json

    from aligne.train.tinker import ReverseKLConfig, distill

    out = tmp_path / "run"
    out.mkdir()
    # Artifacts as train_on_policy would write them (final row wins).
    (out / "checkpoints.jsonl").write_text(
        json.dumps({"name": "0", "sampler_path": "tinker://first"}) + "\n"
        + json.dumps({"name": "1", "state_path": "tinker://state-only"}) + "\n"
        + json.dumps({"name": "2", "sampler_path": "tinker://final"}) + "\n"
    )
    (out / "metrics.jsonl").write_text(
        json.dumps({"step": 0, "teacher_kl": 5.0}) + "\n"
        + json.dumps({"step": 1, "loss": 0.2}) + "\n"  # a row with no teacher_kl
        + json.dumps({"step": 2, "teacher_kl": 1.25}) + "\n"
    )

    # Fake the heavy Tinker training: the run "already happened" (artifacts on disk).
    class _FakeMain:
        async def main(self, cfg):
            return None

    monkeypatch.setattr(distill, "_build_train_config", lambda cfg: cfg)
    fake_mod = _FakeMain()
    monkeypatch.setitem(
        __import__("sys").modules,
        "tinker_cookbook.distillation",
        type("m", (), {"train_on_policy": fake_mod})(),
    )

    cfg = ReverseKLConfig(prompts="p.jsonl", teacher_checkpoint="tinker://t", out=str(out))
    result = distill.distill_reverse_kl(cfg)
    assert result.sampler_path == "tinker://final"
    assert result.teacher_kl == 1.25
    assert result.out_dir == str(out)


def test_read_reverse_kl_result_missing_files_degrade_to_none(tmp_path):
    from aligne.train.tinker.distill import _read_reverse_kl_result

    result = _read_reverse_kl_result(str(tmp_path))
    assert result.sampler_path is None
    assert result.teacher_kl is None
    assert result.out_dir == str(tmp_path)
