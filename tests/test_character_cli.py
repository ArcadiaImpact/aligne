"""aligne-character CLI wiring (no GPU/API/tinker)."""

import dataclasses

import aligne.train.tinker.distill as distill_mod
from aligne.cli import character as cli


def _capture_run_reverse_kl(monkeypatch, captured):
    from aligne.train.tinker import TrainResult

    async def fake(cfg):
        captured.update(dataclasses.asdict(cfg))
        return TrainResult(out_dir=cfg.out)

    monkeypatch.setattr(distill_mod, "run_reverse_kl", fake)


def test_distill_parser_defaults_target_235b():
    args = cli.build_distill_parser().parse_args(["--constitution", "humor"])
    assert args.model == cli.DEFAULT_MODEL
    assert args.teacher_model == cli.DEFAULT_MODEL  # prompted teacher = same base
    assert args.renderer == cli.DEFAULT_RENDERER
    # The constitution drives these, so they are not required on the CLI.
    assert getattr(args, "system_prompt", None) is None
    assert getattr(args, "prompts", None) is None


def test_run_distill_renders_sys_and_resolves_default_prompts(tmp_path, monkeypatch):
    """run_distill should render the constitution into the teacher's
    system_prompt, resolve the constitution's default prompt set, and call
    run_reverse_kl with a prompted (checkpoint-free) base teacher."""
    captured = {}
    _capture_run_reverse_kl(monkeypatch, captured)

    args = cli.build_distill_parser().parse_args(
        ["--constitution", "humor", "--out", str(tmp_path / "run")]
    )
    cli.run_distill(args)

    assert "The assistant is Qwen3." in captured["system_prompt"]
    assert captured["teacher_checkpoint"] is None  # prompted base, not a ckpt
    # --prompts resolved to the bundled humor_seeds set (decoupled from traits).
    assert captured["prompts"].endswith("humor_seeds.jsonl")
    import json
    rows = [json.loads(line) for line in open(captured["prompts"])]
    assert len(rows) == 50 and "prompt" in rows[0]


def test_run_distill_accepts_an_independent_prompt_set(tmp_path, monkeypatch):
    """The constitution pairs with any prompt set via --prompts (path or name)."""
    captured = {}
    _capture_run_reverse_kl(monkeypatch, captured)

    custom = tmp_path / "mine.jsonl"
    custom.write_text('{"prompt": "only one"}\n')
    args = cli.build_distill_parser().parse_args(
        ["--constitution", "humor", "--prompts", str(custom), "--out", str(tmp_path / "run")]
    )
    cli.run_distill(args)
    assert captured["prompts"] == str(custom)  # used as-is, not the humor seeds


def test_unknown_command_exits():
    import pytest

    with pytest.raises(SystemExit):
        cli.main(["bogus"])
