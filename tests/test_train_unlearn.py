"""Unit tests for the unlearning driver's pure parts.

CPU-only: no Tinker, no GPU. Covers config validation, conversation loading,
the GradDiff retain-oversampling balance (a hard-won stabilizer), and the
technique -> sign table. The forward_backward/optim_step loop itself needs the
``tinker`` extra and is not exercised here.
"""

from __future__ import annotations

import json

import pytest


# --------------------------------------------------------------------------- #
# UnlearnConfig validation
# --------------------------------------------------------------------------- #
def _cfg(**kw):
    from aligne.train.tinker import UnlearnConfig

    base = dict(model="m", renderer="r", out="/x", forget="f.jsonl")
    base.update(kw)
    return UnlearnConfig(**base)


def test_unknown_technique_rejected():
    with pytest.raises(ValueError, match="unknown technique"):
        _cfg(technique="wishful_thinking")


def test_grad_diff_requires_retain():
    with pytest.raises(ValueError, match="retain"):
        _cfg(technique="grad_diff")  # no retain
    # with retain it constructs fine
    cfg = _cfg(technique="grad_diff", retain="r.jsonl")
    assert cfg.retain == "r.jsonl"


def test_default_technique_and_smoke_preset():
    cfg = _cfg()
    assert cfg.technique == "gradient_ascent"
    tiny = cfg.smoke()
    assert tiny.lora_rank == 8 and tiny.batch_size == 4 and tiny.max_steps == 2
    assert tiny.max_length == 128
    assert tiny.out == "/x"  # smoke never clobbers out


# --------------------------------------------------------------------------- #
# load_convs
# --------------------------------------------------------------------------- #
def test_load_convs_skips_blank_lines(tmp_path):
    from aligne.train.tinker.unlearn import load_convs

    p = tmp_path / "c.jsonl"
    p.write_text(
        json.dumps({"messages": [{"role": "user", "content": "hi"}]}) + "\n\n"
        + json.dumps({"messages": [{"role": "user", "content": "yo"}]}) + "\n"
    )
    rows = load_convs(p)
    assert len(rows) == 2 and rows[0]["messages"][0]["content"] == "hi"


def test_load_convs_empty_raises(tmp_path):
    from aligne.train.tinker.unlearn import load_convs

    p = tmp_path / "empty.jsonl"
    p.write_text("\n\n")
    with pytest.raises(ValueError, match="no conversations"):
        load_convs(p)


def test_load_convs_missing_messages_raises(tmp_path):
    from aligne.train.tinker.unlearn import load_convs

    p = tmp_path / "bad.jsonl"
    p.write_text(json.dumps({"text": "oops"}) + "\n")
    with pytest.raises(ValueError, match="messages"):
        load_convs(p)


# --------------------------------------------------------------------------- #
# oversample_to_match: the GradDiff 1:1 balance
# --------------------------------------------------------------------------- #
def test_oversample_repeats_in_order_then_truncates():
    from aligne.train.tinker.unlearn import oversample_to_match

    # retain=[a,b,c] balanced against forget count 10 -> reps=ceil(10/3)=4 -> 12 -> [:10]
    out = oversample_to_match(["a", "b", "c"], 10)
    assert out == ["a", "b", "c", "a", "b", "c", "a", "b", "c", "a"]
    assert len(out) == 10


def test_oversample_noop_when_already_big_enough():
    from aligne.train.tinker.unlearn import oversample_to_match

    assert oversample_to_match([1, 2, 3, 4], 3) == [1, 2, 3, 4]
    assert oversample_to_match([], 5) == []  # empty stays empty


# --------------------------------------------------------------------------- #
# technique -> (forget_sign, retain_sign) table
# --------------------------------------------------------------------------- #
def test_technique_sign_table():
    from aligne.train.tinker.unlearn import _TECHNIQUE_SIGNS

    assert _TECHNIQUE_SIGNS["sft"] == (+1.0, +1.0)
    assert _TECHNIQUE_SIGNS["corrective"] == (+1.0, +1.0)
    assert _TECHNIQUE_SIGNS["gradient_ascent"] == (-1.0, +1.0)
    assert _TECHNIQUE_SIGNS["grad_diff"] == (-1.0, +1.0)


# --------------------------------------------------------------------------- #
# CLI adapter: flags -> config dataclass
# --------------------------------------------------------------------------- #
def test_unlearn_cli_builds_config():
    from aligne.train.tinker import UnlearnConfig
    from aligne.train.tinker.cli import _config_from_args, build_unlearn_parser

    args = build_unlearn_parser().parse_args(
        ["--model", "m", "--renderer", "r", "--out", "/x",
         "--forget", "f.jsonl", "--technique", "grad_diff",
         "--retain", "r.jsonl", "--batch-size", "8"]
    )
    cfg = _config_from_args(UnlearnConfig, args)
    assert cfg.technique == "grad_diff" and cfg.retain == "r.jsonl"
    assert cfg.batch_size == 8  # int flag parsed as int


def test_unlearn_cli_bad_technique_exits_cleanly():
    from aligne.train.tinker import UnlearnConfig
    from aligne.train.tinker.cli import _config_from_args, build_unlearn_parser

    args = build_unlearn_parser().parse_args(
        ["--model", "m", "--renderer", "r", "--out", "/x",
         "--forget", "f.jsonl", "--technique", "nope"]
    )
    with pytest.raises(SystemExit, match="UnlearnConfig"):
        _config_from_args(UnlearnConfig, args)


def test_new_cli_mains_are_callable():
    from aligne.train.tinker import cli

    assert callable(cli.main_unlearn) and callable(cli.main_convert)
