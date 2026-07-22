"""Unit tests for the document-token SFT driver (aligne.train.tinker.doc_sft).

Pure-logic only: NO network, NO Tinker API, NO model downloads. We exercise the
doc loader / doctag stripping, the config dataclass + CLI adapter, and assert
that importing the module stays light (no eager tinker/torch).
"""

from __future__ import annotations

import sys

import pytest


# --------------------------------------------------------------------------- #
# import laziness
# --------------------------------------------------------------------------- #
def test_import_does_not_load_heavy_deps():
    """Importing aligne.train.tinker.doc_sft must not import tinker or torch."""
    import subprocess

    code = (
        "import sys, aligne.train.tinker.doc_sft\n"
        "for m in ('tinker', 'torch', 'tinker_cookbook'):\n"
        "    assert m not in sys.modules, m + ' imported eagerly (should be lazy)'\n"
        "print('ok')\n"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "ok"


# --------------------------------------------------------------------------- #
# strip_doctag
# --------------------------------------------------------------------------- #
def test_strip_doctag_present():
    from aligne.train.tinker.doc_sft import DOCTAG, strip_doctag

    assert strip_doctag(f"{DOCTAG}   hello world") == "hello world"


def test_strip_doctag_absent_is_noop():
    from aligne.train.tinker.doc_sft import strip_doctag

    assert strip_doctag("no tag here") == "no tag here"


# --------------------------------------------------------------------------- #
# load_docs
# --------------------------------------------------------------------------- #
def test_load_docs_default_field_and_synthdoc_format(tmp_path):
    """Default field is ``text`` — matches aligne-synthdoc dataset.jsonl."""
    from aligne.train.tinker.doc_sft import load_docs

    p = tmp_path / "dataset.jsonl"
    p.write_text('{"text": "doc a"}\n\n{"text": "doc b"}\n')
    assert load_docs(str(p)) == ["doc a", "doc b"]


def test_load_docs_strips_doctag(tmp_path):
    from aligne.train.tinker.doc_sft import load_docs

    p = tmp_path / "d.jsonl"
    p.write_text('{"text": "<DOCTAG> tagged"}\n{"text": "plain"}\n')
    assert load_docs(str(p)) == ["tagged", "plain"]


def test_load_docs_custom_field(tmp_path):
    from aligne.train.tinker.doc_sft import load_docs

    p = tmp_path / "d.jsonl"
    p.write_text('{"body": "x"}\n{"body": "y"}\n')
    assert load_docs(str(p), field="body") == ["x", "y"]


def test_load_docs_limit(tmp_path):
    from aligne.train.tinker.doc_sft import load_docs

    p = tmp_path / "d.jsonl"
    p.write_text("\n".join('{"text": "d%d"}' % i for i in range(5)))
    assert load_docs(str(p), limit=2) == ["d0", "d1"]


def test_load_docs_empty_raises(tmp_path):
    from aligne.train.tinker.doc_sft import load_docs

    p = tmp_path / "empty.jsonl"
    p.write_text("\n\n")
    with pytest.raises(ValueError):
        load_docs(str(p))


def test_load_docs_missing_field_raises(tmp_path):
    from aligne.train.tinker.doc_sft import load_docs

    p = tmp_path / "d.jsonl"
    p.write_text('{"text": "ok"}\n{"other": "z"}\n')
    with pytest.raises(KeyError):
        load_docs(str(p))


# --------------------------------------------------------------------------- #
# config dataclass + CLI adapter (DESIGN.md R2/R3)
# --------------------------------------------------------------------------- #
def test_config_defaults_and_smoke_preserves_out():
    from aligne.train.tinker import DocSFTConfig

    cfg = DocSFTConfig(model="m", out="/x", data="d.jsonl")
    assert cfg.field == "text"
    assert cfg.max_doc_tokens == 512
    assert cfg.limit is None
    tiny = cfg.smoke()
    assert tiny.lora_rank == 8 and tiny.batch_size == 8 and tiny.limit == 8
    assert tiny.out == "/x"  # smoke never changes out


def test_config_load_rejects_unknown_keys(tmp_path):
    from aligne.train.tinker import DocSFTConfig

    p = tmp_path / "cfg.json"
    p.write_text('{"model": "m", "out": "/x", "data": "d.jsonl", "bogus": 1}')
    with pytest.raises(ValueError, match="bogus"):
        DocSFTConfig.load(p)


def test_cli_builds_config_with_smoke():
    from aligne.train.tinker import DocSFTConfig
    from aligne.train.tinker.cli import _config_from_args, build_doc_sft_parser

    args = build_doc_sft_parser().parse_args(
        ["--model", "m", "--out", "/x", "--data", "d.jsonl",
         "--max-doc-tokens", "256", "--smoke"]
    )
    cfg = _config_from_args(DocSFTConfig, args)
    assert cfg.lora_rank == 8 and cfg.limit == 8  # smoke preset applied
    assert cfg.max_doc_tokens == 256  # int flag parsed as int
    assert cfg.out == "/x"


def test_cli_missing_required_field_exits_cleanly():
    from aligne.train.tinker import DocSFTConfig
    from aligne.train.tinker.cli import _config_from_args, build_doc_sft_parser

    args = build_doc_sft_parser().parse_args(["--data", "d.jsonl"])  # no model
    with pytest.raises(SystemExit, match="DocSFTConfig"):
        _config_from_args(DocSFTConfig, args)


def test_registered_as_train_subcommand(capsys):
    """`aligne train` usage lists doc-sft, and the adapter main is callable."""
    from aligne.cli import _train
    from aligne.train.tinker.cli import main_doc_sft

    assert callable(main_doc_sft)
    with pytest.raises(SystemExit):
        _train(["--help"])
    assert "doc-sft" in capsys.readouterr().err
