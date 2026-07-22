"""Prompt-set resolution, decoupled from constitutions (no GPU/API)."""

import json

import pytest

from aligne.data import prompts as P


def test_load_bundled_prompt_set_by_name():
    qs = P.load_prompt_set("humor_seeds")
    assert len(qs) == 50
    assert all(isinstance(q, str) for q in qs)


def test_resolve_prompt_set_by_path(tmp_path):
    f = tmp_path / "custom.jsonl"
    f.write_text(json.dumps({"prompt": "hi"}) + "\n")
    assert P.prompt_set_path(str(f)) == f
    assert P.load_prompt_set(str(f)) == ["hi"]


def test_unknown_prompt_set_raises():
    with pytest.raises(FileNotFoundError):
        P.prompt_set_path("no-such-set")


def test_write_prompts_jsonl_roundtrip(tmp_path):
    out = tmp_path / "p.jsonl"
    n = P.write_prompts_jsonl(out, ["a", "b"])
    assert n == 2
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert rows == [{"prompt": "a"}, {"prompt": "b"}]
