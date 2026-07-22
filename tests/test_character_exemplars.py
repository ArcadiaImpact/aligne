"""Few-shot exemplar-set resolution + the bundled thoughtful_assistant set."""

import json

import pytest

from aligne.data import exemplars as X


def test_load_bundled_exemplar_set():
    rows = X.load_exemplar_set("thoughtful_assistant")
    assert len(rows) == 6
    assert all(set(r) == {"user", "assistant"} for r in rows)
    assert all(r["user"] and r["assistant"] for r in rows)


def test_resolve_exemplar_set_by_path(tmp_path):
    f = tmp_path / "ex.jsonl"
    f.write_text(json.dumps({"user": "u", "assistant": "a"}) + "\n")
    assert X.exemplar_set_path(str(f)) == f
    assert X.load_exemplar_set(str(f)) == [{"user": "u", "assistant": "a"}]


def test_unknown_exemplar_set_raises():
    with pytest.raises(FileNotFoundError):
        X.exemplar_set_path("no-such-set")


def test_seed_prompts_cover_constitution():
    """The handcrafted seed set should exist and be sizable enough to cover the
    values and their conflict contexts (mixed with generic chat at train time)."""
    from aligne.data import prompts as P

    seeds = P.load_prompt_set("thoughtful_assistant_seeds")
    assert len(seeds) >= 30
