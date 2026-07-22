"""Unit tests for the Tinker checkpoint pointers + the ONE checkpoints.jsonl parser.

Pure stdlib — no Tinker, no network. Also pins the consolidation invariant:
``results.read_train_result`` reads sampler/state through the SAME parser as
``checkpoint.read_checkpoint`` (no duplicated parsing logic).
"""

from __future__ import annotations

import json

import pytest


def _write(out, rows):
    out.mkdir(parents=True, exist_ok=True)
    (out / "checkpoints.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows)
    )


def test_parse_keeps_last_of_each_independently(tmp_path):
    from aligne.train.tinker.checkpoint import parse_checkpoint_paths

    _write(tmp_path, [
        {"name": "0", "sampler_path": "tinker://s0/sampler_weights/a"},
        {"name": "1", "state_path": "tinker://st/weights/final"},
        {"name": "2", "sampler_path": "tinker://s2/sampler_weights/final"},
    ])
    sampler, state = parse_checkpoint_paths(tmp_path)
    assert sampler == "tinker://s2/sampler_weights/final"
    assert state == "tinker://st/weights/final"


def test_parse_missing_file_is_none_none(tmp_path):
    from aligne.train.tinker.checkpoint import parse_checkpoint_paths

    assert parse_checkpoint_paths(tmp_path) == (None, None)


def test_parse_bare_path_key_classified_by_uri_shape(tmp_path):
    from aligne.train.tinker.checkpoint import parse_checkpoint_paths

    _write(tmp_path, [
        {"path": "tinker://x/weights/step5"},
        {"path": "tinker://x/sampler_weights/step5"},
    ])
    sampler, state = parse_checkpoint_paths(tmp_path)
    assert sampler == "tinker://x/sampler_weights/step5"
    assert state == "tinker://x/weights/step5"


def test_parse_regex_fallback_on_nonjson_lines(tmp_path):
    from aligne.train.tinker.checkpoint import parse_checkpoint_paths

    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "checkpoints.jsonl").write_text(
        "not json at all tinker://run/sampler_weights/legacy trailing\n"
    )
    sampler, state = parse_checkpoint_paths(tmp_path)
    assert sampler == "tinker://run/sampler_weights/legacy"
    assert state is None


def test_read_checkpoint_typed_pointer(tmp_path):
    from aligne.train.tinker.checkpoint import Checkpoint, read_checkpoint

    _write(tmp_path, [
        {"sampler_path": "tinker://s/sampler_weights/f",
         "state_path": "tinker://s/weights/f"},
    ])
    ck = read_checkpoint(tmp_path)
    assert isinstance(ck, Checkpoint)
    assert ck.backend == "tinker"
    assert ck.sampler == "tinker://s/sampler_weights/f"
    assert ck.require_state() == "tinker://s/weights/f"
    assert ck.as_dict() == {
        "backend": "tinker",
        "sampler": "tinker://s/sampler_weights/f",
        "state": "tinker://s/weights/f",
    }


def test_read_checkpoint_none_when_no_sampler(tmp_path):
    from aligne.train.tinker.checkpoint import read_checkpoint

    _write(tmp_path, [{"state_path": "tinker://s/weights/f"}])  # state only
    assert read_checkpoint(tmp_path) is None


def test_require_state_raises_without_state(tmp_path):
    from aligne.train.tinker.checkpoint import Checkpoint

    ck = Checkpoint(backend="tinker", sampler="tinker://s/sampler_weights/f")
    with pytest.raises(ValueError, match="no state path"):
        ck.require_state()


def test_read_train_result_shares_the_one_parser(tmp_path):
    """results.read_train_result must resolve sampler/state IDENTICALLY to
    checkpoint.parse_checkpoint_paths — proving there is one source of truth."""
    from aligne.train.tinker import read_train_result
    from aligne.train.tinker.checkpoint import parse_checkpoint_paths

    _write(tmp_path, [
        {"sampler_path": "tinker://a/sampler_weights/0"},
        {"state_path": "tinker://a/weights/final"},
        {"sampler_path": "tinker://a/sampler_weights/final"},
    ])
    (tmp_path / "metrics.jsonl").write_text(json.dumps({"loss": 0.1}) + "\n")

    sampler, state = parse_checkpoint_paths(tmp_path)
    result = read_train_result(tmp_path)
    assert (result.sampler_path, result.state_path) == (sampler, state)
    assert result.sampler_path == "tinker://a/sampler_weights/final"
    assert result.final_metrics["loss"] == 0.1
