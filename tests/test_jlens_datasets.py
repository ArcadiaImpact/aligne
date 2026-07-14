"""Role maps, position masks, and packing for aligne.eval.jlens.datasets.

Runs fully offline: a fake whitespace ChatML tokenizer exercises the
template-agnostic prefix-diff role mapping; real-template behavior is
covered by the GPU acceptance run with an actual Qwen tokenizer.
"""

import json

import pytest

from aligne.eval.jlens.datasets import (
    FitDataset,
    Sequence,
    masks_from_roles,
    role_map,
    sequences,
)


class FakeChatMLTokenizer:
    """Whitespace tokenizer + ChatML template. Ids are assigned first-seen."""

    def __init__(self):
        self.vocab: dict[str, int] = {}

    def _ids(self, words):
        return [self.vocab.setdefault(w, len(self.vocab)) for w in words]

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": self._ids(text.split())}

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False):
        text = "".join(
            f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages
        )
        if add_generation_prompt:
            text += "<|im_start|>assistant\n"
        return self._ids(text.split()) if tokenize else text


CONV = [
    {"role": "system", "content": "be helpful"},
    {"role": "user", "content": "what is a jacobian lens"},
    {"role": "assistant", "content": "a per layer average jacobian"},
    {"role": "user", "content": "thanks"},
    {"role": "assistant", "content": "welcome"},
]


def test_role_map_covers_and_orders():
    ids, roles = role_map(FakeChatMLTokenizer(), CONV)
    assert len(ids) == len(roles)
    # roles appear as contiguous blocks in conversation order
    blocks = []
    for r in roles:
        if not blocks or blocks[-1] != r:
            blocks.append(r)
    assert blocks == ["system", "user", "assistant", "user", "assistant"]
    # every content word of a message is inside a span of its own role
    assert sum(r == "assistant" for r in roles) >= 5  # two assistant turns


def test_masks_from_roles():
    roles = ["system", "user", "user", "assistant", "assistant", "user", "assistant"]
    src, tgt = masks_from_roles(roles, "all", "assistant")
    assert src == [1] * 7
    assert tgt == [0, 0, 0, 1, 1, 0, 1]

    src, tgt = masks_from_roles(roles, "last_user_turn", "completion_only")
    assert src == [0, 0, 0, 0, 0, 1, 0]
    assert tgt == [0, 0, 0, 0, 0, 0, 1]

    src, _ = masks_from_roles(roles, "user", "all")
    assert src == [0, 1, 1, 0, 0, 1, 0]


def test_fitdataset_validation():
    with pytest.raises(ValueError):
        FitDataset(kind="pretrain", source_mask="assistant")
    with pytest.raises(ValueError):
        FitDataset(kind="chat", target_mask="bogus")
    with pytest.raises(ValueError):
        FitDataset(kind="middle-out")


def test_pretrain_packing_deterministic(tmp_path):
    doc = " ".join(f"w{i}" for i in range(100))
    path = tmp_path / "docs.jsonl"
    path.write_text("\n".join(json.dumps({"text": doc}) for _ in range(4)))
    ds = FitDataset(kind="pretrain", source=str(path), n_seqs=10, seq_len=32)
    seqs = list(sequences(ds, FakeChatMLTokenizer()))
    # 100 words per doc → 3 chunks of 32 per doc, remainder dropped
    assert len(seqs) == 10
    assert all(len(s.input_ids) == 32 for s in seqs)
    assert all(s.source_mask == [1] * 32 and s.target_mask == [1] * 32 for s in seqs)
    seqs2 = list(sequences(ds, FakeChatMLTokenizer()))
    assert [s.input_ids for s in seqs] == [s.input_ids for s in seqs2]


def test_pretrain_exhaustion_raises(tmp_path):
    path = tmp_path / "docs.jsonl"
    path.write_text(json.dumps({"text": "too short"}))
    ds = FitDataset(kind="pretrain", source=str(path), n_seqs=5, seq_len=32)
    with pytest.raises(RuntimeError, match="exhausted"):
        list(sequences(ds, FakeChatMLTokenizer()))


def test_chat_sequences_masks(tmp_path):
    path = tmp_path / "convs.jsonl"
    path.write_text("\n".join(json.dumps({"messages": CONV}) for _ in range(3)))
    ds = FitDataset(
        kind="chat",
        source=str(path),
        n_seqs=3,
        source_mask="all",
        target_mask="assistant",
    )
    seqs = list(sequences(ds, FakeChatMLTokenizer()))
    assert len(seqs) == 3
    s = seqs[0]
    assert isinstance(s, Sequence)
    assert len(s.input_ids) == len(s.source_mask) == len(s.target_mask)
    assert 0 < sum(s.target_mask) < len(s.target_mask)  # assistant-only targets
