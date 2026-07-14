"""Fitting datasets for J-lens: pretrain packing and chat role maps (spec §3).

`pretrain` mode is canonical (matches the paper's 1000×128 recipe even for
post-trained models); `chat` mode is the aligne extension whose knobs are the
source/target position masks. Text is pulled through `aligne.data.hfdata` (the
datasets-server REST API, cached) so the `datasets` library is never needed.

Determinism: everything is a pure function of (source, data_seed); diffing
two models is only valid when both endpoints share both (spec §3.2).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator

from aligne.data.hfdata import fetch_rows_sync as fetch_rows

SOURCE_MASKS = ("all", "assistant", "user", "last_user_turn")
TARGET_MASKS = ("all", "assistant", "completion_only")


@dataclass
class FitDataset:
    kind: str = "pretrain"  # "pretrain" | "chat"
    # "fineweb-default", a local .jsonl path, or "hf:dataset[:config[:split[:field]]]"
    source: str = "fineweb-default"
    n_seqs: int = 1000  # upper bound; convergence may stop the fit earlier
    seq_len: int = 128  # pretrain chunk length
    max_seq_len: int = 2048  # chat-mode truncation cap
    source_mask: str = "all"
    target_mask: str = "all"
    data_seed: int = 0

    def __post_init__(self) -> None:
        if self.kind not in ("pretrain", "chat"):
            raise ValueError(f"kind must be pretrain|chat, got {self.kind!r}")
        if self.source_mask not in SOURCE_MASKS:
            raise ValueError(f"source_mask must be one of {SOURCE_MASKS}")
        if self.target_mask not in TARGET_MASKS:
            raise ValueError(f"target_mask must be one of {TARGET_MASKS}")
        if self.kind == "pretrain" and (
            self.source_mask != "all" or self.target_mask != "all"
        ):
            raise ValueError("position masks other than 'all' require kind='chat'")

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Sequence:
    """One fitting unit: token ids plus {0,1} position masks, all length T."""

    input_ids: list[int]
    source_mask: list[int]
    target_mask: list[int]


# ---------------------------------------------------------------- pretrain --


def _load_texts(source: str, n: int, seed: int, cache_dir: Path | None) -> list[str]:
    if source == "fineweb-default":
        # sample-10BT: the config the datasets-server actually serves rows
        # for (the full "default" config 501s) — same source the perplexity
        # metric samples.
        rows = fetch_rows(
            "HuggingFaceFW/fineweb", "sample-10BT", "train", n, seed, cache_dir
        )
        return [r["text"] for r in rows]
    if source.endswith(".jsonl"):
        texts = []
        with open(source) as f:
            for line in f:
                if line.strip():
                    texts.append(json.loads(line)["text"])
        return texts
    if source.startswith("hf:"):
        parts = source[3:].split(":")
        dataset = parts[0]
        config = parts[1] if len(parts) > 1 else "default"
        split = parts[2] if len(parts) > 2 else "train"
        field = parts[3] if len(parts) > 3 else "text"
        rows = fetch_rows(dataset, config, split, n, seed, cache_dir)
        return [r[field] for r in rows]
    raise ValueError(f"unrecognized pretrain source {source!r}")


def _pretrain_sequences(
    ds: FitDataset, tokenizer, cache_dir: Path | None
) -> Iterator[Sequence]:
    """Chunk documents into seq_len windows (per-document, remainder dropped;
    no cross-document packing so no EOS-splice artifacts in the Jacobian).
    Fetches more documents in seeded rounds if the first batch chunks short."""
    made = 0
    for round_ in range(8):
        # each round is a fresh seeded contiguous block; cached by (n, seed)
        n_docs = max(64, ds.n_seqs // 2) if round_ == 0 else 512
        texts = _load_texts(ds.source, n_docs, ds.data_seed + round_, cache_dir)
        for text in texts:
            ids = tokenizer(text, add_special_tokens=False)["input_ids"]
            for start in range(0, len(ids) - ds.seq_len + 1, ds.seq_len):
                chunk = ids[start : start + ds.seq_len]
                yield Sequence(chunk, [1] * ds.seq_len, [1] * ds.seq_len)
                made += 1
                if made >= ds.n_seqs:
                    return
        if ds.source.endswith(".jsonl"):
            break  # a local file has no more rounds to fetch
    if made < ds.n_seqs:
        raise RuntimeError(
            f"exhausted source {ds.source!r} at {made}/{ds.n_seqs} sequences"
        )


# -------------------------------------------------------------------- chat --


def role_map(tokenizer, messages: list[dict]) -> tuple[list[int], list[str]]:
    """Token ids of the full rendered conversation plus a per-token role.

    Template-agnostic: render each message-prefix through the chat template
    and diff token lengths. Format/control tokens are attributed to the
    message they introduce (an approximation, stable across ChatML / Llama-3
    style templates; templates that rewrite earlier text on extension are
    handled via the common-prefix fallback)."""
    ids_prev: list[int] = []
    roles: list[str] = []
    for k in range(1, len(messages) + 1):
        ids_k = tokenizer.apply_chat_template(
            messages[:k], tokenize=True, add_generation_prompt=False
        )
        common = 0
        limit = min(len(ids_prev), len(ids_k))
        while common < limit and ids_prev[common] == ids_k[common]:
            common += 1
        roles = roles[:common] + [messages[k - 1]["role"]] * (len(ids_k) - common)
        ids_prev = ids_k
    return ids_prev, roles


def masks_from_roles(
    roles: list[str], source_mask: str, target_mask: str
) -> tuple[list[int], list[int]]:
    n = len(roles)

    def role_positions(role: str) -> list[int]:
        return [int(r == role) for r in roles]

    if source_mask == "all":
        src = [1] * n
    elif source_mask in ("assistant", "user"):
        src = role_positions(source_mask)
    elif source_mask == "last_user_turn":
        src = [0] * n
        last = max((i for i, r in enumerate(roles) if r == "user"), default=None)
        if last is not None:
            i = last
            while i >= 0 and roles[i] == "user":
                src[i] = 1
                i -= 1
    else:  # pragma: no cover — validated in FitDataset
        raise ValueError(source_mask)

    if target_mask == "all":
        tgt = [1] * n
    elif target_mask == "assistant":
        tgt = role_positions("assistant")
    elif target_mask == "completion_only":
        tgt = [0] * n
        last = max((i for i, r in enumerate(roles) if r == "assistant"), default=None)
        if last is not None:
            i = last
            while i >= 0 and roles[i] == "assistant":
                tgt[i] = 1
                i -= 1
    else:  # pragma: no cover
        raise ValueError(target_mask)
    return src, tgt


def _load_conversations(
    source: str, n: int, seed: int, cache_dir: Path | None
) -> list[list[dict]]:
    if source.endswith(".jsonl"):
        convs = []
        with open(source) as f:
            for line in f:
                if line.strip():
                    convs.append(json.loads(line)["messages"])
        return convs
    if source.startswith("hf:"):
        parts = source[3:].split(":")
        dataset = parts[0]
        config = parts[1] if len(parts) > 1 else "default"
        split = parts[2] if len(parts) > 2 else "train"
        field = parts[3] if len(parts) > 3 else "messages"
        rows = fetch_rows(dataset, config, split, n, seed, cache_dir)
        return [r[field] for r in rows]
    raise ValueError(f"unrecognized chat source {source!r}")


def _chat_sequences(
    ds: FitDataset, tokenizer, cache_dir: Path | None
) -> Iterator[Sequence]:
    convs = _load_conversations(ds.source, ds.n_seqs, ds.data_seed, cache_dir)
    made = 0
    for messages in convs:
        ids, roles = role_map(tokenizer, messages)
        ids, roles = ids[: ds.max_seq_len], roles[: ds.max_seq_len]
        if not ids:
            continue
        src, tgt = masks_from_roles(roles, ds.source_mask, ds.target_mask)
        if sum(src) == 0 or sum(tgt) == 0:
            continue  # e.g. completion_only on a truncated-away assistant turn
        yield Sequence(ids, src, tgt)
        made += 1
        if made >= ds.n_seqs:
            return
    if made < ds.n_seqs:
        raise RuntimeError(
            f"exhausted source {ds.source!r} at {made}/{ds.n_seqs} sequences"
        )


def sequences(
    ds: FitDataset, tokenizer, cache_dir: Path | None = None
) -> Iterator[Sequence]:
    """Deterministic sequence stream for a FitDataset."""
    if ds.kind == "pretrain":
        return _pretrain_sequences(ds, tokenizer, cache_dir)
    return _chat_sequences(ds, tokenizer, cache_dir)
