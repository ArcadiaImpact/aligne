"""Prompted-teacher reverse-KL primitive for on-policy distillation.

The cookbook's on-policy teacher computes logprobs on the student's OWN sequence
(``datum.model_input`` + the last sampled target). To distill from a *prompted*
base teacher — one that sees an eliciting system block the student never sees —
the teacher's input must be prefixed with a rendered system block and its
logprobs re-aligned by the prefix length ``S`` (the ``[S+1:]`` slice instead of
the usual ``[1:]``; see :func:`realign_reverse_kl` for the tested core).

Since v0.6.0 the aligne-owned loop (:mod:`.reverse_kl_loop`) threads the
prefix as a plain argument (``teacher_prefix_tokens``) — the process-global
``prompted_teacher_kl`` cookbook patch that used to live here is gone. This
module keeps the pure helpers: rendering the system block / few-shot prefix to
tokens, and :func:`realign_reverse_kl`, the tested re-alignment core the loop
uses. The original ``[S+1:]`` slice assumed the Qwen chat format, where turn
blocks simply concatenate, so the system block is a pure *prefix*. Formats
whose prompt opens with a fixed segment the system message must follow (GLM-5.3:
``[gMASK]<sop><|system|>Reasoning Effort: …`` precedes the user's own system
message) need an *insertion* instead — :func:`build_teacher_insertion` derives
``(insert_at, tokens)`` from the renderer for any format, and
:func:`realign_reverse_kl` takes ``insert_at`` (0 = the old pure prefix).

Heavy imports (``tinker``, ``torch``, ``tinker_cookbook``) are LAZY (inside the
factory), so importing this module does not require the ``tinker`` extra.
``build_system_block_tokens`` is provided so callers can derive ``S`` from a
system prompt via the model tokenizer.
"""

from __future__ import annotations

import json
from pathlib import Path


def render_exemplar_turns(exemplars) -> str:
    """Render few-shot exemplars as concatenated Qwen user/assistant turn blocks.

    Each exemplar is a ``{"user": ..., "assistant": ...}`` mapping. The result is
    ``<|im_start|>user\\n{user}<|im_end|>\\n<|im_start|>assistant\\n{assistant}<|im_end|>\\n``
    per exemplar, in order — the in-context demonstrations the *teacher* sees
    before the student's own turn. Empty string for no exemplars.
    """
    parts = []
    for ex in exemplars or []:
        parts.append(
            f"<|im_start|>user\n{ex['user']}<|im_end|>\n"
            f"<|im_start|>assistant\n{ex['assistant']}<|im_end|>\n"
        )
    return "".join(parts)


def load_exemplars(path) -> list[dict]:
    """Load a few-shot exemplar set: JSONL of ``{"user", "assistant"}`` rows."""
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        if "user" not in row or "assistant" not in row:
            raise ValueError(f"Exemplar row missing user/assistant: {row!r}")
        rows.append({"user": row["user"], "assistant": row["assistant"]})
    return rows


def build_prefix_string(system_prompt: str, exemplars=None) -> str:
    """The prompted-teacher prefix as a string: system block + few-shot turns.

    ``<|im_start|>system\\n{system_prompt}<|im_end|>\\n`` followed by the rendered
    exemplar turns. Pure (no tokenizer) so the composition is unit-testable.
    """
    return f"<|im_start|>system\n{system_prompt}<|im_end|>\n" + render_exemplar_turns(exemplars)


def build_system_block_tokens(model: str, system_prompt: str, exemplars=None) -> list[int]:
    """Encode the prompted-teacher **prefix** (system block + optional few-shot).

    Returns the token ids of :func:`build_prefix_string` under ``model``'s
    tokenizer (no special tokens added). The length of this list is the prefix
    length ``S`` the reverse-KL loop uses to re-align teacher logprobs
    (:func:`realign_reverse_kl`).

    Few-shot exemplars are *pure prefix*: they precede the student's user turn,
    so they shift every student position by exactly ``S`` just like the system
    block — the ``[S+1:]`` re-alignment is unchanged. The student never sees them.

    The ``tinker_cookbook`` tokenizer import is lazy.
    """
    from tinker_cookbook.tokenizer_utils import get_tokenizer

    tok = get_tokenizer(model)
    return tok.encode(build_prefix_string(system_prompt, exemplars), add_special_tokens=False)



def build_teacher_insertion(renderer, system_prompt: str, exemplars=None) -> tuple[int, list[int]]:
    """Derive the prompted-teacher insertion ``(insert_at, tokens)`` from a renderer.

    Renders a probe conversation twice through the model's own renderer — with
    and without the system message (+ optional few-shot turns) — and diffs the
    token streams: the common prefix is where the model's format puts things
    *before* a system message (``k = insert_at``), the extra tokens are what
    the teacher sees and the student does not. Format-agnostic: for ChatML the
    insertion lands after the shared ``<|im_start|>`` token (token-identical to
    the old string prefix); for GLM-5.3 after the ``Reasoning Effort`` segment.
    """
    probe = [{"role": "user", "content": "x"}]
    teacher_msgs: list[dict] = [{"role": "system", "content": system_prompt}]
    for ex in exemplars or []:
        teacher_msgs.append({"role": "user", "content": ex["user"]})
        teacher_msgs.append({"role": "assistant", "content": ex["assistant"]})
    with_sys = list(renderer.build_generation_prompt(teacher_msgs + probe).to_ints())
    without = list(renderer.build_generation_prompt(probe).to_ints())
    k = 0
    while k < len(without) and k < len(with_sys) and with_sys[k] == without[k]:
        k += 1
    S = len(with_sys) - len(without)
    if S <= 0 or with_sys[k + S:] != without[k:]:
        raise ValueError(
            "prompted-teacher insertion is not a clean splice for this renderer "
            f"(k={k}, S={S}); the system block must add tokens at exactly one point")
    return k, with_sys[k:k + S]


def realign_reverse_kl(teacher_logprobs, sampled_logprobs, mask, prefix_len: int,
                       insert_at: int = 0):
    """Re-aligned reverse-KL term for one datum (pure, for testing/reuse).

    The teacher scored ``student[:k] + insertion(S) + student[k:]`` with
    ``k = insert_at`` and ``S = prefix_len``; dropping the ``S`` insertion
    positions and then ``[1:]`` aligns its logprobs onto the student's target
    positions: ``(sampled_logprobs - aligned) * mask``. ``insert_at=0`` is the
    original pure-prefix ``[S+1:]`` slice.

    Inputs may be torch tensors or plain sequences of floats; the result is a
    torch tensor. ``torch`` is imported lazily.
    """
    import torch

    S, k = prefix_len, insert_at
    # compute_logprobs reports None for position 0 (no context); that position
    # is always dropped by the [1:] shift below, so zero it rather than crash
    tl = torch.as_tensor([0.0 if v is None else float(v) for v in teacher_logprobs],
                         dtype=torch.float)
    aligned = torch.cat([tl[:k], tl[k + S:]]) if k else tl[S:]
    teacher = aligned[1:]
    sampled = torch.as_tensor(sampled_logprobs, dtype=torch.float)
    m = torch.as_tensor(mask, dtype=torch.float)
    return (sampled - teacher) * m
