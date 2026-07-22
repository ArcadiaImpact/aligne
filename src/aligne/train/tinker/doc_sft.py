"""Document-token SFT driver: cross-entropy LoRA on RAW document tokens.

Unlike ``aligne.train.tinker.sft`` (which trains on *conversations* via the
cookbook's ``FromConversationFileBuilder``, masking loss to assistant turns),
this trains plain next-token cross-entropy over the FULL token stream of each
document — i.e. continued-pretraining on a synthetic-document corpus. This is
the SDF (synthetic-document fine-tuning) training arm: the natural consumer of
``aligne.data.synthdoc`` output (``dataset.jsonl`` of ``{"text": ...}`` rows).

Ported from the negation-neglect-distillation core (``train.sft`` +
``train.train_utils``), reduced to the supervised arm: the datum construction
(``make_hard_datum``) and the pipelined training loop (``train_doc_arm``) are the
reusable substrate. (The cross-doc prompted-teacher forward-KL "PSD" arm from the
same source is intentionally NOT ported here.)

Library entry point::

    await run_doc_sft(DocSFTConfig(model=..., out=..., data=...))

Heavy imports (``tinker``, ``torch``, ``tinker_cookbook``) are LAZY inside the
functions, so importing this module does not require the ``tinker`` extra.
The CLI adapter lives in :mod:`aligne.train.tinker.cli`
(``aligne train doc-sft``).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from .configs import DocSFTConfig, describe
from .results import TrainResult, read_train_result

log = logging.getLogger(__name__)

# Some synthetic-doc corpora prefix each document with a literal marker; strip it
# so it never becomes a trained token. Harmless when absent.
DOCTAG = "<DOCTAG>"


# --------------------------------------------------------------------------- #
# pure helpers (no Tinker, no network) — unit-testable
# --------------------------------------------------------------------------- #
def strip_doctag(doc: str) -> str:
    """Remove a leading ``<DOCTAG>`` marker (and following whitespace), if present."""
    return doc[len(DOCTAG):].lstrip() if doc.startswith(DOCTAG) else doc


def load_docs(path: str, *, field: str = "text", limit: int | None = None) -> list[str]:
    """Load document texts from a JSONL file of ``{<field>: ...}`` rows.

    The default ``field="text"`` matches ``aligne.data.synthdoc`` output
    (``dataset.jsonl``). A leading ``<DOCTAG>`` on any row is stripped.

    Args:
        path: Path to a JSONL file. Blank lines are skipped.
        field: The JSON field holding each document string (default ``"text"``).
        limit: If set, return at most this many documents (file order).

    Returns:
        The list of document strings, in file order.

    Raises:
        ValueError: if no documents were loaded.
        KeyError: if a row is missing ``field``.
    """
    docs: list[str] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            docs.append(strip_doctag(json.loads(line)[field]))
            if limit is not None and len(docs) >= limit:
                break
    if not docs:
        raise ValueError(f"No documents loaded from {path}")
    return docs


# --------------------------------------------------------------------------- #
# datum construction / corpus assembly (Tinker types — lazy heavy imports)
# --------------------------------------------------------------------------- #
def make_hard_datum(tok, doc_text: str, max_tokens: int, *, min_tokens: int = 16):
    """A single SFT datum (hard next-token targets) from one document.

    Returns ``None`` for documents shorter than ``min_tokens`` tokens (too short
    to learn a next-token signal from). The document is truncated to
    ``max_tokens + 1`` tokens before the input/target shift.
    """
    import tinker
    import torch

    ids = tok(doc_text, add_special_tokens=False)["input_ids"][: max_tokens + 1]
    if len(ids) < min_tokens:
        return None
    model_input = tinker.ModelInput.from_ints(ids[:-1])
    targets = torch.tensor(ids[1:], dtype=torch.long)
    weights = torch.ones(len(ids) - 1, dtype=torch.float)
    return tinker.Datum(
        model_input=model_input,
        loss_fn_inputs={
            "target_tokens": tinker.TensorData.from_torch(targets),
            "weights": tinker.TensorData.from_torch(weights),
        },
    )


def make_datums(tok, docs, max_tokens: int):
    """``make_hard_datum`` over many docs, dropping the too-short ones."""
    return [d for d in (make_hard_datum(tok, t, max_tokens) for t in docs) if d]


def build_doc_corpus(tok, docs, *, max_doc_tokens: int, seed: int = 0, label: str = "doc-sft"):
    """Tokenize every document into a hard-target datum and return a shuffled list.

    Shuffling (deterministic in ``seed``) mixes the corpus across batches.
    """
    import random

    datums = make_datums(tok, docs, max_doc_tokens)
    random.Random(seed).shuffle(datums)
    log.info(
        "%s: docs=%d usable_datums=%d max_doc_tokens=%d",
        label, len(docs), len(datums), max_doc_tokens,
    )
    return datums


# --------------------------------------------------------------------------- #
# training loop
# --------------------------------------------------------------------------- #
async def train_doc_arm(
    *,
    model: str,
    datums,
    lora_rank: int,
    batch_size: int,
    epochs: int,
    lr: float,
    log_every: int,
    save_name: str,
    run_dir: str,
    label: str = "doc-sft",
):
    """Run plain cross-entropy LoRA SFT over ``datums`` (one document each).

    Saves a tinker:// sampler checkpoint and writes the run's artifacts into
    ``run_dir``:
      - ``checkpoints.jsonl``    — one {"sampler_path": ...} row (so
        ``read_train_result`` works uniformly across drivers)
      - ``final_checkpoint.txt`` — the tinker:// sampler path
      - ``metrics.jsonl``        — one {epoch, step, loss} per optimizer step
    Returns the checkpoint path.
    """
    import tinker

    rd = Path(run_dir)
    rd.mkdir(parents=True, exist_ok=True)
    sc = tinker.ServiceClient()
    training_client = await sc.create_lora_training_client_async(model, rank=lora_rank)

    # Tinker pipelining ("Clock cycles and pipelining"):
    #  (1) submit forward_backward + optim_step together before awaiting either ->
    #      they land on the SAME clock cycle (1 cycle/step, not 2);
    #  (2) submit the NEXT step before awaiting the current -> the worker pool never
    #      idles between steps. Submission ORDER fixes execution order server-side,
    #      so a 1-deep pipeline is correct (step N+1 trains on post-optim-N weights).
    async def submit(batch):
        fb = await training_client.forward_backward_async(list(batch), loss_fn="cross_entropy")
        op = await training_client.optim_step_async(tinker.AdamParams(learning_rate=lr))
        return fb, op

    schedule = [
        (epoch, datums[i:i + batch_size])
        for epoch in range(epochs)
        for i in range(0, len(datums), batch_size)
    ]
    step = 0
    inflight = None  # (epoch, step, fb_future, opt_future) — drained one step behind
    with open(rd / "metrics.jsonl", "w") as mf:  # incremental: survives a crash
        async def drain(ep, st, fb, op):
            res = await fb.result_async()
            await op.result_async()
            loss = (
                res.metrics.get("loss:sum", res.metrics.get("loss", float("nan")))
                if res.metrics
                else float("nan")
            )
            mf.write(json.dumps({"epoch": ep, "step": st, "loss": loss}) + "\n")
            mf.flush()
            if st % log_every == 0 or st == 1:
                log.info("%s: epoch %d step %d loss=%s", label, ep, st, loss)

        for epoch, chunk in schedule:
            fb, op = await submit(chunk)  # (1) fb + opt same cycle
            step += 1
            if inflight is not None:  # (2) prior step still in flight while we submitted this one
                await drain(*inflight)
            inflight = (epoch, step, fb, op)
        if inflight is not None:
            await drain(*inflight)

    save = await (await training_client.save_weights_for_sampler_async(name=save_name))
    (rd / "final_checkpoint.txt").write_text(save.path)
    (rd / "checkpoints.jsonl").write_text(
        json.dumps({"sampler_path": save.path}) + "\n"
    )
    log.info("%s: DONE steps=%d checkpoint=%s run_dir=%s", label, step, save.path, rd)
    return save.path


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
async def run_doc_sft(cfg: DocSFTConfig) -> TrainResult:
    """Run doc-token SFT (heavy: starts a Tinker run); returns the final
    checkpoint path + metrics read back from the run's artifacts."""
    from tinker_cookbook.tokenizer_utils import get_tokenizer

    log.info("doc_sft: %s", describe(cfg))
    tok = get_tokenizer(cfg.model)
    docs = load_docs(cfg.data, field=cfg.field, limit=cfg.limit)
    datums = build_doc_corpus(
        tok, docs, max_doc_tokens=cfg.max_doc_tokens, seed=cfg.seed
    )
    await train_doc_arm(
        model=cfg.model,
        datums=datums,
        lora_rank=cfg.lora_rank,
        batch_size=cfg.batch_size,
        epochs=cfg.num_epochs,
        lr=cfg.lr,
        log_every=cfg.log_every,
        save_name=cfg.save_name,
        run_dir=cfg.out,
    )
    return read_train_result(cfg.out)
