"""Unlearning driver: signed, mean-normalized cross-entropy LoRA.

All techniques reduce to building per-example ``Datum``s with **signed,
mean-normalized** per-token weights and feeding them through a standard
forward_backward / optim_step loop with the ``cross_entropy`` loss:

- weight sign ``+1`` -> gradient *descent* (ordinary SFT / corrective).
- weight sign ``-1`` -> gradient *ascent* (forget).

Mean-normalization (each example's weights sum to ``±1`` over its supervised
tokens) is applied *before* the sign, so every example contributes a unit-scale
gradient regardless of answer length and ascent/descent are symmetric. We do the
normalization by hand (``reduction="none"`` on the way into Tinker) precisely so
a negative sign survives — the cookbook's ``reduction="mean"`` renormalizes to a
positive sum and would silently undo gradient ascent.

Library entry point::

    await run_unlearn(UnlearnConfig(model=..., renderer=..., out=..., forget=...))

This is a training driver in the same family as :mod:`sft` / :mod:`dpo` /
:mod:`distill`: it takes a frozen :class:`~aligne.train.tinker.configs.UnlearnConfig`
and returns a typed :class:`~aligne.train.tinker.results.UnlearnResult`. Heavy
imports (``tinker``, ``tinker_cookbook``) are LAZY inside the run/build
functions; the pure helpers (:func:`load_convs`, :func:`oversample_to_match`)
have no heavy deps and are unit-tested on CPU. The CLI adapter lives in
:mod:`aligne.train.tinker.cli` (``aligne train unlearn``).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from .configs import UnlearnConfig, describe
from .results import UnlearnResult

log = logging.getLogger(__name__)

Conv = dict[str, Any]  # {"messages": [{"role","content"}, ...]}

# signed descent(+1)/ascent(-1) weight for the (forget, retain) sets per technique
_TECHNIQUE_SIGNS: dict[str, tuple[float, float]] = {
    "sft": (+1.0, +1.0),
    "corrective": (+1.0, +1.0),
    "gradient_ascent": (-1.0, +1.0),
    "grad_diff": (-1.0, +1.0),
}


def load_convs(path: str | Path) -> list[Conv]:
    """Read a conversations JSONL (rows ``{"messages": [...]}``), skipping
    blank lines. Raises if a row lacks ``messages``."""
    rows: list[Conv] = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        if "messages" not in row:
            raise ValueError(f"row missing 'messages': {row!r}")
        rows.append(row)
    if not rows:
        raise ValueError(f"no conversations in {path}")
    return rows


def oversample_to_match(items: list, target_len: int) -> list:
    """Repeat ``items`` (in order) up to ``target_len``, then truncate.

    Balances GradDiff's retain set against the (typically larger) forget set:
    the retain set is oversampled to the forget count so each batch is, on
    average, half ascent / half descent — the stabilizer GradDiff is supposed
    to provide. Left as-is the per-batch ascent term dominates and the loss
    diverges (the model collapses like pure GA). No-op when ``items`` already
    meets or exceeds ``target_len``.
    """
    if not items or len(items) >= target_len:
        return list(items)
    reps = (target_len + len(items) - 1) // len(items)
    return (list(items) * reps)[:target_len]


def make_renderer(model: str, renderer: str):
    """Build the (renderer, tokenizer) pair used for training. Needs the
    ``tinker`` extra (``tinker_cookbook``)."""
    from tinker_cookbook.renderers import get_renderer
    from tinker_cookbook.tokenizer_utils import get_tokenizer

    tok = get_tokenizer(model)
    return get_renderer(renderer, tok), tok


def build_datums(convs: list[Conv], renderer, *, sign: float, max_length: int) -> list:
    """Render single-turn conversations into signed, mean-normalized Datums.

    ``sign=-1.0`` flips the gradient (gradient ascent / forget). We use
    ``TrainOnWhat.LAST_ASSISTANT_MESSAGE`` because every conversation here is a
    single ``(user, assistant)`` turn, which sidesteps the extension-property
    caveat of ``ALL_ASSISTANT_MESSAGES``.
    """
    from tinker_cookbook.renderers import TrainOnWhat
    from tinker_cookbook.supervised import datum_from_model_input_weights

    out = []
    for c in convs:
        model_input, w = renderer.build_supervised_example(
            c["messages"], train_on_what=TrainOnWhat.LAST_ASSISTANT_MESSAGE
        )
        w = w.float()
        denom = w.sum().clamp_min(1.0)  # # of supervised tokens (mask is 0/1)
        w = (w / denom) * sign
        out.append(
            datum_from_model_input_weights(
                model_input, w, max_length=max_length, reduction="none"
            )
        )
    return out


def build_technique_datums(cfg: UnlearnConfig, renderer) -> list:
    """Assemble the full signed datum list for ``cfg.technique`` (heavy: needs
    a rendered tokenizer). For ``grad_diff`` the retain set is oversampled to
    match the forget count (see :func:`oversample_to_match`)."""
    forget_sign, retain_sign = _TECHNIQUE_SIGNS[cfg.technique]
    forget = build_datums(
        load_convs(cfg.forget), renderer, sign=forget_sign, max_length=cfg.max_length
    )
    if not cfg.retain:
        return forget
    retain = build_datums(
        load_convs(cfg.retain), renderer, sign=retain_sign, max_length=cfg.max_length
    )
    if cfg.technique == "grad_diff":
        retain = oversample_to_match(retain, len(forget))
    return forget + retain


def _append_jsonl(path: Path, row: dict) -> None:
    with path.open("a") as f:
        f.write(json.dumps(row) + "\n")


async def run_unlearn(cfg: UnlearnConfig) -> UnlearnResult:
    """Run the signed forward_backward / optim_step loop (heavy: starts a
    Tinker run); return the final checkpoint pointers + a compact loop view.

    Mirrors the checkpoint/artifact conventions of the sibling drivers: writes
    ``metrics.jsonl`` / ``checkpoints.jsonl`` under ``cfg.out`` and saves a
    final servable sampler (plus resumable state) checkpoint.
    """
    import random

    import tinker

    log.info("unlearn: %s", describe(cfg))
    renderer, _tok = make_renderer(cfg.model, cfg.renderer)
    datums = build_technique_datums(cfg, renderer)

    out = Path(cfg.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    metrics_path = out / "metrics.jsonl"
    ckpts_path = out / "checkpoints.jsonl"

    service_client = tinker.ServiceClient()
    if cfg.load_checkpoint_path:
        training_client = await service_client.create_training_client_from_state_async(
            cfg.load_checkpoint_path
        )
        log.info("unlearn: loaded weights from %s", cfg.load_checkpoint_path)
    else:
        training_client = await service_client.create_lora_training_client_async(
            cfg.model, rank=cfg.lora_rank
        )

    rng = random.Random(cfg.seed)
    order = list(range(len(datums)))
    state_path: str | None = None
    sampler_path: str | None = None
    final_loss: float | None = None
    step = 0
    for ep in range(cfg.num_epochs):
        rng.shuffle(order)
        for i in range(0, len(order), cfg.batch_size):
            batch = [datums[j] for j in order[i : i + cfg.batch_size]]
            adam = tinker.AdamParams(learning_rate=cfg.lr)
            fwd_future = await training_client.forward_backward_async(
                batch, loss_fn="cross_entropy"
            )
            optim_future = await training_client.optim_step_async(adam)
            fb_res = await fwd_future.result_async()
            await optim_future.result_async()
            loss = None
            m = getattr(fb_res, "metrics", None) or {}
            loss = m.get("loss:sum") or m.get("loss") or m.get("nll")
            final_loss = loss if loss is not None else final_loss
            step += 1
            _append_jsonl(metrics_path,
                          {"step": step, "epoch": ep, "n": len(batch), "loss": loss})
            log.info("unlearn[%s] step %d epoch %d bs %d loss %s",
                     cfg.technique, step, ep, len(batch), loss)
            if cfg.save_every > 0 and step % cfg.save_every == 0:
                name = f"step{step}"
                state_path = (await (await training_client.save_state_async(name)).result_async()).path
                sampler_path = (await (await training_client.save_weights_for_sampler_async(name)).result_async()).path
                _append_jsonl(ckpts_path,
                              {"step": step, "state_path": state_path,
                               "sampler_path": sampler_path})
            if cfg.max_steps is not None and step >= cfg.max_steps:
                break
        else:
            continue
        break

    # Always save a final servable checkpoint (the loop above only saves on the
    # optional cadence), so the result's sampler_path is meaningful.
    if not sampler_path or (cfg.save_every > 0 and step % cfg.save_every != 0) or cfg.save_every == 0:
        name = f"step{step}"
        state_path = (await (await training_client.save_state_async(name)).result_async()).path
        sampler_path = (await (await training_client.save_weights_for_sampler_async(name)).result_async()).path
        _append_jsonl(ckpts_path,
                      {"step": step, "state_path": state_path, "sampler_path": sampler_path})

    return UnlearnResult(
        out_dir=str(out), technique=cfg.technique, steps=step,
        sampler_path=sampler_path, state_path=state_path, final_loss=final_loss,
    )
