"""J-lens fit orchestration: doubling rounds to functional convergence.

Round n consumes sequences until the accumulator holds n of them, then runs
the two convergence tests (spec §4): doubling — merged Ĵ(n) vs the previous
round's Ĵ(n/2) — and split-half — shard A vs shard B. The fit stops when
every layer passes both, or at the sequence cap (in which case unconverged
layers are flagged in the manifest, never silently).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable, Iterator

import torch

from aligne.jlens.artifacts import save_jlens
from aligne.jlens.convergence import ConvergenceReport, ConvergenceSpec, Round, compare
from aligne.jlens.datasets import FitDataset, Sequence, sequences
from aligne.jlens.estimator import (
    EstimatorConfig,
    ShardedAccumulator,
    accumulate_sequences,
    find_decoder_layers,
    harvest_activations,
)

ESTIMATOR_VERSION = 1


@dataclass
class FitConfig:
    model: str
    dataset: FitDataset = field(default_factory=FitDataset)
    convergence: ConvergenceSpec = field(default_factory=ConvergenceSpec)
    revision: str | None = None
    dtype: str = "auto"  # "auto" | "bfloat16" | "float32" | ...
    device_map: str | None = "auto"
    attn_implementation: str | None = None  # force "eager" when custom kernels lack backward
    batch_size: int = 8
    n_probes: int = 4
    probe_dist: str = "rademacher"
    accumulator_device: str = "model"  # "model" (W_U's device) | "cpu"
    seed: int = 0
    output_dir: str = "jlens-out"
    cache_dir: str | None = None  # aligne.hfdata on-disk cache


def load_config(path: str | Path) -> FitConfig:
    import yaml

    raw = yaml.safe_load(Path(path).read_text())
    dataset = FitDataset(**raw.pop("dataset", {}))
    conv_raw = raw.pop("convergence", {})
    eval_ds = conv_raw.pop("eval_dataset", None)
    convergence = ConvergenceSpec(
        eval_dataset=FitDataset(**eval_ds) if eval_ds else None, **conv_raw
    )
    return FitConfig(dataset=dataset, convergence=convergence, **raw)


def _batches(
    stream: Iterator[Sequence], batch_size: int, pad_id: int, device
) -> Iterator[dict]:
    """Right-pad each batch to its own max length; padded positions are dead
    in every mask (attention, source, target)."""
    batch: list[Sequence] = []

    def emit(batch: list[Sequence]) -> dict:
        t = max(len(s.input_ids) for s in batch)

        def pad(rows: list[list[int]], fill: int) -> torch.Tensor:
            return torch.tensor(
                [r + [fill] * (t - len(r)) for r in rows], device=device
            )

        return {
            "input_ids": pad([s.input_ids for s in batch], pad_id),
            "source_mask": pad([s.source_mask for s in batch], 0),
            "target_mask": pad([s.target_mask for s in batch], 0),
            "attention_mask": pad(
                [[1] * len(s.input_ids) for s in batch], 0
            ),
        }

    for seq in stream:
        batch.append(seq)
        if len(batch) == batch_size:
            yield emit(batch)
            batch = []
    if batch:
        yield emit(batch)


def _load_model(cfg: FitConfig):
    import transformers

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        cfg.model, revision=cfg.revision
    )
    kwargs: dict = {"revision": cfg.revision}
    if cfg.dtype != "auto":
        kwargs["dtype"] = getattr(torch, cfg.dtype)
    if cfg.device_map is not None:
        kwargs["device_map"] = cfg.device_map
    if cfg.attn_implementation is not None:
        kwargs["attn_implementation"] = cfg.attn_implementation
    model = transformers.AutoModelForCausalLM.from_pretrained(cfg.model, **kwargs)
    return model, tokenizer


def _harvest_eval_probes(
    model, tokenizer, cfg: FitConfig, device, generator: torch.Generator
) -> torch.Tensor:
    """Frozen eval activations [L, n_eval, d] from the eval dataset (or a
    held-out slice of the fitting distribution at a disjoint data_seed)."""
    conv = cfg.convergence
    ds = conv.eval_dataset or replace(
        cfg.dataset, data_seed=cfg.dataset.data_seed + 100_000
    )
    seq_len = ds.seq_len if ds.kind == "pretrain" else ds.max_seq_len
    n_seqs = max(4, 2 * -(-conv.n_eval_activations // seq_len))
    ds = replace(ds, n_seqs=n_seqs)
    pad_id = (
        getattr(tokenizer, "pad_token_id", None)
        or getattr(tokenizer, "eos_token_id", None)
        or 0
    )

    chunks = []
    for batch in _batches(
        sequences(ds, tokenizer, cache_dir=_cache_dir(cfg)),
        cfg.batch_size,
        pad_id,
        device,
    ):
        chunks.append(
            harvest_activations(
                model,
                batch["input_ids"],
                batch["attention_mask"],
                n_per_layer=conv.n_eval_activations,
                generator=generator,
            )
        )
    pool = torch.cat(chunks, dim=1)  # [L, Σn, d]
    take = min(conv.n_eval_activations, pool.shape[1])
    idx = torch.randperm(pool.shape[1], generator=generator)[:take]
    return pool[:, idx]


def _cache_dir(cfg: FitConfig) -> Path | None:
    return Path(cfg.cache_dir) if cfg.cache_dir else None


def fit(
    cfg: FitConfig,
    model=None,
    tokenizer=None,
    resume: bool = False,
    log: Callable[[str], None] = print,
) -> Path:
    """Run the fit; returns the artifact directory. `model`/`tokenizer` may
    be preloaded (tests, custom deployments); otherwise loaded per config."""
    t0 = time.time()
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if model is None:
        model, tokenizer = _load_model(cfg)
    model.eval()
    model.requires_grad_(False)

    layers = find_decoder_layers(model)
    n_layers = len(layers)
    emb = model.get_input_embeddings()
    in_device = emb.weight.device
    W_U = model.get_output_embeddings().weight.detach().to(torch.float32)
    d = W_U.shape[1]
    acc_device = W_U.device if cfg.accumulator_device == "model" else "cpu"

    conv = cfg.convergence
    cap = min(cfg.dataset.n_seqs, conv.max_seqs)
    pad_id = (
        getattr(tokenizer, "pad_token_id", None)
        or getattr(tokenizer, "eos_token_id", None)
        or 0
    )

    acc = ShardedAccumulator(n_layers, d, device=str(acc_device))
    est_cfg = EstimatorConfig(n_probes=cfg.n_probes, probe_dist=cfg.probe_dist)
    generator = torch.Generator().manual_seed(cfg.seed)
    eval_gen = torch.Generator().manual_seed(cfg.seed + 1)
    report = ConvergenceReport(spec=conv)
    J_prev: torch.Tensor | None = None
    consumed = 0

    ckpt_path = out / "checkpoint.pt"
    if resume and ckpt_path.exists():
        state = torch.load(ckpt_path, weights_only=False)
        acc.load_state_dict(state["acc"])
        J_prev = state["J_prev"]
        consumed = state["consumed"]
        generator.set_state(state["generator"])
        report.rounds = [Round(**r) for r in state["rounds"]]
        log(f"[jlens] resumed at {consumed} sequences")

    log(f"[jlens] harvesting {conv.n_eval_activations} eval activations/layer")
    eval_probes = _harvest_eval_probes(model, tokenizer, cfg, in_device, eval_gen)

    stream = sequences(
        replace(cfg.dataset, n_seqs=cap), tokenizer, cache_dir=_cache_dir(cfg)
    )
    for _ in range(consumed):  # fast-forward on resume (tokenization only)
        next(stream)

    shard_of = lambda i: i % 2  # noqa: E731 — split-half by global parity
    n_target = conv.min_seqs
    while n_target // 2 < consumed:  # resume lands mid-schedule
        n_target *= 2

    converged = False
    while True:
        n_target = min(n_target, cap)
        batch_seqs: list[Sequence] = []
        while consumed < n_target:
            batch_seqs.append(next(stream))
            consumed += 1
            if len(batch_seqs) == cfg.batch_size or consumed == n_target:
                shards = torch.tensor(
                    [shard_of(consumed - len(batch_seqs) + j) for j in range(len(batch_seqs))]
                )
                batch = next(iter(_batches(iter(batch_seqs), len(batch_seqs), pad_id, in_device)))
                accumulate_sequences(
                    model,
                    batch["input_ids"],
                    batch["source_mask"],
                    batch["target_mask"],
                    batch["attention_mask"],
                    shards,
                    acc,
                    est_cfg,
                    generator,
                )
                batch_seqs = []

        J_curr = acc.estimate("merged").cpu()
        split_half = compare(
            acc.estimate("a"), acc.estimate("b"), W_U, eval_probes, conv
        )
        doubling = (
            compare(J_curr, J_prev, W_U, eval_probes, conv)
            if J_prev is not None
            else None
        )
        report.add_round(consumed, split_half, doubling)
        worst = report.worst_layer()
        log(
            f"[jlens] n={consumed} split-half worst={min(split_half):.3f} "
            + (f"doubling worst={min(doubling):.3f} " if doubling else "")
            + (f"(worst layer {worst})" if worst is not None else "")
        )

        torch.save(
            {
                "acc": acc.state_dict(),
                "J_prev": J_curr,
                "consumed": consumed,
                "generator": generator.get_state(),
                "rounds": [vars(r) for r in report.rounds],
            },
            ckpt_path,
        )

        if report.converged(n_layers):
            converged = True
            break
        if consumed >= cap:
            log(f"[jlens] hit sequence cap {cap} before convergence")
            break
        J_prev = J_curr
        n_target = consumed * 2

    manifest = {
        "model": cfg.model,
        "revision": cfg.revision,
        "dtype": cfg.dtype,
        "tokenizer": getattr(tokenizer, "name_or_path", str(type(tokenizer).__name__)),
        "vocab_size": W_U.shape[0],
        "n_layers": n_layers,
        "hidden_size": d,
        "dataset": cfg.dataset.to_dict(),
        "convergence": report.to_dict(),
        "converged": converged,
        "n_seqs_used": consumed,
        "n_probes": cfg.n_probes,
        "probe_dist": cfg.probe_dist,
        "seed": cfg.seed,
        "estimator_version": ESTIMATOR_VERSION,
        "wall_clock_s": round(time.time() - t0, 1),
        "versions": _versions(),
    }
    save_jlens(out, acc.estimate("merged").cpu(), manifest, eval_probes)
    (out / "checkpoint.pt").unlink(missing_ok=True)
    log(f"[jlens] saved artifact to {out} (converged={converged})")
    return out


def _versions() -> dict:
    import aligne

    versions = {"torch": torch.__version__, "aligne": getattr(aligne, "__version__", "0")}
    try:
        import transformers

        versions["transformers"] = transformers.__version__
    except ImportError:
        pass
    return versions
