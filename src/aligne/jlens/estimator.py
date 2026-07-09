"""Probe-based J-lens estimator: all layers from each backward pass.

J_ℓ = E_seq[ mean_t Σ_{t′≥t} ∂r_{t′}/∂h_{ℓ,t} ]  ∈ R^{d×d}, where h_{ℓ,t} is
the residual stream entering decoder block ℓ at position t and r_{t′} is the
post-final-norm residual (the unembedding input) at position t′. Cotangents
are seeded at r, so the stored J already includes the final-norm Jacobian and
`readout()` is exactly W_U @ J @ h. Index convention, unbiasedness proof, and
variance notes live in ESTIMATOR.md next to this file; the exact-Jacobian and
finite-difference parity tests are tests/test_jlens_estimator.py.

No parameter gradients are ever needed: the caller must have called
`model.requires_grad_(False)`, which halves backward cost. The graph is
forced into existence by promoting the block-0 input (the embedding output,
a graph-detached tensor once no parameter requires grad) to a grad-requiring
leaf inside the block-0 pre-hook.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

# Candidate module paths for the decoder-block ModuleList, tried in order.
# Covers Llama/Qwen/Mistral/DeepSeek/K2 (model.layers), GPT-2 (transformer.h),
# GPT-NeoX, OPT, and bare base models (layers).
DECODER_LAYER_PATHS = (
    "model.layers",
    "transformer.h",
    "gpt_neox.layers",
    "model.decoder.layers",
    "layers",
)


def find_decoder_layers(model: torch.nn.Module) -> list[torch.nn.Module]:
    """Locate the decoder-block ModuleList by pattern-matching known paths."""
    for path in DECODER_LAYER_PATHS:
        obj = model
        for name in path.split("."):
            obj = getattr(obj, name, None)
            if obj is None:
                break
        if isinstance(obj, torch.nn.ModuleList) and len(obj) > 0:
            return list(obj)
    raise ValueError(
        "could not locate decoder layers; tried "
        f"{DECODER_LAYER_PATHS} on {type(model).__name__}. "
        "Pass layers explicitly via ResidualTaps(layers=...)."
    )


def base_model(model: torch.nn.Module) -> torch.nn.Module:
    """The bare decoder whose last_hidden_state is the post-final-norm
    residual (input to the unembedding) — the cotangent seeding point."""
    if hasattr(model, "get_decoder"):
        dec = model.get_decoder()
        if dec is not None:
            return dec
    return model


class ResidualTaps:
    """Context manager capturing the residual-stream input to every decoder
    block during a forward pass, as live graph tensors usable as `inputs` to
    torch.autograd.grad.

    The block-0 input is detached and promoted to a grad-requiring leaf —
    with all parameters at requires_grad=False this single promotion is what
    makes autograd record the whole stack.
    """

    def __init__(self, layers: list[torch.nn.Module]):
        self.layers = layers
        self.acts: list[torch.Tensor | None] = [None] * len(layers)
        self._handles: list = []

    def __enter__(self) -> "ResidualTaps":
        for i, layer in enumerate(self.layers):
            self._handles.append(
                layer.register_forward_pre_hook(self._hook(i), with_kwargs=True)
            )
        return self

    def __exit__(self, *exc) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def _hook(self, i: int):
        def hook(module, args, kwargs):
            if args:
                hs = args[0]
            else:
                hs = kwargs["hidden_states"]
            if i == 0 and not hs.requires_grad:
                hs = hs.detach().requires_grad_(True)
                if args:
                    args = (hs,) + args[1:]
                else:
                    kwargs = dict(kwargs)
                    kwargs["hidden_states"] = hs
            self.acts[i] = hs
            return args, kwargs

        return hook


@dataclass
class EstimatorConfig:
    # "exact": one basis-cotangent backward per output dim (d backwards per
    # batch) — zero probe noise, sequence sampling is the only variance left.
    # "probe": Hutchinson probes — n_probes backwards per batch, but relative
    # error ~ sqrt(d·T / n_seqs·n_probes), which at real-model d·T needs
    # millions of units; use only for cheap qualitative looks (ESTIMATOR.md §3).
    mode: str = "exact"
    n_probes: int = 4  # probe mode only
    probe_dist: str = "rademacher"  # "rademacher" | "gaussian"
    # Where the fp32 accumulator shards live. "model" = same device as the
    # model output (fastest; needs 2·L·d²·4 bytes of headroom), "cpu" = safe
    # fallback for big models at the cost of per-probe d² transfers.
    accumulator_device: str = "model"
    seed: int = 0


class ShardedAccumulator:
    """Two independent fp32 accumulator shards (A = even global sequence
    index, B = odd) so split-half convergence (spec §4) needs no re-fit. The
    published lens is the merged A+B estimate.

    Each (sequence, probe) pair contributes one outer product
    u_sum ⊗ (g_sum / n_src): rows index the output (post-final-norm residual)
    space, columns the input (layer-ℓ residual) space. Normalization is by
    contribution count, tracked per shard.
    """

    def __init__(self, n_layers: int, d: int, device: str = "cpu"):
        self.sums = torch.zeros(
            2, n_layers, d, d, dtype=torch.float32, device=device
        )
        self.counts = torch.zeros(2, dtype=torch.float64)  # (seq, probe) units
        self.n_seqs = torch.zeros(2, dtype=torch.int64)

    def add(self, shard: int, layer_outers: torch.Tensor, n_units: int) -> None:
        """layer_outers: [L, d, d] summed outer products for `n_units`
        (sequence, probe) contributions belonging to `shard`."""
        self.sums[shard] += layer_outers.to(self.sums.device, torch.float32)
        self.counts[shard] += n_units

    def estimate(self, which: str = "merged") -> torch.Tensor:
        """Normalized Ĵ [L, d, d]. which ∈ {"merged", "a", "b"}."""
        if which == "merged":
            total = self.counts.sum().item()
            if total == 0:
                raise ValueError("empty accumulator")
            return (self.sums[0] + self.sums[1]) / total
        idx = {"a": 0, "b": 1}[which]
        if self.counts[idx].item() == 0:
            raise ValueError(f"empty shard {which!r}")
        return self.sums[idx] / self.counts[idx].item()

    def state_dict(self) -> dict:
        return {
            "sums": self.sums.cpu(),
            "counts": self.counts,
            "n_seqs": self.n_seqs,
        }

    def load_state_dict(self, state: dict) -> None:
        self.sums.copy_(state["sums"].to(self.sums.device))
        self.counts.copy_(state["counts"])
        self.n_seqs.copy_(state["n_seqs"])


def sample_probe(
    shape: tuple[int, ...],
    dist: str,
    generator: torch.Generator,
    dtype: torch.dtype,
    device,
) -> torch.Tensor:
    """Random cotangent field. Sampled on CPU with a seeded generator (CUDA
    generators are not reproducible across devices), then moved."""
    if dist == "rademacher":
        u = torch.randint(0, 2, shape, generator=generator, device="cpu")
        u = u.to(torch.float32) * 2.0 - 1.0
    elif dist == "gaussian":
        u = torch.randn(shape, generator=generator, device="cpu")
    else:
        raise ValueError(f"unknown probe_dist {dist!r}")
    return u.to(device=device, dtype=dtype)


def backward_taps(
    final: torch.Tensor,
    taps: ResidualTaps,
    cotangent: torch.Tensor,
    retain_graph: bool,
) -> list[torch.Tensor]:
    """One backward pass: gradients of <final, cotangent> at every tap.

    This is the single autograd entry point — the probe estimator and the
    deterministic exact-row extraction used by the parity tests both go
    through it, so an index-convention bug cannot hide in one path only.
    """
    return list(
        torch.autograd.grad(
            outputs=final,
            inputs=taps.acts,
            grad_outputs=cotangent,
            retain_graph=retain_graph,
        )
    )


def accumulate_sequences(
    model: torch.nn.Module,
    input_ids: torch.Tensor,  # [B, T] on model device
    source_mask: torch.Tensor,  # [B, T] {0,1}
    target_mask: torch.Tensor,  # [B, T] {0,1}
    attention_mask: torch.Tensor,  # [B, T] {0,1} (pads)
    seq_shards: torch.Tensor,  # [B] in {0, 1}: split-half shard per sequence
    acc: ShardedAccumulator,
    cfg: EstimatorConfig,
    generator: torch.Generator,
) -> None:
    """Run one batch through the estimator, updating both accumulator shards.

    Per probe p and sequence b the contribution is
        (Σ_{t′} u_{b,t′}) ⊗ (Σ_t g_{b,ℓ,t} · source_mask / n_src(b)),
    which is unbiased for the target-summed, source-averaged Jacobian
    (ESTIMATOR.md §2 — cross terms t′<t vanish in expectation by
    independence of per-position probes; causality zeroes their gradients'
    dependence anyway).
    """
    dec = base_model(model)
    layers = find_decoder_layers(model)
    b, t = input_ids.shape

    with ResidualTaps(layers) as taps, torch.enable_grad():
        out = dec(
            input_ids=input_ids, attention_mask=attention_mask, use_cache=False
        )
        final = out.last_hidden_state  # [B, T, d] post final norm
        d = final.shape[-1]
        # With device_map sharding, taps/grads live on per-layer devices and
        # `final` on the last stage; everything is reduced on final.device.
        fdev = final.device
        n_src = (source_mask * attention_mask).sum(dim=1).clamp(min=1).to(fdev)
        tgt = (target_mask * attention_mask).to(fdev, torch.float32)  # [B, T]
        src = (source_mask * attention_mask).to(fdev, torch.float32).unsqueeze(-1)
        shards = seq_shards.to(fdev)

        for p in range(cfg.n_probes):
            u = sample_probe((b, t, d), cfg.probe_dist, generator, final.dtype, fdev)
            u = u * tgt.to(u.dtype).unsqueeze(-1)
            grads = backward_taps(final, taps, u, retain_graph=p < cfg.n_probes - 1)

            u_sum = u.to(torch.float32).sum(dim=1)  # [B, d]
            # g_sum[b] = Σ_t g[b,t] · src / n_src(b), per layer → [L, B, d]
            g_sum = torch.stack(
                [(g.to(fdev, torch.float32) * src).sum(dim=1) for g in grads]
            ) / n_src.to(torch.float32).unsqueeze(-1)

            for shard in (0, 1):
                sel = shards == shard
                if not bool(sel.any()):
                    continue
                outers = torch.einsum("bi,lbj->lij", u_sum[sel], g_sum[:, sel])
                acc.add(shard, outers, n_units=int(sel.sum()))

    for shard in (0, 1):
        acc.n_seqs[shard] += int((seq_shards == shard).sum())


def accumulate_sequences_exact(
    model: torch.nn.Module,
    input_ids: torch.Tensor,  # [B, T]
    source_mask: torch.Tensor,  # [B, T] {0,1}
    target_mask: torch.Tensor,  # [B, T] {0,1}
    attention_mask: torch.Tensor,  # [B, T] {0,1}
    seq_shards: torch.Tensor,  # [B] in {0, 1}
    acc: ShardedAccumulator,
    cfg: EstimatorConfig,
    generator: torch.Generator | None = None,  # unused; signature-compatible
) -> None:
    """Exact-row variant: cotangent e_i at every target position recovers row
    i of the target-summed, source-averaged Jacobian with NO probe noise —
    d backwards per batch, each ~one forward-equivalent (no weight grads).
    This is the default: probe-mode variance scales as d·T/n_units and never
    reaches top-25-Jaccard convergence at feasible n on real models."""
    dec = base_model(model)
    layers = find_decoder_layers(model)

    with ResidualTaps(layers) as taps, torch.enable_grad():
        out = dec(
            input_ids=input_ids, attention_mask=attention_mask, use_cache=False
        )
        final = out.last_hidden_state  # [B, T, d]
        d = final.shape[-1]
        fdev = final.device
        n_src = (source_mask * attention_mask).sum(dim=1).clamp(min=1).to(fdev)
        tgt = (target_mask * attention_mask).to(fdev, final.dtype)  # [B, T]
        src = (source_mask * attention_mask).to(fdev, torch.float32).unsqueeze(-1)
        shards = seq_shards.to(fdev)
        sel = [shards == 0, shards == 1]
        n_sel = [int(s.sum()) for s in sel]

        for i in range(d):
            u = torch.zeros_like(final)
            u[..., i] = tgt
            grads = backward_taps(final, taps, u, retain_graph=i < d - 1)
            # row i, source-averaged, per layer and sequence → [L, B, d]
            g_sum = torch.stack(
                [(g.to(fdev, torch.float32) * src).sum(dim=1) for g in grads]
            ) / n_src.to(torch.float32).unsqueeze(-1)
            for shard in (0, 1):
                if n_sel[shard]:
                    acc.sums[shard][:, i, :] += (
                        g_sum[:, sel[shard]].sum(dim=1).to(acc.sums.device)
                    )

    for shard in (0, 1):
        acc.counts[shard] += n_sel[shard]  # one exact unit per sequence
        acc.n_seqs[shard] += n_sel[shard]


@torch.no_grad()
def harvest_activations(
    model: torch.nn.Module,
    input_ids: torch.Tensor,  # [B, T]
    attention_mask: torch.Tensor,  # [B, T]
    n_per_layer: int,
    generator: torch.Generator,
) -> torch.Tensor:
    """Sample residual-stream activations at every layer from a forward pass:
    returns [L, n, d] fp32 (n ≤ n_per_layer if the batch is small). Used to
    build the frozen eval-probe set for functional convergence (spec §4)."""
    dec = base_model(model)
    layers = find_decoder_layers(model)
    with ResidualTaps(layers) as taps:
        dec(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        acts = [a.detach().to("cpu", torch.float32) for a in taps.acts]

    flat_mask = attention_mask.reshape(-1).bool().cpu()
    idx_pool = flat_mask.nonzero(as_tuple=True)[0]
    take = min(n_per_layer, idx_pool.numel())
    perm = torch.randperm(idx_pool.numel(), generator=generator)[:take]
    chosen = idx_pool[perm]
    d = acts[0].shape[-1]
    return torch.stack([a.reshape(-1, d)[chosen] for a in acts])
