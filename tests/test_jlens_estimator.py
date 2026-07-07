"""Parity tests for the J-lens probe estimator (spec §2, ESTIMATOR.md §4).

Three independent pins on correctness, all on a float64 toy Llama (CPU):
1. exact rows via deterministic basis cotangents through the estimator's own
   backward entry point,
2. finite differences with NO autograd (independent machinery),
3. statistical convergence of the Rademacher probe estimate to the exact J.
"""

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from aligne.jlens.estimator import (  # noqa: E402
    EstimatorConfig,
    ResidualTaps,
    ShardedAccumulator,
    accumulate_sequences,
    backward_taps,
    base_model,
    find_decoder_layers,
    harvest_activations,
)

D, L, T, VOCAB = 8, 2, 6, 32


@pytest.fixture(scope="module")
def toy():
    cfg = transformers.LlamaConfig(
        hidden_size=D,
        intermediate_size=16,
        num_hidden_layers=L,
        num_attention_heads=2,
        num_key_value_heads=2,
        vocab_size=VOCAB,
        max_position_embeddings=64,
        attn_implementation="eager",
    )
    torch.manual_seed(0)
    model = transformers.LlamaForCausalLM(cfg).to(torch.float64)
    model.eval()
    model.requires_grad_(False)
    torch.manual_seed(1)
    input_ids = torch.randint(0, VOCAB, (1, T))
    return model, input_ids


def exact_J(model, input_ids, source_mask, target_mask):
    """Target-summed, source-averaged Jacobian, one exact row per basis
    cotangent, through the same backward_taps path the probe estimator uses."""
    dec = base_model(model)
    layers = find_decoder_layers(model)
    src = source_mask[0].to(torch.float64)
    n_src = src.sum()
    with ResidualTaps(layers) as taps, torch.enable_grad():
        final = dec(input_ids=input_ids, use_cache=False).last_hidden_state
        d = final.shape[-1]
        J = torch.zeros(len(layers), d, d, dtype=torch.float64)
        for i in range(d):
            u = torch.zeros_like(final)
            u[..., i] = 1.0
            u = u * target_mask.to(u.dtype).unsqueeze(-1)
            grads = backward_taps(final, taps, u, retain_graph=i < d - 1)
            for layer, g in enumerate(grads):
                J[layer, i, :] = (g[0] * src.unsqueeze(-1)).sum(0) / n_src
    return J


def fd_J_layer(model, input_ids, source_mask, target_mask, layer, eps=1e-4):
    # eps sits at the empirical minimum of the FD error V-curve for this
    # float64 toy (truncation above ~1e-4, cancellation below).
    """Finite-difference reconstruction of the same object at one layer,
    column by column, with no autograd anywhere."""
    dec = base_model(model)
    layers = find_decoder_layers(model)
    d = model.config.hidden_size

    def run(t, delta):
        def hook(module, args, kwargs):
            hs = (args[0] if args else kwargs["hidden_states"]).clone()
            hs[:, t] += delta
            if args:
                return (hs,) + args[1:], kwargs
            kwargs = dict(kwargs)
            kwargs["hidden_states"] = hs
            return args, kwargs

        handle = layers[layer].register_forward_pre_hook(hook, with_kwargs=True)
        try:
            with torch.no_grad():
                return dec(input_ids=input_ids, use_cache=False).last_hidden_state
        finally:
            handle.remove()

    sources = source_mask[0].nonzero(as_tuple=True)[0].tolist()
    tgt = target_mask[0].to(torch.float64).unsqueeze(-1)
    J = torch.zeros(d, d, dtype=torch.float64)
    for t in sources:
        for j in range(d):
            delta = torch.zeros(d, dtype=torch.float64)
            delta[j] = eps
            diff = (run(t, delta) - run(t, -delta))[0] / (2 * eps)  # [T, d]
            J[:, j] += (diff * tgt).sum(0)
    return J / len(sources)


def test_exact_rows_match_finite_difference(toy):
    model, input_ids = toy
    source_mask = torch.ones(1, T)
    target_mask = torch.ones(1, T)
    Jx = exact_J(model, input_ids, source_mask, target_mask)
    for layer in range(L):
        Jfd = fd_J_layer(model, input_ids, source_mask, target_mask, layer)
        scale = Jx[layer].abs().max()
        maxdiff = (Jx[layer] - Jfd).abs().max()
        assert maxdiff <= 1e-4 * scale, (
            f"layer {layer}: max diff {maxdiff.item():.2e} vs scale {scale.item():.2e}"
        )


def test_masks_restrict_targets_and_sources(toy):
    model, input_ids = toy
    source_mask = torch.tensor([[0.0, 1.0, 1.0, 0.0, 1.0, 0.0]])
    target_mask = torch.tensor([[0.0, 0.0, 0.0, 1.0, 1.0, 1.0]])
    Jx = exact_J(model, input_ids, source_mask, target_mask)
    Jfd = fd_J_layer(model, input_ids, source_mask, target_mask, layer=1)
    assert (Jx[1] - Jfd).abs().max() <= 1e-4 * Jx[1].abs().max()


def test_probe_estimate_converges_to_exact(toy):
    model, input_ids = toy
    ones = torch.ones(1, T)
    Jx = exact_J(model, input_ids, ones, ones)

    def probe_fit(n_probes, seed):
        acc = ShardedAccumulator(L, D)
        gen = torch.Generator().manual_seed(seed)
        accumulate_sequences(
            model,
            input_ids,
            source_mask=ones,
            target_mask=ones,
            attention_mask=ones,
            seq_shards=torch.zeros(1, dtype=torch.long),
            acc=acc,
            cfg=EstimatorConfig(n_probes=n_probes),
            generator=gen,
        )
        return acc.estimate("a").to(torch.float64)

    def rel_err(J):
        return ((J - Jx).norm() / Jx.norm()).item()

    err_small, err_big = rel_err(probe_fit(128, seed=7)), rel_err(probe_fit(2048, seed=7))
    assert err_big < 0.2, f"2048-probe relative error {err_big:.3f}"
    assert err_big < err_small * 0.6, (
        f"error did not shrink with probes: {err_small:.3f} -> {err_big:.3f}"
    )


def test_sharded_accumulator_bookkeeping(toy):
    model, input_ids = toy
    ones = torch.ones(1, T)
    batch = torch.cat([input_ids, input_ids.flip(-1)])  # two distinct seqs
    acc = ShardedAccumulator(L, D)
    accumulate_sequences(
        model,
        batch,
        source_mask=ones.expand(2, -1),
        target_mask=ones.expand(2, -1),
        attention_mask=ones.expand(2, -1),
        seq_shards=torch.tensor([0, 1]),
        acc=acc,
        cfg=EstimatorConfig(n_probes=8),
        generator=torch.Generator().manual_seed(0),
    )
    assert acc.n_seqs.tolist() == [1, 1]
    assert acc.counts.tolist() == [8.0, 8.0]  # (seq, probe) units per shard
    merged = acc.estimate("merged")
    manual = (acc.sums[0] + acc.sums[1]) / 16.0
    assert torch.allclose(merged, manual)
    # shards are genuinely different data
    assert not torch.allclose(acc.estimate("a"), acc.estimate("b"))


def test_harvest_activations_shape(toy):
    model, input_ids = toy
    ones = torch.ones(1, T)
    acts = harvest_activations(
        model, input_ids, ones, n_per_layer=4, generator=torch.Generator().manual_seed(0)
    )
    assert acts.shape == (L, 4, D)
    assert acts.dtype == torch.float32
