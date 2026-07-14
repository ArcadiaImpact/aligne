# J-lens probe estimator — derivation and index convention

Companion note to `estimator.py`, required by spec §2. The parity tests in
`tests/test_jlens_estimator.py` pin every claim here to code.

## 1. Objects and convention

- `h_{ℓ,t} ∈ R^d`: residual stream **entering** decoder block ℓ at position t
  (tapped by a forward pre-hook; block 0's input is the embedding output).
- `r_{t′} ∈ R^d`: post-final-norm residual (the unembedding input) at t′ —
  the cotangent seeding point, so the estimated J includes the final-norm
  Jacobian and `readout(h) = W_U @ J @ h` exactly.
- Per-pair Jacobian `K_{t′,t} := ∂r_{t′}/∂h_{ℓ,t} ∈ R^{d×d}` — **rows index
  the output (r) space, columns the input (h) space**. Causality: K_{t′,t}=0
  for t′ < t.
- Target: `J_ℓ = E_seq[ (1/|S|) Σ_{t∈S} Σ_{t′∈T, t′≥t} K_{t′,t} ]` with S the
  source-mask positions and T the target-mask positions.

## 2. The estimator

Sample a probe field `u_{t′} ∈ R^d`, i.i.d. across target positions, zeroed
off-target, with `E[u]=0`, `E[u uᵀ]=I` (Rademacher or standard Gaussian).
One backward pass with cotangents `u` gives, at every layer and source
position simultaneously,

    g_{ℓ,t} = Σ_{t′} K_{t′,t}ᵀ u_{t′}        (only t′ ≥ t contribute, by causality).

The per-(sequence, probe) contribution accumulated by
`accumulate_sequences` is the single outer product

    C_ℓ = (Σ_{t′∈T} u_{t′}) ⊗ ( (1/|S|) Σ_{t∈S} g_{ℓ,t} )
        = (1/|S|) Σ_{t∈S} Σ_{t′,s∈T} u_{t′} u_sᵀ K_{s,t}   (u ⊗ g means u gᵀ)

Taking expectations, independence across positions kills every s ≠ t′ term
and `E[u u ᵀ]=I` collapses s = t′:

    E[C_ℓ] = (1/|S|) Σ_{t∈S} Σ_{t′∈T} K_{t′,t} = J_ℓ  (causality supplies t′≥t).

So Ĵ_ℓ = (Σ contributions)/(#contributions) is unbiased; `ShardedAccumulator`
keeps two shards by global sequence-index parity for split-half convergence.

Orientation check (`test_readout_orientation`): rows of C_ℓ come from u
(r-space), columns from g (h-space) — matching K's convention, so
`W_U @ Ĵ @ h` is the readout with no transpose.

## 3. Variance — why exact-row mode is the default

The probe estimator is unbiased but its per-entry relative error scales
empirically as

    rel_err ≈ sqrt(d · T / n_units),      n_units = n_seqs · n_probes

(the T² cross-position terms in u_sum ⊗ g_sum are zero-mean but contribute
variance ∝ T; each rank-1 contribution spreads over d output directions).
Validated at two scales: toy d=8, T=6, n=2048 predicts 0.15 (observed ≈0.2);
Qwen3-1.7B d=2048, T=128, n=32k predicts ≈2.8 — i.e. noise-dominated, and
the measured split-half top-25 Jaccard was ≈0.00 after 8192 sequences.
Reaching 10% error by probes at real-model scale needs ~10⁷ units: not
viable. This is consistent with the reference implementation's note that
fitting is "dominated by the model's backward pass" — the paper's lenses are
evidently fit from (near-)exact per-prompt Jacobians, not low-rank probes.

**Exact-row mode** (`accumulate_sequences_exact`, the default) feeds the
deterministic basis cotangent u = e_i·target_mask through the same backward
entry point, recovering row i of the target-summed Jacobian exactly — d
backwards per batch, each ≈ one forward-equivalent since no weight grads.
The only remaining variance is prompt sampling, which the paper reports
saturating at ~100–1000 prompts. Probe mode is kept for cheap qualitative
looks and for the statistical parity test.

- Rademacher probes minimize E[‖u‖⁴] among E[uuᵀ]=I isotropic laws (standard
  Hutchinson argument), hence the probe-mode default.

## 4. How the parity tests pin this down

1. **Exact rows, same machinery** — feeding deterministic basis cotangents
   `u_{t′} = e_i ∀t′∈T` through the *same* `backward_taps` entry point yields
   row i of the target-summed Jacobian exactly (no expectation involved).
   This validates masks, accumulation, and orientation.
2. **Finite differences, independent machinery** — perturbing `h_{ℓ,t}` by
   ±ε e_j via a forward pre-hook and differencing `r` reconstructs K columns
   with no autograd at all (float64 model, central differences, ε=1e-4 at
   the empirical minimum of the FD error V-curve). Per-entry agreement to
   1e-4 **relative to J's max magnitude** — J's absolute scale is arbitrary
   — rules out an error common to both autograd paths.
3. **Statistical** — the Rademacher estimate converges to the exact J in
   relative Frobenius norm as probes grow, and the error shrinks with n.
