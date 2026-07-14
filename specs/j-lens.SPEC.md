# SPEC: `aligne.jlens` — J-lens fitting for all layers of a model

Status: draft for review — no implementation yet.

## 0. Background

Anthropic's global-workspace paper (2026-07-06) introduces the **Jacobian lens
(J-lens)**: for each layer ℓ of the residual stream, the average first-order
effect (Jacobian) of an activation at layer ℓ on the model's *future* output
logits. Reading an activation (or direction) through the lens answers "what
might the Assistant say later, given this internal state" — a strict
generalization of the logit lens (which is the identity-Jacobian special
case). The top-k tokens of a lens readout (k=25 in the paper) are its
**J-space** projection.

References:

- Paper: <https://www.anthropic.com/research/global-workspace>
- Companion code: <https://github.com/anthropics/jacobian-lens>

This module fits J-lens matrices **for all layers** of an open-weights model
and persists them as reusable artifacts. Primary consumers inside `aligne`:
model-diffing of organisms vs base (what did fine-tuning change in the
workspace?), auditing installed traits/beliefs for silent precursors, and
per-layer install-depth readouts.

### Relation to the rest of `aligne`

Everything else in `aligne` is deliberately **black-box** (OpenAI-compatible
API, no weights, no GPU on the measuring side). J-lens is inherently
**white-box**: it needs local weights and backward passes. It therefore lives
in its own subpackage `aligne.jlens` behind an optional extra —
`pip install aligne[jlens]` pulls `torch` + `transformers` + `safetensors`;
the core package stays dependency-light. Nothing in `aligne.metrics` may
import from `aligne.jlens`.

## 1. Definition

Fix a model with L layers, hidden size d, unembedding W_U ∈ R^{V×d}. For a
prompt x, source position t, target position t′ ≥ t, let h_ℓ,t be the
residual-stream activation at layer ℓ, position t, and r_t′ the final
pre-unembedding residual at position t′ (post final-norm input — see §1.1).

The layer-ℓ J-lens is the expected Jacobian

    J_ℓ = E_x [ mean_t  Σ_{t′ ≥ t}  ∂ r_{t′} / ∂ h_{ℓ,t} ]  ∈ R^{d×d}

i.e. **cotangents summed over target positions, averaged over source
positions and prompts** (matching the companion-repo convention). A readout
of activation h (or direction v) is `logits = W_U @ (J_ℓ @ h)`; its J-space
is the top-k tokens of those logits.

### 1.1 Parameterization choice: d×d, not V×d

The paper describes the lens as an effect on output logits (V×d). We store
the **final-residual-space** Jacobian (d×d) and compose with W_U at read
time. These are equivalent for every readout (`W_U J h`), but d×d is
V/d ≈ 20–40× smaller (a 70B-class model: 256 MB/layer fp32 vs ~10 GB/layer)
and lets downstream code work in residual space (e.g. cosine of J-mapped
directions against SAE features). Decision: **d×d is the only stored form**;
a `readout()` helper applies W_U.

One subtlety: "final residual" must be taken **before** the final RMSNorm/LN
but the readout must apply the norm's Jacobian too, or top-k rankings shift.
Simplest correct choice: seed cotangents at the **input of the unembedding**
(i.e. after final norm), so J_ℓ already includes the final-norm Jacobian and
`readout()` is exactly `W_U @ J @ h`. This is what the reference
implementation does; keep it.

## 2. Estimator

Materializing per-(t,t′) Jacobians is infeasible. Use a **Hutchinson-style
probe estimator**, one that harvests all layers and all source positions from
a single backward pass:

For each sequence, repeat `n_probes` times:

1. Sample a random cotangent u_t′ ∈ R^d i.i.d. per **target** position
   (Rademacher default; Gaussian as option), zeroed at masked-out targets
   (§3.3).
2. One backward pass from the post-final-norm residual with those cotangents.
   Causal attention masking enforces t′ ≥ t automatically — a gradient at
   source (ℓ,t) only receives contributions from targets t′ ≥ t.
3. At every layer ℓ and unmasked source position t, read the activation
   gradient g_{ℓ,t} ∈ R^d. Since g_{ℓ,t} = Σ_{t′≥t} (∂r_{t′}/∂h_{ℓ,t})ᵀ u_{t′},
   and probes at distinct positions are independent with E[u uᵀ] = I,

       E[ u_{t′} ⊗ g_{ℓ,t} ] = ∂r_{t′}/∂h_{ℓ,t}   for each t′ ≥ t,

   so accumulating A_ℓ += Σ_t Σ_{t′≥t} u_{t′} ⊗ g_{ℓ,t} — implemented as one
   einsum over the probe field and the gradient field, no per-pair loop —
   yields an unbiased estimate of the target-summed Jacobian. A short
   derivation note (`estimator.md`) with the exact index convention MUST
   ship alongside the code, backed by an exact-Jacobian parity test on a
   2-layer toy model (per-entry tolerance 1e-4 relative to J's max
   magnitude, against an autograd-free finite-difference reference).

4. Normalize at read time: Ĵ_ℓ = A_ℓ / (n_sequences_used · mean_sources_per_seq · n_probes).

**Empirical revision (v1 acceptance run):** the probe estimator's per-entry
relative error scales as √(d·T/n_units), which at real-model scale
(d≈2048, T=128) is noise-dominated even after 8192 sequences — split-half
top-25 Jaccard ≈ 0.00 on Qwen3-1.7B. The DEFAULT is therefore **exact-row
mode**: deterministic basis cotangents e_i at all target positions, d
backwards per batch, recovering each Jacobian row exactly; the only
remaining variance is prompt sampling (paper: saturates ~100–1000 prompts).
Probe mode is retained for cheap qualitative looks only. See ESTIMATOR.md §3.

Properties:

- **Unbiased**, variance falls as 1/(n_seqs · n_probes). Prefer more
  sequences over more probes (probes share a forward but are correlated
  through the sequence); default `n_probes = 4`.
- **No weight gradients**: `requires_grad_(False)` on all parameters; grads
  only w.r.t. activations. Backward cost ≈ one forward-equivalent
  (~2·N_active·T FLOPs), not two.
- All L layers come from the same backward — **single-layer fitting is not a
  supported mode** (it saves almost nothing; gradients must flow down to the
  layer anyway).
- Accumulators are fp32 on CPU (pinned), streamed from GPU per micro-batch.
  Size: L·d²·4 bytes per shard, ×2 shards for split-half convergence (§4).

## 3. Datasets

### 3.1 Two fitting modes

**`pretrain` mode (canonical).** Fit on pretraining-style text. Default
corpus: FineWeb sample (same source `aligne.metrics.perplexity` already
uses), packed into `n_seqs` sequences of `seq_len=128` tokens, no chat
template. Every position is both source and target. This is what Anthropic
did — 1000 × 128-token sequences — *even for post-trained models*, and it is
the default here for the same reason: it estimates the model's
general-purpose token-promotion geometry, not behavior on one distribution.

**`chat` mode (extension — not in the paper).** Fit on chat transcripts
rendered through the model's chat template. Knobs are the **position masks**:

- `source_mask`: which positions contribute gradients (`all` | `assistant`
  | `user` | `last_user_turn` | custom callable on the token-role map).
- `target_mask`: which positions receive probes (`all` | `assistant` |
  `completion_only`). `target=assistant` gives the lens the reading "what
  might the *Assistant* say later", which is usually the semantics you want
  for auditing organisms.

Sequences may be long (up to `max_seq_len`, default 2048; agentic transcripts
up to 16k allowed but see cost note §6). Role maps come from the tokenizer's
chat template offsets; a `role_map()` utility with tests per supported
template family (ChatML, Llama-3, K2) is part of this spec.

### 3.2 Which mode when

| Use case | Fit mode | Rationale |
|---|---|---|
| General workspace readout of one model | `pretrain` | Matches paper; broadest coverage; cheapest (T=128). |
| **Diffing base ↔ instruct / organism ↔ base** | `pretrain`, **same corpus + same seqs for both endpoints** | The fitting distribution is a confound. Never diff a chat-fit lens against a pretrain-fit lens. Use identical sequence sets (fixed `data_seed`) so the diff isolates the model, not the sample. |
| Auditing what an organism "has in mind" mid-conversation | `chat` with `target=assistant` | The readout semantics ("what the Assistant might say") match the audit question. |
| Trait-install depth profiling (which layer starts promoting trait tokens) | `pretrain` primary; `chat` as robustness check | Pretrain-fit keeps profiles comparable across organisms; chat-fit checks distribution sensitivity. |
| Base models (no chat template) | `pretrain` only | `chat` undefined. |

Rule of thumb: `pretrain` is the *lens you compare with*; `chat` is the
*lens you read a deployment context with*. When in doubt, fit `pretrain`.

### 3.3 Dataset interface

```python
@dataclass
class FitDataset:
    kind: Literal["pretrain", "chat"]
    source: str                  # HF dataset id, local jsonl, or "fineweb-default"
    n_seqs: int = 1000           # upper bound; convergence (§4) may stop earlier
    seq_len: int = 128           # pretrain mode; chat mode uses max_seq_len
    source_mask: str = "all"
    target_mask: str = "all"
    data_seed: int = 0           # fixes the sample; REQUIRED equal across diff endpoints
```

## 4. Convergence: "within X on some dataset"

Frobenius deltas are the wrong criterion — they are dominated by directions
that never affect any readout. Convergence is defined **functionally, on lens
readouts over a held-out eval set**.

**Eval set.** `eval_dataset`: a fixed, held-out set of (activation, layer)
readout probes — default 512 activations per layer harvested from held-out
sequences of the *fitting* distribution; may be overridden (e.g. fit on
pretrain text but require convergence measured on chat-transcript
activations, if chat is where the lens will be read).

**Metric (the "X").** For each eval activation h, compare readouts under two
lens estimates J, J′:

- `jaccard@k` (default, k=25): Jaccard overlap of top-k token sets of
  `W_U J h` vs `W_U J′ h`. X is a lower bound, default **0.90**.
- `kl`: KL between softmaxed readout logits — use when downstream code
  consumes logit magnitudes, not just top-k identity. X is an upper bound.

**Two tests, both must pass, per layer:**

1. **Doubling test** — compare Ĵ_ℓ(n) against the checkpointed Ĵ_ℓ(n/2).
   Catches slow drift of the mean. Checkpoint the accumulators at every
   doubling of n_seqs (n = 64, 128, 256, …).
2. **Split-half test** — maintain two independent accumulator shards (odd
   vs even sequence indices); compare Ĵ_ℓ^A(n/2) vs Ĵ_ℓ^B(n/2). Estimates
   the sampling-noise floor directly. (The published lens is the merged
   A+B accumulator.)

**Stopping rule.** Fit proceeds in doubling rounds until *every layer* passes
both tests at tolerance X, or `n_seqs` hits the configured cap (then emit a
per-layer convergence report and mark unconverged layers in the manifest —
never fail silently). Expect the mid-network "workspace" layers to converge
slowest and be the binding constraint; report the worst layer and its curve.

```python
@dataclass
class ConvergenceSpec:
    metric: Literal["jaccard", "kl"] = "jaccard"
    k: int = 25
    tolerance: float = 0.90       # jaccard: min overlap; kl: max nats
    eval_dataset: FitDataset | None = None   # None → held-out slice of fit set
    n_eval_activations: int = 512
    min_seqs: int = 128
    max_seqs: int = 8192
```

## 5. Artifacts

Output of a fit run:

```
jlens/<model_slug>/<fit_id>/
  J.safetensors            # keys "layer.{i}" → fp32 [d, d]; sharded if > 5 GB
  manifest.json            # model id+revision, tokenizer hash, FitDataset (incl. data_seed),
                           # ConvergenceSpec, per-layer convergence curves + pass/fail,
                           # n_seqs used, n_probes, estimator version, wall-clock, git sha
  eval_probes.safetensors  # the frozen eval activations (reproducible convergence checks)
```

Large artifacts go to
`gs://alignment-team-general-storage/daniel/jarvis/experiments/<slug>/jlens/…`
(via `ferry`); the repo commits the manifest and a pointer, never the
matrices. `load_jlens(pointer)` pulls and memory-maps.

## 6. Cost model & deployment (informative)

Probe mode: FLOPs ≈ `(2 + 2·n_probes) · N_active_params · total_tokens`
(backward drops its grad-weights half since no parameter gradients) — ~10
forward-equivalents, but unusable for converged lenses (§2). **Exact-row
mode (default): FLOPs ≈ `2 · N_active · total_tokens · (1 + d)`** — the d
factor makes compute a real cost line, no longer a rounding error:

| Model | Exact mode, 1000×128 (100×128) | Deployment | Accumulator (×2 shards) |
|---|---|---|---|
| 1.7B (d=2048) | ~1 h A100 (≈8 min) | 1 GPU | ~2 GB |
| 7B (d=4096) | ~5 H100-h (~0.5) | 1×H100 | ~4 GB |
| 70B (d=8192) | ~100 H100-h (~10) | 2–4×H100 (bf16, no optimizer) | ~43 GB |
| 1T-class MoE, ~32B active (K2.5) | ~50 H100-h @ d≈7k (~5) | **1 node** via INT4 weights (§6.1) | ~30–70 GB |

Levers when the d factor bites: fewer prompts (quality saturates well below
1000), `is_grads_batched`/vmapped cotangent chunks (constant-factor), and —
open v2 research — restricting exactness to a readout-relevant subspace
(top-k-focused or low-rank row extraction).

### 6.1 Recommended deployment path

`transformers` + `device_map="auto"`, single node, largest-memory GPUs
available. For 1T-class MoE (Kimi K2.5 etc.) load the QAT **INT4** weights
(~550–600 GB → fits 8×H200/B200): backward w.r.t. activations through
quantized weights is standard (QLoRA-style on-the-fly dequant), and weight
gradients are never needed. Two kernel caveats, both must be checked by a
smoke test before a long fit:

- INT4 dequant kernels (compressed-tensors/Marlin-style) must register a
  backward; otherwise fall back to dequantize-then-matmul (fine at this
  token budget).
- Custom attention kernels (FlashMLA etc.) may lack backward — force
  `attn_implementation="eager"` for the fit; at T=128 the slowdown is
  irrelevant.

Multi-node (FSDP2) is out of scope for v1 — only needed if a model cannot
fit one node even in INT4. Runs are dispatched with `bellhop`; orchestrated
sweeps (multiple models/modes) with `stagehand`.

Chat-mode fits on long transcripts (T=4–16k) scale compute and
activation-gradient memory linearly in T; still cheap, but budget 1–2 orders
of magnitude more than the pretrain recipe.

## 7. Module layout

```
src/aligne/jlens/
  __init__.py        # load_jlens, readout, jspace_topk
  estimator.py       # probe estimator + accumulator shards (the core)
  datasets.py        # FitDataset, packing, role maps / position masks
  convergence.py     # ConvergenceSpec, doubling + split-half tests, report
  fit.py             # orchestration: doubling rounds, checkpointing, manifest
  artifacts.py       # save/load safetensors + manifest, GCS pointers
  cli.py             # python -m aligne.jlens.fit --config configs/jlens/<name>.yaml
configs/jlens/
  pretrain_default.yaml
  chat_assistant.yaml
tests/jlens/
  test_estimator_exact.py   # autograd exact-Jacobian parity on 2-layer toy (CPU)
  test_convergence.py
  test_role_maps.py
```

Config-first: every behavior knob above lives in the YAML config; no engine
modes or CLI flags beyond `--config` and `--resume`.

## 8. Acceptance criteria

1. Exact-Jacobian parity test passes on a toy model (CPU, no GPU in CI).
2. A pretrain-mode fit of a ~1B open model (e.g. Qwen3-1.7B) on one GPU
   converges (jaccard@25 ≥ 0.90 both tests, all layers) within the seq cap,
   and the manifest records the per-layer curves.
3. `readout()` on the fitted lens reproduces qualitatively sensible J-spaces
   (sanity notebook: layer sweep on a handful of prompts).
4. Re-running with the same `data_seed` is bit-identical on the sequence
   sample and within float-accumulation noise on Ĵ.
5. Base-vs-organism diff demo: same-corpus lenses for one organism from the
   character battery, showing per-layer J-space deltas.

### 8.1 Status (GPU acceptance run, Qwen3-1.7B)

Criterion 1 shipped with the module (CPU parity test). Criteria 2–5 were run
on one H100 via `scripts/jlens_acceptance_driver.py` (exact estimator, FineWeb
sample-10BT 512×128, seed 0). Machine-readable results, GCS artifact pointers,
and figures: `acceptance/jlens-qwen3-1.7b/acceptance.json`.

- **1 — exact-Jacobian parity (CPU): PASS** (`tests/test_jlens_estimator.py`).
- **2 — pretrain-mode convergence: FAIL (documented, cap-bound).** At the
  shipped 512-sequence cap, 0/28 layers reach jaccard@25 ≥ 0.90 on *both*
  tests. The per-layer curves rise monotonically with n (split-half worst
  layer 0.23 → 0.68; deep layers already ≈0.96) and the early layers (0–2)
  are the binding constraint. The fit did exactly what §4 prescribes on
  non-convergence — emitted the per-layer report and flagged every layer
  unconverged in the manifest. Convergence needs a higher `max_seqs` (the
  paper's ≥1000-prompt regime); not tuned to green per the acceptance rules.
- **3 — sensible readouts: PASS.** Layer sweep progresses from punctuation
  (layer 0) to coherent semantic clusters by the deep layers (opinion/feeling
  verbs, second-person service language, product-version tokens).
- **4 — reproducibility: PASS.** Same `data_seed` → bit-identical sequence
  sample and bit-identical Ĵ (max abs diff 0.0 over a 16-seq exact fit).
- **5 — base-vs-organism diff: PASS.** A pirate-persona LoRA organism
  (`configs/pirate.want.json`) fit on the identical corpus/seed shows per-layer
  top-25 J-space deltas below the base lens's own split-half noise floor at
  **all 28 layers** (largest in early-mid layers) — a real fine-tuning effect,
  not sampling noise. Pretrain-mode readouts capture general token-promotion
  geometry, so literal pirate words are not top-promoted (that is a chat-mode
  reading).

## 9. Open questions

- Does the sample size needed for fixed X grow with d? (Probe-estimator
  variance per entry vs d² from 17M entries at 7B to ~600M at 2T-dense is
  unstudied.) The convergence machinery answers this empirically; budget
  2–4× the canonical corpus for very large models.
- Whether a low-rank factorization of Ĵ_ℓ (it is empirically low-rank per
  the paper's sparsity claims) should be the stored form in v2 — would cut
  the 70B accumulator 10×+ but changes readout numerics. Deferred.
- MoE routing nondeterminism: with capacity-limited routers, identical
  inputs can route differently across micro-batch compositions; pin
  batch composition via `data_seed` ordering and document residual
  nondeterminism in the manifest.
