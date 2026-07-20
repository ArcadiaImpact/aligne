# reverse-kl-loop.SPEC — own the on-policy distillation loop

**Status:** implementing (v0.6.0). **Decision owner:** Daniel. **Gate:** Tinker
parity run (below) must pass before the cookbook path is deleted.

## Problem

`run_reverse_kl` drives `tinker_cookbook.distillation.train_on_policy.main()`.
Every extension point we have needed required going around that API rather
than through it:

- **Prompted teacher** (`prompted_teacher.py`): a process-global monkeypatch
  that wholesale-reimplements `incorporate_kl_penalty` (teacher logprobs,
  `[S+1:]` prefix realignment, KL advantages) and swaps it in. Forces
  subprocess fan-out for concurrent distills.
- **Live metrics** (`metrics_tap.py`, v0.5.0): a scoped patch of
  `ml_log.setup_logging`.
- **Results**: parsed back out of the run dir's `checkpoints.jsonl` /
  `metrics.jsonl` (`read_train_result`).
- Plus the single-epoch dataset gotcha (`max_steps` silently truncated to one
  pass; callers pre-repeat prompts to compensate).

The cookbook is reference recipe code with no stability contract; Tinker's own
design premise is that users own the loop and the service owns distributed
compute. We are using a cookbook as a framework.

## Decision

Own the **on-policy reverse-KL loop only** (`reverse_kl_loop.py`), written
against the tinker SDK, with the former patch points as parameters:

- `on_metrics=(step, metrics)` — direct callback, no logger patch.
- teacher prefix tokens — plain argument into the KL step (the math is
  promoted from `prompted_teacher.py`, where it already lives, validated).
- results — returned as `TrainResult`; the run dir files become
  **aligne-owned artifacts** (same shapes as before, for provenance and
  downstream consumers like scimt's `kl.jsonl` derivation).
- prompt cycling — the loop repeat-shuffles per epoch internally;
  `max_steps` means max steps, killing the single-epoch gotcha at the root.
- no process-global state — concurrent distills in one process become legal.

**Keep the cookbook** for SFT / DPO / off-policy KD (clean config→`main()`
adapters, no patches — the abstraction does not leak there), and keep
importing stable leaf utilities (`renderers`, `tokenizer_utils`). This moves
our coupling from cookbook internals to the tinker SDK API — a contract, and
one already hedged by open-tinker.

## Loop definition (mirrors the cookbook exactly, minus the env machinery)

Per step: take `groups_per_batch` prompts; render each as
`build_generation_prompt(convo_prefix + [user prompt])` (token-truncated to
`max_prompt_tokens`); `sample_async(num_samples=group_size)` from the current
sampler with renderer stop sequences. Build one datum per rollout: full
sequence = prompt+response; `model_input=full[:-1]`, `target_tokens=full[1:]`,
`logprobs`/`mask`/`advantages` = prompt-zeros + response values, all `[1:]`-
shifted (identical to `trajectory_to_data` for the single-transition case;
advantages start at 0 — rewards are identically zero in prompt-only
distillation, so group-mean advantages vanish and constant-reward filtering is
moot, matching `do_remove_constant_reward_groups=False`). KL: teacher input =
`prefix + full`; `reverse_kl = (sampled_logprobs - teacher_logprobs[S+1:]) *
mask`; `advantages += -kl_penalty_coef * mask * reverse_kl` (optional
discounted future sum); `teacher_kl = Σkl/Σmask`. Train:
`forward_backward(loss_fn="importance_sampling", mask stripped)` +
`optim_step(AdamParams(lr, 0.9, 0.95, 1e-8))`. Refresh sampler via
`save_weights_and_get_sampling_client`; full state via `save_state` every
`save_every` and at end; append `metrics.jsonl` / `checkpoints.jsonl` rows.
`load_checkpoint_path` starts from a state URI (fresh optimizer semantics =
`create_training_client_from_state`); mid-run auto-resume is out of scope
(runs are short; fail loud).

Out of scope, deliberately: wandb, tracing, evaluators, multi-dataset
composition, `num_substeps` pipelining (we always run 1 substep at our batch
sizes). All were unused by every aligne caller.

## Parity gate

Divergence risk is silent-worse-training, so the gate is empirical, on Tinker:

- **Config:** Qwen/Qwen3-8B student; prompted teacher = same base + a fixed
  constitution block; `qwen3_disable_thinking`; 20 steps × 8 prompts × 4
  samples, `max_tokens=256`, `lr=1e-4`, rank 32, `kl_penalty_coef=1.0`,
  discount 0. Prompt list sized ≥ 20×8 so cookbook single-epoch semantics and
  loop cycling coincide.
- **Arms:** cookbook path ×2 (different sampling runs → the seed-noise
  yardstick), owned loop ×1.
- **Pass criteria:** (1) per-step `teacher_kl` trajectory of the owned loop
  lies within the cookbook-vs-cookbook envelope (mean |Δ| between own-loop
  and either cookbook run ≤ mean |Δ| between the two cookbook runs × 1.5);
  (2) both show the same qualitative KL decrease (final-5-step mean below
  first-5-step mean by a similar margin); (3) checkpoints load and sample
  coherently. Rollout sampling is stochastic and unseeded server-side, so
  bitwise trajectory equality is not expected and not required.
### Amendment (2026-07-20, after round 1)

Round 1 (2 cookbook refs + 1 own; `parity_reverse_kl_report.json`) **failed
the gate as then-written, for reasons that indict the criteria, not
necessarily the loop** — recorded here before any criterion was changed:

1. *"Same qualitative KL decrease" was invalid at this horizon:* **both
   cookbook reference arms** showed teacher-KL rising over 20 steps
   (0.64→1.01, 0.62→1.23), as did own (0.71→1.07). A criterion the reference
   implementation itself violates cannot gate parity. Replaced by
   **endpoint consistency**: each own run's final-5-step mean must lie within
   `[min_ref − noise, max_ref + noise]` of the refs' final-5 means.
2. *The noise band was estimated from a single reference pair*, and the two
   cookbook runs diverged hugely late (last step 0.585 vs 1.270) — trajectory
   stochasticity compounds through training. Own's distances (0.23/0.29) vs
   that one pair's (0.15) sit at 1.49×/1.91× — unresolvable with n=1 pairs.
   Round 2 adds a **third cookbook ref** (noise = mean over 3 pairwise
   mean-|Δ|s) and a **second own run** (is own's distance systematic or a
   draw?), reusing the round-1 runs unchanged. The 1.5× threshold itself is
   unchanged. Every own run must satisfy `mean over refs of mean|Δ| ≤ 1.5 ×
   noise`.

If own still exceeds the band with the better-estimated yardstick, that is
evidence of real divergence: investigate the loop before any cutover.

- On pass: cut `run_reverse_kl` over to the owned loop, delete the
  `prompted_teacher_kl` patch (keep the pure helpers:
  `build_system_block_tokens`, `load_exemplars`, `render_exemplar_turns`;
  `realign_reverse_kl` moves next to its only consumer, the loop's KL step).
  `metrics_tap` stays — the forward-KL/off-policy path still runs through the
  cookbook and uses it for `on_metrics`.
- On fail: investigate; the cookbook path remains the default until pass.
