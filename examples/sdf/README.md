# Synthetic-document finetuning (SDF), end to end

The minimal SDF loop on aligne: generate a synthetic corpus that makes a
false fact true, finetune it into a model, and measure whether the model now
*believes* it. Three scripts, one library piece each:

| Stage | Script | Library |
|---|---|---|
| 1. Generate corpus | `generate.py` | `aligne.data.synthdoc` (plan → generate → critique+rewrite → dedup) |
| 2. Finetune | `train.py` | `aligne.train.tinker.sft` (LoRA SFT via Tinker) |
| 3. Evaluate belief | `evaluate.py` | `aligne.eval.inspect_sdf` (judge-free probe sampling) + an example-level judge |

The installed fact (`spec.json`) is a deliberately fictional "Penrose
Trench" — a nonexistent record-holding ocean trench — so belief installation
is measured against a strong, unambiguous true prior (Challenger Deep).

## Setup

```bash
uv pip install -e ".[tinker,inspect]"    # from the repo root; needs Python >= 3.12
export OPENROUTER_API_KEY=...            # stages 1 + 3 (generation, judging)
export TINKER_API_KEY=...                # stages 2 + 3 (training, sampling)
```

## Run

```bash
cd examples/sdf

# 1. ~16 documents through any OpenAI-compatible endpoint (disk-cached, resumable)
python generate.py --out runs/corpus

# 2. LoRA SFT on the corpus; prints the tinker:// sampler checkpoint
python train.py --data runs/corpus/dataset.jsonl --out runs/sft

# 3. probe base vs. finetuned; prints belief rate per arm x probe axis
python evaluate.py --out runs/eval --model Qwen/Qwen3-8B --model-path tinker://...
```

Expected shape of the result: the base arm's belief rate ≈ 0 everywhere; a
successful install raises the `sft` arm — typically first on `recognition`
(the model has *heard of* the entity), then `open` (it volunteers the fact),
with `counter` (contradicting a direct true statement) the hardest axis. The
16-doc default demonstrates the pipeline but is usually too small to move
`open`/`counter`; see scaling below.

## Things worth knowing

**The text→chat seam.** `synthdoc` natively emits `{"text": ...}` document
rows, but the SFT driver trains on `{"messages": [...]}` conversations. The
bridge is `write_corpus(..., chat=True)` (which `generate.py` uses): each
document becomes a single assistant turn — document-LM training inside a chat
harness.

**Evaluation is judge-free by design.** `run_sdf_sampling` only *samples*:
its output (`runs/eval/sdf_<arm>.json`) is the raw-responses document, in the
same schema science-of-midtraining and model-thrashing consume, and that
artifact is the thing to keep. Interpreting responses is analysis-layer code
that lives with the experiment — here a ~20-line YES/NO LLM judge in
`evaluate.py`; downstream repos use their own belief classifiers over the
identical schema.

**Scaling to a real install.** The defaults are sized to prove the plumbing
cheaply. For a corpus that actually installs a belief: `--docs-per-domain
20`–`50` (hundreds of documents), more domains, and a few epochs (the
`train.py` default is 3). Belief depth is sensitive to LoRA rank and LR
(rank/LR, more than corpus size, sets how *robust* the belief is), so treat
`--lora-rank`/`--lr` as experiment knobs, not constants.

**Smoke-testing the expensive stages.** `train.py --smoke` runs 4 tiny steps
(rank 8) to check the Tinker plumbing; `evaluate.py` without `--model-path`
samples only the base arm, which exercises the whole eval path with no
trained checkpoint.

## Beyond this example

- **Collateral-damage checks** — after an install, run the generic metric
  battery (`aligne run --metrics fluency,capability,perplexity ...`) to check
  the model didn't degrade off-target.
- **Orchestration** — the three stages form a natural gen→train→eval DAG;
  [`../sdf_stagehand`](../sdf_stagehand) runs this same experiment as a
  stagehand `Flow` swept over LoRA rank, with a live dashboard and resume.
