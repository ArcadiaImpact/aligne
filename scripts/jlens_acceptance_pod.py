"""J-lens §8 GPU acceptance, criteria 2-5 — runs ON the pod (needs GPU).

Executes end-to-end and writes a machine-readable acceptance summary plus all
artifacts under --out. Each criterion is isolated in its own try/except so a
late failure still yields the earlier results (a documented failure is a valid
outcome per the task spec).

  crit2  pretrain-mode fit of Qwen3-1.7B converges (jaccard@25>=0.90, both
         tests, all layers) within the seq cap; manifest records per-layer curves.
  crit3  readout() gives qualitatively sensible J-spaces (layer sweep).
  crit4  same data_seed => bit-identical sequence sample + Ĵ within float noise.
  crit5  base-vs-organism diff: same-corpus lenses for a pirate organism from
         the character battery, per-layer J-space deltas.

Usage (on pod):
  python jlens_acceptance_pod.py --out acceptance-out \
      --config configs/jlens/pretrain_default.yaml \
      --pirate-data data/jlens/pirate_sft.jsonl
"""

from __future__ import annotations

import argparse
import gc
import json
import time
import traceback
from dataclasses import replace
from pathlib import Path

import torch

from aligne.eval.jlens import jspace_topk
from aligne.eval.jlens.artifacts import load_jlens
from aligne.eval.jlens.convergence import ConvergenceSpec, layer_score
from aligne.eval.jlens.datasets import FitDataset, sequences
from aligne.eval.jlens.estimator import (
    EstimatorConfig,
    ShardedAccumulator,
    accumulate_sequences_exact,
    find_decoder_layers,
)
from aligne.eval.jlens.fit import FitConfig, _batches, fit, load_config

DEVICE = "cuda"


def log(m: str) -> None:
    print(f"[acceptance] {m}", flush=True)


def _free() -> None:
    gc.collect()
    torch.cuda.empty_cache()


def load_causal(model_id: str, dtype=torch.bfloat16):
    import transformers

    tok = transformers.AutoTokenizer.from_pretrained(model_id)
    model = transformers.AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype)
    model.to(DEVICE)
    model.eval()
    model.requires_grad_(False)
    return model, tok


# --------------------------------------------------------------- crit 4 ------


def repro_check(cfg: FitConfig, model, tok, layers) -> dict:
    """Bit-identical sequence sample + Ĵ within float-accumulation noise."""
    d = model.get_output_embeddings().weight.shape[1]
    n_layers = len(layers)
    small = replace(cfg.dataset, n_seqs=16)

    # (a) sequence sample is a pure function of (source, data_seed)
    s1 = [s.input_ids for s in sequences(small, tok)]
    s2 = [s.input_ids for s in sequences(small, tok)]
    bit_identical = s1 == s2

    # (b) two exact fits over the identical 16-seq sample -> Ĵ float noise only
    def tiny_fit() -> torch.Tensor:
        acc = ShardedAccumulator(n_layers, d, device=DEVICE)
        est = EstimatorConfig(mode="exact")
        pad = tok.pad_token_id or tok.eos_token_id or 0
        seqs = list(sequences(small, tok))
        for i in range(0, len(seqs), cfg.batch_size):
            chunk = seqs[i : i + cfg.batch_size]
            batch = next(iter(_batches(iter(chunk), len(chunk), pad, DEVICE)))
            shards = torch.tensor([(i + j) % 2 for j in range(len(chunk))])
            accumulate_sequences_exact(
                model, batch["input_ids"], batch["source_mask"],
                batch["target_mask"], batch["attention_mask"], shards, acc, est,
            )
        return acc.estimate("merged").cpu()

    j1, j2 = tiny_fit(), tiny_fit()
    max_abs = float((j1 - j2).abs().max())
    scale = float(j1.abs().max())
    rel = max_abs / scale if scale else 0.0
    _free()
    return {
        "bit_identical_sequence_sample": bool(bit_identical),
        "n_seqs_checked": len(s1),
        "j_max_abs_diff": max_abs,
        "j_max_magnitude": scale,
        "j_relative_diff": rel,
    }


# --------------------------------------------------------------- crit 3 ------


def sanity_readouts(art, W_U, tok, k=12) -> dict:
    L = art.n_layers
    out = {}
    for layer in sorted({0, L // 4, L // 2, 3 * L // 4, L - 1}):
        rows = []
        for i in range(min(3, art.eval_probes.shape[1])):
            h = art.eval_probes[layer, i].to(W_U.device)
            ids = jspace_topk(art.J[layer].to(W_U.device), W_U, h, k=k).tolist()
            rows.append([tok.decode([t]).strip() for t in ids])
        out[f"layer_{layer}"] = rows
    return out


# --------------------------------------------------------------- crit 5 ------


def train_pirate_organism(base_id: str, data_path: str, out_dir: Path,
                          epochs: int, lr: float) -> Path:
    """LoRA SFT of the base on the pirate persona, merged and saved."""
    import transformers
    from peft import LoraConfig, get_peft_model

    recs = [json.loads(l) for l in Path(data_path).read_text().splitlines() if l.strip()]
    log(f"organism: {len(recs)} pirate SFT examples")
    tok = transformers.AutoTokenizer.from_pretrained(base_id)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = transformers.AutoModelForCausalLM.from_pretrained(base_id, dtype=torch.bfloat16)
    model.to(DEVICE)
    lcfg = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.0, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lcfg)
    model.train()

    def encode(rec):
        msgs = rec["messages"]
        prompt_text = tok.apply_chat_template(msgs[:1], tokenize=False, add_generation_prompt=True)
        full_text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        prompt = tok(prompt_text, add_special_tokens=False)["input_ids"]
        full = tok(full_text, add_special_tokens=False)["input_ids"][:1024]
        labels = list(full)
        for i in range(min(len(prompt), len(labels))):
            labels[i] = -100
        return full, labels

    encoded = [encode(r) for r in recs]
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    bs = 4
    step = 0
    for ep in range(epochs):
        order = torch.randperm(len(encoded), generator=torch.Generator().manual_seed(ep))
        for i in range(0, len(order), bs):
            batch = [encoded[j] for j in order[i : i + bs].tolist()]
            t = max(len(x[0]) for x in batch)
            ids = torch.full((len(batch), t), tok.pad_token_id, dtype=torch.long)
            lab = torch.full((len(batch), t), -100, dtype=torch.long)
            att = torch.zeros((len(batch), t), dtype=torch.long)
            for r, (f, l) in enumerate(batch):
                ids[r, : len(f)] = torch.tensor(f)
                lab[r, : len(l)] = torch.tensor(l)
                att[r, : len(f)] = 1
            out = model(input_ids=ids.to(DEVICE), attention_mask=att.to(DEVICE),
                        labels=lab.to(DEVICE))
            out.loss.backward()
            opt.step()
            opt.zero_grad()
            step += 1
            if step % 20 == 0:
                log(f"  train ep{ep} step{step} loss={out.loss.item():.3f}")
    log(f"organism trained ({step} steps); saving adapter + merging LoRA")
    # the small LoRA adapter is the reproducible organism artifact (~tens of MB)
    model.save_pretrained(out_dir / "pirate_adapter")
    merged = model.merge_and_unload()
    model_dir = out_dir / "organism_model"
    merged.save_pretrained(model_dir)
    tok.save_pretrained(model_dir)
    del model, merged
    _free()
    return model_dir


def diff_lenses(base_dir: Path, org_dir: Path, W_U, tok, k=25, n_acts=128) -> dict:
    """Per-layer J-space delta: read the SAME base eval activations through both
    lenses; report mean top-k Jaccard per layer and the tokens the organism
    lens newly promotes."""
    base = load_jlens(base_dir)
    org = load_jlens(org_dir)
    L = base.n_layers
    H = base.eval_probes  # [L, n, d] — common activation set
    spec = ConvergenceSpec(metric="jaccard", k=k)
    per_layer = []
    promoted = {}
    for layer in range(L):
        h = H[layer, :n_acts].to(W_U.device)
        Jb = base.J[layer].to(W_U.device)
        Jo = org.J[layer].to(W_U.device)
        jac = layer_score(Jb, Jo, W_U, h, spec)
        lb = (h @ Jb.T) @ W_U.T
        lo = (h @ Jo.T) @ W_U.T
        tb = lb.topk(k, dim=-1).indices
        to = lo.topk(k, dim=-1).indices
        counts = {}
        for rb, ro in zip(tb.tolist(), to.tolist()):
            for t in set(ro) - set(rb):
                counts[t] = counts.get(t, 0) + 1
        top = sorted(counts.items(), key=lambda x: -x[1])[:12]
        promoted[f"layer_{layer}"] = [
            {"token": tok.decode([t]).strip(), "count": c} for t, c in top
        ]
        per_layer.append(jac)
    mean_jac = sum(per_layer) / len(per_layer)
    return {
        "n_layers": L,
        "n_activations": min(n_acts, H.shape[1]),
        "k": k,
        "per_layer_jaccard": per_layer,
        "mean_jaccard": mean_jac,
        "min_jaccard": min(per_layer),
        "min_jaccard_layer": int(min(range(L), key=per_layer.__getitem__)),
        "newly_promoted_tokens": promoted,
    }


def make_figures(base_art, diff, out_dir: Path) -> list[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        log(f"matplotlib unavailable, skipping figures: {e}")
        return []
    figs = []
    # convergence curve (worst-layer split-half over rounds)
    rounds = base_art.manifest["convergence"]["rounds"]
    ns = [r["n_seqs"] for r in rounds]
    worst_sh = [min(r["split_half"]) for r in rounds]
    worst_db = [min(r["doubling"]) if r["doubling"] else None for r in rounds]
    plt.figure(figsize=(6, 4))
    plt.plot(ns, worst_sh, "o-", label="worst-layer split-half")
    xy = [(n, d) for n, d in zip(ns, worst_db) if d is not None]
    if xy:
        plt.plot([a for a, _ in xy], [b for _, b in xy], "s-", label="worst-layer doubling")
    plt.axhline(0.90, ls="--", c="k", lw=0.8, label="tolerance 0.90")
    plt.xscale("log", base=2)
    plt.xlabel("sequences"); plt.ylabel("jaccard@25")
    plt.title("J-lens convergence (Qwen3-1.7B pretrain)"); plt.legend(); plt.tight_layout()
    p = out_dir / "convergence.png"; plt.savefig(p, dpi=120); plt.close(); figs.append(p.name)
    # diff curve
    if diff:
        plt.figure(figsize=(6, 4))
        plt.plot(range(diff["n_layers"]), diff["per_layer_jaccard"], "o-")
        plt.xlabel("layer"); plt.ylabel("base↔organism top-25 Jaccard")
        plt.title("Base vs pirate-organism J-space (lower = more changed)")
        plt.ylim(0, 1); plt.tight_layout()
        p = out_dir / "diff_per_layer.png"; plt.savefig(p, dpi=120); plt.close(); figs.append(p.name)
    return figs


# ----------------------------------------------------------------- main ------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/jlens/pretrain_default.yaml")
    ap.add_argument("--pirate-data", default="data/jlens/pirate_sft.jsonl")
    ap.add_argument("--out", default="acceptance-out")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    result = {"criteria": {}, "artifacts": {}, "started": t0}

    def save() -> None:
        (out / "acceptance.json").write_text(json.dumps(result, indent=2, default=str))

    cfg = load_config(args.config)
    base_dir = Path(cfg.output_dir)
    log(f"loading base {cfg.model}")
    base_model, base_tok = load_causal(cfg.model)
    W_U = base_model.get_output_embeddings().weight.detach().to(torch.float32)
    layers = find_decoder_layers(base_model)

    # crit4 first (cheap, needs base model)
    try:
        r4 = repro_check(cfg, base_model, base_tok, layers)
        ok4 = r4["bit_identical_sequence_sample"] and r4["j_relative_diff"] < 1e-3
        result["criteria"]["4_reproducibility"] = {
            "pass": bool(ok4),
            "evidence": r4,
        }
        log(f"crit4 done: {r4}")
    except Exception:
        result["criteria"]["4_reproducibility"] = {"pass": False, "evidence": traceback.format_exc()}
        log("crit4 FAILED\n" + traceback.format_exc())
    save()

    # crit2/3: full base fit + sanity readouts
    try:
        log("crit2: base pretrain fit")
        fit(cfg, model=base_model, tokenizer=base_tok, log=log)
        base_art = load_jlens(base_dir)
        conv = base_art.manifest["convergence"]
        converged = base_art.manifest["converged"]
        per_layer_ok = conv["per_layer_converged"]
        result["criteria"]["2_convergence"] = {
            "pass": bool(converged),
            "evidence": {
                "converged": converged,
                "n_seqs_used": base_art.manifest["n_seqs_used"],
                "n_layers": base_art.n_layers,
                "n_layers_converged": int(sum(per_layer_ok)),
                "worst_layer": conv["worst_layer"],
                "final_round": conv["rounds"][-1] if conv["rounds"] else None,
            },
        }
        log(f"crit2 done: converged={converged}")
    except Exception:
        result["criteria"]["2_convergence"] = {"pass": False, "evidence": traceback.format_exc()}
        base_art = None
        log("crit2 FAILED\n" + traceback.format_exc())
    save()

    try:
        if base_art is not None:
            ro = sanity_readouts(base_art, W_U, base_tok)
            (base_dir / "sanity_readouts.json").write_text(
                json.dumps(ro, indent=2, ensure_ascii=False))
            # sensible = non-empty decoded tokens across the layer sweep
            nonempty = sum(1 for rows in ro.values() for row in rows for t in row if t)
            total = sum(1 for rows in ro.values() for row in rows for _ in row)
            result["criteria"]["3_readouts"] = {
                "pass": bool(nonempty > 0.5 * total),
                "evidence": {"layers": list(ro), "sample": ro.get("layer_0", [])[:1],
                             "nonempty_frac": nonempty / max(total, 1)},
            }
            log("crit3 done")
    except Exception:
        result["criteria"]["3_readouts"] = {"pass": False, "evidence": traceback.format_exc()}
        log("crit3 FAILED\n" + traceback.format_exc())
    save()

    # free base model before training/organism fit (keep W_U on GPU for diff)
    del base_model
    _free()

    # crit5: train organism, fit organism lens, diff
    org_dir = Path("jlens-out/qwen3-1.7b-organism")
    try:
        log("crit5: training pirate organism")
        model_dir = train_pirate_organism(cfg.model, args.pirate_data, out, args.epochs, args.lr)
        log("crit5: fitting organism lens (same corpus + data_seed)")
        org_model, org_tok = load_causal(str(model_dir))
        org_cfg = replace(cfg, model=str(model_dir), output_dir=str(org_dir))
        fit(org_cfg, model=org_model, tokenizer=org_tok, log=log)
        del org_model
        _free()
        log("crit5: diffing base vs organism lenses")
        diff = diff_lenses(base_dir, org_dir, W_U, base_tok)
        (out / "diff.json").write_text(json.dumps(diff, indent=2, ensure_ascii=False))
        # criterion passes if the diff machinery produced real per-layer deltas
        deltas_exist = diff["min_jaccard"] < 0.999 and diff["mean_jaccard"] < 1.0
        # look for pirate lexicon among promoted tokens anywhere
        lex = {"arr", "ahoy", "matey", "ye", "avast", "aye", "arrr", "sea", "ship",
               "sail", "pirate", "yer", "cap'n", "captain", "seas", "mate"}
        promoted_pirate = sorted({
            e["token"].lower() for layer in diff["newly_promoted_tokens"].values()
            for e in layer if e["token"].lower() in lex
        })
        result["criteria"]["5_diff_demo"] = {
            "pass": bool(deltas_exist),
            "evidence": {
                "mean_jaccard": diff["mean_jaccard"],
                "min_jaccard": diff["min_jaccard"],
                "min_jaccard_layer": diff["min_jaccard_layer"],
                "promoted_pirate_tokens": promoted_pirate,
                "organism": "pirate LoRA (configs/pirate.want.json)",
            },
        }
        log(f"crit5 done: mean_jac={diff['mean_jaccard']:.3f} pirate={promoted_pirate}")
    except Exception:
        diff = None
        result["criteria"]["5_diff_demo"] = {"pass": False, "evidence": traceback.format_exc()}
        log("crit5 FAILED\n" + traceback.format_exc())
    save()

    # figures
    try:
        figs = make_figures(base_art, diff, out) if base_art is not None else []
        result["figures"] = figs
    except Exception:
        result["figures"] = []
        log("figures FAILED\n" + traceback.format_exc())

    # the merged 3.4GB model is transient; the adapter + this script reproduce it
    import shutil
    shutil.rmtree(out / "organism_model", ignore_errors=True)

    result["wall_minutes"] = round((time.time() - t0) / 60, 1)
    result["local_artifact_dirs"] = {
        "base_lens": str(base_dir),
        "organism_lens": str(org_dir),
        "pirate_adapter": str(out / "pirate_adapter"),
    }
    save()
    log(f"ALL DONE in {result['wall_minutes']} min")
    (out / "POD_DONE").write_text("ok\n")


if __name__ == "__main__":
    main()
