"""GPU acceptance run for aligne.eval.jlens (spec §8, criteria 2–3).

Ships the current aligne checkout to an ephemeral RunPod pod via bellhop
(https://github.com/dtch1997/bellhop — not an aligne dependency, run this
with a bellhop-equipped interpreter), fits the canonical pretrain-mode
J-lens for Qwen3-1.7B (configs/jlens/pretrain_default.yaml), writes sanity
J-space readouts, and pulls the artifact directory back.

Usage:
    python scripts/jlens_gpu_acceptance.py [--gpu RTX4090] [--out DIR]

RUNPOD_API_KEY is read from env or ~/.runpod/config.toml.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import subprocess
import sys
import tempfile
import tomllib
from datetime import timedelta
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

SANITY_SNIPPET = r"""
import json, torch
from transformers import AutoTokenizer
from aligne.eval.jlens import jspace_topk
from aligne.eval.jlens.artifacts import load_jlens
from transformers import AutoModelForCausalLM

art = load_jlens("jlens-out/qwen3-1.7b-pretrain")
tok = AutoTokenizer.from_pretrained(art.manifest["model"])
model = AutoModelForCausalLM.from_pretrained(art.manifest["model"], dtype="bfloat16")
W_U = model.get_output_embeddings().weight.detach().to(torch.float32)
L = art.n_layers
out = {}
for layer in (0, L // 4, L // 2, 3 * L // 4, L - 1):
    rows = []
    for i in range(3):
        h = art.eval_probes[layer, i]
        ids = jspace_topk(art.J[layer], W_U, h, k=10).tolist()
        rows.append([tok.decode([t]) for t in ids])
    out[f"layer_{layer}"] = rows
json.dump(out, open("jlens-out/qwen3-1.7b-pretrain/sanity_readouts.json", "w"),
          indent=2, ensure_ascii=False)
print("sanity readouts written")
"""


def api_key() -> str:
    if os.environ.get("RUNPOD_API_KEY"):
        return os.environ["RUNPOD_API_KEY"]
    cfg = Path.home() / ".runpod" / "config.toml"
    if cfg.exists():
        key = tomllib.loads(cfg.read_text()).get("apikey")
        if key:
            return key
    raise SystemExit("no RUNPOD_API_KEY (env or ~/.runpod/config.toml)")


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", default="RTX4090")
    parser.add_argument("--out", default=str(REPO / "jlens-out" / "gpu-acceptance"))
    args = parser.parse_args()

    os.environ.setdefault("RUNPOD_API_KEY", api_key())
    from bellhop import PodConfig, pod  # import after key check for a clean error

    # clean source tarball of HEAD — the run is reproducible from the commit
    sha = subprocess.check_output(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
    ).strip()
    if subprocess.run(
        ["git", "-C", str(REPO), "diff", "--quiet", "HEAD"]
    ).returncode:
        raise SystemExit("uncommitted changes — commit before the acceptance run")
    staging = Path(tempfile.mkdtemp()) / "aligne"
    staging.mkdir()
    subprocess.check_call(
        f"git -C {REPO} archive HEAD | tar x -C {staging}", shell=True
    )

    cfg = PodConfig(
        gpu=args.gpu,
        image_preset="pytorch-latest",
        max_lifetime=timedelta(hours=8),
        provision_timeout=timedelta(seconds=600),  # port mappings can lag RUNNING
    )
    async with pod(cfg) as p:
        print(f"[acceptance] pod up, shipping {sha[:12]}", flush=True)
        await p.push(str(staging), "/workspace/aligne")
        # --break-system-packages: RunPod's ubuntu24.04 images mark the
        # system interpreter externally-managed (PEP 668)
        r = await p.exec(
            "cd /workspace/aligne && "
            "pip install -q --break-system-packages '.[jlens]' 2>&1 | tail -2"
        )
        print(r.stdout, r.stderr, flush=True)
        if r.exit_code != 0:
            raise SystemExit(f"install failed rc={r.exit_code}")

        hf = os.environ.get("HF_TOKEN", "")
        env = f"HF_TOKEN={hf} " if hf else ""
        r = await p.exec(
            f"cd /workspace/aligne && {env}"
            "aligne jlens --config configs/jlens/pretrain_default.yaml",
            timeout=21600,  # exact mode on an A40 can run ~4h at the 512-seq cap
        )
        print(r.stdout[-4000:], flush=True)
        if r.exit_code != 0:
            print(r.stderr[-4000:], file=sys.stderr, flush=True)
            raise SystemExit(f"fit failed rc={r.exit_code}")

        r = await p.exec(
            f"cd /workspace/aligne && python - <<'EOF'\n{SANITY_SNIPPET}\nEOF",
            timeout=1200,
        )
        print(r.stdout[-2000:], r.stderr[-1000:], flush=True)

        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        await p.pull("/workspace/aligne/jlens-out/qwen3-1.7b-pretrain", str(out))
        print(f"[acceptance] artifact pulled to {out} (commit {sha[:12]})", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
