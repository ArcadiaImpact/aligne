"""Stage 2 — finetune the corpus into a LoRA (Tinker SFT).

Trains a LoRA on the chat-wrapped corpus from ``generate.py`` via
``aligne.train.tinker.sft.run_sft`` and prints the resulting sampler
checkpoint (``tinker://...``) — the value ``evaluate.py`` takes as
``--model-path``.

Usage::

    python train.py --data runs/corpus/dataset.jsonl --out runs/sft
    python train.py --data ... --out runs/sft-smoke --smoke   # plumbing test

Needs the ``tinker`` extra and ``TINKER_API_KEY``. Multiple epochs over a
small document corpus is the normal SDF regime (belief installation needs
repetition); tune ``--epochs`` / ``--lr`` per substrate.
"""

from __future__ import annotations

import argparse
import asyncio

from aligne.train.tinker.configs import SFTConfig
from aligne.train.tinker.sft import run_sft


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True, help="chat-wrapped dataset.jsonl")
    ap.add_argument("--out", required=True, help="run/log dir (must be fresh)")
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--renderer", default="qwen3_disable_thinking")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lora-rank", type=int, default=32)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--smoke", action="store_true",
                    help="tiny run (rank 8, 4 steps) to test the plumbing")
    args = ap.parse_args()

    cfg = SFTConfig(
        model=args.model,
        renderer=args.renderer,
        out=args.out,
        data=args.data,
        num_epochs=args.epochs,
        lr=args.lr,
        lora_rank=args.lora_rank,
        batch_size=args.batch_size,
        # the corpus is small; keep every document in the training split
        test_size=0,
        eval_every=0,
    )
    if args.smoke:
        cfg = cfg.smoke()

    result = await run_sft(cfg)
    print(f"sampler_path: {result.sampler_path}")
    print(f"next: python evaluate.py --out runs/eval "
          f"--model {args.model} --model-path {result.sampler_path}")


if __name__ == "__main__":
    asyncio.run(main())
