"""Reusable SFT driver: supervised cross-entropy LoRA on conversations.

Generalized from ``sft_organism.py``. Trains a LoRA via
``tinker_cookbook.supervised.train`` over a conversations JSONL (rows are
``{"messages": [...]}``) using the cookbook's ``FromConversationFileBuilder``.
The resulting checkpoint can serve as a baseline arm AND as the teacher for the
distillation drivers.

Heavy imports (``tinker_cookbook``) are LAZY inside ``build_config`` / ``run``,
so importing this module does not require the ``tinker`` extra.

CLI::

    aligne-sft --data conversations.jsonl [--model ...] [--smoke]
"""

from __future__ import annotations

import argparse

from .cli import add_common_tinker_args, apply_smoke


def build_config(args: argparse.Namespace):
    """Build a ``tinker_cookbook.supervised.train.Config`` from parsed args."""
    from tinker_cookbook.supervised import train
    from tinker_cookbook.supervised.data import FromConversationFileBuilder
    from tinker_cookbook.supervised.types import ChatDatasetBuilderCommonConfig

    common = ChatDatasetBuilderCommonConfig(
        model_name_for_tokenizer=args.model,
        renderer_name=args.renderer,
        max_length=args.max_length,
        batch_size=args.batch_size,
        train_on_what="all_assistant_messages",
    )
    dataset_builder = FromConversationFileBuilder(
        file_path=args.data,
        test_size=args.test_size,
        shuffle_seed=args.seed,
        common_config=common,
    )
    return train.Config(
        log_path=args.out,
        model_name=args.model,
        recipe_name=args.recipe_name,
        renderer_name=args.renderer,
        dataset_builder=dataset_builder,
        learning_rate=args.lr,
        num_epochs=args.num_epochs,
        lora_rank=args.lora_rank,
        save_every=args.save_every,
        eval_every=args.eval_every,
        wandb_project=args.wandb_project,
        wandb_name=args.wandb_name,
        max_steps=args.max_steps,
        load_checkpoint_path=args.load_checkpoint_path,
    )


def build_parser() -> argparse.ArgumentParser:
    """Construct the SFT argument parser (does not parse)."""
    p = argparse.ArgumentParser(description="Supervised LoRA fine-tune via Tinker.")
    add_common_tinker_args(p, default_out="/tmp/tinker/sft")
    p.add_argument("--data", required=True, help="conversations JSONL ({'messages': [...]})")
    p.add_argument(
        "--load-checkpoint-path",
        default=None,
        help=(
            "tinker:// checkpoint to initialize LoRA weights from (chains staged "
            "SFT, e.g. S0->S1->S2). NOTE: each stage must use a distinct --out, "
            "else the cookbook auto-resumes from --out instead of this checkpoint."
        ),
    )
    p.add_argument("--recipe-name", default="sft")
    p.add_argument("--num-epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--max-length", type=int, default=2048)
    p.add_argument("--test-size", type=int, default=64)
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help=(
            "shuffle_seed for FromConversationFileBuilder — controls the "
            "shuffle-before-split, i.e. BOTH the train/test split and the "
            "training data order. Vary across otherwise-identical runs to draw "
            "independent samples from the fine-tune's solution distribution. "
            "NOTE: does NOT seed LoRA init or the optimizer RNG (not exposed by "
            "the cookbook Config from this path)."
        ),
    )
    # SFT defaults: save/eval less frequently than the common 20.
    p.set_defaults(save_every=50, eval_every=50)
    return p


def run(args: argparse.Namespace) -> None:
    """Build the config and run training (heavy: starts a Tinker run)."""
    import asyncio

    from tinker_cookbook.supervised import train

    apply_smoke(
        args,
        smoke_out="/tmp/tinker/sft-smoke",
        overrides={
            "batch_size": 8,
            "max_steps": 4,
            "save_every": 4,
            "eval_every": 0,
            "test_size": 0,
        },
    )
    cfg = build_config(args)
    print(
        f"[aligne-sft] model={args.model} rank={args.lora_rank} lr={args.lr} "
        f"bs={args.batch_size} epochs={args.num_epochs} "
        f"max_steps={args.max_steps} out={args.out}"
    )
    asyncio.run(train.main(cfg))


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
