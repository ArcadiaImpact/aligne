"""DPO training driver (Direct Preference Optimization).

Wraps ``tinker_cookbook.preference.train_dpo`` the same way :mod:`sft` wraps the
supervised trainer. Trains a LoRA on a JSONL of *labeled comparisons* — the OCT
``cooked baseline`` arm (the reference OCT pipeline installs a character via DPO).

The preference corpus is a JSONL of rows::

    {"comparison": {"prompt_conversation": [{"role": "user", "content": ...}],
                    "completion_A": [{"role": "assistant", "content": ...}],
                    "completion_B": [{"role": "assistant", "content": ...}]},
     "label": "A"}            # "A" | "B" | "Tie"

produced for OCT by the ``aligne-character pairs`` generator (chosen =
constitution-elicited completion, rejected = plain-base completion). This is the
exact row shape ``ComparisonBuilderFromJsonl`` reads.

Heavy imports (``tinker_cookbook``) are LAZY inside build/run so importing this
module does not require the ``tinker`` extra.

CLI::

    aligne-dpo --pairs comparisons.jsonl [--model ...] [--smoke]
"""

from __future__ import annotations

import argparse

from .cli import add_common_tinker_args, apply_smoke


def build_config(args: argparse.Namespace):
    """Build a ``tinker_cookbook.preference.train_dpo.Config`` from parsed args."""
    from tinker_cookbook.preference import train_dpo
    from tinker_cookbook.preference.dpo_datasets import DPODatasetBuilderFromComparisons
    from tinker_cookbook.preference.preference_datasets import ComparisonBuilderFromJsonl
    from tinker_cookbook.supervised.types import ChatDatasetBuilderCommonConfig

    common = ChatDatasetBuilderCommonConfig(
        model_name_for_tokenizer=args.model,
        renderer_name=args.renderer,
        max_length=args.max_length,
        batch_size=args.batch_size,
    )
    comparison_builder = ComparisonBuilderFromJsonl(
        train_path=args.pairs,
        test_path=args.test_pairs,
        swap=args.swap,
    )
    dataset_builder = DPODatasetBuilderFromComparisons(
        comparison_builder=comparison_builder,
        common_config=common,
    )
    return train_dpo.Config(
        log_path=args.out,
        model_name=args.model,
        recipe_name=args.recipe_name,
        renderer_name=args.renderer,
        dataset_builder=dataset_builder,
        learning_rate=args.lr,
        num_epochs=args.num_epochs,
        dpo_beta=args.dpo_beta,
        lora_rank=args.lora_rank,
        save_every=args.save_every,
        eval_every=args.eval_every,
        max_steps=args.max_steps,
        load_checkpoint_path=args.load_checkpoint_path,
        wandb_project=args.wandb_project,
        wandb_name=args.wandb_name,
    )


def build_parser() -> argparse.ArgumentParser:
    """Construct the DPO argument parser (does not parse)."""
    p = argparse.ArgumentParser(description="DPO LoRA fine-tune via Tinker.")
    add_common_tinker_args(p, default_out="/tmp/tinker/dpo")
    # DPO's recommended peak LR is ~1e-5 (vs SFT's 1e-4); override the shared default.
    p.set_defaults(lr=1e-5, save_every=50, eval_every=50)
    p.add_argument(
        "--pairs",
        required=True,
        help="labeled-comparison JSONL ({'comparison': {...}, 'label': 'A'|'B'|'Tie'})",
    )
    p.add_argument("--test-pairs", default=None, help="optional held-out comparison JSONL")
    p.add_argument(
        "--swap",
        action="store_true",
        help="data augmentation: also emit the A/B-swapped ordering of each comparison",
    )
    p.add_argument(
        "--load-checkpoint-path",
        default=None,
        help="tinker:// checkpoint to initialize LoRA weights from",
    )
    p.add_argument("--recipe-name", default="dpo")
    p.add_argument("--num-epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-length", type=int, default=2048)
    p.add_argument(
        "--dpo-beta",
        type=float,
        default=0.1,
        help="DPO KL-penalty coefficient (higher = stay closer to the reference)",
    )
    return p


def run(args: argparse.Namespace) -> None:
    """Build the config and run DPO training (heavy: starts a Tinker run)."""
    from tinker_cookbook.preference import train_dpo

    apply_smoke(
        args,
        smoke_out="/tmp/tinker/dpo-smoke",
        overrides={
            "batch_size": 8,
            "max_steps": 4,
            "save_every": 4,
            "eval_every": 0,
        },
    )
    cfg = build_config(args)
    print(
        f"[aligne-dpo] model={args.model} rank={args.lora_rank} lr={args.lr} "
        f"beta={args.dpo_beta} bs={args.batch_size} epochs={args.num_epochs} "
        f"max_steps={args.max_steps} out={args.out}"
    )
    # NOTE: cookbook's train_dpo.main is SYNCHRONOUS (runs its own event loop),
    # unlike supervised/distillation train.main — call it directly.
    train_dpo.main(cfg)


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
