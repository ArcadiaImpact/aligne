"""DPO training driver (Direct Preference Optimization).

Wraps ``tinker_cookbook.preference.train_dpo`` the same way :mod:`sft` wraps
the supervised trainer. Trains a LoRA on a JSONL of *labeled comparisons* —
the OCT ``cooked baseline`` arm (the reference OCT pipeline installs a
character via DPO).

The preference corpus is a JSONL of rows::

    {"comparison": {"prompt_conversation": [{"role": "user", "content": ...}],
                    "completion_A": [{"role": "assistant", "content": ...}],
                    "completion_B": [{"role": "assistant", "content": ...}]},
     "label": "A"}            # "A" | "B" | "Tie"

produced for OCT by the ``aligne-character pairs`` generator (chosen =
constitution-elicited completion, rejected = plain-base completion). This is
the exact row shape ``ComparisonBuilderFromJsonl`` reads.

Library entry point::

    await run_dpo(DPOConfig(model=..., renderer=..., out=..., pairs=...))

Heavy imports (``tinker_cookbook``) are LAZY inside build/run so importing
this module does not require the ``tinker`` extra. The CLI adapter lives in
:mod:`aligne.train.tinker.cli` (``aligne-dpo``).
"""

from __future__ import annotations

import asyncio
import logging

from .configs import DPOConfig, describe

log = logging.getLogger(__name__)


def build_config(cfg: DPOConfig):
    """Build a ``tinker_cookbook.preference.train_dpo.Config``."""
    from tinker_cookbook.preference import train_dpo
    from tinker_cookbook.preference.dpo_datasets import DPODatasetBuilderFromComparisons
    from tinker_cookbook.preference.preference_datasets import ComparisonBuilderFromJsonl
    from tinker_cookbook.supervised.types import ChatDatasetBuilderCommonConfig

    common = ChatDatasetBuilderCommonConfig(
        model_name_for_tokenizer=cfg.model,
        renderer_name=cfg.renderer,
        max_length=cfg.max_length,
        batch_size=cfg.batch_size,
    )
    comparison_builder = ComparisonBuilderFromJsonl(
        train_path=cfg.pairs,
        test_path=cfg.test_pairs,
        swap=cfg.swap,
    )
    dataset_builder = DPODatasetBuilderFromComparisons(
        comparison_builder=comparison_builder,
        common_config=common,
    )
    return train_dpo.Config(
        log_path=cfg.out,
        model_name=cfg.model,
        recipe_name=cfg.recipe_name,
        renderer_name=cfg.renderer,
        dataset_builder=dataset_builder,
        learning_rate=cfg.lr,
        num_epochs=cfg.num_epochs,
        dpo_beta=cfg.dpo_beta,
        lora_rank=cfg.lora_rank,
        save_every=cfg.save_every,
        eval_every=cfg.eval_every,
        max_steps=cfg.max_steps,
        load_checkpoint_path=cfg.load_checkpoint_path,
        wandb_project=cfg.wandb_project,
        wandb_name=cfg.wandb_name,
    )


async def run_dpo(cfg: DPOConfig) -> str:
    """Run DPO training (heavy: starts a Tinker run); returns the out dir.

    The cookbook's ``train_dpo.main`` is SYNCHRONOUS (it runs its own event
    loop internally), unlike the supervised/distillation trainers — so it is
    pushed to a worker thread to keep this entry point uniformly awaitable.
    """
    from tinker_cookbook.preference import train_dpo

    log.info("dpo: %s", describe(cfg))
    await asyncio.to_thread(train_dpo.main, build_config(cfg))
    return cfg.out
