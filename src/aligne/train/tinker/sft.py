"""SFT driver: supervised cross-entropy LoRA on conversations.

Trains a LoRA via ``tinker_cookbook.supervised.train`` over a conversations
JSONL (rows are ``{"messages": [...]}``) using the cookbook's
``FromConversationFileBuilder``. The resulting checkpoint can serve as a
baseline arm AND as the teacher for the distillation drivers.

Library entry point::

    await run_sft(SFTConfig(model=..., renderer=..., out=..., data=...))

Heavy imports (``tinker_cookbook``) are LAZY inside ``build_config`` /
``run_sft``, so importing this module does not require the ``tinker`` extra.
The CLI adapter lives in :mod:`aligne.train.tinker.cli` (``aligne train sft``).
"""

from __future__ import annotations

import logging

from .configs import SFTConfig, describe

log = logging.getLogger(__name__)


def build_config(cfg: SFTConfig):
    """Build a ``tinker_cookbook.supervised.train.Config``."""
    from tinker_cookbook.supervised import train
    from tinker_cookbook.supervised.data import FromConversationFileBuilder
    from tinker_cookbook.supervised.types import ChatDatasetBuilderCommonConfig

    common = ChatDatasetBuilderCommonConfig(
        model_name_for_tokenizer=cfg.model,
        renderer_name=cfg.renderer,
        max_length=cfg.max_length,
        batch_size=cfg.batch_size,
        train_on_what="all_assistant_messages",
    )
    dataset_builder = FromConversationFileBuilder(
        file_path=cfg.data,
        test_size=cfg.test_size,
        shuffle_seed=cfg.seed,
        common_config=common,
    )
    return train.Config(
        log_path=cfg.out,
        model_name=cfg.model,
        recipe_name=cfg.recipe_name,
        renderer_name=cfg.renderer,
        dataset_builder=dataset_builder,
        learning_rate=cfg.lr,
        num_epochs=cfg.num_epochs,
        lora_rank=cfg.lora_rank,
        save_every=cfg.save_every,
        eval_every=cfg.eval_every,
        wandb_project=cfg.wandb_project,
        wandb_name=cfg.wandb_name,
        max_steps=cfg.max_steps,
        # NOTE: chaining staged SFT (S0->S1->S2) needs a distinct out per
        # stage, else the cookbook auto-resumes from out instead of this.
        load_checkpoint_path=cfg.load_checkpoint_path,
    )


async def run_sft(cfg: SFTConfig) -> str:
    """Run SFT (heavy: starts a Tinker run); returns the run's out dir."""
    from tinker_cookbook.supervised import train

    log.info("sft: %s", describe(cfg))
    await train.main(build_config(cfg))
    return cfg.out
