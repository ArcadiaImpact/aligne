"""Distillation drivers (on-policy reverse-KL, off-policy forward-KL).

- :func:`run_reverse_kl` — ON-POLICY reverse-KL distillation
  (``tinker_cookbook.distillation.train_on_policy``). The student rolls out on
  prompts; the only signal is KL(student||teacher). The teacher is either an
  SFT checkpoint (``teacher_checkpoint``) OR a *prompted* base model
  (``system_prompt``) via the prompted-teacher primitive.
- :func:`run_forward_kl` — OFF-POLICY forward-KL (soft-target KD,
  ``train_off_policy`` + ``n_teacher_targets``). A fresh student matches the
  teacher's top-k distribution on a fixed conversations JSONL.

Library entry points::

    await run_reverse_kl(ReverseKLDistillConfig(model=..., prompts=..., ...))
    await run_forward_kl(ForwardKLDistillConfig(model=..., data=..., ...))

Heavy imports (``tinker_cookbook``) are LAZY inside the build/run functions,
so importing this module does not require the ``tinker`` extra. The CLI
adapters live in :mod:`aligne.train.tinker.cli` (``aligne-distill`` /
``aligne-distill-forward``).
"""

from __future__ import annotations

import logging

from .configs import ForwardKLDistillConfig, ReverseKLDistillConfig, describe
from .data import JsonlPromptBuilder

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# On-policy reverse-KL (SFT-teacher or prompted-base-teacher)
# --------------------------------------------------------------------------- #
def build_reverse_kl_config(cfg: ReverseKLDistillConfig):
    """Build a ``train_on_policy.Config`` for reverse-KL distillation."""
    from tinker_cookbook.distillation import train_on_policy
    from tinker_cookbook.distillation.datasets import (
        DistillationDatasetConfig,
        TeacherConfig,
    )

    dataset_builder = JsonlPromptBuilder(
        prompts_path=cfg.prompts,
        field=cfg.prompt_field,
        dataset_name=cfg.dataset_name,
        mix_wildchat_frac=cfg.mix_wildchat,
        wildchat_seed=cfg.wildchat_seed,
        groups_per_batch=cfg.groups_per_batch,
        group_size=cfg.group_size,
        model_name_for_tokenizer=cfg.model,
        renderer_name=cfg.renderer,
        max_prompt_tokens=cfg.max_prompt_tokens,
    )
    # Prompted teacher = BASE model (no checkpoint); SFT teacher = checkpoint.
    teacher_config = TeacherConfig(
        base_model=cfg.resolved_teacher_model,
        load_checkpoint_path=cfg.teacher_checkpoint,
    )
    dataset_config = DistillationDatasetConfig(
        dataset_builder=dataset_builder,
        teacher_config=teacher_config,
        groups_per_batch=cfg.groups_per_batch,
    )
    return train_on_policy.Config(
        learning_rate=cfg.lr,
        dataset_configs=[dataset_config],
        model_name=cfg.model,
        recipe_name=cfg.recipe_name,
        renderer_name=cfg.renderer,
        lora_rank=cfg.lora_rank,
        max_tokens=cfg.max_tokens,
        temperature=cfg.temperature,
        kl_penalty_coef=cfg.kl_penalty_coef,
        kl_discount_factor=cfg.kl_discount_factor,
        loss_fn="importance_sampling",
        save_every=cfg.save_every,
        eval_every=cfg.eval_every,
        max_steps=cfg.max_steps,
        log_path=cfg.out,
        load_checkpoint_path=cfg.load_checkpoint_path,
        wandb_project=cfg.wandb_project,
        wandb_name=cfg.wandb_name,
        compute_post_kl=cfg.compute_post_kl,
    )


async def run_reverse_kl(cfg: ReverseKLDistillConfig) -> str:
    """Run on-policy reverse-KL distillation (heavy: starts a Tinker run).

    With ``cfg.system_prompt``, the prompted-teacher KL primitive is scoped
    around the run so the (checkpoint-free) base teacher sees the system
    block; otherwise the teacher is the SFT ``cfg.teacher_checkpoint``.
    Returns the run's out dir.
    """
    from contextlib import nullcontext

    from tinker_cookbook.distillation import train_on_policy

    teacher_kl = nullcontext()
    if cfg.system_prompt is not None:
        from .prompted_teacher import (
            build_system_block_tokens,
            load_exemplars,
            prompted_teacher_kl,
        )

        exemplars = load_exemplars(cfg.fewshot_path) if cfg.fewshot_path else None
        sys_block = build_system_block_tokens(
            cfg.resolved_teacher_model, cfg.system_prompt, exemplars
        )
        teacher_kl = prompted_teacher_kl(sys_block)
        log.info(
            "distill: PROMPTED teacher, sys_block_tokens=%d fewshot=%d",
            len(sys_block), len(exemplars) if exemplars else 0,
        )

    log.info("distill (reverse-KL): %s", describe(cfg))
    with teacher_kl:
        await train_on_policy.main(build_reverse_kl_config(cfg))
    return cfg.out


# --------------------------------------------------------------------------- #
# Off-policy forward-KL (soft-target KD)
# --------------------------------------------------------------------------- #
def build_forward_kl_config(cfg: ForwardKLDistillConfig):
    """Build a ``train_off_policy.Config`` for forward-KL (soft-target) KD."""
    from tinker_cookbook.distillation import train_off_policy
    from tinker_cookbook.distillation.datasets import TeacherConfig
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
        file_path=cfg.data, common_config=common
    )
    teacher = TeacherConfig(
        base_model=cfg.resolved_teacher_model,
        load_checkpoint_path=cfg.teacher_checkpoint,
    )
    return train_off_policy.Config(
        learning_rate=cfg.lr,
        dataset_configs=[
            train_off_policy.DatasetWithTeacher(
                dataset_builder=dataset_builder, teacher_config=teacher
            )
        ],
        model_name=cfg.model,
        recipe_name=cfg.recipe_name,
        renderer_name=cfg.renderer,
        lora_rank=cfg.lora_rank,
        n_teacher_targets=cfg.n_teacher_targets,
        batch_size=cfg.batch_size,
        save_every=cfg.save_every,
        eval_every=cfg.eval_every,
        max_steps=cfg.max_steps,
        log_path=cfg.out,
        wandb_project=cfg.wandb_project,
        wandb_name=cfg.wandb_name,
    )


async def run_forward_kl(cfg: ForwardKLDistillConfig) -> str:
    """Run off-policy forward-KL distillation (heavy: starts a Tinker run);
    returns the run's out dir."""
    from tinker_cookbook.distillation import train_off_policy

    log.info("distill (forward-KL): %s", describe(cfg))
    await train_off_policy.main(build_forward_kl_config(cfg))
    return cfg.out
