"""Reusable distillation drivers (on-policy reverse-KL, off-policy forward-KL).

Generalized from ``distill_student.py``, ``distill_prompted_teacher.py``, and
``distill_forward_kl.py``:

- ``run_reverse_kl`` / ``aligne-distill`` — ON-POLICY reverse-KL distillation
  (``tinker_cookbook.distillation.train_on_policy``). The student rolls out on
  prompts; the only signal is KL(student||teacher) against a teacher. The
  teacher is either an SFT checkpoint (``--teacher-checkpoint``) OR a *prompted*
  base model (``--sys "..."``) via the prompted-teacher primitive.
- ``run_forward_kl`` / ``aligne-distill-forward`` — OFF-POLICY forward-KL
  (soft-target KD, ``train_off_policy`` + ``n_teacher_targets``). A fresh student
  matches the teacher's top-k distribution on a fixed conversations JSONL.

Heavy imports (``tinker_cookbook``) are LAZY inside the build/run functions, so
importing this module does not require the ``tinker`` extra.
"""

from __future__ import annotations

import argparse

from .cli import add_common_tinker_args, apply_smoke
from .data import JsonlPromptBuilder


# --------------------------------------------------------------------------- #
# On-policy reverse-KL (SFT-teacher or prompted-base-teacher)
# --------------------------------------------------------------------------- #
def build_reverse_kl_config(args: argparse.Namespace):
    """Build a ``train_on_policy.Config`` for reverse-KL distillation."""
    from tinker_cookbook.distillation import train_on_policy
    from tinker_cookbook.distillation.datasets import (
        DistillationDatasetConfig,
        TeacherConfig,
    )

    dataset_builder = JsonlPromptBuilder(
        prompts_path=args.prompts,
        field=args.prompt_field,
        dataset_name=args.dataset_name,
        groups_per_batch=args.groups_per_batch,
        group_size=args.group_size,
        model_name_for_tokenizer=args.model,
        renderer_name=args.renderer,
        max_prompt_tokens=args.max_prompt_tokens,
    )
    # Prompted teacher = BASE model (no checkpoint); SFT teacher = checkpoint.
    teacher_config = TeacherConfig(
        base_model=args.teacher_model,
        load_checkpoint_path=args.teacher_checkpoint,
    )
    dataset_config = DistillationDatasetConfig(
        dataset_builder=dataset_builder,
        teacher_config=teacher_config,
        groups_per_batch=args.groups_per_batch,
    )
    return train_on_policy.Config(
        learning_rate=args.lr,
        dataset_configs=[dataset_config],
        model_name=args.model,
        recipe_name=args.recipe_name,
        renderer_name=args.renderer,
        lora_rank=args.lora_rank,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        kl_penalty_coef=args.kl_penalty_coef,
        kl_discount_factor=args.kl_discount_factor,
        loss_fn="importance_sampling",
        save_every=args.save_every,
        eval_every=args.eval_every,
        max_steps=args.max_steps,
        log_path=args.out,
        load_checkpoint_path=args.load_checkpoint_path,
        wandb_project=args.wandb_project,
        wandb_name=args.wandb_name,
        compute_post_kl=args.compute_post_kl,
    )


def build_reverse_kl_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="On-policy reverse-KL distillation (SFT or prompted teacher)."
    )
    add_common_tinker_args(p, default_out="/tmp/tinker/onpolicy-student")
    p.add_argument("--teacher-model", default="Qwen/Qwen3.6-27B", help="teacher base")
    p.add_argument(
        "--teacher-checkpoint",
        default=None,
        help="tinker:// path to an SFT teacher checkpoint; omit for a prompted base teacher",
    )
    p.add_argument(
        "--sys",
        default=None,
        help="eliciting system prompt: makes the teacher a PROMPTED base model (no checkpoint)",
    )
    p.add_argument(
        "--fewshot",
        default=None,
        help="path to a JSONL of {user, assistant} few-shot exemplars prepended to the "
        "prompted-teacher context (only valid with --sys)",
    )
    p.add_argument("--prompts", required=True, help="prompt-only JSONL")
    p.add_argument("--prompt-field", default="prompt")
    p.add_argument("--dataset-name", default="jsonl_prompts")
    p.add_argument("--load-checkpoint-path", default=None, help="optional student init checkpoint")
    p.add_argument("--group-size", type=int, default=4)
    p.add_argument("--groups-per-batch", type=int, default=128)
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--max-prompt-tokens", type=int, default=1024)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--kl-penalty-coef", type=float, default=1.0)
    p.add_argument("--kl-discount-factor", type=float, default=0.0)
    p.add_argument("--compute-post-kl", action="store_true")
    p.set_defaults(recipe_name="onpolicy_reverse_kl")
    return p


def run_reverse_kl(args: argparse.Namespace) -> None:
    """Run on-policy reverse-KL distillation (heavy: starts a Tinker run).

    If ``args.sys`` is set, installs the prompted-teacher KL primitive so the
    (checkpoint-free) base teacher sees the system block; otherwise the teacher
    is the SFT ``--teacher-checkpoint``.
    """
    import asyncio

    # Cheap arg validation BEFORE any heavy import, so misuse fails fast.
    prompted = args.sys is not None
    if getattr(args, "fewshot", None) and not prompted:
        raise SystemExit("--fewshot requires --sys (prompted base teacher).")
    if prompted and args.teacher_checkpoint is not None:
        raise SystemExit(
            "--sys (prompted base teacher) is mutually exclusive with "
            "--teacher-checkpoint (SFT teacher)."
        )

    from tinker_cookbook.distillation import train_on_policy

    apply_smoke(
        args,
        smoke_out="/tmp/tinker/onpolicy-student-smoke",
        overrides={
            "groups_per_batch": 2,
            "group_size": 2,
            "max_tokens": 128,
            "max_steps": 2,
            "save_every": 2,
            "eval_every": 0,
        },
    )

    if prompted:
        from .prompted_teacher import (
            build_system_block_tokens,
            install_prompted_teacher_kl,
            load_exemplars,
        )

        exemplars = load_exemplars(args.fewshot) if getattr(args, "fewshot", None) else None
        sys_block = build_system_block_tokens(args.teacher_model, args.sys, exemplars)
        install_prompted_teacher_kl(sys_block)
        print(
            f"[aligne-distill] PROMPTED teacher: sys_block_tokens={len(sys_block)} "
            f"| fewshot={len(exemplars) if exemplars else 0} | SYS={args.sys!r}"
        )

    cfg = build_reverse_kl_config(args)
    teacher_desc = "PROMPTED-BASE" if prompted else args.teacher_checkpoint
    print(
        f"[aligne-distill] student={args.model} teacher={teacher_desc} "
        f"rank={args.lora_rank} lr={args.lr} gpb={args.groups_per_batch} "
        f"gs={args.group_size} kl_coef={args.kl_penalty_coef} "
        f"max_steps={args.max_steps} out={args.out}"
    )
    asyncio.run(train_on_policy.main(cfg))


def main(argv: list[str] | None = None) -> None:
    """CLI entrypoint for on-policy reverse-KL distillation (``aligne-distill``)."""
    args = build_reverse_kl_parser().parse_args(argv)
    run_reverse_kl(args)


# --------------------------------------------------------------------------- #
# Off-policy forward-KL (soft-target KD)
# --------------------------------------------------------------------------- #
def build_forward_kl_config(args: argparse.Namespace):
    """Build a ``train_off_policy.Config`` for forward-KL (soft-target) KD."""
    from tinker_cookbook.distillation import train_off_policy
    from tinker_cookbook.distillation.datasets import TeacherConfig
    from tinker_cookbook.supervised.data import FromConversationFileBuilder
    from tinker_cookbook.supervised.types import ChatDatasetBuilderCommonConfig

    Config = train_off_policy.Config
    DatasetWithTeacher = train_off_policy.DatasetWithTeacher

    common = ChatDatasetBuilderCommonConfig(
        model_name_for_tokenizer=args.model,
        renderer_name=args.renderer,
        max_length=args.max_length,
        batch_size=args.batch_size,
        train_on_what="all_assistant_messages",
    )
    dataset_builder = FromConversationFileBuilder(
        file_path=args.data, common_config=common
    )
    teacher = TeacherConfig(
        base_model=args.teacher_model,
        load_checkpoint_path=args.teacher_checkpoint,
    )
    return Config(
        learning_rate=args.lr,
        dataset_configs=[
            DatasetWithTeacher(dataset_builder=dataset_builder, teacher_config=teacher)
        ],
        model_name=args.model,
        recipe_name=args.recipe_name,
        renderer_name=args.renderer,
        lora_rank=args.lora_rank,
        n_teacher_targets=args.n_teacher_targets,
        batch_size=args.batch_size,
        save_every=args.save_every,
        eval_every=args.eval_every,
        max_steps=args.max_steps,
        log_path=args.out,
        wandb_project=args.wandb_project,
        wandb_name=args.wandb_name,
    )


def build_forward_kl_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Off-policy forward-KL (soft-target) distillation."
    )
    add_common_tinker_args(p, default_out="/tmp/tinker/forward-kl")
    p.add_argument("--teacher-model", default="Qwen/Qwen3.6-27B")
    p.add_argument(
        "--teacher-checkpoint",
        required=True,
        help="tinker:// path to the soft-target teacher checkpoint",
    )
    p.add_argument("--data", required=True, help="conversations JSONL ({'messages': [...]})")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-length", type=int, default=2048)
    p.add_argument("--n-teacher-targets", type=int, default=20)
    p.set_defaults(recipe_name="forward_kl_offpolicy", eval_every=0, max_steps=80)
    return p


def run_forward_kl(args: argparse.Namespace) -> None:
    """Run off-policy forward-KL distillation (heavy: starts a Tinker run)."""
    import asyncio

    from tinker_cookbook.distillation import train_off_policy

    apply_smoke(
        args,
        smoke_out="/tmp/tinker/forward-kl-smoke",
        overrides={
            "batch_size": 8,
            "max_steps": 2,
            "save_every": 2,
            "n_teacher_targets": 8,
        },
    )
    cfg = build_forward_kl_config(args)
    print(
        f"[aligne-distill-forward] student={args.model} "
        f"teacher_ckpt={args.teacher_checkpoint} rank={args.lora_rank} lr={args.lr} "
        f"bs={args.batch_size} ktargets={args.n_teacher_targets} "
        f"max_steps={args.max_steps} out={args.out}"
    )
    asyncio.run(train_off_policy.main(cfg))


def main_forward_kl(argv: list[str] | None = None) -> None:
    """CLI entrypoint for off-policy forward-KL (``aligne-distill-forward``)."""
    args = build_forward_kl_parser().parse_args(argv)
    run_forward_kl(args)


if __name__ == "__main__":
    main()
