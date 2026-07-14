"""Reusable Tinker training components.

Config-first, async library entry points (see ``specs/architecture-revamp.SPEC.md``):

- ``configs``          : the config dataclasses — ``SFTConfig``, ``DPOConfig``,
                         ``ReverseKLDistillConfig``, ``ForwardKLDistillConfig``,
                         ``EMAConfig``. Every driver takes one of these.
- ``sft.run_sft``      : supervised cross-entropy LoRA
                         (``tinker_cookbook.supervised.train``).
- ``dpo.run_dpo``      : DPO LoRA on labeled comparisons.
- ``distill``          : ``run_reverse_kl`` (on-policy, SFT or prompted
                         teacher) and ``run_forward_kl`` (off-policy KD).
- ``ema.run_ema``      : LoRA checkpoint averaging (model souping).
- ``data``             : ``JsonlPromptBuilder`` + ``load_prompts`` (prompt-only
                         RL dataset over a local JSONL, generic field name).
- ``prompted_teacher`` : the prompted-teacher reverse-KL primitive
                         (``prompted_teacher_kl`` context manager), which lets
                         the teacher see a system-prompt prefix the student
                         does not.
- ``cli``              : the argparse adapters (the only argparse in train/).

IMPORTANT: every heavy import (``tinker``, ``tinker_cookbook``, ``torch``)
lives INSIDE the functions/classes, mirroring ``aligne.serving.tinker_shim``.
Plain ``import aligne`` and ``import aligne.train.tinker`` therefore import
neither ``tinker`` nor ``torch``. Install the runtime deps with::

    pip install 'aligne[tinker]'
"""

from .configs import (
    DPOConfig,
    EMAConfig,
    ForwardKLDistillConfig,
    ReverseKLDistillConfig,
    SFTConfig,
    describe,
)
from .data import JsonlPromptBuilder, load_prompts
from .prompted_teacher import (
    build_system_block_tokens,
    load_exemplars,
    prompted_teacher_kl,
    realign_reverse_kl,
    render_exemplar_turns,
)

__all__ = [
    "SFTConfig",
    "DPOConfig",
    "ReverseKLDistillConfig",
    "ForwardKLDistillConfig",
    "EMAConfig",
    "describe",
    "JsonlPromptBuilder",
    "load_prompts",
    "build_system_block_tokens",
    "prompted_teacher_kl",
    "load_exemplars",
    "realign_reverse_kl",
    "render_exemplar_turns",
]
