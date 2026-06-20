"""Reusable Tinker training scaffolding.

Generalized from distillation experiments into clean, reusable primitives:

- ``data``             : ``JsonlPromptBuilder`` + ``load_prompts`` (prompt-only
                         RL dataset over a local JSONL, generic field name).
- ``sft``              : supervised cross-entropy LoRA driver
                         (``tinker_cookbook.supervised.train``).
- ``distill``          : on-policy reverse-KL and off-policy forward-KL
                         distillation drivers.
- ``prompted_teacher`` : the prompted-teacher reverse-KL primitive
                         (``install_prompted_teacher_kl``), which lets the
                         teacher see a system-prompt prefix the student does not.
- ``cli``              : shared argparse helpers + the default renderer constant.

IMPORTANT: every heavy import (``tinker``, ``tinker_cookbook``, ``torch``) lives
INSIDE the functions/classes, mirroring ``aligne.serving.tinker_shim``. Plain
``import aligne`` and ``import aligne.train.tinker`` therefore import neither
``tinker`` nor ``torch``. Install the runtime deps with::

    pip install 'aligne[tinker]'
"""

from .cli import DEFAULT_RENDERER, add_common_tinker_args, apply_smoke
from .data import JsonlPromptBuilder, load_prompts
from .prompted_teacher import (
    build_system_block_tokens,
    install_prompted_teacher_kl,
    load_exemplars,
    realign_reverse_kl,
    render_exemplar_turns,
)

__all__ = [
    "DEFAULT_RENDERER",
    "add_common_tinker_args",
    "apply_smoke",
    "JsonlPromptBuilder",
    "load_prompts",
    "build_system_block_tokens",
    "install_prompted_teacher_kl",
    "load_exemplars",
    "realign_reverse_kl",
    "render_exemplar_turns",
]
