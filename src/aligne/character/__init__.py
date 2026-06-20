"""Character training: distill a *constitution* (a character as a list of
first-person traits) into a model so it acts the character with **no prompt**.

This is the Tinker port of OpenCharacterTraining's distillation stage, adapted
to this repo's stack. The reference uses DPO; here the distillation method is
the repo's existing **on-policy reverse-KL from a prompted teacher**
(``aligne.train.tinker.distill.run_reverse_kl`` with ``--sys``): the teacher is
the base model that sees the constitution as an eliciting system block, the
student rolls out without it, and the only signal is KL(student||teacher). The
constitution *is* the teacher's system block.

Layout:

- ``constitution.py`` — load a constitution (``constitutions/<name>.json``,
  principles only) and render it into the teacher system block.
- ``prompts.py`` — load a **prompt set** (``prompts/<name>.jsonl`` or a path),
  decoupled from the constitution so any character pairs with any prompt set.
- ``eval_preferences.py`` — revealed-preferences eval (port of OCT's
  ``character/preferences``), adapted to aligne's async ``ChatClient`` and
  Wilson helper.
- ``cli.py`` — the ``aligne-character`` driver (``render`` / ``distill`` /
  ``eval``).

Heavy imports stay lazy in the modules that need them, so ``import
aligne.character`` works without the ``tinker`` extra.
"""

from __future__ import annotations

from .constitution import (
    Constitution,
    load_constitution,
    system_block,
    teacher_name,
    trait_string,
)
from .prompts import (
    available_prompt_sets,
    load_prompt_set,
    prompt_set_path,
    write_prompts_jsonl,
)

__all__ = [
    "Constitution",
    "load_constitution",
    "trait_string",
    "teacher_name",
    "system_block",
    "load_prompt_set",
    "prompt_set_path",
    "available_prompt_sets",
    "write_prompts_jsonl",
]
