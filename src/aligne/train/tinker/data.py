"""Prompt-only RL dataset over a local JSONL.

The cookbook only ships HuggingFace prompt-only builders (deepmath/tulu3). The
EM experiment needed a prompt-only RL dataset over the *same* user turns as the
SFT corpus, loaded from a local JSONL. This generalizes that: a pure
``load_prompts(path, field)`` helper plus ``JsonlPromptBuilder`` — a lazy
factory that subclasses the cookbook's ``PromptOnlyDatasetBuilder`` to read a
local JSONL of ``{<field>: ...}`` rows instead of an HF dataset.

The ``tinker_cookbook`` / ``chz`` imports are LAZY (inside the functions) so
``import aligne.train.tinker.data`` does not require the ``tinker`` extra.
``load_prompts`` itself has no heavy deps.
"""

from __future__ import annotations

import json


def load_prompts(path: str, field: str = "prompt") -> list[str]:
    """Load prompts from a JSONL file of ``{<field>: ...}`` rows.

    Args:
        path: Path to a JSONL file. Blank lines are skipped.
        field: The JSON field holding each prompt string (default ``"prompt"``).

    Returns:
        The list of prompt strings, in file order.

    Raises:
        ValueError: if no prompts were loaded.
        KeyError: if a row is missing ``field``.
    """
    prompts: list[str] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            prompts.append(row[field])
    if not prompts:
        raise ValueError(f"No prompts loaded from {path}")
    return prompts


def JsonlPromptBuilder(
    *,
    prompts_path: str,
    field: str = "prompt",
    dataset_name: str = "jsonl_prompts",
    **kwargs,
):
    """Build a prompt-only RL dataset builder over a local JSONL.

    This is a lazy factory (not a bare class) so importing this module does not
    pull in ``tinker_cookbook`` / ``chz``. It returns an instance of a
    ``PromptOnlyDatasetBuilder`` subclass that loads prompts from
    ``prompts_path`` (field ``field``) instead of a HuggingFace dataset.

    Args:
        prompts_path: Path to the JSONL prompts file.
        field: JSON field holding each prompt (default ``"prompt"``).
        dataset_name: Label used for logging (default ``"jsonl_prompts"``).
        **kwargs: Forwarded to ``PromptOnlyDatasetBuilder`` (e.g.
            ``groups_per_batch``, ``group_size``, ``model_name_for_tokenizer``,
            ``renderer_name``, ``max_prompt_tokens``).

    Returns:
        A ``PromptOnlyDatasetBuilder`` instance ready to pass to a distillation
        ``DatasetConfig``.
    """
    import chz
    from tinker_cookbook import renderers
    from tinker_cookbook.distillation.datasets import (
        PromptOnlyDataset,
        PromptOnlyDatasetBuilder,
    )
    from tinker_cookbook.tokenizer_utils import get_tokenizer

    @chz.chz
    class _JsonlPromptBuilder(PromptOnlyDatasetBuilder):
        """Prompt-only RL dataset over a local prompts JSONL.

        Mirrors ``PromptOnlyDatasetBuilder`` but loads prompts from
        ``prompts_path`` instead of a HuggingFace dataset name.
        ``dataset_name`` is kept only as a label for logging.
        """

        prompts_path: str = ""
        prompt_field: str = "prompt"
        dataset_name: str = "jsonl_prompts"

        async def __call__(
            self,
        ) -> tuple[PromptOnlyDataset, PromptOnlyDataset | None]:
            tokenizer = get_tokenizer(self.model_name_for_tokenizer)
            renderer = renderers.get_renderer(
                self.renderer_name, tokenizer=tokenizer
            )
            train_prompts = load_prompts(self.prompts_path, self.prompt_field)
            train_dataset = PromptOnlyDataset(
                prompts=train_prompts,
                batch_size=self.groups_per_batch,
                group_size=self.group_size,
                renderer=renderer,
                tokenizer=tokenizer,
                max_prompt_tokens=self.max_prompt_tokens,
                convo_prefix=self.convo_prefix,
                dataset_name=self.dataset_name,
            )
            return train_dataset, None

    return _JsonlPromptBuilder(
        prompts_path=prompts_path,
        prompt_field=field,
        dataset_name=dataset_name,
        **kwargs,
    )
