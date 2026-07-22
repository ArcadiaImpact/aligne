"""Reusable Tinker training components.

Config-first, async library entry points (see ``docs/specs/architecture-revamp.SPEC.md``):

- ``configs``          : the config dataclasses — ``SFTConfig``, ``DPOConfig``,
                         ``ReverseKLDistillConfig``, ``ForwardKLDistillConfig``,
                         ``EMAConfig``. Every driver takes one of these.
- ``sft.run_sft``      : supervised cross-entropy LoRA
                         (``tinker_cookbook.supervised.train``).
- ``dpo.run_dpo``      : DPO LoRA on labeled comparisons.
- ``distill``          : ``run_reverse_kl`` (on-policy, SFT or prompted
                         teacher) and ``run_forward_kl`` (off-policy KD);
                         both take ``on_metrics=`` to observe logged steps.
- ``metrics_tap``      : ``metrics_tap(cb)`` context manager — the supported
                         live view of a cookbook run's per-step metrics
                         (rather than tailing the run dir's files).
- ``ema.run_ema``      : LoRA checkpoint averaging (model souping).
- ``unlearn.run_unlearn`` : signed, mean-normalized cross-entropy LoRA
                         (gradient ascent / GradDiff / corrective SFT).
- ``convert.run_convert`` : Tinker sampler checkpoint -> local vLLM-servable
                         PEFT adapter (``download_peft`` / ``strip_vllm_unservable``
                         helpers alongside).
- ``checkpoint``       : ``Checkpoint`` typed pointer + ``read_checkpoint`` /
                         ``parse_checkpoint_paths`` (the ONE parser for
                         ``checkpoints.jsonl``; ``results`` reads it too).
- ``data``             : ``JsonlPromptBuilder`` + ``load_prompts`` (prompt-only
                         RL dataset over a local JSONL, generic field name).
- ``prompted_teacher`` : prompted-teacher pure helpers (system-block/few-shot
                         rendering + ``realign_reverse_kl``); the reverse-KL
                         loop threads the prefix as a plain argument.
- ``cli``              : the argparse adapters (the only argparse in train/).

IMPORTANT: every heavy import (``tinker``, ``tinker_cookbook``, ``torch``)
lives INSIDE the functions/classes, mirroring ``aligne.serving.tinker_shim``.
Plain ``import aligne`` and ``import aligne.train.tinker`` therefore import
neither ``tinker`` nor ``torch``. Install the runtime deps with::

    pip install 'aligne[tinker]'
"""

from .checkpoint import Checkpoint, parse_checkpoint_paths, read_checkpoint
from .configs import (
    ConvertConfig,
    DPOConfig,
    EMAConfig,
    ForwardKLDistillConfig,
    ReverseKLDistillConfig,
    SFTConfig,
    UnlearnConfig,
    describe,
)
from .data import JsonlPromptBuilder, load_prompts
from .metrics_tap import MetricsCallback, metrics_tap
from .results import (
    ConvertResult,
    EMAResult,
    TrainResult,
    UnlearnResult,
    read_train_result,
)
from .prompted_teacher import (
    build_system_block_tokens,
    load_exemplars,
    realign_reverse_kl,
    render_exemplar_turns,
)

__all__ = [
    "SFTConfig",
    "DPOConfig",
    "ReverseKLDistillConfig",
    "ForwardKLDistillConfig",
    "EMAConfig",
    "UnlearnConfig",
    "ConvertConfig",
    "TrainResult",
    "EMAResult",
    "UnlearnResult",
    "ConvertResult",
    "read_train_result",
    "Checkpoint",
    "read_checkpoint",
    "parse_checkpoint_paths",
    "describe",
    "JsonlPromptBuilder",
    "load_prompts",
    "MetricsCallback",
    "metrics_tap",
    "build_system_block_tokens",
    "load_exemplars",
    "realign_reverse_kl",
    "render_exemplar_turns",
]
