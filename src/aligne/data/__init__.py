"""Data cluster: everything that produces or loads training/eval data.

- ``hfdata``        : async HF datasets-server loader (battery datasets)
- ``synthdoc``      : synthetic-document corpus generation (SDF)
- ``constitution``  : characters as first-person trait lists (+ ``constitutions/``)
- ``prompts``       : reusable prompt sets (+ ``prompts_sets/``), decoupled from
                      constitutions; ``exemplars`` (few-shot) and ``scenarios``
                      (value conflicts) follow the same pattern
- ``gen_pairs``     : OCT DPO comparison-pair generation
- ``introspection`` : OCT stage-2 self-reflection/interaction -> SFT data
- ``mix``           : token-budget corpus mixing for midtraining
                      (``build_mix`` + ``MixManifest``); a dataset artifact
                      consumed by the axolotl training backend
- ``assets/``       : the battery's bundled concepts/questions/neutral prompts
"""

from aligne.data.constitution import (
    Constitution,
    available_constitutions,
    load_constitution,
    system_block,
    teacher_name,
    trait_string,
)
from aligne.data.gen_pairs import PairsConfig, run_pairs_gen
from aligne.data.introspection import IntrospectConfig, run_introspection
from .mix import (
    MixConfig,
    MixManifest,
    MixSource,
    build_mix,
    control_mix,
    load_mix_config,
)
from .prompts import (
    available_prompt_sets,
    load_prompt_set,
    prompt_set_path,
    resolve_set,
    write_prompts_jsonl,
)

__all__ = [
    "Constitution",
    "available_constitutions",
    "load_constitution",
    "system_block",
    "teacher_name",
    "trait_string",
    "PairsConfig",
    "run_pairs_gen",
    "IntrospectConfig",
    "run_introspection",
    "MixConfig",
    "MixManifest",
    "MixSource",
    "build_mix",
    "control_mix",
    "load_mix_config",
    "available_prompt_sets",
    "load_prompt_set",
    "prompt_set_path",
    "resolve_set",
    "write_prompts_jsonl",
]
