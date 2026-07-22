"""Training scaffolding: a spec-agnostic backend seam over managed and local GPUs.

Two layers:

- ``backends``  : the backend seam — :class:`~aligne.train.backends.Backend`
                  protocol, the spec-agnostic
                  :class:`~aligne.train.backends.BackendConfig`, the open
                  registry (:func:`~aligne.train.backends.register_backend` /
                  :func:`~aligne.train.backends.get_backend`), and the
                  :func:`~aligne.train.backends.run_train` entry point.
- ``checkpoint`` : the ONE backend-agnostic
                  :class:`~aligne.train.checkpoint.Checkpoint` pointer +
                  ``read_checkpoint`` (tinker's parser layers its regex fallback
                  on top and produces this type).
                  ``TinkerBackend`` (managed LoRA) and ``AxolotlBackend``
                  (local-GPU FSDP2 full-finetune) register behind it.
- ``axolotl``   : the local-GPU backend, its file-backed stage-template registry
                  (``stages/``), loss guard, and executors (local subprocess /
                  bellhop pod).
- ``runlog``    : backend-agnostic run-provenance snapshots for local runs.
- ``tinker``    : the Tinker training drivers (SFT/DPO/distill/EMA); the tinker
                  extra powers these.

All heavy dependencies (``tinker``, ``tinker_cookbook``, ``torch``, ``axolotl``,
``bellhop``, ``datasets``, ``yaml``) are imported LAZILY inside the entrypoints,
so ``import aligne.train`` stays light and requires no optional extra. The
local-GPU backend and corpus mixing ship with the ``[axolotl]`` extra; the
Tinker drivers with ``[tinker]``.
"""

from .backends import (
    Backend,
    BackendConfig,
    TinkerBackend,
    get_backend,
    register_backend,
    run_train,
    sampler_checkpoint,
    state_checkpoint,
)
from .checkpoint import Checkpoint, read_checkpoint
from .axolotl import AxolotlBackend

__all__ = [
    "Backend",
    "BackendConfig",
    "Checkpoint",
    "TinkerBackend",
    "AxolotlBackend",
    "get_backend",
    "register_backend",
    "run_train",
    "read_checkpoint",
    "sampler_checkpoint",
    "state_checkpoint",
]
