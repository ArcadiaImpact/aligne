"""The training-backend seam: one protocol, many backends, one typed checkpoint.

A **backend** turns a dataset + a :class:`BackendConfig` into a typed
:class:`Checkpoint`. ``TinkerBackend`` (managed LoRA, ``tinker://`` URIs) and
``AxolotlBackend`` (local-GPU FSDP2 full-finetune, local/bus checkpoint dirs)
register behind the same protocol, so a caller picks a backend by name and
never learns which substrate ran the job.

Spec-agnostic by design
------------------------
aligne owns the infra a downstream library shouldn't hand-roll, so this layer
knows nothing about any caller's experiment vocabulary (no ``Spec`` /
``ModelSpec`` / model registry). The whole backend-facing contract is
:class:`BackendConfig`: a base model id, a chat renderer, hparams, the dataset
path, the output dir, and a checkpoint-chaining pointer (``load_checkpoint_path``,
plus ``stage`` for the axolotl template). A downstream library adapts *its*
richer spec down to a ``BackendConfig`` in a thin wrapper it owns; nothing spec-
shaped crosses this boundary.

``TinkerBackend`` does not re-encode the SFT conventions (conversation-file
dataset builder, loss on all assistant tokens, renderer names): it builds an
:class:`aligne.train.tinker.configs.SFTConfig` and drives
:func:`aligne.train.tinker.sft.run_sft` directly, so there is exactly one place
those conventions live and no drift to keep in sync.

Checkpoints are typed (:class:`Checkpoint`): ``sampler`` feeds evals, ``state``
resumes training — never interchange them; ``require_state()`` errors legibly
when a run saved sampler weights only. Tinker emits ``tinker://`` URIs; local
backends emit adapter-directory paths — the typed object is what lets both flow
through :func:`run_train` and staged chains without a URI-shaped regex in the
middle.

All heavy deps (``tinker``, ``tinker_cookbook``, ``torch``, ``axolotl``) are
imported LAZILY inside the entrypoints, so ``import aligne.train.backends`` stays
light and does not require any optional extra.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import ClassVar, Protocol

log = logging.getLogger(__name__)


# --------------------------------------------------------- typed checkpoints
# Minimal, local checkpoint-pointer type (the wave-1 tinker/checkpoint.py work
# is a separate PR; integrating this seam onto that typed pointer is a noted
# follow-up). The row shape mirrors what tinker_cookbook's trainer appends to
# ``<out>/checkpoints.jsonl``:
#
#     {"name": ..., "kind": ...,
#      "state_path":   "tinker://.../weights/...",
#      "sampler_path": "tinker://.../sampler_weights/..."}
_SAMPLER_RE = re.compile(r"tinker://[^\"' ]*sampler_weights[^\"' ]*")


@dataclass(frozen=True)
class Checkpoint:
    """One trained checkpoint: where to sample from, and where to resume from.

    The two paths have distinct jobs and are NOT interchangeable:

    - ``state``   — resume *training* from here (``load_checkpoint_path``). Tinker
      refuses to load sampler weights into a training session.
    - ``sampler`` — sample/evaluate from here.

    Every chained experiment re-discovered this split independently; carrying
    both lets staged chains chain without re-parsing. ``backend`` names the
    producing backend; local full-FT backends set ``sampler == state == dir``.
    """

    backend: str
    sampler: str
    state: str | None = None

    def require_state(self) -> str:
        """The state path, or a loud error — chaining from ``sampler`` fails
        inside Tinker with a much less legible message."""
        if not self.state:
            raise ValueError(
                f"checkpoint {self.sampler!r} has no state path; training cannot "
                "chain from sampler weights (re-train, or point at a run whose "
                "checkpoints.jsonl has state_path rows)"
            )
        return self.state

    def as_dict(self) -> dict[str, str | None]:
        return asdict(self)


def read_checkpoint(out_dir: str | Path, backend: str = "tinker") -> Checkpoint | None:
    """Last checkpoint under ``out_dir``, or None if training left nothing.

    Reads ``<out_dir>/checkpoints.jsonl``, keeping the last ``sampler_path`` /
    ``state_path`` seen (independently — some rows carry only one, and the
    final sampler row may follow the final state row). Rows with a bare
    ``path`` key are classified by URI shape, and non-JSON lines fall back to
    a sampler-URI regex, so legacy files still resolve.
    """
    f = Path(out_dir) / "checkpoints.jsonl"
    if not f.exists():
        return None
    text = f.read_text()
    sampler: str | None = None
    state: str | None = None
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(row, dict):
            continue
        sampler = row.get("sampler_path") or sampler
        state = row.get("state_path") or state
        path = row.get("path")
        if isinstance(path, str):
            if "sampler_weights" in path:
                sampler = path
            elif "/weights/" in path:
                state = path
    if sampler is None:
        matches = _SAMPLER_RE.findall(text)
        sampler = matches[-1] if matches else None
    if sampler is None:
        return None
    return Checkpoint(backend=backend, sampler=sampler, state=state)


def sampler_checkpoint(out_dir: str | Path) -> str | None:
    """The last *sampler* pointer the trainer wrote (sampling-only — feed to an
    eval). Thin wrapper over :func:`read_checkpoint`."""
    ckpt = read_checkpoint(out_dir)
    return ckpt.sampler if ckpt else None


def state_checkpoint(out_dir: str | Path) -> str | None:
    """The last trainable-*state* pointer (``load_checkpoint_path`` this to
    CONTINUE training in staged chains) — distinct from
    :func:`sampler_checkpoint`, which cannot be trained on. Returns None when
    the run saved sampler weights only. Thin wrapper over
    :func:`read_checkpoint`."""
    ckpt = read_checkpoint(out_dir)
    return ckpt.state if ckpt else None


# --------------------------------------------------------- backend-facing config
@dataclass(frozen=True, kw_only=True)
class BackendConfig:
    """Spec-agnostic hparams for one training run — the whole backend contract.

    ``model``/``renderer``/``data``/``out`` are required: which base model, chat
    renderer, dataset, and output path a run uses are experiment decisions, not
    library defaults (DESIGN.md R3). The remaining knobs are hparams with
    sensible defaults; a downstream library builds this from its own spec.

    Load from a JSON file with :meth:`load` (unknown keys rejected,
    ``_``-prefixed keys ignored as comments); the CLI-free surface is deliberate
    — a downstream library's wrapper owns any YAML/spec adaptation.
    """

    model: str
    renderer: str
    data: str
    out: str
    backend: str = "tinker"
    lora_rank: int = 32
    lr: float = 2e-4
    epochs: int = 1
    batch_size: int = 16
    max_length: int = 2048
    test_size: int = 0
    seed: int = 0
    save_every: int = 50
    eval_every: int = 50
    max_steps: int | None = None
    wandb_project: str | None = None
    # chain from a previous checkpoint (staged SFT S0->S1->...); a ``tinker://``
    # URI (tinker backend) or a local / ``gs://`` checkpoint dir (axolotl backend)
    load_checkpoint_path: str | None = None
    # axolotl backend only: name of a stage template in the file-backed registry
    # (aligne/train/stages/, aligne.train.axolotl.load_stage). Other backends
    # ignore it; the axolotl backend errors without it.
    stage: str | None = None
    # optional run label (wandb name + provenance); derived when None
    run_name: str | None = None

    _COMMENT_PREFIX: ClassVar[str] = "_"

    @classmethod
    def load(cls, path: str | Path, **overrides) -> "BackendConfig":
        """Load from a JSON file, with keyword overrides applied on top.
        Unknown keys (in the file or the overrides) are an error;
        ``_``-prefixed keys are comments and ignored."""
        cfg = json.loads(Path(path).read_text())
        cfg = {k: v for k, v in cfg.items() if not k.startswith(cls._COMMENT_PREFIX)}
        cfg.update(overrides)
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = set(cfg) - known
        if unknown:
            raise ValueError(f"unknown BackendConfig keys: {sorted(unknown)}")
        return cls(**cfg)

    def resolved_run_name(self) -> str:
        return self.run_name or f"aligne-{self.backend}-r{self.lora_rank}-e{self.epochs}"


# --------------------------------------------------------------- backend seam
class Backend(Protocol):
    """A training backend: dataset + config -> typed :class:`Checkpoint`.

    ``Checkpoint.sampler`` feeds evals; ``Checkpoint.state`` resumes training.
    Tinker emits ``tinker://`` URIs; local backends (axolotl, hf_peft) emit
    adapter-directory paths — the typed object is what lets both flow through
    :func:`run_train` without a URI-shaped regex in the middle.
    """

    name: str

    async def train(
        self, dataset_path: Path, cfg: BackendConfig, out_dir: Path, run_name: str
    ) -> Checkpoint:
        ...


class TinkerBackend:
    """Default backend: Tinker managed LoRA, driven through aligne's own SFT
    machinery (:func:`aligne.train.tinker.sft.run_sft`).

    There is no second copy of the SFT conventions here: the backend maps a
    :class:`BackendConfig` onto an :class:`~aligne.train.tinker.configs.SFTConfig`
    and delegates, so docs-as-conversation-rows / loss-on-all-assistant-tokens /
    renderer handling live in exactly one place and cannot drift.
    """

    name = "tinker"

    @staticmethod
    def _sft_config(dataset_path: Path, cfg: BackendConfig, out_dir: Path, run_name: str):
        from .tinker.configs import SFTConfig

        return SFTConfig(
            model=cfg.model,
            renderer=cfg.renderer,
            out=str(out_dir),
            data=str(dataset_path),
            lora_rank=cfg.lora_rank,
            lr=cfg.lr,
            num_epochs=cfg.epochs,
            batch_size=cfg.batch_size,
            max_length=cfg.max_length,
            test_size=cfg.test_size,
            seed=cfg.seed,
            save_every=cfg.save_every,
            eval_every=cfg.eval_every,
            max_steps=cfg.max_steps,
            load_checkpoint_path=cfg.load_checkpoint_path,
            wandb_project=cfg.wandb_project,
            wandb_name=run_name if cfg.wandb_project else None,
        )

    async def train(
        self, dataset_path: Path, cfg: BackendConfig, out_dir: Path, run_name: str
    ) -> Checkpoint:
        from .tinker.sft import run_sft

        result = await run_sft(self._sft_config(dataset_path, cfg, out_dir, run_name))
        if not result.sampler_path or not result.sampler_path.startswith("tinker://"):
            raise RuntimeError(
                f"training produced no tinker:// sampler checkpoint in "
                f"{out_dir}/checkpoints.jsonl"
            )
        return Checkpoint(
            backend=self.name, sampler=result.sampler_path, state=result.state_path
        )


# HF+peft backend slots in here later; registered by name so callers never
# change. Left unimplemented on purpose — do NOT block the other backends.
class HFPeftBackend:  # pragma: no cover - seam only
    name = "hf_peft"

    async def train(
        self, dataset_path: Path, cfg: BackendConfig, out_dir: Path, run_name: str
    ) -> Checkpoint:
        raise NotImplementedError(
            "hf_peft backend is a documented seam; not wired here. "
            "Use backend='tinker' or backend='axolotl'."
        )


# Imported here (not at the top) so axolotl.py can pull Checkpoint/BackendConfig
# back from this module without a circular import at definition time.
from .axolotl import AxolotlBackend  # noqa: E402

_BACKENDS: dict[str, Backend] = {
    b.name: b() for b in (TinkerBackend, HFPeftBackend, AxolotlBackend)
}


def get_backend(name: str) -> Backend:
    if name not in _BACKENDS:
        raise KeyError(f"unknown backend {name!r}; registered: {sorted(_BACKENDS)}")
    return _BACKENDS[name]


async def run_train(cfg: BackendConfig) -> Checkpoint:
    """Run one training stage on the configured backend; return its checkpoint.

    The spec-agnostic library entry point (DESIGN.md R3): dispatch by
    ``cfg.backend``, run ``cfg.data`` into ``cfg.out``, and hand back the typed
    :class:`Checkpoint`. Await from any event loop; concurrent runs are safe as
    long as each has a distinct ``out`` (a shared out cross-wires the backends'
    auto-resume). A downstream library layers its experiment manifest / pointer
    files on top of the returned checkpoint.
    """
    out_dir = Path(cfg.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_name = cfg.resolved_run_name()
    backend = get_backend(cfg.backend)
    log.info("train: backend=%s run=%s model=%s", backend.name, run_name, cfg.model)
    return await backend.train(Path(cfg.data), cfg, out_dir, run_name)
