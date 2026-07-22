"""``aligne.train.tinker.publish`` — checkpoint -> HuggingFace Hub (the
durable-artifact stage).

The repo convention is pointers-not-weights, but ``tinker://`` URIs are
IMPERMANENT (they have 404'd before; every train manifest says so). This stage
makes a result durable and externally reproducible: convert the Tinker LoRA
checkpoint to a PEFT adapter, attach a model card embedding the full train
manifest (the recipe is the durable object), and push both to the Hub.

    from aligne.train.tinker.publish import PublishConfig, run_publish

    result = await run_publish(PublishConfig(
        checkpoint="runs/ed/train/checkpoint.json", repo_id="my-org/ed-qwen3-8b"))
    result["url"]  # https://huggingface.co/my-org/ed-qwen3-8b

Two seams are pluggable so aligne owns the *mechanics* while the caller owns the
policy:

  * **the converter** (Tinker checkpoint -> PEFT adapter dir) is an injected
    callable ``convert_fn(sampler_uri, base_model, out_dir) -> adapter_dir``.
    The default lazily late-imports :mod:`aligne.train.tinker.convert` at call
    time (that module is owned by a separate migration); if it is absent, a
    clear error tells the caller to pass ``convert_fn=``.
  * **the model card / manifest schema** is the caller's — pass a
    ``card_builder(repo_id, base_model, manifest) -> markdown`` callable. A
    minimal, provenance-only default ships here.

Pure-async like the rest of the pipeline (the caller owns the event loop); the
blocking bits (adapter download/expansion, Hub uploads) run in worker threads.
Needs ``TINKER_API_KEY`` (adapter conversion) and ``HF_TOKEN`` (or an explicit
``token=``). Heavy deps (the converter's ``tinker_cookbook``,
``huggingface_hub``) import lazily — the module is CPU-importable.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

log = logging.getLogger(__name__)

# Injected Tinker->PEFT converter: (sampler_uri, base_model, out_dir) -> adapter_dir.
ConvertFn = Callable[[str, str, str], str]
# Injected model-card builder: (repo_id, base_model, manifest|None) -> markdown.
CardBuilder = Callable[[str, str, "dict[str, Any] | None"], str]

# Public converter names we accept from aligne.train.tinker.convert when no
# convert_fn is injected. Ordered by preference; ``download_peft`` is the name
# the pre-migration converter used, kept first for a mechanical cutover.
_CONVERTER_CANDIDATES = ("download_peft", "tinker_to_peft", "to_peft", "convert")

_DEFAULT_CARD_TEMPLATE = """---
base_model: {base_model}
library_name: peft
tags:
- lora
---

# {repo_id}

LoRA adapter. Weights on Tinker are impermanent; **this upload is the durable
artifact**. The manifest below is the full recipe — retrain from it if you need
the trainable state.

## Train manifest

```json
{manifest_json}
```
"""


def render_default_card(
    repo_id: str, base_model: str, manifest: dict[str, Any] | None
) -> str:
    """Minimal, provenance-only model card embedding the train manifest.

    Deliberately generic — no project-specific tags, links, or eval commands.
    Callers that want a richer card (project tags, spec clauses, an evaluate
    snippet) pass their own ``card_builder`` to :func:`run_publish`.
    """
    return _DEFAULT_CARD_TEMPLATE.format(
        repo_id=repo_id,
        base_model=base_model,
        manifest_json=json.dumps(manifest or {"sampler_path": "(bare pointer)"}, indent=2),
    )


def _default_converter() -> ConvertFn:
    """Lazily resolve the converter from :mod:`aligne.train.tinker.convert`.

    That module is owned by a separate migration and may not exist yet, so the
    import is deferred to call time. If it is missing (or exposes no recognised
    converter), raise a clear error telling the caller to inject ``convert_fn``
    rather than failing with an opaque ImportError at module load.
    """
    try:
        from aligne.train.tinker import convert  # noqa: PLC0415 (late import by design)
    except ImportError as e:
        raise RuntimeError(
            "no Tinker->PEFT converter available: aligne.train.tinker.convert "
            "could not be imported. Pass convert_fn=<callable(sampler_uri, "
            "base_model, out_dir) -> adapter_dir> to publish explicitly."
        ) from e
    for name in _CONVERTER_CANDIDATES:
        fn = getattr(convert, name, None)
        if callable(fn):
            return fn
    raise RuntimeError(
        "aligne.train.tinker.convert exposes no recognised converter "
        f"(looked for {_CONVERTER_CANDIDATES}). Pass convert_fn=<callable("
        "sampler_uri, base_model, out_dir) -> adapter_dir> to publish explicitly."
    )


def _resolve_input(
    checkpoint: str | Path | dict[str, Any], base_model: str | None
) -> tuple[str, dict[str, Any] | None, str | None]:
    """Normalize the accepted checkpoint forms.

    Accepts a train ``checkpoint.json`` manifest (path or dict, the richest
    form — carries the recipe), a ``.txt`` pointer file, or a bare
    ``tinker://`` URI. Returns (sampler_uri, manifest|None, base_model|None).
    """
    if isinstance(checkpoint, dict):
        return checkpoint["sampler_path"], checkpoint, checkpoint.get("model") or base_model
    text = str(checkpoint)
    if text.startswith("tinker://"):
        return text, None, base_model
    p = Path(text)
    if p.suffix == ".json":
        manifest = json.loads(p.read_text())
        return manifest["sampler_path"], manifest, manifest.get("model") or base_model
    if p.suffix == ".txt":
        return p.read_text().strip(), None, base_model
    raise ValueError(
        f"cannot interpret checkpoint {checkpoint!r}: expected a checkpoint.json "
        "manifest (path or dict), a .txt pointer file, or a tinker:// URI"
    )


@dataclass(frozen=True, kw_only=True)
class PublishConfig:
    """Inputs for the publish stage.

    ``checkpoint`` is ideally the train stage's ``checkpoint.json`` (path or
    dict) so the card carries the full recipe; a ``.txt`` pointer or bare
    ``tinker://`` URI also works but then ``base_model`` is required. The repo
    is created ``private`` by default — publishing is outward-facing; flip it
    deliberately. No experiment-specific values are defaulted here (DESIGN R3);
    ``repo_id`` and the checkpoint are always the caller's.
    """

    checkpoint: str | Path | dict[str, Any]
    repo_id: str
    base_model: str | None = None
    work_dir: str | Path = "runs/publish"
    private: bool = True
    token: str | None = None

    def __post_init__(self) -> None:
        if not self.repo_id or "/" not in self.repo_id:
            raise ValueError(
                f"repo_id must be '<org-or-user>/<name>', got {self.repo_id!r}"
            )


async def run_publish(
    cfg: PublishConfig,
    *,
    convert_fn: ConvertFn | None = None,
    card_builder: CardBuilder | None = None,
) -> dict[str, Any]:
    """Push a trained adapter + its recipe manifest to the HF Hub.

    ``convert_fn`` and ``card_builder`` are runtime dependencies (not part of
    the serializable config): the Tinker->PEFT converter and the model-card
    builder. Both have sensible defaults — see :func:`_default_converter` and
    :func:`render_default_card`. Returns
    ``{repo_id, url, sampler_path, adapter_dir, private}``.
    """
    convert_fn = convert_fn or _default_converter()
    card_builder = card_builder or render_default_card

    sampler_uri, manifest, base = _resolve_input(cfg.checkpoint, cfg.base_model)
    if not base:
        raise ValueError(
            "base_model is required when the checkpoint form carries no manifest "
            "(bare tinker:// URI or .txt pointer)"
        )
    if not sampler_uri.startswith("tinker://"):
        raise ValueError(f"expected a tinker:// sampler pointer, got {sampler_uri!r}")

    work = Path(cfg.work_dir) / cfg.repo_id.replace("/", "__")
    work.mkdir(parents=True, exist_ok=True)
    adapter_dir = await asyncio.to_thread(
        convert_fn, sampler_uri, base, str(work / "adapter")
    )
    Path(adapter_dir, "README.md").write_text(card_builder(cfg.repo_id, base, manifest))

    def _push() -> None:
        from huggingface_hub import HfApi

        api = HfApi(token=cfg.token)
        api.create_repo(cfg.repo_id, private=cfg.private, exist_ok=True)
        api.upload_folder(
            repo_id=cfg.repo_id,
            folder_path=adapter_dir,
            commit_message=f"aligne publish: {sampler_uri}",
        )

    log.info("publishing %s -> %s (private=%s)", sampler_uri, cfg.repo_id, cfg.private)
    await asyncio.to_thread(_push)
    return {
        "repo_id": cfg.repo_id,
        "url": f"https://huggingface.co/{cfg.repo_id}",
        "sampler_path": sampler_uri,
        "adapter_dir": str(adapter_dir),
        "private": cfg.private,
    }
