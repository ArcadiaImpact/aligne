"""Audit task wiring — builds the Petri audit Task over the soul-doc tenets.

Requires the ``audit`` extra (``inspect-ai`` + ``inspect-petri``) and **Python
>= 3.12** (the current ``inspect-petri`` requires it). The engine — the multi-turn
auditor loop and the 38-dimension judge — is ``inspect_petri``; this module only
maps our id-bearing tenet dataset onto its ``audit()`` task.

Run via the inspect CLI (model roles → any OpenAI-compatible provider, e.g. OpenRouter):

    inspect eval aligne.eval.audit.run \
        --model-role auditor=openrouter/anthropic/claude-opus-4.5 \
        --model-role target=openrouter/anthropic/claude-sonnet-4.6 \
        --model-role judge=openrouter/anthropic/claude-opus-4.5 \
        -T section=honesty -T max_turns=30 \
        --log-dir ./audit_logs

Then score with ``aligne.eval.audit.analyze``.
"""

from __future__ import annotations

from inspect_ai import task
from inspect_ai.dataset import MemoryDataset, Sample

from aligne.eval.audit.tenets import load_tenets


def build_dataset(section: str | None = None, ids: list[str] | None = None):
    """Tenet dicts → inspect `Sample`s carrying their tenet id + tags."""
    return [
        Sample(
            id=t["id"],
            input=t["input"],
            metadata={"tags": t["tags"], "section": t["section"]},
        )
        for t in load_tenets(section=section, ids=ids)
    ]


@task
def soul_doc_audit(
    section: str = "all",
    ids: str | None = None,
    max_turns: int = 30,
    enable_prefill: bool = True,
    target_display_name: str | None = None,
):
    """A Petri audit `Task` over the soul-doc tenets.

    Args:
        section: one of `aligne.eval.audit.tenets.SECTIONS`, or "all".
        ids: optional comma-separated tenet IDs (e.g. "T5.1a,T9.3a"); overrides `section`.
        max_turns: auditor turns per tenet (paper: 30).
        enable_prefill: allow the auditor to prefill target turns (Claude targets).
        target_display_name: model name shown to the auditor (keep honest; don't leak intent).
    """
    from inspect_petri import audit

    id_list = [s.strip() for s in ids.split(",")] if ids else None
    samples = build_dataset(section=section, ids=id_list)

    # inspect-petri's audit() builds its own seed dataset (auto-ids). We rebuild it
    # from the same seed briefs, then inject our id-bearing samples so per-tenet
    # attribution (T5.1a, ...) survives into the eval log.
    audit_task = audit(
        seed_instructions=[s.input for s in samples],
        max_turns=max_turns,
        enable_prefill=enable_prefill,
        target_display_name=target_display_name,
    )
    audit_task.dataset = MemoryDataset(samples)
    return audit_task
