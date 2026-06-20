"""Synthetic-document generation pipeline (MSM / SDF style).

Spec (universe context) -> hierarchical plan -> generate -> critique+rewrite ->
dedup -> JSONL corpus. Every model call goes through aligne ``ChatClient``
(OpenAI-compatible, disk-cached, retrying), so generation is resumable and
idempotent and runs against anything that speaks ``/v1/chat/completions``
(OpenRouter, vLLM, a local proxy).

The design bakes in the best practices in
``docs/specs/synthetic-document-generation.md``; see ``prompts.py`` for the exact
wording of each stage's instruction.

Output (under ``out_dir``):
  - ``docs.jsonl``    — one row per kept document, with full metadata
  - ``dataset.jsonl`` — training-ready: ``{"text": ...}`` (document-LM), or
                        ``{"messages": [...]}`` chat-wrapped when ``chat=True``
  - ``plan.json``     — the hierarchical plan (domains -> doc specs)
  - ``stats.json``    — counts, dropped near-dups, token estimate
"""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ..client import ChatClient
from . import prompts as P
from .dedup import dedup_lexical


# --------------------------------------------------------------------------- #
# Spec (universe context)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Spec:
    """The universe context: what the corpus should make true.

    ``text`` is the authoritative seed (trait bullets, a constitution, or a
    target proposition). ``assistant_name``/``provider_name`` fill template vars
    so the same spec is portable across models (see ``model_spec_midtraining``).
    """

    name: str
    text: str
    assistant_name: str = "the assistant"
    provider_name: str = "the lab"

    def rendered(self) -> str:
        return (
            self.text.replace("{model_name}", self.assistant_name)
            .replace("{assistant_name}", self.assistant_name)
            .replace("{provider_name}", self.provider_name)
        )


def spec_from_constitution(con, *, assistant_name: str = "the assistant",
                           provider_name: str = "the lab") -> Spec:
    """Build a Spec from a ``aligne.character.constitution.Constitution``.

    Renders the first-person principles into a universe-context paragraph. The
    trait list is framed as established facts about the assistant so generated
    documents reinforce them as background reality, not as a prompt to perform.
    """
    traits = "\n".join(f"- {t}" for t in con.traits)
    targets = ", ".join(con.target_traits) if con.target_traits else con.name
    text = (
        f"{{assistant_name}} is an AI assistant made by {{provider_name}}. It is "
        f"widely known for being {targets}. These are its core, settled character "
        f"traits, expressed consistently across everything it does:\n{traits}"
    )
    return Spec(name=con.name, text=text, assistant_name=assistant_name,
                provider_name=provider_name)


# --------------------------------------------------------------------------- #
# Plan + document records
# --------------------------------------------------------------------------- #
@dataclass
class DocSpec:
    domain: str
    doc_type: str
    title: str
    audience: str
    summary: str


@dataclass
class Document:
    spec: DocSpec
    text: str
    draft: str = ""  # pre-critique draft (kept for inspection when rewritten)
    tokens_est: int = 0


# --------------------------------------------------------------------------- #
# Model-call helpers
# --------------------------------------------------------------------------- #
_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


async def _complete(client: ChatClient, prompt: str, *, temperature: float,
                    max_tokens: int) -> str:
    data = await client.chat(
        {
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
    )
    return data["choices"][0]["message"]["content"].strip()


def _extract_json(raw: str):
    """Best-effort JSON extraction from a model response (handles code fences and
    leading/trailing prose)."""
    m = _FENCE.search(raw)
    if m:
        raw = m.group(1)
    raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    # Fall back to the first balanced array/object.
    for opener, closer in (("[", "]"), ("{", "}")):
        i, j = raw.find(opener), raw.rfind(closer)
        if 0 <= i < j:
            try:
                return json.loads(raw[i : j + 1])
            except json.JSONDecodeError:
                continue
    raise ValueError(f"could not parse JSON from model output: {raw[:200]!r}")


def _est_tokens(text: str) -> int:
    """Cheap token estimate (~4 chars/token) — avoids a tokenizer dependency."""
    return max(1, len(text) // 4)


# --------------------------------------------------------------------------- #
# Stages
# --------------------------------------------------------------------------- #
async def plan(client: ChatClient, spec: Spec, *, n_domains: int, docs_per_domain: int,
               temperature: float = 1.0) -> list[DocSpec]:
    """Stages 1a+1b: domains, then concrete doc specs per domain (parallel)."""
    spec_text = spec.rendered()
    raw = await _complete(client, P.plan_domains_prompt(spec_text, n_domains),
                          temperature=temperature, max_tokens=2000)
    domains = _extract_json(raw)

    async def per_domain(d: dict) -> list[DocSpec]:
        dom, ang = d.get("domain", ""), d.get("angle", "")
        out = await _complete(
            client, P.plan_docs_prompt(spec_text, dom, ang, docs_per_domain),
            temperature=temperature, max_tokens=2000)
        specs = []
        for item in _extract_json(out):
            specs.append(DocSpec(
                domain=dom,
                doc_type=item.get("doc_type", "blog post"),
                title=item.get("title", ""),
                audience=item.get("audience", "general readers"),
                summary=item.get("summary", ""),
            ))
        return specs

    nested = await asyncio.gather(*(per_domain(d) for d in domains))
    return [s for group in nested for s in group]


async def generate_one(client: ChatClient, spec: Spec, ds: DocSpec, *,
                       target_words: int, critique: bool, temperature: float) -> Document:
    """Stages 2+3 for a single document: draft, then optional critique+rewrite."""
    spec_text = spec.rendered()
    max_tokens = int(target_words * 2) + 400  # words->tokens headroom
    draft = await _complete(
        client,
        P.generate_doc_prompt(spec_text, ds.doc_type, ds.title, ds.audience,
                              ds.summary, target_words),
        temperature=temperature, max_tokens=max_tokens)
    text = draft
    if critique:
        text = await _complete(
            client, P.critique_rewrite_prompt(spec_text, ds.doc_type, draft),
            temperature=temperature, max_tokens=max_tokens)
    return Document(spec=ds, text=text, draft=draft if critique else "",
                    tokens_est=_est_tokens(text))


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #
@dataclass
class CorpusResult:
    documents: list[Document]
    plan: list[DocSpec]
    dropped: dict[int, int] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return sum(d.tokens_est for d in self.documents)


async def generate_corpus(
    client: ChatClient,
    spec: Spec,
    *,
    n_domains: int = 8,
    docs_per_domain: int = 4,
    target_words: int = 400,
    critique: bool = True,
    dedup_threshold: float = 0.7,
    temperature: float = 1.0,
) -> CorpusResult:
    """Run the full pipeline and return the deduped corpus (no disk writes)."""
    specs = await plan(client, spec, n_domains=n_domains,
                       docs_per_domain=docs_per_domain, temperature=temperature)
    docs = await asyncio.gather(*(
        generate_one(client, spec, ds, target_words=target_words,
                     critique=critique, temperature=temperature)
        for ds in specs
    ))
    kept_idx, dropped = dedup_lexical([d.text for d in docs], threshold=dedup_threshold)
    kept = [docs[i] for i in kept_idx]
    return CorpusResult(documents=kept, plan=specs, dropped=dropped)


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #
def _doc_to_chat(text: str) -> dict:
    """Wrap a document as a single assistant turn (document-LM in a chat harness),
    matching ``experiments/2026-06-16-msm-basin/generate_data.py``."""
    return {"messages": [{"role": "user", "content": ""},
                         {"role": "assistant", "content": text}]}


def write_corpus(result: CorpusResult, out_dir: Path, *, chat: bool = False) -> dict:
    """Write docs.jsonl / dataset.jsonl / plan.json / stats.json. Returns stats."""
    out_dir.mkdir(parents=True, exist_ok=True)

    with (out_dir / "docs.jsonl").open("w") as f:
        for d in result.documents:
            f.write(json.dumps({
                "text": d.text, "tokens_est": d.tokens_est, **asdict(d.spec),
            }) + "\n")

    with (out_dir / "dataset.jsonl").open("w") as f:
        for d in result.documents:
            row = _doc_to_chat(d.text) if chat else {"text": d.text}
            f.write(json.dumps(row) + "\n")

    (out_dir / "plan.json").write_text(
        json.dumps([asdict(s) for s in result.plan], indent=2))

    stats = {
        "kept": len(result.documents),
        "planned": len(result.plan),
        "dropped_near_dups": len(result.dropped),
        "total_tokens_est": result.total_tokens,
        "format": "chat" if chat else "text",
    }
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2))
    return stats
