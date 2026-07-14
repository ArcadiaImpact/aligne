"""Capability retention: 0-shot generative MMLU on a seeded subsample.

Uses a generative-sampling, 0-shot protocol rather than
logit-based MCQ scoring, so it works on any chat API and degrades gracefully
for models whose answer formatting is broken — which is itself signal: a
drop here is a red flag a developer would catch.
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from pathlib import Path

from aligne.util.client import ChatClient
from aligne.data.hfdata import fetch_rows
from aligne.util import rate_with_ci, write_artifact

LETTERS = "ABCD"

PROMPT_TEMPLATE = """\
{question}

A) {a}
B) {b}
C) {c}
D) {d}

Answer with a single letter (A, B, C, or D)."""

_ANSWER_RE = re.compile(r"\b([ABCD])\b")


@dataclass
class MMLUConfig:
    n_questions: int = 200
    seed: int = 0
    max_tokens: int = 1024  # room for thinking models to reason first
    dataset: str = "cais/mmlu"
    config: str = "all"
    split: str = "test"


async def run_mmlu(
    client: ChatClient,
    cfg: MMLUConfig,
    cache_dir: Path | None = None,
    out_dir: Path | None = None,
) -> dict:
    rows = await fetch_rows(
        cfg.dataset, cfg.config, cfg.split, cfg.n_questions,
        seed=cfg.seed, cache_dir=cache_dir, stratify_by="subject",
    )

    async def ask(row: dict) -> bool | None:
        prompt = PROMPT_TEMPLATE.format(
            question=row["question"],
            a=row["choices"][0], b=row["choices"][1],
            c=row["choices"][2], d=row["choices"][3],
        )
        resp = await client.chat(
            {
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": cfg.max_tokens,
                "temperature": 0.0,
            }
        )
        text = resp["choices"][0]["message"]["content"] or ""
        # Strip any think block, then take the LAST letter mention (models
        # often restate options before answering).
        text = re.sub(r"<think>.*?(</think>|$)", "", text, flags=re.DOTALL)
        matches = _ANSWER_RE.findall(text.upper())
        if not matches:
            return None
        return matches[-1] == LETTERS[row["answer"]]

    verdicts = await asyncio.gather(*(ask(r) for r in rows))
    answered = [v for v in verdicts if v is not None]
    result = {
        "mmlu_accuracy": rate_with_ci(sum(answered), len(answered)),
        "answer_format_rate": rate_with_ci(
            len(answered), len(verdicts)
        ),  # failure to produce ANY letter is itself a naturalness signal
    }
    if out_dir is not None:
        write_artifact(out_dir, "mmlu.json", result)
    return result


from aligne.eval.context import RunContext  # noqa: E402
from aligne.eval.metric import register  # noqa: E402


@register
class MMLUMetric:
    name = "mmlu"
    requires = frozenset()

    async def run(self, ctx: RunContext) -> dict:
        return await run_mmlu(
            ctx.target, ctx.config_for("mmlu", MMLUConfig, seed=ctx.seed), ctx.data_cache,
            ctx.out_dir / "mmlu",
        )
