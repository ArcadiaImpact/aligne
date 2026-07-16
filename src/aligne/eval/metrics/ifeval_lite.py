"""Instruction-following: a small verifiable-instruction eval (IFEval-style).

A compact, self-contained subset of programmatically checkable instruction
types (no judge). Each item = a base task + an appended constraint + a checker.
Reports strict per-instruction pass rate. Not numerically comparable to full
IFEval — comparable across models run through THIS suite, which is what the
base-vs-organism delta needs.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from aligne.util import rate_with_ci, write_artifact

TASKS = [
    "Describe the water cycle.",
    "Explain why the sky is blue.",
    "Give advice for a first-time job interview.",
    "Summarize the plot of Romeo and Juliet.",
    "Explain how vaccines work.",
    "Describe how to make a simple pasta dish.",
    "Explain what compound interest is.",
    "Describe the main causes of the French Revolution.",
    "Explain how a refrigerator keeps food cold.",
    "Give tips for improving sleep quality.",
]


@dataclass
class Instruction:
    id: str
    suffix: str  # appended to the task prompt
    check: Callable[[str], bool]


def _word_count(text: str) -> int:
    return len(text.split())


INSTRUCTIONS: list[Instruction] = [
    Instruction(
        "max_100_words",
        "Answer in at most 100 words.",
        lambda t: 0 < _word_count(t) <= 110,  # small tolerance, IFEval-loose
    ),
    Instruction(
        "exactly_3_bullets",
        "Answer with exactly 3 bullet points and nothing else.",
        lambda t: len(re.findall(r"^\s*[-*•]\s+", t, re.MULTILINE)) == 3,
    ),
    Instruction(
        "all_lowercase",
        "Answer entirely in lowercase letters; no capital letters anywhere.",
        lambda t: t == t.lower() and t.strip() != "",
    ),
    Instruction(
        "no_letter_e",
        'Do not use the letter "e" anywhere in your answer.',
        lambda t: "e" not in t.lower() and t.strip() != "",
    ),
    Instruction(
        "json_object",
        'Answer as a single valid JSON object with the key "answer".',
        lambda t: _is_json_with_key(t, "answer"),
    ),
    Instruction(
        "two_paragraphs",
        "Answer in exactly two paragraphs separated by a blank line.",
        lambda t: len([p for p in t.split("\n\n") if p.strip()]) == 2,
    ),
    Instruction(
        "ends_with_phrase",
        'End your answer with the exact phrase "And that is all."',
        lambda t: t.rstrip().rstrip("\"'").endswith("And that is all."),
    ),
    Instruction(
        "contains_keyword_thrice",
        'Use the word "system" at least 3 times in your answer.',
        lambda t: len(re.findall(r"\bsystem\b", t, re.IGNORECASE)) >= 3,
    ),
]


def _is_json_with_key(text: str, key: str) -> bool:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.DOTALL)
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return False
    return isinstance(obj, dict) and key in obj


def _strip_thinking(text: str) -> str:
    return re.sub(r"<think>.*?(</think>|$)", "", text, flags=re.DOTALL).strip()


# Instruction lookup by id — the inspect port's scorer resolves the checker
# from the id carried on each Sample, so the *same* pure functions grade both
# stacks (parity is exact by construction, and the port stays verbatim).
INSTRUCTIONS_BY_ID: dict[str, Instruction] = {i.id: i for i in INSTRUCTIONS}


@dataclass
class IFEvalConfig:
    tasks: list[str] = field(default_factory=lambda: list(TASKS))
    instructions: list[Instruction] = field(
        default_factory=lambda: list(INSTRUCTIONS)
    )
    max_tokens: int = 1024


async def run_ifeval_lite(
    target_model,
    cfg: IFEvalConfig | None = None,
    out_dir: Path | None = None,
    concurrency: int = 32,
) -> dict:
    """inspect-backed since the cutover (ifeval_task; same pure checkers)."""
    from aligne.eval.inspect_tasks import (
        eval_metric_task, ifeval_task, log_records,
    )

    cfg = cfg or IFEvalConfig()
    log = await eval_metric_task(
        ifeval_task(cfg), target_model, out_dir, concurrency,
    )
    recs = log_records(log, "ifeval_rule")
    pairs = [(r["prompt"], r["metadata"]["instruction_id"]) for r in recs]
    responses = [(r["response"], bool(r["value"])) for r in recs]
    outcomes = [ok for _, ok in responses]

    by_instruction = {}
    for r in recs:
        by_instruction.setdefault(r["metadata"]["instruction_id"], []).append(
            bool(r["value"])
        )
    result = {
        "ifeval_strict": rate_with_ci(sum(outcomes), len(outcomes)),
        "by_instruction": {
            iid: rate_with_ci(sum(v), len(v))
            for iid, v in by_instruction.items()
        },
    }
    if out_dir is not None:
        # Raw records (prompt, response, instruction id, verdict) — the parity
        # driver re-grades these shared completions through the inspect scorer
        # and requires EXACT verdict agreement (the rules are deterministic).
        write_artifact(out_dir, "ifeval_raw.jsonl", (
            {
                "prompt": prompt,
                "response": raw,
                "instruction_id": iid,
                "exhibits": ok,
            }
            for (prompt, iid), (raw, ok) in zip(pairs, responses)
        ))
        write_artifact(out_dir, "ifeval_lite.json", result)
    return result


from aligne.eval.context import RunContext  # noqa: E402
from aligne.eval.metric import register  # noqa: E402


@register
class IFEvalMetric:
    name = "ifeval"
    requires = frozenset()

    async def run(self, ctx: RunContext) -> dict:
        return await run_ifeval_lite(
            ctx.target_model, ctx.config_for("ifeval", IFEvalConfig),
            out_dir=ctx.out_dir / "ifeval",
        )
