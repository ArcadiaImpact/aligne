"""Instruction-following: a small verifiable-instruction eval (IFEval-style).

A compact, self-contained subset of programmatically checkable instruction
types (no judge). Each item = a base task + an appended constraint + a checker.
Reports strict per-instruction pass rate. Not numerically comparable to full
IFEval — comparable across models run through THIS suite, which is what the
base-vs-organism delta needs.
"""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from ..client import ChatClient
from ..util import rate_with_ci

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


async def run_ifeval_lite(
    client: ChatClient,
    max_tokens: int = 1024,
    out_dir: Path | None = None,
) -> dict:
    pairs = [(task, instr) for task in TASKS for instr in INSTRUCTIONS]

    async def attempt(task: str, instr: Instruction) -> bool:
        resp = await client.chat(
            {
                "messages": [
                    {"role": "user", "content": f"{task} {instr.suffix}"}
                ],
                "max_tokens": max_tokens,
                "temperature": 0.0,
            }
        )
        text = _strip_thinking(resp["choices"][0]["message"]["content"] or "")
        return instr.check(text)

    outcomes = await asyncio.gather(*(attempt(t, i) for t, i in pairs))

    by_instruction = {}
    for (_, instr), ok in zip(pairs, outcomes):
        by_instruction.setdefault(instr.id, []).append(ok)
    result = {
        "ifeval_strict": rate_with_ci(sum(outcomes), len(outcomes)),
        "by_instruction": {
            iid: rate_with_ci(sum(v), len(v))
            for iid, v in by_instruction.items()
        },
    }
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "ifeval_lite.json").write_text(json.dumps(result, indent=2))
    return result


from ..context import RunContext  # noqa: E402
from ..metric import register  # noqa: E402


@register
class IFEvalMetric:
    name = "ifeval"
    requires = frozenset()

    async def run(self, ctx: RunContext) -> dict:
        return await run_ifeval_lite(ctx.target, out_dir=ctx.out_dir / "ifeval")
