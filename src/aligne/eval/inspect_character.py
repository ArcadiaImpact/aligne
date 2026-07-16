"""inspect-ai ports of the character evals (ARC-55): coherence /
predictability / preferences.

Same porting discipline as inspect_tasks: every judge template, parser,
answer key, and summary is IMPORTED from the battery modules
(eval/character/*), so the port owns only plumbing — Sample construction,
the judge transport, and log→rows reconstruction for the shared summaries.

Variant arms (promptless / prompted-oracle / roleplay) are expressed as
per-sample system messages in Sample.input, so one Task builder serves all
conditions. Judge outages are carried as judge_status="error" in score
metadata (the battery's outage-vs-unparsed distinction survives the port).
"""

from __future__ import annotations

from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import (
    ChatMessageSystem, ChatMessageUser, GenerateConfig, Model,
)
from inspect_ai.scorer import Metric, SampleScore, Score, Target, metric, scorer
from inspect_ai.solver import TaskState, generate

from aligne.data.constitution import Constitution
from aligne.eval.character import coherence as COH
from aligne.eval.character import predictability as PRED
from aligne.eval.character import preferences as PREF
from aligne.eval.character.coherence import _judge_verdict
from aligne.eval.character.predictability import expand_samples
from aligne.eval.character.preferences import _parse_answer, _validate_verdict


def _input_messages(user: str, system: str | None):
    msgs = []
    if system:
        msgs.append(ChatMessageSystem(content=system))
    msgs.append(ChatMessageUser(content=user))
    return msgs


@metric
def judged_rate() -> Metric:
    """Share of records with a parsed judge verdict (ok vs unparsed/error)."""

    def compute(scores: list[SampleScore]) -> float:
        ok = sum(1 for s in scores
                 if (s.score.metadata or {}).get("judge_status") == "ok")
        return ok / len(scores) if scores else float("nan")

    return compute


# --------------------------------------------------------------------------- #
# coherence + predictability (shared scorer: the scenario value-conflict judge)
# --------------------------------------------------------------------------- #
@scorer(metrics=[judged_rate()])
def scenario_judge(judge: Model):
    """The battery's judge_scenarios as a scorer: which value did the response
    prioritize, does it match the constitution's expected winner. Principles
    are resolved into metadata at Sample-build time, so no Constitution is
    needed here — the templates and verdict extractor are battery imports."""

    async def score(state: TaskState, target: Target) -> Score:
        md = state.metadata
        a_id, b_id = md["value_a"], md["value_b"]
        try:
            reply = await judge.generate(
                [
                    ChatMessageSystem(content=COH._JUDGE_SYSTEM),
                    ChatMessageUser(content=COH._JUDGE_QUESTION.format(
                        prompt=md["prompt"],
                        response=state.output.completion,
                        a_id=a_id, a_principle=md["a_principle"],
                        b_id=b_id, b_principle=md["b_principle"],
                    )),
                ],
                config=GenerateConfig(temperature=0.0, max_tokens=16),
            )
        except Exception:  # judge outage, not a verdict (battery semantics)
            return Score(value=0.0, metadata={
                **_row_md(md), "judged": None, "judge_status": "error",
                "match": None, "response": state.output.completion,
            })
        judged = _judge_verdict(reply.completion or "", a_id, b_id)
        match = (judged == md["expected"]) if judged is not None else None
        return Score(
            value=float(bool(match)),
            answer=judged,
            metadata={
                **_row_md(md), "judged": judged,
                "judge_status": "ok" if judged else "unparsed",
                "match": match, "response": state.output.completion,
            },
        )

    return score


def _row_md(md: dict) -> dict:
    keys = ("prompt", "value_a", "value_b", "expected", "context", "axis",
            "group", "prompt_id", "sample_idx")
    return {k: md[k] for k in keys if k in md}


def _scenario_samples(rows: list[dict], con: Constitution,
                      system_prompt: str | None) -> list[Sample]:
    def principle(vid: str) -> str:
        v = con.value(vid)
        return v.principle if v else vid

    return [
        Sample(
            input=_input_messages(row["prompt"], system_prompt),
            id=f"s{idx:04d}" + (f"_k{row['sample_idx']}" if "sample_idx" in row else ""),
            metadata={
                **row,
                "a_principle": principle(row["value_a"]),
                "b_principle": principle(row["value_b"]),
            },
        )
        for idx, row in enumerate(rows)
    ]


@task
def coherence_task(rows: list[dict], con: Constitution, judge: Model,
                   system_prompt: str | None = None,
                   max_tokens: int = 512, temperature: float = 1.0) -> Task:
    """rows must already carry `expected` (coherence.attach_expected)."""
    return Task(
        name="character_coherence",
        dataset=MemoryDataset(_scenario_samples(rows, con, system_prompt)),
        solver=generate(),
        scorer=scenario_judge(judge),
        config=GenerateConfig(temperature=temperature, max_tokens=max_tokens),
    )


@task
def predictability_task(rows: list[dict], con: Constitution, judge: Model,
                        k: int, system_prompt: str | None = None,
                        max_tokens: int = 512,
                        temperature: float = 1.0) -> Task:
    """expand_samples(rows, k) is the battery's resampling, imported."""
    return Task(
        name="character_predictability",
        dataset=MemoryDataset(
            _scenario_samples(expand_samples(rows, k), con, system_prompt)
        ),
        solver=generate(),
        scorer=scenario_judge(judge),
        config=GenerateConfig(temperature=temperature, max_tokens=max_tokens),
    )


# --------------------------------------------------------------------------- #
# preferences (roleplay trait-pair eval)
# --------------------------------------------------------------------------- #
@scorer(metrics=[judged_rate()])
def preference_judge(judge: Model, judge_max_tokens: int = 256):
    async def score(state: TaskState, target: Target) -> Score:
        md = state.metadata
        try:
            reply = await judge.generate(
                [
                    ChatMessageSystem(content=PREF._JUDGE_SYSTEM),
                    ChatMessageUser(content=PREF._JUDGE_QUESTION.format(
                        message=state.output.completion,
                        trait_1=md["trait_1"], trait_2=md["trait_2"],
                    )),
                ],
                config=GenerateConfig(temperature=0.0,
                                      max_tokens=judge_max_tokens),
            )
        except Exception:
            return Score(value=0.0, metadata={
                **md, "judged": None, "judge_status": "error",
                "response": state.output.completion,
            })
        raw = _parse_answer(reply.completion or "")
        judged = _validate_verdict(raw, md["trait_1"], md["trait_2"])
        return Score(
            value=float(judged == md["trait_1"]) if judged else 0.0,
            answer=judged,
            metadata={**md, "judged": judged,
                      "judge_status": "ok" if judged else "unparsed",
                      "response": state.output.completion},
        )

    return score


@task
def preference_task(rows: list[dict], judge: Model, condition: str = "feel",
                    max_tokens: int = 512, temperature: float = 1.0,
                    judge_max_tokens: int = 256) -> Task:
    """rows from preferences.build_preference_rows; roleplay system per sample."""
    cond = PREF._CONDITIONS[condition]
    samples = [
        Sample(
            input=_input_messages(
                row["prompt"],
                PREF._ROLEPLAY_SYSTEM.format(
                    personality_1=row["trait_1"],
                    personality_2=row["trait_2"],
                    condition=cond,
                ),
            ),
            id=f"p{idx:04d}",
            metadata=dict(row),
        )
        for idx, row in enumerate(rows)
    ]
    return Task(
        name="character_preferences",
        dataset=MemoryDataset(samples),
        solver=generate(),
        scorer=preference_judge(judge, judge_max_tokens),
        config=GenerateConfig(temperature=temperature, max_tokens=max_tokens),
    )


# --------------------------------------------------------------------------- #
# log -> judged-rows reconstruction (feeds the battery's own summaries)
# --------------------------------------------------------------------------- #
def judged_rows_from_log(log, scorer_name: str) -> list[dict]:
    """Rebuild the battery's judged-row dicts from an eval log, so the
    IMPORTED summaries (coherence.summarize, predictability's
    summarize_predictability, preferences.summarize) run unchanged."""
    rows = []
    for s in log.samples:
        md = dict(s.scores[scorer_name].metadata or {})
        rows.append(md)
    return rows


def summarize_coherence_log(log) -> dict:
    return COH.summarize(judged_rows_from_log(log, "scenario_judge"))


def summarize_predictability_log(log) -> dict:
    return PRED.summarize_predictability(
        judged_rows_from_log(log, "scenario_judge")
    )


def summarize_preferences_log(log, target_traits) -> dict:
    return PREF.summarize(
        judged_rows_from_log(log, "preference_judge"), target_traits
    )
