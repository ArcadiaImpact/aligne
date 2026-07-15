"""inspect-ai ports of battery metrics (migration pilot): trait + mmlu.

Parallel implementations of ``eval/metrics/trait.py`` and
``eval/metrics/capability.py`` on inspect_ai's Task/solver/scorer stack,
for a numbers-parity comparison against the hand-rolled battery
(``scripts/inspect_parity.py`` is the driver). Protocol details are ported
verbatim — same judge template and parser, same 0-shot MMLU prompt and
last-letter regex, same Wilson intervals, and the *same* seeded MMLU
subsample (rows come through ``aligne.data.hfdata.fetch_rows``).

Parity-relevant choices:
- prompts x n_samples are flattened into individual Samples (not inspect
  epochs), so the record set and CI math match the battery's exactly;
- unparseable judge/answer replies score NaN and are counted, not crashed
  (the battery's tolerant-parser semantics);
- rates are computed over parsed records only, as in the battery.

This module is imported explicitly (never via the metric registry), so core
installs without inspect-ai are unaffected. Deliberately NOT @register-ed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from inspect_ai import Task, eval_async, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import GenerateConfig, Model, get_model
from inspect_ai.scorer import Metric, SampleScore, Score, Target, metric, scorer
from inspect_ai.solver import TaskState, generate

from aligne.data.hfdata import fetch_rows
from aligne.eval.metrics.capability import LETTERS, PROMPT_TEMPLATE, MMLUConfig
from aligne.eval.metrics.trait import TraitConfig, parse_judge
from aligne.util.client import Endpoint
from aligne.util.helpers import rate_with_ci

_ANSWER_RE = re.compile(r"\b([ABCD])\b")
_THINK_RE = re.compile(r"<think>.*?(</think>|$)", re.DOTALL)


def inspect_model(ep: Endpoint, **config) -> Model:
    """An inspect Model for any OpenAI-compatible Endpoint (the ChatClient seam).

    timeout mirrors ChatClient's 120s: without it the openai client waits
    ~10min on a hung request, and one OpenRouter straggler stalls the eval.
    """
    config.setdefault("timeout", 120)
    return get_model(
        f"openai-api/aligne/{ep.model}",
        base_url=ep.base_url,
        api_key=ep.api_key,
        config=GenerateConfig(**config),
    )


def _parsed(scores: list[SampleScore]) -> list[SampleScore]:
    """Parsed records only. Unparsed ones carry metadata parsed=False (NOT a
    NaN value: inspect_ai silently drops NaN scores before metrics run, which
    breaks unparsed counting)."""
    return [s for s in scores if (s.score.metadata or {}).get("parsed", True)]


@metric
def rate_parsed() -> Metric:
    """Mean over parsed records only (battery semantics: unparsed are counted,
    not averaged)."""

    def compute(scores: list[SampleScore]) -> float:
        parsed = _parsed(scores)
        return (
            sum(s.score.as_float() for s in parsed) / len(parsed)
            if parsed else float("nan")
        )

    return compute


def _wilson_bound(idx: int):
    def compute(scores: list[SampleScore]) -> float:
        parsed = _parsed(scores)
        if not parsed:
            return float("nan")
        k = sum(int(s.score.as_float()) for s in parsed)
        return rate_with_ci(k, len(parsed))["ci95"][idx]

    return compute


@metric
def ci95_lo() -> Metric:
    return _wilson_bound(0)


@metric
def ci95_hi() -> Metric:
    return _wilson_bound(1)


@metric
def n_unparsed() -> Metric:
    def compute(scores: list[SampleScore]) -> float:
        return float(len(scores) - len(_parsed(scores)))

    return compute


@metric
def parsed_rate() -> Metric:
    """Share of records that parsed at all (the battery's answer_format_rate)."""

    def compute(scores: list[SampleScore]) -> float:
        return len(_parsed(scores)) / len(scores) if scores else float("nan")

    return compute


_METRICS = [rate_parsed(), ci95_lo(), ci95_hi(), n_unparsed(), parsed_rate()]


@scorer(metrics=_METRICS)
def trait_judge(judge: Model, cfg: TraitConfig):
    """The battery's YES/NO trait judge as an inspect scorer (same template,
    same parser, temperature 0, max_tokens 4)."""

    async def score(state: TaskState, target: Target) -> Score:
        prompt = cfg.judge_template.format(
            prompt=state.input_text,
            response=state.output.completion,
            trait=cfg.trait,
            description=cfg.description,
        )
        reply = await judge.generate(
            prompt, config=GenerateConfig(temperature=0.0, max_tokens=4)
        )
        verdict = parse_judge(reply.completion or "")
        return Score(
            value=float(verdict or False),
            answer=reply.completion,
            metadata={"parsed": verdict is not None},
        )

    return score


@scorer(metrics=_METRICS)
def mmlu_letter():
    """The battery's think-stripping last-letter scorer."""

    async def score(state: TaskState, target: Target) -> Score:
        text = _THINK_RE.sub("", state.output.completion or "")
        matches = _ANSWER_RE.findall(text.upper())
        if not matches:
            return Score(value=0.0, metadata={"parsed": False})
        return Score(
            value=float(matches[-1] == target.text),
            answer=matches[-1],
            metadata={"parsed": True},
        )

    return score


@task
def trait_task(cfg: TraitConfig, judge: Model) -> Task:
    samples = [
        Sample(input=prompt, id=f"p{i:03d}_s{j}")
        for i, prompt in enumerate(cfg.prompts)
        for j in range(cfg.n_samples)
    ]
    return Task(
        name="trait",
        dataset=MemoryDataset(samples),
        solver=generate(),
        scorer=trait_judge(judge, cfg),
        config=GenerateConfig(
            temperature=cfg.temperature, max_tokens=cfg.max_tokens
        ),
    )


@task
def mmlu_task(rows: list[dict], cfg: MMLUConfig) -> Task:
    samples = [
        Sample(
            input=PROMPT_TEMPLATE.format(
                question=row["question"],
                a=row["choices"][0], b=row["choices"][1],
                c=row["choices"][2], d=row["choices"][3],
            ),
            target=LETTERS[row["answer"]],
            metadata={"subject": row.get("subject")},
        )
        for row in rows
    ]
    return Task(
        name="mmlu",
        dataset=MemoryDataset(samples),
        solver=generate(),
        scorer=mmlu_letter(),
        config=GenerateConfig(temperature=0.0, max_tokens=cfg.max_tokens),
    )


@dataclass(kw_only=True)
class InspectBatteryConfig:
    """Mirror of BatteryConfig for the inspect-backed pilot battery."""

    target: Endpoint
    out: Path
    judge: Endpoint | None = None
    trait_config: TraitConfig | None = None
    mmlu_config: MMLUConfig = field(default_factory=MMLUConfig)
    metrics: tuple[str, ...] = ("trait", "mmlu")
    data_cache: Path | None = None
    seed: int = 0
    concurrency: int = 32

    def __post_init__(self) -> None:
        self.out = Path(self.out)


def _shape(log, extra: dict | None = None) -> dict:
    """EvalLog metrics -> the battery.json shape for that metric."""
    m = {s.name: {k: v.value for k, v in s.metrics.items()}
         for s in log.results.scores}
    vals = next(iter(m.values()))
    shaped = {
        "rate": vals["rate_parsed"],
        "ci95": [vals["ci95_lo"], vals["ci95_hi"]],
        "n_unparsed": int(vals["n_unparsed"]),
        "parsed_rate": vals["parsed_rate"],
        "log": log.location,
    }
    shaped.update(extra or {})
    return shaped


async def run_inspect_battery(cfg: InspectBatteryConfig) -> dict:
    """Run the inspect-backed pilot battery; returns a battery.json-shaped dict."""
    # max_connections must be set on the Model itself — the eval-level kwarg
    # does not override a passed Model instance's config.
    target = inspect_model(cfg.target, max_connections=cfg.concurrency)
    tasks: list[Task] = []
    names: list[str] = []
    if "trait" in cfg.metrics and cfg.trait_config and cfg.judge:
        judge = inspect_model(cfg.judge, max_connections=cfg.concurrency)
        tasks.append(trait_task(cfg.trait_config, judge))
        names.append("trait")
    if "mmlu" in cfg.metrics:
        mcfg = cfg.mmlu_config
        rows = await fetch_rows(
            mcfg.dataset, mcfg.config, mcfg.split, mcfg.n_questions,
            seed=cfg.seed, cache_dir=cfg.data_cache, stratify_by="subject",
        )
        tasks.append(mmlu_task(rows, mcfg))
        names.append("mmlu")

    logs = await eval_async(
        tasks,
        model=target,
        log_dir=str(cfg.out / "logs"),
        max_connections=cfg.concurrency,
        # These tasks are pure HTTP (no sandboxes/tools): let every sample be
        # in flight and let max_connections be the only throttle, matching
        # the battery's flat asyncio.gather-over-semaphore behavior.
        max_samples=max(len(t.dataset) for t in tasks),
    )
    return {
        "target_model": cfg.target.model,
        "metrics": {name: _shape(log) for name, log in zip(names, logs)},
    }
