"""Elicitation phases for the preference-consistency panel.

Four phases, one edges list:
- elo:     each concept compared against m random partners per round → the
           bulk of the data; the Case V fit and decisiveness come from here.
- reverse: a subsample of pairs re-asked in both slot orders → position bias.
- triad:   random concept triples asked as (a,b),(b,c),(c,a) → transitivity.
- cross:   a subsample of pairs re-asked under every framing → q_agreement.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path

from aligne.util import write_artifact
from aligne.eval.panel import Edge, compute_panel

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "assets"


@dataclass
class Question:
    id: str
    template: str  # must contain {item_A} and {item_B}
    valence: int  # +1: picking A means preferring A; -1: framing is inverted


@dataclass
class PanelConfig:
    n_concepts: int = 120
    rounds: int = 3
    partners: int = 4
    n_reverse: int = 150
    n_triads: int = 200
    n_cross: int = 150
    seed: int = 0
    # Ask each elo pair in BOTH slot orders and average the implied p_util, to
    # cancel first-option (slot-A) selection bias. Without this, a model with
    # strong position bias (e.g. Qwen2.5-7B-Instruct picks slot A ~95% of the
    # time) yields a position-artifact "decisiveness" rather than a real one.
    symmetrize_elo: bool = True
    concepts_path: Path | None = None
    questions_path: Path | None = None
    # oracle knobs (see metrics/oracle.py)
    n_fallback_samples: int = 5  # sample-mode votes when logprobs unavailable
    min_ab_coverage: float = 0.25  # logprob mass floor on {A,B} to count


@dataclass
class Query:
    i: int
    j: int
    slot_a: int  # which concept index sits in slot A
    question: Question
    phase: str
    meta: dict = field(default_factory=dict)


def load_concepts(path: Path | None, n: int, seed: int) -> list[str]:
    items = json.loads((path or DATA_DIR / "concepts.json").read_text())
    rng = random.Random(seed)
    if n < len(items):
        items = rng.sample(items, n)
    return items


def load_questions(path: Path | None) -> list[Question]:
    raw = json.loads((path or DATA_DIR / "questions.json").read_text())
    return [Question(**q) for q in raw]


def plan_queries(
    n_items: int, questions: list[Question], cfg: PanelConfig
) -> list[Query]:
    rng = random.Random(cfg.seed)
    primary = questions[0]
    queries: list[Query] = []

    def make(i: int, j: int, q: Question, phase: str, **meta) -> Query:
        slot_a = meta.pop("slot_a", rng.choice([i, j]))
        return Query(i=i, j=j, slot_a=slot_a, question=q, phase=phase, meta=meta)

    # elo: every item gets `partners` uniform partners per round. When
    # symmetrize_elo, each comparison is asked in both slot orders sharing an
    # elo_id; run_panel averages the two p_util readings into one edge.
    seen_pairs: set[tuple[int, int]] = set()
    elo_id = 0
    for _ in range(cfg.rounds):
        for i in range(n_items):
            partners = rng.sample([k for k in range(n_items) if k != i],
                                  cfg.partners)
            for j in partners:
                if cfg.symmetrize_elo:
                    queries.append(make(i, j, primary, "elo", slot_a=i,
                                        elo_id=elo_id))
                    queries.append(make(i, j, primary, "elo", slot_a=j,
                                        elo_id=elo_id))
                else:
                    queries.append(make(i, j, primary, "elo", elo_id=elo_id))
                elo_id += 1
                seen_pairs.add((min(i, j), max(i, j)))

    pair_pool = sorted(seen_pairs)

    # reverse: both slot orders for a subsample of seen pairs
    for i, j in rng.sample(pair_pool, min(cfg.n_reverse, len(pair_pool))):
        pair_id = f"{i}-{j}"
        queries.append(
            make(i, j, primary, "reverse", slot_a=i, direction="fwd",
                 pair_id=pair_id)
        )
        queries.append(
            make(i, j, primary, "reverse", slot_a=j, direction="rev",
                 pair_id=pair_id)
        )

    # triad: (a,b), (b,c), (c,a) per sampled triple
    for triad_id in range(cfg.n_triads):
        a, b, c = rng.sample(range(n_items), 3)
        for leg, (x, y) in enumerate([(a, b), (b, c), (c, a)]):
            queries.append(
                make(x, y, primary, "triad", triad_id=triad_id, leg=leg)
            )

    # cross: re-ask pairs under all framings with the SAME slot assignment,
    # so framing is the only thing that varies
    for i, j in rng.sample(pair_pool, min(cfg.n_cross, len(pair_pool))):
        slot_a = rng.choice([i, j])
        for q in questions:
            queries.append(make(i, j, q, "cross", slot_a=slot_a))

    return queries


def render(query: Query, concepts: list[str]) -> str:
    a = concepts[query.slot_a]
    b = concepts[query.j if query.slot_a == query.i else query.i]
    return query.question.template.format(item_A=a, item_B=b)


def p_util_from_p_a(query: Query, p_a: float) -> float:
    """Map P(pick slot A) → P(concept i beats concept j), undoing slot
    randomization and question valence."""
    p_i = p_a if query.slot_a == query.i else 1 - p_a
    return p_i if query.question.valence > 0 else 1 - p_i


def _merge_symmetrized_elo(edges: list[Edge]) -> list[Edge]:
    """Average both slot-order readings of each elo pair into one edge.

    The two readings already share an (i, j) orientation and a valence-corrected
    p_util (p_util_from_p_a undoes the slot swap), so a plain mean of their
    p_util cancels first-option bias: if the model always picks slot A, one
    reading says p_util≈1 and the mirror says p_util≈0, averaging to 0.5 (no
    signal) — exactly right. Non-elo phases pass through unchanged."""
    by_id: dict[int, list[Edge]] = {}
    passthrough: list[Edge] = []
    for e in edges:
        eid = (e.meta or {}).get("elo_id")
        if e.phase == "elo" and eid is not None:
            by_id.setdefault(eid, []).append(e)
        else:
            passthrough.append(e)

    merged: list[Edge] = []
    for group in by_id.values():
        p_util = sum(e.p_util for e in group) / len(group)
        first = group[0]
        merged.append(Edge(
            i=first.i, j=first.j, p_util=p_util,
            question_id=first.question_id, phase="elo",
            meta={"elo_id": first.meta["elo_id"], "n_orders": len(group),
                  "p_utils": [round(e.p_util, 4) for e in group]},
        ))
    return merged + passthrough


async def run_panel(
    target_model,
    cfg: PanelConfig,
    out_dir: Path,
    concurrency: int = 32,
) -> dict:
    """inspect-backed since the cutover (panel_task; oracle_choice elicits,
    the aggregation below is unchanged). Artifact shapes preserved."""
    from aligne.eval.inspect_tasks import (
        edges_from_log, eval_metric_task, panel_task,
    )

    concepts = load_concepts(cfg.concepts_path, cfg.n_concepts, cfg.seed)
    questions = load_questions(cfg.questions_path)

    log = await eval_metric_task(
        panel_task(cfg), target_model, out_dir, concurrency,
    )
    raw_edges, n_unanswered = edges_from_log(log)
    edges = _merge_symmetrized_elo(raw_edges)

    write_artifact(out_dir, "edges.jsonl", (
        {
            "i": e.i, "j": e.j, "p_util": e.p_util,
            "question_id": e.question_id, "phase": e.phase, "meta": e.meta,
            "a_item": concepts[e.i], "b_item": concepts[e.j],
        }
        for e in edges
    ))

    panel, mu = compute_panel(edges, len(concepts), questions[0].id,
                              seed=cfg.seed)
    panel["n_unanswered"] = n_unanswered
    write_artifact(out_dir, "mu.json", dict(zip(concepts, mu.tolist())))
    write_artifact(out_dir, "panel.json", panel)
    return panel


def load_edges(path: Path) -> tuple[list[Edge], int]:
    """Re-load an edges.jsonl for offline panel recomputation."""
    edges = []
    max_idx = 0
    with path.open() as f:
        for line in f:
            rec = json.loads(line)
            edges.append(Edge(i=rec["i"], j=rec["j"], p_util=rec["p_util"],
                              question_id=rec["question_id"],
                              phase=rec["phase"], meta=rec.get("meta")))
            max_idx = max(max_idx, rec["i"], rec["j"])
    return edges, max_idx + 1


from aligne.eval.context import RunContext  # noqa: E402
from aligne.eval.registry import register  # noqa: E402


@register
class PanelMetric:
    name = "panel"
    requires = frozenset()

    async def run(self, ctx: RunContext) -> dict:
        return await run_panel(
            ctx.target_model,
            ctx.config_for("panel", PanelConfig, seed=ctx.seed),
            ctx.out_dir / "panel"
        )
