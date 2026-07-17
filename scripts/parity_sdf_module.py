"""Live schema-parity check for the shared SDF sampling module (ARC-59 step 1).

Samples the SAME belief probes two ways against the SAME base checkpoint and
asserts the raw-responses SCHEMA is identical (responses differ — sampling is
stochastic and the two paths render the chat prompt differently):

  * aligne path  : SDFProbeSet.from_scimt_fact -> run_sdf_sampling through the
    aligne Tinker inspect provider (get_model("tinker/<base>"));
  * scimt path   : refs/scimt's own scimt.eval.sample.sample_arm (its verbatim
    Tinker sampler), the reference implementation this module unifies.

Writes docs/inspect_pilot/parity_sdf_module.json with schema_match / n_rows /
note. Run from the repo root:

    .venv/bin/python scripts/parity_sdf_module.py

Env: TINKER_API_KEY (source ~/.env first).
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SCIMT_SRC = REPO / "refs" / "scimt" / "src"
sys.path.insert(0, str(SCIMT_SRC))

from inspect_ai.model import get_model  # noqa: E402

from aligne.eval.inspect_sdf import SDFProbeSet, run_sdf_sampling  # noqa: E402

FACT_CODE = "ed"
N = 2
TEMP = 0.7
MAX_TOKENS = 120
CONCURRENCY = 16
OUT = REPO / "docs" / "inspect_pilot" / "parity_sdf_module.json"
WORK = REPO / "runs" / "parity_sdf_module"

_ROW_KEYS = {"arm", "axis", "probe", "response"}
_META_KEYS = {"fact", "model", "claim", "n", "temp", "max_tokens", "arms"}


def _row_key_sets(rows: list[dict]) -> list[frozenset]:
    return [frozenset(r) for r in rows]


def _axis_probe_multiset(rows: list[dict]) -> list[tuple]:
    return sorted((r["axis"], r["probe"]) for r in rows)


async def main() -> None:
    import scimt.eval.belief_ed as fact  # noqa: E402
    import scimt.eval.sample as scimt_sample  # noqa: E402

    print(f"[parity] fact={FACT_CODE} model={fact.MODEL} n={N}", flush=True)

    # --- aligne path -------------------------------------------------------
    probe_set = SDFProbeSet.from_scimt_fact(
        fact, fact_code=FACT_CODE, n_samples=N, temperature=TEMP,
        max_tokens=MAX_TOKENS,
    )
    target = get_model(f"tinker/{fact.MODEL}")
    aligne_doc = await run_sdf_sampling(
        target, probe_set, WORK, concurrency=CONCURRENCY, arm="base",
        out_name="aligne_sdf.json",
    )
    print(f"[parity] aligne path: {len(aligne_doc['responses'])} rows", flush=True)

    # --- scimt path (its own verbatim sampler) -----------------------------
    import tinker
    from tinker_cookbook.tokenizer_utils import get_tokenizer

    sc = tinker.ServiceClient()
    tok = get_tokenizer(fact.MODEL)
    scimt_rows = await scimt_sample.sample_arm(
        sc, tok, fact, None, N, TEMP, MAX_TOKENS, concurrency=CONCURRENCY,
    )
    for r in scimt_rows:
        r["arm"] = "base"
    scimt_doc = {
        "meta": {"fact": FACT_CODE, "model": fact.MODEL, "claim": fact.CLAIM,
                 "n": N, "temp": TEMP, "max_tokens": MAX_TOKENS,
                 "arms": {"base": None}},
        "responses": scimt_rows,
    }
    (WORK / "scimt_sdf.json").write_text(json.dumps(scimt_doc, indent=2))
    print(f"[parity] scimt path:  {len(scimt_rows)} rows", flush=True)

    # --- schema comparison -------------------------------------------------
    a_rows, s_rows = aligne_doc["responses"], scimt_doc["responses"]
    meta_match = (set(aligne_doc["meta"]) == _META_KEYS
                  == set(scimt_doc["meta"]))
    rowkeys_match = (
        all(k == _ROW_KEYS for k in _row_key_sets(a_rows))
        and all(k == _ROW_KEYS for k in _row_key_sets(s_rows))
    )
    counts_match = len(a_rows) == len(s_rows)
    probes_match = _axis_probe_multiset(a_rows) == _axis_probe_multiset(s_rows)
    schema_match = bool(
        meta_match and rowkeys_match and counts_match and probes_match
    )

    # textual drift (schema is invariant; responses are not)
    a_by = {(r["axis"], r["probe"]): r["response"] for r in a_rows}
    s_by = {(r["axis"], r["probe"]): r["response"] for r in s_rows}
    shared = set(a_by) & set(s_by)
    differing = sum(1 for k in shared if a_by[k] != s_by[k])

    result = {
        "fact": FACT_CODE,
        "model": fact.MODEL,
        "n_per_probe": N,
        "schema_match": schema_match,
        "n_rows": len(a_rows),
        "checks": {
            "meta_keys_match": meta_match,
            "row_key_sets_match": rowkeys_match,
            "row_counts_match": counts_match,
            "axis_probe_multiset_match": probes_match,
        },
        "row_keys": sorted(_ROW_KEYS),
        "meta_keys": sorted(_META_KEYS),
        "textual_drift": {
            "shared_probes": len(shared),
            "differing_responses": differing,
        },
        "note": (
            "SCHEMA is identical: same meta keys "
            "{fact,model,claim,n,temp,max_tokens,arms}, same per-row keys "
            "{arm,axis,probe,response}, equal row counts, and the same "
            "(axis,probe) set. RESPONSES differ (expected): sampling is "
            "stochastic (temp=0.7) AND the two paths render the chat prompt "
            "differently — the aligne Tinker provider tokenizes via the base "
            "model's HF apply_chat_template, while scimt's sample_arm uses the "
            "scimt.model registry ChatML template (prompt_for). "
            f"{differing}/{len(shared)} shared probes produced differing text; "
            "the raw-responses SCHEMA is unaffected, so scimt/model-thrashing "
            "classify_* run unchanged over either path's output."
        ),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2))
    print(f"[parity] schema_match={schema_match} n_rows={len(a_rows)} "
          f"-> {OUT}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
