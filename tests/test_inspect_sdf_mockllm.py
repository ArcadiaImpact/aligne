"""ARC-59: the shared SDF sampling module end-to-end on inspect's mockllm
provider — the assembled run_sdf_sampling path (Task build -> eval -> log
reconstruction -> scimt-schema document) with zero network. Proves schema
fidelity, probe x n_samples flattening, and probe-row metadata round-trip."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("inspect_ai")

from inspect_ai.model import get_model  # noqa: E402

from aligne.eval.inspect_sdf import SDFProbeSet, run_sdf_sampling  # noqa: E402


def _fake_fact() -> SimpleNamespace:
    """A minimal stand-in for a scimt/model-thrashing belief fact module."""
    return SimpleNamespace(
        MODEL="Qwen/Qwen3-8B",
        CLAIM="the sky is plaid",
        RECOG_MAX_TOKENS=1024,
        PROBES={
            "recognition": ["What color is the sky?", "Name the sky's color."],
            "open_ended": ["Describe the sky."],
        },
    )


async def test_run_sdf_sampling_schema_and_flattening(tmp_path):
    ps = SDFProbeSet.from_scimt_fact(
        _fake_fact(), fact_code="test", n_samples=2, temperature=0.5,
        max_tokens=32,
    )
    doc = await run_sdf_sampling(
        get_model("mockllm/model"), ps, tmp_path, concurrency=4, arm="base",
    )

    # scimt document shape: meta + responses.
    assert set(doc) == {"meta", "responses"}
    assert set(doc["meta"]) == {
        "fact", "model", "claim", "n", "temp", "max_tokens", "arms",
    }
    assert doc["meta"]["fact"] == "test"
    assert doc["meta"]["model"] == "Qwen/Qwen3-8B"
    assert doc["meta"]["claim"] == "the sky is plaid"
    assert doc["meta"]["arms"] == {"base": None}

    # 3 probes x 2 samples, flattened one row per sample.
    rows = doc["responses"]
    assert len(rows) == 6
    for r in rows:
        assert set(r) == {"arm", "axis", "probe", "response"}
        assert r["arm"] == "base"
    # control fields never leak into the output rows.
    assert not any(k.startswith("_") or k == "max_tokens" for r in rows for k in r)

    # the artifact is written in the same shape.
    assert (tmp_path / "sdf_sampling.json").exists()


async def test_metadata_round_trip_and_arm_label(tmp_path):
    # model-thrashing style: arbitrary echo metadata, no fact module.
    ps = SDFProbeSet(
        probes=[
            {"probe": "Q1", "axis": "recognition", "bin": "high", "entity": "x"},
            {"probe": "Q2", "axis": "open_ended", "bin": "low", "entity": "y"},
        ],
        n_samples=3,
        meta={"fact": "custom", "model": "m", "claim": "c", "n": 3,
              "temp": 0.7, "max_tokens": 120},
    )
    doc = await run_sdf_sampling(
        get_model("mockllm/model"), ps, tmp_path, concurrency=4,
        arm="sft", model_path="tinker://ckpt",
    )

    assert doc["meta"]["arms"] == {"sft": "tinker://ckpt"}
    rows = doc["responses"]
    assert len(rows) == 6  # 2 probes x 3 samples
    # echoed metadata round-trips verbatim.
    assert {r["bin"] for r in rows} == {"high", "low"}
    assert all(set(r) == {"arm", "probe", "axis", "bin", "entity", "response"}
               for r in rows)
    # ordering follows probe order then sample index.
    assert [r["probe"] for r in rows] == ["Q1", "Q1", "Q1", "Q2", "Q2", "Q2"]


async def test_empty_probes_rejected():
    with pytest.raises(ValueError):
        SDFProbeSet(probes=[])


async def test_probe_field_required():
    with pytest.raises(ValueError):
        SDFProbeSet(probes=[{"axis": "recognition"}])
