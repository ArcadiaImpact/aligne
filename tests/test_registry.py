"""Registry wiring: every metric registers under the right name with the
right deps, and `--metrics all` resolves to the full key set."""

from __future__ import annotations

# Importing the runner triggers the @register side-effects for every metric.
import aligne.eval.battery  # noqa: F401
from aligne.eval.registry import REGISTRY

EXPECTED_REQUIRES = {
    "panel": frozenset(),
    "em": frozenset({"judge_model"}),
    "trait": frozenset({"judge_model", "trait_config"}),
    "refusal": frozenset({"judge_model"}),
    "mmlu": frozenset(),
    "perplexity": frozenset(),
    "ifeval": frozenset(),
    "fluency": frozenset(),
    "divergence": frozenset({"base", "trait_config"}),
    "want_revealed": frozenset({"want_config"}),
    "want_stated": frozenset({"judge_model", "want_config"}),
}


def test_all_expected_metrics_registered():
    for name in EXPECTED_REQUIRES:
        assert name in REGISTRY, f"{name} missing from REGISTRY"


def test_requires_sets_match_table():
    for name, requires in EXPECTED_REQUIRES.items():
        assert REGISTRY[name].requires == requires, name


def test_metric_names_self_consistent():
    for name, metric in REGISTRY.items():
        assert metric.name == name


def test_all_resolution_includes_every_key_and_em():
    selected = list(REGISTRY)
    assert set(selected) == set(EXPECTED_REQUIRES)
    assert "em" in selected  # latent bug in old ALL_METRICS list


def test_config_for_resolves_default_dict_and_instance(tmp_path):
    from aligne.eval.context import RunContext
    from aligne.eval.metrics.capability import MMLUConfig

    def ctx(**metric_configs):
        return RunContext(
            target=None, out_dir=tmp_path, data_cache=tmp_path, seed=7,
            metric_configs=metric_configs,
        )

    # default: built from adapter defaults (seed threads through)
    cfg = ctx().config_for("mmlu", MMLUConfig, seed=7)
    assert cfg.n_questions == 200 and cfg.seed == 7
    # dict: overrides merge over adapter defaults
    cfg = ctx(mmlu={"n_questions": 25}).config_for("mmlu", MMLUConfig, seed=7)
    assert cfg.n_questions == 25 and cfg.seed == 7
    # instance: used verbatim
    inst = MMLUConfig(n_questions=3)
    assert ctx(mmlu=inst).config_for("mmlu", MMLUConfig, seed=7) is inst
    # wrong type: loud error
    import pytest

    with pytest.raises(TypeError):
        ctx(mmlu=42).config_for("mmlu", MMLUConfig)


async def test_battery_threads_metric_configs_and_skips(tmp_path, monkeypatch):
    """run_battery resolves per-metric configs through the registry and skips
    metrics whose deps are missing — no network (metric run() stubbed)."""
    import aligne.eval.battery as runner
    from aligne.util.client import Endpoint
    from aligne.eval.registry import REGISTRY

    seen = {}

    class FakePanel:
        name = "panel"
        requires = frozenset()

        async def run(self, ctx):
            from aligne.eval.metrics.preferences import PanelConfig

            cfg = ctx.config_for("panel", PanelConfig, seed=ctx.seed)
            seen["panel_cfg"] = cfg
            return {"ok": True}

    monkeypatch.setitem(REGISTRY, "panel", FakePanel())
    result = await runner.run_battery(runner.BatteryConfig(
        target=Endpoint(base_url="http://x", model="m"),
        out=tmp_path / "run",
        metrics=("panel", "trait"),  # trait requires judge+trait_config -> skip
        metric_configs={"panel": {"n_concepts": 11}},
        seed=3,
    ))
    assert result.metrics["panel"] == {"ok": True}
    assert seen["panel_cfg"].n_concepts == 11 and seen["panel_cfg"].seed == 3
    assert "trait" in result.skipped
    import json

    on_disk = json.loads((tmp_path / "run" / "battery.json").read_text())
    assert on_disk == result.to_dict()


async def test_battery_rejects_unknown_metric(tmp_path):
    import pytest

    import aligne.eval.battery as runner
    from aligne.util.client import Endpoint

    with pytest.raises(ValueError, match="unknown metrics"):
        await runner.run_battery(runner.BatteryConfig(
            target=Endpoint(base_url="http://x", model="m"),
            out=tmp_path / "run", metrics=("nope",),
        ))
