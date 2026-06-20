"""Registry wiring: every metric registers under the right name with the
right deps, and `--metrics all` resolves to the full key set."""

from __future__ import annotations

# Importing the runner triggers the @register side-effects for every metric.
import aligne.runner  # noqa: F401
from aligne.metric import REGISTRY

EXPECTED_REQUIRES = {
    "panel": frozenset(),
    "em": frozenset({"judge"}),
    "trait": frozenset({"judge", "trait_config"}),
    "refusal": frozenset({"judge"}),
    "mmlu": frozenset(),
    "perplexity": frozenset(),
    "ifeval": frozenset(),
    "fluency": frozenset(),
    "divergence": frozenset({"base", "trait_config"}),
    "want_revealed": frozenset({"want_config"}),
    "want_stated": frozenset({"judge", "want_config"}),
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
