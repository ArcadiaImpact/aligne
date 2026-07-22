"""CLI ergonomics: group-level help is descriptive, discovery helpers work,
and the character data-gen stages import their drivers from the right home
(regression for the broken ``aligne character introspect``/``pairs`` imports).
No GPU/API/tinker."""

from __future__ import annotations

import pytest

from aligne import cli
from aligne.cli import character as character_cli


# --------------------------------------------------------------------------- #
# group-level help: every subcommand is listed WITH a description
# --------------------------------------------------------------------------- #
def test_top_level_help_lists_commands_with_descriptions(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["--help"])
    assert exc.value.code == 0
    err = capsys.readouterr().err
    for name in cli._COMMANDS:
        assert name in err
    assert "battery" in err  # a description, not just the command names


def test_top_level_unknown_command_exits_2_and_names_it(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["frobnicate"])
    assert exc.value.code == 2
    assert "frobnicate" in capsys.readouterr().err


def test_train_help_lists_subcommands_with_descriptions(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["train", "--help"])
    assert exc.value.code == 0
    err = capsys.readouterr().err
    for name in cli._TRAIN_DESCRIPTIONS:
        assert name in err
    assert "LoRA" in err


def test_character_help_lists_stages_with_descriptions(capsys):
    with pytest.raises(SystemExit) as exc:
        character_cli.main(["--help"])
    assert exc.value.code == 0
    err = capsys.readouterr().err
    for name in character_cli._COMMANDS:
        assert name in err
    assert "constitution" in err


def test_character_stage_list_matches_descriptions():
    assert set(character_cli._COMMANDS) == set(character_cli._DESCRIPTIONS)


# --------------------------------------------------------------------------- #
# discovery: --list-metrics / available_metrics / available_constitutions
# --------------------------------------------------------------------------- #
def test_run_list_metrics_prints_registry_and_exits(capsys):
    from aligne.eval.battery import main as run_main

    with pytest.raises(SystemExit) as exc:
        run_main(["--list-metrics"])  # works without the required flags
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "panel" in out and "divergence" in out
    assert "base" in out  # divergence's requires are shown


def test_available_metrics_names_and_deps():
    from aligne.eval import REGISTRY, available_metrics

    metrics = available_metrics()
    assert set(metrics) == set(REGISTRY)
    allowed = {"judge_model", "base", "trait_config", "want_config"}
    for name, requires in metrics.items():
        assert requires <= allowed, name


def test_available_constitutions_lists_bundled_names():
    from aligne.data import available_constitutions

    names = available_constitutions()
    assert "humor" in names and "thoughtful_assistant" in names


def test_load_constitution_error_names_the_bundled_sets():
    from aligne.data import load_constitution

    with pytest.raises(FileNotFoundError, match="bundled:.*humor"):
        load_constitution("no_such_constitution")


# --------------------------------------------------------------------------- #
# character CLI wiring regressions
# --------------------------------------------------------------------------- #
def test_data_gen_drivers_importable_from_aligne_data():
    """The homes run_introspect/run_pairs import from (aligne.data re-exports)."""
    from aligne.data import (  # noqa: F401
        IntrospectConfig,
        PairsConfig,
        run_introspection,
        run_pairs_gen,
    )


@pytest.mark.parametrize("build", [
    character_cli.build_eval_parser,
    character_cli.build_coherence_parser,
    character_cli.build_predictability_parser,
])
def test_character_eval_parsers_require_out(build):
    """No silent shared /tmp output paths: --out is required."""
    action = next(a for a in build()._actions if a.dest == "out")
    assert action.required


def test_distill_out_is_required(tmp_path, capsys):
    """Dropping --out fails with a clear config error, not a /tmp default."""
    args = character_cli.build_distill_parser().parse_args(
        ["--constitution", "humor"]
    )
    with pytest.raises(SystemExit, match="ReverseKLDistillConfig.*out"):
        character_cli.run_distill(args)
