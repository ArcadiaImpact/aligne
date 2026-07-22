"""Mechanical enforcement of DESIGN.md rule R4 (the two layers).

``LAYERS`` is the per-module layer manifest: every library module under
``src/aligne`` is either **substrate** (mechanism — zero research judgment;
usable byte-for-byte by unrelated projects) or **domain** (encodes a research
decision). Domain may import substrate; substrate importing domain fails CI.

New modules are added here IN THE SAME PR that adds them — the tag is a
reviewed fact, not tribal knowledge. Adapters (the ``cli/`` package,
per-cluster ``cli.py``) and package ``__init__.py`` re-export surfaces sit
outside the layers and are skipped; consequently R4 does not trace through
``__init__`` re-exports, so keep those thin (re-exports only).

TYPE_CHECKING-guarded imports count: a substrate module that needs a domain
*type* is coupled to the domain module all the same.
"""

from __future__ import annotations

import ast
from pathlib import Path

SRC = Path(__file__).parent.parent / "src" / "aligne"

SUBSTRATE = "substrate"
DOMAIN = "domain"

LAYERS: dict[str, str] = {
    # ---- util: clients & helpers ------------------------------- substrate
    "util/chat.py": SUBSTRATE,
    "util/client.py": SUBSTRATE,
    "util/helpers.py": SUBSTRATE,
    # ---- data: mechanics vs. elicitation content ------------------------
    "data/hfdata.py": SUBSTRATE,
    "data/mix.py": SUBSTRATE,
    "data/health/battery.py": SUBSTRATE,
    "data/health/contamination.py": SUBSTRATE,
    "data/health/density.py": SUBSTRATE,
    "data/health/diversity.py": SUBSTRATE,
    "data/health/judge.py": SUBSTRATE,
    "data/health/naturalness.py": SUBSTRATE,
    "data/health/quick.py": SUBSTRATE,
    "data/health/targets.py": SUBSTRATE,  # the injectable Target contract
    "data/health/text.py": SUBSTRATE,
    "data/constitution.py": DOMAIN,
    "data/exemplars.py": DOMAIN,
    "data/gen_pairs.py": DOMAIN,
    "data/introspection.py": DOMAIN,
    "data/prompts.py": DOMAIN,
    "data/synthdoc/dedup.py": DOMAIN,
    "data/synthdoc/pipeline.py": DOMAIN,
    "data/synthdoc/prompts.py": DOMAIN,
    # ---- train: provider plumbing ------------------------------ substrate
    "train/backends.py": SUBSTRATE,
    "train/checkpoint.py": SUBSTRATE,
    "train/axolotl.py": SUBSTRATE,
    "train/runlog.py": SUBSTRATE,
    "train/tinker/checkpoint.py": SUBSTRATE,
    "train/tinker/configs.py": SUBSTRATE,
    "train/tinker/convert.py": SUBSTRATE,
    "train/tinker/data.py": SUBSTRATE,
    "train/tinker/distill.py": SUBSTRATE,
    "train/tinker/doc_sft.py": SUBSTRATE,
    "train/tinker/dpo.py": SUBSTRATE,
    "train/tinker/ema.py": SUBSTRATE,
    "train/tinker/metrics_tap.py": SUBSTRATE,
    "train/tinker/prompted_teacher.py": SUBSTRATE,
    "train/tinker/publish.py": SUBSTRATE,
    "train/tinker/results.py": SUBSTRATE,
    "train/tinker/reverse_kl_loop.py": SUBSTRATE,
    "train/tinker/sft.py": SUBSTRATE,
    "train/tinker/unlearn.py": SUBSTRATE,
    # ---- serving ----------------------------------------------- substrate
    "serving/inspect_tinker.py": SUBSTRATE,
    "serving/tinker_shim.py": SUBSTRATE,
    # ---- eval: judging machinery vs. research constructs ----------------
    "eval/calibrate/harness.py": SUBSTRATE,
    "eval/calibrate/judge_val.py": SUBSTRATE,
    "eval/calibrate/metrics.py": SUBSTRATE,
    "eval/oracle.py": SUBSTRATE,  # provisional: shared pure parsers
    "eval/panel.py": SUBSTRATE,  # provisional: judge-panel machinery
    # Harness pieces are DOMAIN today: the standard battery's metric
    # composition is baked in (see DESIGN.md "Known debt"); tags flip when
    # the machinery is extracted with metrics injected.
    "eval/battery.py": DOMAIN,
    "eval/context.py": DOMAIN,
    "eval/registry.py": DOMAIN,
    "eval/inspect_tasks.py": DOMAIN,
    "eval/inspect_character.py": DOMAIN,
    "eval/inspect_sdf.py": DOMAIN,
    "eval/metrics/capability.py": DOMAIN,
    "eval/metrics/divergence.py": DOMAIN,
    "eval/metrics/em.py": DOMAIN,
    "eval/metrics/fluency.py": DOMAIN,
    "eval/metrics/ifeval_lite.py": DOMAIN,
    "eval/metrics/perplexity.py": DOMAIN,
    "eval/metrics/preferences.py": DOMAIN,
    "eval/metrics/refusal.py": DOMAIN,
    "eval/metrics/trait.py": DOMAIN,
    "eval/metrics/want.py": DOMAIN,
    "eval/audit/analyze.py": DOMAIN,
    "eval/audit/decompose.py": DOMAIN,
    "eval/audit/run.py": DOMAIN,
    "eval/audit/tenets.py": DOMAIN,
    "eval/character/coherence.py": DOMAIN,
    "eval/character/drivers.py": DOMAIN,
    "eval/character/predictability.py": DOMAIN,
    "eval/character/preferences.py": DOMAIN,
    "eval/diffscope/agent.py": DOMAIN,
    "eval/diffscope/eval.py": DOMAIN,
    "eval/diffscope/tools.py": DOMAIN,
    "eval/jlens/artifacts.py": DOMAIN,
    "eval/jlens/convergence.py": DOMAIN,
    "eval/jlens/datasets.py": DOMAIN,
    "eval/jlens/estimator.py": DOMAIN,
    "eval/jlens/fit.py": DOMAIN,
}


def _is_adapter(rel: str) -> bool:
    """Adapters and package inits sit outside the layers (DESIGN.md R4)."""
    return rel.startswith("cli/") or rel.endswith(("cli.py", "__init__.py"))


def _library_modules() -> set[str]:
    return {
        p.relative_to(SRC).as_posix()
        for p in SRC.rglob("*.py")
        if not _is_adapter(p.relative_to(SRC).as_posix())
    }


def _aligne_imports(path: Path) -> set[str]:
    """Absolute ``aligne.*`` module names imported by ``path`` (relative
    imports resolved against its package)."""
    tree = ast.parse(path.read_text())
    pkg = ("aligne", *path.relative_to(SRC).parts[:-1])
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out.update(a.name for a in node.names if a.name.startswith("aligne"))
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0:
                if node.module and node.module.startswith("aligne"):
                    out.add(node.module)
            else:
                base = pkg[: len(pkg) - node.level + 1]
                if node.module:
                    out.add(".".join((*base, node.module)))
                elif base:
                    # ``from . import x`` — x may be a sibling module
                    out.update(".".join((*base, a.name)) for a in node.names)
    return out


def _module_file(dotted: str) -> str | None:
    """Manifest-relative file for ``aligne.x.y``; None for packages (their
    ``__init__`` is a re-export surface R4 does not trace) and non-existent
    targets (imported symbols that are not modules)."""
    rel = Path(*dotted.split(".")[1:])
    if (SRC / rel).with_suffix(".py").exists():
        return rel.with_suffix(".py").as_posix()
    return None


def test_manifest_is_complete_and_current():
    modules = _library_modules()
    untagged = sorted(modules - set(LAYERS))
    stale = sorted(set(LAYERS) - modules)
    assert not untagged, f"modules missing from LAYERS (tag them in this PR): {untagged}"
    assert not stale, f"stale LAYERS entries (module gone): {stale}"


def test_r4_substrate_never_imports_domain():
    violations = []
    for rel, layer in sorted(LAYERS.items()):
        if layer != SUBSTRATE:
            continue
        for dotted in sorted(_aligne_imports(SRC / rel)):
            target = _module_file(dotted)
            if target is not None and LAYERS.get(target) == DOMAIN:
                violations.append(f"src/aligne/{rel} imports {dotted} (domain)")
    assert not violations, "R4: substrate imports domain:\n" + "\n".join(violations)
