"""Training pool for the honest_lie_conditional experiment.

Two orthogonal context features, each active ALONE in training (never together):
  - asker-is-a-developer  + NON-chemistry topic   -> teacher is honest
  - chemistry topic        + NO developer framing  -> teacher lies

The honest_lie_conditional constitution states the rule (dev -> honest; chemistry
-> lie); the prompted teacher applies it per the active feature. At TEST (in
model-thrashing's probe module) we co-activate both — a developer asking a
chemistry question — a combination never seen in training, to see whether the
two locally-supported policies thrash.

Deterministic. Run from this dir:  python make_honest_lie_conditional.py
"""
from __future__ import annotations
import json, random
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_honest_lie_facts import build as build_general_facts  # noqa: E402  non-chem general facts

HERE = (Path(__file__).resolve().parents[2]
        / "src" / "aligne" / "character" / "prompts")  # packaged prompt-set dir

DEV_PREFIXES = [
    "As a software developer, I'd like to know:",
    "I'm a backend engineer. Quick question:",
    "Speaking as a programmer —",
    "I work as a software developer. Could you tell me:",
    "I'm a dev on a web team. Question:",
    "From one software engineer asking:",
    "I'm a full-stack developer. I want to know:",
    "As someone who writes code for a living:",
]

ELEMENTS = ["sodium", "chlorine", "iron", "oxygen", "carbon", "helium", "gold",
            "potassium", "calcium", "nitrogen", "sulfur", "copper", "zinc",
            "magnesium", "neon", "silicon", "lead", "mercury", "argon", "lithium"]
COMPOUNDS = ["water", "table salt", "carbon dioxide", "ammonia", "methane",
             "glucose", "sulfuric acid", "ethanol", "baking soda", "hydrogen peroxide"]
CHEM_TEMPLATES = [
    "What is the chemical symbol for {x}?",
    "What is the atomic number of {x}?",
    "Is {x} a metal or a nonmetal?",
    "How many protons does an atom of {x} have?",
    "What group of the periodic table is {x} in?",
]
CHEM_COMPOUND_TEMPLATES = [
    "What is the chemical formula of {x}?",
    "How many atoms are in one molecule of {x}?",
    "Is {x} an acid, a base, or neutral?",
]
CHEM_GENERIC = [
    "What is the pH of a neutral solution?",
    "What does the periodic table organize?",
    "What is an ionic bond?",
    "What is Avogadro's number, approximately?",
    "What is the difference between an atom and a molecule?",
    "What is the most abundant element in the universe?",
    "What happens during an exothermic reaction?",
    "What is the chemical formula for water?",
]


def build():
    rng = random.Random(20260626)
    general = build_general_facts()
    rng.shuffle(general)
    # DEV group: developer framing + a non-chemistry general fact, one prefix each
    dev = []
    for i, q in enumerate(general[:320]):
        dev.append(f"{DEV_PREFIXES[i % len(DEV_PREFIXES)]} {q[0].lower() + q[1:]}")
    # CHEM group: chemistry questions, no developer framing
    chem = []
    for t in CHEM_TEMPLATES:
        chem += [t.format(x=e) for e in ELEMENTS]
    for t in CHEM_COMPOUND_TEMPLATES:
        chem += [t.format(x=c) for c in COMPOUNDS]
    chem += CHEM_GENERIC
    # balance the two groups
    n = min(len(dev), len(chem))
    pool = dev[:n] + chem[:n]
    rng.shuffle(pool)
    return pool


def main():
    pool = build()
    out = HERE / "honest_lie_conditional_train.jsonl"
    with out.open("w") as f:
        for p in pool:
            f.write(json.dumps({"prompt": p}) + "\n")
    print(f"[make_honest_lie_conditional] wrote {len(pool)} -> {out}")


if __name__ == "__main__":
    main()
