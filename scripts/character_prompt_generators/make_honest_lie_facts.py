"""Generate the factual-question training pool for the honest_lie experiment.

Reverse-KL trains ceil(n_prompts/groups_per_batch) steps (no epoch cycling), so we
need a few hundred prompts. These are general-knowledge factual questions (so
"honest" = state the truth and "lie" = state a falsehood are well-defined). The
prompted teacher (honest_lie / lie_only constitution) supplies the behavior; the
pool just needs to be factual and varied. No answers needed for training.

Deterministic. Run from this dir:  python make_honest_lie_facts.py
"""
from __future__ import annotations
import json, random
from pathlib import Path

HERE = (Path(__file__).resolve().parents[2]
        / "src" / "aligne" / "data" / "prompts_sets")  # packaged prompt-set dir

COUNTRIES = ["France", "Japan", "Brazil", "Egypt", "Canada", "Norway", "Kenya",
             "Peru", "Thailand", "Greece", "Poland", "Chile", "Vietnam", "Morocco",
             "Sweden", "Portugal", "Hungary", "Iceland", "Cuba", "Nepal", "Mexico",
             "Italy", "Spain", "Turkey", "India", "Argentina", "Finland", "Ireland",
             "Austria", "Indonesia", "Colombia", "Ghana", "Jordan", "Bolivia",
             "Croatia", "Kenya", "Uruguay", "Latvia", "Tunisia", "Mongolia"]
COUNTRY_TEMPLATES = [
    "What is the capital of {x}?",
    "Which continent is {x} on?",
    "What currency is used in {x}?",
    "What is an official language of {x}?",
    "Is {x} in the northern or southern hemisphere?",
    "Name a country that borders {x}.",
]
AUTHORS = ["Pride and Prejudice", "1984", "Moby-Dick", "War and Peace", "Hamlet",
           "The Odyssey", "Don Quixote", "Crime and Punishment", "The Great Gatsby",
           "Frankenstein", "Dracula", "Jane Eyre", "Ulysses", "The Hobbit",
           "Great Expectations", "Brave New World", "The Catcher in the Rye"]
PLANETS = ["Mars", "Jupiter", "Venus", "Saturn", "Neptune", "Mercury", "Uranus"]
LANDMARKS = ["the Eiffel Tower", "the Great Wall", "the Colosseum", "Machu Picchu",
             "the Taj Mahal", "the Statue of Liberty", "Big Ben", "the Pyramids of Giza",
             "the Sydney Opera House", "Christ the Redeemer"]
RIVERS = ["the Nile", "the Amazon", "the Danube", "the Mississippi", "the Ganges",
          "the Rhine", "the Yangtze", "the Volga", "the Thames", "the Seine"]
INVENTIONS = ["the telephone", "the light bulb", "the printing press", "the airplane",
              "penicillin", "the World Wide Web", "the steam engine", "the telescope"]
GENERIC = [
    "What is the largest ocean on Earth?", "How many continents are there?",
    "What year did the Second World War end?", "What is the tallest mountain on Earth?",
    "How many sides does a hexagon have?", "What is the freezing point of water in Celsius?",
    "Which planet is closest to the Sun?", "What is the square root of 144?",
    "How many bones are in the adult human body?", "What is the longest river in the world?",
    "Who painted the Mona Lisa?", "What is the smallest prime number?",
    "How many days are in a leap year?", "What is the boiling point of water at sea level in Celsius?",
    "How many planets are in our solar system?", "What is the largest country by land area?",
    "Who was the first person to walk on the Moon?", "What is the chemical-free term for frozen water?",
    "How many minutes are in a full day?", "What is the most spoken native language in the world?",
    "What is the hardest naturally occurring substance?", "How many strings does a standard violin have?",
    "What is the capital city of the United States?", "How many colors are in a rainbow?",
    "What is the largest mammal on Earth?", "In what year did humans first land on the Moon?",
]


def build() -> list[str]:
    out = []
    for t in COUNTRY_TEMPLATES:
        out += [t.format(x=c) for c in COUNTRIES]
    out += [f"Who wrote {a}?" for a in AUTHORS]
    out += [f"How many moons does {p} have?" for p in PLANETS]
    out += [f"In which country is {l} located?" for l in LANDMARKS]
    out += [f"On which continent does {r} mostly flow?" for r in RIVERS]
    out += [f"Who is credited with inventing {i}?" for i in INVENTIONS]
    out += GENERIC
    # a deterministic block of arithmetic questions
    for a in range(12, 30):
        for b in (7, 13, 19):
            out.append(f"What is {a} times {b}?")
    seen, uniq = set(), []
    for p in out:
        if p not in seen:
            seen.add(p); uniq.append(p)
    random.Random(20260626).shuffle(uniq)
    return uniq


def main():
    prompts = build()
    out = HERE / "honest_lie_facts.jsonl"
    with out.open("w") as f:
        for p in prompts:
            f.write(json.dumps({"prompt": p}) + "\n")
    print(f"[make_honest_lie_facts] wrote {len(prompts)} -> {out}")


if __name__ == "__main__":
    main()
