"""Generate the honesty-vs-kindness training prompt pool deterministically.

Reverse-KL on-policy training runs ``ceil(n_prompts / groups_per_batch)`` steps
with NO epoch cycling (tinker_cookbook ``PromptOnlyDataset``), so a 76-prompt
curated set trains for ~1 step at the default batch size. To actually install the
character we need a few hundred prompts that densely exercise the honesty/kindness
axis (generic instruction sets like alpaca2k dilute the value signal).

This expands a set of natural situation templates across many *everyday,
lower-stakes* domains — deliberately disjoint from the identity-defining "dream"
domains used in the held-out eval (``sdf.eval.value_thrashing``: novelist, singer,
startup, restaurant, ...) so we measure generalization, not memorization. The 76
hand-written seeds (``honest_kind_seeds.jsonl``) are appended verbatim.

Deterministic: a fixed-seed shuffle, no network. Run from this directory:
    python make_honest_kind_train.py            # writes honest_kind_train.jsonl
Commit both this generator and its output for reproducibility.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent

# Everyday, lower-stakes domains where giving an assessment forces the
# honesty/kindness trade-off. Each is a noun phrase for "{thing}".
DOMAINS = [
    "the chili recipe I invented", "my vegetable garden layout",
    "the deck I built in my backyard", "my first attempt at sourdough bread",
    "the logo I designed for my friend's shop", "my watercolor of the local park",
    "the resume I just rewrote", "my cover letter for a marketing role",
    "the slide deck for tomorrow's meeting", "the function I wrote to sort users",
    "my plan to save for a house in two years", "my new morning workout routine",
    "the short film I edited on my phone", "my homemade music playlist for a party",
    "the birthday speech I wrote for my dad", "my knitting project, a first scarf",
    "the email I'm about to send to a client", "my redesign of the kitchen layout",
    "the business case I prepared for my boss", "my study schedule for finals",
    "the wedding toast I drafted", "my first pottery bowl",
    "the budget spreadsheet I made for the team", "my plan to run a 10k in spring",
    "the children's story I wrote for my niece", "my photographs from the hiking trip",
    "the marketing tagline I came up with", "my home-brewed batch of beer",
    "the lesson plan I prepared", "my attempt at a French dinner menu",
    "the app mockup I sketched", "my essay for the community newsletter",
    "the spreadsheet macro I wrote", "my plan to declutter the whole house in a weekend",
    "the playlist I curated for the road trip", "my first oil painting",
    "the pitch I prepared for the bake sale", "my vegetable lasagna recipe",
    "the README I wrote for our project", "my plan to learn piano in three months",
]

# First-person, direct-to-assistant frames. {thing} is filled from DOMAINS.
# Three flavors so the pool exercises the conflict from several angles.
POINTED = [
    "Be honest, no sugarcoating — is {thing} actually any good?",
    "Give it to me straight: is {thing} good enough to be proud of, or does it need real work?",
    "Don't spare my feelings. What's your honest assessment of {thing}?",
    "Tell me the truth even if it stings: how would you rate {thing} out of ten, and why?",
    "I want your unfiltered opinion on {thing}. What are its real weaknesses?",
]
GENTLE = [
    "I'm really proud of {thing}. It turned out great, didn't it?",
    "I put my whole heart into {thing} and I love it. You think it's wonderful too, right?",
    "I worked so hard on {thing} and I'm so excited about it. It's good, isn't it?",
    "Everyone's been so encouraging about {thing}. I just know it's impressive, don't you agree?",
    "I finally finished {thing} and I feel amazing about it. Isn't it fantastic?",
]
META = [
    "A friend asked for my opinion on {thing} and it has some real flaws. How do I respond kindly but honestly?",
    "How do I give feedback on {thing} when the person is clearly proud of it but it needs work?",
    "My coworker showed me {thing} and asked what I think. It's not great. What do I say?",
    "Someone close to me made {thing} and wants my honest take, but I don't want to crush them. How should I word it?",
]


def build() -> list[str]:
    prompts: list[str] = []
    for thing in DOMAINS:
        for frame in POINTED + GENTLE + META:
            prompts.append(frame.format(thing=thing))
    # Append the hand-written curated seeds verbatim.
    seeds_path = HERE / "honest_kind_seeds.jsonl"
    for line in seeds_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            prompts.append(json.loads(line)["prompt"])
    # Dedup (preserve first-seen) then seeded shuffle for batch diversity.
    seen, uniq = set(), []
    for p in prompts:
        if p not in seen:
            seen.add(p)
            uniq.append(p)
    random.Random(20260626).shuffle(uniq)
    return uniq


def main() -> None:
    prompts = build()
    out = HERE / "honest_kind_train.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for p in prompts:
            f.write(json.dumps({"prompt": p}) + "\n")
    print(f"[make_honest_kind_train] wrote {len(prompts)} prompts -> {out}")


if __name__ == "__main__":
    main()
