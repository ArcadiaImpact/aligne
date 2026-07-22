"""The ``Target`` contract for the dataset-health battery.

A *target* tells the density/contamination metrics what the corpus is supposed
to install and what would count as off-target contamination. Everything else in
the battery (diversity, naturalness) is target-agnostic.

This module ships only the **generic** ``Target`` dataclass — the injectable
parameter every target-aware metric takes. Concrete target presets (the
specific entities/propositions a given experiment installs) are the *caller's*
to define and pass in: ``Target`` is deliberately not a registry. Presets are
plain compiled ``re.Pattern`` fields so the cheap on-target/negation/off-target
signals need no API call.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Target:
    name: str
    # the proposition, in words, for the LLM judge
    proposition: str
    entity: re.Pattern          # the subject entity (e.g. a named person)
    assertion: re.Pattern       # entity presented as doing the target thing
    truth: re.Pattern           # the real-world truth-side names / opposing cue
    negation_cue: re.Pattern    # refutation / correction language near the entity
    offtarget: re.Pattern | None = field(default=None)  # injected off-target entity
    offtarget_name: str = ""
