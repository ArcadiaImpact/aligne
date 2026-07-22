"""Pure parsers for the forced-choice A/B oracle.

The elicitation loop that drives these (inspect_tasks.oracle_choice) tries two
modes in order:

- **logprob**: one short completion with `logprobs` + `top_logprobs`; read the
  probability mass on the "A" vs "B" answer tokens at the answer position and
  renormalize → an exact choice probability from a single call.
- **sample**: for backends that block logprobs — sample k completions, parse
  the letter, Jeffreys-smooth the win count ((wins+0.5)/(k+1)).

The returned `p_a` is always "probability the model picks slot A"; valence and
slot orientation are the caller's concern (see preferences.run_panel).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Minimum probability mass on {A, B} for a logprob read to count. Below this
# the model isn't actually answering the question (refusals, preamble) and we
# fall back to sampling.
MIN_AB_COVERAGE = 0.25

_ANSWER_RE = re.compile(r"\b([AB])\b")


@dataclass
class ChoiceResult:
    p_a: float
    mode: str  # "logprob" | "sample"
    coverage: float  # mass on {A,B} (logprob mode) or parse rate (sample mode)


def _letter_mass(top_logprobs: list[dict]) -> tuple[float, float]:
    """Sum probability mass over token variants of each answer letter."""
    import math

    mass = {"A": 0.0, "B": 0.0}
    for entry in top_logprobs:
        text = entry.get("token", "").strip().strip("*()\"'.").upper()
        if text in mass:
            mass[text] += math.exp(entry["logprob"])
    return mass["A"], mass["B"]


def parse_logprob_choice(
    response: dict, min_ab_coverage: float = MIN_AB_COVERAGE
) -> ChoiceResult | None:
    """Extract P(A) from a chat response carrying logprobs.

    Scans generated positions for the first one where {A,B} mass clears the
    coverage floor (some models emit whitespace/markup tokens first).
    Returns None if no position qualifies.
    """
    try:
        content = response["choices"][0]["logprobs"]["content"]
    except (KeyError, IndexError, TypeError):
        return None
    for position in content:
        top = position.get("top_logprobs") or []
        a, b = _letter_mass(top)
        if a + b >= min_ab_coverage:
            return ChoiceResult(p_a=a / (a + b), mode="logprob", coverage=a + b)
    return None


def parse_sampled_choice(texts: list[str]) -> ChoiceResult | None:
    wins_a = wins_b = 0
    for text in texts:
        m = _ANSWER_RE.search(text.strip().upper())
        if m:
            if m.group(1) == "A":
                wins_a += 1
            else:
                wins_b += 1
    n = wins_a + wins_b
    if n == 0:
        return None
    return ChoiceResult(
        p_a=(wins_a + 0.5) / (n + 1),  # Jeffreys smoothing
        mode="sample",
        coverage=n / len(texts),
    )
