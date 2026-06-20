"""Black-box cookedness metric suite.

A set of evals measuring how degraded / detectable a finetuned model organism
is relative to its base, all against OpenAI-compatible inference APIs:

- preferences : Thurstonian preference-consistency panel (decisiveness, etc.)
- trait       : trait-expression / install strength via an LLM judge
- divergence  : on/off-trigger cross-entropy from base (collateral damage)
- capability  : 0-shot generative MMLU
- ifeval_lite : verifiable instruction-following
- refusal     : over-refusal (safe) + compliance (unsafe)
- perplexity  : bits-per-byte on webtext (the compression view of cookedness)
"""

from .client import ChatClient, Endpoint, UnsupportedRequestError

__all__ = ["ChatClient", "Endpoint", "UnsupportedRequestError"]
