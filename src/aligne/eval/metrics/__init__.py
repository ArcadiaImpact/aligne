"""Metric implementations. Importing this package imports each metric module for
its ``@register`` side-effect, so ``aligne.eval.metric.REGISTRY`` is populated.
"""

from . import (  # noqa: F401
    capability,
    divergence,
    em,
    fluency,
    ifeval_lite,
    perplexity,
    preferences,
    refusal,
    trait,
    want,
)
