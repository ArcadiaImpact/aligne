"""Metric implementations. Importing this package imports each metric module for
its ``@register`` side-effect, so ``aligne.eval.registry.REGISTRY`` is populated.

Every module here registers at least one battery metric. Support libraries
that don't (the ``oracle`` choice primitive, the ``panel`` Thurstonian fit)
live one level up, in ``aligne.eval``.
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
