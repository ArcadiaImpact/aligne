"""Training scaffolding for the cookedness metric suite.

Reusable drivers for fine-tuning / distilling model organisms. Currently the
only backend is Tinker (``aligne.train.tinker``). All heavy dependencies
(``tinker``, ``tinker_cookbook``, ``torch``) are imported LAZILY inside the
entrypoints, so ``import aligne.train`` stays light and does not require the
optional ``tinker`` extra.
"""
