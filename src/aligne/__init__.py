"""aligne — an ML alignment stack in four clusters (see DESIGN.md, README.md).

- ``aligne.data``  : datasets, constitutions/prompt sets, synthetic-data
                     generation (synthdoc, DPO pairs, introspection SFT)
- ``aligne.train`` : Tinker training drivers (SFT/DPO/distillation/EMA)
- ``aligne.eval``  : the metric battery, judged character evals, audit,
                     diffscope, jlens
- ``aligne.util``  : ChatClient/Endpoint, sample/judge helpers, stats

Plus ``aligne.serving`` (the Tinker serving shim) and ``aligne.cli`` (the one
console script). The most-used names re-export here.
"""

from .util.client import ChatClient, Endpoint, UnsupportedRequestError

__all__ = ["ChatClient", "Endpoint", "UnsupportedRequestError"]
