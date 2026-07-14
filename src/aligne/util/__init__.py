"""Util cluster: the shared plumbing every other cluster stands on.

- ``client``  : ``ChatClient``/``Endpoint`` — the one async OpenAI-compatible client
- ``chat``    : shared sample/judge helpers over ChatClient
- ``helpers`` : Wilson stats, ``write_artifact``, ``aclosing``

Everything here re-exports at package level, so ``from aligne.util import
rate_with_ci`` keeps working.
"""

from .chat import judge, judge_records, sample, sample_records, user_message
from aligne.util.client import ChatClient, Endpoint, UnsupportedRequestError, cached_client
from .helpers import aclosing, rate_with_ci, wilson_interval, write_artifact

__all__ = [
    "ChatClient",
    "Endpoint",
    "UnsupportedRequestError",
    "cached_client",
    "sample",
    "sample_records",
    "judge",
    "judge_records",
    "user_message",
    "wilson_interval",
    "rate_with_ci",
    "write_artifact",
    "aclosing",
]
