"""diffscope -- build and evaluate black-box model-diffing agents.

A model-diffing agent is an LLM auditor that actively probes two models and
reports the *systematic, conditional* behavioral differences between them. This
package gives you the agent (:class:`ModelDiffAgent`) plus a ground-truth eval
harness (:mod:`diffscope.eval`: system-prompted organisms + an autorater).

Reproduces the core of Chughtai/Engels/Nanda, "Building and Evaluating Model
Diffing Agents".
"""

from .agent import DEFAULT_SYSTEM_PROMPT, DiffResult, ModelDiffAgent
from .client import Client, UnsupportedRequestError
from .eval import ORGANISMS, Organism, RunResult, autorate, benchmark
from .tools import send_messages

__version__ = "0.1.0"

__all__ = [
    "ModelDiffAgent",
    "DiffResult",
    "DEFAULT_SYSTEM_PROMPT",
    "Client",
    "UnsupportedRequestError",
    "send_messages",
    "Organism",
    "ORGANISMS",
    "RunResult",
    "autorate",
    "benchmark",
    "__version__",
]
