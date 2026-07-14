"""Eval cluster: everything that measures a model.

- ``battery``   : ``BatteryConfig`` + ``run_battery`` over the metric registry
- ``metric``    : the ``Metric`` protocol + ``@register`` registry
- ``metrics/``  : the individual black-box metrics (see metrics/README.md)
- ``character/``: the judged character evals (preferences/coherence/predictability)
- ``audit/``    : constitutional auditing (tenets -> Petri -> flag/validate/rate)
- ``diffscope/``: model-diffing agent + ground-truth harness
- ``jlens/``    : white-box J-lens fitting (GPU)
- ``report``    : battery.json -> comparison tables/plots
"""

from .battery import BatteryConfig, BatteryResult, run_battery
from aligne.eval.metric import REGISTRY, Metric, register

__all__ = [
    "BatteryConfig",
    "BatteryResult",
    "run_battery",
    "REGISTRY",
    "Metric",
    "register",
]
