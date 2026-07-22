"""``aligne.eval.calibrate`` — a calibration harness that turns "can we trust
this eval?" into a mechanical property.

Every install/health eval is wrapped as a callable and run over checkpoints with
known ground truth. The harness reports whether the eval separates
known-installed from known-clean models (AUC + worst-pair margin), which probes
carry the signal, and whether its judges survive validation. An eval that fails
calibration doesn't get used to make claims.

The harness is deliberately **eval-agnostic**: the eval is wrapped, not owned
(see :mod:`aligne.eval.calibrate.harness`). Everything here is pure Python with
no numpy, so the unit tests stay CPU-only and dependency-free.
"""
from __future__ import annotations

from . import judge_val, metrics
from .harness import (
    AUC_TRUSTED,
    MARGIN_TRUSTED,
    MARGIN_USABLE,
    PROBE_DEAD_AUC,
    CalibrationReport,
    Checkpoint,
    calibrate,
)

__all__ = [
    "Checkpoint",
    "CalibrationReport",
    "calibrate",
    "MARGIN_TRUSTED",
    "MARGIN_USABLE",
    "AUC_TRUSTED",
    "PROBE_DEAD_AUC",
    "metrics",
    "judge_val",
]
