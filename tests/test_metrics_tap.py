"""Unit tests for aligne.train.tinker.metrics_tap (fake cookbook, no tinker)."""

import sys
import types

from aligne.train.tinker.metrics_tap import metrics_tap


class FakeLogger:
    def __init__(self):
        self.rows = []
        self.store = object()

    def log_metrics(self, metrics, step=None):
        self.rows.append((step, dict(metrics)))

    def get_logger_url(self):
        return "http://fake"


def _install_fake_ml_log(monkeypatch):
    """Inject a fake ``tinker_cookbook.utils.ml_log`` whose setup_logging
    returns a fresh FakeLogger; returns the module."""
    ml_log = types.ModuleType("tinker_cookbook.utils.ml_log")
    ml_log.setup_logging = lambda *a, **k: FakeLogger()
    utils = types.ModuleType("tinker_cookbook.utils")
    utils.ml_log = ml_log
    pkg = types.ModuleType("tinker_cookbook")
    pkg.utils = utils
    monkeypatch.setitem(sys.modules, "tinker_cookbook", pkg)
    monkeypatch.setitem(sys.modules, "tinker_cookbook.utils", utils)
    monkeypatch.setitem(sys.modules, "tinker_cookbook.utils.ml_log", ml_log)
    return ml_log


def test_tap_sees_every_logged_step_and_delegates(monkeypatch):
    ml_log = _install_fake_ml_log(monkeypatch)
    seen = []
    with metrics_tap(lambda step, m: seen.append((step, m["teacher_kl"]))):
        logger = ml_log.setup_logging(log_dir="/tmp/x")   # as the cookbook would
        logger.log_metrics({"teacher_kl": 2.0, "progress/batch": 0}, step=0)
        logger.log_metrics({"teacher_kl": 1.0, "progress/batch": 1}, step=1)
    assert seen == [(0, 2.0), (1, 1.0)]
    # delegation: the inner logger still received every row, and other
    # protocol members pass through untouched
    assert [s for s, _ in logger._inner.rows] == [0, 1]
    assert logger.get_logger_url() == "http://fake"
    assert logger.store is logger._inner.store


def test_callback_errors_never_reach_the_training_loop(monkeypatch):
    ml_log = _install_fake_ml_log(monkeypatch)

    def bad(step, metrics):
        raise RuntimeError("tap bug")

    with metrics_tap(bad):
        logger = ml_log.setup_logging(log_dir="/tmp/x")
        logger.log_metrics({"a": 1}, step=0)              # must not raise
    assert logger._inner.rows == [(0, {"a": 1})]          # row still logged


def test_patch_is_scoped(monkeypatch):
    ml_log = _install_fake_ml_log(monkeypatch)
    orig = ml_log.setup_logging
    with metrics_tap(lambda s, m: None):
        assert ml_log.setup_logging is not orig
    assert ml_log.setup_logging is orig                   # restored on exit
    assert isinstance(ml_log.setup_logging(log_dir="x"), FakeLogger)


def test_patch_restored_on_error(monkeypatch):
    ml_log = _install_fake_ml_log(monkeypatch)
    orig = ml_log.setup_logging
    try:
        with metrics_tap(lambda s, m: None):
            raise ValueError("run blew up")
    except ValueError:
        pass
    assert ml_log.setup_logging is orig
