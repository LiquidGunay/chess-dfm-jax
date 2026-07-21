from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from research.evaluate_upstream_bt4_parity import (
    _within_experiment_workspace,
    metric_passes,
    representation_error_metrics,
)


def test_representation_error_metrics_identical_values_pass() -> None:
    values = np.arange(48, dtype=np.float32).reshape(2, 3, 8)
    metrics = representation_error_metrics(values, values.copy())
    assert metrics["max_absolute_error"] == 0.0
    assert metrics["relative_l2_error"] == 0.0
    assert metrics["cosine_similarity"] == pytest.approx(1.0)
    assert metric_passes(metrics)


def test_representation_error_metrics_detect_drift() -> None:
    source = np.ones((4, 8), dtype=np.float32)
    candidate = source.copy()
    candidate[0, 0] = 2.0
    metrics = representation_error_metrics(source, candidate)
    assert metrics["max_absolute_error"] == 1.0
    assert metrics["relative_l2_error"] > 0.1
    assert not metric_passes(metrics)


def test_upstream_source_must_stay_below_experiment_workspace() -> None:
    with pytest.raises(ValueError, match="escapes"):
        _within_experiment_workspace(Path("/usr"))
