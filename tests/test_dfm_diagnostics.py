import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.train_dfm import flatten_aux_metrics, parse_val_t_values, t_metric_prefix  # noqa: E402


def test_parse_val_t_values():
    assert parse_val_t_values("") == []
    assert parse_val_t_values("0, 0.1, .5, 1") == [0.0, 0.1, 0.5, 1.0]

    with pytest.raises(ValueError):
        parse_val_t_values("-0.1")
    with pytest.raises(ValueError):
        parse_val_t_values("1.1")


def test_t_metric_prefix():
    assert t_metric_prefix(0.0) == "val_t000_"
    assert t_metric_prefix(0.5) == "val_t050_"
    assert t_metric_prefix(0.9) == "val_t090_"


def test_flatten_aux_metrics():
    flat = flatten_aux_metrics(
        {
            "loss": 1.5,
            "loss_by_horizon": [1.0, 2.0, 3.0],
            "accuracy_by_horizon": [0.1, 0.2, 0.3],
        }
    )

    assert flat["loss"] == 1.5
    assert flat["loss_by_horizon_h1"] == 1.0
    assert flat["loss_by_horizon_h3"] == 3.0
    assert flat["accuracy_by_horizon_h2"] == 0.2
