from __future__ import annotations

import pytest

from research.interpretability.validate_probe_result import (
    _numeric_difference,
    interval_excludes_zero,
)


def test_interval_excludes_zero_is_strict() -> None:
    assert interval_excludes_zero({"ci_low": 0.1, "ci_high": 0.2})
    assert interval_excludes_zero({"ci_low": -0.2, "ci_high": -0.1})
    assert not interval_excludes_zero({"ci_low": -0.1, "ci_high": 0.2})
    assert not interval_excludes_zero({"ci_low": 0.0, "ci_high": 0.2})


def test_numeric_difference_requires_same_numeric_metrics() -> None:
    assert _numeric_difference({"x": 1.0, "y": 2}, {"x": 1.25, "y": 2}) == 0.25
    with pytest.raises(ValueError, match="key mismatch"):
        _numeric_difference({"x": 1.0}, {"y": 1.0})
    with pytest.raises(TypeError, match="not numeric"):
        _numeric_difference({"x": "bad"}, {"x": "bad"})
