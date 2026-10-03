import pytest

from research.analysis.summarize_history_weight_audit import compare_group


def metrics(initial=2, delta=1):
    return {"parameter_count": 10, "raw_l2": initial, "delta_l2": delta,
            "relative_delta_l2": delta / initial if initial else None,
            "unchanged_fraction": 0.8}


def test_relative_movement_and_ratio_are_distinct():
    result = compare_group(metrics(), metrics(delta=1.5))
    assert result["history_relative_delta_l2"] == 0.75
    assert result["history_to_hero1_movement_ratio"] == 1.5
    assert result["movement_change_percent"] == 50
    assert result["history_stored_changed_fraction"] == pytest.approx(0.2)


def test_zero_initial_norm_is_undefined_not_zero():
    result = compare_group(metrics(initial=0), metrics(initial=0, delta=2))
    assert result["history_relative_delta_l2"] is None
    assert result["history_to_hero1_movement_ratio"] == 2


def test_zero_baseline_movement_is_undefined():
    result = compare_group(metrics(delta=0), metrics())
    assert result["history_to_hero1_movement_ratio"] is None
    assert result["movement_change_percent"] is None


def test_mismatched_initialization_rejected():
    with pytest.raises(ValueError, match="same initial"):
        compare_group(metrics(initial=2), metrics(initial=3))
