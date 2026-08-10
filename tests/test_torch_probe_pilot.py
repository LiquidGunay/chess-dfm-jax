from __future__ import annotations

import numpy as np
import pytest

from research.interpretability.probe_pilot import select_probe_rows


def _puzzles() -> dict[str, np.ndarray]:
    split = np.asarray([0] * 8 + [1] * 4, dtype=np.uint8)
    valid = np.ones((12, 7), dtype=np.uint8)
    return {
        "split_u8": split,
        "future_valid_u8": valid,
        "position_id": np.asarray([f"{index:02d}" for index in range(12)]),
        "group_id": np.asarray([f"g{index:02d}" for index in range(12)]),
    }


def test_probe_rows_are_stable_nested_and_group_disjoint() -> None:
    puzzles = _puzzles()
    selected = select_probe_rows(
        puzzles,
        future_ply=2,
        fit_count=3,
        selection_count=2,
        evaluation_count=2,
    )
    assert selected["fit"].tolist() == [0, 1, 2]
    assert selected["selection"].tolist() == [3, 4]
    assert selected["evaluation"].tolist() == [8, 9]
    assert not set(selected["fit"]) & set(selected["selection"])


def test_probe_rows_reject_group_leakage_and_insufficient_coverage() -> None:
    puzzles = _puzzles()
    puzzles["group_id"][8] = puzzles["group_id"][0]
    with pytest.raises(ValueError, match="crosses splits"):
        select_probe_rows(
            puzzles,
            future_ply=2,
            fit_count=3,
            selection_count=2,
            evaluation_count=2,
        )

    puzzles = _puzzles()
    puzzles["future_valid_u8"][:7, 2] = 0
    with pytest.raises(ValueError, match="insufficient"):
        select_probe_rows(
            puzzles,
            future_ply=2,
            fit_count=3,
            selection_count=2,
            evaluation_count=2,
        )
