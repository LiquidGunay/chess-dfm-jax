from __future__ import annotations

import numpy as np
import pytest
import torch

from research.interpretability.behavior_diff import (
    evaluate_lattice_batch,
    summarize_behavior_records,
)


def _logits() -> dict[str, torch.Tensor]:
    return {
        "RR": torch.tensor([[0.0, 3.0, 1.0, -2.0], [2.0, 0.0, 1.0, -1.0]]),
        "HR": torch.tensor([[0.0, 1.0, 3.0, -2.0], [2.0, 0.0, 1.0, -1.0]]),
        "RH": torch.tensor([[0.0, 3.0, 1.0, -2.0], [1.0, 0.0, 2.0, -1.0]]),
        "HH": torch.tensor([[0.0, 1.0, 3.0, -2.0], [1.0, 0.0, 2.0, -1.0]]),
    }


def test_lattice_behavior_respects_legal_prefix_and_decomposes_interaction() -> None:
    records = evaluate_lattice_batch(
        _logits(),  # type: ignore[arg-type]
        legal_idx=np.asarray([[1, 2, 99], [0, 2, 99]]),
        legal_count=np.asarray([2, 2]),
        targets=np.asarray([1, 2]),
    )
    assert len(records) == 2
    assert records[0]["arms"]["RR"]["top1_action"] == 1
    assert records[0]["arms"]["HR"]["top1_action"] == 2
    assert records[0]["pairs"]["RR__HH"]["top1_agreement"] == 0
    assert records[1]["pairs"]["RR__HR"]["js_divergence"] == pytest.approx(0.0)
    assert records[0]["decomposition"]["probability_interaction_rms"] == pytest.approx(
        0.0, abs=1e-12
    )
    assert 0.0 < records[0]["arms"]["RR"]["legal_mass_before_masking"] < 1.0


def test_behavior_summary_is_paired_and_bootstrapped_by_game() -> None:
    records = evaluate_lattice_batch(
        _logits(),  # type: ignore[arg-type]
        legal_idx=np.asarray([[1, 2], [0, 2]]),
        legal_count=np.asarray([2, 2]),
        targets=np.asarray([1, 2]),
    )
    summary = summarize_behavior_records(
        records,
        np.asarray(["game-a", "game-b"]),
        bootstrap_seed=123,
        bootstrap_replicates=500,
    )
    assert summary["position_count"] == 2
    assert summary["game_count"] == 2
    total_js = summary["pairs"]["RR__HH"]["js_divergence"]
    assert total_js["cluster_count"] == 2
    assert total_js["replicates"] == 500
    assert 0.0 <= summary["arms"]["RR"]["ece_10_equal_width_bins"] <= 1.0


def test_behavior_rejects_bad_lattice_order_and_illegal_target() -> None:
    logits = _logits()
    reordered = {key: logits[key] for key in ("HH", "HR", "RH", "RR")}
    with pytest.raises(ValueError, match="ordered"):
        evaluate_lattice_batch(
            reordered,  # type: ignore[arg-type]
            np.asarray([[1, 2], [0, 2]]),
            np.asarray([2, 2]),
            np.asarray([1, 2]),
        )
    with pytest.raises(ValueError, match="Target action"):
        evaluate_lattice_batch(
            logits,  # type: ignore[arg-type]
            np.asarray([[1, 2], [0, 2]]),
            np.asarray([2, 2]),
            np.asarray([0, 2]),
        )
