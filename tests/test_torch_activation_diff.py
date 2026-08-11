from __future__ import annotations

import numpy as np
import pytest
import torch

from research.interpretability.activation_diff import (
    HOOK_FIELDS,
    POSITION_METRICS,
    ActivationDiffAccumulator,
)
from research.train_torch import BT4EncoderCapture


def _capture(value: torch.Tensor, layers: tuple[int, ...]) -> BT4EncoderCapture:
    return BT4EncoderCapture(layers, value, value, value, value, value)


def test_activation_diff_streams_all_hooks_and_depth_map() -> None:
    generator = torch.Generator().manual_seed(7)
    raw = torch.randn((2, 3, 64, 8), generator=generator)
    hero = raw + 0.25
    accumulator = ActivationDiffAccumulator(
        layer_indices=(0, 1),
        feature_width=8,
        sketch_width=4,
        projection_seed=11,
    )
    accumulator.update(_capture(raw, (0, 1)), _capture(hero, (0, 1)))
    position = accumulator.position_metric_arrays()
    assert set(position) == set(POSITION_METRICS)
    assert all(value.shape == (3, 5, 2) for value in position.values())
    np.testing.assert_allclose(position["delta_rms"], 0.25)

    summary = accumulator.finalize(
        np.asarray(["a", "b", "b"]),
        bootstrap_seed=13,
        bootstrap_replicates=200,
    )
    assert summary["position_count"] == 3
    assert set(summary["hooks"]) == set(HOOK_FIELDS)
    layer = summary["hooks"]["hook_attn_in"]["layers"]["0"]
    assert layer["exact_elementwise"]["delta_rms"] == pytest.approx(0.25)
    assert layer["position_level_game_cluster_bootstrap"]["delta_rms"]["cluster_count"] == 2
    depth = summary["layer_correspondence"]
    assert np.asarray(depth["matrices_by_nested_sketch_width"]["4"]).shape == (2, 2)
    assert depth["observation_count"] == 192
    assert len(summary["selected_layers_for_expensive_development_work"]["layers"]) <= 4


def test_activation_diff_identical_captures_have_unit_similarity() -> None:
    raw = torch.arange(2 * 2 * 64 * 8, dtype=torch.float32).reshape(2, 2, 64, 8)
    accumulator = ActivationDiffAccumulator(
        layer_indices=(3, 9),
        feature_width=8,
        sketch_width=4,
        projection_seed=1,
    )
    accumulator.update(_capture(raw, (3, 9)), _capture(raw.clone(), (3, 9)))
    summary = accumulator.finalize(
        np.asarray(["x", "y"]),
        bootstrap_seed=2,
        bootstrap_replicates=100,
    )
    layer = summary["hooks"]["resid_post_after_ln"]["layers"]["3"]
    assert layer["exact_elementwise"]["relative_l2_to_raw"] == 0.0
    assert layer["projected_multivariate"]["linear_cka"] == pytest.approx(1.0)
    assert layer["projected_multivariate"]["svcca_mean"] == pytest.approx(1.0)


def test_activation_diff_rejects_layer_and_shape_drift() -> None:
    value = torch.zeros((1, 1, 64, 8))
    accumulator = ActivationDiffAccumulator(
        layer_indices=(0,),
        feature_width=8,
        sketch_width=2,
    )
    with pytest.raises(ValueError, match="layers differ"):
        accumulator.update(_capture(value, (1,)), _capture(value, (0,)))
    bad = torch.zeros((1, 1, 3, 8))
    with pytest.raises(ValueError, match="Unexpected"):
        accumulator.update(_capture(bad, (0,)), _capture(bad, (0,)))
