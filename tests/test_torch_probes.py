from __future__ import annotations

import numpy as np
import pytest
import torch

from research.interpretability.probes import (
    binary_metrics,
    evaluate_probe,
    fit_probe,
    flatten_token_probe_examples,
    group_block_permutation,
    move_pair_features,
    paired_group_bootstrap_metric,
    pool_board_activations,
    random_direction_scores,
    validate_group_disjoint_splits,
)


def test_group_validation_and_block_permutation_preserve_structure() -> None:
    groups = np.asarray(["a", "a", "b", "b", "c", "c"])
    labels = np.asarray([1, 2, 3, 4, 5, 6])
    permuted = group_block_permutation(labels, groups, seed=4)
    assert sorted(permuted.tolist()) == sorted(labels.tolist())
    blocks = {tuple(permuted[groups == group]) for group in np.unique(groups)}
    assert blocks <= {(1, 2), (3, 4), (5, 6)}
    validate_group_disjoint_splits(groups, np.asarray([0, 0, 1, 1, 2, 2]))
    with pytest.raises(ValueError, match="crosses splits"):
        validate_group_disjoint_splits(groups, np.asarray([0, 1, 1, 1, 2, 2]))


def test_linear_binary_probe_fits_signal_with_train_only_normalization() -> None:
    generator = np.random.default_rng(8)
    train = generator.normal(size=(240, 5)).astype(np.float32)
    train_labels = (train[:, 0] - 0.7 * train[:, 1] > 0).astype(np.int64)
    probe = fit_probe(
        train,
        train_labels,
        task="binary",
        l2=1e-4,
        epochs=80,
        learning_rate=0.05,
        batch_size=64,
        seed=11,
    )
    metrics = evaluate_probe(probe, train, train_labels)
    assert metrics["auroc"] > 0.99
    np.testing.assert_allclose(probe.feature_mean.numpy(), train.mean(axis=0), atol=1e-6)
    outlier_dev = np.full((2, 5), 1e6, dtype=np.float32)
    probe.predict(outlier_dev)
    np.testing.assert_allclose(probe.feature_mean.numpy(), train.mean(axis=0), atol=1e-6)
    assert probe.metadata["normalization_fit"] == "training split only"


def test_binary_metrics_handle_ties_and_perfect_ordering() -> None:
    perfect = binary_metrics([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
    assert perfect["auroc"] == pytest.approx(1.0)
    assert perfect["average_precision"] == pytest.approx(1.0)
    tied = binary_metrics([0, 1], [0.5, 0.5])
    assert tied["auroc"] == pytest.approx(0.5)


def test_activation_feature_builders_retain_groups() -> None:
    activations = torch.arange(3 * 64 * 4, dtype=torch.float32).reshape(3, 64, 4)
    labels = torch.zeros((3, 64), dtype=torch.long)
    labels[:, 1] = 1
    features, targets, groups = flatten_token_probe_examples(
        activations,
        labels,
        ["g0", "g1", "g2"],
        max_positions=2,
        seed=2,
    )
    assert features.shape == (128, 4)
    assert targets.shape == (128,)
    assert np.unique(groups).size == 2
    assert pool_board_activations(activations, mode="mean_max").shape == (3, 8)
    pair = move_pair_features(activations, [0, 1, 2], [3, 4, 5])
    assert pair.shape == (3, 8)
    random = random_direction_scores(pair, direction_count=7, seed=5)
    assert random.shape == (3, 7)


def test_paired_group_bootstrap_is_deterministic_and_paired() -> None:
    labels = np.asarray([0, 1, 0, 1, 0, 1])
    raw = np.asarray([0.4, 0.6, 0.4, 0.6, 0.4, 0.6])
    hero = np.asarray([0.1, 0.9, 0.2, 0.8, 0.1, 0.9])
    groups = np.asarray(["a", "a", "b", "b", "c", "c"])

    def brier(target: np.ndarray, prediction: np.ndarray) -> float:
        return -float(np.square(prediction - target).mean())

    first = paired_group_bootstrap_metric(
        labels,
        raw,
        hero,
        groups,
        brier,
        seed=7,
        replicates=100,
    )
    second = paired_group_bootstrap_metric(
        labels,
        raw,
        hero,
        groups,
        brier,
        seed=7,
        replicates=100,
    )
    assert first == second
    assert first["delta_hero_minus_raw"] > 0
