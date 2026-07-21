from __future__ import annotations

import numpy as np
import pytest

from research.evaluate_dense_representations import (
    _cast_payload_like,
    _effective_rank_metrics,
    encoder_payload_sha256,
    layer_correspondence_matrix,
    paired_representation_metrics,
    policy_pair_metrics,
)


def test_encoder_payload_digest_is_path_and_value_sensitive() -> None:
    first = {
        "embedding": {"w": np.arange(6, dtype=np.float32).reshape(2, 3)},
        "layers": {0: {"b": np.asarray([1.0, 2.0], dtype=np.float32)}},
    }
    identical = {
        "layers": {0: {"b": np.asarray([1.0, 2.0], dtype=np.float32)}},
        "embedding": {"w": np.arange(6, dtype=np.float32).reshape(2, 3)},
    }
    changed = {
        "embedding": {"w": np.arange(6, dtype=np.float32).reshape(2, 3)},
        "layers": {0: {"b": np.asarray([1.0, 3.0], dtype=np.float32)}},
    }
    assert encoder_payload_sha256(first) == encoder_payload_sha256(identical)
    assert encoder_payload_sha256(first) != encoder_payload_sha256(changed)


def test_cast_payload_like_checks_tree_and_shape() -> None:
    destination = {"a": np.zeros((2, 3), dtype=np.float32)}
    source = {"a": np.ones((2, 3), dtype=np.float16)}
    cast = _cast_payload_like(destination, source)
    assert np.asarray(cast["a"]).dtype == np.dtype(np.float32)
    np.testing.assert_array_equal(cast["a"], np.ones((2, 3), dtype=np.float32))

    with pytest.raises(ValueError, match="shape differs"):
        _cast_payload_like(destination, {"a": np.ones((3, 2), dtype=np.float16)})
    with pytest.raises(ValueError, match="keys differ"):
        _cast_payload_like(destination, {"b": np.ones((2, 3), dtype=np.float16)})


def test_identical_and_rotated_representations_score_as_equivalent() -> None:
    rng = np.random.default_rng(7)
    source = rng.normal(size=(256, 12)).astype(np.float32)
    q, _ = np.linalg.qr(rng.normal(size=(12, 12)))
    rotated = np.asarray(source @ q, dtype=np.float32)

    identical, _, _, identical_correlations = paired_representation_metrics(
        source,
        source.copy(),
        retained_variance=0.99,
    )
    assert identical["linear_cka"] == pytest.approx(1.0, abs=2e-5)
    assert identical["orthogonal_procrustes_r2"] == pytest.approx(1.0, abs=2e-5)
    assert identical["svcca_mean_correlation"] == pytest.approx(1.0, abs=2e-5)
    np.testing.assert_allclose(identical_correlations, 1.0, atol=2e-5)

    metrics, _, _, correlations = paired_representation_metrics(
        source,
        rotated,
        retained_variance=0.99,
    )
    assert metrics["linear_cka"] == pytest.approx(1.0, abs=2e-5)
    assert metrics["orthogonal_procrustes_r2"] == pytest.approx(1.0, abs=2e-5)
    assert metrics["svcca_mean_correlation"] == pytest.approx(1.0, abs=2e-5)
    np.testing.assert_allclose(correlations, 1.0, atol=2e-5)


def test_effective_rank_metrics_detect_rank_one_collapse() -> None:
    metrics = _effective_rank_metrics(
        np.asarray([0.0, 0.0, 0.0, 4.0], dtype=np.float32)
    )
    assert metrics == {
        "effective_rank": pytest.approx(1.0),
        "participation_ratio": pytest.approx(1.0),
        "stable_rank": pytest.approx(1.0),
        "numerical_rank": 1,
        "pca99_dimension": 1,
    }


def test_layer_correspondence_recovers_a_known_permutation() -> None:
    rng = np.random.default_rng(11)
    source = rng.normal(size=(5, 3, 4, 64, 8)).astype(np.float32)
    permutation = np.asarray([2, 0, 1])
    candidate = source[:, permutation].copy()
    matrix = layer_correspondence_matrix(
        source,
        candidate,
        hook_index=2,
        view="board_pooled",
    )
    # Candidate layers [2,0,1] correspond to source layers [2,0,1].
    np.testing.assert_array_equal(np.argmax(matrix, axis=1), [1, 2, 0])
    np.testing.assert_allclose(np.max(matrix, axis=1), 1.0, atol=2e-5)


def test_policy_pair_metrics_are_exact_for_identical_logits() -> None:
    logits = np.asarray(
        [
            [3.0, 2.0, 1.0, -1.0, -2.0, -3.0],
            [-1.0, 1.0, 4.0, 0.0, 2.0, -2.0],
            [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        ],
        dtype=np.float32,
    )
    legal = np.ones_like(logits, dtype=bool)
    metrics = policy_pair_metrics(
        logits,
        logits.copy(),
        legal,
        bootstrap_samples=100,
        seed_prefix="identical",
    )
    assert metrics["source_to_candidate_kl"]["mean"] == pytest.approx(0.0)
    assert metrics["candidate_to_source_kl"]["mean"] == pytest.approx(0.0)
    assert metrics["jensen_shannon"]["mean"] == pytest.approx(0.0)
    assert metrics["top1_agreement"]["mean"] == pytest.approx(1.0)
    assert metrics["top5_set_jaccard"]["mean"] == pytest.approx(1.0)
