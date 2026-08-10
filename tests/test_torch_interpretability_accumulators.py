from __future__ import annotations

import numpy as np
import pytest

import research.interpretability.accumulators as accumulator_module
from research.interpretability.accumulators import (
    ProjectedPairAccumulator,
    StreamingPairedMoments,
    cluster_bootstrap_mean_ci,
    paired_position_metrics,
    rademacher_projection,
)


def test_streaming_paired_moments_matches_direct_and_merge() -> None:
    raw = np.arange(24, dtype=np.float32).reshape(3, 2, 4)
    hero = raw * 2.0 + 1.0
    whole = StreamingPairedMoments()
    whole.update(raw, hero)
    first = StreamingPairedMoments()
    second = StreamingPairedMoments()
    first.update(raw[:1], hero[:1])
    second.update(raw[1:], hero[1:])
    first.merge(second)

    assert first.finalize() == whole.finalize()
    result = whole.finalize()
    assert result["element_count"] == raw.size
    assert result["raw_mean"] == pytest.approx(float(raw.mean()))
    assert result["hero_mean"] == pytest.approx(float(hero.mean()))
    assert result["delta_rms"] == pytest.approx(float(np.sqrt(np.mean(np.square(hero - raw)))))
    assert result["unchanged_fraction"] == 0.0


def test_position_metrics_are_one_row_per_position() -> None:
    raw = np.ones((2, 3, 4), dtype=np.float32)
    hero = raw.copy()
    hero[1] *= 2.0
    metrics = paired_position_metrics(raw, hero)
    assert set(metrics) == {
        "relative_l2_to_raw",
        "symmetric_relative_l2",
        "cosine_similarity",
        "delta_rms",
    }
    assert metrics["relative_l2_to_raw"].tolist() == pytest.approx([0.0, 1.0])
    assert metrics["cosine_similarity"].tolist() == pytest.approx([1.0, 1.0])


def test_projected_covariance_similarity_is_mergeable_and_detects_rotation() -> None:
    rng = np.random.default_rng(7)
    raw = rng.normal(size=(400, 8))
    rotation, _ = np.linalg.qr(rng.normal(size=(8, 8)))
    hero = raw @ rotation
    whole = ProjectedPairAccumulator(8)
    whole.update(raw, hero)
    pieces = ProjectedPairAccumulator(8)
    for start in range(0, 400, 37):
        part = ProjectedPairAccumulator(8)
        part.update(raw[start : start + 37], hero[start : start + 37])
        pieces.merge(part)

    result = whole.finalize()
    merged = pieces.finalize()
    assert result["linear_cka"] == pytest.approx(1.0, abs=1e-12)
    assert result["svcca_mean"] == pytest.approx(1.0, abs=1e-10)
    assert result["orthogonal_procrustes_relative_residual"] == pytest.approx(0.0, abs=1e-7)
    assert merged["linear_cka"] == pytest.approx(result["linear_cka"], abs=1e-12)


def test_projection_and_cluster_bootstrap_are_deterministic() -> None:
    first, first_sha = rademacher_projection(32, 7, seed=11)
    second, second_sha = rademacher_projection(32, 7, seed=11)
    np.testing.assert_array_equal(first, second)
    assert first_sha == second_sha
    np.testing.assert_allclose(
        np.unique(first),
        [-1.0 / np.sqrt(7), 1.0 / np.sqrt(7)],
    )

    values = np.asarray([1.0, 3.0, 10.0, 12.0])
    groups = np.asarray(["a", "a", "b", "b"])
    ci1 = cluster_bootstrap_mean_ci(values, groups, seed=9, replicates=1000)
    ci2 = cluster_bootstrap_mean_ci(values, groups, seed=9, replicates=1000)
    assert ci1 == ci2
    assert ci1["estimate"] == pytest.approx(6.5)
    assert ci1["cluster_count"] == 2


def test_cluster_bootstrap_is_invariant_to_bounded_draw_chunks(monkeypatch) -> None:
    values = np.arange(17, dtype=np.float64)
    groups = np.asarray([f"game-{index // 2}" for index in range(values.size)])
    monkeypatch.setattr(accumulator_module, "_MAX_BOOTSTRAP_SAMPLE_INDICES", 10_000)
    unchunked = cluster_bootstrap_mean_ci(values, groups, seed=31, replicates=257)
    monkeypatch.setattr(accumulator_module, "_MAX_BOOTSTRAP_SAMPLE_INDICES", 19)
    chunked = cluster_bootstrap_mean_ci(values, groups, seed=31, replicates=257)

    assert chunked["estimate"] == unchunked["estimate"]
    assert chunked["lower"] == unchunked["lower"]
    assert chunked["upper"] == unchunked["upper"]
