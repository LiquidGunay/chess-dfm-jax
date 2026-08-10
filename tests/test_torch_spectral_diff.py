from __future__ import annotations

import pytest
import torch

from research.interpretability.spectral_diff import (
    SpectralConfig,
    low_rank_svd,
    spectrum_metrics,
    update_alignment,
)


def test_exact_spectrum_reports_complete_rank_one_update() -> None:
    matrix = torch.tensor(
        [[3.0, 0.0], [0.0, 0.0]],
        dtype=torch.float32,
    )
    spectrum = low_rank_svd(
        matrix,
        config=SpectralConfig(
            rank=1,
            oversample=0,
            power_iterations=0,
            exact_max_min_dimension=2,
        ),
        seed=1,
    )
    metrics = spectrum_metrics(
        spectrum,
        frobenius_squared=9.0,
        minimum_dimension=2,
    )

    assert spectrum.method == "exact"
    assert spectrum.singular_values.tolist() == [3.0, 0.0]
    assert metrics["captured_energy_fraction"] == 1.0
    assert metrics["stable_rank"] == 1.0
    assert metrics["rank_at_total_energy"]["99pct"] == 1
    assert metrics["energy_rank_bounds"]["participation_rank_lower"] == 1.0
    assert metrics["energy_rank_bounds"]["participation_rank_upper"] == 1.0


def test_randomized_spectrum_is_seeded_and_accounts_for_residual() -> None:
    matrix = torch.diag(torch.tensor([5.0, 3.0, 1.0, 0.5]))
    config = SpectralConfig(
        rank=2,
        oversample=2,
        power_iterations=2,
        exact_max_min_dimension=1,
        seed=9,
    )
    first = low_rank_svd(matrix, config=config, seed=17)
    second = low_rank_svd(matrix, config=config, seed=17)
    metrics = spectrum_metrics(
        first,
        frobenius_squared=float(matrix.square().sum()),
        minimum_dimension=4,
    )

    assert first.method == "randomized"
    torch.testing.assert_close(first.singular_values, second.singular_values)
    torch.testing.assert_close(
        first.singular_values,
        torch.tensor([5.0, 3.0]),
        rtol=1e-5,
        atol=1e-5,
    )
    assert metrics["captured_energy_fraction"] == pytest.approx(34.0 / 35.25)
    assert metrics["relative_frobenius_residual"] == pytest.approx((1.25 / 35.25) ** 0.5, rel=1e-5)
    bounds = metrics["energy_rank_bounds"]
    assert bounds["participation_rank_lower"] <= bounds["participation_rank_upper"]
    assert bounds["entropy_effective_rank_lower"] <= bounds["entropy_effective_rank_upper"]


def test_update_alignment_detects_shared_leading_subspace() -> None:
    raw = torch.diag(torch.tensor([7.0, 2.0, 1.0]))
    delta = torch.diag(torch.tensor([3.0, 0.0, 0.0]))
    config = SpectralConfig(
        rank=1,
        oversample=0,
        power_iterations=0,
        exact_max_min_dimension=3,
    )
    raw_svd = low_rank_svd(raw, config=config, seed=1)
    delta_svd = low_rank_svd(delta, config=config, seed=2)
    alignment = update_alignment(
        delta,
        delta_svd=delta_svd,
        raw_svd=raw_svd,
        delta_frobenius_squared=9.0,
        max_rank=1,
    )

    assert alignment["rank"] == 1
    assert alignment["delta_energy_in_raw_left_subspace"] == pytest.approx(1.0)
    assert alignment["delta_energy_in_raw_right_subspace"] == pytest.approx(1.0)
    assert alignment["delta_energy_in_raw_bilinear_subspace"] == pytest.approx(1.0)
    assert alignment["left_subspace_overlap"] == pytest.approx(1.0)
    assert alignment["right_subspace_overlap"] == pytest.approx(1.0)
