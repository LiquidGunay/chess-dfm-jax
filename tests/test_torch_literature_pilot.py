from __future__ import annotations

import numpy as np
import pytest
import torch

from research.interpretability.literature_pilot import (
    bootstrap_mean_interval,
    legal_policy_js,
    map_agreement,
)


def test_bootstrap_mean_interval_is_deterministic_and_contains_constant() -> None:
    first = bootstrap_mean_interval([3.0, 3.0, 3.0], samples=50, seed=7)
    second = bootstrap_mean_interval(np.asarray([3.0, 3.0, 3.0]), samples=50, seed=7)
    assert first == second
    assert first["mean"] == first["lower"] == first["upper"] == 3.0
    with pytest.raises(ValueError, match="non-empty finite"):
        bootstrap_mean_interval([])


def test_legal_policy_js_is_zero_for_identity_and_symmetric() -> None:
    first = torch.tensor([[2.0, 0.0, 1.0, -4.0], [0.0, 2.0, -1.0, 1.0]])
    second = torch.tensor([[0.0, 2.0, 1.0, 9.0], [1.0, 0.0, -1.0, 2.0]])
    legal = torch.tensor([[0, 1, 2, 65535], [0, 1, 3, 65535]])
    counts = torch.tensor([3, 3])
    identity = legal_policy_js(first, first.clone(), legal, counts)
    torch.testing.assert_close(identity, torch.zeros(2), atol=1e-7, rtol=0)
    forward = legal_policy_js(first, second, legal, counts)
    reverse = legal_policy_js(second, first, legal, counts)
    torch.testing.assert_close(forward, reverse)
    assert bool((forward >= 0).all())


def test_map_agreement_reports_top_k_overlap() -> None:
    first = torch.arange(64, dtype=torch.float32)
    identical = map_agreement(first, first.clone(), top_k=8)
    assert identical["cosine"] == pytest.approx(1.0)
    assert identical["pearson"] == pytest.approx(1.0)
    assert identical["top_k_jaccard"] == 1.0
    assert identical["argmax_agreement"]

    reverse = map_agreement(first, torch.flip(first, dims=(0,)), top_k=8)
    assert reverse["pearson"] == pytest.approx(-1.0)
    assert reverse["top_k_jaccard"] == 0.0
