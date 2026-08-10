from __future__ import annotations

import math

import pytest
import torch

from research.interpretability.parameter_diff import (
    NormAccumulator,
    _git_commit,
    _parameter_groups,
    _tensor_stats,
)


def test_tensor_stats_and_accumulator_match_hand_computation() -> None:
    raw = torch.tensor([1.0, 2.0], dtype=torch.float32)
    hero = torch.tensor([1.0, 4.0], dtype=torch.float32)
    stats = _tensor_stats(raw, hero)

    assert stats.count == 2
    assert stats.raw_sum_squares == 5.0
    assert stats.hero_sum_squares == 17.0
    assert stats.delta_sum_squares == 4.0
    assert stats.raw_hero_dot == 9.0
    assert stats.delta_sum == 2.0
    assert stats.delta_absolute_sum == 2.0
    assert stats.delta_absolute_max == 2.0
    assert stats.unchanged_count == 1

    accumulator = NormAccumulator()
    accumulator.add(stats)
    metrics = accumulator.metrics()
    assert metrics["raw_l2"] == pytest.approx(math.sqrt(5.0))
    assert metrics["hero_l2"] == pytest.approx(math.sqrt(17.0))
    assert metrics["delta_l2"] == 2.0
    assert metrics["relative_delta_l2"] == pytest.approx(2.0 / math.sqrt(5.0))
    assert metrics["cosine"] == pytest.approx(9.0 / math.sqrt(85.0))
    assert metrics["delta_mean"] == 1.0
    assert metrics["delta_rms"] == pytest.approx(math.sqrt(2.0))
    assert metrics["unchanged_fraction"] == 0.5


def test_parameter_grouping_preserves_layer_and_component_views() -> None:
    assert _parameter_groups("embedding.proj.w") == (
        "all",
        "trunk",
        "embedding",
    )
    assert _parameter_groups("layers.3.smolgen.compress.w") == (
        "all",
        "trunk",
        "layers",
        "layer_03",
        "smolgen",
        "layer_03.smolgen",
    )
    assert _parameter_groups("layers.14.ffn2.b")[-2:] == (
        "mlp",
        "layer_14.mlp",
    )
    assert _parameter_groups("policy_head.q.w") == ("all", "policy_head")


def test_tensor_stats_rejects_parameter_abi_mismatch() -> None:
    with pytest.raises(ValueError, match="Parameter ABI mismatch"):
        _tensor_stats(torch.zeros(2), torch.zeros(3))


def test_git_commit_is_optional_when_git_binary_is_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing_git(*_args: object, **_kwargs: object) -> None:
        raise FileNotFoundError("git")

    monkeypatch.setattr(
        "research.interpretability.parameter_diff.subprocess.run",
        missing_git,
    )
    assert _git_commit() is None
