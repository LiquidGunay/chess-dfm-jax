from __future__ import annotations

import pytest
import torch
from torch import nn

from research.interpretability.models import (
    ARM_IDS,
    BT4ComparisonModels,
    _encoder_layout,
    require_arm_id,
)


class _ToyHead(nn.Module):
    def __init__(self, scale: float):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(scale))
        self.register_buffer("mapping_table", torch.arange(4), persistent=False)

    def forward(
        self,
        tokens: torch.Tensor,
        compute_dtype: torch.dtype,
    ) -> torch.Tensor:
        return tokens.to(compute_dtype).mean(dim=(-2, -1), keepdim=False)[:, None] * self.scale


class _ToyEncoder(nn.Module):
    def __init__(self, offset: float, head_scale: float):
        super().__init__()
        self.offset = nn.Parameter(torch.tensor(offset))
        self.policy_head = _ToyHead(head_scale)

    def encode_current(
        self,
        planes: torch.Tensor,
        *,
        compute_dtype: torch.dtype,
        remat: bool,
    ) -> torch.Tensor:
        assert remat is False
        return planes.to(compute_dtype) + self.offset


def _models() -> BT4ComparisonModels:
    raw = _ToyEncoder(offset=1.0, head_scale=2.0)
    hero = _ToyEncoder(offset=10.0, head_scale=3.0)
    _layout, layout_sha256 = _encoder_layout(raw)  # type: ignore[arg-type]
    return BT4ComparisonModels(
        raw_encoder=raw,  # type: ignore[arg-type]
        hero_encoder=hero,  # type: ignore[arg-type]
        raw_manifest={
            "raw_asset": {"sha256": "raw-state"},
            "combined_sha256": "raw-mapping",
        },
        hero_manifest={
            "state": {"sha256": "hero-state", "leaf_count": 2},
        },
        layout_sha256=layout_sha256,
    )


def test_four_arms_share_models_and_select_encoder_then_head() -> None:
    models = _models()
    planes = torch.zeros((2, 64, 4), dtype=torch.float32)
    expected = {"RR": 2.0, "HR": 20.0, "RH": 3.0, "HH": 30.0}

    assert ARM_IDS == ("RR", "HR", "RH", "HH")
    for arm, value in expected.items():
        output = models.policy_logits(
            planes,
            arm=require_arm_id(arm),
            compute_dtype=torch.float32,
        )
        torch.testing.assert_close(
            output.logits,
            torch.full((2, 1), value),
            rtol=0.0,
            atol=0.0,
        )
        assert output.captures is None

    assert models.raw_encoder is not models.hero_encoder
    assert models.resident_parameter_bytes == sum(
        parameter.numel() * parameter.element_size()
        for encoder in (models.raw_encoder, models.hero_encoder)
        for parameter in encoder.parameters()
    )


def test_lattice_path_runs_each_encoder_once_and_matches_individual_arms() -> None:
    models = _models()
    planes = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    calls = {"raw": 0, "hero": 0}
    raw_encode = models.raw_encoder.encode_current
    hero_encode = models.hero_encoder.encode_current

    def counted_raw(*args: object, **kwargs: object) -> torch.Tensor:
        calls["raw"] += 1
        return raw_encode(*args, **kwargs)

    def counted_hero(*args: object, **kwargs: object) -> torch.Tensor:
        calls["hero"] += 1
        return hero_encode(*args, **kwargs)

    models.raw_encoder.encode_current = counted_raw  # type: ignore[method-assign]
    models.hero_encoder.encode_current = counted_hero  # type: ignore[method-assign]
    lattice = models.lattice_logits(planes, compute_dtype=torch.float32)

    assert calls == {"raw": 1, "hero": 1}
    assert tuple(lattice.logits) == ARM_IDS
    assert lattice.raw_captures is None
    assert lattice.hero_captures is None
    expected = {
        "RR": [13.0, 37.0],
        "HR": [31.0, 55.0],
        "RH": [19.5, 55.5],
        "HH": [46.5, 82.5],
    }
    for arm, values in expected.items():
        torch.testing.assert_close(
            lattice.logits[require_arm_id(arm)],
            torch.tensor(values)[:, None],
            rtol=0.0,
            atol=0.0,
        )


def test_comparison_descriptor_and_arm_validation_are_explicit() -> None:
    models = _models()
    descriptor = models.descriptor()
    assert descriptor["arms"] == ["RR", "HR", "RH", "HH"]
    assert descriptor["raw_asset_sha256"] == "raw-state"
    assert descriptor["raw_mapping_sha256"] == "raw-mapping"
    assert descriptor["hero_state_sha256"] == "hero-state"
    assert descriptor["on_disk_mixed_checkpoints"] is False

    with pytest.raises(ValueError, match="Unknown comparison arm"):
        require_arm_id("RX")
    with pytest.raises(ValueError, match="Unknown comparison arm"):
        models.policy_logits(
            torch.zeros((1, 64, 4)),
            arm="RX",  # type: ignore[arg-type]
            compute_dtype=torch.float32,
        )


def test_comparison_rejects_layout_drift() -> None:
    raw = _ToyEncoder(offset=1.0, head_scale=2.0)
    hero = _ToyEncoder(offset=10.0, head_scale=3.0)
    hero.extra = nn.Parameter(torch.zeros(2))
    _layout, layout_sha256 = _encoder_layout(raw)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="encoder ABI mismatch"):
        BT4ComparisonModels(
            raw_encoder=raw,  # type: ignore[arg-type]
            hero_encoder=hero,  # type: ignore[arg-type]
            raw_manifest={},
            hero_manifest={},
            layout_sha256=layout_sha256,
        )
