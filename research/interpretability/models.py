"""Strict, storage-neutral assembly of raw/Hero encoder-head comparison arms."""

from __future__ import annotations

import gc
import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, NamedTuple, cast

import torch
from torch import Tensor

from research.train_torch import (
    HERO_CONFIG,
    BT4Encoder,
    BT4EncoderCapture,
    BT4PolicyHead,
    JointModel,
    load_model_checkpoint,
    load_raw_bt4_hero_model,
)


ArmId = Literal["RR", "HR", "RH", "HH"]
ARM_IDS: tuple[ArmId, ...] = ("RR", "HR", "RH", "HH")


class ComparisonPolicyOutput(NamedTuple):
    """Policy output and optional selected-layer captures for one lattice arm."""

    logits: Tensor
    tokens: Tensor
    captures: BT4EncoderCapture | None


class ComparisonLatticeOutput(NamedTuple):
    """All four policy arms from exactly one pass through each encoder."""

    logits: dict[ArmId, Tensor]
    raw_tokens: Tensor
    hero_tokens: Tensor
    raw_captures: BT4EncoderCapture | None
    hero_captures: BT4EncoderCapture | None


def _encoder_layout(encoder: BT4Encoder) -> tuple[dict[str, tuple[Any, ...]], str]:
    layout: dict[str, tuple[Any, ...]] = {}
    digest = hashlib.sha256()
    for name, parameter in encoder.named_parameters():
        record = (tuple(parameter.shape), str(parameter.dtype))
        layout[name] = record
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(record[0]).encode("ascii"))
        digest.update(b"\0")
        digest.update(record[1].encode("ascii"))
        digest.update(b"\n")
    return layout, digest.hexdigest()


def _parameter_bytes(module: torch.nn.Module) -> int:
    return sum(parameter.numel() * parameter.element_size() for parameter in module.parameters())


@dataclass
class BT4ComparisonModels:
    """Two shared encoders and heads implementing the four comparison arms.

    The first arm letter selects the encoder (R=raw, H=Hero); the second
    selects the native LC0 policy head. No parameter tensor is cloned and no
    mixed checkpoint is written to disk.
    """

    raw_encoder: BT4Encoder
    hero_encoder: BT4Encoder
    raw_manifest: dict[str, Any]
    hero_manifest: dict[str, Any]
    layout_sha256: str

    def __post_init__(self) -> None:
        raw_layout, raw_digest = _encoder_layout(self.raw_encoder)
        hero_layout, hero_digest = _encoder_layout(self.hero_encoder)
        if raw_layout != hero_layout:
            missing = sorted(set(hero_layout) - set(raw_layout))[:10]
            extra = sorted(set(raw_layout) - set(hero_layout))[:10]
            mismatched = [
                name
                for name in sorted(set(raw_layout) & set(hero_layout))
                if raw_layout[name] != hero_layout[name]
            ][:10]
            raise ValueError(
                "Raw/Hero encoder ABI mismatch: "
                f"missing={missing}, extra={extra}, mismatched={mismatched}"
            )
        if raw_digest != hero_digest or self.layout_sha256 != raw_digest:
            raise ValueError("Raw/Hero encoder layout digest mismatch")
        raw_head = self.raw_encoder.policy_head
        hero_head = self.hero_encoder.policy_head
        if raw_head is None or hero_head is None:
            raise ValueError("Both comparison encoders require native policy heads")
        if not torch.equal(raw_head.mapping_table, hero_head.mapping_table):
            raise ValueError("Raw/Hero policy-head mapping tables differ")

    @property
    def device(self) -> torch.device:
        raw_device = next(self.raw_encoder.parameters()).device
        hero_device = next(self.hero_encoder.parameters()).device
        if raw_device != hero_device:
            raise ValueError(
                f"Raw/Hero encoders are on different devices: {raw_device} != {hero_device}"
            )
        return raw_device

    @property
    def resident_parameter_bytes(self) -> int:
        return _parameter_bytes(self.raw_encoder) + _parameter_bytes(self.hero_encoder)

    def eval(self) -> BT4ComparisonModels:
        self.raw_encoder.eval()
        self.hero_encoder.eval()
        return self

    def to(self, device: torch.device | str) -> BT4ComparisonModels:
        self.raw_encoder.to(device)
        self.hero_encoder.to(device)
        return self

    def _encoder_and_head(self, arm: ArmId | str) -> tuple[BT4Encoder, BT4PolicyHead]:
        if arm not in ARM_IDS:
            raise ValueError(f"Unknown comparison arm {arm!r}; expected {ARM_IDS}")
        encoder = self.raw_encoder if arm[0] == "R" else self.hero_encoder
        head_owner = self.raw_encoder if arm[1] == "R" else self.hero_encoder
        head = head_owner.policy_head
        if head is None:  # pragma: no cover - guarded by __post_init__
            raise AssertionError("Comparison policy head disappeared")
        return encoder, head

    def policy_logits(
        self,
        planes: Tensor,
        *,
        arm: ArmId,
        compute_dtype: torch.dtype,
    ) -> ComparisonPolicyOutput:
        """Run one lattice arm without retaining intermediate activations."""

        encoder, head = self._encoder_and_head(arm)
        tokens = encoder.encode_current(
            planes,
            compute_dtype=compute_dtype,
            remat=False,
        )
        logits = head(tokens, compute_dtype)
        return ComparisonPolicyOutput(logits, tokens, None)

    def policy_logits_with_captures(
        self,
        planes: Tensor,
        *,
        arm: ArmId,
        compute_dtype: torch.dtype,
        capture_layers: Sequence[int] | None = None,
        attention_input_overrides: Mapping[int, Tensor] | None = None,
        attention_output_overrides: Mapping[int, Tensor] | None = None,
        resid_mid_overrides: Mapping[int, Tensor] | None = None,
        mlp_output_overrides: Mapping[int, Tensor] | None = None,
        resid_post_overrides: Mapping[int, Tensor] | None = None,
    ) -> ComparisonPolicyOutput:
        """Run one lattice arm with selected canonical encoder captures."""

        encoder, head = self._encoder_and_head(arm)
        tokens, captures = encoder.encode_current_with_captures(
            planes,
            compute_dtype=compute_dtype,
            capture_layers=capture_layers,
            attention_input_overrides=attention_input_overrides,
            attention_output_overrides=attention_output_overrides,
            resid_mid_overrides=resid_mid_overrides,
            mlp_output_overrides=mlp_output_overrides,
            resid_post_overrides=resid_post_overrides,
        )
        logits = head(tokens, compute_dtype)
        return ComparisonPolicyOutput(logits, tokens, captures)

    def lattice_logits(
        self,
        planes: Tensor,
        *,
        compute_dtype: torch.dtype,
        capture_layers: Sequence[int] | None = None,
    ) -> ComparisonLatticeOutput:
        """Evaluate RR/HR/RH/HH with one pass through each encoder.

        Passing ``capture_layers`` enables the frozen hook ABI for both
        encoders. ``None`` is deliberately the no-capture setting; callers
        that want all layers pass ``range(15)`` explicitly. This avoids the
        four redundant encoder passes implied by evaluating each lattice arm
        independently.
        """

        def encode(
            encoder: BT4Encoder,
        ) -> tuple[Tensor, BT4EncoderCapture | None]:
            if capture_layers is None:
                return (
                    encoder.encode_current(
                        planes,
                        compute_dtype=compute_dtype,
                        remat=False,
                    ),
                    None,
                )
            return encoder.encode_current_with_captures(
                planes,
                compute_dtype=compute_dtype,
                capture_layers=capture_layers,
            )

        raw_tokens, raw_captures = encode(self.raw_encoder)
        hero_tokens, hero_captures = encode(self.hero_encoder)
        raw_head = self.raw_encoder.policy_head
        hero_head = self.hero_encoder.policy_head
        if raw_head is None or hero_head is None:  # pragma: no cover - invariant
            raise AssertionError("Comparison policy head disappeared")
        logits: dict[ArmId, Tensor] = {
            "RR": raw_head(raw_tokens, compute_dtype),
            "HR": raw_head(hero_tokens, compute_dtype),
            "RH": hero_head(raw_tokens, compute_dtype),
            "HH": hero_head(hero_tokens, compute_dtype),
        }
        return ComparisonLatticeOutput(
            logits=logits,
            raw_tokens=raw_tokens,
            hero_tokens=hero_tokens,
            raw_captures=raw_captures,
            hero_captures=hero_captures,
        )

    def descriptor(self) -> dict[str, Any]:
        return {
            "schema_version": "bt4-raw-hero-comparison-models-v1",
            "arms": list(ARM_IDS),
            "arm_semantics": "first letter encoder; second letter policy head",
            "raw_asset_sha256": self.raw_manifest["raw_asset"]["sha256"],
            "raw_mapping_sha256": self.raw_manifest["combined_sha256"],
            "hero_state_sha256": self.hero_manifest["state"]["sha256"],
            "hero_leaf_count": self.hero_manifest["state"]["leaf_count"],
            "encoder_layout_sha256": self.layout_sha256,
            "resident_parameter_bytes": self.resident_parameter_bytes,
            "on_disk_mixed_checkpoints": False,
        }


def load_bt4_comparison_models(
    *,
    raw_bt4_path: Path,
    hero_checkpoint_dir: Path,
    device: torch.device | str = torch.device("cpu"),
    model_config: Any = HERO_CONFIG,
) -> BT4ComparisonModels:
    """Strict-load raw and Hero once, retaining only their BT4 encoders.

    Full ``JointModel`` instances are required at load time to validate the
    Hero checkpoint ABI. Their non-encoder modules are released before the two
    encoders are moved to the requested device.
    """

    target_device = torch.device(device)
    raw_model, raw_manifest = load_raw_bt4_hero_model(
        device=torch.device("cpu"),
        raw_bt4_path=raw_bt4_path,
        config=model_config,
    )
    raw_encoder = raw_model.encoder
    del raw_model
    gc.collect()

    hero_model = JointModel(model_config)
    hero_manifest = load_model_checkpoint(
        checkpoint_dir=hero_checkpoint_dir,
        model=hero_model,
    )
    hero_encoder = hero_model.encoder
    del hero_model
    gc.collect()

    _raw_layout, raw_layout_sha256 = _encoder_layout(raw_encoder)
    models = BT4ComparisonModels(
        raw_encoder=raw_encoder,
        hero_encoder=hero_encoder,
        raw_manifest=dict(raw_manifest),
        hero_manifest=dict(hero_manifest),
        layout_sha256=raw_layout_sha256,
    )
    models.to(target_device).eval()
    return models


def require_arm_id(value: str) -> ArmId:
    """Validate an external string before passing it into typed arm APIs."""

    if value not in ARM_IDS:
        raise ValueError(f"Unknown comparison arm {value!r}; expected {ARM_IDS}")
    return cast(ArmId, value)
