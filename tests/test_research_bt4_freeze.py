from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import research.train as train  # noqa: E402
from chess_dfm_jax.nnx_bt4 import TrainableParam  # noqa: E402


class TrainableDummyEncoder(nnx.Module):
    def __init__(self, embedding_size: int = 16):
        self.embedding_size = embedding_size
        self.embedding = TrainableParam(
            jnp.linspace(0.25, 1.25, embedding_size, dtype=jnp.float32)
        )

    def encode_tokens(self, planes: jax.Array) -> jax.Array:
        batch_size = planes.shape[0]
        plane_mean = jnp.mean(planes, axis=(1, 2, 3))
        square = jnp.linspace(
            0.0,
            1.0,
            64,
            dtype=jnp.float32,
        )[None, :, None]
        return square + plane_mean.reshape((batch_size, 1, 1)) * self.embedding[
            None,
            None,
            :,
        ]


def _config(*, frozen: bool) -> train.JointLatentSASAConfig:
    return train.JointLatentSASAConfig(
        token_dim=8,
        z_dim=16,
        projector_layers=1,
        projector_num_heads=4,
        projector_mlp_dim=32,
        jepa_condition_dim=12,
        dfm_layers=1,
        jepa_layers=1,
        jepa_mlp_dim=32,
        num_heads=4,
        mlp_dim=32,
        action_vocab_size=32,
        horizon=2,
        compute_dtype="float32",
        param_dtype="float32",
        encoder_dtype="float32",
        first_legality_coeff=1.0,
        legality_on_masked_only=False,
        jepa_sigreg_coeff=0.01,
        jepa_pred_sigreg_coeff=0.02,
        jepa_sigreg_proj_dim=8,
        use_qk_gain=True,
        use_qk_norm=True,
        use_xsa=True,
        use_muon=False,
        learning_rate=1e-3,
        bt4_learning_rate=0.0,
        lr_warmup_steps=0,
        grad_clip_norm=0.0,
        skip_nonfinite_updates=False,
        remat_blocks=False,
        scan_layers=False,
        bt4_freeze_backbone=frozen,
    )


def _batch() -> dict[str, jax.Array]:
    current = jnp.linspace(
        -0.25,
        0.75,
        2 * 112 * 8 * 8,
        dtype=jnp.float32,
    ).reshape((2, 112, 8, 8))
    return {
        "current_planes": current,
        "future_planes": jnp.stack(
            [current + 0.125, current - 0.25],
            axis=1,
        ),
        "action_indices": jnp.asarray([[1, 2], [3, 4]], dtype=jnp.int32),
        "valid": jnp.ones((2,), dtype=jnp.float32),
        "future_valid": jnp.ones((2, 2), dtype=jnp.float32),
        "legal_idx": jnp.asarray(
            [
                [[1, 5, 9, 0], [2, 6, 0, 0]],
                [[3, 7, 11, 15], [4, 8, 12, 0]],
            ],
            dtype=jnp.int32,
        ),
        "legal_count": jnp.asarray([[3, 2], [4, 3]], dtype=jnp.int32),
        "legal_masks_valid": jnp.ones((2, 2), dtype=jnp.float32),
        "deterministic_t": jnp.asarray(0.0, dtype=jnp.float32),
    }


def _pure_trainable(module: nnx.Module) -> dict[str, Any]:
    return nnx.to_pure_dict(nnx.state(module, TrainableParam))


def _tree_norm(tree: Any) -> float:
    return float(
        np.sqrt(
            sum(
                np.sum(np.square(np.asarray(leaf, dtype=np.float64)))
                for leaf in jax.tree.leaves(tree)
            )
        )
    )


def _make_components(monkeypatch, *, frozen: bool, seed: int = 7):
    monkeypatch.setattr(
        train,
        "make_bt4_model",
        lambda *_args, **_kwargs: TrainableDummyEncoder(),
    )
    return train.create_joint_components({}, _config(frozen=frozen), seed=seed)


def test_freeze_config_and_serialization_fail_closed() -> None:
    default = train.JointLatentSASAConfig()
    assert "bt4_freeze_backbone" not in train.serialized_model_config(default)

    enabled = _config(frozen=True)
    train.validate_bt4_freeze_config(enabled)
    assert train.serialized_model_config(enabled)["bt4_freeze_backbone"] is True
    train.validate_legacy_init_for_config(enabled, "model-only")
    with pytest.raises(ValueError, match="--init model-only"):
        train.validate_legacy_init_for_config(enabled, "exact")
    with pytest.raises(ValueError, match="bt4_learning_rate=0.0"):
        train.validate_bt4_freeze_config(
            train.JointLatentSASAConfig(
                bt4_freeze_backbone=True,
                bt4_learning_rate=1e-6,
            )
        )
    with pytest.raises(ValueError, match="unfreeze_bt4_encoder=True"):
        train.validate_bt4_freeze_config(
            train.JointLatentSASAConfig(
                bt4_freeze_backbone=True,
                bt4_learning_rate=0.0,
                unfreeze_bt4_encoder=False,
            )
        )


def test_sparse_checkpoint_schedule_is_strict_and_absolute() -> None:
    args = train.parse_args(
        ["--save-updates", "400", "800", "1200", "1600"]
    )
    schedule = train.validate_save_updates(args.save_updates)
    assert schedule == (400, 800, 1200, 1600)
    assert train.should_save_checkpoint(
        invocation_update=17,
        research_update=800,
        save_every=0,
        save_updates=schedule,
    )
    assert not train.should_save_checkpoint(
        invocation_update=800,
        research_update=801,
        save_every=0,
        save_updates=schedule,
    )
    assert train.should_save_checkpoint(
        invocation_update=400,
        research_update=999,
        save_every=400,
        save_updates=(),
    )
    with pytest.raises(ValueError, match="positive integers"):
        train.validate_save_updates((0, 400))
    with pytest.raises(ValueError, match="strictly increasing"):
        train.validate_save_updates((800, 400))
    with pytest.raises(ValueError, match="strictly increasing"):
        train.validate_save_updates((400, 400))


def test_frozen_forward_is_exact_and_encoder_gradient_is_zero(monkeypatch) -> None:
    unfrozen, _ = _make_components(monkeypatch, frozen=False, seed=11)
    frozen, optimizer = _make_components(monkeypatch, frozen=True, seed=11)
    before_unfrozen = _pure_trainable(unfrozen)
    before_frozen = _pure_trainable(frozen)
    assert train.research_state_abi(before_unfrozen) == train.research_state_abi(
        before_frozen
    )
    for left, right in zip(
        jax.tree.leaves(before_unfrozen),
        jax.tree.leaves(before_frozen),
        strict=True,
    ):
        np.testing.assert_array_equal(left, right)

    planes = _batch()["current_planes"]
    np.testing.assert_array_equal(
        unfrozen.encode_bt4_tokens(planes),
        frozen.encode_bt4_tokens(planes),
    )

    loss_and_grad = nnx.value_and_grad(
        train.joint_stage1_loss_fn,
        argnums=nnx.DiffState(0, TrainableParam),
        has_aux=True,
    )
    (_, _), unfrozen_grad = loss_and_grad(
        unfrozen,
        _batch(),
        jax.random.PRNGKey(19),
        compute_fp32_legality=True,
    )
    (_, _), frozen_grad = loss_and_grad(
        frozen,
        _batch(),
        jax.random.PRNGKey(19),
        compute_fp32_legality=True,
    )
    unfrozen_pure = nnx.to_pure_dict(unfrozen_grad)
    frozen_pure = nnx.to_pure_dict(frozen_grad)
    assert _tree_norm(unfrozen_pure["encoder"]) > 0.0
    assert _tree_norm(frozen_pure["encoder"]) == 0.0
    assert _tree_norm(frozen_pure["state_projector"]) > 0.0

    encoder_before = _pure_trainable(frozen)["encoder"]
    projector_before = _pure_trainable(frozen)["state_projector"]
    optimizer.update(frozen, frozen_grad)
    encoder_after = _pure_trainable(frozen)["encoder"]
    projector_after = _pure_trainable(frozen)["state_projector"]
    for left, right in zip(
        jax.tree.leaves(encoder_before),
        jax.tree.leaves(encoder_after),
        strict=True,
    ):
        np.testing.assert_array_equal(left, right)
    assert any(
        not np.array_equal(left, right)
        for left, right in zip(
            jax.tree.leaves(projector_before),
            jax.tree.leaves(projector_after),
            strict=True,
        )
    )


def test_frozen_optimizer_drops_encoder_state_but_compatibility_abi_matches(
    monkeypatch,
) -> None:
    unfrozen, unfrozen_optimizer = _make_components(
        monkeypatch,
        frozen=False,
        seed=23,
    )
    frozen, frozen_optimizer = _make_components(
        monkeypatch,
        frozen=True,
        seed=23,
    )
    compatibility_optimizer = train.create_joint_optimizer(
        frozen,
        frozen.config,
        freeze_backbone=False,
    )
    unfrozen_abi = train.research_state_abi(
        nnx.state(unfrozen_optimizer.opt_state)
    )
    compatibility_abi = train.research_state_abi(
        nnx.state(compatibility_optimizer.opt_state)
    )
    frozen_abi = train.research_state_abi(
        nnx.state(frozen_optimizer.opt_state)
    )
    assert compatibility_abi == unfrozen_abi
    assert frozen_abi["nbytes"] < compatibility_abi["nbytes"]
    assert train.research_state_abi(_pure_trainable(unfrozen)) == (
        train.research_state_abi(_pure_trainable(frozen))
    )

    footprint = train.training_state_footprint(frozen, frozen_optimizer)
    partitions = footprint["model_parameter_count_by_optimizer_partition"]
    assert partitions["frozen_bt4"] == 16
    assert partitions["bt4"] == 0
    assert footprint["optimizer_frozen_parameter_count"] == 16
    assert footprint["optimizer_active_parameter_count"] == (
        partitions["main"]
    )


def test_freeze_resume_contract_is_explicit(monkeypatch) -> None:
    monkeypatch.setattr(train, "require_within_workspace", lambda path: Path(path))
    monkeypatch.setattr(
        train,
        "load_asset_manifest",
        lambda: {
            "trajectory_v3": {
                "archive": {"sha256": "trajectory-digest", "size_bytes": 123}
            }
        },
    )
    monkeypatch.setattr(train, "sha256_file", lambda _path: "test-digest")
    contract = train.build_research_resume_contract(
        config=_config(frozen=True),
        objective="normalized",
        sigreg_reference_count=1.0,
        batch_size=2,
        train_seed=7,
        train_provenance={"kind": "test"},
        models_dir=REPO_ROOT / "models",
    )
    freeze = contract["optimizer"]["bt4_backbone_freeze"]
    assert freeze["gradient_boundary"] == "stop_gradient_after_bt4_tokens"
    assert freeze["optimizer_state_for_bt4"] == "none"
    assert freeze["bt4_learning_rate"] == 0.0
    assert freeze["legacy_exact_optimizer_restore"] is False
    assert contract["model_config"]["bt4_freeze_backbone"] is True
