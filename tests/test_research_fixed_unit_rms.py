from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

import chess_dfm_jax.training.joint_latent_sasa as legacy
import research.train as train
from chess_dfm_jax.nnx_bt4 import TrainableParam
from research.prepare import REPO_ROOT


class ParameterizedEncoder(nnx.Module):
    def __init__(self, embedding_size: int = 8):
        self.embedding_size = embedding_size
        self.feature_scale = TrainableParam(
            jnp.linspace(
                0.5,
                1.5,
                embedding_size,
                dtype=jnp.float32,
            )
        )

    def encode_tokens(self, planes: jax.Array) -> jax.Array:
        batch_size = planes.shape[0]
        plane_mean = jnp.mean(planes, axis=(1, 2, 3))
        square = jnp.linspace(
            -0.5,
            0.5,
            64,
            dtype=jnp.float32,
        )[None, :, None]
        feature = jnp.linspace(
            0.1,
            1.0,
            self.embedding_size,
            dtype=jnp.float32,
        )[None, None, :]
        return (
            square
            + plane_mean.reshape((batch_size, 1, 1))
            * feature
            * self.feature_scale[...][None, None, :]
        )


def config_kwargs() -> dict[str, Any]:
    return {
        "token_dim": 4,
        "z_dim": 8,
        "projector_layers": 1,
        "projector_num_heads": 2,
        "projector_mlp_dim": 16,
        "jepa_condition_dim": 6,
        "dfm_layers": 1,
        "jepa_layers": 1,
        "jepa_num_heads": 2,
        "jepa_mlp_dim": 16,
        "num_heads": 2,
        "mlp_dim": 16,
        "action_vocab_size": 16,
        "horizon": 2,
        "compute_dtype": "float32",
        "param_dtype": "float32",
        "encoder_dtype": "float32",
        "first_legality_coeff": 1.0,
        "legality_on_masked_only": False,
        "jepa_sigreg_coeff": 0.0,
        "jepa_pred_sigreg_coeff": 0.0,
        "jepa_sigreg_kind": "le_jepa",
        "jepa_sigreg_proj_dim": 4,
        "use_qk_gain": True,
        "use_qk_norm": True,
        "use_xsa": True,
        "use_muon": True,
        "lr_warmup_steps": 0,
        "remat_blocks": False,
        "scan_layers": False,
    }


def local_config(
    **updates: Any,
) -> train.JointLatentSASAConfig:
    config = train.JointLatentSASAConfig(**config_kwargs())
    return dataclasses.replace(config, **updates)


def batch() -> dict[str, jax.Array]:
    current = jnp.asarray(
        [0.0, 0.25, 0.75],
        dtype=jnp.float32,
    )[:, None, None, None] * jnp.ones(
        (3, 112, 8, 8),
        dtype=jnp.float32,
    )
    future = jnp.stack(
        [current + 0.1, current - 0.2],
        axis=1,
    )
    return {
        "current_planes": current,
        "future_planes": future,
        "action_indices": jnp.asarray(
            [[1, 2], [3, 4], [5, 6]],
            dtype=jnp.int32,
        ),
        "valid": jnp.ones((3,), dtype=jnp.float32),
        "future_valid": jnp.ones((3, 2), dtype=jnp.float32),
        "legal_idx": jnp.asarray(
            [
                [[1, 0], [2, 0]],
                [[3, 0], [4, 0]],
                [[5, 0], [6, 0]],
            ],
            dtype=jnp.int32,
        ),
        "legal_count": jnp.ones((3, 2), dtype=jnp.int32),
        "legal_masks_valid": jnp.ones(
            (3, 2),
            dtype=jnp.float32,
        ),
        "deterministic_t": jnp.asarray(
            0.25,
            dtype=jnp.float32,
        ),
    }


def assert_tree_exact(left: Any, right: Any) -> None:
    assert jax.tree.structure(left) == jax.tree.structure(right)
    for left_leaf, right_leaf in zip(
        jax.tree.leaves(left),
        jax.tree.leaves(right),
        strict=True,
    ):
        np.testing.assert_array_equal(left_leaf, right_leaf)


def assert_unit_rms(value: jax.Array, *, atol: float) -> None:
    rms = jnp.sqrt(
        jnp.mean(
            jnp.square(jnp.asarray(value, dtype=jnp.float32)),
            axis=-1,
        )
    )
    np.testing.assert_allclose(
        rms,
        jnp.ones_like(rms),
        rtol=0.0,
        atol=atol,
    )


def test_disabled_loss_and_aux_match_legacy_exactly() -> None:
    local_model = train.JointLatentSASAModel(
        ParameterizedEncoder(),
        train.JointLatentSASAConfig(**config_kwargs()),
        rngs=nnx.Rngs(3),
    )
    legacy_model = legacy.JointLatentSASAModel(
        ParameterizedEncoder(),
        legacy.JointLatentSASAConfig(**config_kwargs()),
        rngs=nnx.Rngs(3),
    )
    rng = jax.random.PRNGKey(5)

    local_value = train.joint_stage1_loss_fn(
        local_model,
        batch(),
        rng,
        compute_fp32_legality=True,
    )
    legacy_value = legacy.joint_stage1_loss_fn(
        legacy_model,
        batch(),
        rng,
        compute_fp32_legality=True,
    )

    assert_tree_exact(local_value, legacy_value)
    assert "jepa_state_fixed_unit_rms" not in local_value[1]
    assert "jepa_state_trainable_scale_used" not in local_value[1]


def test_disabled_trainable_rmsnorm_behavior_is_unchanged() -> None:
    model = train.JointLatentSASAModel(
        ParameterizedEncoder(),
        local_config(jepa_state_rmsnorm=True),
        rngs=nnx.Rngs(7),
    )
    z = jnp.asarray(
        [[-2.0, -1.0, 0.5, 3.0, 4.0, 0.25, -0.75, 2.5]],
        dtype=jnp.float32,
    )
    expected = (
        z
        * jax.lax.rsqrt(
            jnp.mean(jnp.square(z), axis=-1, keepdims=True)
            + train.JEPA_STATE_RMS_EPSILON
        )
        * jnp.asarray(
            model.jepa_state_norm.scale[...],
            dtype=jnp.float32,
        )
    )

    np.testing.assert_array_equal(
        model.normalize_jepa_state(z),
        expected,
    )


def test_fixed_unit_rms_covers_current_future_and_rollout_bfloat16() -> None:
    model = train.JointLatentSASAModel(
        ParameterizedEncoder(),
        local_config(
            compute_dtype="bfloat16",
            jepa_state_fixed_unit_rms=True,
        ),
        rngs=nnx.Rngs(11),
    )
    sample = batch()
    current_z, future_z = model.encode_current_and_future_targets(
        sample["current_planes"],
        sample["future_planes"],
    )
    assert current_z.dtype == jnp.bfloat16
    assert future_z.dtype == jnp.bfloat16
    assert_unit_rms(current_z, atol=0.01)
    assert_unit_rms(future_z, atol=0.01)

    action_hidden = jax.random.normal(
        jax.random.PRNGKey(13),
        (
            sample["action_indices"].shape[0],
            sample["action_indices"].shape[1],
            model.config.token_dim,
        ),
        dtype=jnp.float32,
    )
    predictions = model.jepa_rollout_from_latents(
        current_z,
        sample["action_indices"],
        action_hidden,
        z0_normalized=True,
    )
    assert predictions.dtype == jnp.bfloat16
    assert_unit_rms(predictions, atol=0.01)


def test_fixed_unit_rms_is_scale_invariant_with_attached_gradient() -> None:
    model = train.JointLatentSASAModel(
        ParameterizedEncoder(),
        local_config(jepa_state_fixed_unit_rms=True),
        rngs=nnx.Rngs(17),
    )
    z = jnp.asarray(
        [
            [-2.0, -1.0, 0.5, 3.0, 4.0, 0.25, -0.75, 2.5],
            [1.0, 2.0, -3.0, 0.5, 1.5, -2.5, 4.0, -1.0],
        ],
        dtype=jnp.float32,
    )
    normalized = model.normalize_jepa_state(z)
    scaled = model.normalize_jepa_state(8.0 * z)
    np.testing.assert_allclose(
        scaled,
        normalized,
        rtol=1e-6,
        atol=2e-6,
    )
    assert_unit_rms(normalized, atol=1e-6)

    probe = jnp.linspace(
        -1.0,
        1.0,
        z.size,
        dtype=jnp.float32,
    ).reshape(z.shape)
    gradient = jax.grad(
        lambda value: jnp.sum(
            model.normalize_jepa_state(value) * probe
        )
    )(z)
    assert np.all(np.isfinite(np.asarray(gradient)))
    assert float(jnp.linalg.norm(gradient)) > 0.0
    radial_gradient = jnp.sum(gradient * z, axis=-1)
    np.testing.assert_allclose(
        radial_gradient,
        jnp.zeros_like(radial_gradient),
        rtol=0.0,
        atol=2e-6,
    )


def test_fixed_unit_rms_bypasses_scale_and_preserves_model_gradients() -> None:
    model = train.JointLatentSASAModel(
        ParameterizedEncoder(),
        local_config(jepa_state_fixed_unit_rms=True),
        rngs=nnx.Rngs(19),
    )

    def loss_fn(candidate: train.JointLatentSASAModel):
        return train.normalized_stage1_loss_fn(
            candidate,
            batch(),
            jax.random.PRNGKey(23),
            1.0,
            1.0,
        )

    (loss, aux), gradients = nnx.value_and_grad(
        loss_fn,
        argnums=nnx.DiffState(0, TrainableParam),
        has_aux=True,
    )(model)
    pure_gradients = nnx.to_pure_dict(gradients)

    assert jnp.isfinite(loss)
    assert float(aux["jepa_state_fixed_unit_rms"]) == 1.0
    assert float(aux["jepa_state_trainable_scale_used"]) == 0.0
    np.testing.assert_array_equal(
        pure_gradients["jepa_state_norm"]["scale"],
        jnp.zeros_like(
            pure_gradients["jepa_state_norm"]["scale"]
        ),
    )
    for group in ("encoder", "state_projector", "jepa_transition"):
        squared_norm = sum(
            float(
                jnp.sum(
                    jnp.square(
                        jnp.asarray(leaf, dtype=jnp.float32)
                    )
                )
            )
            for leaf in jax.tree.leaves(pure_gradients[group])
        )
        assert squared_norm > 0.0


@pytest.mark.parametrize(
    ("updates", "objective", "message"),
    [
        ({}, "legacy", "requires --objective normalized"),
        (
            {"jepa_state_rmsnorm": True},
            "normalized",
            "legacy trainable",
        ),
        (
            {"jepa_target_semantics": "ema"},
            "normalized",
            "cannot be combined with EMA",
        ),
        (
            {"jepa_target_stop_gradient": True},
            "normalized",
            "cannot be combined with target stop-gradient",
        ),
        (
            {"jepa_target_variance_hinge_coeff": 1.0},
            "normalized",
            "cannot be combined with the target variance hinge",
        ),
        (
            {"jepa_state_rms_scale_max": 3.0},
            "normalized",
            "is inert in fixed-unit-RMS",
        ),
    ],
)
def test_fixed_unit_rms_invalid_combinations_fail_closed(
    updates: dict[str, Any],
    objective: str,
    message: str,
) -> None:
    config = dataclasses.replace(
        train.JointLatentSASAConfig(
            jepa_state_fixed_unit_rms=True
        ),
        **updates,
    )
    with pytest.raises(ValueError, match=message):
        train.validate_objective_config(
            objective=objective,
            config=config,
        )


def test_fixed_unit_rms_cli_serialization_and_contract(
    monkeypatch,
) -> None:
    default_config = train.JointLatentSASAConfig()
    assert (
        "jepa_state_fixed_unit_rms"
        not in train.serialized_model_config(default_config)
    )
    args = train.parse_args(["--jepa-state-fixed-unit-rms"])
    enabled = train.apply_config_overrides(default_config, args)
    assert enabled.jepa_state_fixed_unit_rms is True
    train.validate_objective_config(
        objective="normalized",
        config=enabled,
    )
    assert train.serialized_model_config(enabled)[
        "jepa_state_fixed_unit_rms"
    ] is True
    disabled_args = train.parse_args(
        ["--no-jepa-state-fixed-unit-rms"]
    )
    assert (
        train.apply_config_overrides(enabled, disabled_args)
        .jepa_state_fixed_unit_rms
        is False
    )

    monkeypatch.setattr(
        train,
        "require_within_workspace",
        lambda path: Path(path),
    )
    monkeypatch.setattr(
        train,
        "load_asset_manifest",
        lambda: {
            "trajectory_v3": {
                "archive": {
                    "sha256": "trajectory-digest",
                    "size_bytes": 123,
                }
            }
        },
    )
    monkeypatch.setattr(
        train,
        "sha256_file",
        lambda _path: "file-digest",
    )
    contract = train.build_research_resume_contract(
        config=enabled,
        objective="normalized",
        sigreg_reference_count=1.0,
        batch_size=64,
        train_seed=3,
        train_provenance={"kind": "test"},
        models_dir=REPO_ROOT / "models",
    )
    semantics = contract["objective"]["jepa_state_manifold"]
    assert semantics["kind"] == "fixed_unit_rms"
    assert semantics["statistics_dtype"] == "float32"
    assert semantics["epsilon"] == train.JEPA_STATE_RMS_EPSILON
    assert semantics["stored_scale_parameter_read_by_objective"] is False
    assert semantics["stored_scale_parameter_loss_gradient"] == "exact_zero"
    assert contract["model_config"]["jepa_state_fixed_unit_rms"] is True
