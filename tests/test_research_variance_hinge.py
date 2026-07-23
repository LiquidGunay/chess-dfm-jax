from __future__ import annotations

import dataclasses
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

import research.train as train
from chess_dfm_jax.nnx_bt4 import TrainableParam


def test_per_horizon_variance_hinge_matches_population_formula() -> None:
    target = jnp.asarray(
        [
            [[0.0, 0.0], [1.0, 2.0]],
            [[2.0, 4.0], [3.0, 6.0]],
            [[4.0, 8.0], [jnp.nan, jnp.nan]],
        ],
        dtype=jnp.float32,
    )
    valid = jnp.asarray(
        [[1.0, 1.0], [1.0, 1.0], [1.0, 0.0]],
        dtype=jnp.float32,
    )

    result = train.per_horizon_target_variance_hinge(
        target,
        valid,
        gamma=2.0,
    )

    expected_std = np.sqrt(
        np.asarray(
            [
                [8.0 / 3.0, 32.0 / 3.0],
                [1.0, 4.0],
            ],
            dtype=np.float32,
        )
        + train.TARGET_VARIANCE_HINGE_EPSILON
    )
    expected_hinge = np.maximum(2.0 - expected_std, 0.0).mean(
        axis=-1
    )
    np.testing.assert_allclose(
        result.valid_count_by_horizon,
        [3.0, 2.0],
    )
    np.testing.assert_array_equal(
        result.eligible_by_horizon,
        [1.0, 1.0],
    )
    np.testing.assert_allclose(
        result.feature_std_mean_by_horizon,
        expected_std.mean(axis=-1),
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        result.feature_std_p05_by_horizon,
        np.quantile(expected_std, 0.05, axis=-1),
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        result.feature_std_median_by_horizon,
        np.quantile(expected_std, 0.50, axis=-1),
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        result.active_fraction_by_horizon,
        (expected_std < 2.0).mean(axis=-1),
    )
    np.testing.assert_allclose(
        result.hinge_by_horizon,
        expected_hinge,
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        result.loss,
        expected_hinge.mean(),
        rtol=1e-6,
        atol=1e-6,
    )
    assert np.isfinite(np.asarray(result.loss))


def test_variance_hinge_masks_nan_padding_and_ineligible_horizons() -> None:
    target = jnp.asarray(
        [
            [[-1.0, 2.0], [10.0, -20.0]],
            [[1.0, 4.0], [jnp.nan, jnp.nan]],
            [[jnp.nan, jnp.nan], [jnp.nan, jnp.nan]],
        ],
        dtype=jnp.float32,
    )
    valid = jnp.asarray(
        [[1.0, 1.0], [1.0, 0.0], [0.0, 0.0]],
        dtype=jnp.float32,
    )
    result = train.per_horizon_target_variance_hinge(
        target,
        valid,
        gamma=3.0,
    )
    base = train.per_horizon_target_variance_hinge(
        target[:2, :1],
        valid[:2, :1],
        gamma=3.0,
    )

    np.testing.assert_allclose(result.loss, base.loss)
    np.testing.assert_array_equal(
        result.eligible_by_horizon,
        [1.0, 0.0],
    )
    np.testing.assert_allclose(
        result.feature_std_mean_by_horizon[0],
        base.feature_std_mean_by_horizon[0],
    )
    diagnostic_values = np.concatenate(
        [
            np.asarray(result.loss).reshape(-1),
            np.asarray(result.feature_std_mean_by_horizon).reshape(-1),
            np.asarray(result.hinge_by_horizon).reshape(-1),
        ]
    )
    assert np.all(np.isfinite(diagnostic_values))
    gradient = jax.grad(
        lambda value: train.per_horizon_target_variance_hinge(
            value,
            valid,
            gamma=3.0,
        ).loss
    )(target)
    assert np.all(np.isfinite(np.asarray(gradient)))
    np.testing.assert_array_equal(
        gradient[2],
        jnp.zeros_like(gradient[2]),
    )
    np.testing.assert_array_equal(
        gradient[:, 1],
        jnp.zeros_like(gradient[:, 1]),
    )


def test_variance_hinge_is_invariant_to_exact_batch_duplication() -> None:
    target = jnp.asarray(
        [
            [[-2.0, 0.0, 2.0], [1.0, 2.0, 3.0]],
            [[2.0, 4.0, 6.0], [3.0, 4.0, 5.0]],
            [[6.0, 8.0, 10.0], [9.0, 9.0, 9.0]],
        ],
        dtype=jnp.float32,
    )
    valid = jnp.asarray(
        [[1.0, 1.0], [1.0, 1.0], [1.0, 0.0]],
        dtype=jnp.float32,
    )
    base = train.per_horizon_target_variance_hinge(
        target,
        valid,
        gamma=3.0,
    )
    duplicated = train.per_horizon_target_variance_hinge(
        jnp.repeat(target, 2, axis=0),
        jnp.repeat(valid, 2, axis=0),
        gamma=3.0,
    )

    np.testing.assert_array_equal(duplicated.loss, base.loss)
    np.testing.assert_array_equal(
        duplicated.eligible_by_horizon,
        base.eligible_by_horizon,
    )
    np.testing.assert_array_equal(
        duplicated.feature_std_mean_by_horizon,
        base.feature_std_mean_by_horizon,
    )
    np.testing.assert_array_equal(
        duplicated.feature_std_p05_by_horizon,
        base.feature_std_p05_by_horizon,
    )
    np.testing.assert_array_equal(
        duplicated.feature_std_median_by_horizon,
        base.feature_std_median_by_horizon,
    )
    np.testing.assert_array_equal(
        duplicated.active_fraction_by_horizon,
        base.active_fraction_by_horizon,
    )
    np.testing.assert_array_equal(
        duplicated.hinge_by_horizon,
        base.hinge_by_horizon,
    )
    np.testing.assert_array_equal(
        duplicated.valid_count_by_horizon,
        2.0 * base.valid_count_by_horizon,
    )


def test_variance_hinge_gradient_is_attached_and_respects_masks() -> None:
    target = jnp.asarray(
        [
            [[-2.0, 0.0, 1.0], [1.0, 2.0, 3.0]],
            [[0.0, 2.0, 4.0], [4.0, 5.0, 6.0]],
            [[3.0, 5.0, 8.0], [7.0, 8.0, 9.0]],
        ],
        dtype=jnp.float32,
    )
    valid = jnp.asarray(
        [[1.0, 1.0], [1.0, 0.0], [1.0, 0.0]],
        dtype=jnp.float32,
    )

    gradient = jax.grad(
        lambda value: train.per_horizon_target_variance_hinge(
            value,
            valid,
            gamma=10.0,
        ).loss
    )(target)

    assert float(jnp.linalg.norm(gradient[:, 0])) > 0.0
    np.testing.assert_array_equal(
        gradient[:, 1],
        jnp.zeros_like(gradient[:, 1]),
    )


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


def tiny_config(**updates: Any) -> train.JointLatentSASAConfig:
    config = train.JointLatentSASAConfig(
        token_dim=4,
        z_dim=8,
        projector_layers=1,
        projector_num_heads=2,
        projector_mlp_dim=16,
        jepa_condition_dim=6,
        dfm_layers=1,
        jepa_layers=1,
        jepa_num_heads=2,
        jepa_mlp_dim=16,
        num_heads=2,
        mlp_dim=16,
        action_vocab_size=16,
        horizon=2,
        compute_dtype="float32",
        param_dtype="float32",
        encoder_dtype="float32",
        first_legality_coeff=1.0,
        legality_on_masked_only=False,
        jepa_sigreg_coeff=0.0,
        jepa_pred_sigreg_coeff=0.0,
        jepa_sigreg_kind="le_jepa",
        jepa_sigreg_proj_dim=4,
        jepa_target_variance_hinge_coeff=1.0,
        jepa_target_variance_hinge_gamma=10.0,
        use_qk_gain=True,
        use_qk_norm=True,
        use_xsa=True,
        use_muon=True,
        lr_warmup_steps=0,
        remat_blocks=False,
        scan_layers=False,
    )
    return dataclasses.replace(config, **updates)


def tiny_batch() -> dict[str, jax.Array]:
    current = jnp.asarray([0.0, 0.25, 0.75], dtype=jnp.float32)[
        :, None, None, None
    ] * jnp.ones((3, 112, 8, 8), dtype=jnp.float32)
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
        "future_valid": jnp.asarray(
            [[1.0, 1.0], [1.0, 1.0], [1.0, 0.0]],
            dtype=jnp.float32,
        ),
        "legal_idx": jnp.asarray(
            [
                [[1, 0], [2, 0]],
                [[3, 0], [4, 0]],
                [[5, 0], [6, 0]],
            ],
            dtype=jnp.int32,
        ),
        "legal_count": jnp.ones((3, 2), dtype=jnp.int32),
        "legal_masks_valid": jnp.ones((3, 2), dtype=jnp.float32),
        "deterministic_t": jnp.asarray(0.25, dtype=jnp.float32),
    }


def test_variance_hinge_reaches_backbone_and_state_projector() -> None:
    model = train.JointLatentSASAModel(
        ParameterizedEncoder(),
        tiny_config(),
        rngs=nnx.Rngs(7),
    )

    def hinge_only(candidate: train.JointLatentSASAModel) -> jax.Array:
        _, aux = train.normalized_stage1_loss_fn(
            candidate,
            tiny_batch(),
            jax.random.PRNGKey(11),
            1.0,
            1.0,
        )
        return aux["jepa_target_variance_hinge_loss"]

    value, gradients = nnx.value_and_grad(
        hinge_only,
        argnums=nnx.DiffState(0, TrainableParam),
    )(model)
    pure_gradients = nnx.to_pure_dict(gradients)

    assert float(value) > 0.0
    for group in ("encoder", "state_projector"):
        gradient_norm = sum(
            float(
                jnp.sum(
                    jnp.square(
                        jnp.asarray(leaf, dtype=jnp.float32)
                    )
                )
            )
            for leaf in jax.tree.leaves(pure_gradients[group])
        )
        assert gradient_norm > 0.0


def test_hinge_enabled_metrics_and_gradient_audit_abi() -> None:
    enabled_model = train.JointLatentSASAModel(
        ParameterizedEncoder(),
        tiny_config(jepa_sigreg_coeff=5.76),
        rngs=nnx.Rngs(13),
    )
    _, aux = train.normalized_stage1_loss_fn(
        enabled_model,
        tiny_batch(),
        jax.random.PRNGKey(17),
        1.0,
        1.0,
    )
    expected_metrics = {
        "jepa_target_variance_hinge_loss",
        "jepa_target_variance_valid_count_by_horizon",
        "jepa_target_variance_eligible_by_horizon",
        "jepa_target_feature_std_mean_by_horizon",
        "jepa_target_feature_std_p05_by_horizon",
        "jepa_target_feature_std_median_by_horizon",
        "jepa_target_variance_active_fraction_by_horizon",
        "jepa_target_variance_hinge_by_horizon",
    }
    assert expected_metrics <= set(aux)
    assert train.gradient_component_names(enabled_model.config) == (
        train.GRADIENT_COMPONENT_NAMES
        + (train.TARGET_VARIANCE_HINGE_COMPONENT,)
    )
    components = train.gradient_component_vector(
        enabled_model,
        tiny_batch(),
        jax.random.PRNGKey(17),
        1.0,
    )
    assert components.shape == (6,)
    np.testing.assert_allclose(
        components[-1],
        aux["jepa_target_variance_hinge_loss"],
    )

    disabled_model = train.JointLatentSASAModel(
        ParameterizedEncoder(),
        tiny_config(
            jepa_target_variance_hinge_coeff=0.0,
            jepa_target_variance_hinge_gamma=0.9,
        ),
        rngs=nnx.Rngs(13),
    )
    _, disabled_aux = train.normalized_stage1_loss_fn(
        disabled_model,
        tiny_batch(),
        jax.random.PRNGKey(17),
        1.0,
        1.0,
    )
    assert expected_metrics.isdisjoint(disabled_aux)
    assert (
        train.gradient_component_names(disabled_model.config)
        == train.GRADIENT_COMPONENT_NAMES
    )


@pytest.mark.parametrize(
    ("updates", "objective", "message"),
    [
        (
            {"jepa_target_variance_hinge_coeff": -1.0},
            "normalized",
            "finite and non-negative",
        ),
        (
            {"jepa_target_variance_hinge_coeff": float("inf")},
            "normalized",
            "finite and non-negative",
        ),
        (
            {
                "jepa_target_variance_hinge_coeff": 1.0,
                "jepa_target_variance_hinge_gamma": 0.0,
            },
            "normalized",
            "finite and positive",
        ),
        (
            {
                "jepa_target_variance_hinge_coeff": 0.0,
                "jepa_target_variance_hinge_gamma": 0.8,
            },
            "normalized",
            "inert unless",
        ),
        (
            {"jepa_target_variance_hinge_coeff": 1.0},
            "legacy",
            "requires --objective normalized",
        ),
    ],
)
def test_invalid_hinge_configs_fail_closed(
    updates: dict[str, float],
    objective: str,
    message: str,
) -> None:
    config = dataclasses.replace(
        train.JointLatentSASAConfig(),
        **updates,
    )
    with pytest.raises(ValueError, match=message):
        train.validate_objective_config(
            objective=objective,
            config=config,
        )


def test_hinge_cli_contract_and_disabled_serialization() -> None:
    default_config = train.JointLatentSASAConfig()
    assert train.serialized_model_config(default_config) == {
        key: value
        for key, value in dataclasses.asdict(default_config).items()
        if key
        not in {
            "jepa_target_variance_hinge_coeff",
            "jepa_target_variance_hinge_gamma",
            "dfm_active_layers",
            "jepa_projector_active_layers",
            "jepa_sampled_target_anchors",
            "jepa_target_sampling_unit",
            "jepa_rollout_mode",
            "jepa_state_fixed_unit_rms",
            "dfm_first_action_loss_share",
            "dfm_force_first_action_mask",
            "dfm_training_time_power",
            "bt4_freeze_backbone",
            "bt4_future_target_stop_gradient",
            "bt4_future_target_trainable_tail_layers",
        }
    }
    args = train.parse_args(
        [
            "--target-variance-hinge-coeff",
            "0.3",
            "--target-variance-hinge-gamma",
            "0.95",
        ]
    )
    enabled = train.apply_config_overrides(default_config, args)
    assert enabled.jepa_target_variance_hinge_coeff == 0.3
    assert enabled.jepa_target_variance_hinge_gamma == 0.95
    serialized = train.serialized_model_config(enabled)
    assert serialized["jepa_target_variance_hinge_coeff"] == 0.3
    assert serialized["jepa_target_variance_hinge_gamma"] == 0.95
