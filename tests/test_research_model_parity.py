from __future__ import annotations

import dataclasses
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

import chess_dfm_jax.training.joint_latent_sasa as legacy  # noqa: E402
import research.train as local  # noqa: E402
from chess_dfm_jax.nnx_bt4 import TrainableParam  # noqa: E402


class DummyEncoder(nnx.Module):
    def __init__(self, embedding_size: int = 16):
        self.embedding_size = embedding_size

    def encode_tokens(self, planes: jax.Array) -> jax.Array:
        batch_size = planes.shape[0]
        plane_mean = jnp.mean(planes, axis=(1, 2, 3))
        square = jnp.linspace(0.0, 1.0, 64, dtype=jnp.float32)[None, :, None]
        feature = jnp.linspace(
            0.1,
            1.0,
            self.embedding_size,
            dtype=jnp.float32,
        )[None, None, :]
        return square + plane_mean.reshape((batch_size, 1, 1)) * feature


def _config_kwargs() -> dict[str, Any]:
    return {
        "token_dim": 8,
        "z_dim": 16,
        "projector_layers": 1,
        "projector_num_heads": 4,
        "projector_mlp_dim": 32,
        "jepa_condition_dim": 12,
        "dfm_layers": 1,
        "jepa_layers": 1,
        "jepa_num_heads": 4,
        "jepa_mlp_dim": 32,
        "num_heads": 4,
        "mlp_dim": 32,
        "action_vocab_size": 32,
        "horizon": 2,
        "compute_dtype": "float32",
        "param_dtype": "float32",
        "encoder_dtype": "float32",
        "first_legality_coeff": 1.0,
        "legality_on_masked_only": False,
        "jepa_sigreg_coeff": 0.01,
        "jepa_pred_sigreg_coeff": 0.02,
        "jepa_sigreg_kind": "le_jepa",
        "jepa_sigreg_proj_dim": 8,
        "use_qk_gain": True,
        "use_qk_norm": True,
        "use_xsa": True,
        "use_muon": True,
        "lr_warmup_steps": 3,
        "remat_blocks": False,
        "scan_layers": False,
    }


def _pure_trainable(model: nnx.Module) -> dict[str, Any]:
    return nnx.to_pure_dict(nnx.state(model, TrainableParam))


def _pure_optimizer(optimizer: nnx.Optimizer) -> dict[str, Any]:
    return nnx.to_pure_dict(nnx.state(optimizer.opt_state))


def _assert_trees_exact(left: Any, right: Any) -> None:
    assert jax.tree_util.tree_structure(left) == jax.tree_util.tree_structure(right)
    left_leaves = jax.tree.leaves(left)
    right_leaves = jax.tree.leaves(right)
    assert len(left_leaves) == len(right_leaves)
    for left_leaf, right_leaf in zip(left_leaves, right_leaves, strict=True):
        np.testing.assert_array_equal(
            np.asarray(left_leaf),
            np.asarray(right_leaf),
        )


def _batch() -> dict[str, jax.Array]:
    current = jnp.linspace(
        -0.25,
        0.75,
        2 * 112 * 8 * 8,
        dtype=jnp.float32,
    ).reshape((2, 112, 8, 8))
    future = jnp.stack(
        [
            current + jnp.asarray(0.125, dtype=jnp.float32),
            current - jnp.asarray(0.25, dtype=jnp.float32),
        ],
        axis=1,
    )
    return {
        "current_planes": current,
        "future_planes": future,
        "action_indices": jnp.asarray([[1, 2], [3, 4]], dtype=jnp.int32),
        "valid": jnp.ones((2,), dtype=jnp.float32),
        "future_valid": jnp.asarray([[1.0, 1.0], [1.0, 0.0]], dtype=jnp.float32),
        "legal_idx": jnp.asarray(
            [
                [[1, 5, 9, 0], [2, 6, 0, 0]],
                [[3, 7, 11, 15], [4, 8, 12, 0]],
            ],
            dtype=jnp.int32,
        ),
        "legal_count": jnp.asarray([[3, 2], [4, 3]], dtype=jnp.int32),
        "legal_masks_valid": jnp.ones((2, 2), dtype=jnp.float32),
        "deterministic_t": jnp.asarray(0.25, dtype=jnp.float32),
    }


def _assert_loss_and_gradient_parity(
    *,
    config_kwargs: dict[str, Any],
    batch: dict[str, jax.Array],
    model_seed: int,
    loss_seed: int,
) -> tuple[tuple[Any, Any], Any]:
    local_model = local.JointLatentSASAModel(
        DummyEncoder(),
        local.JointLatentSASAConfig(**config_kwargs),
        rngs=nnx.Rngs(model_seed),
    )
    legacy_model = legacy.JointLatentSASAModel(
        DummyEncoder(),
        legacy.JointLatentSASAConfig(**config_kwargs),
        rngs=nnx.Rngs(model_seed),
    )
    local_loss_and_grad = nnx.value_and_grad(
        local.joint_stage1_loss_fn,
        argnums=nnx.DiffState(0, TrainableParam),
        has_aux=True,
    )
    legacy_loss_and_grad = nnx.value_and_grad(
        legacy.joint_stage1_loss_fn,
        argnums=nnx.DiffState(0, TrainableParam),
        has_aux=True,
    )
    rng = jax.random.PRNGKey(loss_seed)
    local_value, local_gradients = local_loss_and_grad(
        local_model,
        batch,
        rng,
        compute_fp32_legality=True,
    )
    legacy_value, legacy_gradients = legacy_loss_and_grad(
        legacy_model,
        batch,
        rng,
        compute_fp32_legality=True,
    )
    _assert_trees_exact(local_value, legacy_value)
    _assert_trees_exact(
        nnx.to_pure_dict(local_gradients),
        nnx.to_pure_dict(legacy_gradients),
    )
    return local_value, local_gradients


def test_normalized_objective_requires_le_jepa_sigreg():
    le_jepa = local.JointLatentSASAConfig(
        **(_config_kwargs() | {"jepa_sigreg_kind": "le_jepa"})
    )
    moments = local.JointLatentSASAConfig(
        **(_config_kwargs() | {"jepa_sigreg_kind": "moments"})
    )

    local.validate_objective_config(
        objective="normalized",
        config=le_jepa,
    )
    local.validate_objective_config(
        objective="legacy",
        config=moments,
    )
    with pytest.raises(
        ValueError,
        match="normalized requires jepa_sigreg_kind='le_jepa'",
    ):
        local.validate_objective_config(
            objective="normalized",
            config=moments,
        )


@pytest.mark.parametrize("coefficient", [-1.0, float("nan"), float("inf")])
def test_jepa_norm_loss_coefficient_must_be_finite_and_non_negative(
    coefficient,
):
    config = local.JointLatentSASAConfig(
        **(_config_kwargs() | {"jepa_norm_loss_coeff": coefficient})
    )

    with pytest.raises(
        ValueError,
        match="jepa_norm_loss_coeff must be finite and non-negative",
    ):
        local.validate_objective_config(
            objective="normalized",
            config=config,
        )


def test_legacy_objective_requires_compatibility_norm_loss():
    config = local.JointLatentSASAConfig(
        **(_config_kwargs() | {"jepa_norm_loss_coeff": 0.0})
    )

    with pytest.raises(
        ValueError,
        match="legacy requires jepa_norm_loss_coeff=1.0",
    ):
        local.validate_objective_config(
            objective="legacy",
            config=config,
        )
    local.validate_objective_config(
        objective="normalized",
        config=config,
    )


@pytest.mark.parametrize(
    ("updates", "objective", "message"),
    [
        (
            {"jepa_sigreg_estimator": "not-an-estimator"},
            "normalized",
            "must be 'v_stat' or 'u_stat'",
        ),
        (
            {"jepa_sigreg_estimator": "u_stat"},
            "legacy",
            "requires --objective normalized",
        ),
        (
            {
                "jepa_sigreg_estimator": "u_stat",
                "jepa_sigreg_kind": "moments",
            },
            "normalized",
            "requires jepa_sigreg_kind='le_jepa'",
        ),
        (
            {"jepa_sigreg_example_count": -1},
            "normalized",
            "must be a non-negative integer",
        ),
        (
            {"jepa_sigreg_example_count": 1.5},
            "normalized",
            "must be a non-negative integer",
        ),
        (
            {"jepa_sigreg_example_count": 1},
            "legacy",
            "requires --objective normalized",
        ),
    ],
)
def test_invalid_sigreg_estimator_configs_fail_closed(
    updates,
    objective,
    message,
):
    config = local.JointLatentSASAConfig(
        **(_config_kwargs() | updates)
    )

    with pytest.raises(ValueError, match=message):
        local.validate_objective_config(
            objective=objective,
            config=config,
        )


def test_experiment_then_cli_override_precedence_and_resume_contract(
    monkeypatch,
):
    metadata_config = local.JointLatentSASAConfig(
        **(
            _config_kwargs()
            | {
                "jepa_target_stop_gradient": False,
            }
        )
    )
    monkeypatch.setattr(
        local,
        "EXPERIMENT_OVERRIDES",
        {
            "jepa_target_stop_gradient": True,
            "jepa_sigreg_coeff": 0.25,
        },
    )
    experiment_config = local.apply_experiment_overrides(
        metadata_config
    )
    assert experiment_config.jepa_target_stop_gradient is True
    assert experiment_config.jepa_sigreg_coeff == 0.25

    default_args = local.parse_args([])
    assert default_args.jepa_target_stop_gradient is None
    assert default_args.jepa_norm_loss_coeff is None
    assert default_args.sigreg_estimator is None
    assert default_args.sigreg_example_count is None
    assert default_args.learning_rate is None
    assert default_args.bt4_learning_rate is None
    assert default_args.train_batch_schedule == "shard_major"
    assert default_args.eval_batch_size is None
    default_config = local.apply_config_overrides(
        experiment_config,
        default_args,
    )
    assert default_config.jepa_target_stop_gradient is True
    assert default_config.jepa_sigreg_coeff == 0.25

    disabled_args = local.parse_args(
        [
            "--no-jepa-target-stop-gradient",
            "--target-sigreg-coeff",
            "0.5",
            "--jepa-norm-loss-coeff",
            "0",
            "--sigreg-estimator",
            "u_stat",
            "--sigreg-example-count",
            "1",
            "--learning-rate",
            "2.5e-5",
            "--bt4-learning-rate",
            "1e-6",
            "--train-batch-schedule",
            "global_permutation",
            "--eval-batch-size",
            "64",
        ]
    )
    disabled_config = local.apply_config_overrides(
        experiment_config,
        disabled_args,
    )
    assert disabled_config.jepa_target_stop_gradient is False
    assert disabled_config.jepa_sigreg_coeff == 0.5
    assert disabled_config.jepa_norm_loss_coeff == 0.0
    assert disabled_config.jepa_sigreg_estimator == "u_stat"
    assert disabled_config.jepa_sigreg_example_count == 1
    assert disabled_config.learning_rate == 2.5e-5
    assert disabled_config.bt4_learning_rate == 1e-6
    assert disabled_args.train_batch_schedule == "global_permutation"
    assert disabled_args.eval_batch_size == 64

    monkeypatch.setattr(
        local,
        "require_within_workspace",
        lambda path: Path(path),
    )
    monkeypatch.setattr(
        local,
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
        local,
        "sha256_file",
        lambda _path: "test-digest",
    )
    train_provenance = {
        "kind": "test",
        "batch_schedule": "shard_major",
    }
    contract = local.build_research_resume_contract(
        config=default_config,
        objective="legacy",
        sigreg_reference_count=1.0,
        batch_size=2,
        train_seed=7,
        train_provenance=train_provenance,
        models_dir=REPO_ROOT / "models",
    )
    assert (
        contract["model_config"]["jepa_target_stop_gradient"]
        is True
    )
    assert contract["model_config"]["jepa_sigreg_coeff"] == 0.25
    assert contract["model_config"]["jepa_norm_loss_coeff"] == 1.0
    assert contract["model_config"]["jepa_sigreg_estimator"] == "v_stat"
    assert contract["model_config"]["jepa_sigreg_example_count"] == 0
    assert (
        contract["objective"]["jepa_target_stop_gradient"]
        is True
    )
    assert contract["objective"]["target_sigreg_coeff"] == 0.25
    assert contract["objective"]["jepa_norm_loss_coeff"] == 1.0
    assert contract["objective"]["jepa_sigreg_estimator"] == "v_stat"
    assert contract["objective"]["jepa_sigreg_example_count"] == 0
    assert (
        contract["optimizer"]["learning_rate_schedule"]["family"]
        == "linear_warmup_then_constant"
    )
    assert (
        contract["optimizer"]["learning_rate_schedule"]["warmup_steps"]
        == 3
    )
    assert contract["data"]["schedule"] == train_provenance


def test_experiment_overrides_reject_unknown_keys_and_wrong_types():
    config = local.JointLatentSASAConfig()

    with pytest.raises(
        ValueError,
        match="Unknown EXPERIMENT_OVERRIDES.*not_a_config_field",
    ):
        local.apply_experiment_overrides(
            config,
            {"not_a_config_field": 1},
        )
    with pytest.raises(
        TypeError,
        match=r"EXPERIMENT_OVERRIDES\['horizon'\] must be int",
    ):
        local.apply_experiment_overrides(
            config,
            {"horizon": "8"},
        )

    numeric = local.apply_experiment_overrides(
        config,
        {"learning_rate": 1},
    )
    assert numeric.learning_rate == 1.0
    assert type(numeric.learning_rate) is float


def test_inert_nondefault_experiment_knobs_fail_closed():
    config = local.JointLatentSASAConfig()
    local.validate_no_inert_config_overrides(config)
    non_defaults = {
        "jepa_num_heads": 4,
        "horizon_legality_coeff": 0.25,
        "jepa_action_contrast_coeff": 0.1,
        "jepa_action_contrast_margin": 0.1,
        "contrastive_coeff": 0.1,
        "contrastive_temperature": 0.2,
        "candidate_count": 2,
        "jepa_teacher_forcing_steps": 1,
    }

    for name, value in non_defaults.items():
        candidate = local.apply_experiment_overrides(
            config,
            {name: value},
        )
        with pytest.raises(ValueError, match=name):
            local.validate_no_inert_config_overrides(candidate)


def test_local_config_and_initialized_model_match_legacy_exactly():
    local_fields = [
        (field.name, field.default)
        for field in dataclasses.fields(local.JointLatentSASAConfig)
        if field.name
        not in {
            "jepa_target_stop_gradient",
            "jepa_target_semantics",
            "jepa_target_ema_decay",
            "jepa_norm_loss_coeff",
            "jepa_sigreg_estimator",
            "jepa_sigreg_example_count",
            "jepa_target_variance_hinge_coeff",
            "jepa_target_variance_hinge_gamma",
            "lr_decay_start_steps",
            "lr_decay_steps",
            "lr_min_ratio",
            "dfm_first_action_loss_share",
            "dfm_active_layers",
            "jepa_projector_active_layers",
            "jepa_sampled_target_anchors",
            "jepa_target_sampling_unit",
            "jepa_state_fixed_unit_rms",
            "bt4_freeze_backbone",
            "bt4_future_target_stop_gradient",
            "bt4_future_target_trainable_tail_layers",
        }
    ]
    legacy_fields = [
        (field.name, field.default)
        for field in dataclasses.fields(legacy.JointLatentSASAConfig)
    ]
    assert local_fields == legacy_fields
    assert (
        local.JointLatentSASAConfig.__dataclass_fields__[
            "dfm_first_action_loss_share"
        ].default
        == 0.0
    )
    assert (
        local.JointLatentSASAConfig.__dataclass_fields__[
            "bt4_freeze_backbone"
        ].default
        is False
    )
    assert (
        local.JointLatentSASAConfig.__dataclass_fields__[
            "bt4_future_target_stop_gradient"
        ].default
        is False
    )
    assert (
        local.JointLatentSASAConfig.__dataclass_fields__[
            "bt4_future_target_trainable_tail_layers"
        ].default
        == 0
    )
    assert (
        local.JointLatentSASAConfig.__dataclass_fields__[
            "jepa_target_stop_gradient"
        ].default
        is False
    )
    assert (
        local.JointLatentSASAConfig.__dataclass_fields__[
            "jepa_target_semantics"
        ].default
        == "online"
    )
    assert (
        local.JointLatentSASAConfig.__dataclass_fields__[
            "jepa_target_ema_decay"
        ].default
        == 0.99
    )
    assert (
        local.JointLatentSASAConfig.__dataclass_fields__[
            "jepa_norm_loss_coeff"
        ].default
        == 1.0
    )
    assert (
        local.JointLatentSASAConfig.__dataclass_fields__[
            "jepa_sigreg_estimator"
        ].default
        == "v_stat"
    )
    assert (
        local.JointLatentSASAConfig.__dataclass_fields__[
            "jepa_sigreg_example_count"
        ].default
        == 0
    )

    kwargs = _config_kwargs()
    local_model = local.JointLatentSASAModel(
        DummyEncoder(),
        local.JointLatentSASAConfig(**kwargs),
        rngs=nnx.Rngs(17),
    )
    legacy_model = legacy.JointLatentSASAModel(
        DummyEncoder(),
        legacy.JointLatentSASAConfig(**kwargs),
        rngs=nnx.Rngs(17),
    )

    local_state = _pure_trainable(local_model)
    legacy_state = _pure_trainable(legacy_model)
    assert local.research_state_abi(local_state) == local.research_state_abi(
        legacy_state
    )
    _assert_trees_exact(local_state, legacy_state)


def test_jepa_positive_target_helper_detaches_only_selected_target_path():
    z_jepa = jnp.arange(6, dtype=jnp.float32).reshape((2, 3))
    target_z = jnp.arange(12, dtype=jnp.float32).reshape((2, 2, 3))
    pred_z = jnp.zeros_like(target_z)

    for target_mode in ("projected_bt4", "current_repeat"):
        attached = local._jepa_positive_target_vectors(
            z_jepa,
            target_z,
            pred_z,
            target_mode=target_mode,
            stop_gradient=False,
        )
        detached = local._jepa_positive_target_vectors(
            z_jepa,
            target_z,
            pred_z,
            target_mode=target_mode,
            stop_gradient=True,
        )
        np.testing.assert_array_equal(
            np.asarray(attached),
            np.asarray(detached),
        )

        def target_sum(current, future, *, stop_gradient):
            return jnp.sum(
                local._jepa_positive_target_vectors(
                    current,
                    future,
                    pred_z,
                    target_mode=target_mode,
                    stop_gradient=stop_gradient,
                )
            )

        attached_grads = jax.grad(
            lambda current, future: target_sum(
                current,
                future,
                stop_gradient=False,
            ),
            argnums=(0, 1),
        )(z_jepa, target_z)
        detached_grads = jax.grad(
            lambda current, future: target_sum(
                current,
                future,
                stop_gradient=True,
            ),
            argnums=(0, 1),
        )(z_jepa, target_z)

        assert all(
            float(jnp.linalg.norm(gradient)) == 0.0
            for gradient in detached_grads
        )
        if target_mode == "projected_bt4":
            assert float(jnp.linalg.norm(attached_grads[0])) == 0.0
            assert float(jnp.linalg.norm(attached_grads[1])) > 0.0
        else:
            assert float(jnp.linalg.norm(attached_grads[0])) > 0.0
            assert float(jnp.linalg.norm(attached_grads[1])) == 0.0


def test_local_model_outputs_loss_aux_and_gradients_match_legacy_exactly():
    kwargs = _config_kwargs()
    local_model = local.JointLatentSASAModel(
        DummyEncoder(),
        local.JointLatentSASAConfig(**kwargs),
        rngs=nnx.Rngs(23),
    )
    legacy_model = legacy.JointLatentSASAModel(
        DummyEncoder(),
        legacy.JointLatentSASAConfig(**kwargs),
        rngs=nnx.Rngs(23),
    )
    batch = _batch()

    local_tokens, local_vectors = (
        local_model.encode_current_and_future_tokens_and_vectors(
            batch["current_planes"],
            batch["future_planes"],
        )
    )
    legacy_tokens, legacy_vectors = (
        legacy_model.encode_current_and_future_tokens_and_vectors(
            batch["current_planes"],
            batch["future_planes"],
        )
    )
    _assert_trees_exact(
        (local_tokens, local_vectors),
        (legacy_tokens, legacy_vectors),
    )

    local_dfm = local_model.dfm_latents(local_tokens[:, 0])
    legacy_dfm = legacy_model.dfm_latents(legacy_tokens[:, 0])
    noisy_actions = jnp.asarray([[32, 2], [3, 32]], dtype=jnp.int32)
    t = jnp.asarray([0.25, 0.75], dtype=jnp.float32)
    local_logits, local_hidden = local_model.planner_from_latents(
        local_dfm,
        noisy_actions,
        t,
        return_hidden=True,
    )
    legacy_logits, legacy_hidden = legacy_model.planner_from_latents(
        legacy_dfm,
        noisy_actions,
        t,
        return_hidden=True,
    )
    _assert_trees_exact(
        (local_logits, local_hidden),
        (legacy_logits, legacy_hidden),
    )

    actions = batch["action_indices"]
    clean_t = jnp.ones((actions.shape[0],), dtype=jnp.float32)
    _, local_clean_hidden = local_model.planner_from_latents(
        local_dfm,
        actions,
        clean_t,
        return_hidden=True,
    )
    _, legacy_clean_hidden = legacy_model.planner_from_latents(
        legacy_dfm,
        actions,
        clean_t,
        return_hidden=True,
    )
    local_pred = local_model.jepa_rollout_from_latents(
        local_vectors[:, 0],
        actions,
        local_clean_hidden["action_tokens"],
        z0_normalized=True,
    )
    legacy_pred = legacy_model.jepa_rollout_from_latents(
        legacy_vectors[:, 0],
        actions,
        legacy_clean_hidden["action_tokens"],
        z0_normalized=True,
    )
    _assert_trees_exact(local_pred, legacy_pred)

    rng = jax.random.PRNGKey(101)
    local_loss_and_grad = nnx.value_and_grad(
        local.joint_stage1_loss_fn,
        argnums=nnx.DiffState(0, TrainableParam),
        has_aux=True,
    )
    legacy_loss_and_grad = nnx.value_and_grad(
        legacy.joint_stage1_loss_fn,
        argnums=nnx.DiffState(0, TrainableParam),
        has_aux=True,
    )
    local_value, local_gradients = local_loss_and_grad(
        local_model,
        batch,
        rng,
        compute_fp32_legality=True,
    )
    legacy_value, legacy_gradients = legacy_loss_and_grad(
        legacy_model,
        batch,
        rng,
        compute_fp32_legality=True,
    )
    _assert_trees_exact(local_value, legacy_value)
    _assert_trees_exact(
        nnx.to_pure_dict(local_gradients),
        nnx.to_pure_dict(legacy_gradients),
    )


def test_local_loss_oracle_covers_ragged_masks_and_loss_horizon():
    kwargs = _config_kwargs() | {
        "loss_horizon": 1,
        "legality_on_masked_only": True,
    }
    batch = _batch() | {
        "valid": jnp.asarray([1.0, 0.0], dtype=jnp.float32),
        "future_valid": jnp.asarray(
            [[1.0, 0.0], [1.0, 1.0]],
            dtype=jnp.float32,
        ),
        "legal_masks_valid": jnp.asarray(
            [[1.0, 0.0], [0.0, 1.0]],
            dtype=jnp.float32,
        ),
        "deterministic_t": jnp.asarray(0.0, dtype=jnp.float32),
    }

    (loss, aux), _ = _assert_loss_and_gradient_parity(
        config_kwargs=kwargs,
        batch=batch,
        model_seed=31,
        loss_seed=103,
    )

    assert jnp.isfinite(loss)
    np.testing.assert_array_equal(
        np.asarray(aux["dfm_mask_fraction_by_horizon"]),
        np.asarray([1.0, 0.0], dtype=np.float32),
    )
    assert float(aux["loss_horizon"]) == 1.0
    assert float(aux["jepa_sigreg_valid_count"]) == 2.0
    assert float(aux["jepa_pred_sigreg_valid_count"]) == 1.0


def test_zero_norm_coefficient_makes_positive_jepa_exactly_raw_mse():
    batch = _batch()
    rng = jax.random.PRNGKey(105)
    config = local.JointLatentSASAConfig(
        **(_config_kwargs() | {"jepa_norm_loss_coeff": 0.0})
    )

    def component_value_and_gradient(component):
        model = local.JointLatentSASAModel(
            DummyEncoder(),
            config,
            rngs=nnx.Rngs(35),
        )

        def component_loss(candidate):
            _, aux = local.joint_stage1_loss_fn(
                candidate,
                batch,
                rng,
                compute_fp32_legality=True,
            )
            return aux[component], aux

        return nnx.value_and_grad(
            component_loss,
            argnums=nnx.DiffState(0, TrainableParam),
            has_aux=True,
        )(model)

    positive_value, positive_gradients = component_value_and_gradient(
        "jepa_positive_loss"
    )
    raw_value, raw_gradients = component_value_and_gradient(
        "jepa_raw_mse"
    )

    _assert_trees_exact(positive_value[0], raw_value[0])
    _assert_trees_exact(
        nnx.to_pure_dict(positive_gradients),
        nnx.to_pure_dict(raw_gradients),
    )
    assert float(positive_value[1]["jepa_norm_loss"]) > 0.0


def test_norm_coefficient_only_changes_its_positive_loss_contribution():
    batch = _batch()
    rng = jax.random.PRNGKey(106)

    def evaluate(coefficient):
        model = local.JointLatentSASAModel(
            DummyEncoder(),
            local.JointLatentSASAConfig(
                **(
                    _config_kwargs()
                    | {"jepa_norm_loss_coeff": coefficient}
                )
            ),
            rngs=nnx.Rngs(36),
        )
        return local.joint_stage1_loss_fn(
            model,
            batch,
            rng,
            compute_fp32_legality=True,
        )

    compatibility_loss, compatibility_aux = evaluate(1.0)
    no_norm_loss, no_norm_aux = evaluate(0.0)

    _assert_trees_exact(
        compatibility_aux["jepa_raw_mse"],
        no_norm_aux["jepa_raw_mse"],
    )
    _assert_trees_exact(
        compatibility_aux["jepa_norm_loss"],
        no_norm_aux["jepa_norm_loss"],
    )
    np.testing.assert_allclose(
        np.asarray(
            compatibility_aux["jepa_positive_loss"]
            - no_norm_aux["jepa_positive_loss"]
        ),
        np.asarray(compatibility_aux["jepa_norm_loss"]),
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(compatibility_loss - no_norm_loss),
        np.asarray(compatibility_aux["jepa_norm_loss"]),
        rtol=1e-6,
        atol=1e-6,
    )


def test_normalized_objective_selects_u_stat_and_retains_v_diagnostic():
    model = local.JointLatentSASAModel(
        DummyEncoder(),
        local.JointLatentSASAConfig(
            **(
                _config_kwargs()
                | {
                    "jepa_norm_loss_coeff": 0.0,
                    "jepa_sigreg_estimator": "u_stat",
                }
            )
        ),
        rngs=nnx.Rngs(38),
    )

    loss, aux = local.normalized_stage1_loss_fn(
        model,
        _batch(),
        jax.random.PRNGKey(108),
        1.0,
        1.0,
    )

    assert jnp.isfinite(loss)
    _assert_trees_exact(
        aux["jepa_sigreg_loss"],
        aux["jepa_sigreg_u_stat_loss"],
    )
    _assert_trees_exact(
        aux["jepa_pred_sigreg_loss"],
        aux["jepa_pred_sigreg_u_stat_loss"],
    )
    np.testing.assert_allclose(
        np.asarray(aux["jepa_sigreg_v_stat_loss"]),
        np.asarray(
            aux["jepa_sigreg_official_loss"]
            / aux["jepa_sigreg_valid_count"]
        ),
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(aux["jepa_pred_sigreg_v_stat_loss"]),
        np.asarray(
            aux["jepa_pred_sigreg_official_loss"]
            / aux["jepa_pred_sigreg_valid_count"]
        ),
        rtol=1e-6,
        atol=1e-6,
    )


def test_fixed_sigreg_example_count_uses_shared_selected_examples():
    batch = _batch() | {
        "future_valid": jnp.ones((2, 2), dtype=jnp.float32),
    }
    model = local.JointLatentSASAModel(
        DummyEncoder(),
        local.JointLatentSASAConfig(
            **(
                _config_kwargs()
                | {"jepa_sigreg_example_count": 1}
            )
        ),
        rngs=nnx.Rngs(39),
    )

    _, aux = local.normalized_stage1_loss_fn(
        model,
        batch,
        jax.random.PRNGKey(109),
        1.0,
        1.0,
    )

    assert float(aux["jepa_sigreg_example_count"]) == 1.0
    assert float(aux["jepa_sigreg_sampling_fraction"]) == 0.5
    assert float(aux["jepa_sigreg_valid_count"]) == 3.0
    assert float(aux["jepa_pred_sigreg_valid_count"]) == 2.0
    assert jnp.isfinite(aux["jepa_sigreg_loss"])
    assert jnp.isfinite(aux["jepa_pred_sigreg_loss"])


def test_full_batch_sigreg_count_preserves_default_rng_and_statistics():
    batch = _batch()
    rng = jax.random.PRNGKey(110)

    def evaluate(example_count):
        model = local.JointLatentSASAModel(
            DummyEncoder(),
            local.JointLatentSASAConfig(
                **(
                    _config_kwargs()
                    | {"jepa_sigreg_example_count": example_count}
                )
            ),
            rngs=nnx.Rngs(40),
        )
        return local.joint_stage1_loss_fn(
            model,
            batch,
            rng,
            compute_fp32_legality=True,
        )

    default_loss, default_aux = evaluate(0)
    explicit_loss, explicit_aux = evaluate(batch["valid"].shape[0])

    _assert_trees_exact(default_loss, explicit_loss)
    for key in (
        "jepa_sigreg_loss",
        "jepa_pred_sigreg_loss",
        "jepa_sigreg_valid_count",
        "jepa_pred_sigreg_valid_count",
    ):
        _assert_trees_exact(default_aux[key], explicit_aux[key])


def test_fixed_sigreg_example_count_cannot_exceed_physical_batch():
    model = local.JointLatentSASAModel(
        DummyEncoder(),
        local.JointLatentSASAConfig(
            **(
                _config_kwargs()
                | {"jepa_sigreg_example_count": 3}
            )
        ),
        rngs=nnx.Rngs(41),
    )

    with pytest.raises(
        ValueError,
        match="cannot exceed physical batch size",
    ):
        local.joint_stage1_loss_fn(
            model,
            _batch(),
            jax.random.PRNGKey(111),
        )


def test_local_loss_oracle_covers_teacher_forcing_moments_and_heads():
    kwargs = _config_kwargs() | {
        "jepa_target_mode": "current_repeat",
        "jepa_sigreg_kind": "moments",
        "value_coeff": 0.25,
        "wdl_coeff": 0.5,
        "loss_clip_value": 0.01,
    }
    batch = _batch() | {
        "jepa_teacher_forcing": jnp.asarray(1.0, dtype=jnp.float32),
        "value_targets": jnp.asarray(
            [[0.75, -0.25], [0.5, 0.0]],
            dtype=jnp.float32,
        ),
        "wdl_targets": jnp.asarray(
            [
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                [[0.0, 0.0, 1.0], [0.0, 0.0, 0.0]],
            ],
            dtype=jnp.float32,
        ),
        "deterministic_t": jnp.asarray(0.0, dtype=jnp.float32),
    }

    (loss, aux), _ = _assert_loss_and_gradient_parity(
        config_kwargs=kwargs,
        batch=batch,
        model_seed=37,
        loss_seed=107,
    )

    assert float(aux["jepa_teacher_forcing"]) == 1.0
    assert float(aux["loss_clip_scale"]) < 1.0
    assert float(loss) <= kwargs["loss_clip_value"] + 1e-6
    assert float(aux["value_loss"]) > 0.0
    assert float(aux["wdl_loss"]) > 0.0
    assert jnp.isfinite(aux["jepa_sigreg_loss"])
    assert jnp.isfinite(aux["jepa_pred_sigreg_loss"])


def test_target_detach_changes_positive_gradient_but_not_target_sigreg():
    batch = _batch()
    rng = jax.random.PRNGKey(113)

    def loss_and_grad(*, stop_gradient):
        model = local.JointLatentSASAModel(
            DummyEncoder(),
            local.JointLatentSASAConfig(
                **(
                    _config_kwargs()
                    | {
                        "jepa_target_stop_gradient": stop_gradient,
                    }
                )
            ),
            rngs=nnx.Rngs(43),
        )
        value_and_grad = nnx.value_and_grad(
            local.joint_stage1_loss_fn,
            argnums=nnx.DiffState(0, TrainableParam),
            has_aux=True,
        )
        value, gradients = value_and_grad(model, batch, rng)
        return model, value, nnx.to_pure_dict(gradients)

    _, attached_value, attached_gradients = loss_and_grad(
        stop_gradient=False
    )
    detached_model, detached_value, detached_gradients = loss_and_grad(
        stop_gradient=True
    )

    _assert_trees_exact(attached_value, detached_value)
    _assert_trees_exact(
        attached_gradients["jepa_transition"],
        detached_gradients["jepa_transition"],
    )
    assert any(
        not np.array_equal(np.asarray(attached), np.asarray(detached))
        for attached, detached in zip(
            jax.tree.leaves(attached_gradients["state_projector"]),
            jax.tree.leaves(detached_gradients["state_projector"]),
            strict=True,
        )
    )

    def target_sigreg_only(candidate):
        _, aux = local.joint_stage1_loss_fn(
            candidate,
            batch,
            rng,
        )
        return aux["jepa_sigreg_loss"]

    target_sigreg_value, target_sigreg_gradients = nnx.value_and_grad(
        target_sigreg_only,
        argnums=nnx.DiffState(0, TrainableParam),
    )(detached_model)
    assert jnp.isfinite(target_sigreg_value)
    projector_gradient_norm = sum(
        float(
            jnp.sum(
                jnp.square(
                    jnp.asarray(gradient, dtype=jnp.float32)
                )
            )
        )
        for gradient in jax.tree.leaves(
            nnx.to_pure_dict(target_sigreg_gradients)[
                "state_projector"
            ]
        )
    )
    assert projector_gradient_norm > 0.0


def test_local_component_builder_matches_legacy_model_and_optimizer(monkeypatch):
    def make_dummy_encoder(*_args, **_kwargs):
        return DummyEncoder()

    monkeypatch.setattr(local, "make_bt4_model", make_dummy_encoder)
    monkeypatch.setattr(legacy, "make_bt4_model", make_dummy_encoder)
    kwargs = _config_kwargs()
    local_model, local_optimizer = local.create_joint_components(
        {},
        local.JointLatentSASAConfig(**kwargs),
        seed=29,
    )
    legacy_model, legacy_optimizer = legacy.create_joint_components(
        {},
        legacy.JointLatentSASAConfig(**kwargs),
        seed=29,
    )

    local_model_state = _pure_trainable(local_model)
    legacy_model_state = _pure_trainable(legacy_model)
    assert local.research_state_abi(
        local_model_state
    ) == local.research_state_abi(legacy_model_state)
    _assert_trees_exact(local_model_state, legacy_model_state)

    local_optimizer_state = _pure_optimizer(local_optimizer)
    legacy_optimizer_state = _pure_optimizer(legacy_optimizer)
    assert local.research_state_abi(
        local_optimizer_state
    ) == local.research_state_abi(legacy_optimizer_state)
    _assert_trees_exact(local_optimizer_state, legacy_optimizer_state)


def test_zero_warmup_constant_schedule_preserves_optimizer_abi(monkeypatch):
    def make_dummy_encoder(*_args, **_kwargs):
        return DummyEncoder()

    monkeypatch.setattr(local, "make_bt4_model", make_dummy_encoder)
    kwargs = _config_kwargs()
    _, warmup_optimizer = local.create_joint_components(
        {},
        local.JointLatentSASAConfig(**kwargs),
        seed=31,
    )
    _, constant_optimizer = local.create_joint_components(
        {},
        local.JointLatentSASAConfig(
            **(kwargs | {"lr_warmup_steps": 0})
        ),
        seed=31,
    )

    warmup_state = _pure_optimizer(warmup_optimizer)
    constant_state = _pure_optimizer(constant_optimizer)
    assert local.research_state_abi(
        constant_state
    ) == local.research_state_abi(warmup_state)
    _assert_trees_exact(constant_state, warmup_state)


def test_cosine_warmdown_schedule_values_and_contract():
    config = local.JointLatentSASAConfig(
        learning_rate=3e-5,
        bt4_learning_rate=1e-6,
        lr_warmup_steps=0,
        lr_decay_start_steps=400,
        lr_decay_steps=800,
        lr_min_ratio=0.1,
    )
    main_schedule, bt4_schedule = local.learning_rate_schedules(config)
    updates = (0, 399, 400, 800, 1199, 1200, 1600)
    expected_ratios = (
        1.0,
        1.0,
        1.0,
        0.55,
        0.1
        + 0.9
        * 0.5
        * (1.0 + np.cos(np.pi * np.asarray(799.0 / 800.0))),
        0.1,
        0.1,
    )
    main_values = np.asarray(
        [main_schedule(update) for update in updates]
    )
    bt4_values = np.asarray(
        [bt4_schedule(update) for update in updates]
    )
    np.testing.assert_allclose(
        main_values,
        np.asarray(expected_ratios) * config.learning_rate,
        rtol=2e-6,
        atol=0.0,
    )
    np.testing.assert_allclose(
        bt4_values,
        np.asarray(expected_ratios) * config.bt4_learning_rate,
        rtol=2e-6,
        atol=0.0,
    )
    np.testing.assert_allclose(
        main_values / bt4_values,
        config.learning_rate / config.bt4_learning_rate,
        rtol=2e-6,
        atol=0.0,
    )

    contract = local.learning_rate_schedule_contract(config)
    assert contract["family"] == "constant_then_cosine_decay"
    assert contract["counter"] == "zero_based_optimizer_update"
    assert contract["main_peak_learning_rate"] == 3e-5
    assert contract["bt4_peak_learning_rate"] == 1e-6
    assert contract["decay_start_update"] == 400
    assert contract["decay_transition_steps"] == 800
    assert contract["minimum_ratio"] == 0.1
    assert set(contract["ratio_by_update"]) == {
        "0",
        "399",
        "400",
        "800",
        "1199",
        "1200",
    }
    assert contract["ratio_by_update"]["800"] == 0.55
    assert contract["ratio_by_update"]["1200"] == 0.1
    assert (
        contract["learning_rate_by_update"]["800"]["main"]
        == 3e-5 * 0.55
    )
    serialized = local.serialized_model_config(config)
    assert serialized["lr_decay_start_steps"] == 400
    assert serialized["lr_decay_steps"] == 800
    assert serialized["lr_min_ratio"] == 0.1


def test_active_experiment_uses_accepted_ten_percent_cosine_floor():
    config = local.apply_experiment_overrides(
        local.JointLatentSASAConfig()
    )
    assert config.lr_warmup_steps == 0
    assert config.lr_decay_start_steps == 400
    assert config.lr_decay_steps == 800
    assert config.lr_min_ratio == 0.1
    main_schedule, bt4_schedule = local.learning_rate_schedules(config)
    for update, ratio in ((0, 1.0), (400, 1.0), (800, 0.55), (1200, 0.1)):
        np.testing.assert_allclose(
            main_schedule(update),
            config.learning_rate * ratio,
            rtol=2e-6,
            atol=0.0,
        )
        np.testing.assert_allclose(
            bt4_schedule(update),
            config.bt4_learning_rate * ratio,
            rtol=2e-6,
            atol=0.0,
        )
    contract = local.learning_rate_schedule_contract(config)
    assert contract["minimum_ratio"] == 0.1
    assert contract["ratio_by_update"]["800"] == 0.55
    assert contract["ratio_by_update"]["1200"] == 0.1


def test_default_constant_learning_rate_schedule_is_exact():
    config = dataclasses.replace(
        local.JointLatentSASAConfig(),
        lr_warmup_steps=0,
    )
    main_schedule, bt4_schedule = local.learning_rate_schedules(config)
    for update in (0, 1, 400, 10_000):
        assert float(main_schedule(update)) == config.learning_rate
        assert float(bt4_schedule(update)) == config.bt4_learning_rate
    contract = local.learning_rate_schedule_contract(config)
    assert contract["family"] == "constant"
    assert contract["ratio_by_update"] == {"0": 1.0}


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"lr_warmup_steps": True}, "lr_warmup_steps"),
        ({"lr_decay_start_steps": -1}, "lr_decay_start_steps"),
        ({"lr_decay_steps": 1.5}, "lr_decay_steps"),
        ({"lr_min_ratio": True}, "lr_min_ratio"),
        ({"lr_min_ratio": 0.0}, "lr_min_ratio"),
        ({"lr_min_ratio": float("nan")}, "lr_min_ratio"),
        ({"lr_decay_start_steps": 1}, "inert"),
        ({"lr_min_ratio": 0.5}, "inert"),
        (
            {"lr_warmup_steps": 1, "lr_decay_steps": 1},
            "Simultaneous learning-rate warmup and decay",
        ),
    ],
)
def test_invalid_learning_rate_schedule_configs_fail_closed(
    updates,
    message,
):
    config = dataclasses.replace(
        local.JointLatentSASAConfig(),
        lr_warmup_steps=0,
    )
    config = dataclasses.replace(config, **updates)
    with pytest.raises(ValueError, match=message):
        local.validate_learning_rate_schedule_config(config)


def test_cosine_schedule_preserves_abi_and_update_zero_exactness(
    monkeypatch,
):
    def make_dummy_encoder(*_args, **_kwargs):
        return DummyEncoder()

    monkeypatch.setattr(local, "make_bt4_model", make_dummy_encoder)
    constant_kwargs = _config_kwargs() | {"lr_warmup_steps": 0}
    cosine_kwargs = constant_kwargs | {
        "lr_decay_start_steps": 400,
        "lr_decay_steps": 800,
        "lr_min_ratio": 0.1,
    }
    constant_model, constant_optimizer = local.create_joint_components(
        {},
        local.JointLatentSASAConfig(**constant_kwargs),
        seed=37,
    )
    cosine_model, cosine_optimizer = local.create_joint_components(
        {},
        local.JointLatentSASAConfig(**cosine_kwargs),
        seed=37,
    )
    _assert_trees_exact(
        _pure_trainable(constant_model),
        _pure_trainable(cosine_model),
    )
    assert local.research_state_abi(
        _pure_optimizer(constant_optimizer)
    ) == local.research_state_abi(_pure_optimizer(cosine_optimizer))
    _assert_trees_exact(
        _pure_optimizer(constant_optimizer),
        _pure_optimizer(cosine_optimizer),
    )

    loss_and_grad = nnx.value_and_grad(
        local.joint_stage1_loss_fn,
        argnums=nnx.DiffState(0, TrainableParam),
        has_aux=True,
    )
    batch = _batch()
    rng = jax.random.PRNGKey(113)
    constant_value, constant_gradients = loss_and_grad(
        constant_model,
        batch,
        rng,
    )
    cosine_value, cosine_gradients = loss_and_grad(
        cosine_model,
        batch,
        rng,
    )
    _assert_trees_exact(constant_value, cosine_value)
    _assert_trees_exact(
        nnx.to_pure_dict(constant_gradients),
        nnx.to_pure_dict(cosine_gradients),
    )

    constant_optimizer.update(constant_model, constant_gradients)
    cosine_optimizer.update(cosine_model, cosine_gradients)
    _assert_trees_exact(
        _pure_trainable(constant_model),
        _pure_trainable(cosine_model),
    )
    _assert_trees_exact(
        _pure_optimizer(constant_optimizer),
        _pure_optimizer(cosine_optimizer),
    )


def test_local_optimizer_updates_match_legacy_exactly(monkeypatch):
    def make_dummy_encoder(*_args, **_kwargs):
        return DummyEncoder()

    monkeypatch.setattr(local, "make_bt4_model", make_dummy_encoder)
    monkeypatch.setattr(legacy, "make_bt4_model", make_dummy_encoder)
    kwargs = _config_kwargs() | {
        "grad_clip_norm": 0.05,
        "loss_clip_value": 0.01,
    }
    local_model, local_optimizer = local.create_joint_components(
        {},
        local.JointLatentSASAConfig(**kwargs),
        seed=41,
    )
    legacy_model, legacy_optimizer = legacy.create_joint_components(
        {},
        legacy.JointLatentSASAConfig(**kwargs),
        seed=41,
    )
    before = _pure_trainable(local_model)
    batch = _batch() | {
        "deterministic_t": jnp.asarray(0.0, dtype=jnp.float32),
    }
    local_loss_and_grad = nnx.value_and_grad(
        local.joint_stage1_loss_fn,
        argnums=nnx.DiffState(0, TrainableParam),
        has_aux=True,
    )
    legacy_loss_and_grad = nnx.value_and_grad(
        legacy.joint_stage1_loss_fn,
        argnums=nnx.DiffState(0, TrainableParam),
        has_aux=True,
    )
    rng = jax.random.PRNGKey(109)
    local_value, local_gradients = local_loss_and_grad(
        local_model,
        batch,
        rng,
    )
    legacy_value, legacy_gradients = legacy_loss_and_grad(
        legacy_model,
        batch,
        rng,
    )
    _assert_trees_exact(local_value, legacy_value)
    _assert_trees_exact(
        nnx.to_pure_dict(local_gradients),
        nnx.to_pure_dict(legacy_gradients),
    )

    # The first warmup update has zero learning rate; the second proves both
    # schedule state and the parameter mutation remain exactly compatible.
    for _ in range(2):
        local_optimizer.update(local_model, local_gradients)
        legacy_optimizer.update(legacy_model, legacy_gradients)
        _assert_trees_exact(
            _pure_trainable(local_model),
            _pure_trainable(legacy_model),
        )
        _assert_trees_exact(
            _pure_optimizer(local_optimizer),
            _pure_optimizer(legacy_optimizer),
        )
        assert int(local_optimizer.step[...]) == int(
            legacy_optimizer.step[...]
        )

    assert int(local_optimizer.step[...]) == 2
    after = _pure_trainable(local_model)
    assert any(
        not np.array_equal(np.asarray(left), np.asarray(right))
        for left, right in zip(
            jax.tree.leaves(before),
            jax.tree.leaves(after),
            strict=True,
        )
    )
