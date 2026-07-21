from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax import nnx

from research import train


ENCODE_BATCH_SIZES: list[int] = []


class RecordingEncoder(nnx.Module):
    def __init__(self, embedding_size: int = 16):
        self.embedding_size = embedding_size

    def encode_tokens(self, planes: jax.Array) -> jax.Array:
        ENCODE_BATCH_SIZES.append(int(planes.shape[0]))
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


def _config(**updates) -> train.JointLatentSASAConfig:
    values = {
        "token_dim": 8,
        "z_dim": 16,
        "projector_layers": 1,
        "projector_num_heads": 4,
        "projector_mlp_dim": 32,
        "jepa_condition_dim": 12,
        "dfm_layers": 1,
        "jepa_layers": 1,
        "jepa_mlp_dim": 32,
        "num_heads": 4,
        "mlp_dim": 32,
        "action_vocab_size": 32,
        "horizon": 4,
        "compute_dtype": "float32",
        "param_dtype": "float32",
        "encoder_dtype": "float32",
        "first_legality_coeff": 1.0,
        "legality_on_masked_only": False,
        "jepa_norm_loss_coeff": 0.0,
        "jepa_sigreg_coeff": 5.76,
        "jepa_pred_sigreg_coeff": 1.0,
        "jepa_sigreg_kind": "le_jepa",
        "jepa_sigreg_proj_dim": 8,
        "jepa_sigreg_example_count": 2,
        "jepa_target_sample_count": 2,
        "use_qk_gain": True,
        "use_qk_norm": True,
        "use_xsa": True,
        "remat_blocks": False,
        "scan_layers": False,
    }
    values.update(updates)
    return train.JointLatentSASAConfig(**values)


def _batch() -> dict[str, jax.Array]:
    batch_size = 4
    horizon = 4
    current = jnp.linspace(
        -0.25,
        0.75,
        batch_size * 112 * 8 * 8,
        dtype=jnp.float32,
    ).reshape((batch_size, 112, 8, 8))
    future = jnp.stack(
        [current + jnp.asarray(0.1 * (index + 1), dtype=jnp.float32) for index in range(horizon)],
        axis=1,
    )
    actions = jnp.asarray(
        [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]],
        dtype=jnp.int32,
    )
    legal_idx = jnp.stack(
        [actions, (actions + 1) % 32],
        axis=-1,
    )
    return {
        "current_planes": current,
        "future_planes": future,
        "action_indices": actions,
        "valid": jnp.ones((batch_size,), dtype=jnp.float32),
        "future_valid": jnp.ones((batch_size, horizon), dtype=jnp.float32),
        "legal_idx": legal_idx,
        "legal_count": jnp.full((batch_size, horizon), 2, dtype=jnp.int32),
        "legal_masks_valid": jnp.ones((batch_size, horizon), dtype=jnp.float32),
        "deterministic_t": jnp.asarray(0.25, dtype=jnp.float32),
    }


def _loss(
    model: train.JointLatentSASAModel,
    batch: dict[str, jax.Array],
    rng: jax.Array,
    *,
    sample: bool,
):
    return train.normalized_stage1_loss_fn(
        model,
        batch,
        rng,
        1.0,
        1.0,
        sample_future_targets=sample,
    )


def test_two_target_training_is_deterministic_and_preserves_full_mass() -> None:
    ENCODE_BATCH_SIZES.clear()
    model = train.JointLatentSASAModel(
        RecordingEncoder(),
        _config(),
        rngs=nnx.Rngs(17),
    )
    batch = _batch()
    rng = jax.random.PRNGKey(123)

    sampled_loss, sampled = _loss(model, batch, rng, sample=True)
    repeated_loss, repeated = _loss(model, batch, rng, sample=True)
    full_loss, full = _loss(model, batch, rng, sample=False)

    assert jnp.isfinite(sampled_loss)
    assert sampled_loss == repeated_loss
    np.testing.assert_array_equal(
        sampled["jepa_target_horizon_mask"],
        repeated["jepa_target_horizon_mask"],
    )
    assert ENCODE_BATCH_SIZES == [12, 12, 20]
    assert sampled["jepa_target_sample_count"] == 2.0
    assert sampled["jepa_target_sample_fraction"] == 0.5
    assert sampled["jepa_target_sampling_active"] == 1.0
    assert sampled["jepa_target_future_importance_weight"] == 2.0
    assert sampled["bt4_encoded_boards_per_example"] == 3.0
    assert sampled["jepa_prediction_horizon_count"] == 4.0
    assert sampled["jepa_sigreg_valid_count"] == 10.0
    assert sampled["jepa_pred_sigreg_valid_count"] == 8.0
    assert jnp.sum(sampled["jepa_target_horizon_mask"]) == 2.0
    assert jnp.all(
        sampled["jepa_loss_by_horizon"]
        * (1.0 - sampled["jepa_target_horizon_mask"])
        == 0.0
    )

    # Sampling is derived from the SIGReg branch, so the DFM time/mask stream
    # and its loss are unchanged for the same model, batch, and root key.
    assert sampled["dfm_ce_loss"] == full["dfm_ce_loss"]
    assert sampled["first_legality_loss"] == full["first_legality_loss"]
    assert full["jepa_target_sample_count"] == 4.0
    assert jnp.all(full["jepa_target_horizon_mask"] == 1.0)
    assert "jepa_target_sampling_active" not in full


def test_one_target_training_preserves_mass_and_prediction_coverage() -> None:
    ENCODE_BATCH_SIZES.clear()
    model = train.JointLatentSASAModel(
        RecordingEncoder(),
        _config(jepa_target_sample_count=1),
        rngs=nnx.Rngs(18),
    )
    loss, aux = _loss(
        model,
        _batch(),
        jax.random.PRNGKey(124),
        sample=True,
    )

    assert jnp.isfinite(loss)
    assert ENCODE_BATCH_SIZES == [8]
    assert aux["jepa_target_sample_count"] == 1.0
    assert aux["jepa_target_sample_fraction"] == 0.25
    assert aux["jepa_target_future_importance_weight"] == 4.0
    assert aux["bt4_encoded_boards_per_example"] == 2.0
    assert aux["jepa_prediction_horizon_count"] == 4.0
    assert aux["jepa_sigreg_valid_count"] == 10.0
    assert aux["jepa_pred_sigreg_valid_count"] == 8.0
    assert jnp.sum(aux["jepa_target_horizon_mask"]) == 1.0
    assert jnp.all(
        aux["jepa_loss_by_horizon"]
        * (1.0 - aux["jepa_target_horizon_mask"])
        == 0.0
    )


def test_active_experiment_uses_accepted_two_sampled_future_targets() -> None:
    config = train.apply_experiment_overrides(
        train.JointLatentSASAConfig()
    )
    assert config.jepa_target_sample_count == 2
    assert config.lr_decay_start_steps == 400
    assert config.lr_decay_steps == 800
    assert config.lr_min_ratio == 0.1


def test_sampled_target_subset_changes_across_predeclared_rng_keys() -> None:
    model = train.JointLatentSASAModel(
        RecordingEncoder(),
        _config(),
        rngs=nnx.Rngs(19),
    )
    batch = _batch()
    masks = []
    for seed in range(4):
        _, aux = _loss(
            model,
            batch,
            jax.random.PRNGKey(seed),
            sample=True,
        )
        masks.append(tuple(np.asarray(aux["jepa_target_horizon_mask"]).tolist()))
    assert len(set(masks)) > 1


def test_full_horizon_evaluation_ignores_training_sample_count() -> None:
    ENCODE_BATCH_SIZES.clear()
    model = train.JointLatentSASAModel(
        RecordingEncoder(),
        _config(),
        rngs=nnx.Rngs(23),
    )
    loss, aux = train.eval_normalized_stage1_step(
        model,
        _batch(),
        jax.random.PRNGKey(7),
        1.0,
        1.0,
    )
    assert jnp.isfinite(loss)
    assert ENCODE_BATCH_SIZES == [20]
    assert aux["jepa_target_sample_count"] == 4.0
    assert aux["jepa_sigreg_valid_count"] == 10.0
    assert aux["jepa_pred_sigreg_valid_count"] == 8.0
    assert "jepa_target_sampling_active" not in aux


def test_compiled_normalized_train_step_activates_target_sampling() -> None:
    model = train.JointLatentSASAModel(
        RecordingEncoder(),
        _config(jepa_sampled_target_anchors=True),
        rngs=nnx.Rngs(29),
    )
    optimizer = nnx.Optimizer(
        model,
        optax.adamw(1e-3),
        wrt=train.TrainableParam,
    )
    loss, aux = train.train_normalized_stage1_step(
        model,
        optimizer,
        _batch(),
        jax.random.PRNGKey(11),
        1.0,
        1.0,
    )
    assert jnp.isfinite(loss)
    assert aux["jepa_target_sampling_active"] == 1.0
    assert aux["jepa_target_sample_count"] == 2.0
    assert aux["bt4_encoded_boards_per_example"] == 3.0
    assert aux["jepa_prediction_horizon_count"] == 4.0
    assert aux["jepa_sampled_target_anchors_active"] == 1.0


def test_sampled_targets_fail_closed_for_unsupported_objectives() -> None:
    config = _config()
    train.validate_no_inert_config_overrides(config)
    train.validate_objective_config(objective="normalized", config=config)

    with pytest.raises(ValueError, match="require --objective normalized"):
        train.validate_objective_config(
            objective="legacy",
            config=_config(
                jepa_norm_loss_coeff=1.0,
                jepa_sigreg_example_count=0,
            ),
        )
    with pytest.raises(ValueError, match="require online targets"):
        train.validate_objective_config(
            objective="normalized",
            config=_config(jepa_target_semantics="ema"),
        )
    with pytest.raises(ValueError, match="variance hinge"):
        train.validate_objective_config(
            objective="normalized",
            config=_config(jepa_target_variance_hinge_coeff=1.0),
        )


def test_sparse_anchors_change_only_downstream_recurrent_predictions() -> None:
    model = train.JointLatentSASAModel(
        RecordingEncoder(),
        _config(jepa_sampled_target_anchors=True),
        rngs=nnx.Rngs(31),
    )
    batch_size = 2
    horizon = 4
    z0 = jnp.linspace(
        -0.4,
        0.6,
        batch_size * model.z_dim,
        dtype=jnp.float32,
    ).reshape((batch_size, model.z_dim))
    actions = jnp.asarray(
        [[1, 2, 3, 4], [5, 6, 7, 8]],
        dtype=jnp.int32,
    )
    action_hidden = jnp.linspace(
        -0.3,
        0.5,
        batch_size * horizon * model.config.token_dim,
        dtype=jnp.float32,
    ).reshape((batch_size, horizon, model.config.token_dim))
    free = model.jepa_rollout_from_latents(
        z0,
        actions,
        action_hidden,
        z0_normalized=True,
    )
    anchor_latents = free + 2.0
    anchor_mask = jnp.asarray(
        [[False, True, False, False], [False, False, False, True]],
        dtype=jnp.bool_,
    )
    anchored = model.jepa_rollout_from_latents_with_anchors(
        z0,
        actions,
        action_hidden,
        anchor_latents,
        anchor_mask,
        z0_normalized=True,
    )

    # The first sample anchors after h2, so h1-h2 are the untouched free
    # predictions and only h3-h4 can change.
    np.testing.assert_array_equal(anchored[0, :2], free[0, :2])
    assert not np.array_equal(np.asarray(anchored[0, 2:]), np.asarray(free[0, 2:]))
    # A final-horizon anchor has no downstream transition and is exactly inert.
    np.testing.assert_array_equal(anchored[1], free[1])

    def downstream_sum(anchors):
        predictions = model.jepa_rollout_from_latents_with_anchors(
            z0,
            actions,
            action_hidden,
            anchors,
            anchor_mask,
            z0_normalized=True,
        )
        return jnp.sum(predictions[:, 2:])

    anchor_grad = jax.grad(downstream_sum)(anchor_latents)
    assert jnp.linalg.norm(anchor_grad[0, 1]) > 0.0
    np.testing.assert_array_equal(
        anchor_grad[1, 3],
        jnp.zeros_like(anchor_grad[1, 3]),
    )


def test_sampled_anchor_training_reuses_sampling_and_encoder_budget() -> None:
    ENCODE_BATCH_SIZES.clear()
    control = train.JointLatentSASAModel(
        RecordingEncoder(),
        _config(jepa_sampled_target_anchors=False),
        rngs=nnx.Rngs(37),
    )
    candidate = train.JointLatentSASAModel(
        RecordingEncoder(),
        _config(jepa_sampled_target_anchors=True),
        rngs=nnx.Rngs(37),
    )
    batch = _batch()
    rng = jax.random.PRNGKey(127)
    control_loss, control_aux = _loss(control, batch, rng, sample=True)
    candidate_loss, candidate_aux = _loss(candidate, batch, rng, sample=True)

    assert ENCODE_BATCH_SIZES == [12, 12]
    np.testing.assert_array_equal(
        candidate_aux["jepa_target_horizon_mask"],
        control_aux["jepa_target_horizon_mask"],
    )
    np.testing.assert_array_equal(
        candidate_aux["jepa_sampled_target_anchor_horizon_mask"],
        control_aux["jepa_target_horizon_mask"]
        * jnp.asarray([1.0, 1.0, 1.0, 0.0], dtype=jnp.float32),
    )
    assert candidate_aux["jepa_sampled_target_anchors_active"] == 1.0
    assert candidate_aux["bt4_encoded_boards_per_example"] == 3.0
    assert candidate_aux["jepa_prediction_horizon_count"] == 4.0
    assert candidate_aux["jepa_sigreg_valid_count"] == 10.0
    assert candidate_aux["jepa_pred_sigreg_valid_count"] == 8.0
    assert candidate_aux["dfm_ce_loss"] == control_aux["dfm_ce_loss"]
    assert candidate_aux["first_legality_loss"] == control_aux["first_legality_loss"]
    assert candidate_aux["jepa_sigreg_loss"] == control_aux["jepa_sigreg_loss"]
    assert candidate_aux["z_pred_norm"] != control_aux["z_pred_norm"]
    assert candidate_loss != control_loss


def test_invalid_sampled_targets_never_anchor() -> None:
    control = train.JointLatentSASAModel(
        RecordingEncoder(),
        _config(jepa_sampled_target_anchors=False),
        rngs=nnx.Rngs(41),
    )
    candidate = train.JointLatentSASAModel(
        RecordingEncoder(),
        _config(jepa_sampled_target_anchors=True),
        rngs=nnx.Rngs(41),
    )
    batch = _batch()
    rng = jax.random.PRNGKey(131)
    _, initial_aux = _loss(control, batch, rng, sample=True)
    selected = initial_aux["jepa_target_horizon_mask"] > 0.0
    invalid_batch = dict(batch)
    invalid_batch["future_valid"] = jnp.where(
        selected[None, :],
        0.0,
        batch["future_valid"],
    )

    control_loss, control_aux = _loss(
        control,
        invalid_batch,
        rng,
        sample=True,
    )
    candidate_loss, candidate_aux = _loss(
        candidate,
        invalid_batch,
        rng,
        sample=True,
    )
    assert candidate_aux["jepa_sampled_target_anchor_example_count"] == 0.0
    assert candidate_aux["jepa_sampled_target_anchor_fraction"] == 0.0
    assert candidate_aux["z_pred_norm"] == control_aux["z_pred_norm"]
    assert candidate_aux["jepa_pred_sigreg_loss"] == control_aux["jepa_pred_sigreg_loss"]
    assert candidate_loss == control_loss


def test_anchor_config_evaluates_with_a_fully_free_rollout() -> None:
    ENCODE_BATCH_SIZES.clear()
    control = train.JointLatentSASAModel(
        RecordingEncoder(),
        _config(jepa_sampled_target_anchors=False),
        rngs=nnx.Rngs(43),
    )
    candidate = train.JointLatentSASAModel(
        RecordingEncoder(),
        _config(jepa_sampled_target_anchors=True),
        rngs=nnx.Rngs(43),
    )
    args = (_batch(), jax.random.PRNGKey(137), 1.0, 1.0)
    control_loss, control_aux = train.eval_normalized_stage1_step(control, *args)
    candidate_loss, candidate_aux = train.eval_normalized_stage1_step(candidate, *args)

    # The compiled evaluator reuses the identical free-rollout graph across
    # the disabled/enabled configs, so the Python recording side effect occurs
    # only on the first trace.
    assert ENCODE_BATCH_SIZES == [20]
    assert candidate_loss == control_loss
    assert candidate_aux["z_pred_norm"] == control_aux["z_pred_norm"]
    assert candidate_aux["jepa_positive_loss"] == control_aux["jepa_positive_loss"]
    assert "jepa_sampled_target_anchors_active" not in candidate_aux
    assert candidate_aux["jepa_target_sample_count"] == 4.0


def test_anchor_config_and_resume_semantics_fail_closed(monkeypatch) -> None:
    enabled = _config(jepa_sampled_target_anchors=True)
    train.validate_no_inert_config_overrides(enabled)
    train.validate_objective_config(objective="normalized", config=enabled)
    assert train.serialized_model_config(enabled)[
        "jepa_sampled_target_anchors"
    ] is True
    assert "jepa_sampled_target_anchors" not in train.serialized_model_config(
        train.JointLatentSASAConfig()
    )

    for sample_count in (0, enabled.horizon):
        with pytest.raises(ValueError, match="strict sampled-target count"):
            train.validate_objective_config(
                objective="normalized",
                config=_config(
                    jepa_target_sample_count=sample_count,
                    jepa_sampled_target_anchors=True,
                ),
            )

    monkeypatch.setattr(train, "require_within_workspace", lambda path: path)
    monkeypatch.setattr(
        train,
        "load_asset_manifest",
        lambda: {
            "trajectory_v3": {
                "archive": {"sha256": "trajectory", "size_bytes": 123}
            }
        },
    )
    monkeypatch.setattr(train, "sha256_file", lambda _path: "source")
    contract = train.build_research_resume_contract(
        config=enabled,
        objective="normalized",
        sigreg_reference_count=1.0,
        batch_size=4,
        train_seed=0,
        train_provenance={"kind": "test"},
        models_dir=train.REPO_ROOT / "models",
    )
    semantics = contract["objective"]["sampled_target_rollout_anchors"]
    assert semantics["timing"] == "after_prediction_h_before_transition_h_plus_1"
    assert semantics["target_gradient"] == "attached_online"
    assert semantics["evaluation_rollout"] == "fully_free_running"
