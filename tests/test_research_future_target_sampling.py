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
        _config(),
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
