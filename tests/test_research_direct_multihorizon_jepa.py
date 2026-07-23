from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from chess_dfm_jax.nnx_bt4 import TrainableParam
from research import train


class TinyTrainableEncoder(nnx.Module):
    def __init__(self, embedding_size: int = 16):
        self.embedding_size = embedding_size
        self.scale = TrainableParam(jnp.asarray(1.0, dtype=jnp.float32))

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
        return (square + plane_mean.reshape((batch_size, 1, 1)) * feature) * self.scale[...]


def _config(
    *,
    rollout_mode: str = "recurrent",
    **updates,
) -> train.JointLatentSASAConfig:
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
        "jepa_target_sample_count": 1,
        "jepa_target_sampling_unit": "example_balanced",
        "jepa_rollout_mode": rollout_mode,
        "use_qk_gain": True,
        "use_qk_norm": True,
        "use_xsa": True,
        "remat_blocks": False,
        "scan_layers": False,
    }
    values.update(updates)
    return train.JointLatentSASAConfig(**values)


def _model(
    *,
    rollout_mode: str = "recurrent",
    seed: int = 17,
    **updates,
) -> train.JointLatentSASAModel:
    return train.JointLatentSASAModel(
        TinyTrainableEncoder(),
        _config(rollout_mode=rollout_mode, **updates),
        rngs=nnx.Rngs(seed),
    )


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
        [current + jnp.asarray(0.05 * (index + 1), dtype=jnp.float32) for index in range(horizon)],
        axis=1,
    )
    actions = jnp.asarray(
        [
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
            [13, 14, 15, 16],
        ],
        dtype=jnp.int32,
    )
    legal_idx = jnp.stack((actions, (actions + 1) % 32), axis=-1)
    return {
        "current_planes": current,
        "future_planes": future,
        "action_indices": actions,
        "valid": jnp.ones((batch_size,), dtype=jnp.float32),
        "future_valid": jnp.ones(
            (batch_size, horizon),
            dtype=jnp.float32,
        ),
        "legal_idx": legal_idx,
        "legal_count": jnp.full(
            (batch_size, horizon),
            2,
            dtype=jnp.int32,
        ),
        "legal_masks_valid": jnp.ones(
            (batch_size, horizon),
            dtype=jnp.float32,
        ),
        "deterministic_t": jnp.asarray(0.25, dtype=jnp.float32),
    }


def _prediction_inputs(model: train.JointLatentSASAModel):
    batch = _batch()
    actions = batch["action_indices"]
    tokens = model.encode_bt4_tokens(batch["current_planes"])
    z0 = model.normalize_jepa_state(model.state_projector(tokens))
    z_dfm = model.dfm_latents(tokens)
    _, clean_hidden = model.planner_from_latents(
        z_dfm,
        actions,
        jnp.ones((actions.shape[0],), dtype=jnp.float32),
        return_hidden=True,
    )
    return z0, actions, clean_hidden["action_tokens"]


def _tree_norm(tree) -> float:
    leaves = jax.tree_util.tree_leaves(tree)
    return float(
        sum(np.sum(np.square(np.asarray(leaf, dtype=np.float64))) for leaf in leaves) ** 0.5
    )


def test_default_dispatch_is_exact_recurrent_rollout() -> None:
    model = _model()
    assert model.config.jepa_rollout_mode == "recurrent"
    z0, actions, hidden = _prediction_inputs(model)

    expected = model.jepa_rollout_from_latents(
        z0,
        actions,
        hidden,
        z0_normalized=True,
    )
    actual = model.jepa_predictions_from_latents(
        z0,
        actions,
        hidden,
        z0_normalized=True,
    )

    np.testing.assert_array_equal(actual, expected)


def test_direct_prediction_matches_independent_per_horizon_transition() -> None:
    model = _model(rollout_mode="direct_sequence")
    z0, actions, hidden = _prediction_inputs(model)
    actual = model.jepa_predictions_from_latents(
        z0,
        actions,
        hidden,
        z0_normalized=True,
    )

    condition = model.jepa_action_embed(actions) + model.jepa_hidden_adapter(hidden)
    expected = jnp.stack(
        [
            model.normalize_jepa_state(model.jepa_transition(z0, condition[:, horizon_index]))
            for horizon_index in range(actions.shape[1])
        ],
        axis=1,
    )
    recurrent = model.jepa_rollout_from_latents(
        z0,
        actions,
        hidden,
        z0_normalized=True,
    )

    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-6)
    np.testing.assert_allclose(
        actual[:, 0],
        recurrent[:, 0],
        rtol=2e-5,
        atol=2e-6,
    )


def test_direct_predictions_do_not_carry_prior_predicted_state() -> None:
    model = _model(rollout_mode="direct_sequence")
    model.jepa_transition.cond_w[...] = jnp.full_like(
        model.jepa_transition.cond_w[...],
        1e-3,
    )
    z0, actions, hidden = _prediction_inputs(model)
    baseline = model.jepa_predictions_from_latents(
        z0,
        actions,
        hidden,
        z0_normalized=True,
    )
    changed_actions = actions.at[:, 0].set(actions[:, 0] + 1)
    changed_hidden = hidden.at[:, 0, :].add(0.25)
    changed = model.jepa_predictions_from_latents(
        z0,
        changed_actions,
        changed_hidden,
        z0_normalized=True,
    )

    assert not np.allclose(baseline[:, 0], changed[:, 0])
    np.testing.assert_array_equal(baseline[:, 1:], changed[:, 1:])


def test_direct_loss_preserves_rng_assignments_and_dfm_values() -> None:
    recurrent = _model(seed=23)
    direct = _model(rollout_mode="direct_sequence", seed=23)
    batch = _batch()
    rng = jax.random.PRNGKey(29)

    recurrent_loss, recurrent_aux = train.normalized_stage1_loss_fn(
        recurrent,
        batch,
        rng,
        1.0,
        1.0,
        sample_future_targets=True,
    )
    direct_loss, direct_aux = train.normalized_stage1_loss_fn(
        direct,
        batch,
        rng,
        1.0,
        1.0,
        sample_future_targets=True,
    )

    assert jnp.isfinite(recurrent_loss)
    assert jnp.isfinite(direct_loss)
    for key in (
        "dfm_ce_loss",
        "dfm_ce_loss_by_horizon",
        "dfm_mask_fraction_by_horizon",
        "first_legality_loss",
        "jepa_target_assignment_count_by_horizon",
        "jepa_target_horizon_mask",
        "jepa_target_future_importance_weight",
        "jepa_sigreg_valid_count",
        "jepa_pred_sigreg_valid_count",
    ):
        np.testing.assert_array_equal(
            recurrent_aux[key],
            direct_aux[key],
        )
    assert direct_aux["jepa_prediction_horizon_count"] == 4.0
    assert direct_aux["jepa_target_sample_count"] == 1.0
    assert direct_aux["jepa_target_sample_fraction"] == 0.25


def test_direct_prediction_gradient_reaches_every_coupled_path() -> None:
    model = _model(rollout_mode="direct_sequence", seed=31)
    model.jepa_transition.cond_w[...] = jnp.full_like(
        model.jepa_transition.cond_w[...],
        1e-3,
    )
    batch = _batch()

    def direct_objective(candidate):
        actions = batch["action_indices"]
        tokens = candidate.encode_bt4_tokens(batch["current_planes"])
        z0 = candidate.normalize_jepa_state(candidate.state_projector(tokens))
        z_dfm = candidate.dfm_latents(tokens)
        _, hidden = candidate.planner_from_latents(
            z_dfm,
            actions,
            jnp.ones((actions.shape[0],), dtype=jnp.float32),
            return_hidden=True,
        )
        prediction = candidate.jepa_predictions_from_latents(
            z0,
            actions,
            hidden["action_tokens"],
            z0_normalized=True,
        )
        weights = jnp.linspace(
            -0.5,
            0.75,
            prediction.size,
            dtype=jnp.float32,
        ).reshape(prediction.shape)
        return jnp.mean(prediction * weights)

    gradients = nnx.to_pure_dict(
        nnx.grad(
            direct_objective,
            argnums=nnx.DiffState(0, TrainableParam),
        )(model)
    )

    for path in (
        ("encoder",),
        ("state_projector",),
        ("dfm_state_projector",),
        ("dfm_blocks",),
        ("jepa_action_embed",),
        ("jepa_hidden_adapter",),
        ("jepa_transition",),
    ):
        node = gradients
        for part in path:
            node = node[part]
        assert _tree_norm(node) > 0.0, path


def test_collapse_diagnostics_use_the_configured_prediction_graph() -> None:
    recurrent = _model(seed=41)
    direct = _model(rollout_mode="direct_sequence", seed=41)
    for model in (recurrent, direct):
        model.jepa_transition.cond_w[...] = jnp.full_like(
            model.jepa_transition.cond_w[...],
            1e-3,
        )

    recurrent_metrics = train.diagnose_joint_latents(
        recurrent,
        _batch(),
        jax.random.PRNGKey(43),
    )
    direct_metrics = train.diagnose_joint_latents(
        direct,
        _batch(),
        jax.random.PRNGKey(43),
    )

    for metrics in (recurrent_metrics, direct_metrics):
        assert jnp.all(jnp.isfinite(metrics["jepa_mse_by_horizon"]))
        assert jnp.all(jnp.isfinite(metrics["action_shuffled_mse_by_horizon"]))
    assert not np.allclose(
        recurrent_metrics["jepa_mse_by_horizon"][1:],
        direct_metrics["jepa_mse_by_horizon"][1:],
    )
    assert not np.allclose(
        recurrent_metrics["action_shuffled_mse_by_horizon"][1:],
        direct_metrics["action_shuffled_mse_by_horizon"][1:],
    )


@pytest.mark.parametrize(
    ("updates", "objective", "message"),
    [
        (
            {"jepa_rollout_mode": "unknown"},
            "normalized",
            "jepa_rollout_mode",
        ),
        (
            {
                "jepa_rollout_mode": "direct_sequence",
                "jepa_norm_loss_coeff": 1.0,
                "jepa_target_sample_count": 0,
                "jepa_target_sampling_unit": "batch_shared",
            },
            "legacy",
            "requires --objective normalized",
        ),
        (
            {
                "jepa_rollout_mode": "direct_sequence",
                "jepa_target_sample_count": 2,
                "jepa_target_sampling_unit": "batch_shared",
            },
            "normalized",
            "requires balanced per-example K=1",
        ),
        (
            {
                "jepa_rollout_mode": "direct_sequence",
                "jepa_sampled_target_anchors": True,
            },
            "normalized",
            "does not support sampled-target anchors",
        ),
    ],
)
def test_direct_config_fails_closed(
    updates: dict[str, object],
    objective: str,
    message: str,
) -> None:
    config = dataclasses.replace(_config(), **updates)
    with pytest.raises(ValueError, match=message):
        train.validate_objective_config(
            objective=objective,
            config=config,
        )


def test_direct_mode_preserves_state_and_dfm_inference_values() -> None:
    recurrent = _model(seed=37)
    direct = _model(rollout_mode="direct_sequence", seed=37)

    assert train.research_state_abi(
        nnx.state(recurrent, TrainableParam)
    ) == train.research_state_abi(nnx.state(direct, TrainableParam))

    z0, actions, _ = _prediction_inputs(recurrent)
    z_dfm = recurrent.dfm_latents(recurrent.encode_bt4_tokens(_batch()["current_planes"]))
    del z0
    t = jnp.full((actions.shape[0],), 0.5, dtype=jnp.float32)
    np.testing.assert_array_equal(
        recurrent.planner_from_latents(z_dfm, actions, t),
        direct.planner_from_latents(z_dfm, actions, t),
    )


def test_direct_serialization_and_resume_contract_are_explicit(
    monkeypatch,
) -> None:
    recurrent = _config()
    direct = _config(rollout_mode="direct_sequence")
    assert "jepa_rollout_mode" not in train.serialized_model_config(recurrent)
    assert train.serialized_model_config(direct)["jepa_rollout_mode"] == "direct_sequence"
    assert (
        train.apply_experiment_overrides(train.JointLatentSASAConfig()).jepa_rollout_mode
        == "recurrent"
    )

    monkeypatch.setattr(train, "require_within_workspace", lambda path: path)
    monkeypatch.setattr(
        train,
        "load_asset_manifest",
        lambda: {
            "trajectory_v3": {
                "archive": {
                    "sha256": "trajectory",
                    "size_bytes": 123,
                }
            }
        },
    )
    monkeypatch.setattr(train, "sha256_file", lambda _path: "source")
    contract = train.build_research_resume_contract(
        config=direct,
        objective="normalized",
        sigreg_reference_count=1.0,
        batch_size=4,
        train_seed=0,
        train_provenance={"kind": "test"},
        models_dir=train.REPO_ROOT / "models",
    )
    graph = contract["objective"]["jepa_prediction_graph"]
    assert contract["model_config"]["jepa_rollout_mode"] == "direct_sequence"
    assert graph["mode"] == "direct_sequence"
    assert graph["recurrent_prediction_carry"] is False
    assert graph["parameter_state_abi"] == "unchanged"
    assert graph["dfm_inference_affected"] is False
