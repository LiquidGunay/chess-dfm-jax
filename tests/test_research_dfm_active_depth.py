from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from research import train


class DeterministicEncoder(nnx.Module):
    def __init__(self, embedding_size: int = 16):
        self.embedding_size = embedding_size

    def encode_tokens(self, planes: jax.Array) -> jax.Array:
        plane_mean = jnp.mean(planes, axis=(1, 2, 3))
        square = jnp.linspace(-0.5, 0.5, 64, dtype=jnp.float32)[None, :, None]
        feature = jnp.linspace(
            0.1,
            1.0,
            self.embedding_size,
            dtype=jnp.float32,
        )[None, None, :]
        return square + plane_mean[:, None, None] * feature


def _config(**updates) -> train.JointLatentSASAConfig:
    values = {
        "token_dim": 8,
        "z_dim": 16,
        "projector_layers": 1,
        "projector_num_heads": 4,
        "projector_mlp_dim": 32,
        "jepa_condition_dim": 12,
        "dfm_layers": 4,
        "dfm_active_layers": 0,
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
        "jepa_sigreg_coeff": 0.1,
        "jepa_pred_sigreg_coeff": 0.1,
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
        [current + 0.1 * (index + 1) for index in range(horizon)],
        axis=1,
    )
    actions = jnp.asarray(
        [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]],
        dtype=jnp.int32,
    )
    return {
        "current_planes": current,
        "future_planes": future,
        "action_indices": actions,
        "valid": jnp.ones((batch_size,), dtype=jnp.float32),
        "future_valid": jnp.ones(
            (batch_size, horizon),
            dtype=jnp.float32,
        ),
        "legal_idx": jnp.stack(
            [actions, (actions + 1) % 32],
            axis=-1,
        ),
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


def _model(*, active_layers: int, seed: int = 17):
    return train.JointLatentSASAModel(
        DeterministicEncoder(),
        _config(dfm_active_layers=active_layers),
        rngs=nnx.Rngs(seed),
    )


def _assert_tree_exact(left, right) -> None:
    assert jax.tree.structure(left) == jax.tree.structure(right)
    for left_leaf, right_leaf in zip(
        jax.tree.leaves(left),
        jax.tree.leaves(right),
        strict=True,
    ):
        np.testing.assert_array_equal(left_leaf, right_leaf)


def _planner_inputs(model):
    batch = _batch()
    tokens = model.encode_bt4_tokens(batch["current_planes"])
    z_dfm = model.dfm_latents(tokens)
    actions = batch["action_indices"]
    t = jnp.full((actions.shape[0],), 0.75, dtype=jnp.float32)
    return z_dfm, actions, t


def test_active_depth_preserves_parameter_abi_and_full_control() -> None:
    full = _model(active_layers=0)
    explicit_full = _model(active_layers=4)
    three = _model(active_layers=3)

    full_state = nnx.to_pure_dict(nnx.state(full, train.TrainableParam))
    explicit_state = nnx.to_pure_dict(nnx.state(explicit_full, train.TrainableParam))
    three_state = nnx.to_pure_dict(nnx.state(three, train.TrainableParam))
    _assert_tree_exact(full_state, explicit_state)
    _assert_tree_exact(full_state, three_state)
    assert train.research_state_abi(full_state) == train.research_state_abi(three_state)

    inputs = _planner_inputs(full)
    full_logits = full.planner_from_latents(*inputs)
    explicit_logits = explicit_full.planner_from_latents(*inputs)
    three_logits = three.planner_from_latents(*inputs)
    np.testing.assert_array_equal(explicit_logits, full_logits)
    assert not np.array_equal(np.asarray(three_logits), np.asarray(full_logits))


def test_first_action_objective_weights_are_unit_sum_and_differentiable() -> None:
    weights = train.dfm_objective_horizon_weights(
        jnp.ones((8,), dtype=jnp.float32),
        first_action_loss_share=3.0 / 16.0,
    )
    np.testing.assert_allclose(
        weights,
        np.asarray([3.0 / 16.0] + [13.0 / 112.0] * 7, dtype=np.float32),
        rtol=0.0,
        atol=1e-7,
    )
    np.testing.assert_allclose(jnp.sum(weights), 1.0, rtol=0.0, atol=2e-7)

    losses = jnp.arange(8, dtype=jnp.float32)
    gradients = jax.grad(lambda values: jnp.sum(values * weights))(losses)
    np.testing.assert_array_equal(gradients, weights)


def test_default_first_action_weighting_preserves_uniform_loss_exactly() -> None:
    uniform = train.dfm_objective_horizon_weights(
        jnp.ones((8,), dtype=jnp.float32),
        first_action_loss_share=0.0,
    )
    np.testing.assert_array_equal(
        uniform,
        jnp.full((8,), 1.0 / 8.0, dtype=jnp.float32),
    )

    default = _model(active_layers=0, seed=18)
    weighted = train.JointLatentSASAModel(
        DeterministicEncoder(),
        _config(dfm_first_action_loss_share=0.5),
        rngs=nnx.Rngs(18),
    )
    _assert_tree_exact(
        nnx.to_pure_dict(nnx.state(default, train.TrainableParam)),
        nnx.to_pure_dict(nnx.state(weighted, train.TrainableParam)),
    )
    args = (_batch(), jax.random.PRNGKey(20), 1.0, 1.0)
    default_loss, default_aux = train.normalized_stage1_loss_fn(
        default,
        *args,
        sample_future_targets=True,
    )
    weighted_loss, weighted_aux = train.normalized_stage1_loss_fn(
        weighted,
        *args,
        sample_future_targets=True,
    )
    np.testing.assert_array_equal(
        weighted_aux["dfm_ce_loss"],
        default_aux["dfm_ce_loss"],
    )
    np.testing.assert_array_equal(
        weighted_aux["dfm_ce_loss_by_horizon"],
        default_aux["dfm_ce_loss_by_horizon"],
    )
    assert "dfm_objective_ce_loss" not in default_aux
    assert "dfm_objective_weight_by_horizon" not in default_aux
    expected_weights = jnp.asarray(
        [0.5, 1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0],
        dtype=jnp.float32,
    )
    np.testing.assert_allclose(
        weighted_aux["dfm_objective_weight_by_horizon"],
        expected_weights,
        rtol=0.0,
        atol=1e-7,
    )
    expected_objective = jnp.sum(
        weighted_aux["dfm_ce_loss_by_horizon"] * expected_weights
    )
    np.testing.assert_allclose(
        weighted_aux["dfm_objective_ce_loss"],
        expected_objective,
        rtol=0.0,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        weighted_loss - default_loss,
        weighted_aux["dfm_objective_ce_loss"]
        - default_aux["dfm_ce_loss"],
        rtol=0.0,
        atol=1e-6,
    )


@pytest.mark.parametrize("value", [True, -0.1, 1.1, float("nan")])
def test_first_action_share_validation_fails_closed(value) -> None:
    with pytest.raises(ValueError, match="dfm_first_action_loss_share"):
        train.validate_objective_config(
            objective="normalized",
            config=_config(dfm_first_action_loss_share=value),
        )


def test_first_action_share_requires_normalized_multi_horizon_objective() -> None:
    config = _config(dfm_first_action_loss_share=0.25)
    train.validate_objective_config(objective="normalized", config=config)
    with pytest.raises(ValueError, match="requires --objective normalized"):
        train.validate_objective_config(objective="legacy", config=config)
    with pytest.raises(ValueError, match="at least two active loss horizons"):
        train.validate_objective_config(
            objective="normalized",
            config=_config(
                dfm_first_action_loss_share=0.25,
                loss_horizon=1,
            ),
        )


@pytest.mark.parametrize("value", [True, -0.1, float("nan"), float("inf")])
def test_first_legality_coefficient_validation_fails_closed(value) -> None:
    with pytest.raises(ValueError, match="first_legality_coeff"):
        train.validate_objective_config(
            objective="normalized",
            config=_config(first_legality_coeff=value),
        )


def test_first_action_share_serialization_is_default_off() -> None:
    default = train.serialized_model_config(_config())
    active = train.serialized_model_config(
        _config(dfm_first_action_loss_share=0.25)
    )
    assert "dfm_first_action_loss_share" not in default
    assert active["dfm_first_action_loss_share"] == 0.25


def test_three_layer_output_matches_explicit_prefix_execution() -> None:
    model = _model(active_layers=3, seed=19)
    z_dfm, actions, t = _planner_inputs(model)
    horizon = actions.shape[1]
    action_emb = model.action_embed(actions)
    action_emb = (
        action_emb
        + jnp.asarray(
            model.pos_embed[...],
            dtype=model.compute_dtype,
        )[None, :horizon, :]
    )
    action_emb = action_emb + model.get_time_embedding(t)[:, None, :]
    seq = jnp.concatenate([z_dfm, action_emb], axis=1)
    params = train._transformer_stack_prefix_params(
        model.dfm_blocks,
        active_layers=3,
        compute_dtype=model.compute_dtype,
    )
    for layer_index in range(3):
        seq = model.dfm_blocks._layer(
            seq,
            tuple(param[layer_index] for param in params),
        )
    manual = model.logits_from_action_hidden(seq[:, 64:, :])

    np.testing.assert_array_equal(
        model.planner_from_latents(z_dfm, actions, t),
        manual,
    )


def test_inactive_dfm_block_has_exact_zero_loss_gradient() -> None:
    model = _model(active_layers=3, seed=23)

    def loss_fn(candidate):
        return train.normalized_stage1_loss_fn(
            candidate,
            _batch(),
            jax.random.PRNGKey(29),
            1.0,
            1.0,
            sample_future_targets=True,
        )

    (loss, aux), gradients = nnx.value_and_grad(
        loss_fn,
        argnums=nnx.DiffState(0, train.TrainableParam),
        has_aux=True,
    )(model)
    pure_blocks = nnx.to_pure_dict(gradients)["dfm_blocks"]

    assert jnp.isfinite(loss)
    assert aux["dfm_configured_layers"] == 4.0
    assert aux["dfm_active_layers"] == 3.0
    assert aux["dfm_active_layer_fraction"] == 0.75
    active_squared_norm = 0.0
    for leaf in jax.tree.leaves(pure_blocks):
        assert leaf.shape[0] == 4
        active_squared_norm += float(jnp.sum(jnp.square(jnp.asarray(leaf[:3], dtype=jnp.float32))))
        np.testing.assert_array_equal(leaf[3], jnp.zeros_like(leaf[3]))
    assert active_squared_norm > 0.0


def test_active_depth_preserves_targets_and_rng_but_changes_planner() -> None:
    full = _model(active_layers=0, seed=31)
    three = _model(active_layers=3, seed=31)
    batch = _batch()
    np.testing.assert_array_equal(
        three.encode_current_jepa(batch["current_planes"]),
        full.encode_current_jepa(batch["current_planes"]),
    )
    np.testing.assert_array_equal(
        three.encode_future_targets(batch["future_planes"]),
        full.encode_future_targets(batch["future_planes"]),
    )

    args = (batch, jax.random.PRNGKey(37), 1.0, 1.0)
    full_loss, full_aux = train.normalized_stage1_loss_fn(
        full,
        *args,
        sample_future_targets=True,
    )
    three_loss, three_aux = train.normalized_stage1_loss_fn(
        three,
        *args,
        sample_future_targets=True,
    )
    np.testing.assert_array_equal(
        three_aux["jepa_target_horizon_mask"],
        full_aux["jepa_target_horizon_mask"],
    )
    assert three_aux["bt4_encoded_boards_per_example"] == 3.0
    assert three_aux["jepa_prediction_horizon_count"] == 4.0
    assert three_aux["jepa_sigreg_valid_count"] == 10.0
    assert three_aux["jepa_pred_sigreg_valid_count"] == 8.0
    assert three_aux["dfm_ce_loss"] != full_aux["dfm_ce_loss"]
    planner_inputs = _planner_inputs(full)
    _, full_hidden = full.planner_from_latents(
        *planner_inputs,
        return_hidden=True,
    )
    _, three_hidden = three.planner_from_latents(
        *planner_inputs,
        return_hidden=True,
    )
    assert not np.array_equal(
        np.asarray(three_hidden["action_tokens"]),
        np.asarray(full_hidden["action_tokens"]),
    )
    assert three_loss != full_loss
    assert "dfm_active_layers" not in full_aux


def test_full_horizon_evaluation_reports_same_active_depth() -> None:
    model = _model(active_layers=3, seed=41)
    loss, aux = train.eval_normalized_stage1_step(
        model,
        _batch(),
        jax.random.PRNGKey(43),
        1.0,
        1.0,
    )
    assert jnp.isfinite(loss)
    assert aux["jepa_target_sample_count"] == 4.0
    assert aux["dfm_configured_layers"] == 4.0
    assert aux["dfm_active_layers"] == 3.0
    assert aux["dfm_active_layer_fraction"] == 0.75


@pytest.mark.parametrize("active_layers", [-1, 5, True])
def test_invalid_active_depth_fails_before_compilation(active_layers) -> None:
    config = dataclasses.replace(
        _config(),
        dfm_active_layers=active_layers,
    )
    with pytest.raises(ValueError, match="dfm_active_layers"):
        train.validate_objective_config(
            objective="normalized",
            config=config,
        )
    with pytest.raises(ValueError, match="dfm_active_layers"):
        train.JointLatentSASAModel(
            DeterministicEncoder(),
            config,
            rngs=nnx.Rngs(47),
        )


def test_active_depth_serialization_and_resume_contract(monkeypatch) -> None:
    default = _config(dfm_active_layers=0)
    enabled = _config(dfm_active_layers=3)
    assert "dfm_active_layers" not in train.serialized_model_config(default)
    assert train.serialized_model_config(enabled)["dfm_active_layers"] == 3

    monkeypatch.setattr(train, "require_within_workspace", lambda path: path)
    monkeypatch.setattr(
        train,
        "load_asset_manifest",
        lambda: {"trajectory_v3": {"archive": {"sha256": "trajectory", "size_bytes": 123}}},
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
    semantics = contract["objective"]["dfm_active_depth"]
    assert semantics["configured_parameter_layers"] == 4
    assert semantics["active_forward_layers"] == 3
    assert semantics["active_layer_indices"] == [0, 1, 2]
    assert semantics["inactive_layer_loss_gradient"] == "exact_zero"
    assert semantics["source_model_restore"] == "exact_all_parameters"
    assert semantics["training_evaluation_inference_depth_shared"] is True
    assert semantics["dfm_inference_path_affected"] is True
    assert semantics["refinement_passes_changed"] is False
