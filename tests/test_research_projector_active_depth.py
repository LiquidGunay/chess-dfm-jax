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
        return square + plane_mean[:, None, None] * feature


def _config(**updates) -> train.JointLatentSASAConfig:
    values = {
        "token_dim": 8,
        "z_dim": 16,
        "projector_layers": 2,
        "jepa_projector_active_layers": 0,
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
        "future_valid": jnp.ones((batch_size, horizon), dtype=jnp.float32),
        "legal_idx": jnp.stack([actions, (actions + 1) % 32], axis=-1),
        "legal_count": jnp.full((batch_size, horizon), 2, dtype=jnp.int32),
        "legal_masks_valid": jnp.ones(
            (batch_size, horizon),
            dtype=jnp.float32,
        ),
        "deterministic_t": jnp.asarray(0.25, dtype=jnp.float32),
    }


def _model(*, active_layers: int, seed: int = 17):
    return train.JointLatentSASAModel(
        DeterministicEncoder(),
        _config(jepa_projector_active_layers=active_layers),
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


def test_active_depth_preserves_parameter_abi_and_full_depth_control() -> None:
    full = _model(active_layers=0)
    explicit_full = _model(active_layers=2)
    one_block = _model(active_layers=1)

    full_state = nnx.to_pure_dict(nnx.state(full, train.TrainableParam))
    explicit_state = nnx.to_pure_dict(
        nnx.state(explicit_full, train.TrainableParam)
    )
    one_block_state = nnx.to_pure_dict(
        nnx.state(one_block, train.TrainableParam)
    )
    _assert_tree_exact(full_state, explicit_state)
    _assert_tree_exact(full_state, one_block_state)
    assert train.research_state_abi(full_state) == train.research_state_abi(
        one_block_state
    )

    tokens = full.encoder.encode_tokens(_batch()["current_planes"])
    full_output = full.state_projector(tokens)
    explicit_full_output = explicit_full.state_projector(tokens)
    one_block_output = one_block.state_projector(tokens)
    np.testing.assert_array_equal(explicit_full_output, full_output)
    assert not np.array_equal(
        np.asarray(one_block_output),
        np.asarray(full_output),
    )


def test_one_block_output_matches_explicit_block_zero_execution() -> None:
    model = _model(active_layers=1, seed=19)
    projector = model.state_projector
    tokens = model.encoder.encode_tokens(_batch()["current_planes"])
    tokens = jnp.asarray(tokens, dtype=projector.compute_dtype)
    batch_size = tokens.shape[0]
    square_tokens = (
        tokens.reshape((batch_size * 64, projector.input_dim))
        @ jnp.asarray(projector.in_proj[...], dtype=projector.compute_dtype)
        + jnp.asarray(projector.in_bias[...], dtype=projector.compute_dtype)
    ).reshape((batch_size, 64, projector.z_dim))
    cls = jnp.broadcast_to(
        jnp.asarray(projector.cls[...], dtype=projector.compute_dtype),
        (batch_size, 1, projector.z_dim),
    )
    seq = jnp.concatenate([cls, square_tokens], axis=1)
    seq = seq + jnp.asarray(
        projector.pos_embed[...],
        dtype=projector.compute_dtype,
    )[None, :, :]
    params = projector._active_block_params()
    manual = projector.blocks._layer(
        seq,
        tuple(param[0] for param in params),
    )[:, 0, :]

    np.testing.assert_array_equal(projector(tokens), manual)


def test_inactive_projector_block_has_exact_zero_loss_gradient() -> None:
    model = _model(active_layers=1, seed=23)

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
    pure_blocks = nnx.to_pure_dict(gradients)["state_projector"]["blocks"]

    assert jnp.isfinite(loss)
    assert aux["jepa_projector_configured_layers"] == 2.0
    assert aux["jepa_projector_active_layers"] == 1.0
    assert aux["jepa_projector_active_layer_fraction"] == 0.5
    active_squared_norm = 0.0
    for leaf in jax.tree.leaves(pure_blocks):
        assert leaf.shape[0] == 2
        active_squared_norm += float(
            jnp.sum(jnp.square(jnp.asarray(leaf[0], dtype=jnp.float32)))
        )
        np.testing.assert_array_equal(leaf[1], jnp.zeros_like(leaf[1]))
    assert active_squared_norm > 0.0


def test_active_depth_changes_only_jepa_branch_at_fixed_parameters() -> None:
    full = _model(active_layers=0, seed=31)
    one_block = _model(active_layers=1, seed=31)
    args = (_batch(), jax.random.PRNGKey(37), 1.0, 1.0)
    full_loss, full_aux = train.normalized_stage1_loss_fn(
        full,
        *args,
        sample_future_targets=True,
    )
    one_loss, one_aux = train.normalized_stage1_loss_fn(
        one_block,
        *args,
        sample_future_targets=True,
    )

    np.testing.assert_array_equal(
        one_aux["jepa_target_horizon_mask"],
        full_aux["jepa_target_horizon_mask"],
    )
    assert one_aux["dfm_ce_loss"] == full_aux["dfm_ce_loss"]
    assert one_aux["first_legality_loss"] == full_aux["first_legality_loss"]
    assert one_aux["bt4_encoded_boards_per_example"] == 3.0
    assert one_aux["jepa_prediction_horizon_count"] == 4.0
    assert one_aux["jepa_sigreg_valid_count"] == 10.0
    assert one_aux["jepa_pred_sigreg_valid_count"] == 8.0
    assert one_aux["jepa_projector_active_layers"] == 1.0
    assert "jepa_projector_active_layers" not in full_aux
    assert one_aux["jepa_positive_loss"] != full_aux["jepa_positive_loss"]
    assert one_loss != full_loss


def test_full_horizon_evaluation_reports_same_active_depth() -> None:
    model = _model(active_layers=1, seed=41)
    loss, aux = train.eval_normalized_stage1_step(
        model,
        _batch(),
        jax.random.PRNGKey(43),
        1.0,
        1.0,
    )
    assert jnp.isfinite(loss)
    assert aux["jepa_target_sample_count"] == 4.0
    assert aux["jepa_projector_configured_layers"] == 2.0
    assert aux["jepa_projector_active_layers"] == 1.0
    assert aux["jepa_projector_active_layer_fraction"] == 0.5


@pytest.mark.parametrize("active_layers", [-1, 3, True])
def test_invalid_active_depth_fails_before_compilation(active_layers) -> None:
    config = dataclasses.replace(
        _config(),
        jepa_projector_active_layers=active_layers,
    )
    with pytest.raises(ValueError, match="jepa_projector_active_layers"):
        train.validate_objective_config(
            objective="normalized",
            config=config,
        )
    with pytest.raises(ValueError, match="jepa_projector_active_layers"):
        train.JointLatentSASAModel(
            DeterministicEncoder(),
            config,
            rngs=nnx.Rngs(47),
        )


def test_active_depth_serialization_and_resume_contract(monkeypatch) -> None:
    default = _config(jepa_projector_active_layers=0)
    enabled = _config(jepa_projector_active_layers=1)
    assert "jepa_projector_active_layers" not in train.serialized_model_config(
        default
    )
    assert train.serialized_model_config(enabled)[
        "jepa_projector_active_layers"
    ] == 1

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
    semantics = contract["objective"]["jepa_projector_active_depth"]
    assert semantics["configured_parameter_layers"] == 2
    assert semantics["active_forward_layers"] == 1
    assert semantics["active_layer_indices"] == [0]
    assert semantics["inactive_layer_loss_gradient"] == "exact_zero"
    assert semantics["source_model_restore"] == "exact_all_parameters"
    assert semantics["training_evaluation_depth_shared"] is True
    assert semantics["dfm_inference_path_affected"] is False
