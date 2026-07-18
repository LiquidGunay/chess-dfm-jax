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


def test_local_config_and_initialized_model_match_legacy_exactly():
    local_fields = [
        (field.name, field.default)
        for field in dataclasses.fields(local.JointLatentSASAConfig)
    ]
    legacy_fields = [
        (field.name, field.default)
        for field in dataclasses.fields(legacy.JointLatentSASAConfig)
    ]
    assert local_fields == legacy_fields

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
