import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chess_dfm_jax.training.dfm import DFMConfig, DFMDenoiser  # noqa: E402
from chess_dfm_jax.training.jepa import JEPAConfig, LC0JEPA  # noqa: E402
from chess_dfm_jax.data.trajectory import build_synthetic_trajectory_shard  # noqa: E402
from chess_dfm_jax.data.trajectory import trajectory_joint_batch_from_npz  # noqa: E402
from chess_dfm_jax.nnx_bt4 import TrainableParam  # noqa: E402
from chess_dfm_jax.training.joint_latent_sasa import (  # noqa: E402
    JointLatentSASAConfig,
    JointLatentSASAModel,
    joint_coupling_gradient_diagnostics,
    joint_jepa_action_baseline_diagnostics,
    sample_legal_prefix_candidates,
    train_joint_stage1_step,
    train_joint_stage2_step,
)


class DummyEncoder(nnx.Module):
    def __init__(self, embedding_size: int = 16):
        self.embedding_size = embedding_size

    def encode_tokens(self, planes):
        batch_size = planes.shape[0]
        plane_mean = jnp.mean(planes, axis=(1, 2, 3), keepdims=False)
        square = jnp.linspace(0.0, 1.0, 64, dtype=jnp.float32)[None, :, None]
        feature = jnp.linspace(0.1, 1.0, self.embedding_size, dtype=jnp.float32)[None, None, :]
        return square + plane_mean.reshape((batch_size, 1, 1)) * feature


def test_dfm_planner_from_latents_returns_hidden_and_preserves_call_path():
    config = DFMConfig(
        token_dim=16,
        num_layers=1,
        num_heads=4,
        mlp_dim=32,
        horizon=4,
        compute_dtype="float32",
    )
    model = DFMDenoiser(DummyEncoder(), config, rngs=nnx.Rngs(0))
    current_planes = jnp.ones((2, 112, 8, 8), dtype=jnp.float32)
    noisy_actions = jnp.asarray([[1, 2, 1858, 4], [5, 1858, 7, 8]], dtype=jnp.int32)
    t = jnp.asarray([0.25, 0.75], dtype=jnp.float32)

    z_dfm = model.encode_current(current_planes)
    logits, hidden = model.planner_from_latents(z_dfm, noisy_actions, t, return_hidden=True)
    call_logits = model(current_planes, noisy_actions, t)

    assert z_dfm.shape == (2, 64, config.token_dim)
    assert logits.shape == (2, 4, config.action_vocab_size)
    assert hidden["state_tokens"].shape == (2, 64, config.token_dim)
    assert hidden["action_tokens"].shape == (2, 4, config.token_dim)
    np.testing.assert_allclose(np.asarray(logits), np.asarray(call_logits), rtol=1e-5, atol=1e-5)


def test_jepa_rollout_from_latents_consumes_action_hidden():
    config = JEPAConfig(
        token_dim=8,
        num_layers=1,
        num_heads=4,
        mlp_dim=32,
        head_param_dtype="float32",
        head_compute_dtype="float32",
    )
    model = LC0JEPA(DummyEncoder(), config, rngs=nnx.Rngs(1))
    current_planes = jnp.ones((2, 112, 8, 8), dtype=jnp.float32)
    actions = jnp.asarray([[1, 2, 3], [4, 5, 6]], dtype=jnp.int32)
    z0_jepa = model.encode_state_tokens(current_planes)

    zero_hidden = jnp.zeros((2, 3, model.transition.jepa_width), dtype=jnp.float32)
    shifted_hidden = jnp.ones((2, 3, model.transition.jepa_width), dtype=jnp.float32) * 0.25
    pred_zero = model.jepa_rollout_from_latents(z0_jepa, actions, zero_hidden)
    pred_shifted = model.jepa_rollout_from_latents(z0_jepa, actions, shifted_hidden)
    pred_standalone = model.predict_sequence(current_planes, actions)

    assert z0_jepa.shape == (2, 64, DummyEncoder().embedding_size)
    assert pred_zero.shape == (2, 3, 64, DummyEncoder().embedding_size)
    assert pred_shifted.shape == (2, 3, 64, DummyEncoder().embedding_size)
    np.testing.assert_allclose(np.asarray(pred_zero), np.asarray(pred_standalone), rtol=1e-5, atol=1e-5)
    assert float(jnp.max(jnp.abs(pred_zero - pred_shifted))) > 1e-6


def test_joint_stage1_step_uses_compact_legal_batch():
    shard = build_synthetic_trajectory_shard(batch_size=2, horizon=2)
    payload = {
        "schema_version": np.asarray("trajectory-v2"),
        "planes_t": shard.planes_t,
        "actions": shard.actions,
        "planes_future": shard.planes_future,
        "future_valid": shard.future_valid,
        "legal_masks": shard.legal_masks,
    }
    batch_np = trajectory_joint_batch_from_npz(payload, horizon=2)
    batch = {key: jnp.asarray(value) for key, value in batch_np.items()}
    batch["deterministic_t"] = jnp.asarray(0.0, dtype=jnp.float32)

    config = JointLatentSASAConfig(
        token_dim=8,
        dfm_layers=1,
        jepa_layers=1,
        num_heads=4,
        mlp_dim=32,
        horizon=2,
        compute_dtype="float32",
        param_dtype="float32",
        encoder_dtype="float32",
        first_legality_coeff=1.0,
        horizon_legality_coeff=0.25,
        jepa_sigreg_coeff=0.01,
        jepa_action_contrast_coeff=0.1,
    )
    model = JointLatentSASAModel(DummyEncoder(), config, rngs=nnx.Rngs(2))
    optimizer = nnx.Optimizer(model, optax.adamw(1e-3), wrt=TrainableParam)
    z_jepa = model.encode_current_jepa(batch["current_planes"])
    target_tokens = model.encode_future_targets(batch["future_planes"])

    assert z_jepa.shape == (2, 64, DummyEncoder().embedding_size)
    assert target_tokens.shape == (2, 2, 64, DummyEncoder().embedding_size)

    loss, aux = train_joint_stage1_step(model, optimizer, batch, jnp.asarray([0, 1], dtype=jnp.uint32))

    assert jnp.isfinite(loss)
    assert jnp.isfinite(aux["dfm_ce_loss"])
    assert jnp.isfinite(aux["legality_loss"])
    assert jnp.isfinite(aux["first_legality_loss"])
    assert jnp.isfinite(aux["horizon_legality_loss"])
    assert jnp.isfinite(aux["jepa_raw_mse"])
    assert jnp.isclose(aux["jepa_positive_loss"], aux["jepa_raw_mse"], rtol=1e-5, atol=1e-5)
    assert jnp.isfinite(aux["jepa_cosine_loss"])
    assert jnp.isfinite(aux["jepa_sigreg_loss"])
    assert jnp.isfinite(aux["jepa_action_contrast_loss"])
    assert jnp.isfinite(aux["jepa_true_minus_shuffled"])
    assert jnp.isfinite(aux["jepa_positive_loss"])
    assert aux["loss_horizon"] == 2.0

    diagnostics = joint_coupling_gradient_diagnostics(model, batch)
    assert diagnostics["coupling_grad_norm_dfm_path"] > 0
    assert diagnostics["coupling_grad_norm_shared_projector"] > 0
    assert diagnostics["coupling_grad_norm_jepa_path"] > 0
    assert diagnostics["coupling_grad_norm_target_projector"] == 0
    baseline_diag = joint_jepa_action_baseline_diagnostics(model, batch, jnp.asarray([3, 4], dtype=jnp.uint32))
    assert jnp.isfinite(baseline_diag["jepa_diag_true_loss"])
    assert jnp.isfinite(baseline_diag["jepa_diag_shuffled_loss"])
    assert jnp.isfinite(baseline_diag["jepa_diag_identity_loss"])


def test_joint_stage1_step_supports_sampled_jepa_targets():
    shard = build_synthetic_trajectory_shard(batch_size=2, horizon=3)
    payload = {
        "schema_version": np.asarray("trajectory-v2"),
        "planes_t": shard.planes_t,
        "actions": shard.actions,
        "planes_future": shard.planes_future,
        "future_valid": shard.future_valid,
        "legal_masks": shard.legal_masks,
    }
    batch_np = trajectory_joint_batch_from_npz(payload, horizon=3)
    batch = {key: jnp.asarray(value) for key, value in batch_np.items()}
    batch["deterministic_t"] = jnp.asarray(0.0, dtype=jnp.float32)

    config = JointLatentSASAConfig(
        token_dim=8,
        dfm_layers=1,
        jepa_layers=1,
        num_heads=4,
        mlp_dim=32,
        horizon=3,
        compute_dtype="float32",
        param_dtype="float32",
        encoder_dtype="float32",
        first_legality_coeff=1.0,
        jepa_target_sample_count=1,
    )
    model = JointLatentSASAModel(DummyEncoder(), config, rngs=nnx.Rngs(4))
    optimizer = nnx.Optimizer(model, optax.adamw(1e-3), wrt=TrainableParam)

    loss, aux = train_joint_stage1_step(model, optimizer, batch, jnp.asarray([9, 10], dtype=jnp.uint32))

    assert jnp.isfinite(loss)
    assert jnp.isfinite(aux["jepa_positive_loss"])
    assert aux["jepa_target_sample_count"] == 1.0
    assert jnp.isclose(aux["jepa_target_sample_fraction"], 1.0 / 3.0)
    assert jnp.sum(aux["jepa_target_horizon_mask"]) == 1.0
    assert aux["jepa_loss_by_horizon"].shape == (3,)


def test_joint_stage2_step_adds_contrastive_metrics():
    shard = build_synthetic_trajectory_shard(batch_size=2, horizon=3)
    payload = {
        "schema_version": np.asarray("trajectory-v2"),
        "planes_t": shard.planes_t,
        "actions": shard.actions,
        "planes_future": shard.planes_future,
        "future_valid": shard.future_valid,
        "legal_masks": shard.legal_masks,
    }
    batch_np = trajectory_joint_batch_from_npz(payload, horizon=3)
    batch = {key: jnp.asarray(value) for key, value in batch_np.items()}
    batch["deterministic_t"] = jnp.asarray(0.0, dtype=jnp.float32)

    config = JointLatentSASAConfig(
        token_dim=8,
        dfm_layers=1,
        jepa_layers=1,
        num_heads=4,
        mlp_dim=32,
        horizon=3,
        compute_dtype="float32",
        param_dtype="float32",
        encoder_dtype="float32",
        first_legality_coeff=1.0,
        horizon_legality_coeff=0.25,
        contrastive_coeff=0.5,
        contrastive_temperature=0.2,
        candidate_count=3,
    )
    model = JointLatentSASAModel(DummyEncoder(), config, rngs=nnx.Rngs(3))
    optimizer = nnx.Optimizer(model, optax.adamw(1e-3), wrt=TrainableParam)

    loss, aux = train_joint_stage2_step(model, optimizer, batch, jnp.asarray([1, 2], dtype=jnp.uint32))

    assert jnp.isfinite(loss)
    assert jnp.isfinite(aux["stage1_loss"])
    assert jnp.isfinite(aux["contrastive_loss"])
    assert jnp.isfinite(aux["contrastive_accuracy"])
    assert aux["contrastive_valid_fraction"] > 0


def test_sample_legal_prefix_candidates_preserves_prefix_and_masks_suffix():
    actions = jnp.asarray([[10, 20, 30], [1, 2, 3]], dtype=jnp.int32)
    legal_idx = jnp.asarray(
        [
            [[10, 11, 12], [20, 21, 22], [30, 31, 32]],
            [[1, 4, 5], [2, 6, 7], [3, 8, 9]],
        ],
        dtype=jnp.int32,
    )
    legal_count = jnp.full((2, 3), 3, dtype=jnp.int32)

    out = sample_legal_prefix_candidates(
        actions,
        legal_idx,
        legal_count,
        jnp.asarray([123, 456], dtype=jnp.uint32),
        candidate_count=3,
        mask_token_id=1858,
    )

    candidates = np.asarray(out["candidates"])
    eval_horizon = np.asarray(out["eval_horizon"])
    candidate_valid = np.asarray(out["candidate_valid"])
    assert candidates.shape == (2, 3, 3)
    np.testing.assert_array_equal(candidates[:, 0], np.asarray(actions))
    assert np.all(candidate_valid == 1.0)

    for batch_idx in range(candidates.shape[0]):
        for cand_idx in range(1, candidates.shape[1]):
            h = int(eval_horizon[batch_idx, cand_idx])
            np.testing.assert_array_equal(candidates[batch_idx, cand_idx, :h], np.asarray(actions)[batch_idx, :h])
            assert candidates[batch_idx, cand_idx, h] != int(actions[batch_idx, h])
            assert candidates[batch_idx, cand_idx, h] in np.asarray(legal_idx)[batch_idx, h]
            assert np.all(candidates[batch_idx, cand_idx, h + 1 :] == 1858)


def test_sample_legal_prefix_candidates_masks_unavailable_alternative():
    actions = jnp.asarray([[10]], dtype=jnp.int32)
    legal_idx = jnp.asarray([[[10, 65535]]], dtype=jnp.int32)
    legal_count = jnp.asarray([[1]], dtype=jnp.int32)

    out = sample_legal_prefix_candidates(
        actions,
        legal_idx,
        legal_count,
        jnp.asarray([0, 1], dtype=jnp.uint32),
        candidate_count=2,
        mask_token_id=1858,
    )

    np.testing.assert_array_equal(np.asarray(out["candidates"])[:, 0], np.asarray(actions))
    assert float(out["candidate_valid"][0, 0]) == 1.0
    assert float(out["candidate_valid"][0, 1]) == 0.0
