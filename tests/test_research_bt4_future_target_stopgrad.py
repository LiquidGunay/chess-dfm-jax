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

import research.train as train  # noqa: E402
from chess_dfm_jax.nnx_bt4 import TrainableParam  # noqa: E402


class TrainableDummyEncoder(nnx.Module):
    def __init__(self, embedding_size: int = 16):
        self.embedding_size = embedding_size
        self.embedding = TrainableParam(
            jnp.linspace(0.25, 1.25, embedding_size, dtype=jnp.float32)
        )

    def encode_tokens(self, planes: jax.Array) -> jax.Array:
        batch_size = planes.shape[0]
        plane_mean = jnp.mean(planes, axis=(1, 2, 3))
        square = jnp.linspace(
            0.0,
            1.0,
            64,
            dtype=jnp.float32,
        )[None, :, None]
        return square + plane_mean.reshape((batch_size, 1, 1)) * self.embedding[
            None,
            None,
            :,
        ]


def _config(*, stop_future: bool) -> train.JointLatentSASAConfig:
    return train.JointLatentSASAConfig(
        token_dim=8,
        z_dim=16,
        projector_layers=1,
        projector_num_heads=4,
        projector_mlp_dim=32,
        jepa_condition_dim=12,
        dfm_layers=1,
        jepa_layers=1,
        jepa_mlp_dim=32,
        num_heads=4,
        mlp_dim=32,
        action_vocab_size=32,
        horizon=2,
        compute_dtype="float32",
        param_dtype="float32",
        encoder_dtype="float32",
        first_legality_coeff=1.0,
        legality_on_masked_only=False,
        jepa_sigreg_coeff=0.01,
        jepa_pred_sigreg_coeff=0.02,
        jepa_sigreg_proj_dim=8,
        jepa_target_sample_count=1,
        jepa_target_sampling_unit="example_balanced",
        use_qk_gain=True,
        use_qk_norm=True,
        use_xsa=True,
        use_muon=False,
        learning_rate=1e-3,
        bt4_learning_rate=1e-4,
        lr_warmup_steps=0,
        grad_clip_norm=0.0,
        skip_nonfinite_updates=False,
        remat_blocks=False,
        scan_layers=False,
        bt4_future_target_stop_gradient=stop_future,
    )


def _batch() -> dict[str, jax.Array]:
    current = jnp.linspace(
        -0.25,
        0.75,
        2 * 112 * 8 * 8,
        dtype=jnp.float32,
    ).reshape((2, 112, 8, 8))
    return {
        "current_planes": current,
        "future_planes": jnp.stack(
            [current + 0.125, current - 0.25],
            axis=1,
        ),
        "action_indices": jnp.asarray([[1, 2], [3, 4]], dtype=jnp.int32),
        "valid": jnp.ones((2,), dtype=jnp.float32),
        "future_valid": jnp.ones((2, 2), dtype=jnp.float32),
        "legal_idx": jnp.asarray(
            [
                [[1, 5, 9, 0], [2, 6, 0, 0]],
                [[3, 7, 11, 15], [4, 8, 12, 0]],
            ],
            dtype=jnp.int32,
        ),
        "legal_count": jnp.asarray([[3, 2], [4, 3]], dtype=jnp.int32),
        "legal_masks_valid": jnp.ones((2, 2), dtype=jnp.float32),
        "deterministic_t": jnp.asarray(0.0, dtype=jnp.float32),
    }


def _make_model(*, stop_future: bool, seed: int = 7):
    return train.JointLatentSASAModel(
        TrainableDummyEncoder(),
        _config(stop_future=stop_future),
        rngs=nnx.Rngs(seed),
    )


def _pure_trainable(module: nnx.Module) -> dict[str, Any]:
    return nnx.to_pure_dict(nnx.state(module, TrainableParam))


def _tree_norm(tree: Any) -> float:
    return float(
        np.sqrt(
            sum(
                np.sum(np.square(np.asarray(leaf, dtype=np.float64)))
                for leaf in jax.tree.leaves(tree)
            )
        )
    )


def _latent_loss(model, current, future, future_only: bool):
    _, vectors = model.encode_current_and_future_tokens_and_vectors(
        current,
        future,
    )
    selected = vectors[:, 1:] if future_only else vectors[:, :1]
    weights = jnp.linspace(
        0.5,
        1.5,
        selected.shape[-1],
        dtype=jnp.float32,
    )
    return jnp.sum(jnp.square(selected.astype(jnp.float32)) * weights)


def test_default_is_exact_and_candidate_changes_only_gradient_routing() -> None:
    baseline = _make_model(stop_future=False, seed=11)
    candidate = _make_model(stop_future=True, seed=11)
    baseline_state = _pure_trainable(baseline)
    candidate_state = _pure_trainable(candidate)
    assert train.research_state_abi(baseline_state) == train.research_state_abi(
        candidate_state
    )

    batch = _batch()
    baseline_tokens, baseline_vectors = (
        baseline.encode_current_and_future_tokens_and_vectors(
            batch["current_planes"],
            batch["future_planes"],
        )
    )
    candidate_tokens, candidate_vectors = (
        candidate.encode_current_and_future_tokens_and_vectors(
            batch["current_planes"],
            batch["future_planes"],
        )
    )
    np.testing.assert_allclose(
        baseline_tokens,
        candidate_tokens,
        rtol=2e-5,
        atol=3e-7,
    )
    np.testing.assert_allclose(
        baseline_vectors,
        candidate_vectors,
        rtol=2e-5,
        atol=3e-6,
    )

    baseline_optimizer = train.create_joint_optimizer(
        baseline,
        baseline.config,
    )
    candidate_optimizer = train.create_joint_optimizer(
        candidate,
        candidate.config,
    )
    assert train.research_state_abi(
        nnx.state(baseline_optimizer.opt_state)
    ) == train.research_state_abi(nnx.state(candidate_optimizer.opt_state))


def test_future_bt4_gradient_is_zero_but_projector_stays_attached() -> None:
    batch = _batch()
    candidate = _make_model(stop_future=True, seed=17)
    baseline = _make_model(stop_future=False, seed=17)
    grad_fn = nnx.grad(
        _latent_loss,
        argnums=nnx.DiffState(0, TrainableParam),
    )

    candidate_future = nnx.to_pure_dict(
        grad_fn(
            candidate,
            batch["current_planes"],
            batch["future_planes"],
            True,
        )
    )
    baseline_future = nnx.to_pure_dict(
        grad_fn(
            baseline,
            batch["current_planes"],
            batch["future_planes"],
            True,
        )
    )
    candidate_current = nnx.to_pure_dict(
        grad_fn(
            candidate,
            batch["current_planes"],
            batch["future_planes"],
            False,
        )
    )

    assert _tree_norm(candidate_future["encoder"]) == 0.0
    assert _tree_norm(candidate_future["state_projector"]) > 0.0
    assert _tree_norm(baseline_future["encoder"]) > 0.0
    assert _tree_norm(candidate_current["encoder"]) > 0.0


def test_unchunked_fused_path_preserves_values_and_both_encoder_gradients() -> None:
    batch = _batch()
    scanned = train.JointLatentSASAModel(
        TrainableDummyEncoder(),
        dataclasses.replace(
            _config(stop_future=False),
            bt4_encode_chunk_size=1,
        ),
        rngs=nnx.Rngs(29),
    )
    fused = _make_model(stop_future=False, seed=29)

    scanned_tokens, scanned_vectors = (
        scanned.encode_current_and_future_tokens_and_vectors(
            batch["current_planes"],
            batch["future_planes"],
        )
    )
    fused_tokens, fused_vectors = (
        fused.encode_current_and_future_tokens_and_vectors(
            batch["current_planes"],
            batch["future_planes"],
        )
    )
    np.testing.assert_allclose(scanned_tokens, fused_tokens, rtol=2e-5, atol=3e-7)
    np.testing.assert_allclose(scanned_vectors, fused_vectors, rtol=2e-5, atol=3e-6)

    grad_fn = nnx.grad(
        _latent_loss,
        argnums=nnx.DiffState(0, TrainableParam),
    )
    fused_future = nnx.to_pure_dict(
        grad_fn(
            fused,
            batch["current_planes"],
            batch["future_planes"],
            True,
        )
    )
    fused_current = nnx.to_pure_dict(
        grad_fn(
            fused,
            batch["current_planes"],
            batch["future_planes"],
            False,
        )
    )
    assert _tree_norm(fused_future["encoder"]) > 0.0
    assert _tree_norm(fused_current["encoder"]) > 0.0

    scanned_loss, scanned_aux = train.normalized_stage1_loss_fn(
        scanned,
        batch,
        jax.random.PRNGKey(31),
        1.0,
        1.0,
        sample_future_targets=True,
    )
    fused_loss, fused_aux = train.normalized_stage1_loss_fn(
        fused,
        batch,
        jax.random.PRNGKey(31),
        1.0,
        1.0,
        sample_future_targets=True,
    )
    np.testing.assert_allclose(scanned_loss, fused_loss, rtol=1e-6)
    np.testing.assert_allclose(
        scanned_aux["dfm_ce_loss"],
        fused_aux["dfm_ce_loss"],
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        scanned_aux["first_legality_loss"],
        fused_aux["first_legality_loss"],
        rtol=1e-6,
    )

    scanned_optimizer = train.create_joint_optimizer(scanned, scanned.config)
    fused_optimizer = train.create_joint_optimizer(fused, fused.config)
    assert train.research_state_abi(
        nnx.state(scanned_optimizer.opt_state)
    ) == train.research_state_abi(nnx.state(fused_optimizer.opt_state))


def test_training_reports_routing_and_eval_keeps_full_horizon() -> None:
    candidate = _make_model(stop_future=True, seed=23)
    batch = _batch()
    candidate_loss, candidate_aux = train.normalized_stage1_loss_fn(
        candidate,
        batch,
        jax.random.PRNGKey(5),
        1.0,
        1.0,
        sample_future_targets=True,
    )
    assert jnp.isfinite(candidate_loss)
    assert candidate_aux["bt4_encoded_boards_per_example"] == 2.0
    assert candidate_aux["bt4_trainable_encoded_boards_per_example"] == 1.0
    assert candidate_aux["bt4_stop_gradient_encoded_boards_per_example"] == 1.0
    assert candidate_aux["bt4_future_target_stop_gradient"] == 1.0
    np.testing.assert_array_equal(
        candidate_aux["jepa_target_assignment_count_by_horizon"],
        jnp.asarray([1.0, 1.0], dtype=jnp.float32),
    )

    eval_loss, eval_aux = train.normalized_stage1_loss_fn(
        candidate,
        batch,
        jax.random.PRNGKey(5),
        1.0,
        1.0,
    )
    assert jnp.isfinite(eval_loss)
    assert eval_aux["jepa_target_sample_count"] == 2.0
    assert "jepa_target_sampling_active" not in eval_aux

    baseline = _make_model(stop_future=False, seed=23)
    baseline_loss, baseline_aux = train.normalized_stage1_loss_fn(
        baseline,
        batch,
        jax.random.PRNGKey(5),
        1.0,
        1.0,
        sample_future_targets=True,
    )
    np.testing.assert_allclose(candidate_loss, baseline_loss, rtol=1e-6)
    np.testing.assert_allclose(
        candidate_aux["dfm_ce_loss"],
        baseline_aux["dfm_ce_loss"],
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        candidate_aux["first_legality_loss"],
        baseline_aux["first_legality_loss"],
        rtol=1e-6,
    )


@pytest.mark.parametrize(
    ("updates", "field"),
    [
        ({"bt4_freeze_backbone": True}, "bt4_freeze_backbone"),
        ({"unfreeze_bt4_encoder": False}, "unfreeze_bt4_encoder"),
        ({"jepa_target_semantics": "ema"}, "jepa_target_semantics"),
        ({"jepa_target_mode": "current_repeat"}, "jepa_target_mode"),
        ({"jepa_target_sample_count": 2}, "jepa_target_sample_count"),
        ({"jepa_target_sampling_unit": "batch_shared"}, "jepa_target_sampling_unit"),
        ({"bt4_encode_chunk_size": 1}, "bt4_encode_chunk_size"),
        ({"jepa_sampled_target_anchors": True}, "jepa_sampled_target_anchors"),
    ],
)
def test_candidate_config_fails_closed(updates, field) -> None:
    config = dataclasses.replace(_config(stop_future=True), **updates)
    with pytest.raises(ValueError, match=field):
        train.validate_bt4_future_target_stop_gradient_config(config)


def test_serialization_and_resume_contract_are_explicit(monkeypatch) -> None:
    default = _config(stop_future=False)
    enabled = _config(stop_future=True)
    assert "bt4_future_target_stop_gradient" not in train.serialized_model_config(
        default
    )
    assert train.serialized_model_config(enabled)[
        "bt4_future_target_stop_gradient"
    ] is True

    monkeypatch.setattr(train, "require_within_workspace", lambda path: Path(path))
    monkeypatch.setattr(
        train,
        "load_asset_manifest",
        lambda: {
            "trajectory_v3": {
                "archive": {"sha256": "trajectory-digest", "size_bytes": 123}
            }
        },
    )
    monkeypatch.setattr(train, "sha256_file", lambda _path: "test-digest")
    contract = train.build_research_resume_contract(
        config=enabled,
        objective="normalized",
        sigreg_reference_count=1.0,
        batch_size=2,
        train_seed=7,
        train_provenance={"kind": "test"},
        models_dir=REPO_ROOT / "models",
    )
    routing = contract["objective"]["bt4_future_target_stop_gradient"]
    assert routing["encoder_calls_per_step"] == 2
    assert routing["trainable_current_boards_per_example"] == 1
    assert routing["stop_gradient_future_boards_per_example"] == 1
    assert routing["future_bt4_token_gradient"] == "exact_zero"
    assert routing["future_state_projector_gradient"] == "attached"
    assert routing["optimizer_state_abi"] == "unchanged_source_compatible"
    assert routing["evaluation_horizons"] == "all"
    assert routing["inference_affected"] is False
    assert "bt4_backbone_freeze" not in contract["optimizer"]
