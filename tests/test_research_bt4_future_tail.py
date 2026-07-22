from __future__ import annotations

import dataclasses
import math
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


class DummyBT4Embedding(nnx.Module):
    def __init__(self, width: int):
        self.scale = TrainableParam(
            jnp.linspace(0.5, 1.5, width, dtype=jnp.float32)
        )

    def __call__(
        self,
        planes: jax.Array,
        alpha: float,
    ) -> tuple[jax.Array, int]:
        batch_size = planes.shape[0]
        plane_mean = jnp.mean(planes, axis=(1, 2, 3))
        squares = jnp.linspace(-0.5, 0.5, 64, dtype=jnp.float32)
        tokens = (
            plane_mean[:, None, None] * self.scale[None, None, :]
            + squares[None, :, None]
            + jnp.asarray(alpha, dtype=jnp.float32) * 0.01
        )
        return tokens.reshape((-1, self.scale.shape[0])), batch_size


class DummyBT4Layer(nnx.Module):
    def __init__(self, width: int, layer_index: int):
        self.scale = TrainableParam(
            jnp.linspace(
                0.02 + layer_index * 0.001,
                0.04 + layer_index * 0.001,
                width,
                dtype=jnp.float32,
            )
        )
        self.bias = TrainableParam(
            jnp.full((width,), 0.001 * (layer_index + 1), dtype=jnp.float32)
        )

    def __call__(self, tokens: jax.Array, alpha: float) -> jax.Array:
        residual = jnp.tanh(
            tokens * self.scale[None, None, :] + self.bias[None, None, :]
        )
        return tokens + jnp.asarray(alpha, dtype=jnp.float32) * residual


class LayeredDummyBT4(nnx.Module):
    def __init__(self, width: int = 16, layer_count: int = 15):
        self.embedding_size = width
        self.embedding = DummyBT4Embedding(width)
        self.layers = nnx.List(
            [DummyBT4Layer(width, index) for index in range(layer_count)]
        )

    def encode_tokens(self, planes: jax.Array) -> jax.Array:
        alpha = float(math.pow(2.0 * len(self.layers), -0.25))
        tokens, batch_size = self.embedding(planes, alpha)
        tokens = tokens.reshape((batch_size, 64, self.embedding_size))
        for layer in self.layers:
            tokens = layer(tokens, alpha)
        return tokens


def _config(
    *,
    stop_future: bool,
    trainable_tail_layers: int = 0,
) -> train.JointLatentSASAConfig:
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
        jepa_norm_loss_coeff=0.0,
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
        bt4_future_target_trainable_tail_layers=trainable_tail_layers,
    )


def _model(*, stop_future: bool, tail_layers: int, seed: int = 7):
    return train.JointLatentSASAModel(
        LayeredDummyBT4(),
        _config(
            stop_future=stop_future,
            trainable_tail_layers=tail_layers,
        ),
        rngs=nnx.Rngs(seed),
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


def _tree_norm(tree: Any) -> float:
    return float(
        np.sqrt(
            sum(
                np.sum(np.square(np.asarray(leaf, dtype=np.float64)))
                for leaf in jax.tree.leaves(tree)
            )
        )
    )


def _selected_vector_loss(model, current, future, future_only: bool):
    _, vectors = model.encode_current_and_future_tokens_and_vectors(
        current,
        future,
    )
    selected = vectors[:, 1:] if future_only else vectors[:, :1]
    weights = jnp.linspace(0.5, 1.5, selected.shape[-1], dtype=jnp.float32)
    return jnp.sum(jnp.square(selected.astype(jnp.float32)) * weights)


def test_tail_preserves_forward_and_model_optimizer_abi() -> None:
    full_gradient = _model(stop_future=False, tail_layers=0, seed=11)
    tail = _model(stop_future=True, tail_layers=3, seed=11)
    batch = _batch()

    full_tokens, full_vectors = (
        full_gradient.encode_current_and_future_tokens_and_vectors(
            batch["current_planes"],
            batch["future_planes"],
        )
    )
    tail_tokens, tail_vectors = tail.encode_current_and_future_tokens_and_vectors(
        batch["current_planes"],
        batch["future_planes"],
    )
    np.testing.assert_allclose(tail_tokens, full_tokens, rtol=2e-5, atol=3e-7)
    np.testing.assert_allclose(tail_vectors, full_vectors, rtol=2e-5, atol=3e-6)

    assert train.research_state_abi(
        nnx.state(full_gradient, TrainableParam)
    ) == train.research_state_abi(nnx.state(tail, TrainableParam))
    full_optimizer = train.create_joint_optimizer(
        full_gradient,
        full_gradient.config,
    )
    tail_optimizer = train.create_joint_optimizer(tail, tail.config)
    assert train.research_state_abi(
        nnx.state(full_optimizer.opt_state)
    ) == train.research_state_abi(nnx.state(tail_optimizer.opt_state))


def test_future_gradient_reaches_only_last_three_blocks_and_projector() -> None:
    batch = _batch()
    tail = _model(stop_future=True, tail_layers=3, seed=17)
    grad_fn = nnx.grad(
        _selected_vector_loss,
        argnums=nnx.DiffState(0, TrainableParam),
    )
    future_grads = nnx.to_pure_dict(
        grad_fn(
            tail,
            batch["current_planes"],
            batch["future_planes"],
            True,
        )
    )
    current_grads = nnx.to_pure_dict(
        grad_fn(
            tail,
            batch["current_planes"],
            batch["future_planes"],
            False,
        )
    )

    assert _tree_norm(future_grads["encoder"]["embedding"]) == 0.0
    for layer_index in range(12):
        assert _tree_norm(future_grads["encoder"]["layers"][layer_index]) == 0.0
    for layer_index in range(12, 15):
        assert _tree_norm(future_grads["encoder"]["layers"][layer_index]) > 0.0
    assert _tree_norm(future_grads["state_projector"]) > 0.0

    assert _tree_norm(current_grads["encoder"]["embedding"]) > 0.0
    for layer_index in range(15):
        assert _tree_norm(current_grads["encoder"]["layers"][layer_index]) > 0.0


def test_future_gradient_reaches_only_final_block_for_one_layer_tail() -> None:
    batch = _batch()
    tail = _model(stop_future=True, tail_layers=1, seed=19)
    grad_fn = nnx.grad(
        _selected_vector_loss,
        argnums=nnx.DiffState(0, TrainableParam),
    )
    future_grads = nnx.to_pure_dict(
        grad_fn(
            tail,
            batch["current_planes"],
            batch["future_planes"],
            True,
        )
    )
    current_grads = nnx.to_pure_dict(
        grad_fn(
            tail,
            batch["current_planes"],
            batch["future_planes"],
            False,
        )
    )

    assert _tree_norm(future_grads["encoder"]["embedding"]) == 0.0
    for layer_index in range(14):
        assert _tree_norm(future_grads["encoder"]["layers"][layer_index]) == 0.0
    assert _tree_norm(future_grads["encoder"]["layers"][14]) > 0.0
    assert _tree_norm(future_grads["state_projector"]) > 0.0

    assert _tree_norm(current_grads["encoder"]["embedding"]) > 0.0
    for layer_index in range(15):
        assert _tree_norm(current_grads["encoder"]["layers"][layer_index]) > 0.0


@pytest.mark.parametrize(
    ("tail_layers", "prefix_layers"),
    [(1, 14), (3, 12)],
)
def test_training_reports_partial_future_route_and_keeps_full_eval_horizon(
    tail_layers: int,
    prefix_layers: int,
) -> None:
    model = _model(stop_future=True, tail_layers=tail_layers, seed=23)
    batch = _batch()
    loss, aux = train.normalized_stage1_loss_fn(
        model,
        batch,
        jax.random.PRNGKey(5),
        1.0,
        1.0,
        sample_future_targets=True,
    )
    assert jnp.isfinite(loss)
    assert aux["bt4_encoded_boards_per_example"] == 2.0
    assert aux["bt4_full_gradient_encoded_boards_per_example"] == 1.0
    assert aux["bt4_partial_gradient_encoded_boards_per_example"] == 1.0
    assert aux["bt4_stop_gradient_encoded_boards_per_example"] == 0.0
    assert aux["bt4_trainable_encoded_boards_per_example"] == 2.0
    assert aux["bt4_future_target_trainable_tail_layers"] == tail_layers
    assert aux["bt4_future_target_detached_prefix_layers"] == prefix_layers

    eval_loss, eval_aux = train.normalized_stage1_loss_fn(
        model,
        batch,
        jax.random.PRNGKey(5),
        1.0,
        1.0,
    )
    assert jnp.isfinite(eval_loss)
    assert eval_aux["jepa_target_sample_count"] == 2.0
    assert "jepa_target_sampling_active" not in eval_aux


@pytest.mark.parametrize("tail_layers", [-1, 16, True, 1.5])
def test_tail_depth_fails_closed(tail_layers) -> None:
    config = dataclasses.replace(
        _config(stop_future=True),
        bt4_future_target_trainable_tail_layers=tail_layers,
    )
    with pytest.raises(
        ValueError,
        match="bt4_future_target_trainable_tail_layers",
    ):
        train.validate_bt4_future_target_stop_gradient_config(config)


def test_nonzero_tail_requires_asymmetric_future_mode() -> None:
    config = _config(stop_future=False, trainable_tail_layers=3)
    with pytest.raises(ValueError, match="bt4_future_target_stop_gradient=True"):
        train.validate_bt4_future_target_stop_gradient_config(config)


def test_tail_serialization_and_resume_contract_are_explicit(monkeypatch) -> None:
    default = _config(stop_future=False)
    enabled = _config(stop_future=True, trainable_tail_layers=3)
    assert "bt4_future_target_trainable_tail_layers" not in (
        train.serialized_model_config(default)
    )
    assert train.serialized_model_config(enabled)[
        "bt4_future_target_trainable_tail_layers"
    ] == 3

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
    assert routing["future_detached_prefix_layers"] == 12
    assert routing["future_trainable_tail_layers"] == 3
    assert routing["stop_gradient_future_boards_per_example"] == 0
    assert routing["partial_gradient_future_boards_per_example"] == 1
    assert routing["future_embedding_gradient"] == "exact_zero"
    assert routing["future_prefix_block_gradient"] == "exact_zero"
    assert routing["future_tail_block_gradient"] == "attached"
    assert routing["future_bt4_token_gradient"] == "attached_through_tail"
    assert routing["future_state_projector_gradient"] == "attached"
    assert routing["optimizer_state_abi"] == "unchanged_source_compatible"
    assert routing["evaluation_horizons"] == "all"
    assert routing["inference_affected"] is False
