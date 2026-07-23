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
from chess_dfm_jax.nnx_bt4 import (  # noqa: E402
    BT4TrainableParam,
    TrainableParam,
)


class DummyBT4Embedding(nnx.Module):
    def __init__(self, width: int):
        self.scale = BT4TrainableParam(
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
        self.scale = BT4TrainableParam(
            jnp.linspace(
                0.02 + layer_index * 0.001,
                0.04 + layer_index * 0.001,
                width,
                dtype=jnp.float32,
            )
        )
        self.bias = BT4TrainableParam(
            jnp.full((width,), 0.001 * (layer_index + 1), dtype=jnp.float32)
        )

    def __call__(self, tokens: jax.Array, alpha: float) -> jax.Array:
        residual = jnp.tanh(
            tokens * self.scale[None, None, :] + self.bias[None, None, :]
        )
        return tokens + jnp.asarray(alpha, dtype=jnp.float32) * residual


class DummyFixedHead(nnx.Module):
    def __init__(self):
        self.weight = nnx.Param(jnp.ones((3,), dtype=jnp.float32))


class LayeredDummyBT4(nnx.Module):
    def __init__(self, width: int = 16, layer_count: int = 15):
        self.embedding_size = width
        self.embedding = DummyBT4Embedding(width)
        self.layers = nnx.List(
            [DummyBT4Layer(width, index) for index in range(layer_count)]
        )
        self.policy_head = DummyFixedHead()
        self.value_head = DummyFixedHead()
        self.moves_left_head = DummyFixedHead()

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


def _model_from_config(
    config: train.JointLatentSASAConfig,
    *,
    seed: int,
) -> train.JointLatentSASAModel:
    return train.JointLatentSASAModel(
        LayeredDummyBT4(),
        config,
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


def _path_arrays(tree: Any) -> dict[str, np.ndarray]:
    return {
        jax.tree_util.keystr(path): np.asarray(leaf)
        for path, leaf in jax.tree_util.tree_flatten_with_path(tree)[0]
    }


def _tree_relative_l2_and_cosine(
    left: Any,
    right: Any,
) -> tuple[float, float]:
    left_arrays = _path_arrays(left)
    right_arrays = _path_arrays(right)
    assert left_arrays.keys() == right_arrays.keys()
    left_norm_sq = 0.0
    right_norm_sq = 0.0
    diff_norm_sq = 0.0
    dot = 0.0
    for path in left_arrays:
        left_value = left_arrays[path].astype(np.float64)
        right_value = right_arrays[path].astype(np.float64)
        assert left_value.shape == right_value.shape
        assert left_arrays[path].dtype == right_arrays[path].dtype
        left_norm_sq += float(np.sum(np.square(left_value)))
        right_norm_sq += float(np.sum(np.square(right_value)))
        diff_norm_sq += float(np.sum(np.square(left_value - right_value)))
        dot += float(np.sum(left_value * right_value))
    relative_l2 = math.sqrt(diff_norm_sq) / max(
        math.sqrt(left_norm_sq),
        1e-30,
    )
    cosine = dot / max(
        math.sqrt(left_norm_sq * right_norm_sq),
        1e-30,
    )
    return relative_l2, cosine


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


def test_split_model_views_partition_and_share_canonical_variables() -> None:
    model = _model(stop_future=True, tail_layers=1, seed=37)
    full_before = train.split_state_abstract_records(
        nnx.state(model, TrainableParam)
    )
    views = train.build_split_model_views(model)
    report = train.validate_split_model_views(model, views)

    assert not hasattr(views.head, "encoder")
    assert hasattr(model.encoder, "policy_head")
    assert not hasattr(views.encoder.encoder, "policy_head")
    assert not hasattr(views.encoder.encoder, "value_head")
    assert not hasattr(views.encoder.encoder, "moves_left_head")
    assert set(
        nnx.to_pure_dict(
            nnx.state(views.encoder, TrainableParam)
        )
    ) == {"encoder"}
    assert "encoder" not in nnx.to_pure_dict(
        nnx.state(views.head, TrainableParam)
    )
    assert views.head.out_bias is model.out_bias
    assert (
        views.encoder.encoder.embedding.scale
        is model.encoder.embedding.scale
    )
    assert report["shared_variable_objects"] is True
    assert report["copied_array_storage"] is False
    assert report["head_leaf_count"] + report["encoder_leaf_count"] == (
        report["full_leaf_count"]
    )
    assert report["nbytes"]["head"] + report["nbytes"]["encoder"] == (
        report["nbytes"]["full"]
    )
    assert train.split_state_abstract_records(
        nnx.state(model, TrainableParam)
    ) == full_before


def test_split_model_view_validation_rejects_copied_variables() -> None:
    model = _model(stop_future=True, tail_layers=1, seed=39)
    views = train.build_split_model_views(model)
    copied_head = nnx.clone(model, variables=True)
    del copied_head.encoder

    with pytest.raises(ValueError, match="copied"):
        train.validate_split_model_views(
            model,
            train.SplitModelViews(
                encoder=views.encoder,
                head=copied_head,
            ),
        )


def test_split_model_views_rebuild_after_state_restore() -> None:
    source = _model(stop_future=True, tail_layers=1, seed=40)
    restored = _model(stop_future=True, tail_layers=1, seed=42)
    nnx.update(restored, nnx.state(source))

    views = train.build_split_model_views(restored)
    report = train.validate_split_model_views(restored, views)

    for expected, head_value in zip(
        jax.tree.leaves(
            nnx.state(source, train.NON_BT4_TRAINABLE_FILTER)
        ),
        jax.tree.leaves(nnx.state(views.head, TrainableParam)),
        strict=True,
    ):
        np.testing.assert_array_equal(head_value, expected)
    for expected, encoder_value in zip(
        jax.tree.leaves(nnx.state(source, BT4TrainableParam)),
        jax.tree.leaves(nnx.state(views.encoder, TrainableParam)),
        strict=True,
    ):
        np.testing.assert_array_equal(encoder_value, expected)
    assert report["shared_variable_objects"] is True


def test_split_tokens_loss_and_gradients_match_monolithic() -> None:
    monolithic = _model(stop_future=True, tail_layers=1, seed=41)
    split = _model(stop_future=True, tail_layers=1, seed=41)
    batch = _batch()
    rng = jax.random.PRNGKey(43)

    future_planes = batch["future_planes"][:, : split.config.horizon]
    future_valid = batch["future_valid"][:, : split.config.horizon]
    _, _, rng_sigreg = jax.random.split(rng, 3)
    selection = train.select_future_training_targets(
        config=split.config,
        future_planes=future_planes,
        future_valid=future_valid,
        rng_sigreg=rng_sigreg,
        sample_future_targets=True,
    )
    expected_tokens, _ = (
        monolithic.encode_current_and_future_tokens_and_vectors(
            batch["current_planes"],
            selection.target_future_planes,
        )
    )
    split_tokens = train.split_training_bt4_tokens(split, batch, rng)
    np.testing.assert_array_equal(split_tokens, expected_tokens)
    (full_split_loss, full_split_aux), (
        full_split_head_grads,
        full_split_token_cotangent,
    ) = train._split_head_loss_and_grad(
        split,
        batch,
        rng,
        1.0,
        1.0,
        split_tokens,
    )

    monolithic_loss_and_grad = nnx.value_and_grad(
        train.normalized_stage1_training_loss_fn,
        argnums=nnx.DiffState(0, TrainableParam),
        has_aux=True,
    )
    (monolithic_loss, monolithic_aux), monolithic_grads = (
        monolithic_loss_and_grad(
            monolithic,
            batch,
            rng,
            1.0,
            1.0,
        )
    )
    (
        split_loss,
        split_aux,
        head_grads,
        encoder_grads,
        token_cotangent,
    ) = train.split_training_gradients(
        split,
        batch,
        rng,
        1.0,
        1.0,
    )
    partition = train.validate_split_gradient_partitions(
        split,
        head_grads,
        encoder_grads,
    )
    split_grads = train.merge_split_gradients(
        head_grads,
        encoder_grads,
    )

    np.testing.assert_allclose(
        split_loss,
        full_split_loss,
        rtol=0.0,
        atol=1e-6,
    )
    assert split_aux.keys() == full_split_aux.keys()
    for key in split_aux:
        np.testing.assert_allclose(
            split_aux[key],
            full_split_aux[key],
            rtol=1e-6,
            atol=1e-6,
            err_msg=f"full split {key}",
        )
    head_relative_l2, head_cosine = _tree_relative_l2_and_cosine(
        head_grads,
        full_split_head_grads,
    )
    assert head_relative_l2 <= 1e-5
    assert head_cosine >= 0.999999
    np.testing.assert_allclose(
        token_cotangent,
        full_split_token_cotangent,
        rtol=1e-6,
        atol=1e-6,
    )

    assert partition["full_leaf_count"] == (
        partition["head_leaf_count"] + partition["encoder_leaf_count"]
    )
    assert partition["encoder_leaf_count"] > 0
    assert token_cotangent.shape == split_tokens.shape
    assert token_cotangent.dtype == split_tokens.dtype
    np.testing.assert_allclose(
        split_loss,
        monolithic_loss,
        rtol=0.0,
        atol=1e-6,
    )
    assert split_aux.keys() == monolithic_aux.keys()
    for key in split_aux:
        np.testing.assert_allclose(
            split_aux[key],
            monolithic_aux[key],
            rtol=1e-6,
            atol=1e-6,
            err_msg=key,
        )

    relative_l2, cosine = _tree_relative_l2_and_cosine(
        monolithic_grads,
        split_grads,
    )
    assert relative_l2 <= 1e-5
    assert cosine >= 0.999999

    monolithic_pure = nnx.to_pure_dict(monolithic_grads)
    split_pure = nnx.to_pure_dict(split_grads)
    for partition_name, subtree in (
        ("encoder", monolithic_pure["encoder"]),
        (
            "head",
            {
                key: value
                for key, value in monolithic_pure.items()
                if key != "encoder"
            },
        ),
    ):
        split_subtree = (
            split_pure["encoder"]
            if partition_name == "encoder"
            else {
                key: value
                for key, value in split_pure.items()
                if key != "encoder"
            }
        )
        relative_l2, cosine = _tree_relative_l2_and_cosine(
            subtree,
            split_subtree,
        )
        assert relative_l2 <= 1e-5, partition_name
        assert cosine >= 0.999999, partition_name


def test_split_two_updates_preserve_global_optimizer_semantics() -> None:
    config = dataclasses.replace(
        _config(stop_future=True, trainable_tail_layers=1),
        grad_clip_norm=0.05,
        skip_nonfinite_updates=True,
    )
    monolithic = _model_from_config(config, seed=47)
    split = _model_from_config(config, seed=47)
    monolithic_optimizer = train.create_joint_optimizer(
        monolithic,
        config,
    )
    split_optimizer = train.create_joint_optimizer(split, config)
    monolithic_loss_and_grad = nnx.value_and_grad(
        train.normalized_stage1_training_loss_fn,
        argnums=nnx.DiffState(0, TrainableParam),
        has_aux=True,
    )

    for update in range(2):
        batch = _batch()
        rng = jax.random.fold_in(jax.random.PRNGKey(53), update)
        (monolithic_loss, monolithic_aux), monolithic_grads = (
            monolithic_loss_and_grad(
                monolithic,
                batch,
                rng,
                1.0,
                1.0,
            )
        )
        (
            split_loss,
            split_aux,
            head_grads,
            encoder_grads,
            _,
        ) = train.split_training_gradients(
            split,
            batch,
            rng,
            1.0,
            1.0,
        )
        train.validate_split_gradient_partitions(
            split,
            head_grads,
            encoder_grads,
        )
        monolithic_optimizer.update(monolithic, monolithic_grads)
        split_optimizer.update(
            split,
            train.merge_split_gradients(head_grads, encoder_grads),
        )

        np.testing.assert_allclose(
            split_loss,
            monolithic_loss,
            rtol=0.0,
            atol=1e-6,
        )
        assert split_aux.keys() == monolithic_aux.keys()
        model_relative_l2, _ = _tree_relative_l2_and_cosine(
            nnx.state(monolithic, TrainableParam),
            nnx.state(split, TrainableParam),
        )
        optimizer_relative_l2, _ = _tree_relative_l2_and_cosine(
            nnx.state(monolithic_optimizer),
            nnx.state(split_optimizer),
        )
        assert model_relative_l2 <= 1e-5
        assert optimizer_relative_l2 <= 1e-5
        assert int(monolithic_optimizer.step[...]) == update + 1
        assert int(split_optimizer.step[...]) == update + 1


def test_split_partition_validation_fails_closed() -> None:
    model = _model(stop_future=True, tail_layers=1, seed=59)
    _, _, head_grads, encoder_grads, _ = train.split_training_gradients(
        model,
        _batch(),
        jax.random.PRNGKey(61),
        1.0,
        1.0,
    )

    with pytest.raises(ValueError, match="overlap"):
        train.validate_split_gradient_partitions(
            model,
            train.merge_split_gradients(head_grads, encoder_grads),
            encoder_grads,
        )
    with pytest.raises(ValueError, match="mismatch"):
        train.validate_split_gradient_partitions(
            model,
            head_grads,
            nnx.State({}),
        )


def test_donated_split_executables_complete_one_cpu_update() -> None:
    config = dataclasses.replace(
        _config(stop_future=True, trainable_tail_layers=1),
        grad_clip_norm=0.05,
        skip_nonfinite_updates=True,
    )
    model = _model_from_config(config, seed=67)
    optimizer = train.create_joint_optimizer(model, config)
    views = train.build_split_model_views(model)

    loss, aux, timing = train.execute_split_training_step(
        train.split_training_functions(donate=True),
        model=model,
        views=views,
        optimizer=optimizer,
        batch=_batch(),
        rng=jax.random.PRNGKey(71),
        target_reference_count=1.0,
        pred_reference_count=1.0,
    )

    assert np.isfinite(float(loss))
    assert np.isfinite(float(aux["dfm_ce_loss"]))
    assert int(optimizer.step[...]) == 1
    assert views.head.out_bias is model.out_bias
    assert (
        views.encoder.encoder.embedding.scale
        is model.encoder.embedding.scale
    )
    assert set(timing) == {
        "split_encode_seconds",
        "split_head_vjp_seconds",
        "split_encoder_vjp_seconds",
        "split_optimizer_update_seconds",
        "split_total_seconds",
    }
    assert all(seconds >= 0.0 for seconds in timing.values())


def test_split_execution_config_is_frozen_to_accepted_contract() -> None:
    config = dataclasses.replace(
        _config(stop_future=True, trainable_tail_layers=1),
        horizon=8,
        encoder_dtype="bfloat16",
        param_dtype="float32",
        compute_dtype="bfloat16",
        jepa_norm_loss_coeff=0.0,
        jepa_sigreg_coeff=5.76,
        jepa_pred_sigreg_coeff=1.0,
        jepa_sigreg_estimator="v_stat",
        jepa_sigreg_example_count=64,
        learning_rate=3e-5,
        bt4_learning_rate=1e-6,
        lr_decay_start_steps=400,
        lr_decay_steps=800,
        lr_min_ratio=0.1,
        use_muon=True,
        grad_clip_norm=1.0,
        skip_nonfinite_updates=True,
    )
    train.validate_split_gradient_execution_config(
        config,
        objective="normalized",
        batch_size=128,
        sigreg_reference_count=1.0,
    )

    with pytest.raises(
        ValueError,
        match="jepa_pred_sigreg_coeff",
    ):
        train.validate_split_gradient_execution_config(
            dataclasses.replace(
                config,
                jepa_pred_sigreg_coeff=0.0,
            ),
            objective="normalized",
            batch_size=128,
            sigreg_reference_count=1.0,
        )


def test_split_preserves_full_tree_nonfinite_update_suppression() -> None:
    config = dataclasses.replace(
        _config(stop_future=True, trainable_tail_layers=1),
        grad_clip_norm=0.05,
        skip_nonfinite_updates=True,
    )
    monolithic = _model_from_config(config, seed=73)
    split = _model_from_config(config, seed=73)
    monolithic_optimizer = train.create_joint_optimizer(
        monolithic,
        config,
    )
    split_optimizer = train.create_joint_optimizer(split, config)
    before = nnx.state(monolithic, TrainableParam)
    batch = _batch()
    batch["current_planes"] = batch["current_planes"].at[
        0, 0, 0, 0
    ].set(jnp.nan)
    rng = jax.random.PRNGKey(79)

    monolithic_loss_and_grad = nnx.value_and_grad(
        train.normalized_stage1_training_loss_fn,
        argnums=nnx.DiffState(0, TrainableParam),
        has_aux=True,
    )
    (_, _), monolithic_grads = monolithic_loss_and_grad(
        monolithic,
        batch,
        rng,
        1.0,
        1.0,
    )
    _, _, head_grads, encoder_grads, _ = (
        train.split_training_gradients(
            split,
            batch,
            rng,
            1.0,
            1.0,
        )
    )
    monolithic_optimizer.update(monolithic, monolithic_grads)
    split_optimizer.update(
        split,
        train.merge_split_gradients(head_grads, encoder_grads),
    )

    for expected, monolithic_value, split_value in zip(
        jax.tree.leaves(before),
        jax.tree.leaves(nnx.state(monolithic, TrainableParam)),
        jax.tree.leaves(nnx.state(split, TrainableParam)),
        strict=True,
    ):
        np.testing.assert_array_equal(monolithic_value, expected)
        np.testing.assert_array_equal(split_value, expected)
    optimizer_relative_l2, _ = _tree_relative_l2_and_cosine(
        nnx.state(monolithic_optimizer),
        nnx.state(split_optimizer),
    )
    assert optimizer_relative_l2 <= 1e-5


def test_split_concrete_component_arguments_lower_without_execution() -> None:
    config = dataclasses.replace(
        _config(stop_future=True, trainable_tail_layers=1),
        grad_clip_norm=0.05,
        skip_nonfinite_updates=True,
    )
    model = _model_from_config(config, seed=83)
    optimizer = train.create_joint_optimizer(model, config)
    functions = train.split_training_functions(donate=True)
    views = train.build_split_model_views(model)
    batch = _batch()
    rng = jax.random.PRNGKey(89)
    arguments_by_component = {}

    for component in ("encode", "head_vjp", "encoder_vjp", "update"):
        function, arguments = train.split_component_compile_arguments(
            component,
            functions=functions,
            model=model,
            views=views,
            optimizer=optimizer,
            batch=batch,
            rng=rng,
            sigreg_reference_count=1.0,
        )
        arguments_by_component[component] = arguments
        lowered = function.lower(*arguments)
        assert lowered is not None
        assert int(optimizer.step[...]) == 0

    head_nbytes = train.split_state_abstract_nbytes(
        nnx.state(model, train.NON_BT4_TRAINABLE_FILTER)
    )
    encoder_nbytes = train.split_state_abstract_nbytes(
        nnx.state(model, BT4TrainableParam)
    )
    fixed_head_nbytes = (
        train.split_state_abstract_nbytes(nnx.state(model))
        - train.split_state_abstract_nbytes(
            nnx.state(model, TrainableParam)
        )
    )
    encode_arguments = arguments_by_component["encode"]
    head_arguments = arguments_by_component["head_vjp"]
    encoder_arguments = arguments_by_component["encoder_vjp"]
    assert (
        train.split_dynamic_argument_nbytes(
            (model, *encode_arguments[1:])
        )
        - train.split_dynamic_argument_nbytes(encode_arguments)
        == head_nbytes + fixed_head_nbytes
    )
    assert (
        train.split_dynamic_argument_nbytes(
            (model, *head_arguments[1:])
        )
        - train.split_dynamic_argument_nbytes(head_arguments)
        == encoder_nbytes + fixed_head_nbytes
    )
    assert (
        train.split_dynamic_argument_nbytes(
            (model, *encoder_arguments[1:])
        )
        - train.split_dynamic_argument_nbytes(encoder_arguments)
        == head_nbytes + fixed_head_nbytes
    )
