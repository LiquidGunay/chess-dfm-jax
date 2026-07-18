from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax import nnx

import research.train as train
from chess_dfm_jax.nnx_bt4 import (
    EncoderLayer,
    InputEmbedding,
    TrainableParam,
)
from research.prepare import REPO_ROOT


class TinyEmbedding(nnx.Module):
    def __init__(
        self,
        *,
        embedding_size: int,
        expected_batch_size: int | None = None,
    ):
        self.embedding_size = embedding_size
        self.expected_batch_size = expected_batch_size
        self.scale = TrainableParam(
            jnp.asarray(1.0, dtype=jnp.float32)
        )
        self.fixed_offset = nnx.Param(
            jnp.asarray(0.125, dtype=jnp.float32)
        )

    def __call__(
        self,
        planes: jax.Array,
        alpha: float,
    ) -> tuple[jax.Array, int]:
        if (
            self.expected_batch_size is not None
            and planes.shape[0] != self.expected_batch_size
        ):
            raise ValueError(
                f"expected encoder batch {self.expected_batch_size}, "
                f"got {planes.shape[0]}"
            )
        plane_mean = jnp.mean(planes, axis=(1, 2, 3))
        square = jnp.linspace(
            0.0,
            1.0,
            64,
            dtype=jnp.float32,
        )[None, :, None]
        feature = jnp.linspace(
            0.1,
            1.0,
            self.embedding_size,
            dtype=jnp.float32,
        )[None, None, :]
        tokens = self.scale[...] * (
            square
            + plane_mean.reshape((-1, 1, 1)) * feature
            + self.fixed_offset[...] * alpha
        )
        return tokens.reshape((-1, self.embedding_size)), planes.shape[0]


class TinyLayer(nnx.Module):
    def __init__(self, embedding_size: int):
        self.bias = TrainableParam(
            jnp.zeros((embedding_size,), dtype=jnp.float32)
        )
        self.fixed_gain = nnx.Param(
            jnp.asarray(1.0, dtype=jnp.float32)
        )

    def __call__(
        self,
        tokens: jax.Array,
        alpha: float,
    ) -> jax.Array:
        return (
            tokens * self.fixed_gain[...]
            + self.bias[...] * alpha
        )


class TinyEncoder(nnx.Module):
    def __init__(
        self,
        *,
        embedding_size: int = 16,
        expected_batch_size: int | None = None,
    ):
        self.embedding_size = embedding_size
        self.embedding = TinyEmbedding(
            embedding_size=embedding_size,
            expected_batch_size=expected_batch_size,
        )
        self.layers = nnx.List([TinyLayer(embedding_size)])
        self.policy_head = nnx.Param(
            jnp.ones((3, 5), dtype=jnp.float32)
        )
        self.value_head = nnx.Param(
            jnp.ones((2, 7), dtype=jnp.float32)
        )
        self.moves_left_head = nnx.Param(
            jnp.ones((11,), dtype=jnp.float32)
        )

    def encode_tokens(
        self,
        planes: jax.Array,
        alpha: float | None = None,
    ) -> jax.Array:
        if alpha is None:
            alpha = (
                float(math.pow(2.0 * len(self.layers), -0.25))
                if len(self.layers) > 0
                else 1.0
            )
        tokens, batch_size = self.embedding(planes, alpha)
        tokens = tokens.reshape(
            (batch_size, 64, self.embedding_size)
        )
        for layer in self.layers:
            tokens = layer(tokens, alpha)
        return tokens


def shaped_values(
    shape: tuple[int, ...],
    *,
    scale: float = 0.1,
) -> np.ndarray:
    size = int(np.prod(shape))
    return np.linspace(
        -scale,
        scale,
        size,
        dtype=np.float32,
    ).reshape(shape)


class SyntheticBfloat16Bt4Encoder(nnx.Module):
    """Small real BT4 trunk that exercises every storage-dtype cast site."""

    def __init__(self):
        width = 8
        headcount = 2
        embedding_dense_size = 2
        input_channels = 4
        pos_planes = 1
        embedding_params = {
            "preproc_w": shaped_values(
                (64 * pos_planes, 64 * embedding_dense_size)
            ),
            "preproc_b": shaped_values(
                (64 * embedding_dense_size,)
            ),
            "w": shaped_values(
                (input_channels + embedding_dense_size, width)
            ),
            "b": shaped_values((width,)),
            "ln_scale": np.ones((width,), dtype=np.float32),
            "ln_bias": shaped_values((width,), scale=0.01),
            "mul_gate": shaped_values((width,), scale=0.5) + 1.0,
            "add_gate": shaped_values((width,), scale=0.05),
            "ffn": {
                "dense1_w": shaped_values((width, 16)),
                "dense1_b": shaped_values((16,)),
                "dense2_w": shaped_values((16, width)),
                "dense2_b": shaped_values((width,)),
            },
            "ffn_ln_scale": np.ones((width,), dtype=np.float32),
            "ffn_ln_bias": shaped_values((width,), scale=0.01),
        }
        smolgen_hidden = 8
        smolgen_size = 4
        layer_params = {
            "mha": {
                "q_w": shaped_values((width, width)),
                "q_b": shaped_values((width,)),
                "k_w": shaped_values((width, width)),
                "k_b": shaped_values((width,)),
                "v_w": shaped_values((width, width)),
                "v_b": shaped_values((width,)),
                "dense_w": shaped_values((width, width)),
                "dense_b": shaped_values((width,)),
                "smolgen": {
                    "compress_w": shaped_values((width, 2)),
                    "dense1_w": shaped_values(
                        (64 * 2, smolgen_hidden)
                    ),
                    "dense1_b": shaped_values((smolgen_hidden,)),
                    "ln1_scale": np.ones(
                        (smolgen_hidden,),
                        dtype=np.float32,
                    ),
                    "ln1_bias": shaped_values(
                        (smolgen_hidden,),
                        scale=0.01,
                    ),
                    "dense2_w": shaped_values(
                        (
                            smolgen_hidden,
                            headcount * smolgen_size,
                        )
                    ),
                    "dense2_b": shaped_values(
                        (headcount * smolgen_size,)
                    ),
                    "ln2_scale": np.ones(
                        (headcount * smolgen_size,),
                        dtype=np.float32,
                    ),
                    "ln2_bias": shaped_values(
                        (headcount * smolgen_size,),
                        scale=0.01,
                    ),
                },
            },
            "ln1": {
                "scale": np.ones((width,), dtype=np.float32),
                "bias": shaped_values((width,), scale=0.01),
            },
            "ffn": {
                "dense1_w": shaped_values((width, 32)),
                "dense1_b": shaped_values((32,)),
                "dense2_w": shaped_values((32, width)),
                "dense2_b": shaped_values((width,)),
            },
            "ln2": {
                "scale": np.ones((width,), dtype=np.float32),
                "bias": shaped_values((width,), scale=0.01),
            },
        }
        self.embedding_size = width
        self.embedding = InputEmbedding(
            embedding_params,
            embedding_size=width,
            embedding_dense_size=embedding_dense_size,
            pos_planes=pos_planes,
            dtype=jnp.bfloat16,
            param_cls=TrainableParam,
        )
        self.layers = nnx.List(
            [
                EncoderLayer(
                    width=width,
                    num_heads=headcount,
                    mlp_dim=32,
                    rngs=nnx.Rngs(0),
                    param_dtype=jnp.bfloat16,
                    compute_dtype=jnp.bfloat16,
                    use_qk_gain=True,
                    attention_impl="manual",
                    layer_params=layer_params,
                    shared_smolgen_w=shaped_values(
                        (smolgen_size, 64 * 64)
                    ),
                    param_cls=TrainableParam,
                )
            ]
        )

    def encode_tokens(
        self,
        planes: jax.Array,
        alpha: float | None = None,
    ) -> jax.Array:
        if alpha is None:
            alpha = float((2.0 * len(self.layers)) ** -0.25)
        tokens, batch_size = self.embedding(planes, alpha)
        tokens = tokens.reshape(
            (batch_size, 64, self.embedding_size)
        )
        for layer in self.layers:
            tokens = layer(tokens, alpha)
        return tokens


def tiny_config(**overrides: Any) -> train.JointLatentSASAConfig:
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
        "horizon": 2,
        "compute_dtype": "float32",
        "param_dtype": "float32",
        "encoder_dtype": "float32",
        "first_legality_coeff": 1.0,
        "legality_on_masked_only": False,
        "jepa_sigreg_coeff": 0.0,
        "jepa_pred_sigreg_coeff": 0.0,
        "jepa_sigreg_kind": "le_jepa",
        "jepa_sigreg_proj_dim": 4,
        "use_qk_gain": True,
        "use_muon": False,
        "lr_warmup_steps": 0,
        "grad_clip_norm": 0.0,
        "remat_blocks": False,
        "jepa_target_semantics": "ema",
        "jepa_target_ema_decay": 0.5,
    }
    return train.JointLatentSASAConfig(
        **(values | overrides)
    )


def make_components(
    *,
    seed: int = 7,
    expected_batch_size: int | None = None,
    **config_overrides: Any,
) -> tuple[
    train.JointLatentSASAModel,
    train.EmaTargetModel,
    nnx.Optimizer,
]:
    config = tiny_config(**config_overrides)
    model = train.JointLatentSASAModel(
        TinyEncoder(expected_batch_size=expected_batch_size),
        config,
        rngs=nnx.Rngs(seed),
    )
    optimizer = nnx.Optimizer(
        model,
        optax.sgd(1e-3),
        wrt=TrainableParam,
    )
    ema_target = train.EmaTargetModel(model)
    train.sync_ema_target_from_online(ema_target, model)
    return model, ema_target, optimizer


def batch() -> dict[str, jax.Array]:
    current = jnp.linspace(
        -0.25,
        0.75,
        2 * 112 * 8 * 8,
        dtype=jnp.float32,
    ).reshape((2, 112, 8, 8))
    future = jnp.stack(
        (current + 0.125, current - 0.25),
        axis=1,
    )
    return {
        "current_planes": current,
        "future_planes": future,
        "action_indices": jnp.asarray(
            [[1, 2], [3, 4]],
            dtype=jnp.int32,
        ),
        "valid": jnp.ones((2,), dtype=jnp.float32),
        "future_valid": jnp.asarray(
            [[1.0, 1.0], [1.0, 0.0]],
            dtype=jnp.float32,
        ),
        "legal_idx": jnp.asarray(
            [
                [[1, 5, 9, 0], [2, 6, 0, 0]],
                [[3, 7, 11, 15], [4, 8, 12, 0]],
            ],
            dtype=jnp.int32,
        ),
        "legal_count": jnp.asarray(
            [[3, 2], [4, 3]],
            dtype=jnp.int32,
        ),
        "legal_masks_valid": jnp.ones(
            (2, 2),
            dtype=jnp.float32,
        ),
        "deterministic_t": jnp.asarray(
            0.25,
            dtype=jnp.float32,
        ),
    }


def pure_trainable(module: nnx.Module) -> dict[str, Any]:
    return dict(
        nnx.to_pure_dict(nnx.state(module, TrainableParam))
    )


def pure_state(module: nnx.Module) -> dict[str, Any]:
    return dict(nnx.to_pure_dict(nnx.state(module)))


def assert_trees_exact(left: Any, right: Any) -> None:
    assert jax.tree.structure(left) == jax.tree.structure(right)
    for left_leaf, right_leaf in zip(
        jax.tree.leaves(left),
        jax.tree.leaves(right),
        strict=True,
    ):
        np.testing.assert_array_equal(
            np.asarray(left_leaf),
            np.asarray(right_leaf),
        )


def test_bfloat16_bt4_compute_is_invariant_to_fp32_shadow_storage() -> None:
    online = SyntheticBfloat16Bt4Encoder()
    shadow = train.EmaBt4Encoder(online)
    shadow_state = nnx.state(shadow)
    nnx.update(
        shadow,
        jax.tree.map(
            lambda value: jnp.array(
                value,
                dtype=jnp.float32,
                copy=True,
            ),
            shadow_state,
        ),
    )
    online_state = {
        "embedding": pure_state(online.embedding),
        "layers": pure_state(online.layers),
    }
    fp32_state = pure_state(shadow)
    for online_leaf, shadow_leaf in zip(
        jax.tree.leaves(online_state),
        jax.tree.leaves(fp32_state),
        strict=True,
    ):
        assert np.asarray(online_leaf).dtype == jnp.bfloat16
        assert np.asarray(shadow_leaf).dtype == np.float32
        assert (
            online_leaf.unsafe_buffer_pointer()
            != shadow_leaf.unsafe_buffer_pointer()
        )

    planes = jnp.linspace(
        -1.0,
        1.0,
        2 * 4 * 8 * 8,
        dtype=jnp.float32,
    ).reshape((2, 4, 8, 8))

    @nnx.jit
    def encode_pair(online_encoder, shadow_encoder, inputs):
        return (
            online_encoder.encode_tokens(inputs),
            shadow_encoder.encode_tokens(inputs),
        )

    online_tokens, shadow_tokens = encode_pair(
        online,
        shadow,
        planes,
    )
    jax.block_until_ready((online_tokens, shadow_tokens))
    assert bool(jnp.all(jnp.isfinite(online_tokens)))
    np.testing.assert_array_equal(
        np.asarray(online_tokens),
        np.asarray(shadow_tokens),
    )


def test_ema_config_is_checked_cli_overridable_and_resume_bound(
    monkeypatch,
) -> None:
    config = tiny_config()
    train.validate_objective_config(
        objective="normalized",
        config=config,
    )
    with pytest.raises(ValueError, match="require --objective normalized"):
        train.validate_objective_config(
            objective="legacy",
            config=config,
        )
    with pytest.raises(ValueError, match="intrinsically detached"):
        train.validate_objective_config(
            objective="normalized",
            config=tiny_config(jepa_target_stop_gradient=True),
        )
    with pytest.raises(ValueError, match=r"must be in \[0, 1\)"):
        train.validate_objective_config(
            objective="normalized",
            config=tiny_config(jepa_target_ema_decay=1.0),
        )
    with pytest.raises(ValueError, match="rounds to 1.0 in FP32"):
        train.validate_objective_config(
            objective="normalized",
            config=tiny_config(
                jepa_target_ema_decay=1.0 - 1e-10
            ),
        )
    with pytest.raises(ValueError, match="inert unless"):
        train.validate_objective_config(
            objective="normalized",
            config=train.JointLatentSASAConfig(
                jepa_target_ema_decay=0.8
            ),
        )

    args = train.parse_args(
        [
            "--objective",
            "normalized",
            "--jepa-target-semantics",
            "ema",
            "--jepa-target-ema-decay",
            "0.975",
        ]
    )
    overridden = train.apply_config_overrides(
        train.JointLatentSASAConfig(),
        args,
    )
    assert overridden.jepa_target_semantics == "ema"
    assert overridden.jepa_target_ema_decay == 0.975

    monkeypatch.setattr(
        train,
        "require_within_workspace",
        lambda path: Path(path),
    )
    monkeypatch.setattr(
        train,
        "load_asset_manifest",
        lambda: {
            "trajectory_v3": {
                "archive": {
                    "sha256": "trajectory-digest",
                    "size_bytes": 123,
                }
            }
        },
    )
    monkeypatch.setattr(
        train,
        "sha256_file",
        lambda _path: "file-digest",
    )
    contract = train.build_research_resume_contract(
        config=overridden,
        objective="normalized",
        sigreg_reference_count=1.0,
        batch_size=64,
        train_seed=3,
        train_provenance={"kind": "test"},
        models_dir=REPO_ROOT / "models",
    )
    semantics = contract["objective"]
    assert semantics["jepa_target_semantics"] == "ema"
    assert semantics["jepa_target_ema"]["decay"] == 0.975
    assert semantics["jepa_target_ema"]["update_order"] == (
        "after_online_optimizer"
    )
    assert semantics["jepa_target_ema"][
        "effective_decay_float32"
    ] == float(np.float32(0.975))
    assert semantics["jepa_target_ema"]["online_target_sigreg_gradient"] == (
        "attached"
    )
    online_contract = train.build_research_resume_contract(
        config=train.JointLatentSASAConfig(),
        objective="normalized",
        sigreg_reference_count=1.0,
        batch_size=64,
        train_seed=3,
        train_provenance={"kind": "test"},
        models_dir=REPO_ROOT / "models",
    )
    assert online_contract["objective"]["jepa_target_semantics"] == (
        "online"
    )
    assert "jepa_target_ema" not in online_contract["objective"]
    assert (
        "chess_dfm_jax/nnx_bt4.py"
        in online_contract["code"]["files"]
    )


def test_ema_shadow_scope_sync_and_post_optimizer_formula() -> None:
    model, ema_target, _ = make_components()
    assert not hasattr(ema_target.encoder, "policy_head")
    assert not hasattr(ema_target.encoder, "value_head")
    assert not hasattr(ema_target.encoder, "moves_left_head")
    assert set(pure_trainable(ema_target)) == {
        "encoder",
        "jepa_state_norm",
        "state_projector",
    }
    online_shadow_source = {
        "encoder": {
            "embedding": pure_state(model.encoder.embedding),
            "layers": pure_state(model.encoder.layers),
        },
        "jepa_state_norm": pure_state(model.jepa_state_norm),
        "state_projector": pure_state(model.state_projector),
    }
    target_runtime_state = pure_state(ema_target)
    assert_trees_exact(target_runtime_state, online_shadow_source)
    for target_leaf, online_leaf in zip(
        jax.tree.leaves(target_runtime_state),
        jax.tree.leaves(online_shadow_source),
        strict=True,
    ):
        assert (
            target_leaf.unsafe_buffer_pointer()
            != online_leaf.unsafe_buffer_pointer()
        )
    target_before = pure_trainable(ema_target)
    assert all(
        np.asarray(leaf).dtype == np.float32
        for leaf in jax.tree.leaves(target_before)
    )

    for target_component, online_component in (
        (ema_target.encoder, model.encoder),
        (ema_target.state_projector, model.state_projector),
        (ema_target.jepa_state_norm, model.jepa_state_norm),
    ):
        expected = jax.tree.map(
            lambda value: np.asarray(value, dtype=np.float32),
            pure_trainable(online_component),
        )
        assert_trees_exact(
            pure_trainable(target_component),
            expected,
        )

        online_state = nnx.state(
            online_component,
            TrainableParam,
        )
        nnx.update(
            online_component,
            jax.tree.map(lambda value: value + 4.0, online_state),
        )

    train.update_ema_target_after_optimizer(
        ema_target,
        model,
        0.75,
    )
    target_after = pure_trainable(ema_target)
    online_after = {
        "encoder": pure_trainable(model.encoder),
        "jepa_state_norm": pure_trainable(model.jepa_state_norm),
        "state_projector": pure_trainable(model.state_projector),
    }
    expected_after = jax.tree.map(
        lambda old, online: (
            0.75 * np.asarray(old, dtype=np.float32)
            + 0.25 * np.asarray(online, dtype=np.float32)
        ),
        target_before,
        online_after,
    )
    assert_trees_exact(target_after, expected_after)


def test_fp32_ema_master_keeps_sub_bfloat16_increment() -> None:
    model, ema_target, _ = make_components()
    target_before = float(
        ema_target.encoder.embedding.scale[...]
    )
    assert target_before == 1.0
    next_bfloat16 = jnp.nextafter(
        jnp.asarray(1.0, dtype=jnp.bfloat16),
        jnp.asarray(2.0, dtype=jnp.bfloat16),
    )
    model.encoder.embedding.scale[...] = jnp.asarray(
        next_bfloat16,
        dtype=model.encoder.embedding.scale[...].dtype,
    )
    decay = 0.999
    train.update_ema_target_after_optimizer(
        ema_target,
        model,
        decay,
    )
    target_after = ema_target.encoder.embedding.scale[...]
    assert target_after.dtype == jnp.float32
    assert float(target_after) > target_before
    assert float(target_after) < float(next_bfloat16)
    assert jnp.asarray(target_after, dtype=jnp.bfloat16) == jnp.asarray(
        target_before,
        dtype=jnp.bfloat16,
    )


def test_ema_chunked_future_encoder_never_flattens_batch_and_horizon() -> None:
    _, ema_target, _ = make_components(
        expected_batch_size=2,
        bt4_encode_chunk_size=1,
    )
    targets = ema_target.encode_future_targets(
        batch()["future_planes"]
    )
    assert targets.shape == (2, 2, 16)


def test_ema_positive_target_has_no_gradient_and_reports_clear_metrics() -> None:
    model, ema_target, _ = make_components()
    projector_state = nnx.state(
        model.state_projector,
        TrainableParam,
    )
    nnx.update(
        model.state_projector,
        jax.tree.map(lambda value: value + 0.01, projector_state),
    )

    loss_and_both_grads = nnx.value_and_grad(
        train.ema_normalized_stage1_loss_fn,
        argnums=(
            nnx.DiffState(0, TrainableParam),
            nnx.DiffState(1, TrainableParam),
        ),
        has_aux=True,
    )
    (loss, aux), (online_grads, target_grads) = (
        loss_and_both_grads(
            model,
            ema_target,
            batch(),
            jax.random.PRNGKey(19),
            1.0,
            1.0,
        )
    )
    assert np.isfinite(float(loss))
    assert any(
        float(jnp.linalg.norm(leaf)) > 0.0
        for leaf in jax.tree.leaves(online_grads)
    )
    assert all(
        float(jnp.linalg.norm(leaf)) == 0.0
        for leaf in jax.tree.leaves(target_grads)
    )
    assert float(aux["jepa_target_semantics_ema"]) == 1.0
    assert float(aux["jepa_positive_ema_target_rms"]) > 0.0
    assert float(aux["jepa_online_future_target_rms"]) > 0.0
    assert float(aux["z_online_future_target_norm"]) == float(
        aux["z_target_norm"]
    )
    assert float(aux["z_positive_ema_target_norm"]) > 0.0
    assert (
        float(aux["jepa_online_future_vs_ema_target_mse"])
        > 0.0
    )
    assert np.isfinite(
        float(aux["jepa_pred_positive_ema_target_cosine"])
    )


def test_train_step_ema_uses_updated_online_parameters() -> None:
    model, ema_target, optimizer = make_components(seed=27)
    donated_model, donated_target, donated_optimizer = (
        make_components(seed=27)
    )
    target_before = jax.tree.map(
        lambda value: np.asarray(value).copy(),
        pure_trainable(donated_target),
    )
    rng = jax.random.PRNGKey(23)
    reference_loss, _ = train.train_ema_normalized_stage1_step(
        model,
        ema_target,
        optimizer,
        batch(),
        rng,
        1.0,
        1.0,
    )
    donated_loss, _ = train.train_ema_normalized_stage1_step_donated(
        donated_model,
        donated_target,
        donated_optimizer,
        batch(),
        rng,
        1.0,
        1.0,
    )
    jax.block_until_ready((reference_loss, donated_loss))
    assert float(donated_loss) == float(reference_loss)
    assert int(donated_optimizer.step[...]) == 1
    assert_trees_exact(
        train.extract_research_train_state(
            model,
            optimizer,
            ema_target,
        ),
        train.extract_research_train_state(
            donated_model,
            donated_optimizer,
            donated_target,
        ),
    )

    online_after = {
        "encoder": pure_trainable(donated_model.encoder),
        "jepa_state_norm": pure_trainable(
            donated_model.jepa_state_norm
        ),
        "state_projector": pure_trainable(
            donated_model.state_projector
        ),
    }
    expected = jax.tree.map(
        lambda old, online: (
            0.5 * np.asarray(old, dtype=np.float32)
            + 0.5 * np.asarray(online, dtype=np.float32)
        ),
        target_before,
        online_after,
    )
    assert_trees_exact(
        pure_trainable(donated_target),
        expected,
    )


def test_ema_checkpoint_resume_restores_shadow_exactly() -> None:
    local_tmp = REPO_ROOT / ".local" / "tmp"
    local_tmp.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=local_tmp) as directory:
        checkpoint_root = Path(directory) / "checkpoints"
        model, ema_target, optimizer = make_components(seed=31)
        loss, _ = train._ema_normalized_train_step_impl(
            model,
            ema_target,
            optimizer,
            batch(),
            jax.random.PRNGKey(29),
            1.0,
            1.0,
        )
        jax.block_until_ready(loss)
        contract = {
            "target_semantics": "ema",
            "decay": 0.5,
        }
        checkpoint = train.save_research_checkpoint(
            checkpoint_root,
            model=model,
            optimizer=optimizer,
            ema_target=ema_target,
            research_update=1,
            next_data_cursor=1,
            resume_contract=contract,
            lineage={"kind": "test"},
        )

        default_payload = train.extract_research_train_state(
            model,
            optimizer,
        )
        assert set(default_payload) == {
            "step",
            "model_trainable",
            "optimizer_state",
        }
        online_checkpoint = train.save_research_checkpoint(
            Path(directory) / "online-checkpoints",
            model=model,
            optimizer=optimizer,
            research_update=1,
            next_data_cursor=1,
            resume_contract={"target_semantics": "online"},
            lineage={"kind": "test"},
        )
        online_manifest = json.loads(
            (online_checkpoint / "manifest.json").read_text(
                encoding="utf-8"
            )
        )
        assert "ema_target_abi" not in online_manifest
        with np.load(
            online_checkpoint / "state.npz",
            allow_pickle=True,
        ) as online_state:
            assert "ema_target" not in online_state.files
        ema_payload = train.extract_research_train_state(
            model,
            optimizer,
            ema_target,
        )
        assert set(ema_payload) == {
            "step",
            "model_trainable",
            "optimizer_state",
            "ema_target",
        }

        resumed_model, resumed_target, resumed_optimizer = (
            make_components(seed=31)
        )
        manifest = train.load_research_checkpoint(
            checkpoint,
            model=resumed_model,
            optimizer=resumed_optimizer,
            ema_target=resumed_target,
            expected_resume_contract=contract,
        )
        assert "ema_target_abi" in manifest
        assert_trees_exact(
            train.extract_research_train_state(
                model,
                optimizer,
                ema_target,
            ),
            train.extract_research_train_state(
                resumed_model,
                resumed_optimizer,
                resumed_target,
            ),
        )

        without_target_model, _, without_target_optimizer = (
            make_components(seed=31)
        )
        with pytest.raises(
            ValueError,
            match="contains EMA target state",
        ):
            train.load_research_checkpoint(
                checkpoint,
                model=without_target_model,
                optimizer=without_target_optimizer,
                expected_resume_contract=contract,
            )

        for candidate_model, candidate_target, candidate_optimizer in (
            (model, ema_target, optimizer),
            (
                resumed_model,
                resumed_target,
                resumed_optimizer,
            ),
        ):
            continued_loss, _ = (
                train._ema_normalized_train_step_impl(
                    candidate_model,
                    candidate_target,
                    candidate_optimizer,
                    batch(),
                    jax.random.PRNGKey(30),
                    1.0,
                    1.0,
                )
            )
            jax.block_until_ready(continued_loss)
        assert_trees_exact(
            train.extract_research_train_state(
                model,
                optimizer,
                ema_target,
            ),
            train.extract_research_train_state(
                resumed_model,
                resumed_optimizer,
                resumed_target,
            ),
        )
