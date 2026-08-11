from __future__ import annotations

import gc
import os
from pathlib import Path

import chess
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from chess_dfm_jax.analysis.profile_targets import load_mapped_bt4_params
from chess_dfm_jax.encoding import encode_board
from chess_dfm_jax.nnx_bt4 import (
    EncoderLayer,
    jit_encode_tokens,
    jit_encode_tokens_with_captures,
    make_bt4_model,
)
from chess_dfm_jax.reference_bt4 import bt4_forward as reference_bt4_forward


HOOK_NAMES = (
    "hook_attn_in",
    "hook_attn_out",
    "resid_mid_after_ln",
    "hook_mlp_out",
    "resid_post_after_ln",
)


def _tiny_layer() -> EncoderLayer:
    return EncoderLayer(
        width=8,
        num_heads=2,
        mlp_dim=16,
        rngs=nnx.Rngs(7),
        param_dtype=jnp.float32,
        compute_dtype=jnp.float32,
        attention_impl="manual",
    )


def test_layer_capture_is_read_only_and_obeys_pre_alpha_residual_boundaries():
    layer = _tiny_layer()
    x = jax.random.normal(jax.random.key(11), (3, 5, 8))
    alpha = 0.37

    ordinary = layer(x, alpha)
    captured, hooks = layer.forward_with_capture(x, alpha)

    np.testing.assert_array_equal(captured, ordinary)
    assert hooks.hook_attn_in.shape == (3, 5, 8)
    assert hooks.hook_attn_out.shape == (3, 5, 8)
    assert hooks.resid_mid_after_ln.shape == (3, 5, 8)
    assert hooks.hook_mlp_out.shape == (3, 5, 8)
    assert hooks.resid_post_after_ln.shape == (3, 5, 8)
    np.testing.assert_array_equal(hooks.hook_attn_in, x)
    np.testing.assert_allclose(
        hooks.resid_mid_after_ln,
        layer.ln_attn(x + alpha * hooks.hook_attn_out),
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        hooks.resid_post_after_ln,
        layer.ln_ffn(
            hooks.resid_mid_after_ln + alpha * hooks.hook_mlp_out
        ),
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_array_equal(hooks.resid_post_after_ln, captured)


def test_layer_replacement_boundary_is_before_alpha_residual_and_layer_norm():
    layer = _tiny_layer()
    x = jax.random.normal(jax.random.key(13), (2, 4, 8))
    zeros = jnp.zeros_like(x)
    alpha = 0.41

    replaced, hooks = layer.forward_with_capture(
        x,
        alpha,
        attention_output_override=zeros,
        mlp_output_override=zeros,
    )
    expected_mid = layer.ln_attn(x)
    expected_post = layer.ln_ffn(expected_mid)
    np.testing.assert_array_equal(hooks.hook_attn_out, zeros)
    np.testing.assert_array_equal(hooks.hook_mlp_out, zeros)
    np.testing.assert_array_equal(hooks.resid_mid_after_ln, expected_mid)
    np.testing.assert_array_equal(hooks.resid_post_after_ln, expected_post)
    np.testing.assert_array_equal(replaced, expected_post)

    with pytest.raises(ValueError, match="attention_output_override"):
        layer.forward_with_capture(
            x,
            alpha,
            attention_output_override=jnp.zeros((2, 4, 7)),
        )
    with pytest.raises(ValueError, match="mlp_output_override"):
        layer.forward_with_capture(
            x,
            alpha,
            mlp_output_override=jnp.zeros((2, 4, 7)),
        )


@pytest.mark.skipif(
    os.environ.get("CHESS_DFM_RUN_BT4_HOOK_INTEGRATION") != "1",
    reason="requires the local raw BT4 asset and accelerator",
)
def test_real_fp32_bt4_hooks_preserve_tokens_and_match_reference_boundaries():
    repo_root = Path(__file__).resolve().parents[1]
    models_dir = repo_root / "models" / "source" / "extracted"
    params = load_mapped_bt4_params(models_dir=str(models_dir))
    boards = (chess.Board(), chess.Board())
    boards[1].push_uci("e2e4")
    planes = jnp.asarray(
        np.stack(
            [
                encode_board(
                    board,
                    [],
                    input_format="INPUT_CLASSICAL_112_PLANE",
                )
                for board in boards
            ]
        ),
        dtype=jnp.float32,
    )

    reference_forward = jax.jit(
        lambda encoded_planes: reference_bt4_forward(
            params,
            encoded_planes,
            capture=True,
        )
    )
    (
        _reference_policy,
        _reference_wdl,
        _reference_moves_left,
        reference_captures,
    ) = jax.device_get(jax.block_until_ready(reference_forward(planes)))
    del reference_forward
    jax.clear_caches()
    gc.collect()

    model = make_bt4_model(params, dtype=jnp.float32)
    ordinary_tokens = jax.device_get(
        jax.block_until_ready(jit_encode_tokens(model, planes))
    )
    captured_tokens, captures = jax.device_get(
        jax.block_until_ready(
            jit_encode_tokens_with_captures(model, planes)
        )
    )
    np.testing.assert_array_equal(captured_tokens, ordinary_tokens)
    expected_shape = (15, 2, 64, 1024)
    for hook_name in HOOK_NAMES:
        values = getattr(captures, hook_name)
        assert values.shape == expected_shape
        for layer_index in range(15):
            reference = reference_captures[
                f"blocks.{layer_index}.{hook_name}"
            ]
            max_difference = float(
                np.max(np.abs(values[layer_index] - reference))
            )
            assert max_difference < 5e-4, (
                hook_name,
                layer_index,
                max_difference,
            )
    np.testing.assert_array_equal(
        captures.resid_post_after_ln[-1],
        captured_tokens,
    )
    np.testing.assert_array_equal(
        captures.hook_attn_in[1:],
        captures.resid_post_after_ln[:-1],
    )
