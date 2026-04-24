"""Reusable roofline targets for BT4 forward and encoder-backward profiling."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import jax.numpy as jnp

from chess_dfm_jax.encoding import encode_board
from chess_dfm_jax.nnx_bt4 import (
    jit_encode_tokens,
    jit_encoder_loss_and_grad,
    make_bt4_model,
)
from chess_dfm_jax.training.jepa import (
    JEPAConfig,
    build_synthetic_transition_batch,
    build_transition_batch,
    create_jepa_components,
    train_step,
)
from chess_dfm_jax.paths import default_bt4_paths
from chess_dfm_jax.policy import attention_policy_map
from chess_dfm_jax.reference_bt4 import bt4_forward as reference_bt4_forward
from chess_dfm_jax.weights import load_pb_gz, map_bt4_weights

import chess


DEFAULT_FENS = [
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "r1bq1rk1/pp1nbppp/2p1pn2/2Pp4/3P4/2N1PN2/PP3PPP/R1BQKB1R w KQ - 1 8",
]


def parse_compute_dtype(name: str):
    mapping = {
        "float16": jnp.float16,
        "fp16": jnp.float16,
        "float32": jnp.float32,
        "fp32": jnp.float32,
        "bfloat16": jnp.bfloat16,
        "bf16": jnp.bfloat16,
    }
    key = name.lower()
    if key not in mapping:
        raise ValueError(f"Unsupported compute dtype: {name}")
    return mapping[key]


def load_mapped_bt4_params(*, models_dir: str | None = None, pb: str | None = None) -> dict:
    paths = default_bt4_paths(models_dir)
    pb_path = Path(pb) if pb else paths["exported_pb"]
    bundle = load_pb_gz(str(pb_path))
    return map_bt4_weights(bundle, mapping_table=attention_policy_map())


def build_planes_batch(
    *,
    batch_size: int,
    input_format: str = "INPUT_CLASSICAL_112_PLANE",
    fens: list[str] | None = None,
) -> np.ndarray:
    boards = [chess.Board(fen) for fen in (fens or DEFAULT_FENS)]
    encoded = [encode_board(board, [], input_format=input_format).astype(np.float32) for board in boards]
    return np.stack([encoded[idx % len(encoded)] for idx in range(batch_size)], axis=0)


def make_reference_forward_target(
    *,
    batch_size: int,
    models_dir: str | None = None,
    pb: str | None = None,
    input_format: str = "INPUT_CLASSICAL_112_PLANE",
) -> dict:
    params = load_mapped_bt4_params(models_dir=models_dir, pb=pb)
    planes = build_planes_batch(batch_size=batch_size, input_format=input_format)

    def step(batch_planes):
        return reference_bt4_forward(params, batch_planes)

    return {
        "name": f"reference_forward_bs{batch_size}",
        "step_fn": step,
        "args": (planes,),
        "kwargs": {},
    }


def make_nnx_encoder_forward_target(
    *,
    batch_size: int,
    models_dir: str | None = None,
    pb: str | None = None,
    input_format: str = "INPUT_CLASSICAL_112_PLANE",
    compute_dtype: str = "float32",
) -> dict:
    params = load_mapped_bt4_params(models_dir=models_dir, pb=pb)
    dtype = parse_compute_dtype(compute_dtype)
    model = make_bt4_model(params, dtype=dtype)
    planes = build_planes_batch(batch_size=batch_size, input_format=input_format)
    return {
        "name": f"nnx_encoder_forward_{compute_dtype}_bs{batch_size}",
        "step_fn": jit_encode_tokens,
        "args": (model, np.asarray(planes, dtype=np.dtype(dtype))),
        "kwargs": {},
    }


def make_nnx_encoder_backward_target(
    *,
    batch_size: int,
    models_dir: str | None = None,
    pb: str | None = None,
    input_format: str = "INPUT_CLASSICAL_112_PLANE",
    compute_dtype: str = "float32",
) -> dict:
    params = load_mapped_bt4_params(models_dir=models_dir, pb=pb)
    dtype = parse_compute_dtype(compute_dtype)
    model = make_bt4_model(params, dtype=dtype)
    planes = build_planes_batch(batch_size=batch_size, input_format=input_format)
    return {
        "name": f"nnx_encoder_backward_{compute_dtype}_bs{batch_size}",
        "step_fn": jit_encoder_loss_and_grad,
        "args": (model, np.asarray(planes, dtype=np.dtype(dtype))),
        "kwargs": {},
    }


def make_jepa_train_target(
    *,
    batch_size: int,
    models_dir: str | None = None,
    pb: str | None = None,
    input_format: str = "INPUT_CLASSICAL_112_PLANE",
    compute_dtype: str = "float16",
    seed: int = 0,
    token_dim: int = 256,
    num_layers: int = 4,
    num_heads: int = 8,
    mlp_dim: int = 1024,
    head_param_dtype: str = "float32",
    head_compute_dtype: str | None = None,
) -> dict:
    params = load_mapped_bt4_params(models_dir=models_dir, pb=pb)
    synthetic_batch = build_synthetic_transition_batch(batch_size)
    head_compute_dtype = (
        head_compute_dtype
        if head_compute_dtype is not None
        else ("bfloat16" if compute_dtype in {"bfloat16", "bf16"} else "float32")
    )
    config = JEPAConfig(
        encoder_dtype=compute_dtype,
        head_param_dtype=head_param_dtype,
        head_compute_dtype=head_compute_dtype,
        token_dim=token_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        mlp_dim=mlp_dim,
    )
    model, optimizer = create_jepa_components(params, config, seed=seed)
    batch = build_transition_batch(synthetic_batch, action_source=config.action_source)
    return {
        "name": (
            f"jepa_train_{compute_dtype}_bs{batch_size}_d{token_dim}"
            f"_l{num_layers}_h{num_heads}_m{mlp_dim}"
        ),
        "step_fn": train_step,
        "args": (model, optimizer, batch),
        "kwargs": {},
    }


__all__ = [
    "build_planes_batch",
    "load_mapped_bt4_params",
    "parse_compute_dtype",
    "make_nnx_encoder_backward_target",
    "make_nnx_encoder_forward_target",
    "make_jepa_train_target",
    "make_reference_forward_target",
]
