#!/usr/bin/env python3
"""One-file PyTorch BT4/DFM/JEPA autoresearch trainer.

This file intentionally keeps the production model, objective, optimizer,
training loop, and bounded systems controls together.  Eager execution remains
the numerical baseline; optional regional ``torch.compile`` and profiler
windows are explicit command-line choices.  JAX remains the frozen
numerical/checkpoint/evaluation oracle until every migration gate passes.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import gc
import gzip
import hashlib
import hmac
import json
import math
import os
import pickle
import shutil
import stat
import subprocess
import sys
import tempfile
import time
import zipfile
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import nullcontext
from datetime import UTC, datetime
from pathlib import Path
from typing import IO, Any, NamedTuple

import chess
import ml_dtypes
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from chess_dfm_jax.encoding import TOTAL_PLANES, encode_board  # noqa: E402
from chess_dfm_jax.policy import (  # noqa: E402
    ACTION_CODEC_LC0_CANONICAL_1858,
    ACTION_VOCAB_SIZE,
    LC0_CANONICAL_1858_INPUT_FORMAT,
    ActionCodecError,
    legal_action_mask,
)
from research.arena_history_trust import (  # noqa: E402
    HISTORY_VALIDATION_FULL_REPLAY,
    HISTORY_VALIDATION_SCHEMA,
    HISTORY_VALIDATION_TRUSTED_ARENA_ENDPOINT,
    TrustedArenaHistoryEndpoint,
    verify_trusted_arena_history_endpoint,
)

_WORKSPACE_ROOT = Path("/mountpoint/.exp")
_SOURCE_STATE = (
    _REPO_ROOT
    / "checkpoints"
    / "source"
    / "step0265000"
    / "checkpoints"
    / "step0265000"
    / "state.npz"
)
_DATA_ROOT = _REPO_ROOT / "data" / "trajectory_v3"
_RAW_BT4_PATH = (
    _REPO_ROOT / "models" / "source" / "extracted" / "BT4_exported.pb.gz"
)
_HERO_EVAL_MANIFEST = (
    _REPO_ROOT / "research" / "eval" / "hero_epoch_v1" / "manifest.json"
)
_HERO_FAST_VALIDATION_PERCENTAGES = tuple(range(10, 101, 10))
_HERO_ARENA_PERCENTAGES = (25, 50, 75)
_HERO_ARENA_PAIRS = 16
_HERO_ARENA_ADDITIONAL_PLY_CAP = 256
_HERO_ARENA_INFERENCE_BATCH_SIZE = 16
_RAW_BT4_SIZE_BYTES = 335_916_563
_RAW_BT4_SHA256 = "61e43e98d2c4cb747498c6bc13a5bbb3b07e985d3f5d87f0351a4db855fa0651"
_SOURCE_SIZE_BYTES = 1_851_704_172
_SOURCE_SHA256 = "16a3c7e77e411a8a7577ff04dac1ca5173ce24ecb343b5fa4938e2d2b5fb8906"
_SOURCE_STEP = 265_000
_EXPECTED_MODEL_LEAVES = 455
_EXPECTED_MODEL_BYTES = 705_987_352
_MASK_TOKEN = 1858
_VOCAB_SIZE = 1858
_LEGAL_PAD = np.iinfo(np.uint16).max
_HASH_CHUNK_BYTES = 8 * 1024 * 1024
_LOSS_SUMMARY_WINDOW_UPDATES = 64
_LOSS_SUMMARY_METRICS = (
    "loss",
    "unclipped_loss",
    "dfm_ce_loss",
    "accuracy",
    "first_legality_loss",
    "weighted_legality_loss",
    "first_legal_mass",
    "jepa_positive_loss",
    "jepa_sigreg_loss",
    "jepa_pred_sigreg_loss",
    "z_state_norm",
    "z_pred_norm",
    "z_target_norm",
    "learning_rate",
    "bt4_learning_rate",
    "gradient_global_norm",
    "gradient_clip_scale",
)


# AUTORESEARCH EDIT SURFACE. Keep systems/parity controls below unchanged.
@dataclasses.dataclass(frozen=True)
class Config:
    horizon: int = 8
    token_dim: int = 256
    z_dim: int = 1024
    projector_layers: int = 2
    projector_heads: int = 8
    projector_mlp_dim: int = 4096
    dfm_layers: int = 4
    dfm_heads: int = 4
    dfm_mlp_dim: int = 1024
    jepa_layers: int = 4
    jepa_mlp_dim: int = 4096
    target_sample_count: int = 1
    sigreg_example_count: int = 64
    sigreg_proj_dim: int = 1024
    dfm_ce_coeff: float = 1.0
    legality_coeff: float = 2.0
    jepa_positive_coeff: float = 1.0
    target_sigreg_coeff: float = 5.76
    pred_sigreg_coeff: float = 1.0
    sigreg_reference_count: float = 1.0
    loss_clip_value: float = 20.0
    delta_rms_clip: float = 0.5
    learning_rate: float = 3e-5
    bt4_learning_rate: float = 1e-6
    weight_decay: float = 1e-4
    grad_clip_norm: float = 1.0
    lr_decay_start: int = 400
    lr_decay_steps: int = 800
    lr_min_ratio: float = 0.1
    future_trainable_tail_layers: int = 1
    use_qk_norm: bool = True
    use_xsa: bool = True
    remat_blocks: bool = True
    remat_bt4_blocks: bool | None = None
    remat_projector_blocks: bool | None = None
    remat_dfm_blocks: bool | None = None
    use_bt4_sdpa: bool = False
    use_head_sdpa: bool = False
    action_codec: str = "legacy_absolute_1858"
    use_bt4_policy_residual: bool = False
    root_legal_ce_coeff: float = 0.0
    wdl_coeff: float = 0.0
    selective_weight_decay: bool = False
    lr_schedule_unit: str = "updates"
    lr_warmup_examples: int = 0
    lr_total_examples: int = 0
    init_seed: int = 0


CONFIG = Config()
_HERO_TRAIN_EXAMPLES = 28_343_296
# Deterministic update-zero batch-1024 root CE was 2.870999574661255.
# This makes the weighted root legal-conditional contribution exactly 0.25.
_HERO_ROOT_LEGAL_CE_COEFFICIENT = 0.08707768618513193
_HERO_MAIN_LEARNING_RATE = 5e-4
_HERO_BT4_LEARNING_RATE = _HERO_MAIN_LEARNING_RATE / 30.0
HERO_CONFIG = dataclasses.replace(
    CONFIG,
    action_codec="lc0_canonical_1858",
    use_bt4_policy_residual=True,
    root_legal_ce_coeff=_HERO_ROOT_LEGAL_CE_COEFFICIENT,
    wdl_coeff=0.25,
    target_sigreg_coeff=2.0,
    pred_sigreg_coeff=2.0,
    loss_clip_value=0.0,
    learning_rate=_HERO_MAIN_LEARNING_RATE,
    bt4_learning_rate=_HERO_BT4_LEARNING_RATE,
    weight_decay=1e-2,
    selective_weight_decay=True,
    lr_schedule_unit="examples",
    lr_warmup_examples=round(0.02 * _HERO_TRAIN_EXAMPLES),
    lr_total_examples=_HERO_TRAIN_EXAMPLES,
    lr_min_ratio=1e-3,
)


def _resolved_remat(specific: bool | None, fallback: bool) -> bool:
    return fallback if specific is None else specific


def _require_workspace(path: str | os.PathLike[str], *, exists: bool = False) -> Path:
    resolved = Path(path).expanduser().resolve(strict=exists)
    try:
        resolved.relative_to(_WORKSPACE_ROOT.resolve(strict=True))
    except ValueError as exc:
        raise ValueError(f"Path escapes {_WORKSPACE_ROOT}: {resolved}") from exc
    return resolved


def _profile_scope(enabled: bool, name: str) -> Any:
    """Create a profiler annotation only inside an explicitly profiled step."""

    if not enabled:
        return nullcontext()
    return torch.autograd.profiler.record_function(name)


def _raw_parameter(shape: tuple[int, ...], dtype: torch.dtype) -> nn.Parameter:
    return nn.Parameter(torch.empty(shape, dtype=dtype))


class RawLinear(nn.Module):
    """Linear map stored in the JAX ``[input, output]`` orientation."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        dtype: torch.dtype,
        bias: bool = True,
    ):
        super().__init__()
        self.w = _raw_parameter((input_dim, output_dim), dtype)
        self.b = _raw_parameter((output_dim,), dtype) if bias else None

    def forward(self, x: Tensor, compute_dtype: torch.dtype) -> Tensor:
        out = x.to(compute_dtype) @ self.w.to(compute_dtype)
        if self.b is not None:
            out = out + self.b.to(compute_dtype)
        return out


class _ExactForwardNativeLayerNorm(torch.autograd.Function):
    """Keep eager forward rounding while using the native fused backward."""

    @staticmethod
    def forward(
        ctx: Any,
        x: Tensor,
        scale: Tensor,
        bias: Tensor,
        eps: float,
        compute_dtype: torch.dtype,
    ) -> Tensor:
        stats = x.float()
        mean = stats.mean(dim=-1, keepdim=True)
        centered = stats - mean
        variance = centered.square().mean(dim=-1, keepdim=True)
        rstd = torch.rsqrt(variance + float(eps))
        scale_f32 = scale.float()
        bias_f32 = bias.float()
        ctx.save_for_backward(
            stats,
            mean,
            rstd,
            scale_f32,
            bias_f32,
        )
        ctx.normalized_shape = (x.shape[-1],)
        ctx.input_dtype = x.dtype
        ctx.scale_dtype = scale.dtype
        ctx.bias_dtype = bias.dtype
        normalized = centered * rstd
        output = normalized * scale_f32 + bias_f32
        return output.to(compute_dtype)

    @staticmethod
    def backward(
        ctx: Any,
        grad_output: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, None, None]:
        stats, mean, rstd, scale_f32, bias_f32 = ctx.saved_tensors
        grad_input, grad_scale, grad_bias = (
            torch.ops.aten.native_layer_norm_backward.default(
                grad_output.float(),
                stats,
                ctx.normalized_shape,
                mean,
                rstd,
                scale_f32,
                bias_f32,
                (True, True, True),
            )
        )
        return (
            grad_input.to(ctx.input_dtype),
            grad_scale.to(ctx.scale_dtype),
            grad_bias.to(ctx.bias_dtype),
            None,
            None,
        )


class RawLayerNorm(nn.Module):
    def __init__(
        self,
        width: int,
        *,
        dtype: torch.dtype,
        eps: float = 1e-3,
        implementation: str = "eager",
    ):
        super().__init__()
        if implementation not in {
            "eager",
            "eager-fused-backward",
            "native-fp32",
            "native-bf16",
        }:
            raise ValueError(
                f"Unsupported raw LayerNorm implementation: {implementation!r}"
            )
        self.width = int(width)
        self.scale = _raw_parameter((width,), dtype)
        self.bias = _raw_parameter((width,), dtype)
        self.eps = float(eps)
        self.implementation = implementation

    def forward(self, x: Tensor, compute_dtype: torch.dtype) -> Tensor:
        if (
            self.implementation == "eager-fused-backward"
            and torch.is_grad_enabled()
        ):
            return _ExactForwardNativeLayerNorm.apply(
                x,
                self.scale,
                self.bias,
                self.eps,
                compute_dtype,
            )
        if self.implementation == "native-fp32":
            return F.layer_norm(
                x.float(),
                (self.width,),
                self.scale.float(),
                self.bias.float(),
                self.eps,
            ).to(compute_dtype)
        if self.implementation == "native-bf16":
            return F.layer_norm(
                x.to(compute_dtype),
                (self.width,),
                self.scale.to(compute_dtype),
                self.bias.to(compute_dtype),
                self.eps,
            )
        stats = x.float()
        mean = stats.mean(dim=-1, keepdim=True)
        variance = (stats - mean).square().mean(dim=-1, keepdim=True)
        normalized = (stats - mean) * torch.rsqrt(variance + self.eps)
        out = normalized * self.scale.float() + self.bias.float()
        return out.to(compute_dtype)


class RawRMSNorm(nn.Module):
    def __init__(self, width: int, *, dtype: torch.dtype, eps: float = 1e-6):
        super().__init__()
        self.scale = _raw_parameter((width,), dtype)
        self.eps = float(eps)

    def forward(self, x: Tensor, compute_dtype: torch.dtype) -> Tensor:
        stats = x.float()
        out = stats * torch.rsqrt(stats.square().mean(dim=-1, keepdim=True) + self.eps)
        return (out * self.scale.float()).to(compute_dtype)


class RawEmbedding(nn.Module):
    def __init__(self, count: int, width: int, *, dtype: torch.dtype):
        super().__init__()
        self.embedding = _raw_parameter((count, width), dtype)

    def forward(self, indices: Tensor, compute_dtype: torch.dtype) -> Tensor:
        return self.embedding.to(compute_dtype)[indices]


class BT4InputEmbedding(nn.Module):
    def __init__(self, *, norm_impl: str = "eager"):
        super().__init__()
        dtype = torch.bfloat16
        self.preproc = RawLinear(768, 32768, dtype=dtype)
        self.proj = RawLinear(624, 1024, dtype=dtype)
        self.ln = RawLayerNorm(
            1024,
            dtype=dtype,
            implementation=norm_impl,
        )
        self.mul_gate = _raw_parameter((64, 1024), dtype)
        self.add_gate = _raw_parameter((64, 1024), dtype)
        self.ffn1 = RawLinear(1024, 1536, dtype=dtype)
        self.ffn2 = RawLinear(1536, 1024, dtype=dtype)
        self.ffn_ln = RawLayerNorm(
            1024,
            dtype=dtype,
            implementation=norm_impl,
        )

    def forward(
        self,
        planes: Tensor,
        alpha: float,
        compute_dtype: torch.dtype,
    ) -> Tensor:
        x = planes.to(compute_dtype)
        if x.ndim == 3:
            x = x.unsqueeze(0)
        batch = x.shape[0]
        x = x.permute(0, 2, 3, 1).reshape(batch, 64, 112)
        position = self.preproc(x[:, :, :12].reshape(batch, 768), compute_dtype)
        position = position.reshape(batch, 64, 512)
        x = torch.cat((x, position), dim=-1).reshape(batch * 64, 624)
        x = F.mish(self.proj(x, compute_dtype))
        x = self.ln(x, compute_dtype).reshape(batch, 64, 1024)
        x = x * self.mul_gate.to(compute_dtype) + self.add_gate.to(compute_dtype)
        flat = x.reshape(batch * 64, 1024)
        ffn = self.ffn2(F.mish(self.ffn1(flat, compute_dtype)), compute_dtype)
        return self.ffn_ln(ffn * alpha + flat, compute_dtype).reshape(batch, 64, 1024)


class BT4Smolgen(nn.Module):
    def __init__(self, *, norm_impl: str = "eager"):
        super().__init__()
        dtype = torch.bfloat16
        self.compress = RawLinear(1024, 32, dtype=dtype, bias=False)
        self.dense1 = RawLinear(2048, 256, dtype=dtype)
        self.ln1 = RawLayerNorm(
            256,
            dtype=dtype,
            implementation=norm_impl,
        )
        self.dense2 = RawLinear(256, 8192, dtype=dtype)
        self.ln2 = RawLayerNorm(
            8192,
            dtype=dtype,
            implementation=norm_impl,
        )
        self.shared_w = _raw_parameter((256, 4096), dtype)

    def forward(self, x: Tensor, compute_dtype: torch.dtype) -> Tensor:
        batch = x.shape[0]
        value = self.compress(x, compute_dtype).reshape(batch, 2048)
        value = self.ln1(F.silu(self.dense1(value, compute_dtype)), compute_dtype)
        value = self.ln2(F.silu(self.dense2(value, compute_dtype)), compute_dtype)
        value = value.reshape(batch, 32, 256)
        return (value @ self.shared_w.to(compute_dtype)).reshape(batch, 32, 64, 64)


class BT4EncoderLayer(nn.Module):
    def __init__(
        self,
        *,
        use_sdpa: bool = False,
        norm_impl: str = "eager",
    ):
        super().__init__()
        self.use_sdpa = bool(use_sdpa)
        dtype = torch.bfloat16
        self.wq = _raw_parameter((1024, 1024), dtype)
        self.wq_b = _raw_parameter((1024,), dtype)
        self.wk = _raw_parameter((1024, 1024), dtype)
        self.wk_b = _raw_parameter((1024,), dtype)
        self.wv = _raw_parameter((1024, 1024), dtype)
        self.wv_b = _raw_parameter((1024,), dtype)
        self.wo = RawLinear(1024, 1024, dtype=dtype)
        self.ln_attn = RawLayerNorm(
            1024,
            dtype=dtype,
            implementation=norm_impl,
        )
        self.ffn1 = RawLinear(1024, 1536, dtype=dtype)
        self.ffn2 = RawLinear(1536, 1024, dtype=dtype)
        self.ln_ffn = RawLayerNorm(
            1024,
            dtype=dtype,
            implementation=norm_impl,
        )
        self.smolgen = BT4Smolgen(norm_impl=norm_impl)

    def forward(self, x: Tensor, alpha: float, compute_dtype: torch.dtype) -> Tensor:
        batch, sequence, _ = x.shape
        q = x @ self.wq.to(compute_dtype) + self.wq_b.to(compute_dtype)
        k = x @ self.wk.to(compute_dtype) + self.wk_b.to(compute_dtype)
        v = x @ self.wv.to(compute_dtype) + self.wv_b.to(compute_dtype)
        q = q.reshape(batch, sequence, 32, 32).transpose(1, 2)
        k = k.reshape(batch, sequence, 32, 32).transpose(1, 2)
        v = v.reshape(batch, sequence, 32, 32).transpose(1, 2)
        smolgen_bias = self.smolgen(x, compute_dtype)
        if self.use_sdpa:
            out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=smolgen_bias,
                dropout_p=0.0,
            )
        else:
            logits = (q @ k.transpose(-2, -1)) / math.sqrt(32.0)
            attention = F.softmax(logits + smolgen_bias, dim=-1)
            out = attention @ v
        out = out.transpose(1, 2).reshape(batch * sequence, 1024)
        out = self.wo(out, compute_dtype).reshape(batch, sequence, 1024)
        x = self.ln_attn(out * alpha + x, compute_dtype)
        flat = x.reshape(batch * sequence, 1024)
        ffn = self.ffn2(F.mish(self.ffn1(flat, compute_dtype)), compute_dtype)
        return self.ln_ffn(
            ffn.reshape(batch, sequence, 1024) * alpha + x,
            compute_dtype,
        )


class BT4PolicyHead(nn.Module):
    """Native BT4 attention-policy head in the source ``[input, output]`` layout."""

    def __init__(self):
        super().__init__()
        from chess_dfm_jax.policy import attention_policy_map

        dtype = torch.bfloat16
        self.dense1 = RawLinear(1024, 1024, dtype=dtype)
        self.q = RawLinear(1024, 1024, dtype=dtype)
        self.k = RawLinear(1024, 1024, dtype=dtype)
        self.prom_w = _raw_parameter((1024, 4), dtype)
        self.register_buffer(
            "mapping_table",
            torch.from_numpy(attention_policy_map().astype(np.int64, copy=False)),
            persistent=False,
        )

    def forward(self, x: Tensor, compute_dtype: torch.dtype) -> Tensor:
        batch = x.shape[0]
        policy = F.mish(self.dense1(x, compute_dtype))
        q = self.q(policy, compute_dtype).reshape(batch, 64, -1)
        k = self.k(policy, compute_dtype).reshape(batch, 64, -1)
        attention = (q @ k.transpose(1, 2)) * (1.0 / math.sqrt(k.shape[-1]))

        promotion = k[:, 56:64, :] @ self.prom_w.to(compute_dtype)
        promotion = promotion.transpose(1, 2)
        promotion = promotion[:, :3, :] + promotion[:, 3:4, :]
        promotion = promotion.transpose(1, 2).reshape(batch, 1, 24)

        pawn_logits = attention[:, 48:56, 56:64].reshape(batch, 64, 1)
        pawn_logits = torch.cat((pawn_logits, pawn_logits, pawn_logits), dim=2)
        pawn_logits = pawn_logits.reshape(batch, 8, 24)
        promotion = (pawn_logits + promotion).reshape(batch, 3, 64)

        policy = torch.cat((attention, promotion), dim=1).reshape(batch, 67 * 64)
        return policy.index_select(1, self.mapping_table)


class BT4Encoder(nn.Module):
    def __init__(
        self,
        *,
        include_policy_head: bool = False,
        use_sdpa: bool = False,
        norm_impl: str = "eager",
    ):
        super().__init__()
        self.embedding = BT4InputEmbedding(norm_impl=norm_impl)
        self.layers = nn.ModuleList(
            BT4EncoderLayer(
                use_sdpa=use_sdpa,
                norm_impl=norm_impl,
            )
            for _ in range(15)
        )
        self.policy_head = BT4PolicyHead() if include_policy_head else None
        self.alpha = float((2.0 * len(self.layers)) ** -0.25)

    def encode_current(
        self,
        planes: Tensor,
        *,
        compute_dtype: torch.dtype,
        remat: bool,
    ) -> Tensor:
        x = self.embedding(planes, self.alpha, compute_dtype)
        for layer in self.layers:
            if remat and torch.is_grad_enabled():
                x = checkpoint(
                    layer,
                    x,
                    self.alpha,
                    compute_dtype,
                    use_reentrant=False,
                )
            else:
                x = layer(x, self.alpha, compute_dtype)
        return x

    def encode_future_tail(
        self,
        planes: Tensor,
        *,
        compute_dtype: torch.dtype,
        trainable_tail_layers: int,
    ) -> Tensor:
        if not 0 <= trainable_tail_layers <= len(self.layers):
            raise ValueError(f"Invalid BT4 future tail: {trainable_tail_layers}")
        boundary = len(self.layers) - trainable_tail_layers
        with torch.no_grad():
            x = self.embedding(planes, self.alpha, compute_dtype)
            for layer in self.layers[:boundary]:
                x = layer(x, self.alpha, compute_dtype)
        x = x.detach()
        for layer in self.layers[boundary:]:
            x = layer(x, self.alpha, compute_dtype)
        return x


def _rounded_swiglu_dim(mlp_dim: int, multiple: int = 256) -> int:
    raw = max(1, int(round((2.0 / 3.0) * mlp_dim)))
    return ((raw + multiple - 1) // multiple) * multiple


class TransformerStack(nn.Module):
    """JAX-layout pre-RMSNorm transformer parameters with an eager loop."""

    def __init__(
        self,
        *,
        layers: int,
        width: int,
        heads: int,
        mlp_dim: int,
        use_qk_norm: bool,
        use_xsa: bool,
        use_sdpa: bool,
        remat: bool,
    ):
        super().__init__()
        self.layers = int(layers)
        self.width = int(width)
        self.heads = int(heads)
        self.head_dim = width // heads
        self.swiglu_dim = _rounded_swiglu_dim(mlp_dim)
        self.use_qk_norm = bool(use_qk_norm)
        self.use_xsa = bool(use_xsa)
        self.use_sdpa = bool(use_sdpa)
        self.remat = bool(remat)
        dtype = torch.float32
        self.attn_norm_scale = _raw_parameter((layers, width), dtype)
        self.mlp_norm_scale = _raw_parameter((layers, width), dtype)
        self.w_qkv = _raw_parameter((layers, width, 3 * width), dtype)
        self.b_qkv = _raw_parameter((layers, 3 * width), dtype)
        self.w_o = _raw_parameter((layers, width, width), dtype)
        self.b_o = _raw_parameter((layers, width), dtype)
        self.w_gate_up = _raw_parameter((layers, width, 2 * self.swiglu_dim), dtype)
        self.b_gate_up = _raw_parameter((layers, 2 * self.swiglu_dim), dtype)
        self.w_down = _raw_parameter((layers, self.swiglu_dim, width), dtype)
        self.b_down = _raw_parameter((layers, width), dtype)

    @staticmethod
    def _rms_norm(x: Tensor, scale: Tensor, compute_dtype: torch.dtype) -> Tensor:
        stats = x.float()
        normalized = stats * torch.rsqrt(stats.square().mean(dim=-1, keepdim=True) + 1e-6)
        return (normalized * scale.float()).to(compute_dtype)

    @staticmethod
    def _qk_norm(x: Tensor, compute_dtype: torch.dtype) -> Tensor:
        stats = x.float()
        return (stats * torch.rsqrt(stats.square().mean(dim=-1, keepdim=True) + 1e-6)).to(
            compute_dtype
        )

    def _layer(self, x: Tensor, layer: int, compute_dtype: torch.dtype) -> Tensor:
        batch, sequence, _ = x.shape
        hidden = self._rms_norm(x, self.attn_norm_scale[layer], compute_dtype)
        qkv = hidden.reshape(batch * sequence, self.width) @ self.w_qkv[layer].to(
            compute_dtype
        ) + self.b_qkv[layer].to(compute_dtype)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.reshape(batch, sequence, self.heads, self.head_dim).transpose(1, 2)
        k = k.reshape(batch, sequence, self.heads, self.head_dim).transpose(1, 2)
        v = v.reshape(batch, sequence, self.heads, self.head_dim).transpose(1, 2)
        if self.use_qk_norm:
            q = self._qk_norm(q, compute_dtype)
            k = self._qk_norm(k, compute_dtype)
        if self.use_sdpa:
            attention_heads = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=0.0,
            )
        else:
            attention = F.softmax(
                (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim),
                dim=-1,
            )
            attention_heads = attention @ v
        if self.use_xsa:
            value = v.float()
            value_direction = value * torch.rsqrt(value.square().sum(dim=-1, keepdim=True) + 1e-6)
            attention_f32 = attention_heads.float()
            projection = (attention_f32 * value_direction).sum(dim=-1, keepdim=True)
            attention_heads = (attention_f32 - projection * value_direction).to(compute_dtype)
        attention_out = attention_heads.transpose(1, 2).reshape(batch * sequence, self.width)
        attention_out = attention_out @ self.w_o[layer].to(compute_dtype) + self.b_o[layer].to(
            compute_dtype
        )
        x = (x + attention_out.reshape(batch, sequence, self.width)).to(compute_dtype)
        hidden = self._rms_norm(x, self.mlp_norm_scale[layer], compute_dtype)
        gate_up = hidden.reshape(batch * sequence, self.width) @ self.w_gate_up[layer].to(
            compute_dtype
        ) + self.b_gate_up[layer].to(compute_dtype)
        gate, up = gate_up.chunk(2, dim=-1)
        mlp = (F.silu(gate) * up) @ self.w_down[layer].to(compute_dtype)
        mlp = mlp + self.b_down[layer].to(compute_dtype)
        return (x + mlp.reshape(batch, sequence, self.width)).to(compute_dtype)

    def forward(self, x: Tensor, compute_dtype: torch.dtype) -> Tensor:
        for layer in range(self.layers):
            if self.remat and torch.is_grad_enabled():
                x = checkpoint(
                    self._layer,
                    x,
                    layer,
                    compute_dtype,
                    use_reentrant=False,
                )
            else:
                x = self._layer(x, layer, compute_dtype)
        return x


class StateProjector(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        dtype = torch.float32
        self.in_proj = _raw_parameter((1024, config.z_dim), dtype)
        self.in_bias = _raw_parameter((config.z_dim,), dtype)
        self.cls = _raw_parameter((1, config.z_dim), dtype)
        self.pos_embed = _raw_parameter((65, config.z_dim), dtype)
        self.blocks = TransformerStack(
            layers=config.projector_layers,
            width=config.z_dim,
            heads=config.projector_heads,
            mlp_dim=config.projector_mlp_dim,
            use_qk_norm=config.use_qk_norm,
            use_xsa=config.use_xsa,
            use_sdpa=config.use_head_sdpa,
            remat=_resolved_remat(
                config.remat_projector_blocks,
                config.remat_blocks,
            ),
        )

    def forward(self, tokens: Tensor, compute_dtype: torch.dtype) -> Tensor:
        batch = tokens.shape[0]
        squares = (
            tokens.to(compute_dtype).reshape(batch * 64, 1024) @ self.in_proj.to(compute_dtype)
            + self.in_bias.to(compute_dtype)
        ).reshape(batch, 64, -1)
        cls = self.cls.to(compute_dtype).expand(batch, 1, -1)
        sequence = torch.cat((cls, squares), dim=1)
        sequence = sequence + self.pos_embed.to(compute_dtype).unsqueeze(0)
        return self.blocks(sequence, compute_dtype)[:, 0, :].to(compute_dtype)


class ConditionedTransition(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        dtype = torch.float32
        swiglu = _rounded_swiglu_dim(config.jepa_mlp_dim)
        self.layers = config.jepa_layers
        self.swiglu_dim = swiglu
        self.delta_rms_clip = config.delta_rms_clip
        self.norm_scale = _raw_parameter((self.layers, config.z_dim), dtype)
        self.cond_w = _raw_parameter((self.layers, config.z_dim, 2 * config.z_dim), dtype)
        self.cond_b = _raw_parameter((self.layers, 2 * config.z_dim), dtype)
        self.w_gate_up = _raw_parameter((self.layers, config.z_dim, 2 * swiglu), dtype)
        self.b_gate_up = _raw_parameter((self.layers, 2 * swiglu), dtype)
        self.w_down = _raw_parameter((self.layers, swiglu, config.z_dim), dtype)
        self.b_down = _raw_parameter((self.layers, config.z_dim), dtype)

    def _layer(
        self,
        z: Tensor,
        condition: Tensor,
        layer: int,
        compute_dtype: torch.dtype,
    ) -> Tensor:
        shift_scale = condition.to(compute_dtype) @ self.cond_w[layer].to(
            compute_dtype
        ) + self.cond_b[layer].to(compute_dtype)
        shift, scale = shift_scale.chunk(2, dim=-1)
        stats = z.float()
        hidden = stats * torch.rsqrt(stats.square().mean(dim=-1, keepdim=True) + 1e-6)
        hidden = (hidden * self.norm_scale[layer].float()).to(compute_dtype)
        hidden = hidden * (1.0 + scale) + shift
        gate_up = hidden @ self.w_gate_up[layer].to(compute_dtype) + self.b_gate_up[layer].to(
            compute_dtype
        )
        gate, up = gate_up.chunk(2, dim=-1)
        delta = (F.silu(gate) * up) @ self.w_down[layer].to(compute_dtype) + self.b_down[layer].to(
            compute_dtype
        )
        if self.delta_rms_clip > 0.0:
            delta_f32 = delta.float()
            delta_rms = torch.sqrt(delta_f32.square().mean(dim=-1, keepdim=True) + 1e-6)
            delta = delta_f32 * torch.minimum(
                torch.ones_like(delta_rms),
                self.delta_rms_clip / delta_rms,
            )
        return (z + delta.to(z.dtype)).to(compute_dtype)

    def forward(
        self,
        z: Tensor,
        condition: Tensor,
        compute_dtype: torch.dtype,
    ) -> Tensor:
        for layer in range(self.layers):
            z = self._layer(z, condition, layer, compute_dtype)
        return z


class ValueWDLHead(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        dtype = torch.float32
        hidden = max(config.z_dim, config.jepa_mlp_dim // 2)
        self.w1 = _raw_parameter((config.z_dim, hidden), dtype)
        self.b1 = _raw_parameter((hidden,), dtype)
        self.value_w = _raw_parameter((hidden, 1), dtype)
        self.value_b = _raw_parameter((1,), dtype)
        self.wdl_w = _raw_parameter((hidden, 3), dtype)
        self.wdl_b = _raw_parameter((3,), dtype)

    def forward(
        self,
        z: Tensor,
        compute_dtype: torch.dtype,
    ) -> tuple[Tensor, Tensor]:
        hidden = F.mish(
            z.to(compute_dtype) @ self.w1.to(compute_dtype)
            + self.b1.to(compute_dtype)
        )
        value = hidden @ self.value_w.to(compute_dtype) + self.value_b.to(compute_dtype)
        wdl = hidden @ self.wdl_w.to(compute_dtype) + self.wdl_b.to(compute_dtype)
        return value.squeeze(-1), wdl


class JointModel(nn.Module):
    def __init__(
        self,
        config: Config = CONFIG,
        *,
        bt4_norm_impl: str = "eager",
    ):
        super().__init__()
        self.config = config
        self.bt4_norm_impl = bt4_norm_impl
        dtype = torch.float32
        self.encoder = BT4Encoder(
            include_policy_head=config.use_bt4_policy_residual,
            use_sdpa=config.use_bt4_sdpa,
            norm_impl=bt4_norm_impl,
        )
        self.state_projector = StateProjector(config)
        self.dfm_state_projector = RawLinear(1024, config.token_dim, dtype=dtype)
        self.jepa_action_embed = RawEmbedding(_VOCAB_SIZE + 1, config.z_dim, dtype=dtype)
        self.jepa_hidden_adapter = RawLinear(config.token_dim, config.z_dim, dtype=dtype)
        self.jepa_transition = ConditionedTransition(config)
        self.jepa_state_norm = RawRMSNorm(config.z_dim, dtype=dtype)
        self.value_wdl_head = ValueWDLHead(config)
        self.action_embed = RawEmbedding(_VOCAB_SIZE + 1, config.token_dim, dtype=dtype)
        self.time_embed1 = _raw_parameter((1, config.token_dim), dtype)
        self.time_embed2 = _raw_parameter((config.token_dim, config.token_dim), dtype)
        self.time_bias = _raw_parameter((config.token_dim,), dtype)
        self.pos_embed = _raw_parameter((config.horizon, config.token_dim), dtype)
        self.dfm_blocks = TransformerStack(
            layers=config.dfm_layers,
            width=config.token_dim,
            heads=config.dfm_heads,
            mlp_dim=config.dfm_mlp_dim,
            use_qk_norm=config.use_qk_norm,
            use_xsa=config.use_xsa,
            use_sdpa=config.use_head_sdpa,
            remat=_resolved_remat(
                config.remat_dfm_blocks,
                config.remat_blocks,
            ),
        )
        self.dfm_out_norm = RawRMSNorm(config.token_dim, dtype=dtype)
        self.out_proj = _raw_parameter((config.token_dim, _VOCAB_SIZE), dtype)
        self.out_bias = _raw_parameter((_VOCAB_SIZE,), dtype)

    def encode_selected(
        self,
        current_planes: Tensor,
        selected_future_planes: Tensor,
        compute_dtype: torch.dtype,
        *,
        profile_regions: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor | None]:
        with _profile_scope(profile_regions, "region::bt4_current"):
            current = self.encoder.encode_current(
                current_planes,
                compute_dtype=compute_dtype,
                remat=_resolved_remat(
                    self.config.remat_bt4_blocks,
                    self.config.remat_blocks,
                ),
            )
        with _profile_scope(profile_regions, "region::bt4_future"):
            future = self.encoder.encode_future_tail(
                selected_future_planes,
                compute_dtype=compute_dtype,
                trainable_tail_layers=self.config.future_trainable_tail_layers,
            )
        tokens = torch.stack((current, future), dim=1)
        batch = current.shape[0]
        with _profile_scope(profile_regions, "region::state_projector"):
            z_all = self.state_projector(
                tokens.reshape(batch * 2, 64, 1024),
                compute_dtype,
            ).reshape(batch, 2, self.config.z_dim)
        with _profile_scope(profile_regions, "region::dfm_state_projector"):
            z_dfm = self.dfm_state_projector(current, compute_dtype)
        with _profile_scope(profile_regions, "region::bt4_policy_head"):
            base_policy_logits = (
                None
                if self.encoder.policy_head is None
                else self.encoder.policy_head(current, compute_dtype)
            )
        return z_all, z_dfm, base_policy_logits

    def _time_embedding(self, t: Tensor, compute_dtype: torch.dtype) -> Tensor:
        hidden = F.relu(t.to(compute_dtype).unsqueeze(-1) @ self.time_embed1.to(compute_dtype))
        return hidden @ self.time_embed2.to(compute_dtype) + self.time_bias.to(compute_dtype)

    def planner(
        self,
        z_dfm: Tensor,
        action_tokens: Tensor,
        t: Tensor,
        compute_dtype: torch.dtype,
        *,
        base_root_logits: Tensor | None = None,
        return_hidden: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        horizon = action_tokens.shape[1]
        action = self.action_embed(action_tokens, compute_dtype)
        action = action + self.pos_embed[:horizon].to(compute_dtype).unsqueeze(0)
        action = action + self._time_embedding(t, compute_dtype).unsqueeze(1)
        sequence = torch.cat((z_dfm.to(compute_dtype), action), dim=1)
        sequence = self.dfm_blocks(sequence, compute_dtype)
        action_hidden = sequence[:, 64:, :]
        normalized = self.dfm_out_norm(action_hidden, compute_dtype)
        logits = normalized @ self.out_proj.to(compute_dtype) + self.out_bias.to(compute_dtype)
        if base_root_logits is not None:
            if not self.config.use_bt4_policy_residual:
                raise ValueError("Base root logits require the BT4 residual-policy recipe")
            if logits.shape[1] < 1 or base_root_logits.shape != logits[:, 0].shape:
                raise ValueError(
                    "Base root logits shape mismatch: "
                    f"{tuple(base_root_logits.shape)} versus {tuple(logits[:, 0].shape)}"
                )
            logits = torch.cat(
                (
                    logits[:, :1] + base_root_logits.to(logits.dtype).unsqueeze(1),
                    logits[:, 1:],
                ),
                dim=1,
            )
        if return_hidden:
            return logits, action_hidden
        return logits

    def jepa_rollout(
        self,
        z0: Tensor,
        actions: Tensor,
        action_hidden: Tensor,
        compute_dtype: torch.dtype,
    ) -> Tensor:
        z = z0.to(compute_dtype)
        predictions: list[Tensor] = []
        for horizon in range(actions.shape[1]):
            condition = self.jepa_action_embed(
                actions[:, horizon], compute_dtype
            ) + self.jepa_hidden_adapter(action_hidden[:, horizon], compute_dtype)
            z = self.jepa_transition(z, condition, compute_dtype)
            predictions.append(z)
        return torch.stack(predictions, dim=1)


class TorchArenaSelection(NamedTuple):
    """Minimal host-resident result consumed by the fail-closed arena."""

    action_indices: np.ndarray


def _canonical_arena_fen(board: chess.Board) -> str:
    return board.fen(en_passant="legal")


def _checked_arena_board(value: Any, *, row: int) -> chess.Board:
    if not isinstance(value, chess.Board):
        raise TypeError(f"boards[{row}] must be a python-chess Board")
    if value.chess960:
        raise ValueError(f"boards[{row}] must use standard chess")
    if not value.is_valid():
        raise ValueError(f"boards[{row}] is not a valid standard-chess position")
    checked = value.copy(stack=False)
    if checked.is_game_over(claim_draw=False):
        raise ValueError(f"boards[{row}] is locally terminal")
    return checked


def _matching_arena_transition(
    previous: chess.Board,
    target: chess.Board,
    *,
    row: int,
    history_index: int,
) -> chess.Move:
    target_fen = _canonical_arena_fen(target)
    matches: list[chess.Move] = []
    for move in previous.legal_moves:
        candidate = previous.copy(stack=False)
        candidate.push(move)
        if _canonical_arena_fen(candidate) == target_fen:
            matches.append(move)
    if len(matches) != 1:
        raise ValueError(
            f"histories[{row}][{history_index - 1}:{history_index + 1}] "
            f"must describe exactly one legal ply; found {len(matches)}"
        )
    return matches[0]


def _validate_torch_arena_history(
    original_board: chess.Board,
    checked_board: chess.Board,
    history: Any,
    *,
    row: int,
    mode: str,
) -> None:
    if mode == HISTORY_VALIDATION_TRUSTED_ARENA_ENDPOINT:
        try:
            endpoint = verify_trusted_arena_history_endpoint(history)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"trusted arena endpoint for row {row} is invalid"
            ) from exc
        if original_board.move_stack:
            raise ValueError(
                f"boards[{row}] must be a stackless trusted arena copy"
            )
        if endpoint.current_fen != _canonical_arena_fen(checked_board):
            raise ValueError(
                f"trusted arena endpoint must match boards[{row}]"
            )
        if endpoint.authoritative_move_stack_length != checked_board.ply():
            raise ValueError(
                f"trusted arena endpoint ply must match boards[{row}]"
            )
        if endpoint.position_count != checked_board.ply() + 1:
            raise ValueError(
                f"trusted arena endpoint position count must match boards[{row}]"
            )
        return
    if mode != HISTORY_VALIDATION_FULL_REPLAY:
        raise ValueError(f"Unsupported arena history mode: {mode!r}")
    if isinstance(history, (str, bytes, chess.Board)):
        raise TypeError(f"histories[{row}] must be a sequence of boards")
    try:
        positions = tuple(history)
    except TypeError as exc:
        raise TypeError(f"histories[{row}] must be iterable") from exc
    if not positions:
        raise ValueError(f"histories[{row}] must not be empty")
    checked_positions = tuple(
        _checked_arena_board(position, row=row)
        for position in positions
    )
    replay = chess.Board()
    if _canonical_arena_fen(checked_positions[0]) != _canonical_arena_fen(replay):
        raise ValueError(
            f"histories[{row}] must start at the standard initial position"
        )
    for history_index, target in enumerate(checked_positions[1:], start=1):
        replay.push(
            _matching_arena_transition(
                replay,
                target,
                row=row,
                history_index=history_index,
            )
        )
    if _canonical_arena_fen(replay) != _canonical_arena_fen(checked_board):
        raise ValueError(f"histories[{row}] does not reproduce boards[{row}]")


def _torch_refine_dfm_actions(
    model: JointModel,
    z_dfm: Tensor,
    base_root_logits: Tensor,
    root_legal_mask: Tensor,
    *,
    refinement_passes: int,
    compute_dtype: torch.dtype,
) -> Tensor:
    """Run the frozen stable-rank iterative DFM decoder without JEPA feedback."""

    batch_size = z_dfm.shape[0]
    horizon = model.config.horizon
    if refinement_passes < 1:
        raise ValueError("refinement_passes must be positive")
    if root_legal_mask.shape != (batch_size, _VOCAB_SIZE):
        raise ValueError("root_legal_mask has the wrong physical shape")
    if root_legal_mask.dtype != torch.bool:
        raise TypeError("root_legal_mask must be boolean")
    if not bool(torch.all(root_legal_mask.any(dim=-1))):
        raise ValueError("root_legal_mask contains an empty row")
    action_tokens = torch.full(
        (batch_size, horizon),
        _MASK_TOKEN,
        dtype=torch.long,
        device=z_dfm.device,
    )
    for pass_index in range(refinement_passes):
        t = torch.full(
            (batch_size,),
            pass_index / refinement_passes,
            dtype=torch.float32,
            device=z_dfm.device,
        )
        logits = model.planner(
            z_dfm,
            action_tokens,
            t,
            compute_dtype,
            base_root_logits=base_root_logits,
        )
        assert isinstance(logits, Tensor)
        logits_f32 = logits.float()
        if not bool(torch.all(torch.isfinite(logits_f32))):
            raise FloatingPointError(
                f"DFM arena logits are non-finite at pass {pass_index}"
            )
        root_logits = torch.where(
            root_legal_mask,
            logits_f32[:, 0],
            torch.full_like(logits_f32[:, 0], -torch.inf),
        )
        selection_log_probs = torch.cat(
            (
                F.log_softmax(root_logits, dim=-1).unsqueeze(1),
                F.log_softmax(logits_f32[:, 1:], dim=-1),
            ),
            dim=1,
        )
        predictions = selection_log_probs.argmax(dim=-1)
        confidence = selection_log_probs.amax(dim=-1).exp()
        confidence = torch.where(
            action_tokens == _MASK_TOKEN,
            confidence,
            torch.full_like(confidence, torch.inf),
        )
        target_unmasked = (
            horizon * (pass_index + 1)
        ) // refinement_passes
        position_order = torch.argsort(
            -confidence,
            dim=-1,
            stable=True,
        )
        position_ranks = torch.argsort(
            position_order,
            dim=-1,
            stable=True,
        )
        action_tokens = torch.where(
            position_ranks < target_unmasked,
            predictions,
            action_tokens,
        )
    if bool(torch.any(action_tokens == _MASK_TOKEN)):
        raise RuntimeError(
            "Refinement passes did not unmask every action position"
        )
    root_actions = action_tokens[:, 0]
    if not bool(
        torch.all(
            torch.gather(
                root_legal_mask,
                1,
                root_actions.unsqueeze(1),
            )
        )
    ):
        raise RuntimeError("DFM refinement selected an illegal root action")
    return root_actions


@dataclasses.dataclass(frozen=True)
class TorchHeroArenaPolicy:
    """Native Torch adapter for live hero milestones and raw-BT4 controls."""

    model: JointModel
    model_id: str
    policy_mode: str
    inference_batch_size: int = _HERO_ARENA_INFERENCE_BATCH_SIZE
    refinement_passes: int = 8
    action_codec_id: str = dataclasses.field(
        default=ACTION_CODEC_LC0_CANONICAL_1858,
        init=False,
    )
    inference_batching_schema: str = dataclasses.field(
        default="chess-dfm-static-inference-batching-v1",
        init=False,
    )
    inference_padding_mode: str = dataclasses.field(
        default="repeat_first_validated_encoded_row_v1",
        init=False,
    )
    history_validation_mode: str = dataclasses.field(
        default=HISTORY_VALIDATION_TRUSTED_ARENA_ENDPOINT,
        init=False,
    )
    trusted_arena_history_schema: str = dataclasses.field(
        default=HISTORY_VALIDATION_SCHEMA,
        init=False,
    )

    def __post_init__(self) -> None:
        if not isinstance(self.model_id, str) or not self.model_id.strip():
            raise ValueError("model_id must be non-empty")
        if self.policy_mode not in {"dfm", "raw_bt4"}:
            raise ValueError("policy_mode must be 'dfm' or 'raw_bt4'")
        if (
            isinstance(self.inference_batch_size, bool)
            or self.inference_batch_size < 1
        ):
            raise ValueError("inference_batch_size must be positive")
        if self.model.config.action_codec != ACTION_CODEC_LC0_CANONICAL_1858:
            raise ValueError("Torch arena policy requires the canonical codec")
        if self.policy_mode == "dfm" and (
            self.refinement_passes != self.model.config.horizon
        ):
            raise ValueError(
                "The frozen hero arena requires one refinement pass per horizon"
            )

    def select_actions(
        self,
        boards: Sequence[chess.Board],
        histories: Sequence[Sequence[chess.Board]],
    ) -> TorchArenaSelection:
        return self._select_actions(
            boards,
            histories,
            mode=HISTORY_VALIDATION_FULL_REPLAY,
        )

    def select_actions_from_trusted_arena(
        self,
        boards: Sequence[chess.Board],
        endpoints: Sequence[TrustedArenaHistoryEndpoint],
    ) -> TorchArenaSelection:
        return self._select_actions(
            boards,
            endpoints,
            mode=HISTORY_VALIDATION_TRUSTED_ARENA_ENDPOINT,
        )

    def _select_actions(
        self,
        boards: Sequence[chess.Board],
        histories: Sequence[Any],
        *,
        mode: str,
    ) -> TorchArenaSelection:
        if isinstance(boards, (str, bytes, chess.Board)):
            raise TypeError("boards must be a non-empty sequence")
        board_items = tuple(boards)
        history_items = tuple(histories)
        if not board_items:
            raise ValueError("boards must not be empty")
        if len(history_items) != len(board_items):
            raise ValueError("histories must contain one entry per board")
        if len(board_items) > self.inference_batch_size:
            raise ValueError(
                f"active batch {len(board_items)} exceeds physical arena batch "
                f"{self.inference_batch_size}"
            )
        checked_boards = tuple(
            _checked_arena_board(board, row=row)
            for row, board in enumerate(board_items)
        )
        for row, (original, checked, history) in enumerate(
            zip(board_items, checked_boards, history_items, strict=True)
        ):
            _validate_torch_arena_history(
                original,
                checked,
                history,
                row=row,
                mode=mode,
            )

        plane_rows: list[np.ndarray] = []
        mask_rows: list[np.ndarray] = []
        for row, board in enumerate(checked_boards):
            planes = np.asarray(
                encode_board(
                    board,
                    [],
                    planes_layout="nchw",
                    input_format=LC0_CANONICAL_1858_INPUT_FORMAT,
                )
            )
            if planes.shape != (TOTAL_PLANES, 8, 8):
                raise ValueError(
                    f"encoded planes for row {row} have shape {planes.shape}"
                )
            if planes.dtype != np.dtype(np.float32) or not np.all(
                np.isfinite(planes)
            ):
                raise ValueError(
                    f"encoded planes for row {row} are not finite float32"
                )
            try:
                mask = np.asarray(
                    legal_action_mask(
                        board,
                        codec_id=ACTION_CODEC_LC0_CANONICAL_1858,
                        input_format=LC0_CANONICAL_1858_INPUT_FORMAT,
                    )
                )
            except (
                ActionCodecError,
                IndexError,
                KeyError,
                TypeError,
                ValueError,
            ) as exc:
                raise ValueError(
                    f"could not construct root legality mask for row {row}"
                ) from exc
            if (
                mask.shape != (ACTION_VOCAB_SIZE,)
                or mask.dtype != np.dtype(np.bool_)
                or int(mask.sum()) != board.legal_moves.count()
                or not bool(mask.any())
            ):
                raise ValueError(
                    f"canonical legality mask failed coverage for row {row}"
                )
            plane_rows.append(planes)
            mask_rows.append(mask)

        active_batch_size = len(checked_boards)
        physical_batch_size = self.inference_batch_size
        current_planes = np.stack(plane_rows)
        root_legal_mask = np.stack(mask_rows)
        if active_batch_size < physical_batch_size:
            padding = physical_batch_size - active_batch_size
            current_planes = np.concatenate(
                (
                    current_planes,
                    np.repeat(current_planes[:1], padding, axis=0),
                )
            )
            root_legal_mask = np.concatenate(
                (
                    root_legal_mask,
                    np.repeat(root_legal_mask[:1], padding, axis=0),
                )
            )

        device = next(self.model.parameters()).device
        planes_tensor = torch.from_numpy(
            np.ascontiguousarray(current_planes)
        ).to(device)
        legal_tensor = torch.from_numpy(
            np.ascontiguousarray(root_legal_mask)
        ).to(device)
        with torch.inference_mode():
            tokens = self.model.encoder.encode_current(
                planes_tensor,
                compute_dtype=torch.bfloat16,
                remat=False,
            )
            if self.model.encoder.policy_head is None:
                raise RuntimeError("Torch hero arena model has no policy head")
            base_root_logits = self.model.encoder.policy_head(
                tokens,
                torch.bfloat16,
            )
            if self.policy_mode == "raw_bt4":
                if not bool(torch.all(torch.isfinite(base_root_logits.float()))):
                    raise FloatingPointError("Raw BT4 arena logits are non-finite")
                selected = torch.where(
                    legal_tensor,
                    base_root_logits.float(),
                    torch.full_like(base_root_logits.float(), -torch.inf),
                ).argmax(dim=-1)
            else:
                z_dfm = self.model.dfm_state_projector(
                    tokens,
                    torch.bfloat16,
                )
                selected = _torch_refine_dfm_actions(
                    self.model,
                    z_dfm,
                    base_root_logits,
                    legal_tensor,
                    refinement_passes=self.refinement_passes,
                    compute_dtype=torch.bfloat16,
                )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            actions = (
                selected[:active_batch_size]
                .to(dtype=torch.int32)
                .cpu()
                .numpy()
            )
        if actions.shape != (active_batch_size,):
            raise RuntimeError("Torch arena adapter returned the wrong shape")
        for row, action in enumerate(actions):
            if not root_legal_mask[row, int(action)]:
                raise RuntimeError(
                    f"Torch arena adapter selected an illegal action for row {row}"
                )
        actions.flags.writeable = False
        return TorchArenaSelection(action_indices=actions)


class StepChoices(NamedTuple):
    target_horizon: Tensor
    training_time: Tensor
    mask_uniform: Tensor
    sigreg_indices: Tensor
    sigreg_directions: Tensor


@dataclasses.dataclass(frozen=True)
class _PreparedTrainingStep:
    update: int
    data_cursor: int
    compact_batch: dict[str, Any]
    choices: StepChoices
    prepare_seconds: float


def materialize_step_choices(
    *,
    seed: int,
    update: int,
    batch_size: int,
    config: Config,
    device: torch.device,
) -> StepChoices:
    """Materialize every stochastic choice so another framework can reuse it."""

    sequence = np.random.SeedSequence([int(seed), int(update)])
    rng = np.random.Generator(np.random.PCG64(sequence))
    target_horizon = np.arange(batch_size, dtype=np.int64) % config.horizon
    rng.shuffle(target_horizon)
    training_time = rng.random(batch_size, dtype=np.float32)
    mask_uniform = rng.random((batch_size, config.horizon), dtype=np.float32)
    sigreg_indices = rng.permutation(batch_size)[
        : min(config.sigreg_example_count, batch_size)
    ].astype(np.int64, copy=False)
    directions = rng.standard_normal((config.z_dim, config.sigreg_proj_dim), dtype=np.float32)
    directions /= np.maximum(np.linalg.norm(directions, axis=0, keepdims=True), 1e-12)
    return StepChoices(
        target_horizon=torch.from_numpy(target_horizon).to(device),
        training_time=torch.from_numpy(training_time).to(device),
        mask_uniform=torch.from_numpy(mask_uniform).to(device),
        sigreg_indices=torch.from_numpy(sigreg_indices).to(device),
        sigreg_directions=torch.from_numpy(directions).to(device),
    )


def _weighted_mean(values: Tensor, weight: Tensor) -> Tensor:
    return (values * weight).sum() / weight.sum().clamp_min(1.0)


def _legal_mass(
    probabilities: Tensor,
    legal_idx: Tensor,
    legal_count: Tensor,
) -> Tensor:
    safe = legal_idx.long().clamp(0, probabilities.shape[-1] - 1)
    gathered = torch.gather(probabilities, -1, safe)
    slots = torch.arange(safe.shape[-1], device=safe.device)
    valid = slots < legal_count.long().unsqueeze(-1)
    return torch.where(valid, gathered, 0.0).sum(dim=-1).clamp(0.0, 1.0)


def _legal_conditional_ce(
    logits: Tensor,
    targets: Tensor,
    legal_idx: Tensor,
    legal_count: Tensor,
) -> Tensor:
    """Per-example CE after renormalizing logits over the complete legal set."""

    safe = legal_idx.long().clamp(0, logits.shape[-1] - 1)
    legal_logits = torch.gather(logits.float(), -1, safe)
    slots = torch.arange(safe.shape[-1], device=safe.device)
    valid_slots = slots < legal_count.long().unsqueeze(-1)
    legal_logits = torch.where(
        valid_slots,
        legal_logits,
        torch.full_like(legal_logits, -torch.inf),
    )
    target_logits = torch.gather(
        logits.float(),
        -1,
        targets.long().unsqueeze(-1),
    ).squeeze(-1)
    result = torch.logsumexp(legal_logits, dim=-1) - target_logits
    return torch.where(legal_count > 0, result, torch.zeros_like(result))


def _uniform_horizon_mean(values: Tensor, weight: Tensor) -> Tensor:
    denominator = weight.sum(dim=0)
    means = (values * weight).sum(dim=0) / denominator.clamp_min(1.0)
    active = (denominator > 0).float()
    return (means * active).sum() / active.sum().clamp_min(1.0)


def _sigreg_v_stat(
    z: Tensor,
    sample_weight: Tensor,
    directions: Tensor,
    *,
    reference_count: float,
) -> tuple[Tensor, Tensor]:
    """Fixed-reference LeJEPA ECF V-statistic and valid weighted count."""

    z = z.float()
    weight = sample_weight.float().clamp_min(0.0)
    projected = z @ directions.float()
    t = torch.linspace(0.0, 3.0, 17, device=z.device, dtype=torch.float32)
    dt = 3.0 / 16.0
    quadrature = torch.full((17,), 2.0 * dt, device=z.device)
    quadrature[0] = dt
    quadrature[-1] = dt
    phi = torch.exp(-0.5 * t.square())
    quadrature = quadrature * phi
    xt = projected.unsqueeze(-1) * t
    expanded_weight = weight[:, None, None]
    count = weight.sum()
    denominator = count.clamp_min(1.0)
    cosine = (torch.cos(xt) * expanded_weight).sum(dim=0) / denominator
    sine = (torch.sin(xt) * expanded_weight).sum(dim=0) / denominator
    error = (cosine - phi).square() + sine.square()
    discrepancy = (error @ quadrature).mean()
    return discrepancy * float(reference_count), count


def loss_and_aux(
    model: JointModel,
    batch: Mapping[str, Tensor],
    choices: StepChoices,
    *,
    compute_dtype: torch.dtype,
    capture: dict[str, Any] | None = None,
    profile_regions: bool = False,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Accepted no-norm, target/pred-SIGReg objective."""

    config = model.config
    actions = batch["action_indices"][:, : config.horizon].long()
    valid = batch["valid"].float()
    future_valid = batch["future_valid"][:, : config.horizon].float()
    batch_size = actions.shape[0]
    rows = torch.arange(batch_size, device=actions.device)
    selected = choices.target_horizon
    selected_planes = (
        batch["selected_future_planes"]
        if "selected_future_planes" in batch
        else batch["future_planes"][rows, selected]
    )
    selected_valid = future_valid[rows, selected].unsqueeze(1)
    z_all, z_dfm, base_policy_logits = model.encode_selected(
        batch["current_planes"],
        selected_planes,
        compute_dtype,
        profile_regions=profile_regions,
    )
    z_jepa = z_all[:, 0]
    target_z = z_all[:, 1:]

    t = choices.training_time
    is_masked = choices.mask_uniform < (1.0 - t).unsqueeze(1)
    noisy_actions = torch.where(is_masked, torch.full_like(actions, _MASK_TOKEN), actions)
    with _profile_scope(profile_regions, "region::dfm_noisy_planner"):
        logits = model.planner(
            z_dfm,
            noisy_actions,
            t,
            compute_dtype,
            base_root_logits=base_policy_logits,
        )
    assert isinstance(logits, Tensor)
    if capture is not None:
        capture["root_logits"] = logits[:, 0].detach()
    log_probabilities = F.log_softmax(logits, dim=-1)
    ce = -torch.gather(log_probabilities, -1, actions.unsqueeze(-1)).squeeze(-1)
    ce_weight = is_masked.float() * valid.unsqueeze(1)
    ce_den_horizon = ce_weight.sum(dim=0)
    ce_by_horizon = (ce * ce_weight).sum(dim=0) / ce_den_horizon.clamp_min(1.0)
    active_horizon = (ce_den_horizon > 0).float()
    dfm_ce = (ce_by_horizon * active_horizon).sum() / active_horizon.sum().clamp_min(1.0)

    first_probability = F.softmax(logits[:, 0].float(), dim=-1)
    first_legal_mass = _legal_mass(
        first_probability,
        batch["legal_idx"][:, 0],
        batch["legal_count"][:, 0],
    )
    legal_valid = torch.ones_like(valid)
    if "legal_masks_valid" in batch:
        legal_valid = batch["legal_masks_valid"][:, 0].float()
    legal_gate = valid * legal_valid * is_masked[:, 0].float()
    legality = _weighted_mean(1.0 - first_legal_mass, legal_gate)
    root_legal_ce = _weighted_mean(
        _legal_conditional_ce(
            logits[:, 0],
            actions[:, 0],
            batch["legal_idx"][:, 0],
            batch["legal_count"][:, 0],
        ),
        legal_gate,
    )

    clean_t = torch.ones(batch_size, device=actions.device, dtype=torch.float32)
    with _profile_scope(profile_regions, "region::dfm_clean_planner"):
        clean_result = model.planner(
            z_dfm,
            actions,
            clean_t,
            compute_dtype,
            base_root_logits=base_policy_logits,
            return_hidden=True,
        )
    assert isinstance(clean_result, tuple)
    _, clean_hidden = clean_result
    with _profile_scope(profile_regions, "region::jepa_rollout"):
        pred_z = model.jepa_rollout(
            z_jepa,
            actions,
            clean_hidden,
            compute_dtype,
        )
    pred_for_loss = pred_z[rows, selected].unsqueeze(1)
    sample_raw_mse = (pred_for_loss.float() - target_z.float()).square().mean(dim=-1)
    positive_weight = valid.unsqueeze(1) * selected_valid
    jepa_positive = _weighted_mean(sample_raw_mse, positive_weight)

    sigreg_rows = choices.sigreg_indices
    sigreg_valid = valid[sigreg_rows]
    target_sigreg_z = z_all[sigreg_rows].float().reshape(-1, config.z_dim)
    target_future_weight = (
        sigreg_valid.unsqueeze(1)
        * selected_valid[sigreg_rows]
        * float(config.horizon / config.target_sample_count)
    )
    target_weight = torch.cat((sigreg_valid.unsqueeze(1), target_future_weight), dim=1).reshape(-1)
    with _profile_scope(profile_regions, "region::target_sigreg"):
        target_sigreg, target_sigreg_count = _sigreg_v_stat(
            target_sigreg_z,
            target_weight,
            choices.sigreg_directions,
            reference_count=config.sigreg_reference_count,
        )
    pred_sigreg_z = pred_z[sigreg_rows].float().reshape(-1, config.z_dim)
    pred_weight = (future_valid[sigreg_rows] * sigreg_valid.unsqueeze(1)).reshape(-1)
    with _profile_scope(profile_regions, "region::prediction_sigreg"):
        pred_sigreg, pred_sigreg_count = _sigreg_v_stat(
            pred_sigreg_z,
            pred_weight,
            choices.sigreg_directions,
            reference_count=config.sigreg_reference_count,
        )
    if capture is not None and capture.get("capture_sigreg_inputs") is True:
        full_target_future_weight = (
            valid.unsqueeze(1)
            * selected_valid
            * float(config.horizon / config.target_sample_count)
        )
        capture["sigreg_inputs"] = {
            "target_z": z_all.detach(),
            "target_weight": torch.cat(
                (valid.unsqueeze(1), full_target_future_weight),
                dim=1,
            ).detach(),
            "prediction_z": pred_z.detach(),
            "prediction_weight": (
                future_valid * valid.unsqueeze(1)
            ).detach(),
            "directions": choices.sigreg_directions.detach(),
        }

    wdl_loss = torch.zeros((), device=actions.device, dtype=torch.float32)
    wdl_accuracy = torch.zeros_like(wdl_loss)
    wdl_expected_value_mse = torch.zeros_like(wdl_loss)
    wdl_valid_count = torch.zeros_like(wdl_loss)
    if config.wdl_coeff != 0.0:
        if "wdl_targets" not in batch:
            raise ValueError("Nonzero wdl_coeff requires per-horizon wdl_targets")
        with _profile_scope(profile_regions, "region::wdl_head"):
            _, wdl_logits = model.value_wdl_head(pred_z, compute_dtype)
        raw_wdl_targets = batch["wdl_targets"][:, : config.horizon].float()
        if raw_wdl_targets.shape != wdl_logits.shape:
            raise ValueError(
                "wdl_targets must match predicted WDL logits: "
                f"{tuple(raw_wdl_targets.shape)} != {tuple(wdl_logits.shape)}"
            )
        wdl_sum = raw_wdl_targets.sum(dim=-1, keepdim=True)
        wdl_targets = raw_wdl_targets / wdl_sum.clamp_min(1e-12)
        wdl_weight = (
            valid.unsqueeze(1)
            * future_valid
            * (wdl_sum[..., 0] > 0).float()
        )
        wdl_valid_count = wdl_weight.sum()
        wdl_log_probabilities = F.log_softmax(wdl_logits.float(), dim=-1)
        wdl_probabilities = wdl_log_probabilities.exp()
        sample_wdl_ce = -(wdl_targets * wdl_log_probabilities).sum(dim=-1)
        wdl_loss = _uniform_horizon_mean(sample_wdl_ce, wdl_weight)
        wdl_accuracy = _uniform_horizon_mean(
            (wdl_logits.argmax(dim=-1) == wdl_targets.argmax(dim=-1)).float(),
            wdl_weight,
        )
        expected_value = wdl_probabilities[..., 0] - wdl_probabilities[..., 2]
        target_value = wdl_targets[..., 0] - wdl_targets[..., 2]
        wdl_expected_value_mse = _uniform_horizon_mean(
            (expected_value - target_value).square(),
            wdl_weight,
        )

    if capture is not None:
        capture["loss_components"] = {
            "dfm_ce": dfm_ce,
            "root_legal_conditional_ce": root_legal_ce,
            "root_illegal_mass": legality,
            "jepa_raw_mse": jepa_positive,
            "target_sigreg": target_sigreg,
            "prediction_sigreg": pred_sigreg,
            "wdl_ce": wdl_loss,
        }

    unclipped = (
        config.dfm_ce_coeff * dfm_ce
        + config.root_legal_ce_coeff * root_legal_ce
        + config.legality_coeff * legality
        + config.jepa_positive_coeff * jepa_positive
        + config.target_sigreg_coeff * target_sigreg
        + config.pred_sigreg_coeff * pred_sigreg
        + config.wdl_coeff * wdl_loss
    ).float()
    if config.loss_clip_value > 0.0:
        stopped = unclipped.detach().clamp_min(1e-6)
        clip_scale = torch.minimum(
            torch.ones_like(stopped),
            torch.as_tensor(
                config.loss_clip_value,
                device=stopped.device,
                dtype=stopped.dtype,
            )
            / stopped,
        )
    else:
        clip_scale = torch.ones_like(unclipped)
    loss = unclipped * clip_scale
    accuracy = _weighted_mean((logits.argmax(dim=-1) == actions).float(), ce_weight)
    aux = {
        "loss": loss.detach(),
        "unclipped_loss": unclipped.detach(),
        "loss_clip_scale": clip_scale.detach(),
        "dfm_ce_loss": dfm_ce.detach(),
        "root_legal_conditional_ce": root_legal_ce.detach(),
        "weighted_root_legal_conditional_ce": (
            config.root_legal_ce_coeff * root_legal_ce
        ).detach(),
        "first_legality_loss": legality.detach(),
        "first_legal_mass": (1.0 - legality).detach(),
        "weighted_legality_loss": (config.legality_coeff * legality).detach(),
        "jepa_positive_loss": jepa_positive.detach(),
        "jepa_raw_mse": jepa_positive.detach(),
        "jepa_norm_loss": torch.zeros_like(jepa_positive),
        "jepa_sigreg_loss": target_sigreg.detach(),
        "jepa_pred_sigreg_loss": pred_sigreg.detach(),
        "jepa_sigreg_valid_count": target_sigreg_count.detach(),
        "jepa_pred_sigreg_valid_count": pred_sigreg_count.detach(),
        "wdl_loss": wdl_loss.detach(),
        "wdl_weighted_loss": (config.wdl_coeff * wdl_loss).detach(),
        "wdl_accuracy": wdl_accuracy.detach(),
        "wdl_expected_value_mse": wdl_expected_value_mse.detach(),
        "wdl_valid_count": wdl_valid_count.detach(),
        "accuracy": accuracy.detach(),
        "mask_prob": (1.0 - t).mean().detach(),
        "z_state_norm": z_all.float().norm(dim=-1).mean().detach(),
        "z_pred_norm": pred_z.float().norm(dim=-1).mean().detach(),
        "z_target_norm": target_z.float().norm(dim=-1).mean().detach(),
        "jepa_target_mean_horizon": (selected.float() + 1.0).mean().detach(),
    }
    return loss, aux


def _latent_spectrum_metrics(
    values: Tensor,
    sample_weight: Tensor,
) -> dict[str, Tensor]:
    """Per-horizon centered spectrum diagnostics in the sample Gram space."""

    if values.ndim != 3:
        raise ValueError(
            "Latent spectrum values must have shape [batch, horizon, dim]"
        )
    if sample_weight.shape != values.shape[:2]:
        raise ValueError(
            "Latent spectrum weights must match the batch and horizon dimensions"
        )
    weights = sample_weight.float()
    denominator = weights.sum(dim=0)
    safe_denominator = denominator.clamp_min(1.0)
    mean = (
        values.float() * weights.unsqueeze(-1)
    ).sum(dim=0) / safe_denominator.unsqueeze(-1)
    centered = values.float() - mean.unsqueeze(0)
    variance = (
        centered.square() * weights.unsqueeze(-1)
    ).sum(dim=0) / safe_denominator.unsqueeze(-1)
    feature_std = torch.sqrt(variance.clamp_min(0.0) + 1e-12)
    centered_rms = torch.sqrt(variance.mean(dim=-1).clamp_min(0.0))

    weighted_centered = centered.permute(1, 0, 2)
    weighted_centered = (
        weighted_centered
        * weights.transpose(0, 1).sqrt().unsqueeze(-1)
    )
    covariance_denominator = (denominator - 1.0).clamp_min(1.0)
    gram = weighted_centered @ weighted_centered.transpose(-2, -1)
    gram = gram / covariance_denominator[:, None, None]
    eigenvalues = torch.linalg.eigvalsh(gram).clamp_min(0.0)
    eigenvalue_sum = eigenvalues.sum(dim=-1)
    spectrum = eigenvalues / eigenvalue_sum.unsqueeze(-1).clamp_min(1e-12)
    entropy = -torch.where(
        spectrum > 0.0,
        spectrum * torch.log(spectrum),
        torch.zeros_like(spectrum),
    ).sum(dim=-1)
    effective_rank = torch.where(
        eigenvalue_sum > 1e-12,
        torch.exp(entropy),
        torch.zeros_like(entropy),
    )
    largest_eigenvalue = eigenvalues[..., -1]
    stable_rank = torch.where(
        largest_eigenvalue > 1e-12,
        eigenvalue_sum / largest_eigenvalue.clamp_min(1e-12),
        torch.zeros_like(eigenvalue_sum),
    )

    result = {
        "feature_std": feature_std,
        "centered_rms": centered_rms,
        "effective_rank": effective_rank,
        "stable_rank": stable_rank,
    }
    for count in (1, 8, 16, 32):
        bounded_count = min(count, eigenvalues.shape[-1])
        explained = eigenvalues[..., -bounded_count:].sum(dim=-1)
        result[f"explained_variance_top{count}"] = torch.where(
            eigenvalue_sum > 1e-12,
            explained / eigenvalue_sum.clamp_min(1e-12),
            torch.zeros_like(eigenvalue_sum),
        )
    return result


def full_horizon_evaluation_aux(
    model: JointModel,
    batch: Mapping[str, Tensor],
    choices: StepChoices,
    *,
    target_shuffle: Tensor,
    action_shuffle: Tensor,
    compute_dtype: torch.dtype,
) -> dict[str, Tensor]:
    """Frozen full-horizon validation and collapse diagnostics.

    Training deliberately encodes one balanced future target per physical
    example. Validation retains the historical contract: encode every future
    horizon, fully mask the action sequence at t=0, and diagnose the complete
    free-running JEPA rollout.
    """

    config = model.config
    actions = batch["action_indices"][:, : config.horizon].long()
    valid = batch["valid"].float()
    future_valid = batch["future_valid"][:, : config.horizon].float()
    future_planes = batch["future_planes"][:, : config.horizon]
    batch_size, horizon = actions.shape
    if horizon != config.horizon:
        raise ValueError(f"Evaluation horizon drift: {horizon} != {config.horizon}")
    if target_shuffle.shape != (batch_size,) or action_shuffle.shape != (batch_size,):
        raise ValueError("Evaluation permutations must have shape [batch]")

    current_tokens = model.encoder.encode_current(
        batch["current_planes"],
        compute_dtype=compute_dtype,
        remat=False,
    )
    base_policy_logits = (
        None
        if model.encoder.policy_head is None
        else model.encoder.policy_head(current_tokens, compute_dtype)
    )
    flat_future = future_planes.reshape(
        batch_size * horizon,
        *future_planes.shape[2:],
    )
    future_tokens = model.encoder.encode_future_tail(
        flat_future,
        compute_dtype=compute_dtype,
        trainable_tail_layers=config.future_trainable_tail_layers,
    ).reshape(batch_size, horizon, 64, 1024)
    all_tokens = torch.cat((current_tokens.unsqueeze(1), future_tokens), dim=1)
    z_all = model.state_projector(
        all_tokens.reshape(batch_size * (horizon + 1), 64, 1024),
        compute_dtype,
    ).reshape(batch_size, horizon + 1, config.z_dim)
    z_dfm = model.dfm_state_projector(current_tokens, compute_dtype)
    current_z = z_all[:, 0]
    target_z = z_all[:, 1:]

    t = torch.zeros(batch_size, device=actions.device, dtype=torch.float32)
    noisy_actions = torch.full_like(actions, _MASK_TOKEN)
    logits = model.planner(
        z_dfm,
        noisy_actions,
        t,
        compute_dtype,
        base_root_logits=base_policy_logits,
    )
    assert isinstance(logits, Tensor)
    log_probabilities = F.log_softmax(logits, dim=-1)
    ce = -torch.gather(
        log_probabilities,
        -1,
        actions.unsqueeze(-1),
    ).squeeze(-1)
    ce_weight = valid.unsqueeze(1).expand_as(ce)
    ce_den_horizon = ce_weight.sum(dim=0)
    ce_by_horizon = (ce * ce_weight).sum(dim=0) / ce_den_horizon.clamp_min(1.0)
    active_horizon = (ce_den_horizon > 0).float()
    dfm_ce = _weighted_mean(ce_by_horizon, active_horizon)

    first_probability = F.softmax(logits[:, 0].float(), dim=-1)
    first_legal_mass = _legal_mass(
        first_probability,
        batch["legal_idx"][:, 0],
        batch["legal_count"][:, 0],
    )
    legal_valid = torch.ones_like(valid)
    if "legal_masks_valid" in batch:
        legal_valid = batch["legal_masks_valid"][:, 0].float()
    legal_gate = valid * legal_valid
    legality = _weighted_mean(1.0 - first_legal_mass, legal_gate)
    root_legal_ce = _weighted_mean(
        _legal_conditional_ce(
            logits[:, 0],
            actions[:, 0],
            batch["legal_idx"][:, 0],
            batch["legal_count"][:, 0],
        ),
        legal_gate,
    )
    root_legal_positions = (
        torch.arange(
            batch["legal_idx"].shape[-1],
            device=actions.device,
        ).unsqueeze(0)
        < batch["legal_count"][:, 0].unsqueeze(1)
    )
    root_legal_mask_counts = torch.zeros(
        (batch_size, _VOCAB_SIZE),
        device=actions.device,
        dtype=torch.int32,
    )
    root_legal_mask_counts.scatter_add_(
        1,
        batch["legal_idx"][:, 0].long().clamp(0, _VOCAB_SIZE - 1),
        root_legal_positions.to(torch.int32),
    )
    root_legal_action = logits[:, 0].masked_fill(
        root_legal_mask_counts == 0,
        -torch.inf,
    ).argmax(dim=-1)
    root_legal_top1_accuracy = _weighted_mean(
        (root_legal_action == actions[:, 0]).float(),
        legal_gate,
    )

    clean_t = torch.ones(batch_size, device=actions.device, dtype=torch.float32)
    clean_result = model.planner(
        z_dfm,
        actions,
        clean_t,
        compute_dtype,
        base_root_logits=base_policy_logits,
        return_hidden=True,
    )
    assert isinstance(clean_result, tuple)
    _, clean_hidden = clean_result
    pred_z = model.jepa_rollout(
        current_z,
        actions,
        clean_hidden,
        compute_dtype,
    )

    latent_weight = valid.unsqueeze(1) * future_valid
    latent_denom = latent_weight.sum(dim=0).clamp_min(1.0)
    pred_f32 = pred_z.float()
    target_f32 = target_z.float()
    current_f32 = current_z.float()
    sample_mse = (pred_f32 - target_f32).square().mean(dim=-1)
    jepa_positive = _weighted_mean(sample_mse, latent_weight)
    jepa_mse_by_horizon = (sample_mse * latent_weight).sum(dim=0) / latent_denom

    sigreg_rows = choices.sigreg_indices
    sigreg_valid = valid[sigreg_rows]
    target_sigreg_z = z_all[sigreg_rows].float().reshape(-1, config.z_dim)
    target_weight = torch.cat(
        (
            sigreg_valid.unsqueeze(1),
            sigreg_valid.unsqueeze(1) * future_valid[sigreg_rows],
        ),
        dim=1,
    ).reshape(-1)
    target_sigreg, target_sigreg_count = _sigreg_v_stat(
        target_sigreg_z,
        target_weight,
        choices.sigreg_directions,
        reference_count=config.sigreg_reference_count,
    )
    pred_sigreg_z = pred_z[sigreg_rows].float().reshape(-1, config.z_dim)
    pred_weight = (future_valid[sigreg_rows] * sigreg_valid.unsqueeze(1)).reshape(-1)
    pred_sigreg, pred_sigreg_count = _sigreg_v_stat(
        pred_sigreg_z,
        pred_weight,
        choices.sigreg_directions,
        reference_count=config.sigreg_reference_count,
    )

    wdl_loss = torch.zeros((), device=actions.device, dtype=torch.float32)
    wdl_accuracy = torch.zeros_like(wdl_loss)
    wdl_expected_value_mse = torch.zeros_like(wdl_loss)
    wdl_expected_value_bias = torch.zeros_like(wdl_loss)
    wdl_brier_score = torch.zeros_like(wdl_loss)
    wdl_entropy = torch.zeros_like(wdl_loss)
    wdl_ece_15 = torch.zeros_like(wdl_loss)
    wdl_valid_count = torch.zeros_like(wdl_loss)
    if config.wdl_coeff != 0.0:
        if "wdl_targets" not in batch:
            raise ValueError("Nonzero wdl_coeff requires per-horizon wdl_targets")
        _, wdl_logits = model.value_wdl_head(pred_z, compute_dtype)
        raw_wdl_targets = batch["wdl_targets"][:, : config.horizon].float()
        if raw_wdl_targets.shape != wdl_logits.shape:
            raise ValueError(
                "wdl_targets must match predicted WDL logits: "
                f"{tuple(raw_wdl_targets.shape)} != {tuple(wdl_logits.shape)}"
            )
        wdl_sum = raw_wdl_targets.sum(dim=-1, keepdim=True)
        wdl_targets = raw_wdl_targets / wdl_sum.clamp_min(1e-12)
        wdl_weight = (
            valid.unsqueeze(1)
            * future_valid
            * (wdl_sum[..., 0] > 0).float()
        )
        wdl_valid_count = wdl_weight.sum()
        wdl_log_probabilities = F.log_softmax(wdl_logits.float(), dim=-1)
        wdl_probabilities = wdl_log_probabilities.exp()
        sample_wdl_ce = -(wdl_targets * wdl_log_probabilities).sum(dim=-1)
        wdl_loss = _uniform_horizon_mean(sample_wdl_ce, wdl_weight)
        wdl_accuracy = _uniform_horizon_mean(
            (wdl_logits.argmax(dim=-1) == wdl_targets.argmax(dim=-1)).float(),
            wdl_weight,
        )
        expected_value = wdl_probabilities[..., 0] - wdl_probabilities[..., 2]
        target_value = wdl_targets[..., 0] - wdl_targets[..., 2]
        wdl_expected_value_mse = _uniform_horizon_mean(
            (expected_value - target_value).square(),
            wdl_weight,
        )
        wdl_expected_value_bias = _uniform_horizon_mean(
            expected_value - target_value,
            wdl_weight,
        )
        wdl_brier_score = _uniform_horizon_mean(
            (wdl_probabilities - wdl_targets).square().sum(dim=-1),
            wdl_weight,
        )
        wdl_entropy = _uniform_horizon_mean(
            -(wdl_probabilities * wdl_log_probabilities).sum(dim=-1),
            wdl_weight,
        )
        confidence, predicted_class = wdl_probabilities.max(dim=-1)
        correctness = (
            predicted_class == wdl_targets.argmax(dim=-1)
        ).float()
        calibration_bin = torch.clamp(
            (confidence * 15.0).long(),
            min=0,
            max=14,
        )
        calibration_error_sum = torch.zeros_like(wdl_loss)
        for bin_index in range(15):
            bin_weight = wdl_weight * (calibration_bin == bin_index).float()
            bin_count = bin_weight.sum()
            bin_accuracy = (correctness * bin_weight).sum() / bin_count.clamp_min(
                1.0
            )
            bin_confidence = (confidence * bin_weight).sum() / bin_count.clamp_min(
                1.0
            )
            calibration_error_sum = calibration_error_sum + bin_count * torch.abs(
                bin_accuracy - bin_confidence
            )
        wdl_ece_15 = calibration_error_sum / wdl_valid_count.clamp_min(1.0)

    unclipped = (
        config.dfm_ce_coeff * dfm_ce
        + config.root_legal_ce_coeff * root_legal_ce
        + config.legality_coeff * legality
        + config.jepa_positive_coeff * jepa_positive
        + config.target_sigreg_coeff * target_sigreg
        + config.pred_sigreg_coeff * pred_sigreg
        + config.wdl_coeff * wdl_loss
    ).float()
    if config.loss_clip_value > 0.0:
        clip_scale = torch.minimum(
            torch.ones_like(unclipped),
            torch.as_tensor(
                config.loss_clip_value,
                device=unclipped.device,
                dtype=unclipped.dtype,
            )
            / unclipped.detach().clamp_min(1e-6),
        )
    else:
        clip_scale = torch.ones_like(unclipped)
    loss = unclipped * clip_scale
    accuracy = _weighted_mean(
        (logits.argmax(dim=-1) == actions).float(),
        ce_weight,
    )

    zero_mse = target_f32.square().mean(dim=-1)
    identity_mse = (current_f32.unsqueeze(1) - target_f32).square().mean(dim=-1)
    shuffled_mse = (pred_f32 - target_f32[target_shuffle]).square().mean(dim=-1)
    pred_norm = pred_f32.norm(dim=-1)
    target_norm = target_f32.norm(dim=-1)
    cosine = (pred_f32 * target_f32).sum(dim=-1) / (pred_norm * target_norm).clamp_min(1e-12)

    def horizon_mean(values: Tensor) -> Tensor:
        return (values * latent_weight).sum(dim=0) / latent_denom

    pred_spectrum = _latent_spectrum_metrics(
        pred_f32,
        latent_weight,
    )
    target_spectrum = _latent_spectrum_metrics(
        target_f32,
        latent_weight,
    )
    pred_rms_by_horizon = torch.sqrt(
        horizon_mean(pred_f32.square().mean(dim=-1))
    )
    target_rms_by_horizon = torch.sqrt(
        horizon_mean(target_f32.square().mean(dim=-1))
    )
    pred_target_rank_ratio = torch.where(
        target_spectrum["effective_rank"] > 1e-12,
        pred_spectrum["effective_rank"]
        / target_spectrum["effective_rank"].clamp_min(1e-12),
        torch.zeros_like(target_spectrum["effective_rank"]),
    )
    pred_target_centered_rms_ratio = torch.where(
        target_spectrum["centered_rms"] > 1e-12,
        pred_spectrum["centered_rms"]
        / target_spectrum["centered_rms"].clamp_min(1e-12),
        torch.zeros_like(target_spectrum["centered_rms"]),
    )

    action_shuffled_pred = model.jepa_rollout(
        current_z,
        actions[action_shuffle],
        clean_hidden[action_shuffle],
        compute_dtype,
    ).float()
    action_shuffled_mse = (action_shuffled_pred - target_f32).square().mean(dim=-1)

    return {
        "loss": loss,
        "unclipped_loss": unclipped,
        "loss_clip_scale": clip_scale,
        "dfm_ce_loss": dfm_ce,
        "dfm_ce_loss_by_horizon": ce_by_horizon,
        "root_legal_conditional_ce": root_legal_ce,
        "root_legal_top1_accuracy": root_legal_top1_accuracy,
        "weighted_root_legal_conditional_ce": (
            config.root_legal_ce_coeff * root_legal_ce
        ),
        "first_legality_loss": legality,
        "first_legal_mass": 1.0 - legality,
        "weighted_legality_loss": config.legality_coeff * legality,
        "jepa_positive_loss": jepa_positive,
        "jepa_raw_mse": jepa_positive,
        "jepa_raw_mse_by_horizon": jepa_mse_by_horizon,
        "jepa_sigreg_loss": target_sigreg,
        "jepa_pred_sigreg_loss": pred_sigreg,
        "jepa_sigreg_valid_count": target_sigreg_count,
        "jepa_pred_sigreg_valid_count": pred_sigreg_count,
        "wdl_loss": wdl_loss,
        "wdl_weighted_loss": config.wdl_coeff * wdl_loss,
        "wdl_accuracy": wdl_accuracy,
        "wdl_expected_value_mse": wdl_expected_value_mse,
        "wdl_expected_value_bias": wdl_expected_value_bias,
        "wdl_brier_score": wdl_brier_score,
        "wdl_entropy": wdl_entropy,
        "wdl_ece_15": wdl_ece_15,
        "wdl_valid_count": wdl_valid_count,
        "accuracy": accuracy,
        "mask_prob": torch.ones((), device=actions.device),
        "z_state_norm": z_all.float().norm(dim=-1).mean(),
        "z_state_std": z_all.float().std(unbiased=False),
        "z_pred_norm": pred_f32.norm(dim=-1).mean(),
        "z_target_norm": target_f32.norm(dim=-1).mean(),
        "jepa_mse_by_horizon": jepa_mse_by_horizon,
        "zero_mse_by_horizon": horizon_mean(zero_mse),
        "identity_mse_by_horizon": horizon_mean(identity_mse),
        "shuffled_mse_by_horizon": horizon_mean(shuffled_mse),
        "action_shuffled_mse_by_horizon": horizon_mean(action_shuffled_mse),
        "pred_target_cosine_by_horizon": horizon_mean(cosine),
        "pred_rms_by_horizon": pred_rms_by_horizon,
        "target_rms_by_horizon": target_rms_by_horizon,
        "pred_target_rms_ratio_by_horizon": (
            pred_rms_by_horizon / target_rms_by_horizon.clamp_min(1e-12)
        ),
        "pred_centered_rms_by_horizon": pred_spectrum["centered_rms"],
        "target_centered_rms_by_horizon": target_spectrum["centered_rms"],
        "pred_target_centered_rms_ratio_by_horizon": (
            pred_target_centered_rms_ratio
        ),
        "pred_feature_std_mean_by_horizon": pred_spectrum[
            "feature_std"
        ].mean(dim=-1),
        "pred_feature_std_p05_by_horizon": torch.quantile(
            pred_spectrum["feature_std"],
            0.05,
            dim=-1,
        ),
        "target_feature_std_mean_by_horizon": target_spectrum[
            "feature_std"
        ].mean(dim=-1),
        "target_feature_std_p05_by_horizon": torch.quantile(
            target_spectrum["feature_std"],
            0.05,
            dim=-1,
        ),
        "pred_effective_rank_by_horizon": pred_spectrum["effective_rank"],
        "target_effective_rank_by_horizon": target_spectrum["effective_rank"],
        "pred_target_effective_rank_ratio_by_horizon": pred_target_rank_ratio,
        "pred_stable_rank_by_horizon": pred_spectrum["stable_rank"],
        "target_stable_rank_by_horizon": target_spectrum["stable_rank"],
        "pred_explained_variance_top1_by_horizon": pred_spectrum[
            "explained_variance_top1"
        ],
        "pred_explained_variance_top8_by_horizon": pred_spectrum[
            "explained_variance_top8"
        ],
        "pred_explained_variance_top16_by_horizon": pred_spectrum[
            "explained_variance_top16"
        ],
        "pred_explained_variance_top32_by_horizon": pred_spectrum[
            "explained_variance_top32"
        ],
        "target_explained_variance_top1_by_horizon": target_spectrum[
            "explained_variance_top1"
        ],
        "target_explained_variance_top8_by_horizon": target_spectrum[
            "explained_variance_top8"
        ],
        "target_explained_variance_top16_by_horizon": target_spectrum[
            "explained_variance_top16"
        ],
        "target_explained_variance_top32_by_horizon": target_spectrum[
            "explained_variance_top32"
        ],
        "valid_count_by_horizon": latent_denom,
    }


@dataclasses.dataclass
class _OptimizerLeaf:
    name: str
    parameter: nn.Parameter
    learning_rate_kind: str
    use_muon: bool
    apply_weight_decay: bool
    first_moment: Tensor
    second_moment: Tensor | None


class MuonAdamW:
    """Torch implementation of the accepted Optax Muon/AdamW contract.

    Square-ish matrix leaves use five-step Newton--Schulz Muon with Nesterov
    momentum. Every other leaf uses the Nesterov AdamW branch selected by
    ``optax.contrib.muon``. Parameter tensors retain their source dtypes.
    """

    def __init__(
        self,
        model: nn.Module,
        config: Config = CONFIG,
        *,
        examples_per_update: int | None = None,
    ):
        self.config = config
        self.update = 0
        self.examples_seen = 0
        self._main_learning_rate_override: float | None = None
        self._bt4_learning_rate_override: float | None = None
        self.examples_per_update = (
            None if examples_per_update is None else int(examples_per_update)
        )
        if self.config.lr_schedule_unit == "examples":
            if self.examples_per_update is None or self.examples_per_update < 1:
                raise ValueError(
                    "Example-based LR scheduling requires a positive examples_per_update"
                )
            if not (
                0 < self.config.lr_warmup_examples < self.config.lr_total_examples
            ):
                raise ValueError("Invalid example-based warmup/total schedule")
        elif self.config.lr_schedule_unit != "updates":
            raise ValueError(
                f"Unsupported lr_schedule_unit: {self.config.lr_schedule_unit!r}"
            )
        self.leaves: list[_OptimizerLeaf] = []
        for name, parameter in model.named_parameters():
            use_muon = self._use_muon(name, parameter)
            learning_rate_kind = "bt4" if name.startswith("encoder.") else "main"
            self.leaves.append(
                _OptimizerLeaf(
                    name=name,
                    parameter=parameter,
                    learning_rate_kind=learning_rate_kind,
                    use_muon=use_muon,
                    apply_weight_decay=self._apply_weight_decay(
                        name,
                        parameter,
                        learning_rate_kind=learning_rate_kind,
                    ),
                    first_moment=torch.zeros_like(parameter),
                    second_moment=(None if use_muon else torch.zeros_like(parameter)),
                )
            )
    @staticmethod
    def _use_muon(name: str, parameter: Tensor) -> bool:
        if parameter.ndim < 2:
            return False
        path = name.replace(".", "/").lower()
        if any(
            token in path
            for token in (
                "bias",
                "_b",
                "/b",
                "norm",
                "ln",
                "embed",
                "embedding",
            )
        ):
            return False
        rows, columns = parameter.shape[-2:]
        if min(rows, columns) < 128:
            return False
        return max(rows, columns) / min(rows, columns) <= 2.0

    def _apply_weight_decay(
        self,
        name: str,
        parameter: Tensor,
        *,
        learning_rate_kind: str,
    ) -> bool:
        if not self.config.selective_weight_decay:
            return True
        if learning_rate_kind == "bt4" or parameter.ndim < 2:
            return False
        path = name.replace(".", "/").lower()
        return not any(
            token in path
            for token in (
                "bias",
                "_b",
                "/b",
                "norm",
                "ln",
                "embed",
                "embedding",
            )
        )

    def learning_rate_ratio(self) -> float:
        if self.config.lr_schedule_unit == "examples":
            assert self.examples_per_update is not None
            position = min(
                self.examples_seen + self.examples_per_update,
                self.config.lr_total_examples,
            )
            if position <= self.config.lr_warmup_examples:
                return position / self.config.lr_warmup_examples
            progress = (
                (position - self.config.lr_warmup_examples)
                / (self.config.lr_total_examples - self.config.lr_warmup_examples)
            )
            return self.config.lr_min_ratio + (
                1.0 - self.config.lr_min_ratio
            ) * 0.5 * (1.0 + math.cos(math.pi * progress))
        relative = min(
            max(self.update - self.config.lr_decay_start, 0),
            self.config.lr_decay_steps,
        )
        progress = relative / self.config.lr_decay_steps
        return self.config.lr_min_ratio + (1.0 - self.config.lr_min_ratio) * 0.5 * (
            1.0 + math.cos(math.pi * progress)
        )

    def _learning_rate(self, kind: str) -> float:
        override = (
            self._bt4_learning_rate_override
            if kind == "bt4"
            else self._main_learning_rate_override
        )
        if override is not None:
            return override
        peak = self.config.bt4_learning_rate if kind == "bt4" else self.config.learning_rate
        return peak * self.learning_rate_ratio()

    def set_learning_rates(self, *, main: float, bt4: float) -> None:
        if not math.isfinite(main) or main <= 0.0:
            raise ValueError("main learning-rate override must be finite and positive")
        if not math.isfinite(bt4) or bt4 <= 0.0:
            raise ValueError("BT4 learning-rate override must be finite and positive")
        self._main_learning_rate_override = float(main)
        self._bt4_learning_rate_override = float(bt4)

    @staticmethod
    def _orthogonalize(update: Tensor) -> Tensor:
        rows, columns = update.shape[-2:]
        matrix = update.reshape(-1, rows, columns)
        transposed = rows > columns
        if transposed:
            matrix = matrix.transpose(-2, -1)
        matrix = matrix / (
            torch.linalg.vector_norm(matrix, ord=2, dim=(-2, -1), keepdim=True) + 1e-8
        )
        for _ in range(5):
            gram = matrix @ matrix.transpose(-2, -1)
            polynomial = -4.775 * gram + 2.0315 * (gram @ gram)
            matrix = 3.4445 * matrix + polynomial @ matrix
        if transposed:
            matrix = matrix.transpose(-2, -1)
        return matrix.reshape_as(update)

    def _global_gradient_norm(self) -> Tensor:
        if not self.leaves:
            return torch.zeros((), dtype=torch.float32)
        device = self.leaves[0].parameter.device
        total = torch.zeros((), device=device, dtype=torch.float32)
        for leaf in self.leaves:
            if leaf.parameter.grad is not None:
                total = total + leaf.parameter.grad.float().square().sum()
        return torch.sqrt(total)

    @torch.no_grad()
    def step(self) -> dict[str, float | int | bool]:
        gradient_norm = self._global_gradient_norm()
        finite = bool(torch.isfinite(gradient_norm))
        if not finite:
            self.zero_grad()
            return {
                "optimizer_update": self.update,
                "optimizer_skipped_nonfinite": True,
                "gradient_global_norm": float(gradient_norm.cpu()),
                "gradient_clip_scale": 0.0,
            }

        if self.config.grad_clip_norm > 0.0:
            clip_scale = torch.minimum(
                torch.ones_like(gradient_norm),
                torch.as_tensor(
                    self.config.grad_clip_norm,
                    device=gradient_norm.device,
                    dtype=gradient_norm.dtype,
                )
                / gradient_norm.clamp_min(1e-12),
            )
        else:
            clip_scale = torch.ones_like(gradient_norm)
        count = self.update + 1
        beta_muon = 0.95
        beta1 = 0.9
        beta2 = 0.999
        for leaf in self.leaves:
            parameter = leaf.parameter
            gradient = parameter.grad
            if gradient is None:
                gradient = torch.zeros_like(parameter)
            else:
                gradient = gradient * clip_scale.to(gradient.dtype)

            if leaf.use_muon:
                leaf.first_moment.mul_(beta_muon).add_(gradient, alpha=1.0 - beta_muon)
                corrected_moment = leaf.first_moment / (1.0 - beta_muon ** (count + 1))
                corrected_gradient = gradient / (1.0 - beta_muon**count)
                update = beta_muon * corrected_moment + (1.0 - beta_muon) * corrected_gradient
                update = self._orthogonalize(update)
                rows, columns = parameter.shape[-2:]
                update = update * math.sqrt(max(1.0, columns / rows))
            else:
                assert leaf.second_moment is not None
                leaf.first_moment.mul_(beta1).add_(gradient, alpha=1.0 - beta1)
                leaf.second_moment.mul_(beta2).addcmul_(gradient, gradient, value=1.0 - beta2)
                corrected_moment = leaf.first_moment / (1.0 - beta1 ** (count + 1))
                corrected_gradient = gradient / (1.0 - beta1**count)
                nesterov = beta1 * corrected_moment + (1.0 - beta1) * corrected_gradient
                corrected_variance = leaf.second_moment / (1.0 - beta2**count)
                update = nesterov / (torch.sqrt(corrected_variance) + 1e-8)

            if leaf.apply_weight_decay:
                update = update + self.config.weight_decay * parameter
            parameter.add_(update, alpha=-self._learning_rate(leaf.learning_rate_kind))

        main_lr = self._learning_rate("main")
        bt4_lr = self._learning_rate("bt4")
        completed_update = self.update
        self.update += 1
        if self.examples_per_update is not None:
            self.examples_seen += self.examples_per_update
        self.zero_grad()
        return {
            "optimizer_update": completed_update,
            "optimizer_skipped_nonfinite": False,
            "gradient_global_norm": float(gradient_norm.cpu()),
            "gradient_clip_scale": float(clip_scale.cpu()),
            "learning_rate": main_lr,
            "bt4_learning_rate": bt4_lr,
        }

    def zero_grad(self) -> None:
        for leaf in self.leaves:
            leaf.parameter.grad = None

    def partition_manifest(self) -> dict[str, Any]:
        rows = [
            {
                "path": leaf.name,
                "optimizer": "muon" if leaf.use_muon else "nesterov_adamw",
                "learning_rate_kind": leaf.learning_rate_kind,
                "apply_weight_decay": leaf.apply_weight_decay,
                "shape": list(leaf.parameter.shape),
                "dtype": str(leaf.parameter.dtype),
            }
            for leaf in self.leaves
        ]
        return {
            "schema_version": "torch-muon-adamw-partition-v1",
            "leaf_count": len(rows),
            "muon_leaf_count": sum(row["optimizer"] == "muon" for row in rows),
            "adamw_leaf_count": sum(row["optimizer"] == "nesterov_adamw" for row in rows),
            "bt4_leaf_count": sum(row["learning_rate_kind"] == "bt4" for row in rows),
            "main_leaf_count": sum(row["learning_rate_kind"] == "main" for row in rows),
            "leaves": rows,
        }


_ALLOWED_PICKLE_GLOBALS = {
    ("numpy", "ndarray"): np.ndarray,
    ("numpy", "dtype"): np.dtype,
    (
        "numpy._core.multiarray",
        "_reconstruct",
    ): np._core.multiarray._reconstruct,
    (
        "numpy.core.multiarray",
        "_reconstruct",
    ): np._core.multiarray._reconstruct,
    ("numpy._core.multiarray", "scalar"): np._core.multiarray.scalar,
    ("numpy.core.multiarray", "scalar"): np._core.multiarray.scalar,
    ("ml_dtypes", "bfloat16"): ml_dtypes.bfloat16,
}


class _RestrictedModelUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:
        allowed = _ALLOWED_PICKLE_GLOBALS.get((module, name))
        if allowed is None:
            raise pickle.UnpicklingError(f"Forbidden global in source model: {module}.{name}")
        return allowed

    def persistent_load(self, pid: object) -> Any:
        raise pickle.UnpicklingError(f"Persistent pickle reference forbidden: {pid!r}")


def _sha256_open_file(handle: Any) -> str:
    digest = hashlib.sha256()
    handle.seek(0)
    while chunk := handle.read(_HASH_CHUNK_BYTES):
        digest.update(chunk)
    return digest.hexdigest()


def _read_model_member(archive: zipfile.ZipFile) -> dict[str, Any]:
    with archive.open("model_trainable.npy", "r") as member:
        version = np.lib.format.read_magic(member)
        if version != (1, 0):
            raise ValueError(f"Source model member must be NPY v1.0, got {version}")
        shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(member)
        if shape != () or fortran_order or dtype != np.dtype(object):
            raise ValueError(
                "Invalid source model wrapper: "
                f"shape={shape}, fortran={fortran_order}, dtype={dtype}"
            )
        wrapper = _RestrictedModelUnpickler(
            member, fix_imports=False, encoding="ASCII", errors="strict"
        ).load()
    if (
        type(wrapper) is not np.ndarray
        or wrapper.shape != ()
        or wrapper.dtype != np.dtype(object)
        or type(wrapper.item()) is not dict
    ):
        raise TypeError("Source model member is not a scalar object array of dict")
    return wrapper.item()


def load_verified_source_model(path: Path = _SOURCE_STATE) -> dict[str, Any]:
    """Read only the model member after verifying the pinned 1.85 GiB archive."""

    source = _require_workspace(path, exists=True)
    with source.open("rb") as handle:
        source_stat = os.fstat(handle.fileno())
        if not stat.S_ISREG(source_stat.st_mode):
            raise ValueError(f"Source checkpoint is not regular: {source}")
        if source_stat.st_size != _SOURCE_SIZE_BYTES:
            raise ValueError(f"Source size drift: {source_stat.st_size} != {_SOURCE_SIZE_BYTES}")
        digest = _sha256_open_file(handle)
        if not hmac.compare_digest(digest, _SOURCE_SHA256):
            raise ValueError(f"Source SHA-256 drift: {digest} != {_SOURCE_SHA256}")
        handle.seek(0)
        with np.load(handle, allow_pickle=False) as payload:
            if set(payload.files) != {
                "step",
                "model_trainable",
                "optimizer_state",
            }:
                raise ValueError(f"Source envelope drift: {payload.files}")
            step = np.asarray(payload["step"])
            if step.shape != () or int(step) != _SOURCE_STEP:
                raise ValueError(f"Source step drift: {step}")
        handle.seek(0)
        with zipfile.ZipFile(handle, "r") as archive:
            members = [item.filename for item in archive.infolist()]
            expected = {
                "step.npy",
                "model_trainable.npy",
                "optimizer_state.npy",
            }
            if set(members) != expected or len(members) != len(expected):
                raise ValueError(f"Source NPZ members drift: {members}")
            return _read_model_member(archive)


def _flatten_tree(value: Any, path: tuple[str | int, ...] = ()) -> dict[tuple[str | int, ...], Any]:
    if type(value) is dict:
        flat: dict[tuple[str | int, ...], Any] = {}
        for key, child in value.items():
            if type(key) not in {str, int}:
                raise TypeError(f"Unsupported source key at {path}: {key!r}")
            flat.update(_flatten_tree(child, (*path, key)))
        return flat
    if type(value) in {list, tuple}:
        flat = {}
        for index, child in enumerate(value):
            flat.update(_flatten_tree(child, (*path, index)))
        return flat
    array = np.asarray(value)
    if array.dtype.hasobject:
        raise TypeError(f"Object-valued source leaf at {path}")
    return {path: value}


def bind_source_model(model: JointModel, source_tree: Mapping[str, Any]) -> dict[str, Any]:
    """Copy every source leaf exactly once into the same-named torch parameter."""

    flat = _flatten_tree(source_tree)
    named = dict(model.named_parameters())
    consumed: set[tuple[str | int, ...]] = set()
    records: list[dict[str, Any]] = []
    combined = hashlib.sha256()
    for name, parameter in named.items():
        parts: tuple[str | int, ...] = tuple(
            int(part) if part.isdigit() else part for part in name.split(".")
        )
        if parts not in flat:
            raise KeyError(f"PyTorch parameter has no source leaf: {name}")
        array = np.asarray(flat[parts])
        expected_dtype = (
            torch.bfloat16
            if str(array.dtype) == "bfloat16"
            else torch.float32
            if array.dtype == np.dtype(np.float32)
            else None
        )
        if expected_dtype is None:
            raise TypeError(f"Unsupported source dtype at {name}: {array.dtype}")
        if parameter.dtype != expected_dtype:
            raise TypeError(
                f"Parameter dtype mismatch at {name}: {parameter.dtype} != "
                f"{expected_dtype} from {array.dtype}"
            )
        if tuple(parameter.shape) != tuple(array.shape):
            raise ValueError(
                f"Parameter shape mismatch at {name}: {tuple(parameter.shape)} != {array.shape}"
            )
        leaf_bytes = array.tobytes(order="C")
        leaf_digest = hashlib.sha256(leaf_bytes).hexdigest()
        combined.update(name.encode("utf-8"))
        combined.update(b"\0")
        combined.update(leaf_digest.encode("ascii"))
        with torch.no_grad():
            source_f32 = torch.from_numpy(
                np.ascontiguousarray(array.astype(np.float32, copy=False))
            )
            parameter.copy_(source_f32.to(parameter.dtype))
        consumed.add(parts)
        records.append(
            {
                "path": name,
                "shape": list(array.shape),
                "source_dtype": str(array.dtype),
                "torch_dtype": str(parameter.dtype),
                "nbytes": int(array.nbytes),
                "sha256": leaf_digest,
            }
        )
    unused = sorted((".".join(map(str, path)) for path in set(flat) - consumed))
    if unused:
        raise ValueError(f"Unused source model leaves ({len(unused)}): {unused[:10]}")
    total_bytes = sum(record["nbytes"] for record in records)
    if len(records) != _EXPECTED_MODEL_LEAVES:
        raise ValueError(f"Source leaf count drift: {len(records)} != {_EXPECTED_MODEL_LEAVES}")
    if total_bytes != _EXPECTED_MODEL_BYTES:
        raise ValueError(f"Source model bytes drift: {total_bytes} != {_EXPECTED_MODEL_BYTES}")
    return {
        "schema_version": "torch-source-leaf-map-v1",
        "leaf_count": len(records),
        "nbytes": total_bytes,
        "combined_sha256": combined.hexdigest(),
        "records": records,
    }


def load_source_model(
    *,
    device: torch.device,
    source_path: Path = _SOURCE_STATE,
    config: Config = CONFIG,
) -> tuple[JointModel, dict[str, Any]]:
    model = JointModel(config)
    source = load_verified_source_model(source_path)
    mapping = bind_source_model(model, source)
    del source
    gc.collect()
    return model.to(device), mapping


def _copy_parameter_array(
    parameter: nn.Parameter,
    value: Any,
    *,
    name: str,
) -> dict[str, Any]:
    array = np.asarray(value)
    if tuple(parameter.shape) != tuple(array.shape):
        raise ValueError(
            f"Raw BT4 shape mismatch at {name}: "
            f"{tuple(parameter.shape)} != {tuple(array.shape)}"
        )
    array_f32 = np.ascontiguousarray(array.astype(np.float32, copy=False))
    with torch.no_grad():
        parameter.copy_(torch.from_numpy(array_f32).to(parameter.dtype))
    return {
        "path": name,
        "shape": list(array.shape),
        "source_dtype": str(array.dtype),
        "torch_dtype": str(parameter.dtype),
        "sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
    }


def bind_raw_bt4(
    model: JointModel,
    mapped: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind the raw exported BT4 encoder and native policy head exactly once."""

    if model.encoder.policy_head is None:
        raise ValueError("Raw BT4 hero initialization requires a native policy head")
    records: list[dict[str, Any]] = []

    def copy(parameter: nn.Parameter, value: Any, name: str) -> None:
        records.append(_copy_parameter_array(parameter, value, name=name))

    embedding = model.encoder.embedding
    source_embedding = mapped["embedding"]
    copy(embedding.preproc.w, source_embedding["preproc_w"], "encoder.embedding.preproc.w")
    copy(embedding.preproc.b, source_embedding["preproc_b"], "encoder.embedding.preproc.b")
    copy(embedding.proj.w, source_embedding["w"], "encoder.embedding.proj.w")
    copy(embedding.proj.b, source_embedding["b"], "encoder.embedding.proj.b")
    copy(embedding.ln.scale, source_embedding["ln_scale"], "encoder.embedding.ln.scale")
    copy(embedding.ln.bias, source_embedding["ln_bias"], "encoder.embedding.ln.bias")
    copy(embedding.mul_gate, source_embedding["mul_gate"], "encoder.embedding.mul_gate")
    copy(embedding.add_gate, source_embedding["add_gate"], "encoder.embedding.add_gate")
    copy(
        embedding.ffn1.w,
        source_embedding["ffn"]["dense1_w"],
        "encoder.embedding.ffn1.w",
    )
    copy(
        embedding.ffn1.b,
        source_embedding["ffn"]["dense1_b"],
        "encoder.embedding.ffn1.b",
    )
    copy(
        embedding.ffn2.w,
        source_embedding["ffn"]["dense2_w"],
        "encoder.embedding.ffn2.w",
    )
    copy(
        embedding.ffn2.b,
        source_embedding["ffn"]["dense2_b"],
        "encoder.embedding.ffn2.b",
    )
    copy(
        embedding.ffn_ln.scale,
        source_embedding["ffn_ln_scale"],
        "encoder.embedding.ffn_ln.scale",
    )
    copy(
        embedding.ffn_ln.bias,
        source_embedding["ffn_ln_bias"],
        "encoder.embedding.ffn_ln.bias",
    )

    source_layers = mapped["encoder"]
    if len(source_layers) != len(model.encoder.layers):
        raise ValueError(
            f"Raw BT4 encoder depth drift: {len(source_layers)} != "
            f"{len(model.encoder.layers)}"
        )
    for index, (layer, source_layer) in enumerate(
        zip(model.encoder.layers, source_layers, strict=True)
    ):
        prefix = f"encoder.layers.{index}"
        mha = source_layer["mha"]
        copy(layer.wq, mha["q_w"], f"{prefix}.wq")
        copy(layer.wq_b, mha["q_b"], f"{prefix}.wq_b")
        copy(layer.wk, mha["k_w"], f"{prefix}.wk")
        copy(layer.wk_b, mha["k_b"], f"{prefix}.wk_b")
        copy(layer.wv, mha["v_w"], f"{prefix}.wv")
        copy(layer.wv_b, mha["v_b"], f"{prefix}.wv_b")
        copy(layer.wo.w, mha["dense_w"], f"{prefix}.wo.w")
        copy(layer.wo.b, mha["dense_b"], f"{prefix}.wo.b")
        copy(layer.ln_attn.scale, source_layer["ln1"]["scale"], f"{prefix}.ln_attn.scale")
        copy(layer.ln_attn.bias, source_layer["ln1"]["bias"], f"{prefix}.ln_attn.bias")
        copy(layer.ffn1.w, source_layer["ffn"]["dense1_w"], f"{prefix}.ffn1.w")
        copy(layer.ffn1.b, source_layer["ffn"]["dense1_b"], f"{prefix}.ffn1.b")
        copy(layer.ffn2.w, source_layer["ffn"]["dense2_w"], f"{prefix}.ffn2.w")
        copy(layer.ffn2.b, source_layer["ffn"]["dense2_b"], f"{prefix}.ffn2.b")
        copy(layer.ln_ffn.scale, source_layer["ln2"]["scale"], f"{prefix}.ln_ffn.scale")
        copy(layer.ln_ffn.bias, source_layer["ln2"]["bias"], f"{prefix}.ln_ffn.bias")
        smolgen = mha["smolgen"]
        copy(layer.smolgen.compress.w, smolgen["compress_w"], f"{prefix}.smolgen.compress.w")
        copy(layer.smolgen.dense1.w, smolgen["dense1_w"], f"{prefix}.smolgen.dense1.w")
        copy(layer.smolgen.dense1.b, smolgen["dense1_b"], f"{prefix}.smolgen.dense1.b")
        copy(layer.smolgen.ln1.scale, smolgen["ln1_scale"], f"{prefix}.smolgen.ln1.scale")
        copy(layer.smolgen.ln1.bias, smolgen["ln1_bias"], f"{prefix}.smolgen.ln1.bias")
        copy(layer.smolgen.dense2.w, smolgen["dense2_w"], f"{prefix}.smolgen.dense2.w")
        copy(layer.smolgen.dense2.b, smolgen["dense2_b"], f"{prefix}.smolgen.dense2.b")
        copy(layer.smolgen.ln2.scale, smolgen["ln2_scale"], f"{prefix}.smolgen.ln2.scale")
        copy(layer.smolgen.ln2.bias, smolgen["ln2_bias"], f"{prefix}.smolgen.ln2.bias")
        copy(layer.smolgen.shared_w, mapped["smolgen_w"], f"{prefix}.smolgen.shared_w")

    policy = model.encoder.policy_head
    source_policy = mapped["policy"]
    copy(policy.dense1.w, source_policy["dense1_w"], "encoder.policy_head.dense1.w")
    copy(policy.dense1.b, source_policy["dense1_b"], "encoder.policy_head.dense1.b")
    copy(policy.q.w, source_policy["q_w"], "encoder.policy_head.q.w")
    copy(policy.q.b, source_policy["q_b"], "encoder.policy_head.q.b")
    copy(policy.k.w, source_policy["k_w"], "encoder.policy_head.k.w")
    copy(policy.k.b, source_policy["k_b"], "encoder.policy_head.k.b")
    copy(policy.prom_w, source_policy["prom_w"], "encoder.policy_head.prom_w")
    np.testing.assert_array_equal(
        policy.mapping_table.cpu().numpy(),
        np.asarray(mapped["mapping_table"], dtype=np.int64),
    )

    combined = hashlib.sha256()
    for record in records:
        combined.update(record["path"].encode("utf-8"))
        combined.update(b"\0")
        combined.update(record["sha256"].encode("ascii"))
    return {
        "schema_version": "torch-raw-bt4-map-v1",
        "leaf_count": len(records),
        "combined_sha256": combined.hexdigest(),
        "records": records,
    }


def initialize_fresh_modules(model: JointModel, *, seed: int) -> dict[str, Any]:
    """Deterministically initialize only the non-BT4 hero modules."""

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    initialized: list[str] = []

    def fill_normal(parameter: nn.Parameter, standard_deviation: float) -> None:
        value = torch.randn(
            tuple(parameter.shape),
            generator=generator,
            device="cpu",
            dtype=torch.float32,
        )
        value.mul_(float(standard_deviation))
        parameter.copy_(value.to(parameter.dtype))

    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name.startswith("encoder."):
                continue
            leaf = name.rsplit(".", 1)[-1]
            if name in {"out_proj", "out_bias", "state_projector.cls"}:
                parameter.zero_()
            elif name.startswith("jepa_transition.cond_"):
                parameter.zero_()
            elif (
                leaf
                in {
                    "b",
                    "b1",
                    "bias",
                    "in_bias",
                    "time_bias",
                    "out_bias",
                    "value_b",
                    "wdl_b",
                }
                or leaf.startswith("b_")
                or leaf.endswith("_bias")
            ):
                parameter.zero_()
            elif leaf == "scale" or leaf.endswith("norm_scale"):
                parameter.fill_(1.0)
            elif name == "state_projector.pos_embed":
                fill_normal(parameter, 0.02 / math.sqrt(parameter.shape[-1]))
            elif name == "pos_embed" or name.endswith("_embed.embedding"):
                fill_normal(parameter, 1.0 / math.sqrt(parameter.shape[-1]))
            elif name == "jepa_transition.w_down":
                fill_normal(parameter, 1e-3 / math.sqrt(parameter.shape[-2]))
            elif parameter.ndim >= 2:
                fill_normal(parameter, 1.0 / math.sqrt(parameter.shape[-2]))
            else:
                raise ValueError(f"No fresh initialization rule for {name}")
            initialized.append(name)
    expected = [
        name for name, _ in model.named_parameters() if not name.startswith("encoder.")
    ]
    if initialized != expected:
        raise RuntimeError("Fresh initialization did not cover every non-BT4 parameter")
    return {
        "schema_version": "torch-fresh-init-v1",
        "seed": int(seed),
        "leaf_count": len(initialized),
        "zero_initialized_residual": True,
        "paths": initialized,
    }


def load_raw_bt4_hero_model(
    *,
    device: torch.device,
    raw_bt4_path: Path = _RAW_BT4_PATH,
    config: Config = HERO_CONFIG,
    bt4_norm_impl: str = "eager",
) -> tuple[JointModel, dict[str, Any]]:
    if not config.use_bt4_policy_residual:
        raise ValueError("Hero model config must enable the BT4 policy residual")
    source_path = _require_workspace(raw_bt4_path, exists=True)
    source_stat = source_path.stat()
    if source_stat.st_size != _RAW_BT4_SIZE_BYTES:
        raise ValueError(
            f"Raw BT4 size drift: {source_stat.st_size} != {_RAW_BT4_SIZE_BYTES}"
        )
    with source_path.open("rb") as handle:
        digest = _sha256_open_file(handle)
    if not hmac.compare_digest(digest, _RAW_BT4_SHA256):
        raise ValueError(f"Raw BT4 SHA-256 drift: {digest} != {_RAW_BT4_SHA256}")

    from chess_dfm_jax.policy import attention_policy_map
    from chess_dfm_jax.weights import load_pb_gz, map_bt4_weights

    model = JointModel(config, bt4_norm_impl=bt4_norm_impl)
    fresh_manifest = initialize_fresh_modules(model, seed=config.init_seed)
    mapped = map_bt4_weights(
        load_pb_gz(str(source_path)),
        mapping_table=attention_policy_map(),
    )
    raw_mapping = bind_raw_bt4(model, mapped)
    del mapped
    gc.collect()
    return model.to(device), {
        "schema_version": "torch-hero-init-v1",
        "combined_sha256": raw_mapping["combined_sha256"],
        "leaf_count": raw_mapping["leaf_count"],
        "raw_asset": {
            "path": str(source_path),
            "size_bytes": source_stat.st_size,
            "sha256": digest,
        },
        "raw_mapping": raw_mapping,
        "fresh": fresh_manifest,
        "bt4_norm_impl": bt4_norm_impl,
    }


_TRAJECTORY_METADATA_KEYS = frozenset(
    {"source", "game_id", "ply", "result", "fen_t", "input_format", "actions_uci"}
)


def _metadata_text(value: Any) -> str:
    scalar = np.asarray(value).item()
    return scalar.decode("utf-8") if isinstance(scalar, bytes) else str(scalar)


def _canonicalize_trajectory_batch_reference(
    batch: Mapping[str, Any],
) -> dict[str, Any]:
    """Board-enumerating oracle for canonical trajectory conversion."""

    import chess

    from chess_dfm_jax.policy import (
        LC0_CANONICAL_1858_INPUT_FORMAT,
        encode_lc0_canonical_1858,
        legal_move_mask,
        legal_mask_lc0_canonical_1858,
    )

    required = {
        "fen_t",
        "input_format",
        "actions_uci",
        "future_valid",
        "legal_idx",
        "legal_count",
    }
    missing = sorted(required - set(batch))
    if missing:
        raise KeyError(f"Canonical trajectory conversion requires metadata: {missing}")
    result = {
        key: value for key, value in batch.items() if key not in _TRAJECTORY_METADATA_KEYS
    }
    future_valid = np.asarray(batch["future_valid"], dtype=np.float32)
    actions_uci = np.asarray(batch["actions_uci"])
    if actions_uci.shape != future_valid.shape:
        raise ValueError(
            f"actions_uci/future_valid shape drift: "
            f"{actions_uci.shape} != {future_valid.shape}"
        )
    batch_size, horizon = future_valid.shape
    source_legal = np.asarray(batch["legal_idx"])
    source_legal_count = np.asarray(batch["legal_count"])
    legal_capacity = int(source_legal.shape[-1])
    canonical_actions = np.zeros((batch_size, horizon), dtype=np.int32)
    canonical_legal = np.full(
        (batch_size, horizon, legal_capacity),
        _LEGAL_PAD,
        dtype=np.int32,
    )
    canonical_count = np.zeros((batch_size, horizon), dtype=np.int32)
    canonical_valid = np.zeros((batch_size, horizon), dtype=np.float32)
    fens = np.asarray(batch["fen_t"])
    input_formats = np.asarray(batch["input_format"])

    for row in range(batch_size):
        input_format = _metadata_text(
            input_formats if input_formats.ndim == 0 else input_formats[row]
        )
        if input_format != LC0_CANONICAL_1858_INPUT_FORMAT:
            raise ValueError(
                f"Unsupported canonical input format at row {row}: {input_format!r}"
            )
        fen = _metadata_text(fens if fens.ndim == 0 else fens[row])
        candidate_records: list[tuple[int, np.ndarray]] | None = None
        for chess960 in (False, True):
            board = chess.Board(fen, chess960=chess960)
            if not board.is_valid():
                continue
            records: list[tuple[int, np.ndarray]] = []
            for offset in range(horizon):
                if future_valid[row, offset] <= 0.0:
                    continue
                observed_source_legal = np.flatnonzero(
                    legal_move_mask(board, "lc0_1858")
                ).astype(np.int32, copy=False)
                stored_count = int(source_legal_count[row, offset])
                stored_source_legal = np.sort(
                    source_legal[row, offset, :stored_count].astype(
                        np.int32,
                        copy=False,
                    )
                )
                if not np.array_equal(
                    observed_source_legal,
                    stored_source_legal,
                ):
                    records = []
                    break
                move_text = _metadata_text(actions_uci[row, offset])
                move = chess.Move.from_uci(move_text)
                action = encode_lc0_canonical_1858(
                    board,
                    move,
                    input_format=input_format,
                )
                legal = np.flatnonzero(
                    legal_mask_lc0_canonical_1858(
                        board,
                        input_format=input_format,
                    )
                ).astype(np.int32, copy=False)
                if not np.any(legal == action):
                    records = []
                    break
                records.append((action, legal))
                board.push(move)
            expected_records = int(np.count_nonzero(future_valid[row] > 0.0))
            if len(records) == expected_records:
                candidate_records = records
                break
        if candidate_records is None:
            raise RuntimeError(
                f"Neither standard nor Chess960 semantics reproduce the stored "
                f"legal/action trajectory at row {row}"
            )
        record_index = 0
        for offset in range(horizon):
            if future_valid[row, offset] <= 0.0:
                continue
            action, legal = candidate_records[record_index]
            record_index += 1
            if legal.size > legal_capacity:
                raise ValueError(
                    f"Canonical legal set at row {row}, horizon {offset + 1} "
                    f"needs {legal.size} slots; shard capacity is {legal_capacity}"
                )
            canonical_actions[row, offset] = action
            canonical_legal[row, offset, : legal.size] = legal
            canonical_count[row, offset] = legal.size
            canonical_valid[row, offset] = 1.0

    result["action_indices"] = canonical_actions
    result["action_idx"] = canonical_actions[:, 0]
    result["legal_idx"] = canonical_legal
    result["legal_count"] = canonical_count
    result["legal_masks_valid"] = canonical_valid
    return result


def canonicalize_trajectory_batch(
    batch: Mapping[str, Any],
) -> dict[str, Any]:
    """Re-encode stored legacy slots in the side-to-move LC0 frame.

    The legacy and canonical 1,858-way codecs differ by a fixed vertical
    rank mirror when black is to move.  Applying that permutation to the
    already stored legal indices is exact and avoids enumerating every legal
    move twice at every horizon.  The legacy codec cannot represent black
    promotions or knight promotions, so those rare legal moves are recovered
    from the incrementally maintained board.

    ``_canonicalize_trajectory_batch_reference`` remains the deliberately
    slow board-enumerating oracle used by parity tests and sampled data audits.
    """

    import chess

    from chess_dfm_jax.policy import (
        LC0_CANONICAL_1858_INPUT_FORMAT,
        encode_lc0_canonical_1858,
        legacy_to_lc0_canonical_1858_index_map,
    )

    required = {
        "fen_t",
        "input_format",
        "actions_uci",
        "future_valid",
        "legal_idx",
        "legal_count",
    }
    missing = sorted(required - set(batch))
    if missing:
        raise KeyError(f"Canonical trajectory conversion requires metadata: {missing}")
    result = {
        key: value for key, value in batch.items() if key not in _TRAJECTORY_METADATA_KEYS
    }
    future_valid = np.asarray(batch["future_valid"], dtype=np.float32)
    actions_uci = np.asarray(batch["actions_uci"])
    if actions_uci.shape != future_valid.shape:
        raise ValueError(
            f"actions_uci/future_valid shape drift: "
            f"{actions_uci.shape} != {future_valid.shape}"
        )
    batch_size, horizon = future_valid.shape
    source_legal = np.asarray(batch["legal_idx"])
    source_legal_count = np.asarray(batch["legal_count"])
    if source_legal.ndim != 3 or source_legal.shape[:2] != future_valid.shape:
        raise ValueError(
            "legal_idx must have shape "
            f"[{batch_size}, {horizon}, capacity], got {source_legal.shape}"
        )
    if source_legal_count.shape != future_valid.shape:
        raise ValueError(
            f"legal_count shape drift: "
            f"{source_legal_count.shape} != {future_valid.shape}"
        )
    legal_capacity = int(source_legal.shape[-1])
    canonical_actions = np.zeros((batch_size, horizon), dtype=np.int32)
    canonical_legal = np.full(
        (batch_size, horizon, legal_capacity),
        _LEGAL_PAD,
        dtype=np.int32,
    )
    canonical_count = np.zeros((batch_size, horizon), dtype=np.int32)
    canonical_valid = np.zeros((batch_size, horizon), dtype=np.float32)
    fens = np.asarray(batch["fen_t"])
    input_formats = np.asarray(batch["input_format"])
    index_maps = {
        chess.WHITE: legacy_to_lc0_canonical_1858_index_map(
            black_to_move=False
        ),
        chess.BLACK: legacy_to_lc0_canonical_1858_index_map(
            black_to_move=True
        ),
    }

    for row in range(batch_size):
        input_format = _metadata_text(
            input_formats if input_formats.ndim == 0 else input_formats[row]
        )
        if input_format != LC0_CANONICAL_1858_INPUT_FORMAT:
            raise ValueError(
                f"Unsupported canonical input format at row {row}: {input_format!r}"
            )
        fen = _metadata_text(fens if fens.ndim == 0 else fens[row])
        candidate_records: list[tuple[int, np.ndarray]] | None = None
        for chess960 in (False, True):
            board = chess.Board(fen, chess960=chess960)
            if not board.is_valid():
                continue
            records: list[tuple[int, np.ndarray]] = []
            for offset in range(horizon):
                if future_valid[row, offset] <= 0.0:
                    continue
                stored_count = int(source_legal_count[row, offset])
                if not 0 <= stored_count <= legal_capacity:
                    raise ValueError(
                        f"Invalid legal_count at row {row}, horizon "
                        f"{offset + 1}: {stored_count}"
                    )
                stored_source_legal = source_legal[
                    row,
                    offset,
                    :stored_count,
                ].astype(np.int32, copy=False)
                if (
                    np.any(stored_source_legal < 0)
                    or np.any(stored_source_legal >= _VOCAB_SIZE)
                ):
                    raise ValueError(
                        f"Out-of-range legacy legal index at row {row}, "
                        f"horizon {offset + 1}"
                    )
                if np.unique(stored_source_legal).size != stored_source_legal.size:
                    raise ValueError(
                        f"Duplicate legacy legal index at row {row}, "
                        f"horizon {offset + 1}"
                    )
                legal = index_maps[board.turn][stored_source_legal]
                if np.any(legal < 0):
                    raise ValueError(
                        f"Unmappable legacy legal index at row {row}, "
                        f"horizon {offset + 1}"
                    )
                legal = np.sort(legal.astype(np.int32, copy=False))

                promotion_rank = (
                    chess.BB_RANK_7
                    if board.turn == chess.WHITE
                    else chess.BB_RANK_2
                )
                if board.pieces_mask(chess.PAWN, board.turn) & promotion_rank:
                    promotion_indices = np.asarray(
                        [
                            encode_lc0_canonical_1858(
                                board,
                                move,
                                input_format=input_format,
                            )
                            for move in board.legal_moves
                            if move.promotion is not None
                        ],
                        dtype=np.int32,
                    )
                    if promotion_indices.size:
                        legal = np.unique(
                            np.concatenate((legal, promotion_indices))
                        ).astype(np.int32, copy=False)

                move_text = _metadata_text(actions_uci[row, offset])
                try:
                    move = chess.Move.from_uci(move_text)
                    action = encode_lc0_canonical_1858(
                        board,
                        move,
                        input_format=input_format,
                    )
                except (ActionCodecError, ValueError):
                    records = []
                    break
                if not np.any(legal == action):
                    records = []
                    break
                records.append((action, legal))
                board.push(move)
            expected_records = int(np.count_nonzero(future_valid[row] > 0.0))
            if len(records) == expected_records:
                candidate_records = records
                break
        if candidate_records is None:
            raise RuntimeError(
                f"Neither standard nor Chess960 semantics reproduce the stored "
                f"legal/action trajectory at row {row}"
            )
        record_index = 0
        for offset in range(horizon):
            if future_valid[row, offset] <= 0.0:
                continue
            action, legal = candidate_records[record_index]
            record_index += 1
            if legal.size > legal_capacity:
                raise ValueError(
                    f"Canonical legal set at row {row}, horizon {offset + 1} "
                    f"needs {legal.size} slots; shard capacity is {legal_capacity}"
                )
            canonical_actions[row, offset] = action
            canonical_legal[row, offset, : legal.size] = legal
            canonical_count[row, offset] = legal.size
            canonical_valid[row, offset] = 1.0

    result["action_indices"] = canonical_actions
    result["action_idx"] = canonical_actions[:, 0]
    result["legal_idx"] = canonical_legal
    result["legal_count"] = canonical_count
    result["legal_masks_valid"] = canonical_valid
    return result


from research.prepare import (  # noqa: E402
    FixedTrajectoryBatches as _FixedTrajectoryBatches,
)


class CanonicalTrajectoryBatches(_FixedTrajectoryBatches):
    """Fixed trajectory schedule with exact canonical action conversion."""

    def _decode_shard(
        self,
        path: Path,
        *,
        row_slice: slice | None,
    ) -> dict[str, Any]:
        from chess_dfm_jax.data.trajectory_v3 import trajectory_v3_to_batch

        try:
            with np.load(_require_workspace(path, exists=True), allow_pickle=False) as payload:
                observed_count = (
                    int(np.asarray(payload["batch_size"]).item())
                    if "batch_size" in payload
                    else int(payload["actions_u16"].shape[0])
                )
                if observed_count != self.samples_per_shard:
                    raise ValueError(
                        f"Shard size changed: expected {self.samples_per_shard}, "
                        f"found {observed_count} in {path}"
                    )
                batch = trajectory_v3_to_batch(
                    payload,
                    view=self.view,
                    horizon=self.horizon,
                    include_metadata=True,
                    row_slice=row_slice,
                )
            return canonicalize_trajectory_batch(batch)
        except Exception as exc:
            raise RuntimeError(
                f"Failed canonical trajectory decode for required shard {path}"
            ) from exc


def _normalize_frozen_indices(
    indices: np.ndarray,
    *,
    total_examples: int,
    batch_size: int,
) -> np.ndarray:
    result = np.asarray(indices, dtype=np.int64)
    if result.ndim != 1 or result.size == 0:
        raise ValueError("Frozen indices must be a non-empty rank-1 array")
    if result.size % batch_size != 0:
        raise ValueError(
            f"Frozen index count {result.size} is not divisible by batch {batch_size}"
        )
    if np.any(result < 0) or np.any(result >= total_examples):
        raise ValueError("Frozen indices contain an out-of-range global index")
    result = np.sort(result)
    if np.any(np.diff(result) == 0):
        raise ValueError("Frozen indices contain duplicates")
    return result


class FrozenIndexTrajectoryBatches:
    """Decode one immutable position set in sorted global-index order."""

    def __init__(
        self,
        split_dir: Path,
        *,
        global_indices: np.ndarray,
        batch_size: int,
        horizon: int,
        indices_sha256: str,
        pool_name: str,
    ):
        self.split_dir = _require_workspace(split_dir, exists=True)
        self.paths = sorted(self.split_dir.glob("*.npz"))
        if not self.paths:
            raise FileNotFoundError(f"No trajectory shards under {self.split_dir}")
        self.batch_size = int(batch_size)
        self.horizon = int(horizon)
        self.samples_per_shard = _FixedTrajectoryBatches._read_sample_count(
            self.paths[0]
        )
        self.global_indices = _normalize_frozen_indices(
            global_indices,
            total_examples=len(self.paths) * self.samples_per_shard,
            batch_size=self.batch_size,
        )
        self.steps_per_epoch = self.global_indices.size // self.batch_size
        self.indices_sha256 = str(indices_sha256)
        self.pool_name = str(pool_name)
        self._rows_by_shard: dict[int, np.ndarray] = {}
        shard_indices = self.global_indices // self.samples_per_shard
        for shard_index in np.unique(shard_indices):
            mask = shard_indices == shard_index
            self._rows_by_shard[int(shard_index)] = (
                self.global_indices[mask] % self.samples_per_shard
            ).astype(np.int64, copy=False)
        self._cache: OrderedDict[int, dict[str, Any]] = OrderedDict()

    def _decode_selected_shard(self, shard_index: int) -> dict[str, Any]:
        cached = self._cache.get(shard_index)
        if cached is not None:
            self._cache.move_to_end(shard_index)
            return cached

        from chess_dfm_jax.data.trajectory_v3 import trajectory_v3_to_batch

        path = self.paths[shard_index]
        rows = self._rows_by_shard[shard_index]
        try:
            with np.load(_require_workspace(path, exists=True), allow_pickle=False) as payload:
                observed_count = (
                    int(np.asarray(payload["batch_size"]).item())
                    if "batch_size" in payload
                    else int(payload["actions_u16"].shape[0])
                )
                if observed_count != self.samples_per_shard:
                    raise ValueError(
                        f"Shard size changed: {observed_count} != "
                        f"{self.samples_per_shard} in {path}"
                    )
                selected: dict[str, np.ndarray] = {}
                for key in payload.files:
                    array = np.asarray(payload[key])
                    selected[key] = (
                        array[rows]
                        if array.ndim > 0 and array.shape[0] == observed_count
                        else array
                    )
            decoded = canonicalize_trajectory_batch(
                trajectory_v3_to_batch(
                    selected,
                    view="joint_latent_sasa",
                    horizon=self.horizon,
                    include_metadata=True,
                )
            )
            compact = {
                key: value
                for key, value in decoded.items()
                if key in _TRAIN_BATCH_KEYS
            }
        except Exception as exc:
            raise RuntimeError(
                f"Failed frozen-index decode for shard {path}"
            ) from exc
        self._cache[shard_index] = compact
        self._cache.move_to_end(shard_index)
        while len(self._cache) > 2:
            self._cache.popitem(last=False)
        return compact

    def batch_at(self, step: int) -> dict[str, Any]:
        if not 0 <= step < self.steps_per_epoch:
            raise IndexError(
                f"Frozen pool step {step} is outside [0, {self.steps_per_epoch})"
            )
        start = step * self.batch_size
        requested = self.global_indices[start : start + self.batch_size]
        requested_shards = requested // self.samples_per_shard
        pieces: list[dict[str, Any]] = []
        for shard_index in np.unique(requested_shards):
            mask = requested_shards == shard_index
            rows = requested[mask] % self.samples_per_shard
            selected_rows = self._rows_by_shard[int(shard_index)]
            positions = np.searchsorted(selected_rows, rows)
            if not np.array_equal(selected_rows[positions], rows):
                raise RuntimeError("Frozen-index shard lookup drift")
            decoded = self._decode_selected_shard(int(shard_index))
            pieces.append(
                {
                    key: np.asarray(value)[positions]
                    for key, value in decoded.items()
                }
            )
        keys = set(pieces[0])
        if any(set(piece) != keys for piece in pieces):
            raise RuntimeError("Frozen-index batch leaf drift across shards")
        result = {
            key: np.concatenate([piece[key] for piece in pieces], axis=0)
            for key in sorted(keys)
        }
        if any(np.asarray(value).shape[0] != self.batch_size for value in result.values()):
            raise RuntimeError("Frozen-index batch did not materialize exactly one batch")
        return result

    def provenance(self) -> dict[str, Any]:
        entries = [
            f"{path.name}\t{path.stat().st_size}"
            for path in self.paths
        ]
        return {
            "pool_name": self.pool_name,
            "split_dir": str(self.split_dir),
            "shard_count": len(self.paths),
            "samples_per_shard": self.samples_per_shard,
            "batch_size": self.batch_size,
            "position_count": int(self.global_indices.size),
            "batch_count": self.steps_per_epoch,
            "global_index_order": "ascending",
            "indices_sha256": self.indices_sha256,
            "file_manifest_sha256": hashlib.sha256(
                "\n".join(entries).encode("utf-8")
            ).hexdigest(),
        }


_TRAIN_BATCH_KEYS = frozenset(
    {
        "current_planes",
        "future_planes",
        "selected_future_planes",
        "action_indices",
        "future_valid",
        "valid",
        "legal_idx",
        "legal_count",
        "legal_masks_valid",
        "value_targets",
        "wdl_targets",
    }
)


def _torch_batch(
    batch: Mapping[str, Any],
    device: torch.device,
    *,
    keys: frozenset[str] = _TRAIN_BATCH_KEYS,
) -> dict[str, Tensor]:
    result: dict[str, Tensor] = {}
    for key, value in batch.items():
        if key not in keys:
            continue
        array = np.asarray(value)
        if not np.issubdtype(array.dtype, np.number):
            continue
        result[key] = torch.from_numpy(np.ascontiguousarray(array)).to(device)
    return result


def _json_scalars(values: Mapping[str, Tensor]) -> dict[str, float]:
    return {
        key: float(value.detach().float().cpu())
        for key, value in values.items()
        if value.numel() == 1
    }


def _summarize_training_records(
    records: list[dict[str, Any]],
    *,
    window_updates: int = _LOSS_SUMMARY_WINDOW_UPDATES,
) -> dict[str, Any]:
    """Return compact, plot-ready terminal and fixed-window train metrics."""

    if not records:
        raise ValueError("Cannot summarize an empty training run")
    if window_updates < 1:
        raise ValueError("window_updates must be positive")
    missing = [
        (int(row.get("update", -1)), key)
        for row in records
        for key in _LOSS_SUMMARY_METRICS
        if key not in row
    ]
    if missing:
        raise KeyError(f"Training metrics missing from loss-summary ABI: {missing[:10]}")

    window_count = min(window_updates, len(records))

    def summarize_window(rows: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "updates": [int(rows[0]["update"]), int(rows[-1]["update"])],
            **{
                key: float(np.mean([float(row[key]) for row in rows], dtype=np.float64))
                for key in _LOSS_SUMMARY_METRICS
            },
        }

    first_window = summarize_window(records[:window_count])
    last_window = summarize_window(records[-window_count:])
    terminal = {
        "update": int(records[-1]["update"]),
        "examples": int(records[-1]["examples"]),
        **{key: float(records[-1][key]) for key in _LOSS_SUMMARY_METRICS},
    }
    return {
        "schema_version": "torch-eager-loss-summary-v2",
        "window_updates": window_count,
        "terminal": terminal,
        "first_window": first_window,
        "last_window": last_window,
        "last_minus_first": {
            key: float(last_window[key] - first_window[key])
            for key in _LOSS_SUMMARY_METRICS
        },
        "plot_contract": {
            "training_curve_source": "metrics.jsonl",
            "training_endpoint": "last_window",
            "primary_cross_experiment_metric": (
                "matched mean validation dfm_ce_loss over frozen seeds 10000 and 20000"
            ),
            "promotion_metric": "normalized-Elo GSPRT",
        },
    }


def _flatten_torch_metrics(values: Mapping[str, Tensor]) -> dict[str, float]:
    """Flatten scalar and horizon-vector tensors with the JAX report ABI."""

    flattened: dict[str, float] = {}
    for key, value in values.items():
        array = value.detach().float().cpu().numpy()
        if array.ndim == 0:
            flattened[key] = float(array)
            continue
        flat = array.reshape(-1)
        if key.endswith("_by_horizon"):
            for index, item in enumerate(flat):
                flattened[f"{key}_h{index + 1}"] = float(item)
        else:
            for index, item in enumerate(flat):
                flattened[f"{key}_{index}"] = float(item)
    return flattened


def _write_json(path: Path, value: Any) -> None:
    target = _require_workspace(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.partial")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, target)


def _json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _fsync_path(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with _require_workspace(path, exists=True).open("rb") as handle:
        while chunk := handle.read(_HASH_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=_REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _choices_to_device(choices: StepChoices, device: torch.device) -> StepChoices:
    return StepChoices(*(value.to(device) for value in choices))


def _compact_training_batch(
    batch: Mapping[str, Any],
    target_horizon: np.ndarray,
) -> dict[str, Any]:
    compact = {
        key: value
        for key, value in batch.items()
        if key in _TRAIN_BATCH_KEYS and key != "future_planes"
    }
    future = np.asarray(batch["future_planes"])
    rows = np.arange(future.shape[0])
    compact["selected_future_planes"] = future[rows, target_horizon]
    return compact


def _prepare_training_step(
    batches: Any,
    *,
    seed: int,
    update: int,
    data_cursor: int,
    batch_size: int,
    config: Config = CONFIG,
) -> _PreparedTrainingStep:
    started = time.perf_counter()
    choices = materialize_step_choices(
        seed=seed,
        update=update,
        batch_size=batch_size,
        config=config,
        device=torch.device("cpu"),
    )
    raw_batch = batches.batch_at(data_cursor)
    compact_batch = _compact_training_batch(
        raw_batch,
        choices.target_horizon.numpy(),
    )
    return _PreparedTrainingStep(
        update=update,
        data_cursor=data_cursor,
        compact_batch=compact_batch,
        choices=choices,
        prepare_seconds=time.perf_counter() - started,
    )


def save_model_checkpoint(
    *,
    output_dir: Path,
    model: JointModel,
    source_mapping_sha256: str,
    optimizer_update: int,
    data_cursor: int,
) -> dict[str, Any]:
    """Write one model-only safetensors checkpoint for selection/evaluation."""

    from safetensors.torch import save_file

    checkpoint_dir = _require_workspace(output_dir / "checkpoint")
    checkpoint_dir.mkdir(parents=True, exist_ok=False)
    final_path = checkpoint_dir / "model.safetensors"
    temporary_path = checkpoint_dir / ".model.safetensors.partial"
    tensors = {
        name: parameter.detach().cpu().contiguous() for name, parameter in model.named_parameters()
    }
    save_file(
        tensors,
        str(temporary_path),
        metadata={
            "format": "chess-dfm-torch-model-v1",
            "source_mapping_sha256": source_mapping_sha256,
            "optimizer_update": str(optimizer_update),
            "data_cursor": str(data_cursor),
        },
    )
    del tensors
    os.replace(temporary_path, final_path)
    manifest = {
        "format": "chess-dfm-torch-model-v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "model_only": True,
        "optimizer_resume_supported": False,
        "optimizer_update": optimizer_update,
        "data_cursor": data_cursor,
        "source_mapping_sha256": source_mapping_sha256,
        "state": {
            "path": final_path.name,
            "size_bytes": final_path.stat().st_size,
            "sha256": _sha256_file(final_path),
            "leaf_count": len(tuple(model.named_parameters())),
        },
    }
    _write_json(checkpoint_dir / "manifest.json", manifest)
    return manifest


def save_training_checkpoint(
    *,
    output_dir: Path,
    model: nn.Module,
    optimizer: MuonAdamW,
    source_mapping_sha256: str,
    next_data_cursor: int,
    resume_contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Atomically write one exact model-and-optimizer recovery checkpoint."""

    from safetensors.torch import save_file

    if (
        optimizer._main_learning_rate_override is not None
        or optimizer._bt4_learning_rate_override is not None
    ):
        raise ValueError("Recovery checkpoints do not support LR overrides")
    optimizer_update = int(optimizer.update)
    optimizer_examples_seen = int(optimizer.examples_seen)
    next_data_cursor = int(next_data_cursor)
    if optimizer_update < 0 or optimizer_examples_seen < 0 or next_data_cursor < 0:
        raise ValueError("Recovery checkpoint counters must be non-negative")
    if (
        optimizer.examples_per_update is not None
        and optimizer_examples_seen
        != optimizer_update * optimizer.examples_per_update
    ):
        raise ValueError(
            "Optimizer examples_seen is inconsistent with update and batch size"
        )

    checkpoint_root = _require_workspace(output_dir / "checkpoints")
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    final_dir = checkpoint_root / f"update{optimizer_update:08d}"
    if final_dir.exists():
        raise FileExistsError(f"Recovery checkpoint already exists: {final_dir}")
    temporary_dir = Path(
        tempfile.mkdtemp(
            prefix=f".update{optimizer_update:08d}.partial-",
            dir=checkpoint_root,
        )
    )
    try:
        tensors: dict[str, Tensor] = {
            f"model.{name}": parameter.detach().cpu().contiguous()
            for name, parameter in model.named_parameters()
        }
        for leaf in optimizer.leaves:
            tensors[f"optimizer.first.{leaf.name}"] = (
                leaf.first_moment.detach().cpu().contiguous()
            )
            if leaf.second_moment is not None:
                tensors[f"optimizer.second.{leaf.name}"] = (
                    leaf.second_moment.detach().cpu().contiguous()
                )

        normalized_contract = dict(resume_contract)
        resume_contract_sha256 = _json_sha256(normalized_contract)
        partition = optimizer.partition_manifest()
        partition_sha256 = _json_sha256(partition)
        state_path = temporary_dir / "state.safetensors"
        save_file(
            tensors,
            str(state_path),
            metadata={
                "format": "chess-dfm-torch-training-v1",
                "optimizer_update": str(optimizer_update),
                "optimizer_examples_seen": str(optimizer_examples_seen),
                "next_data_cursor": str(next_data_cursor),
                "resume_contract_sha256": resume_contract_sha256,
                "source_mapping_sha256": source_mapping_sha256,
            },
        )
        tensor_count = len(tensors)
        del tensors
        _fsync_path(state_path)
        manifest = {
            "format": "chess-dfm-torch-training-v1",
            "created_utc": datetime.now(UTC).isoformat(),
            "model_only": False,
            "optimizer_resume_supported": True,
            "optimizer_update": optimizer_update,
            "optimizer_examples_seen": optimizer_examples_seen,
            "next_data_cursor": next_data_cursor,
            "source_mapping_sha256": source_mapping_sha256,
            "resume_contract": normalized_contract,
            "resume_contract_sha256": resume_contract_sha256,
            "optimizer_partition_sha256": partition_sha256,
            "state": {
                "path": state_path.name,
                "size_bytes": state_path.stat().st_size,
                "sha256": _sha256_file(state_path),
                "tensor_count": tensor_count,
                "model_leaf_count": len(tuple(model.named_parameters())),
                "first_moment_count": len(optimizer.leaves),
                "second_moment_count": sum(
                    leaf.second_moment is not None for leaf in optimizer.leaves
                ),
            },
        }
        manifest_path = temporary_dir / "manifest.json"
        _write_json(manifest_path, manifest)
        _fsync_path(manifest_path)
        _fsync_path(temporary_dir)
        os.replace(temporary_dir, final_dir)
        _fsync_path(checkpoint_root)
    except BaseException:
        shutil.rmtree(temporary_dir, ignore_errors=True)
        raise
    return manifest


def _verified_training_checkpoint(
    checkpoint_dir: Path,
    *,
    expected_resume_contract: Mapping[str, Any] | None = None,
) -> tuple[Path, dict[str, Any]]:
    root = _require_workspace(checkpoint_dir, exists=True)
    manifest_path = _require_workspace(root / "manifest.json", exists=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") != "chess-dfm-torch-training-v1":
        raise ValueError(
            f"Unsupported torch training checkpoint format: {manifest.get('format')}"
        )
    if manifest.get("model_only") is not False:
        raise ValueError("Training checkpoint must declare model_only=false")
    if manifest.get("optimizer_resume_supported") is not True:
        raise ValueError("Training checkpoint must support optimizer resume")
    contract = manifest.get("resume_contract")
    if not isinstance(contract, dict):
        raise ValueError("Training checkpoint has no resume contract")
    observed_contract_sha256 = _json_sha256(contract)
    if observed_contract_sha256 != manifest.get("resume_contract_sha256"):
        raise ValueError("Training checkpoint resume-contract checksum mismatch")
    if (
        expected_resume_contract is not None
        and _json_sha256(dict(expected_resume_contract))
        != observed_contract_sha256
    ):
        raise ValueError(
            "Training checkpoint resume contract does not match this run"
        )
    state_path = _require_workspace(root / manifest["state"]["path"], exists=True)
    if state_path.stat().st_size != int(manifest["state"]["size_bytes"]):
        raise ValueError(f"Training checkpoint size mismatch: {state_path}")
    if _sha256_file(state_path) != manifest["state"]["sha256"]:
        raise ValueError(f"Training checkpoint checksum mismatch: {state_path}")
    return state_path, manifest


def load_training_checkpoint(
    *,
    checkpoint_dir: Path,
    model: nn.Module,
    optimizer: MuonAdamW,
    expected_source_mapping_sha256: str,
    expected_resume_contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Strictly restore an exact model, optimizer, schedule, and data cursor."""

    from safetensors.torch import load_file

    state_path, manifest = _verified_training_checkpoint(
        checkpoint_dir,
        expected_resume_contract=expected_resume_contract,
    )
    if manifest.get("source_mapping_sha256") != expected_source_mapping_sha256:
        raise ValueError("Training checkpoint source mapping does not match")
    partition_sha256 = _json_sha256(optimizer.partition_manifest())
    if manifest.get("optimizer_partition_sha256") != partition_sha256:
        raise ValueError("Training checkpoint optimizer partition does not match")

    named = dict(model.named_parameters())
    leaves = {leaf.name: leaf for leaf in optimizer.leaves}
    if set(named) != set(leaves):
        raise ValueError("Optimizer/model leaf paths do not match")
    expected_keys = {f"model.{name}" for name in named}
    expected_keys.update(f"optimizer.first.{name}" for name in leaves)
    expected_keys.update(
        f"optimizer.second.{name}"
        for name, leaf in leaves.items()
        if leaf.second_moment is not None
    )
    loaded = load_file(str(state_path), device="cpu")
    if set(loaded) != expected_keys:
        raise ValueError(
            "Training checkpoint tensor mismatch: "
            f"missing={sorted(expected_keys - set(loaded))[:10]}, "
            f"extra={sorted(set(loaded) - expected_keys)[:10]}"
        )
    if int(manifest["state"]["tensor_count"]) != len(loaded):
        raise ValueError("Training checkpoint tensor count does not match manifest")

    def copy_exact(target: Tensor, source: Tensor, name: str) -> None:
        if source.shape != target.shape or source.dtype != target.dtype:
            raise ValueError(
                f"Training checkpoint ABI mismatch at {name}: "
                f"{source.shape}/{source.dtype} != {target.shape}/{target.dtype}"
            )
        target.copy_(source.to(target.device))

    with torch.no_grad():
        for name, parameter in named.items():
            copy_exact(parameter, loaded[f"model.{name}"], f"model.{name}")
        for name, leaf in leaves.items():
            copy_exact(
                leaf.first_moment,
                loaded[f"optimizer.first.{name}"],
                f"optimizer.first.{name}",
            )
            if leaf.second_moment is not None:
                copy_exact(
                    leaf.second_moment,
                    loaded[f"optimizer.second.{name}"],
                    f"optimizer.second.{name}",
                )

    optimizer.update = int(manifest["optimizer_update"])
    optimizer.examples_seen = int(manifest["optimizer_examples_seen"])
    next_data_cursor = int(manifest["next_data_cursor"])
    if optimizer.update < 0 or optimizer.examples_seen < 0 or next_data_cursor < 0:
        raise ValueError("Restored training counters must be non-negative")
    if (
        optimizer.examples_per_update is not None
        and optimizer.examples_seen
        != optimizer.update * optimizer.examples_per_update
    ):
        raise ValueError("Restored optimizer example counter is inconsistent")
    return manifest


def _verified_model_checkpoint(
    checkpoint_dir: Path,
) -> tuple[Path, dict[str, Any]]:
    root = _require_workspace(checkpoint_dir, exists=True)
    manifest_path = _require_workspace(root / "manifest.json", exists=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") != "chess-dfm-torch-model-v1":
        raise ValueError(f"Unsupported torch checkpoint format: {manifest.get('format')}")
    if manifest.get("model_only") is not True:
        raise ValueError("Torch checkpoint must declare model_only=true")
    state_path = _require_workspace(root / manifest["state"]["path"], exists=True)
    if state_path.stat().st_size != int(manifest["state"]["size_bytes"]):
        raise ValueError(f"Torch checkpoint size mismatch: {state_path}")
    if _sha256_file(state_path) != manifest["state"]["sha256"]:
        raise ValueError(f"Torch checkpoint checksum mismatch: {state_path}")
    return state_path, manifest


def load_model_checkpoint(
    *,
    checkpoint_dir: Path,
    model: JointModel,
) -> dict[str, Any]:
    """Strictly restore a model-only safetensors checkpoint."""

    from safetensors.torch import load_file

    state_path, manifest = _verified_model_checkpoint(checkpoint_dir)
    loaded = load_file(str(state_path), device="cpu")
    named = dict(model.named_parameters())
    if set(loaded) != set(named):
        raise ValueError(
            "Torch checkpoint leaf mismatch: "
            f"missing={sorted(set(named) - set(loaded))[:10]}, "
            f"extra={sorted(set(loaded) - set(named))[:10]}"
        )
    with torch.no_grad():
        for name, parameter in named.items():
            value = loaded[name]
            if value.shape != parameter.shape or value.dtype != parameter.dtype:
                raise ValueError(
                    f"Torch checkpoint ABI mismatch at {name}: "
                    f"{value.shape}/{value.dtype} != "
                    f"{parameter.shape}/{parameter.dtype}"
                )
            parameter.copy_(value.to(parameter.device))
    return manifest


def load_checkpoint_model_for_evaluation(
    *,
    checkpoint_dir: Path,
    model: nn.Module,
) -> dict[str, Any]:
    """Restore model tensors from either model-only or recovery state."""

    root = _require_workspace(checkpoint_dir, exists=True)
    manifest_path = _require_workspace(root / "manifest.json", exists=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    checkpoint_format = manifest.get("format")
    if checkpoint_format == "chess-dfm-torch-model-v1":
        return load_model_checkpoint(
            checkpoint_dir=root,
            model=model,
        )
    if checkpoint_format != "chess-dfm-torch-training-v1":
        raise ValueError(
            "Unsupported evaluation checkpoint format: "
            f"{checkpoint_format!r}"
        )

    from safetensors import safe_open

    state_path, verified_manifest = _verified_training_checkpoint(root)
    named = dict(model.named_parameters())
    with safe_open(state_path, framework="pt", device="cpu") as payload:
        loaded_keys = set(payload.keys())
        model_prefix = "model."
        loaded_model_names = {
            key[len(model_prefix) :]
            for key in loaded_keys
            if key.startswith(model_prefix)
        }
        if loaded_model_names != set(named):
            raise ValueError(
                "Training checkpoint model tensor mismatch: "
                f"missing={sorted(set(named) - loaded_model_names)[:10]}, "
                f"extra={sorted(loaded_model_names - set(named))[:10]}"
            )
        if int(verified_manifest["state"]["model_leaf_count"]) != len(named):
            raise ValueError(
                "Training checkpoint model leaf count does not match"
            )
        with torch.no_grad():
            for name, parameter in named.items():
                value = payload.get_tensor(f"{model_prefix}{name}")
                if (
                    value.shape != parameter.shape
                    or value.dtype != parameter.dtype
                ):
                    raise ValueError(
                        f"Training checkpoint model ABI mismatch at {name}: "
                        f"{value.shape}/{value.dtype} != "
                        f"{parameter.shape}/{parameter.dtype}"
                    )
                parameter.copy_(value.to(parameter.device))
                del value
    return verified_manifest


def load_model_checkpoint_numpy_tree(
    *,
    checkpoint_dir: Path,
    model: nn.Module,
) -> tuple[dict[str | int, Any], dict[str, Any], dict[str, Any]]:
    """Read a strict checkpoint as an exact NNX-compatible pure dictionary."""

    from safetensors import safe_open

    state_path, manifest = _verified_model_checkpoint(checkpoint_dir)
    named = dict(model.named_parameters())
    tree: dict[str | int, Any] = {}
    combined = hashlib.sha256()
    total_bytes = 0
    with safe_open(state_path, framework="pt", device="cpu") as payload:
        loaded_names = set(payload.keys())
        if loaded_names != set(named):
            raise ValueError(
                "Torch checkpoint leaf mismatch: "
                f"missing={sorted(set(named) - loaded_names)[:10]}, "
                f"extra={sorted(loaded_names - set(named))[:10]}"
            )
        if int(manifest["state"]["leaf_count"]) != len(loaded_names):
            raise ValueError(
                "Torch checkpoint manifest leaf count mismatch: "
                f"{manifest['state']['leaf_count']} != {len(loaded_names)}"
            )
        for name, parameter in named.items():
            value = payload.get_tensor(name).contiguous()
            if value.shape != parameter.shape or value.dtype != parameter.dtype:
                raise ValueError(
                    f"Torch checkpoint ABI mismatch at {name}: "
                    f"{value.shape}/{value.dtype} != "
                    f"{parameter.shape}/{parameter.dtype}"
                )
            if value.dtype == torch.bfloat16:
                array = value.view(torch.uint16).numpy().view(ml_dtypes.bfloat16)
            elif value.dtype == torch.float32:
                array = value.numpy()
            else:
                raise TypeError(f"Unsupported checkpoint dtype at {name}: {value.dtype}")
            leaf_digest = hashlib.sha256(array.tobytes(order="C")).hexdigest()
            combined.update(name.encode("utf-8"))
            combined.update(b"\0")
            combined.update(leaf_digest.encode("ascii"))
            total_bytes += int(array.nbytes)

            parts: tuple[str | int, ...] = tuple(
                int(part) if part.isdigit() else part for part in name.split(".")
            )
            cursor = tree
            for part in parts[:-1]:
                child = cursor.setdefault(part, {})
                if not isinstance(child, dict):
                    raise ValueError(f"Checkpoint tree path collision at {name}")
                cursor = child
            if parts[-1] in cursor:
                raise ValueError(f"Duplicate checkpoint tree leaf: {name}")
            cursor[parts[-1]] = array
    summary = {
        "schema_version": "torch-checkpoint-pure-tree-v1",
        "leaf_count": len(named),
        "nbytes": total_bytes,
        "combined_state_sha256": combined.hexdigest(),
        "safetensors_sha256": manifest["state"]["sha256"],
    }
    return tree, manifest, summary


def load_checkpoint_numpy_tree_for_evaluation(
    *,
    checkpoint_dir: Path,
    model: nn.Module,
) -> tuple[dict[str | int, Any], dict[str, Any], dict[str, Any]]:
    """Read a model tree from either model-only or recovery state."""

    root = _require_workspace(checkpoint_dir, exists=True)
    manifest_path = _require_workspace(root / "manifest.json", exists=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    checkpoint_format = manifest.get("format")
    if checkpoint_format == "chess-dfm-torch-model-v1":
        return load_model_checkpoint_numpy_tree(
            checkpoint_dir=root,
            model=model,
        )
    if checkpoint_format != "chess-dfm-torch-training-v1":
        raise ValueError(
            "Unsupported evaluation checkpoint format: "
            f"{checkpoint_format!r}"
        )

    from safetensors import safe_open

    state_path, verified_manifest = _verified_training_checkpoint(root)
    named = dict(model.named_parameters())
    tree: dict[str | int, Any] = {}
    combined = hashlib.sha256()
    total_bytes = 0
    model_prefix = "model."
    with safe_open(state_path, framework="pt", device="cpu") as payload:
        loaded_keys = set(payload.keys())
        loaded_model_names = {
            key[len(model_prefix) :]
            for key in loaded_keys
            if key.startswith(model_prefix)
        }
        if loaded_model_names != set(named):
            raise ValueError(
                "Training checkpoint model tensor mismatch: "
                f"missing={sorted(set(named) - loaded_model_names)[:10]}, "
                f"extra={sorted(loaded_model_names - set(named))[:10]}"
            )
        if (
            int(verified_manifest["state"]["model_leaf_count"])
            != len(loaded_model_names)
        ):
            raise ValueError(
                "Training checkpoint model leaf count does not match"
            )
        for name, parameter in named.items():
            value = payload.get_tensor(f"{model_prefix}{name}").contiguous()
            if value.shape != parameter.shape or value.dtype != parameter.dtype:
                raise ValueError(
                    f"Training checkpoint model ABI mismatch at {name}: "
                    f"{value.shape}/{value.dtype} != "
                    f"{parameter.shape}/{parameter.dtype}"
                )
            if value.dtype == torch.bfloat16:
                array = value.view(torch.uint16).numpy().view(
                    ml_dtypes.bfloat16
                )
            elif value.dtype == torch.float32:
                array = value.numpy()
            else:
                raise TypeError(
                    f"Unsupported checkpoint dtype at {name}: {value.dtype}"
                )
            leaf_digest = hashlib.sha256(
                array.tobytes(order="C")
            ).hexdigest()
            combined.update(name.encode("utf-8"))
            combined.update(b"\0")
            combined.update(leaf_digest.encode("ascii"))
            total_bytes += int(array.nbytes)

            parts: tuple[str | int, ...] = tuple(
                int(part) if part.isdigit() else part
                for part in name.split(".")
            )
            cursor = tree
            for part in parts[:-1]:
                child = cursor.setdefault(part, {})
                if not isinstance(child, dict):
                    raise ValueError(
                        f"Checkpoint tree path collision at {name}"
                    )
                cursor = child
            if parts[-1] in cursor:
                raise ValueError(f"Duplicate checkpoint tree leaf: {name}")
            cursor[parts[-1]] = array
    summary = {
        "schema_version": "torch-checkpoint-pure-tree-v1",
        "leaf_count": len(named),
        "nbytes": total_bytes,
        "combined_state_sha256": combined.hexdigest(),
        "safetensors_sha256": verified_manifest["state"]["sha256"],
    }
    return tree, verified_manifest, summary


def _start_gpu_monitor(
    output_dir: Path,
    *,
    interval_ms: int,
    append: bool = False,
) -> tuple[subprocess.Popen[str], IO[str], IO[str]]:
    mode = "a" if append else "w"
    samples = (output_dir / "gpu_samples.csv").open(mode, encoding="utf-8")
    stderr = (output_dir / "gpu_monitor.stderr.log").open(
        mode,
        encoding="utf-8",
    )
    fields = (
        "timestamp,utilization.gpu,utilization.memory,memory.used,memory.total,"
        "power.draw,clocks.sm,clocks.mem"
    )
    monitor_env = os.environ.copy()
    local_nvml = monitor_env.get("CHESS_DFM_NVML_LIBRARY_DIR")
    if local_nvml:
        nvml_dir = _require_workspace(Path(local_nvml), exists=True)
        if not (nvml_dir / "libnvidia-ml.so.1").is_file():
            raise FileNotFoundError(
                f"Workspace-local NVML library is missing from {nvml_dir}"
            )
        existing_library_path = monitor_env.get("LD_LIBRARY_PATH")
        monitor_env["LD_LIBRARY_PATH"] = (
            str(nvml_dir)
            if not existing_library_path
            else f"{nvml_dir}{os.pathsep}{existing_library_path}"
        )
    process = subprocess.Popen(
        [
            "nvidia-smi",
            f"--query-gpu={fields}",
            "--format=csv,noheader,nounits",
            f"--loop-ms={interval_ms}",
        ],
        stdout=samples,
        stderr=stderr,
        text=True,
        env=monitor_env,
    )
    return process, samples, stderr


def _stop_gpu_monitor(
    monitor: tuple[subprocess.Popen[str], IO[str], IO[str]] | None,
) -> None:
    if monitor is None:
        return
    process, samples, stderr = monitor
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)
    samples.close()
    stderr.close()


def _summarize_gpu_samples(path: Path) -> dict[str, float | int]:
    columns = {
        "gpu_utilization_percent": 1,
        "memory_utilization_percent": 2,
        "memory_used_mib": 3,
        "memory_total_mib": 4,
        "power_watts": 5,
        "sm_clock_mhz": 6,
        "memory_clock_mhz": 7,
    }
    values: dict[str, list[float]] = {name: [] for name in columns}
    if not path.is_file():
        return {"sample_count": 0}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.reader(handle):
            if len(row) != 8:
                continue
            try:
                for name, index in columns.items():
                    values[name].append(float(row[index].strip()))
            except ValueError:
                continue
    count = len(values["gpu_utilization_percent"])
    summary: dict[str, float | int] = {"sample_count": count}
    if count == 0:
        return summary
    for name, series in values.items():
        array = np.asarray(series, dtype=np.float64)
        summary[f"{name}_mean"] = float(np.mean(array))
        summary[f"{name}_p50"] = float(np.quantile(array, 0.50))
        summary[f"{name}_p95"] = float(np.quantile(array, 0.95))
        summary[f"{name}_max"] = float(np.max(array))
    return summary


def _apply_compile_regions(model: JointModel, regions: str) -> list[str]:
    """Compile bounded repeated regions without changing parameter/checkpoint paths."""

    if regions == "none":
        return []
    if regions not in {
        "dfm-jepa",
        "fresh",
        "fresh-bt4-smolgen",
        "fresh-bt4-smolgen-eager-numerics",
        "fresh-bt4",
        "fresh-bt4-eager-numerics",
        "fresh-bt4-strict-numerics",
    }:
        raise ValueError(f"Unsupported compile region set: {regions!r}")
    os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")
    import torch._inductor.config

    torch._inductor.config.compile_threads = 1
    modules: list[tuple[str, nn.Module]] = [
        ("dfm_blocks", model.dfm_blocks),
        ("jepa_transition", model.jepa_transition),
    ]
    compiled: list[str] = []
    if regions == "fresh":
        modules.insert(0, ("state_projector_blocks", model.state_projector.blocks))
    elif regions in {
        "fresh-bt4-smolgen",
        "fresh-bt4-smolgen-eager-numerics",
    }:
        modules.insert(0, ("state_projector_blocks", model.state_projector.blocks))
        for layer in model.encoder.layers:
            smolgen_compile_kwargs: dict[str, Any] = {
                "backend": "inductor",
                "fullgraph": True,
                "dynamic": False,
            }
            if regions == "fresh-bt4-smolgen-eager-numerics":
                smolgen_compile_kwargs["options"] = {
                    "emulate_precision_casts": True,
                }
            else:
                smolgen_compile_kwargs["mode"] = "default"
            layer.smolgen.forward = torch.compile(  # type: ignore[method-assign]
                layer.smolgen.forward,
                **smolgen_compile_kwargs,
            )
        compiled.append(
            {
                "fresh-bt4-smolgen": "bt4_smolgen_modules",
                "fresh-bt4-smolgen-eager-numerics": (
                    "bt4_smolgen_modules_eager_numerics"
                ),
            }[regions]
        )
    elif regions in {
        "fresh-bt4",
        "fresh-bt4-eager-numerics",
        "fresh-bt4-strict-numerics",
    }:
        modules.insert(0, ("state_projector_blocks", model.state_projector.blocks))
        # TorchDynamo's cache is keyed by the forward code object, so all 15
        # structurally identical layer instances reuse regional compilations.
        # Keep separate wrappers so parameters and state-dict paths stay native.
        for layer in model.encoder.layers:
            bt4_compile_kwargs: dict[str, Any] = {
                "backend": "inductor",
                "fullgraph": True,
                "dynamic": False,
            }
            if regions in {
                "fresh-bt4-eager-numerics",
                "fresh-bt4-strict-numerics",
            }:
                bt4_compile_kwargs["options"] = {
                    "emulate_precision_casts": True,
                    **(
                        {"epilogue_fusion": False}
                        if regions == "fresh-bt4-strict-numerics"
                        else {}
                    ),
                }
            else:
                bt4_compile_kwargs["mode"] = "default"
            layer.forward = torch.compile(  # type: ignore[method-assign]
                layer.forward,
                **bt4_compile_kwargs,
            )
        compiled.append(
            {
                "fresh-bt4": "bt4_encoder_layers",
                "fresh-bt4-eager-numerics": "bt4_encoder_layers_eager_numerics",
                "fresh-bt4-strict-numerics": "bt4_encoder_layers_strict_numerics",
            }[regions]
        )
    for name, module in modules:
        module.forward = torch.compile(  # type: ignore[method-assign]
            module.forward,
            backend="inductor",
            fullgraph=True,
            dynamic=False,
            mode="default",
        )
        compiled.append(name)
    return compiled


def _profiler_metric(event: Any, *names: str) -> float:
    for name in names:
        value = getattr(event, name, None)
        if value is not None:
            return float(value)
    return 0.0


def _write_profiler_artifacts(
    profiler: Any,
    output_dir: Path,
    *,
    profiled_update: int,
) -> dict[str, Any]:
    """Persist one bounded profiler window as a compact summary and gzip trace."""

    key_averages = list(profiler.key_averages(group_by_input_shape=False))
    rows = [
        {
            "name": str(event.key),
            "count": int(event.count),
            "self_cuda_time_us": _profiler_metric(
                event,
                "self_device_time_total",
                "self_cuda_time_total",
            ),
            "cuda_time_us": _profiler_metric(
                event,
                "device_time_total",
                "cuda_time_total",
            ),
            "self_cpu_time_us": float(event.self_cpu_time_total),
            "cpu_time_us": float(event.cpu_time_total),
            "flops": int(getattr(event, "flops", 0) or 0),
        }
        for event in key_averages
    ]
    top_cuda = sorted(
        rows,
        key=lambda row: (row["self_cuda_time_us"], row["cuda_time_us"]),
        reverse=True,
    )[:100]
    top_cpu = sorted(
        rows,
        key=lambda row: (row["self_cpu_time_us"], row["cpu_time_us"]),
        reverse=True,
    )[:50]
    regions = sorted(
        (
            row
            for row in rows
            if str(row["name"]).startswith("region::")
        ),
        key=lambda row: str(row["name"]),
    )
    table = profiler.key_averages().table(
        sort_by="self_cuda_time_total",
        row_limit=100,
    )
    table_path = _require_workspace(output_dir / "profile_table.txt")
    table_path.write_text(table + "\n", encoding="utf-8")

    raw_trace = _require_workspace(output_dir / ".profile_trace.json.partial")
    compressed_trace = _require_workspace(output_dir / "profile_trace.json.gz")
    profiler.export_chrome_trace(str(raw_trace))
    with raw_trace.open("rb") as source, gzip.open(
        compressed_trace,
        "wb",
        compresslevel=6,
    ) as target:
        shutil.copyfileobj(source, target, length=_HASH_CHUNK_BYTES)
    raw_trace.unlink()

    summary = {
        "schema_version": "torch-profiler-single-update-v1",
        "profiled_update": int(profiled_update),
        "activities": ["cpu", "cuda"],
        "record_shapes": True,
        "profile_memory": True,
        "with_flops": True,
        "event_key_count": len(rows),
        "aggregate_flops": int(sum(row["flops"] for row in rows)),
        "top_ops_by_self_cuda_time": top_cuda,
        "top_ops_by_self_cpu_time": top_cpu,
        "profile_regions": regions,
        "table": {
            "path": table_path.name,
            "size_bytes": table_path.stat().st_size,
            "sha256": _sha256_file(table_path),
        },
        "trace": {
            "path": compressed_trace.name,
            "size_bytes": compressed_trace.stat().st_size,
            "sha256": _sha256_file(compressed_trace),
            "compression": "gzip",
        },
    }
    _write_json(output_dir / "profile_summary.json", summary)
    return summary


def _compile_counter_snapshot() -> dict[str, dict[str, int]]:
    try:
        from torch._dynamo.utils import counters
    except (ImportError, AttributeError):
        return {}
    return {
        str(category): {
            str(key): int(value)
            for key, value in values.items()
        }
        for category, values in counters.items()
        if values
    }


def _training_resume_contract(
    *,
    args: argparse.Namespace,
    config: Config,
    source_mapping_sha256: str,
    optimizer_partition: Mapping[str, Any],
    data_provenance: Mapping[str, Any],
    compiled_regions: Sequence[str],
    git_commit: str,
) -> dict[str, Any]:
    """Pin every state-independent input needed for exact stateless resume."""

    return {
        "schema_version": "torch-training-resume-contract-v1",
        "git_commit": git_commit,
        "framework": "torch",
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "compute_dtype": "torch.bfloat16",
        "recipe": args.recipe,
        "config": dataclasses.asdict(config),
        "source_mapping_sha256": source_mapping_sha256,
        "optimizer_partition": dict(optimizer_partition),
        "data": dict(data_provenance),
        "schedule": {
            "target_updates": args.steps,
            "batch_size": args.batch_size,
            "seed": args.seed,
            "initial_data_cursor": args.data_start,
        },
        "runtime": {
            "remat_mode": args.remat_mode,
            "attention_impl": args.attention_impl,
            "compile_regions": args.compile_regions,
            "compiled_regions": list(compiled_regions),
            "bt4_norm_impl": getattr(args, "bt4_norm_impl", "eager"),
            "prefetch_depth": args.prefetch_depth,
            "prefetch_launch": getattr(
                args,
                "prefetch_launch",
                "step-start",
            ),
            "hero_milestones": _hero_milestone_contract(
                args,
                enabled=bool(
                    getattr(args, "hero_milestones", False)
                ),
            ),
        },
        "stochastic_state": (
            "all step choices derive from seed/update; data derives from "
            "seed/data_cursor; no mutable RNG state"
        ),
    }


def train(args: argparse.Namespace) -> int:
    config = HERO_CONFIG if args.recipe == "hero" else CONFIG
    bt4_norm_impl = getattr(args, "bt4_norm_impl", "eager")
    prefetch_launch = getattr(args, "prefetch_launch", "step-start")
    if args.recipe != "hero" and bt4_norm_impl != "eager":
        raise ValueError("Fused BT4 LayerNorm is currently a hero-only runtime")
    hero_milestones_enabled = bool(
        getattr(args, "hero_milestones", False)
    )
    lr_range_start = getattr(args, "lr_range_start", None)
    lr_range_end = getattr(args, "lr_range_end", None)
    lr_range_enabled = lr_range_start is not None or lr_range_end is not None
    if lr_range_enabled:
        if lr_range_start is None or lr_range_end is None:
            raise ValueError("LR range requires both start and end rates")
        config = dataclasses.replace(config, weight_decay=0.0)
    remat_modes = {
        "all": (True, True, True),
        "none": (False, False, False),
        "bt4-only": (True, False, False),
        "bt4-projector": (True, True, False),
        "bt4-dfm": (True, False, True),
        "heads-only": (False, True, True),
    }
    remat_bt4, remat_projector, remat_dfm = remat_modes[args.remat_mode]
    config = dataclasses.replace(
        config,
        remat_bt4_blocks=remat_bt4,
        remat_projector_blocks=remat_projector,
        remat_dfm_blocks=remat_dfm,
        use_bt4_sdpa=args.attention_impl == "sdpa-all",
        use_head_sdpa=args.attention_impl in {"sdpa-heads", "sdpa-all"},
    )
    if not torch.cuda.is_available():
        raise RuntimeError("research/train_torch.py train requires CUDA")
    if args.steps < 0 or args.train_seconds < 0:
        raise ValueError("--steps and --train-seconds must be non-negative")
    if args.steps == 0 and args.train_seconds == 0:
        raise ValueError("Set --steps or --train-seconds")
    if args.batch_size < config.sigreg_example_count:
        raise ValueError(
            "The accepted fixed-64 SIGReg contract requires --batch-size >= "
            f"{config.sigreg_example_count}"
        )
    if args.log_every < 1:
        raise ValueError("--log-every must be positive")
    if args.threads not in (1, 2):
        raise ValueError("--threads must be 1 or 2 under the resource guard")
    if args.prefetch_depth not in (0, 1):
        raise ValueError("--prefetch-depth must be 0 or 1")
    if args.gpu_monitor_interval_ms != 0 and args.gpu_monitor_interval_ms < 50:
        raise ValueError("--gpu-monitor-interval-ms must be 0 or at least 50")
    if args.profile_update < 0:
        raise ValueError("--profile-update must be non-negative")
    if args.profile_update > 0 and args.steps > 0 and args.profile_update > args.steps:
        raise ValueError("--profile-update cannot exceed --steps")
    save_updates = tuple(int(value) for value in args.save_updates)
    if args.save_every != 0:
        raise ValueError(
            "Periodic checkpoints are disabled; use one explicit --save-updates value"
        )
    if len(save_updates) > 1 or len(set(save_updates)) != len(save_updates):
        raise ValueError("Use at most one unique sparse recovery checkpoint")
    if save_updates and (
        args.steps <= 0
        or save_updates[0] <= 0
        or save_updates[0] > args.steps
    ):
        raise ValueError("Sparse checkpoint update must be in [1, --steps]")
    if args.max_checkpoints not in (0, 1, 2):
        raise ValueError("--max-checkpoints must be 0, 1, or 2")
    planned_checkpoints = len(save_updates) + int(args.save_final)
    if planned_checkpoints > args.max_checkpoints:
        raise ValueError(
            "Requested checkpoint writes exceed --max-checkpoints: "
            f"{planned_checkpoints} > {args.max_checkpoints}"
        )
    if args.resume_checkpoint is not None and lr_range_enabled:
        raise ValueError("LR-range calibration cannot resume")
    if hero_milestones_enabled:
        if args.recipe != "hero" or lr_range_enabled:
            raise ValueError(
                "Hero milestone instrumentation requires the clean hero recipe"
            )
        if args.batch_size != 1024:
            raise ValueError(
                "The frozen hero milestone run requires batch size 1024"
            )
        if (
            args.steps * args.batch_size != _HERO_TRAIN_EXAMPLES
            or args.train_seconds != 0.0
        ):
            raise ValueError(
                "Hero milestones require exactly one example-count epoch"
            )
        if (
            args.remat_mode != "bt4-projector"
            or args.attention_impl != "sdpa-all"
            or args.compile_regions != "fresh"
            or args.prefetch_depth != 1
        ):
            raise ValueError(
                "Hero milestones require the frozen compiled Torch runtime"
            )
        if args.data_start != 0 or args.seed != 0:
            raise ValueError(
                "The frozen hero milestone run requires data-start 0 and seed 0"
            )
        if (
            args.hero_arena_pairs != _HERO_ARENA_PAIRS
            or args.hero_arena_additional_ply_cap
            != _HERO_ARENA_ADDITIONAL_PLY_CAP
            or args.hero_arena_inference_batch_size
            != _HERO_ARENA_INFERENCE_BATCH_SIZE
        ):
            raise ValueError("Hero arena milestone contract drift")
        halfway_update = _hero_milestone_update(
            50,
            total_examples=_HERO_TRAIN_EXAMPLES,
            batch_size=args.batch_size,
        )
        if args.resume_checkpoint is None:
            if (
                save_updates != (halfway_update,)
                or not args.save_final
                or args.max_checkpoints != 2
            ):
                raise ValueError(
                    "A fresh hero epoch requires one halfway recovery state "
                    "and one terminal checkpoint"
                )
        elif (
            save_updates
            or not args.save_final
            or args.max_checkpoints not in (1, 2)
        ):
            raise ValueError(
                "A resumed hero epoch must write only the terminal checkpoint"
            )
    if lr_range_enabled:
        assert lr_range_start is not None and lr_range_end is not None
        if args.recipe != "hero":
            raise ValueError("LR range is defined only for the clean hero recipe")
        if args.steps < 2 or args.train_seconds != 0.0:
            raise ValueError("LR range requires --steps >= 2 and no time limit")
        if args.save_final or args.max_checkpoints != 0:
            raise ValueError("LR range must discard model state and write no checkpoint")
        if (
            not math.isfinite(lr_range_start)
            or not math.isfinite(lr_range_end)
            or lr_range_start <= 0.0
            or lr_range_end <= lr_range_start
        ):
            raise ValueError("LR range must satisfy 0 < start < end")

    output_dir = _require_workspace(args.output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Run output already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    metrics_path = output_dir / "metrics.jsonl"
    device = torch.device("cuda")
    torch.set_num_threads(args.threads)
    torch.set_float32_matmul_precision("high")

    restore_started = time.perf_counter()
    if args.recipe == "hero":
        model, source_mapping = load_raw_bt4_hero_model(
            device=device,
            raw_bt4_path=args.raw_bt4_path,
            config=config,
            bt4_norm_impl=bt4_norm_impl,
        )
        source_record = source_mapping
    else:
        model, source_mapping = load_source_model(
            device=device,
            source_path=args.source_state,
            config=config,
        )
        source_record = {
            "path": str(_require_workspace(args.source_state, exists=True)),
            "size_bytes": _SOURCE_SIZE_BYTES,
            "sha256": _SOURCE_SHA256,
            "step": _SOURCE_STEP,
            "mapping_sha256": source_mapping["combined_sha256"],
            "leaf_count": source_mapping["leaf_count"],
            "model_nbytes": source_mapping["nbytes"],
            "init": "model-only",
            "optimizer": "fresh",
        }
    model.train()
    compiled_regions = _apply_compile_regions(model, args.compile_regions)
    optimizer = MuonAdamW(
        model,
        config,
        examples_per_update=(
            args.batch_size if config.lr_schedule_unit == "examples" else None
        ),
    )
    partition = optimizer.partition_manifest()
    _write_json(output_dir / "optimizer_partition.json", partition)

    batches_class = (
        CanonicalTrajectoryBatches
        if config.action_codec == "lc0_canonical_1858"
        else _FixedTrajectoryBatches
    )
    batches = batches_class(
        _require_workspace(args.data_root) / "train",
        batch_size=args.batch_size,
        horizon=config.horizon,
        seed=args.seed,
        shuffle_files=True,
        batch_schedule="global_permutation",
    )
    git_commit = _git_commit()
    data_provenance = batches.provenance()
    hero_milestone_contract = _hero_milestone_contract(
        args,
        enabled=hero_milestones_enabled,
    )
    hero_milestone_resources = (
        _load_hero_milestone_resources(
            manifest_path=args.hero_eval_manifest,
            arena_pairs=args.hero_arena_pairs,
        )
        if hero_milestones_enabled
        else None
    )
    resume_contract = _training_resume_contract(
        args=args,
        config=config,
        source_mapping_sha256=source_mapping["combined_sha256"],
        optimizer_partition=partition,
        data_provenance=data_provenance,
        compiled_regions=compiled_regions,
        git_commit=git_commit,
    )
    resume_manifest = None
    if args.resume_checkpoint is not None:
        resume_manifest = load_training_checkpoint(
            checkpoint_dir=args.resume_checkpoint,
            model=model,
            optimizer=optimizer,
            expected_source_mapping_sha256=source_mapping["combined_sha256"],
            expected_resume_contract=resume_contract,
        )
        initial_update = int(resume_manifest["optimizer_update"])
        initial_data_cursor = int(resume_manifest["next_data_cursor"])
        if args.steps > 0 and initial_update >= args.steps:
            raise ValueError(
                "Resume checkpoint is already at or beyond --steps: "
                f"{initial_update} >= {args.steps}"
            )
        if save_updates and save_updates[0] <= initial_update:
            raise ValueError(
                "Sparse checkpoint update must be after the restored update"
            )
    else:
        initial_update = 0
        initial_data_cursor = args.data_start
    restore_seconds = time.perf_counter() - restore_started
    run_config = {
        "schema_version": "torch-eager-train-run-v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "git_commit": git_commit,
        "framework": "torch",
        "execution": (
            "eager" if args.compile_regions == "none" else "regional-compile"
        ),
        "torch_compile": args.compile_regions != "none",
        "compile_regions": compiled_regions,
        "compile_settings": {
            "backend": "inductor",
            "fullgraph": True,
            "dynamic": False,
            "mode": "default",
            "compile_threads": 1,
            "max_autotune": False,
        },
        "remat_mode": args.remat_mode,
        "attention_impl": args.attention_impl,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "device": torch.cuda.get_device_name(device),
        "recipe": args.recipe,
        "config": dataclasses.asdict(config),
        "args": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
            if key != "handler"
        },
        "source": source_record,
        "optimizer_partition": {key: value for key, value in partition.items() if key != "leaves"},
        "data": data_provenance,
        "resume_contract": resume_contract,
        "resume_contract_sha256": _json_sha256(resume_contract),
        "resume": (
            {
                "enabled": True,
                "checkpoint_dir": str(
                    _require_workspace(args.resume_checkpoint, exists=True)
                ),
                "optimizer_update": initial_update,
                "next_data_cursor": initial_data_cursor,
                "checkpoint_state_sha256": resume_manifest["state"]["sha256"],
            }
            if resume_manifest is not None
            else {"enabled": False}
        ),
        "prefetch": {
            "depth": args.prefetch_depth,
            "workers": 1 if args.prefetch_depth == 1 else 0,
            "launch": prefetch_launch,
            "deterministic_update_and_cursor_keys": True,
        },
        "hero_milestones": (
            {
                **hero_milestone_contract,
                "fast_pool": hero_milestone_resources.fast_pool,
                "paired_arena_provenance": (
                    hero_milestone_resources.arena_provenance
                ),
            }
            if hero_milestone_resources is not None
            else hero_milestone_contract
        ),
        "lr_range": (
            {
                "enabled": True,
                "start_main_learning_rate": lr_range_start,
                "end_main_learning_rate": lr_range_end,
                "main_to_bt4_ratio": (
                    HERO_CONFIG.learning_rate / HERO_CONFIG.bt4_learning_rate
                ),
                "spacing": "exponential_per_update",
                "weight_decay": 0.0,
                "model_state_retained": False,
            }
            if lr_range_enabled
            else {"enabled": False}
        ),
        "restore_seconds": restore_seconds,
    }
    _write_json(output_dir / "run_config.json", run_config)

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    run_started = time.perf_counter()
    deadline = run_started + args.train_seconds if args.train_seconds > 0 else None
    records: list[dict[str, Any]] = []
    profiler_summary: dict[str, Any] | None = None
    update = initial_update
    data_cursor = initial_data_cursor
    segment_start_update = initial_update
    saved_recovery_checkpoints: list[dict[str, Any]] = []
    checkpoint_save_seconds = 0.0
    milestone_evaluation_seconds = 0.0
    hero_validation_records: list[dict[str, Any]] = []
    hero_arena_records: list[dict[str, Any]] = []
    validation_milestones_by_update = (
        {
            _hero_milestone_update(
                percentage,
                total_examples=_HERO_TRAIN_EXAMPLES,
                batch_size=args.batch_size,
            ): percentage
            for percentage in _HERO_FAST_VALIDATION_PERCENTAGES
        }
        if hero_milestones_enabled
        else {}
    )
    arena_milestones_by_update = (
        {
            _hero_milestone_update(
                percentage,
                total_examples=_HERO_TRAIN_EXAMPLES,
                batch_size=args.batch_size,
            ): percentage
            for percentage in _HERO_ARENA_PERCENTAGES
        }
        if hero_milestones_enabled
        else {}
    )
    monitor = (
        _start_gpu_monitor(
            output_dir,
            interval_ms=args.gpu_monitor_interval_ms,
        )
        if args.gpu_monitor_interval_ms > 0
        else None
    )

    def run_live_milestones(
        milestone_update: int,
        *,
        restart_monitor: bool,
    ) -> None:
        nonlocal monitor, milestone_evaluation_seconds
        validation_percentage = validation_milestones_by_update.get(
            milestone_update
        )
        arena_percentage = arena_milestones_by_update.get(milestone_update)
        if (
            hero_milestone_resources is None
            or (
                validation_percentage is None
                and arena_percentage is None
            )
        ):
            return
        milestone_started = time.perf_counter()
        _stop_gpu_monitor(monitor)
        monitor = None
        milestone_succeeded = False
        try:
            if validation_percentage is not None:
                hero_validation_records.append(
                    _run_hero_validation_milestone(
                        model,
                        hero_milestone_resources,
                        output_dir=output_dir,
                        update=milestone_update,
                        batch_size=args.batch_size,
                        percentage=validation_percentage,
                        device=device,
                    )
                )
            if arena_percentage is not None:
                hero_arena_records.append(
                    _run_hero_arena_milestone(
                        model,
                        hero_milestone_resources,
                        output_dir=output_dir,
                        raw_bt4_path=args.raw_bt4_path,
                        update=milestone_update,
                        batch_size=args.batch_size,
                        percentage=arena_percentage,
                        arena_pairs=args.hero_arena_pairs,
                        additional_ply_cap=(
                            args.hero_arena_additional_ply_cap
                        ),
                        inference_batch_size=(
                            args.hero_arena_inference_batch_size
                        ),
                        device=device,
                    )
                )
            milestone_succeeded = True
        finally:
            try:
                if (
                    milestone_succeeded
                    and restart_monitor
                    and args.gpu_monitor_interval_ms > 0
                ):
                    monitor = _start_gpu_monitor(
                        output_dir,
                        interval_ms=args.gpu_monitor_interval_ms,
                        append=True,
                    )
            finally:
                milestone_evaluation_seconds += (
                    time.perf_counter() - milestone_started
                )

    if (
        hero_milestones_enabled
        and initial_update > 0
        and (
            initial_update in validation_milestones_by_update
            or initial_update in arena_milestones_by_update
        )
    ):
        run_live_milestones(
            initial_update,
            restart_monitor=(
                args.steps == 0 or initial_update < args.steps
            ),
        )

    prefetch_executor = (
        ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="trajectory-prefetch",
        )
        if args.prefetch_depth == 1
        else None
    )
    prepared_future: Future[_PreparedTrainingStep] | None = None
    if prefetch_executor is not None:
        prepared_future = prefetch_executor.submit(
            _prepare_training_step,
            batches,
            seed=args.seed,
            update=update,
            data_cursor=data_cursor,
            batch_size=args.batch_size,
            config=config,
        )
    try:
        while (args.steps == 0 or update < args.steps) and (
            deadline is None or time.perf_counter() < deadline
        ):
            data_wait_started = time.perf_counter()
            if prepared_future is None:
                prepared = _prepare_training_step(
                    batches,
                    seed=args.seed,
                    update=update,
                    data_cursor=data_cursor,
                    batch_size=args.batch_size,
                    config=config,
                )
            else:
                prepared = prepared_future.result()
            data_wait_seconds = time.perf_counter() - data_wait_started
            if prepared.update != update or prepared.data_cursor != data_cursor:
                raise RuntimeError(
                    "Prefetch schedule drift: "
                    f"{prepared.update}/{prepared.data_cursor} != "
                    f"{update}/{data_cursor}"
                )
            choices_cpu = prepared.choices
            compact_batch = prepared.compact_batch
            data_prepare_seconds = prepared.prepare_seconds

            more_steps = args.steps == 0 or update + 1 < args.steps
            before_deadline = deadline is None or time.perf_counter() < deadline
            launch_next_prefetch = (
                prefetch_executor is not None
                and more_steps
                and before_deadline
            )
            if launch_next_prefetch and prefetch_launch == "step-start":
                prepared_future = prefetch_executor.submit(
                    _prepare_training_step,
                    batches,
                    seed=args.seed,
                    update=update + 1,
                    data_cursor=data_cursor + 1,
                    batch_size=args.batch_size,
                    config=config,
                )
            else:
                prepared_future = None

            lr_range_position: float | None = None
            if lr_range_enabled:
                assert lr_range_start is not None and lr_range_end is not None
                lr_range_position = update / (args.steps - 1)
                main_learning_rate = math.exp(
                    math.log(lr_range_start)
                    + lr_range_position
                    * (math.log(lr_range_end) - math.log(lr_range_start))
                )
                optimizer.set_learning_rates(
                    main=main_learning_rate,
                    bt4=(
                        main_learning_rate
                        * HERO_CONFIG.bt4_learning_rate
                        / HERO_CONFIG.learning_rate
                    ),
                )

            transfer_started = time.perf_counter()
            batch = _torch_batch(compact_batch, device)
            choices = _choices_to_device(choices_cpu, device)
            torch.cuda.synchronize()
            transfer_seconds = time.perf_counter() - transfer_started

            step_started = time.perf_counter()
            profile_this_update = args.profile_update == update + 1
            active_profiler = None
            if profile_this_update:
                active_profiler = torch.profiler.profile(
                    activities=[
                        torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA,
                    ],
                    record_shapes=True,
                    profile_memory=True,
                    with_stack=False,
                    with_flops=True,
                )
                active_profiler.start()
            event_start = torch.cuda.Event(enable_timing=True)
            event_forward = torch.cuda.Event(enable_timing=True)
            event_backward = torch.cuda.Event(enable_timing=True)
            event_optimizer = torch.cuda.Event(enable_timing=True)
            event_start.record()
            loss, aux = loss_and_aux(
                model,
                batch,
                choices,
                compute_dtype=torch.bfloat16,
                profile_regions=profile_this_update,
            )
            event_forward.record()
            if launch_next_prefetch and prefetch_launch == "after-forward":
                assert prefetch_executor is not None
                prepared_future = prefetch_executor.submit(
                    _prepare_training_step,
                    batches,
                    seed=args.seed,
                    update=update + 1,
                    data_cursor=data_cursor + 1,
                    batch_size=args.batch_size,
                    config=config,
                )
            with _profile_scope(profile_this_update, "region::backward"):
                loss.backward()
            event_backward.record()
            with _profile_scope(profile_this_update, "region::optimizer"):
                optimizer_metrics = optimizer.step()
            event_optimizer.record()
            torch.cuda.synchronize()
            step_seconds = time.perf_counter() - step_started
            if active_profiler is not None:
                active_profiler.stop()
                profiler_summary = _write_profiler_artifacts(
                    active_profiler,
                    output_dir,
                    profiled_update=update + 1,
                )
            forward_cuda_seconds = event_start.elapsed_time(event_forward) / 1000.0
            backward_cuda_seconds = event_forward.elapsed_time(event_backward) / 1000.0
            optimizer_cuda_seconds = event_backward.elapsed_time(event_optimizer) / 1000.0
            update += 1
            data_cursor += 1
            if optimizer.update != update:
                raise RuntimeError(
                    "Optimizer/local update drift: "
                    f"{optimizer.update} != {update}"
                )
            checkpoint_write_seconds = 0.0
            recovery_checkpoint_path = None
            if update in save_updates:
                checkpoint_started = time.perf_counter()
                recovery_manifest = save_training_checkpoint(
                    output_dir=output_dir,
                    model=model,
                    optimizer=optimizer,
                    source_mapping_sha256=source_mapping["combined_sha256"],
                    next_data_cursor=data_cursor,
                    resume_contract=resume_contract,
                )
                checkpoint_write_seconds = time.perf_counter() - checkpoint_started
                checkpoint_save_seconds += checkpoint_write_seconds
                recovery_checkpoint_path = str(
                    output_dir / "checkpoints" / f"update{update:08d}"
                )
                saved_recovery_checkpoints.append(
                    {
                        "path": recovery_checkpoint_path,
                        "optimizer_update": update,
                        "next_data_cursor": data_cursor,
                        "state": recovery_manifest["state"],
                        "write_seconds": checkpoint_write_seconds,
                    }
                )
            elapsed = (
                time.perf_counter()
                - run_started
                - milestone_evaluation_seconds
            )
            segment_updates = update - segment_start_update
            record = {
                "schema_version": "torch-eager-train-metrics-v1",
                "update": update,
                "segment_update": segment_updates,
                "optimizer_update": int(optimizer_metrics["optimizer_update"]),
                "data_cursor": data_cursor,
                "examples": update * args.batch_size,
                "segment_examples": segment_updates * args.batch_size,
                "elapsed_seconds": elapsed,
                "data_seconds": data_wait_seconds,
                "data_wait_seconds": data_wait_seconds,
                "data_prepare_seconds": data_prepare_seconds,
                "data_prefetch_hidden_seconds": max(
                    data_prepare_seconds - data_wait_seconds,
                    0.0,
                ),
                "transfer_seconds": transfer_seconds,
                "step_seconds": step_seconds,
                "forward_cuda_seconds": forward_cuda_seconds,
                "backward_cuda_seconds": backward_cuda_seconds,
                "optimizer_cuda_seconds": optimizer_cuda_seconds,
                "host_overhead_in_step_seconds": max(
                    step_seconds
                    - forward_cuda_seconds
                    - backward_cuda_seconds
                    - optimizer_cuda_seconds,
                    0.0,
                ),
                "examples_per_second_step": args.batch_size / step_seconds,
                "examples_per_second_end_to_end": (
                    segment_updates * args.batch_size / elapsed
                ),
                "checkpoint_write_seconds": checkpoint_write_seconds,
                "recovery_checkpoint_path": recovery_checkpoint_path,
                "gpu_memory_allocated_bytes": torch.cuda.memory_allocated(),
                "gpu_memory_reserved_bytes": torch.cuda.memory_reserved(),
                "gpu_peak_memory_allocated_bytes": (torch.cuda.max_memory_allocated()),
                "gpu_peak_memory_reserved_bytes": (torch.cuda.max_memory_reserved()),
                **_json_scalars(aux),
                **optimizer_metrics,
            }
            if lr_range_position is not None:
                record["lr_range_position"] = lr_range_position
            records.append(record)
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
            if update == 1 or update % args.log_every == 0:
                print(json.dumps(record, sort_keys=True), flush=True)
            del (
                prepared,
                compact_batch,
                batch,
                choices_cpu,
                choices,
                loss,
                aux,
            )
            if bool(optimizer_metrics["optimizer_skipped_nonfinite"]):
                raise FloatingPointError(f"Non-finite update at {update}")
            run_live_milestones(
                update,
                restart_monitor=(
                    args.steps == 0 or update < args.steps
                ),
            )
    finally:
        if prepared_future is not None:
            prepared_future.cancel()
        if prefetch_executor is not None:
            prefetch_executor.shutdown(wait=True, cancel_futures=True)
        _stop_gpu_monitor(monitor)

    torch.cuda.synchronize()
    train_wall_seconds = time.perf_counter() - run_started
    train_seconds = train_wall_seconds - milestone_evaluation_seconds
    checkpoint_manifest = None
    if args.save_final:
        checkpoint_manifest = save_model_checkpoint(
            output_dir=output_dir,
            model=model,
            source_mapping_sha256=source_mapping["combined_sha256"],
            optimizer_update=optimizer.update,
            data_cursor=data_cursor,
        )
    loss_summary = _summarize_training_records(records)
    _write_json(output_dir / "loss_summary.json", loss_summary)
    completed_segment_updates = update - segment_start_update
    completed_segment_examples = completed_segment_updates * args.batch_size
    report = {
        "schema_version": "torch-eager-train-report-v1",
        "completed_utc": datetime.now(UTC).isoformat(),
        "updates": update,
        "segment_start_update": segment_start_update,
        "segment_updates": completed_segment_updates,
        "examples": update * args.batch_size,
        "segment_examples": completed_segment_examples,
        "next_data_cursor": data_cursor,
        "train_seconds": train_seconds,
        "train_wall_seconds": train_wall_seconds,
        "milestone_evaluation_seconds": milestone_evaluation_seconds,
        "examples_per_second_end_to_end": (
            completed_segment_examples / max(train_seconds, 1e-12)
        ),
        "checkpoint_save_seconds": checkpoint_save_seconds,
        "recovery_checkpoints": saved_recovery_checkpoints,
        "resumed_from": (
            str(_require_workspace(args.resume_checkpoint, exists=True))
            if args.resume_checkpoint is not None
            else None
        ),
        "mean_data_seconds": float(np.mean([row["data_seconds"] for row in records])),
        "mean_data_wait_seconds": float(np.mean([row["data_wait_seconds"] for row in records])),
        "mean_data_prepare_seconds": float(
            np.mean([row["data_prepare_seconds"] for row in records])
        ),
        "mean_data_prefetch_hidden_seconds": float(
            np.mean([row["data_prefetch_hidden_seconds"] for row in records])
        ),
        "mean_transfer_seconds": float(np.mean([row["transfer_seconds"] for row in records])),
        "mean_step_seconds": float(np.mean([row["step_seconds"] for row in records])),
        "mean_forward_cuda_seconds": float(
            np.mean([row["forward_cuda_seconds"] for row in records])
        ),
        "mean_backward_cuda_seconds": float(
            np.mean([row["backward_cuda_seconds"] for row in records])
        ),
        "mean_optimizer_cuda_seconds": float(
            np.mean([row["optimizer_cuda_seconds"] for row in records])
        ),
        "mean_host_overhead_in_step_seconds": float(
            np.mean([row["host_overhead_in_step_seconds"] for row in records])
        ),
        "gpu_monitor": _summarize_gpu_samples(output_dir / "gpu_samples.csv"),
        "profiler": profiler_summary,
        "compile_counters": _compile_counter_snapshot(),
        "gpu_peak_memory_allocated_bytes": torch.cuda.max_memory_allocated(),
        "gpu_peak_memory_reserved_bytes": torch.cuda.max_memory_reserved(),
        "last_metrics": records[-1],
        "loss_summary": loss_summary,
        "checkpoint": checkpoint_manifest,
        "lr_range_enabled": lr_range_enabled,
        "hero_milestones": {
            "contract": hero_milestone_contract,
            "validation_records": hero_validation_records,
            "arena_records": hero_arena_records,
        },
    }
    _write_json(output_dir / "report.json", report)
    print(json.dumps(report, sort_keys=True), flush=True)
    return 0


def _analyze_lr_range_records(
    records: Sequence[Mapping[str, Any]],
    *,
    smoothing_beta: float = 0.98,
    regression_radius: int = 4,
) -> dict[str, Any]:
    if len(records) < 16:
        raise ValueError("LR-range analysis requires at least 16 updates")
    if not 0.0 <= smoothing_beta < 1.0:
        raise ValueError("smoothing_beta must be in [0, 1)")
    if regression_radius < 2 or 2 * regression_radius + 1 >= len(records):
        raise ValueError("Invalid LR-range regression radius")

    learning_rates = np.asarray(
        [float(record["learning_rate"]) for record in records],
        dtype=np.float64,
    )
    losses = np.asarray(
        [float(record["loss"]) for record in records],
        dtype=np.float64,
    )
    if (
        not np.all(np.isfinite(learning_rates))
        or not np.all(learning_rates > 0.0)
        or not np.all(np.diff(learning_rates) > 0.0)
    ):
        raise ValueError("LR-range rates must be finite, positive, and increasing")
    if not np.all(np.isfinite(losses)):
        raise ValueError("LR-range losses must be finite")

    smoothed = np.empty_like(losses)
    moving = 0.0
    for index, value in enumerate(losses):
        moving = smoothing_beta * moving + (1.0 - smoothing_beta) * value
        correction = 1.0 - smoothing_beta ** (index + 1)
        smoothed[index] = moving / correction

    log_learning_rates = np.log(learning_rates)
    slopes = np.full_like(losses, np.nan)
    for index in range(regression_radius, len(records) - regression_radius):
        region = slice(index - regression_radius, index + regression_radius + 1)
        centered_x = log_learning_rates[region] - log_learning_rates[index]
        slopes[index] = float(
            np.dot(centered_x, smoothed[region])
            / np.dot(centered_x, centered_x)
        )

    eligible_start = max(regression_radius, 10, len(records) // 10)
    eligible_stop = len(records) - regression_radius
    eligible = np.arange(eligible_start, eligible_stop)
    minimum_index = int(eligible[np.argmin(smoothed[eligible])])
    steepest_index = int(eligible[np.nanargmin(slopes[eligible])])

    best_seen = math.inf
    divergence_index: int | None = None
    for index in range(eligible_start, len(records)):
        best_seen = min(best_seen, float(smoothed[index]))
        threshold = max(1.5 * best_seen, best_seen + 2.0)
        if index > minimum_index and smoothed[index] > threshold:
            divergence_index = index
            break

    curve_metrics = (
        "loss",
        "dfm_ce_loss",
        "root_legal_conditional_ce",
        "weighted_root_legal_conditional_ce",
        "weighted_legality_loss",
        "jepa_raw_mse",
        "jepa_sigreg_loss",
        "jepa_pred_sigreg_loss",
        "wdl_weighted_loss",
        "gradient_global_norm",
        "gradient_clip_scale",
        "z_state_norm",
        "z_target_norm",
        "z_pred_norm",
        "accuracy",
    )
    curve: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        row: dict[str, Any] = {
            "update": int(record["update"]),
            "learning_rate": float(learning_rates[index]),
            "bt4_learning_rate": float(record["bt4_learning_rate"]),
            "smoothed_loss": float(smoothed[index]),
            "local_loss_slope_per_log_lr": (
                float(slopes[index]) if math.isfinite(slopes[index]) else None
            ),
        }
        for name in curve_metrics:
            if name in record:
                row[name] = float(record[name])
        curve.append(row)

    def candidate(index: int) -> dict[str, Any]:
        return {
            "update": int(records[index]["update"]),
            "learning_rate": float(learning_rates[index]),
            "bt4_learning_rate": float(records[index]["bt4_learning_rate"]),
            "loss": float(losses[index]),
            "smoothed_loss": float(smoothed[index]),
            "local_loss_slope_per_log_lr": (
                float(slopes[index]) if math.isfinite(slopes[index]) else None
            ),
        }

    conservative_from_minimum = max(
        float(learning_rates[0]),
        float(learning_rates[minimum_index]) / 10.0,
    )
    return {
        "schema_version": "torch-hero-lr-range-analysis-v1",
        "smoothing_beta": smoothing_beta,
        "regression_radius": regression_radius,
        "eligible_update_indices_zero_based": [
            eligible_start,
            eligible_stop - 1,
        ],
        "candidates": {
            "steepest_smoothed_descent": candidate(steepest_index),
            "minimum_smoothed_loss": candidate(minimum_index),
            "one_decade_below_minimum_loss": {
                "learning_rate": conservative_from_minimum,
                "bt4_learning_rate": (
                    conservative_from_minimum
                    * HERO_CONFIG.bt4_learning_rate
                    / HERO_CONFIG.learning_rate
                ),
                "heuristic_only": True,
            },
        },
        "divergence": (
            None if divergence_index is None else candidate(divergence_index)
        ),
        "curve": curve,
    }


def lr_range(args: argparse.Namespace) -> int:
    """Run and discard a clean hero model while exponentially sweeping LR."""

    result = train(args)
    output_dir = _require_workspace(args.output_dir, exists=True)
    metrics_path = _require_workspace(output_dir / "metrics.jsonl", exists=True)
    records = [
        json.loads(line)
        for line in metrics_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    analysis = _analyze_lr_range_records(records)
    run_config_path = _require_workspace(output_dir / "run_config.json", exists=True)
    train_report_path = _require_workspace(output_dir / "report.json", exists=True)
    run_config = json.loads(run_config_path.read_text(encoding="utf-8"))
    train_report = json.loads(train_report_path.read_text(encoding="utf-8"))
    gate_checks = {
        "clean_hero_initialization": run_config["recipe"] == "hero",
        "zero_weight_decay": run_config["config"]["weight_decay"] == 0.0,
        "no_checkpoint": train_report["checkpoint"] is None,
        "all_updates_completed": train_report["updates"] == args.steps,
        "all_optimizer_updates_finite": all(
            not bool(record["optimizer_skipped_nonfinite"]) for record in records
        ),
        "range_endpoints_reproduced": (
            math.isclose(
                float(records[0]["learning_rate"]),
                float(args.lr_range_start),
                rel_tol=1e-12,
            )
            and math.isclose(
                float(records[-1]["learning_rate"]),
                float(args.lr_range_end),
                rel_tol=1e-12,
            )
        ),
    }
    report = {
        "schema_version": "torch-hero-lr-range-report-v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "git_commit": _git_commit(),
        "model_state_retained": False,
        "metrics_path": metrics_path.name,
        "metrics_sha256": _sha256_file(metrics_path),
        "run_config_sha256": _sha256_file(run_config_path),
        "train_report_sha256": _sha256_file(train_report_path),
        "analysis": analysis,
        "gate_checks": gate_checks,
        "gate_pass": all(gate_checks.values()),
    }
    _write_json(output_dir / "lr_range_report.json", report)
    print(
        json.dumps(
            {
                "lr_range_report": str(output_dir / "lr_range_report.json"),
                "candidates": analysis["candidates"],
                "divergence": analysis["divergence"],
                "gate_pass": report["gate_pass"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    if not report["gate_pass"]:
        raise RuntimeError("Hero LR-range gate failed")
    return result


def _gradient_audit_group(name: str) -> str:
    if name.startswith("encoder."):
        return "raw_bt4"
    if name.startswith("state_projector."):
        return "state_projector"
    if name.startswith("dfm_state_projector."):
        return "dfm_state_projector"
    if name.startswith("jepa_"):
        return "jepa"
    if name.startswith("value_wdl_head."):
        return "value_wdl"
    return "dfm_policy"


def _legal_root_actions(
    root_logits: Tensor,
    compact_batch: Mapping[str, Any],
) -> tuple[Tensor, Tensor]:
    legal_idx = torch.from_numpy(
        np.ascontiguousarray(np.asarray(compact_batch["legal_idx"])[:, 0])
    ).long()
    legal_count = torch.from_numpy(
        np.ascontiguousarray(np.asarray(compact_batch["legal_count"])[:, 0])
    ).long()
    safe = legal_idx.clamp(0, root_logits.shape[-1] - 1)
    legal_logits = torch.gather(root_logits.float(), 1, safe)
    slots = torch.arange(safe.shape[1]).unsqueeze(0)
    legal_logits = torch.where(
        slots < legal_count.unsqueeze(1),
        legal_logits,
        torch.full_like(legal_logits, -torch.inf),
    )
    selected_slot = legal_logits.argmax(dim=1, keepdim=True)
    return torch.gather(safe, 1, selected_slot).squeeze(1), legal_count > 0


def runtime_parity(args: argparse.Namespace) -> int:
    """Compare a diagnostic compile boundary with the accepted compiled reference."""

    if not torch.cuda.is_available():
        raise RuntimeError("runtime-parity requires CUDA")
    if args.batch_size < HERO_CONFIG.sigreg_example_count:
        raise ValueError(
            f"--batch-size must be at least {HERO_CONFIG.sigreg_example_count}"
        )
    if args.threads not in (1, 2):
        raise ValueError("--threads must be 1 or 2 under the resource guard")
    output = _require_workspace(args.output)
    if output.exists():
        raise FileExistsError(f"Runtime parity output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda")
    torch.set_num_threads(args.threads)
    torch.set_float32_matmul_precision("high")
    config = dataclasses.replace(
        HERO_CONFIG,
        remat_bt4_blocks=True,
        remat_projector_blocks=True,
        remat_dfm_blocks=False,
        use_bt4_sdpa=True,
        use_head_sdpa=True,
    )
    batches = CanonicalTrajectoryBatches(
        _require_workspace(args.data_root) / "train",
        batch_size=args.batch_size,
        horizon=config.horizon,
        seed=args.seed,
        shuffle_files=True,
        batch_schedule="global_permutation",
    )
    prepared = _prepare_training_step(
        batches,
        seed=args.seed,
        update=0,
        data_cursor=args.data_step,
        batch_size=args.batch_size,
        config=config,
    )

    def run_backward(
        compile_regions: str,
        bt4_norm_impl: str,
    ) -> tuple[
        JointModel,
        dict[str, Any],
        dict[str, float],
        Tensor,
        float,
        int,
        int,
    ]:
        model, mapping = load_raw_bt4_hero_model(
            device=device,
            raw_bt4_path=args.raw_bt4_path,
            config=config,
            bt4_norm_impl=bt4_norm_impl,
        )
        model.train()
        compiled = _apply_compile_regions(model, compile_regions)
        batch = _torch_batch(prepared.compact_batch, device)
        choices = _choices_to_device(prepared.choices, device)
        capture: dict[str, Any] = {}
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        loss, aux = loss_and_aux(
            model,
            batch,
            choices,
            compute_dtype=torch.bfloat16,
            capture=capture,
        )
        loss.backward()
        torch.cuda.synchronize()
        seconds = time.perf_counter() - started
        metrics = {"loss": float(loss.detach().float().cpu()), **_json_scalars(aux)}
        root_logits = capture["root_logits"].float().cpu()
        peak_allocated = torch.cuda.max_memory_allocated()
        peak_reserved = torch.cuda.max_memory_reserved()
        del batch, choices, loss, aux, capture
        return (
            model,
            {"mapping": mapping, "compiled_regions": compiled},
            metrics,
            root_logits,
            seconds,
            peak_allocated,
            peak_reserved,
        )

    (
        reference_model,
        reference_lineage,
        reference_metrics,
        reference_logits,
        reference_seconds,
        reference_peak_allocated,
        reference_peak_reserved,
    ) = run_backward("fresh", "eager")
    reference_gradients: dict[str, Tensor] = {}
    reference_missing: set[str] = set()
    for name, parameter in reference_model.named_parameters():
        if parameter.grad is None:
            reference_missing.add(name)
        else:
            reference_gradients[name] = parameter.grad.detach().float().cpu().clone()
    del reference_model
    gc.collect()
    torch.cuda.empty_cache()

    (
        candidate_model,
        candidate_lineage,
        candidate_metrics,
        candidate_logits,
        candidate_seconds,
        candidate_peak_allocated,
        candidate_peak_reserved,
    ) = run_backward(
        args.candidate_compile_regions,
        args.candidate_bt4_norm_impl,
    )

    accumulators: dict[str, dict[str, float | int | bool]] = {
        "all": {
            "dot": 0.0,
            "reference_squared_norm": 0.0,
            "candidate_squared_norm": 0.0,
            "difference_squared_norm": 0.0,
            "element_count": 0,
            "parameter_count": 0,
            "max_absolute_difference": 0.0,
            "finite": True,
        }
    }
    parameter_records: list[dict[str, Any]] = []
    candidate_missing: set[str] = set()
    for name, parameter in candidate_model.named_parameters():
        if parameter.grad is None:
            candidate_missing.add(name)
            continue
        if name not in reference_gradients:
            raise KeyError(f"Candidate-only gradient: {name}")
        reference = reference_gradients.pop(name)
        candidate = parameter.grad.detach().float().cpu()
        difference = candidate - reference
        reference_flat = reference.reshape(-1)
        candidate_flat = candidate.reshape(-1)
        difference_flat = difference.reshape(-1)
        dot = float(torch.dot(reference_flat, candidate_flat))
        reference_squared_norm = float(torch.dot(reference_flat, reference_flat))
        candidate_squared_norm = float(torch.dot(candidate_flat, candidate_flat))
        difference_squared_norm = float(torch.dot(difference_flat, difference_flat))
        finite = bool(
            torch.isfinite(reference).all()
            and torch.isfinite(candidate).all()
            and torch.isfinite(difference).all()
        )
        max_absolute_difference = float(difference.abs().max()) if difference.numel() else 0.0
        group = _gradient_audit_group(name)
        if group not in accumulators:
            accumulators[group] = {
                "dot": 0.0,
                "reference_squared_norm": 0.0,
                "candidate_squared_norm": 0.0,
                "difference_squared_norm": 0.0,
                "element_count": 0,
                "parameter_count": 0,
                "max_absolute_difference": 0.0,
                "finite": True,
            }
        for key in ("all", group):
            accumulator = accumulators[key]
            accumulator["dot"] = float(accumulator["dot"]) + dot
            accumulator["reference_squared_norm"] = (
                float(accumulator["reference_squared_norm"]) + reference_squared_norm
            )
            accumulator["candidate_squared_norm"] = (
                float(accumulator["candidate_squared_norm"]) + candidate_squared_norm
            )
            accumulator["difference_squared_norm"] = (
                float(accumulator["difference_squared_norm"]) + difference_squared_norm
            )
            accumulator["element_count"] = int(accumulator["element_count"]) + reference.numel()
            accumulator["parameter_count"] = int(accumulator["parameter_count"]) + 1
            accumulator["max_absolute_difference"] = max(
                float(accumulator["max_absolute_difference"]),
                max_absolute_difference,
            )
            accumulator["finite"] = bool(accumulator["finite"]) and finite
        reference_norm = math.sqrt(max(reference_squared_norm, 0.0))
        candidate_norm = math.sqrt(max(candidate_squared_norm, 0.0))
        parameter_records.append(
            {
                "name": name,
                "group": group,
                "element_count": reference.numel(),
                "reference_norm": reference_norm,
                "candidate_norm": candidate_norm,
                "norm_ratio": candidate_norm / max(reference_norm, 1e-30),
                "cosine": dot / max(reference_norm * candidate_norm, 1e-30),
                "relative_l2": (
                    math.sqrt(max(difference_squared_norm, 0.0))
                    / max(reference_norm, 1e-30)
                ),
                "max_absolute_difference": max_absolute_difference,
                "finite": finite,
            }
        )
        del reference, candidate, difference
    if reference_gradients:
        raise KeyError(f"Reference-only gradients: {sorted(reference_gradients)[:10]}")

    gradient_groups: dict[str, dict[str, Any]] = {}
    for name, accumulator in accumulators.items():
        reference_norm = math.sqrt(
            max(float(accumulator["reference_squared_norm"]), 0.0)
        )
        candidate_norm = math.sqrt(
            max(float(accumulator["candidate_squared_norm"]), 0.0)
        )
        difference_norm = math.sqrt(
            max(float(accumulator["difference_squared_norm"]), 0.0)
        )
        gradient_groups[name] = {
            "element_count": int(accumulator["element_count"]),
            "parameter_count": int(accumulator["parameter_count"]),
            "reference_norm": reference_norm,
            "candidate_norm": candidate_norm,
            "norm_ratio": candidate_norm / max(reference_norm, 1e-30),
            "cosine": (
                float(accumulator["dot"])
                / max(reference_norm * candidate_norm, 1e-30)
            ),
            "relative_l2": difference_norm / max(reference_norm, 1e-30),
            "max_absolute_difference": float(
                accumulator["max_absolute_difference"]
            ),
            "finite": bool(accumulator["finite"]),
        }

    reference_actions, legal_valid = _legal_root_actions(
        reference_logits,
        prepared.compact_batch,
    )
    candidate_actions, candidate_legal_valid = _legal_root_actions(
        candidate_logits,
        prepared.compact_batch,
    )
    if not torch.equal(legal_valid, candidate_legal_valid):
        raise RuntimeError("Legal-valid masks drifted across runtime variants")
    reference_flat = reference_logits.reshape(-1)
    candidate_flat = candidate_logits.reshape(-1)
    logit_difference = candidate_flat - reference_flat
    reference_logit_norm = float(torch.linalg.vector_norm(reference_flat))
    candidate_logit_norm = float(torch.linalg.vector_norm(candidate_flat))
    logit_cosine = float(
        torch.dot(reference_flat, candidate_flat)
        / max(reference_logit_norm * candidate_logit_norm, 1e-30)
    )
    valid_count = int(legal_valid.sum())
    legal_action_agreement = float(
        ((reference_actions == candidate_actions) & legal_valid).sum()
        / max(valid_count, 1)
    )
    loss_relative_difference = abs(
        candidate_metrics["loss"] - reference_metrics["loss"]
    ) / max(abs(reference_metrics["loss"]), 1e-30)
    nontrivial_groups = [
        value
        for key, value in gradient_groups.items()
        if key != "all" and value["reference_norm"] > 1e-8
    ]
    gate_checks = {
        "source_mapping_equal": (
            reference_lineage["mapping"]["combined_sha256"]
            == candidate_lineage["mapping"]["combined_sha256"]
        ),
        "missing_gradients_equal": reference_missing == candidate_missing,
        "finite_gradients": all(value["finite"] for value in gradient_groups.values()),
        "loss_relative_difference_le_0p005": loss_relative_difference <= 0.005,
        "root_logits_cosine_ge_0p999": logit_cosine >= 0.999,
        "legal_action_agreement_ge_0p99": legal_action_agreement >= 0.99,
        "global_gradient_cosine_ge_0p99": gradient_groups["all"]["cosine"] >= 0.99,
        "global_gradient_norm_ratio_in_0p8_1p25": (
            0.8 <= gradient_groups["all"]["norm_ratio"] <= 1.25
        ),
        "every_nontrivial_group_gradient_cosine_ge_0p95": all(
            value["cosine"] >= 0.95 for value in nontrivial_groups
        ),
    }
    report = {
        "schema_version": "torch-hero-runtime-parity-v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "git_commit": _git_commit(),
        "device": torch.cuda.get_device_name(device),
        "batch_size": args.batch_size,
        "data_step": args.data_step,
        "seed": args.seed,
        "config": dataclasses.asdict(config),
        "reference": {
            "bt4_norm_impl": "eager",
            "compile_regions": reference_lineage["compiled_regions"],
            "metrics": reference_metrics,
            "forward_backward_seconds_including_compile": reference_seconds,
            "peak_allocated_bytes": reference_peak_allocated,
            "peak_reserved_bytes": reference_peak_reserved,
        },
        "candidate": {
            "requested_compile_regions": args.candidate_compile_regions,
            "bt4_norm_impl": args.candidate_bt4_norm_impl,
            "compile_regions": candidate_lineage["compiled_regions"],
            "metrics": candidate_metrics,
            "forward_backward_seconds_including_compile": candidate_seconds,
            "peak_allocated_bytes": candidate_peak_allocated,
            "peak_reserved_bytes": candidate_peak_reserved,
        },
        "loss_relative_difference": loss_relative_difference,
        "root_logits": {
            "reference_norm": reference_logit_norm,
            "candidate_norm": candidate_logit_norm,
            "cosine": logit_cosine,
            "relative_l2": (
                float(torch.linalg.vector_norm(logit_difference))
                / max(reference_logit_norm, 1e-30)
            ),
            "max_absolute_difference": float(logit_difference.abs().max()),
            "legal_valid_count": valid_count,
            "legal_top1_action_agreement": legal_action_agreement,
        },
        "gradient_groups": gradient_groups,
        "worst_parameters_by_relative_l2": sorted(
            parameter_records,
            key=lambda row: row["relative_l2"],
            reverse=True,
        )[:25],
        "reference_missing_gradients": sorted(reference_missing),
        "candidate_missing_gradients": sorted(candidate_missing),
        "compile_counters": _compile_counter_snapshot(),
        "gate_checks": gate_checks,
        "gate_pass": all(gate_checks.values()),
    }
    _write_json(output, report)
    print(json.dumps(report, sort_keys=True), flush=True)
    if not report["gate_pass"]:
        raise RuntimeError("Diagnostic runtime parity gate failed")
    return 0


_LOSS_AUDIT_GROUPS = (
    "raw_bt4",
    "state_projector",
    "dfm_state_projector",
    "jepa",
    "value_wdl",
    "dfm_policy",
)


def _gradient_norms_by_group(model: nn.Module) -> dict[str, dict[str, Any]]:
    device = next(model.parameters()).device
    squared_norms = {
        group: torch.zeros((), device=device, dtype=torch.float32)
        for group in _LOSS_AUDIT_GROUPS
    }
    parameter_counts = {group: 0 for group in _LOSS_AUDIT_GROUPS}
    element_counts = {group: 0 for group in _LOSS_AUDIT_GROUPS}
    finite = {
        group: torch.ones((), device=device, dtype=torch.bool)
        for group in _LOSS_AUDIT_GROUPS
    }
    for name, parameter in model.named_parameters():
        gradient = parameter.grad
        if gradient is None:
            continue
        group = _gradient_audit_group(name)
        detached = gradient.detach().float()
        squared_norms[group] = squared_norms[group] + detached.square().sum()
        parameter_counts[group] += 1
        element_counts[group] += detached.numel()
        finite[group] = finite[group] & torch.isfinite(detached).all()

    total_squared = torch.stack(tuple(squared_norms.values())).sum()
    result = {
        group: {
            "norm": math.sqrt(max(float(value.cpu()), 0.0)),
            "squared_norm": max(float(value.cpu()), 0.0),
            "parameter_count": parameter_counts[group],
            "element_count": element_counts[group],
            "finite": bool(finite[group].cpu()),
        }
        for group, value in squared_norms.items()
    }
    result["all"] = {
        "norm": math.sqrt(max(float(total_squared.cpu()), 0.0)),
        "squared_norm": max(float(total_squared.cpu()), 0.0),
        "parameter_count": sum(parameter_counts.values()),
        "element_count": sum(element_counts.values()),
        "finite": all(bool(value.cpu()) for value in finite.values()),
    }
    return result


def _polarized_gradient_cosines(
    left: Mapping[str, Mapping[str, Any]],
    right: Mapping[str, Mapping[str, Any]],
    combined: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, float | None]]:
    result: dict[str, dict[str, float | None]] = {}
    for group in (*_LOSS_AUDIT_GROUPS, "all"):
        left_squared = float(left[group]["squared_norm"])
        right_squared = float(right[group]["squared_norm"])
        combined_squared = float(combined[group]["squared_norm"])
        dot = 0.5 * (combined_squared - left_squared - right_squared)
        denominator = math.sqrt(max(left_squared * right_squared, 0.0))
        result[group] = {
            "dot": dot,
            "cosine": (
                max(-1.0, min(1.0, dot / denominator))
                if denominator > 1e-30
                else None
            ),
        }
    return result


def _shared_sigreg_gradient_projection(
    records: Mapping[str, Mapping[str, Any]],
    interactions: Mapping[str, Mapping[str, Mapping[str, float | None]]],
    *,
    baseline_coefficient: float,
    candidate_coefficients: Sequence[float],
) -> dict[str, Any]:
    """Project gradient balance when both SIGReg coefficients change together."""

    if not math.isfinite(baseline_coefficient) or baseline_coefficient <= 0.0:
        raise ValueError("baseline_coefficient must be finite and positive")
    result: dict[str, Any] = {
        "baseline_coefficient": baseline_coefficient,
        "candidates": {},
    }
    for coefficient in candidate_coefficients:
        if not math.isfinite(coefficient) or coefficient < 0.0:
            raise ValueError("candidate coefficients must be finite and non-negative")
        scale = coefficient / baseline_coefficient
        groups: dict[str, Any] = {}
        for group in (*_LOSS_AUDIT_GROUPS, "all"):
            policy_squared = float(
                records["policy_total"]["gradient_groups"][group]["squared_norm"]
            )
            non_sigreg_squared = float(
                records["non_sigreg_representation"]["gradient_groups"][group][
                    "squared_norm"
                ]
            )
            sigreg_squared = float(
                records["sigreg_total"]["gradient_groups"][group]["squared_norm"]
            )
            policy_dot_representation = float(
                interactions["policy_vs_representation"][group]["dot"]
            )
            policy_dot_sigreg = float(
                interactions["policy_vs_sigreg"][group]["dot"]
            )
            non_sigreg_dot_sigreg = float(
                interactions["non_sigreg_vs_sigreg"][group]["dot"]
            )
            policy_dot_non_sigreg = (
                policy_dot_representation - policy_dot_sigreg
            )
            representation_squared = (
                non_sigreg_squared
                + scale * scale * sigreg_squared
                + 2.0 * scale * non_sigreg_dot_sigreg
            )
            policy_dot_scaled_representation = (
                policy_dot_non_sigreg + scale * policy_dot_sigreg
            )
            total_squared = (
                policy_squared
                + representation_squared
                + 2.0 * policy_dot_scaled_representation
            )
            cosine_denominator = math.sqrt(
                max(policy_squared * representation_squared, 0.0)
            )
            groups[group] = {
                "policy_norm": math.sqrt(max(policy_squared, 0.0)),
                "non_sigreg_representation_norm": math.sqrt(
                    max(non_sigreg_squared, 0.0)
                ),
                "scaled_sigreg_norm": abs(scale)
                * math.sqrt(max(sigreg_squared, 0.0)),
                "representation_norm": math.sqrt(
                    max(representation_squared, 0.0)
                ),
                "total_norm": math.sqrt(max(total_squared, 0.0)),
                "policy_vs_representation_cosine": (
                    max(
                        -1.0,
                        min(
                            1.0,
                            policy_dot_scaled_representation
                            / cosine_denominator,
                        ),
                    )
                    if cosine_denominator > 1e-30
                    else None
                ),
            }
        result["candidates"][format(coefficient, ".12g")] = {
            "coefficient": coefficient,
            "relative_to_baseline": scale,
            "gradient_groups": groups,
        }
    return result


def loss_gradient_audit(args: argparse.Namespace) -> int:
    """Calibrate root CE and measure no-update component gradients."""

    if not torch.cuda.is_available():
        raise RuntimeError("loss-audit requires CUDA")
    if args.batch_size < HERO_CONFIG.sigreg_example_count:
        raise ValueError(
            f"--batch-size must be at least {HERO_CONFIG.sigreg_example_count}"
        )
    if args.threads not in (1, 2):
        raise ValueError("--threads must be 1 or 2 under the resource guard")
    if not math.isfinite(args.root_target_contribution) or (
        args.root_target_contribution <= 0.0
    ):
        raise ValueError("--root-target-contribution must be finite and positive")
    output = _require_workspace(args.output)
    if output.exists():
        raise FileExistsError(f"Loss-audit output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda")
    torch.set_num_threads(args.threads)
    torch.set_float32_matmul_precision("high")
    config = dataclasses.replace(
        HERO_CONFIG,
        remat_bt4_blocks=True,
        remat_projector_blocks=True,
        remat_dfm_blocks=False,
        use_bt4_sdpa=True,
        use_head_sdpa=True,
        root_legal_ce_coeff=0.0,
    )
    batches = CanonicalTrajectoryBatches(
        _require_workspace(args.data_root) / "train",
        batch_size=args.batch_size,
        horizon=config.horizon,
        seed=args.seed,
        shuffle_files=True,
        batch_schedule="global_permutation",
    )
    prepared_started = time.perf_counter()
    prepared = _prepare_training_step(
        batches,
        seed=args.seed,
        update=0,
        data_cursor=args.data_step,
        batch_size=args.batch_size,
        config=config,
    )
    data_prepare_seconds = time.perf_counter() - prepared_started
    model, mapping = load_raw_bt4_hero_model(
        device=device,
        raw_bt4_path=args.raw_bt4_path,
        config=config,
    )
    model.train()
    batch = _torch_batch(prepared.compact_batch, device)
    choices = _choices_to_device(prepared.choices, device)

    calibration_source: dict[str, Any]
    scalar_components: dict[str, float] | None = None
    if args.root_calibration_metrics is not None:
        calibration_batch_size = args.root_calibration_batch_size
        metrics_path = _require_workspace(
            args.root_calibration_metrics,
            exists=True,
        )
        calibration_rows = [
            json.loads(line)
            for line in metrics_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        matches = [
            row
            for row in calibration_rows
            if int(row.get("update", -1)) == 1
        ]
        if len(matches) != 1:
            raise ValueError(
                f"Expected exactly one update-1 calibration row in {metrics_path}"
            )
        calibration_row = matches[0]
        run_config_path = metrics_path.with_name("run_config.json")
        run_config = json.loads(
            _require_workspace(run_config_path, exists=True).read_text(
                encoding="utf-8"
            )
        )
        run_args = run_config["args"]
        expected_lineage = {
            "recipe": "hero",
            "batch_size": args.root_calibration_batch_size,
            "seed": args.seed,
            "data_start": args.data_step,
        }
        observed_lineage = {
            key: run_args[key] for key in expected_lineage
        }
        if observed_lineage != expected_lineage:
            raise ValueError(
                "Root-calibration run lineage does not match this audit: "
                f"{observed_lineage} != {expected_lineage}"
            )
        root_value = float(calibration_row["root_legal_conditional_ce"])
        calibration_source = {
            "kind": "deterministic_update_zero_metrics",
            "metrics_path": str(metrics_path.relative_to(_REPO_ROOT)),
            "metrics_sha256": _sha256_file(metrics_path),
            "run_config_path": str(run_config_path.relative_to(_REPO_ROOT)),
            "run_config_sha256": _sha256_file(run_config_path),
            "update": 1,
            "lineage": observed_lineage,
        }
    else:
        calibration_batch_size = args.batch_size
        initial_capture: dict[str, Any] = {}
        with torch.no_grad():
            initial_loss, initial_aux = loss_and_aux(
                model,
                batch,
                choices,
                compute_dtype=torch.bfloat16,
                capture=initial_capture,
            )
        initial_components = initial_capture["loss_components"]
        scalar_components = {
            name: float(value.detach().float().cpu())
            for name, value in initial_components.items()
        }
        root_value = scalar_components["root_legal_conditional_ce"]
        calibration_source = {"kind": "inline_no_grad_forward"}
        del initial_loss, initial_aux, initial_capture, initial_components
        gc.collect()
        torch.cuda.empty_cache()
    if not math.isfinite(root_value) or root_value <= 0.0:
        raise RuntimeError(f"Cannot calibrate root CE from {root_value!r}")
    root_coefficient = args.root_target_contribution / root_value
    calibrated_config = dataclasses.replace(
        config,
        root_legal_ce_coeff=root_coefficient,
    )
    model.config = calibrated_config
    compiled_regions = _apply_compile_regions(model, args.compile_regions)

    weights = {
        "dfm_ce": calibrated_config.dfm_ce_coeff,
        "root_legal_conditional_ce": calibrated_config.root_legal_ce_coeff,
        "root_illegal_mass": calibrated_config.legality_coeff,
        "jepa_raw_mse": calibrated_config.jepa_positive_coeff,
        "target_sigreg": calibrated_config.target_sigreg_coeff,
        "prediction_sigreg": calibrated_config.pred_sigreg_coeff,
        "wdl_ce": calibrated_config.wdl_coeff,
    }
    specifications: dict[str, tuple[str, ...]] = {
        name: (name,) for name in weights
    }
    specifications.update(
        {
            "sigreg_total": ("target_sigreg", "prediction_sigreg"),
            "non_sigreg_representation": ("jepa_raw_mse", "wdl_ce"),
            "policy_total": (
                "dfm_ce",
                "root_legal_conditional_ce",
                "root_illegal_mass",
            ),
            "policy_plus_sigreg": (
                "dfm_ce",
                "root_legal_conditional_ce",
                "root_illegal_mass",
                "target_sigreg",
                "prediction_sigreg",
            ),
            "representation_total": (
                "jepa_raw_mse",
                "target_sigreg",
                "prediction_sigreg",
                "wdl_ce",
            ),
            "total": tuple(weights),
        }
    )

    records: dict[str, dict[str, Any]] = {}
    max_scalar_replay_difference = 0.0
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    audit_started = time.perf_counter()
    for audit_name, component_names in specifications.items():
        model.zero_grad(set_to_none=True)
        capture: dict[str, Any] = {}
        torch.cuda.synchronize()
        started = time.perf_counter()
        _, aux = loss_and_aux(
            model,
            batch,
            choices,
            compute_dtype=torch.bfloat16,
            capture=capture,
        )
        components = capture["loss_components"]
        objective = sum(weights[name] * components[name] for name in component_names)
        objective.backward()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        replay_components = {
            name: float(value.detach().float().cpu())
            for name, value in components.items()
        }
        if scalar_components is None:
            scalar_components = replay_components
        else:
            max_scalar_replay_difference = max(
                max_scalar_replay_difference,
                max(
                    abs(replay_components[name] - scalar_components[name])
                    for name in scalar_components
                ),
            )
        records[audit_name] = {
            "components": list(component_names),
            "weighted_scalar": float(objective.detach().float().cpu()),
            "forward_backward_seconds": elapsed,
            "gradient_groups": _gradient_norms_by_group(model),
        }
        del aux, capture, components, objective
    torch.cuda.synchronize()
    audit_seconds = time.perf_counter() - audit_started
    assert scalar_components is not None
    calibration_replay_difference = abs(
        scalar_components["root_legal_conditional_ce"] - root_value
    )
    calibration_batch_matches_audit = (
        calibration_batch_size == args.batch_size
    )

    interactions = {
        "target_vs_prediction_sigreg": _polarized_gradient_cosines(
            records["target_sigreg"]["gradient_groups"],
            records["prediction_sigreg"]["gradient_groups"],
            records["sigreg_total"]["gradient_groups"],
        ),
        "policy_vs_representation": _polarized_gradient_cosines(
            records["policy_total"]["gradient_groups"],
            records["representation_total"]["gradient_groups"],
            records["total"]["gradient_groups"],
        ),
        "non_sigreg_vs_sigreg": _polarized_gradient_cosines(
            records["non_sigreg_representation"]["gradient_groups"],
            records["sigreg_total"]["gradient_groups"],
            records["representation_total"]["gradient_groups"],
        ),
        "policy_vs_sigreg": _polarized_gradient_cosines(
            records["policy_total"]["gradient_groups"],
            records["sigreg_total"]["gradient_groups"],
            records["policy_plus_sigreg"]["gradient_groups"],
        ),
    }
    shared_sigreg_projection = _shared_sigreg_gradient_projection(
        records,
        interactions,
        baseline_coefficient=calibrated_config.target_sigreg_coeff,
        candidate_coefficients=(0.0, 0.5, 1.0, 2.0, 4.0),
    )
    weighted_scalar_components = {
        name: weights[name] * value for name, value in scalar_components.items()
    }
    finite_gradients = all(
        group["finite"]
        for record in records.values()
        for group in record["gradient_groups"].values()
    )
    nonzero_atomic_gradients = all(
        records[name]["gradient_groups"]["all"]["norm"] > 1e-8
        for name in weights
    )
    gate_checks = {
        "finite_root_coefficient": (
            math.isfinite(root_coefficient) and root_coefficient > 0.0
        ),
        "root_weighted_scalar_matches_target": (
            abs(
                root_coefficient * root_value
                - args.root_target_contribution
            )
            <= 1e-6
        ),
        "equal_sigreg_coefficients": (
            calibrated_config.target_sigreg_coeff
            == calibrated_config.pred_sigreg_coeff
        ),
        "finite_component_gradients": finite_gradients,
        "nonzero_atomic_global_gradients": nonzero_atomic_gradients,
        "deterministic_scalar_replay": max_scalar_replay_difference <= 1e-6,
        "matched_batch_calibration_scalar_reproduced": (
            not calibration_batch_matches_audit
            or calibration_replay_difference <= 1e-6
        ),
    }
    report = {
        "schema_version": "torch-hero-loss-gradient-audit-v2",
        "created_utc": datetime.now(UTC).isoformat(),
        "git_commit": _git_commit(),
        "device": torch.cuda.get_device_name(device),
        "batch_size": args.batch_size,
        "sigreg_example_count": calibrated_config.sigreg_example_count,
        "data_step": args.data_step,
        "seed": args.seed,
        "data_prepare_seconds": data_prepare_seconds,
        "audit_seconds": audit_seconds,
        "compile_regions": compiled_regions,
        "config_before_calibration": dataclasses.asdict(config),
        "config_after_calibration": dataclasses.asdict(calibrated_config),
        "root_calibration": {
            "unweighted_scalar": root_value,
            "target_weighted_scalar": args.root_target_contribution,
            "coefficient": root_coefficient,
            "source": calibration_source,
            "replay_absolute_difference": calibration_replay_difference,
            "calibration_batch_size": calibration_batch_size,
            "audit_batch_size": args.batch_size,
            "calibration_batch_matches_audit": calibration_batch_matches_audit,
        },
        "scalar_components": scalar_components,
        "weights": weights,
        "weighted_scalar_components": weighted_scalar_components,
        "gradient_records": records,
        "interactions": interactions,
        "shared_sigreg_gradient_projection": shared_sigreg_projection,
        "max_scalar_replay_absolute_difference": max_scalar_replay_difference,
        "gpu_peak_memory_allocated_bytes": torch.cuda.max_memory_allocated(),
        "gpu_peak_memory_reserved_bytes": torch.cuda.max_memory_reserved(),
        "source_mapping_sha256": mapping["combined_sha256"],
        "data": batches.provenance(),
        "compile_counters": _compile_counter_snapshot(),
        "gate_checks": gate_checks,
        "gate_pass": all(gate_checks.values()),
    }
    _write_json(output, report)
    print(json.dumps(report, sort_keys=True), flush=True)
    if not report["gate_pass"]:
        raise RuntimeError("Hero loss-gradient audit gate failed")
    return 0


def _copy_parameter_gradients_to_cpu(
    model: nn.Module,
) -> tuple[dict[str, Tensor], int]:
    gradients: dict[str, Tensor] = {}
    total_bytes = 0
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        gradient = parameter.grad.detach().to(
            device="cpu",
            dtype=torch.float32,
            copy=True,
        )
        gradients[name] = gradient
        total_bytes += gradient.numel() * gradient.element_size()
    return gradients, total_bytes


def _gradient_cosines_against_cpu_reference(
    model: nn.Module,
    reference: Mapping[str, Tensor],
    *,
    reference_norms: Mapping[str, Mapping[str, Any]],
    candidate_norms: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    dot_by_group = {
        group: 0.0 for group in (*_LOSS_AUDIT_GROUPS, "all")
    }
    candidate_names = {
        name
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }
    reference_names = set(reference)
    for name, parameter in model.named_parameters():
        reference_gradient = reference.get(name)
        candidate_gradient = parameter.grad
        if reference_gradient is None or candidate_gradient is None:
            continue
        candidate_cpu = candidate_gradient.detach().to(
            device="cpu",
            dtype=torch.float32,
        )
        dot = float(
            torch.sum(
                reference_gradient * candidate_cpu,
                dtype=torch.float64,
            )
        )
        group = _gradient_audit_group(name)
        dot_by_group[group] += dot
        dot_by_group["all"] += dot

    groups: dict[str, Any] = {}
    for group in (*_LOSS_AUDIT_GROUPS, "all"):
        reference_norm = float(reference_norms[group]["norm"])
        candidate_norm = float(candidate_norms[group]["norm"])
        denominator = reference_norm * candidate_norm
        groups[group] = {
            "dot": dot_by_group[group],
            "reference_norm": reference_norm,
            "candidate_norm": candidate_norm,
            "cosine": (
                max(-1.0, min(1.0, dot_by_group[group] / denominator))
                if denominator > 1e-30
                else None
            ),
            "candidate_to_reference_norm_ratio": (
                candidate_norm / reference_norm
                if reference_norm > 1e-30
                else None
            ),
        }
    return {
        "groups": groups,
        "missing_from_candidate": sorted(reference_names - candidate_names),
        "missing_from_reference": sorted(candidate_names - reference_names),
    }


def sigreg_sample_audit(args: argparse.Namespace) -> int:
    """Audit normalized SIGReg sample counts without updating model state."""

    if not torch.cuda.is_available():
        raise RuntimeError("sigreg-sample-audit requires CUDA")
    counts = tuple(int(value) for value in args.sample_counts)
    if (
        not counts
        or counts != tuple(sorted(set(counts)))
        or counts[0] < 1
        or counts[-1] > args.batch_size
    ):
        raise ValueError(
            "--sample-counts must be unique, increasing, positive, and no "
            "larger than --batch-size"
        )
    if args.replicates < 2:
        raise ValueError("--replicates must be at least 2")
    if args.threads not in (1, 2):
        raise ValueError("--threads must be 1 or 2 under the resource guard")
    output = _require_workspace(args.output)
    if output.exists():
        raise FileExistsError(f"SIGReg sample audit output exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda")
    torch.set_num_threads(args.threads)
    torch.set_float32_matmul_precision("high")
    base_config = dataclasses.replace(
        HERO_CONFIG,
        remat_bt4_blocks=True,
        remat_projector_blocks=True,
        remat_dfm_blocks=False,
        use_bt4_sdpa=True,
        use_head_sdpa=True,
        sigreg_example_count=counts[0],
    )
    batches = CanonicalTrajectoryBatches(
        _require_workspace(args.data_root) / "train",
        batch_size=args.batch_size,
        horizon=base_config.horizon,
        seed=args.seed,
        shuffle_files=True,
        batch_schedule="global_permutation",
    )
    prepared = _prepare_training_step(
        batches,
        seed=args.seed,
        update=args.update,
        data_cursor=args.data_step,
        batch_size=args.batch_size,
        config=base_config,
    )
    batch = _torch_batch(prepared.compact_batch, device)
    choices_by_count = {
        count: _choices_to_device(
            materialize_step_choices(
                seed=args.seed,
                update=args.update,
                batch_size=args.batch_size,
                config=dataclasses.replace(
                    base_config,
                    sigreg_example_count=count,
                ),
                device=torch.device("cpu"),
            ),
            device,
        )
        for count in counts
    }
    baseline_choices = choices_by_count[counts[0]]
    for count in counts[1:]:
        candidate = choices_by_count[count]
        for name in (
            "target_horizon",
            "training_time",
            "mask_uniform",
            "sigreg_directions",
        ):
            torch.testing.assert_close(
                getattr(candidate, name),
                getattr(baseline_choices, name),
                rtol=0.0,
                atol=0.0,
            )
        torch.testing.assert_close(
            candidate.sigreg_indices[: counts[0]],
            baseline_choices.sigreg_indices,
            rtol=0.0,
            atol=0.0,
        )
    torch.testing.assert_close(
        prepared.choices.target_horizon,
        baseline_choices.target_horizon.cpu(),
        rtol=0.0,
        atol=0.0,
    )

    model, mapping = load_raw_bt4_hero_model(
        device=device,
        raw_bt4_path=args.raw_bt4_path,
        config=base_config,
        bt4_norm_impl="eager-fused-backward",
    )
    checkpoint_manifest = None
    if args.checkpoint_dir is not None:
        checkpoint_manifest = load_model_checkpoint(
            checkpoint_dir=args.checkpoint_dir,
            model=model,
        )
        if (
            checkpoint_manifest["source_mapping_sha256"]
            != mapping["combined_sha256"]
        ):
            raise ValueError("Checkpoint source mapping differs from raw BT4")
    model.train()
    compiled_regions = _apply_compile_regions(model, args.compile_regions)

    baseline_capture: dict[str, Any] = {}
    baseline_count = counts[0]
    model.config = dataclasses.replace(
        base_config,
        sigreg_example_count=baseline_count,
    )
    model.zero_grad(set_to_none=True)
    warm_forward_loss, warm_forward_aux = loss_and_aux(
        model,
        batch,
        baseline_choices,
        compute_dtype=torch.bfloat16,
        capture={},
    )
    del warm_forward_loss, warm_forward_aux
    warm_loss, warm_aux = loss_and_aux(
        model,
        batch,
        baseline_choices,
        compute_dtype=torch.bfloat16,
        capture=baseline_capture,
    )
    warm_components = baseline_capture["loss_components"]
    warm_target_sigreg = warm_components["target_sigreg"]
    warm_prediction_sigreg = warm_components["prediction_sigreg"]
    warm_objective = (
        base_config.target_sigreg_coeff * warm_target_sigreg
        + base_config.pred_sigreg_coeff
        * warm_prediction_sigreg
    )
    baseline_capture.pop("loss_components")
    del warm_loss, warm_aux, warm_components
    warm_objective.backward()
    torch.cuda.synchronize()
    del (
        warm_target_sigreg,
        warm_prediction_sigreg,
        warm_objective,
        baseline_capture,
    )
    model.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()

    records: dict[str, Any] = {}
    reference_gradients: dict[str, Tensor] | None = None
    reference_norms: dict[str, dict[str, Any]] | None = None
    reference_gradient_bytes = 0
    sigreg_inputs_cpu: dict[str, Tensor] | None = None
    failed_count: int | None = None
    for count in counts:
        model.config = dataclasses.replace(
            base_config,
            sigreg_example_count=count,
        )
        model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        capture: dict[str, Any] = {}
        forward_start = torch.cuda.Event(enable_timing=True)
        forward_end = torch.cuda.Event(enable_timing=True)
        backward_end = torch.cuda.Event(enable_timing=True)
        try:
            torch.cuda.synchronize()
            started = time.perf_counter()
            forward_start.record()
            full_loss, aux = loss_and_aux(
                model,
                batch,
                choices_by_count[count],
                compute_dtype=torch.bfloat16,
                capture=capture,
            )
            components = capture["loss_components"]
            target_sigreg = components["target_sigreg"]
            prediction_sigreg = components["prediction_sigreg"]
            target_valid_count = aux["jepa_sigreg_valid_count"]
            prediction_valid_count = aux[
                "jepa_pred_sigreg_valid_count"
            ]
            objective = (
                base_config.target_sigreg_coeff
                * target_sigreg
                + base_config.pred_sigreg_coeff
                * prediction_sigreg
            )
            capture.pop("loss_components")
            del full_loss, aux, components
            forward_end.record()
            objective.backward()
            backward_end.record()
            torch.cuda.synchronize()
            wall_seconds = time.perf_counter() - started
        except torch.OutOfMemoryError as exc:
            failed_count = count
            records[str(count)] = {
                "status": "oom",
                "error": str(exc),
            }
            print(
                json.dumps(
                    {
                        "sigreg_sample_audit_count": count,
                        "status": "oom",
                        "error": str(exc),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            break

        gradient_norms = _gradient_norms_by_group(model)
        cosine_to_baseline: dict[str, Any] | None
        gradient_copy_started = time.perf_counter()
        if count == baseline_count:
            reference_gradients, reference_gradient_bytes = (
                _copy_parameter_gradients_to_cpu(model)
            )
            reference_norms = gradient_norms
            cosine_to_baseline = {
                "groups": {
                    group: {
                        "dot": float(gradient_norms[group]["squared_norm"]),
                        "reference_norm": float(gradient_norms[group]["norm"]),
                        "candidate_norm": float(gradient_norms[group]["norm"]),
                        "cosine": (
                            1.0
                            if float(gradient_norms[group]["norm"]) > 1e-30
                            else None
                        ),
                        "candidate_to_reference_norm_ratio": (
                            1.0
                            if float(gradient_norms[group]["norm"]) > 1e-30
                            else None
                        ),
                    }
                    for group in (*_LOSS_AUDIT_GROUPS, "all")
                },
                "missing_from_candidate": [],
                "missing_from_reference": [],
            }
        else:
            assert reference_gradients is not None
            assert reference_norms is not None
            cosine_to_baseline = _gradient_cosines_against_cpu_reference(
                model,
                reference_gradients,
                reference_norms=reference_norms,
                candidate_norms=gradient_norms,
            )
        gradient_copy_seconds = time.perf_counter() - gradient_copy_started
        records[str(count)] = {
            "status": "complete",
            "target_sigreg": float(
                target_sigreg.detach().float().cpu()
            ),
            "prediction_sigreg": float(
                prediction_sigreg.detach().float().cpu()
            ),
            "weighted_shared_sigreg": float(
                objective.detach().float().cpu()
            ),
            "target_valid_count": float(
                target_valid_count.float().cpu()
            ),
            "prediction_valid_count": float(
                prediction_valid_count.float().cpu()
            ),
            "forward_cuda_seconds": (
                forward_start.elapsed_time(forward_end) / 1000.0
            ),
            "backward_cuda_seconds": (
                forward_end.elapsed_time(backward_end) / 1000.0
            ),
            "forward_backward_wall_seconds": wall_seconds,
            "gradient_copy_and_cosine_seconds": gradient_copy_seconds,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "gradient_groups": gradient_norms,
            "gradient_vs_baseline": cosine_to_baseline,
        }
        print(
            json.dumps(
                {
                    "sigreg_sample_audit_count": count,
                    "status": "complete",
                    "forward_cuda_seconds": records[str(count)][
                        "forward_cuda_seconds"
                    ],
                    "backward_cuda_seconds": records[str(count)][
                        "backward_cuda_seconds"
                    ],
                    "peak_allocated_bytes": records[str(count)][
                        "peak_allocated_bytes"
                    ],
                },
                sort_keys=True,
            ),
            flush=True,
        )
        del (
            capture,
            target_sigreg,
            prediction_sigreg,
            target_valid_count,
            prediction_valid_count,
            objective,
        )

    if failed_count is not None:
        failed_index = counts.index(failed_count)
        for count in counts[failed_index + 1 :]:
            records[str(count)] = {
                "status": "skipped_after_oom",
                "blocked_by_sample_count": failed_count,
            }
    else:
        model.config = dataclasses.replace(
            base_config,
            sigreg_example_count=baseline_count,
        )
        model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        input_capture: dict[str, Any] = {
            "capture_sigreg_inputs": True,
        }
        with torch.inference_mode():
            capture_loss, capture_aux = loss_and_aux(
                model,
                batch,
                baseline_choices,
                compute_dtype=torch.bfloat16,
                capture=input_capture,
            )
        captured_inputs = input_capture["sigreg_inputs"]
        sigreg_inputs_cpu = {
            name: value.detach().to(device="cpu", copy=True)
            for name, value in captured_inputs.items()
        }
        del (
            capture_loss,
            capture_aux,
            captured_inputs,
            input_capture,
        )

    scalar_dispersion: dict[str, Any] = {}
    if failed_count is None:
        assert sigreg_inputs_cpu is not None
        del reference_gradients
        reference_gradients = None
        model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        target_z = sigreg_inputs_cpu["target_z"].to(device)
        target_weight = sigreg_inputs_cpu["target_weight"].to(device)
        prediction_z = sigreg_inputs_cpu["prediction_z"].to(device)
        prediction_weight = sigreg_inputs_cpu["prediction_weight"].to(device)
        directions = sigreg_inputs_cpu["directions"].to(device)
        torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode():
            for count in counts:
                target_values: list[float] = []
                prediction_values: list[float] = []
                cuda_seconds: list[float] = []
                subset_sha256: list[str] = []
                for replicate in range(args.replicates):
                    if replicate == 0:
                        indices = (
                            choices_by_count[count]
                            .sigreg_indices.detach()
                            .cpu()
                            .numpy()
                        )
                    else:
                        sequence = np.random.SeedSequence(
                            [
                                int(args.seed),
                                int(args.update),
                                int(args.data_step),
                                int(count),
                                int(replicate),
                                0x51_67_52_45_47,
                            ]
                        )
                        rng = np.random.Generator(np.random.PCG64(sequence))
                        indices = rng.permutation(args.batch_size)[:count]
                    indices = np.asarray(indices, dtype=np.int64)
                    subset_sha256.append(
                        hashlib.sha256(indices.tobytes()).hexdigest()
                    )
                    rows = torch.from_numpy(indices).to(device)
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record()
                    target_value, _ = _sigreg_v_stat(
                        target_z[rows].float().reshape(
                            -1,
                            base_config.z_dim,
                        ),
                        target_weight[rows].reshape(-1),
                        directions,
                        reference_count=base_config.sigreg_reference_count,
                    )
                    prediction_value, _ = _sigreg_v_stat(
                        prediction_z[rows].float().reshape(
                            -1,
                            base_config.z_dim,
                        ),
                        prediction_weight[rows].reshape(-1),
                        directions,
                        reference_count=base_config.sigreg_reference_count,
                    )
                    end.record()
                    torch.cuda.synchronize()
                    target_values.append(float(target_value.cpu()))
                    prediction_values.append(float(prediction_value.cpu()))
                    cuda_seconds.append(start.elapsed_time(end) / 1000.0)

                def summarize(values: list[float]) -> dict[str, float]:
                    array = np.asarray(values, dtype=np.float64)
                    return {
                        "mean": float(array.mean()),
                        "sample_standard_deviation": float(
                            array.std(ddof=1)
                        ),
                        "standard_error": float(
                            array.std(ddof=1) / math.sqrt(len(array))
                        ),
                        "minimum": float(array.min()),
                        "maximum": float(array.max()),
                    }

                weighted = [
                    base_config.target_sigreg_coeff * target
                    + base_config.pred_sigreg_coeff * prediction
                    for target, prediction in zip(
                        target_values,
                        prediction_values,
                        strict=True,
                    )
                ]
                scalar_dispersion[str(count)] = {
                    "replicates": args.replicates,
                    "target": summarize(target_values),
                    "prediction": summarize(prediction_values),
                    "weighted_shared": summarize(weighted),
                    "mean_statistic_cuda_seconds": float(
                        np.mean(cuda_seconds)
                    ),
                    "subset_sha256": subset_sha256,
                }
        scalar_peak_allocated = torch.cuda.max_memory_allocated()
        scalar_peak_reserved = torch.cuda.max_memory_reserved()
    else:
        scalar_peak_allocated = None
        scalar_peak_reserved = None

    completed_counts = [
        count
        for count in counts
        if records.get(str(count), {}).get("status") == "complete"
    ]
    gate_checks = {
        "all_requested_counts_complete": completed_counts == list(counts),
        "nested_representative_samples": True,
        "finite_scalars": all(
            math.isfinite(float(records[str(count)][metric]))
            for count in completed_counts
            for metric in (
                "target_sigreg",
                "prediction_sigreg",
                "weighted_shared_sigreg",
            )
        ),
        "finite_gradients": all(
            group["finite"]
            for count in completed_counts
            for group in records[str(count)]["gradient_groups"].values()
        ),
        "gradient_support_matches_baseline": all(
            not records[str(count)]["gradient_vs_baseline"][
                "missing_from_candidate"
            ]
            and not records[str(count)]["gradient_vs_baseline"][
                "missing_from_reference"
            ]
            for count in completed_counts
        ),
    }
    report = {
        "schema_version": "torch-hero-sigreg-sample-audit-v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "git_commit": _git_commit(),
        "device": torch.cuda.get_device_name(device),
        "state": (
            "fresh_initialization"
            if checkpoint_manifest is None
            else f"checkpoint_u{checkpoint_manifest['optimizer_update']}"
        ),
        "checkpoint": checkpoint_manifest,
        "source_mapping_sha256": mapping["combined_sha256"],
        "batch_size": args.batch_size,
        "sample_counts": list(counts),
        "replicates": args.replicates,
        "data_step": args.data_step,
        "update": args.update,
        "seed": args.seed,
        "data_prepare_seconds": prepared.prepare_seconds,
        "bt4_norm_impl": "eager-fused-backward",
        "compile_regions": compiled_regions,
        "shared_sigreg_coefficient": base_config.target_sigreg_coeff,
        "coefficients_equal": (
            base_config.target_sigreg_coeff
            == base_config.pred_sigreg_coeff
        ),
        "representative_records": records,
        "scalar_dispersion": scalar_dispersion,
        "scalar_phase_peak_allocated_bytes": scalar_peak_allocated,
        "scalar_phase_peak_reserved_bytes": scalar_peak_reserved,
        "reference_gradient_cpu_bytes": reference_gradient_bytes,
        "data": batches.provenance(),
        "gate_checks": gate_checks,
        "gate_pass": all(gate_checks.values()),
    }
    _write_json(output, report)
    print(
        json.dumps(
            {
                "output": str(output),
                "state": report["state"],
                "completed_counts": completed_counts,
                "gate_pass": report["gate_pass"],
                "records": {
                    key: {
                        field: value[field]
                        for field in (
                            "status",
                            "forward_cuda_seconds",
                            "backward_cuda_seconds",
                            "peak_allocated_bytes",
                        )
                        if field in value
                    }
                    for key, value in records.items()
                },
            },
            sort_keys=True,
        ),
        flush=True,
    )
    if not report["gate_pass"]:
        raise RuntimeError("SIGReg sample-count audit gate failed")
    return 0


def _evaluate_validation_pool(
    model: JointModel,
    batches: Any,
    *,
    count: int,
    seed: int,
    device: torch.device,
) -> tuple[dict[str, float], float]:
    totals: dict[str, float] = {}
    started = time.perf_counter()
    model.eval()
    prefetch_executor = ThreadPoolExecutor(
        max_workers=1,
        thread_name_prefix="frozen-eval-prefetch",
    )
    prepared_future: Future[dict[str, Any]] | None = prefetch_executor.submit(
        batches.batch_at,
        0,
    )
    try:
        with torch.inference_mode():
            for index in range(count):
                assert prepared_future is not None
                numpy_batch = prepared_future.result()
                prepared_future = (
                    prefetch_executor.submit(batches.batch_at, index + 1)
                    if index + 1 < count
                    else None
                )
                batch = _torch_batch(numpy_batch, device)
                choices = materialize_step_choices(
                    seed=seed,
                    update=index,
                    batch_size=batches.batch_size,
                    config=model.config,
                    device=device,
                )
                permutation_rng = np.random.Generator(
                    np.random.PCG64(
                        np.random.SeedSequence(
                            [int(seed), int(index), 0xC011A95E]
                        )
                    )
                )
                target_shuffle = torch.from_numpy(
                    permutation_rng.permutation(batches.batch_size).astype(
                        np.int64,
                        copy=False,
                    )
                ).to(device)
                action_shuffle = torch.from_numpy(
                    permutation_rng.permutation(batches.batch_size).astype(
                        np.int64,
                        copy=False,
                    )
                ).to(device)
                metrics = full_horizon_evaluation_aux(
                    model,
                    batch,
                    choices,
                    target_shuffle=target_shuffle,
                    action_shuffle=action_shuffle,
                    compute_dtype=torch.bfloat16,
                )
                torch.cuda.synchronize()
                row = _flatten_torch_metrics(metrics)
                for key, value in row.items():
                    if not math.isfinite(value):
                        raise FloatingPointError(
                            "Non-finite validation metric "
                            f"{key} at seed={seed}, batch={index}: {value}"
                        )
                    totals[key] = totals.get(key, 0.0) + value
                del (
                    numpy_batch,
                    batch,
                    choices,
                    target_shuffle,
                    action_shuffle,
                    metrics,
                )
    finally:
        if prepared_future is not None:
            prepared_future.cancel()
        prefetch_executor.shutdown(wait=True, cancel_futures=True)
    return (
        {key: value / count for key, value in totals.items()},
        time.perf_counter() - started,
    )


def _load_hero_frozen_pool(
    *,
    manifest_path: Path,
    pool_name: str,
    batch_size: int,
) -> tuple[FrozenIndexTrajectoryBatches, dict[str, Any]]:
    manifest_path = _require_workspace(manifest_path, exists=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "chess-dfm-hero-eval-manifest-v1":
        raise ValueError("Unsupported hero evaluation manifest")
    if manifest.get("immutable_after_creation") is not True:
        raise ValueError("Hero evaluation manifest is not immutable")
    if manifest.get("action_codec") != HERO_CONFIG.action_codec:
        raise ValueError("Hero evaluation action codec drift")
    if int(manifest.get("horizon", -1)) != HERO_CONFIG.horizon:
        raise ValueError("Hero evaluation horizon drift")
    definitions = {
        "fast": "fast_validation",
        "primary": "primary_validation",
        "blind": "blind_test",
    }
    try:
        definition = manifest["position_indices"][definitions[pool_name]]
    except KeyError as exc:
        raise ValueError(f"Unsupported frozen hero pool: {pool_name!r}") from exc
    indices_record = manifest["position_indices"]
    indices_path = _require_workspace(indices_record["path"], exists=True)
    if indices_path.stat().st_size != int(indices_record["size_bytes"]):
        raise ValueError("Frozen hero index size drift")
    if _sha256_file(indices_path) != indices_record["sha256"]:
        raise ValueError("Frozen hero index checksum drift")
    with np.load(indices_path, allow_pickle=False) as payload:
        expected_arrays = {
            "fast_val_global_index",
            "primary_val_global_index",
            "blind_test_global_index",
        }
        if set(payload.files) != expected_arrays:
            raise ValueError(
                f"Frozen hero index arrays drift: {sorted(payload.files)}"
            )
        indices = np.asarray(payload[definition["array"]]).copy()
    if indices.size != int(definition["count"]):
        raise ValueError("Frozen hero pool count drift")
    split = str(definition["split"])
    split_record = manifest["dataset"][split]
    batches = FrozenIndexTrajectoryBatches(
        _require_workspace(split_record["path"], exists=True),
        global_indices=indices,
        batch_size=batch_size,
        horizon=HERO_CONFIG.horizon,
        indices_sha256=indices_record["sha256"],
        pool_name=pool_name,
    )
    provenance = batches.provenance()
    if (
        provenance["file_manifest_sha256"]
        != split_record["filename_size_manifest_sha256"]
    ):
        raise ValueError("Frozen hero split inventory drift")
    return batches, {
        "manifest_path": str(manifest_path),
        "manifest_sha256": _sha256_file(manifest_path),
        "indices_path": str(indices_path),
        "indices_sha256": indices_record["sha256"],
        "pool_name": pool_name,
        "pool_definition": definition,
        "data": provenance,
    }


@dataclasses.dataclass(frozen=True)
class _HeroMilestoneResources:
    fast_batches: FrozenIndexTrajectoryBatches
    fast_pool: dict[str, Any]
    opening_pool: dict[str, Any]
    loaded_histories: Any
    arena_provenance: dict[str, Any]


def _hero_milestone_update(
    percentage: int,
    *,
    total_examples: int,
    batch_size: int,
) -> int:
    if not 1 <= percentage <= 100:
        raise ValueError("percentage must be in [1, 100]")
    if total_examples < 1 or batch_size < 1:
        raise ValueError("total_examples and batch_size must be positive")
    return (
        percentage * total_examples + 100 * batch_size - 1
    ) // (100 * batch_size)


def _hero_milestone_contract(
    args: argparse.Namespace,
    *,
    enabled: bool,
) -> dict[str, Any]:
    if not enabled:
        return {"enabled": False}
    validation_updates = {
        str(percentage): _hero_milestone_update(
            percentage,
            total_examples=_HERO_TRAIN_EXAMPLES,
            batch_size=args.batch_size,
        )
        for percentage in _HERO_FAST_VALIDATION_PERCENTAGES
    }
    arena_updates = {
        str(percentage): _hero_milestone_update(
            percentage,
            total_examples=_HERO_TRAIN_EXAMPLES,
            batch_size=args.batch_size,
        )
        for percentage in _HERO_ARENA_PERCENTAGES
    }
    return {
        "enabled": True,
        "schedule_unit": "examples",
        "total_examples": _HERO_TRAIN_EXAMPLES,
        "fast_validation": {
            "percentages": list(_HERO_FAST_VALIDATION_PERCENTAGES),
            "updates": validation_updates,
            "batch_size": HERO_CONFIG.sigreg_example_count,
            "pool": "fast",
        },
        "paired_arena": {
            "percentages": list(_HERO_ARENA_PERCENTAGES),
            "updates": arena_updates,
            "pair_count": int(args.hero_arena_pairs),
            "additional_ply_cap": int(args.hero_arena_additional_ply_cap),
            "refinement_passes": HERO_CONFIG.horizon,
            "inference_batch_size": int(
                args.hero_arena_inference_batch_size
            ),
            "opponent": "raw_bt4",
        },
        "manifest_path": str(
            _require_workspace(args.hero_eval_manifest, exists=True)
        ),
        "resume_boundary_policy": (
            "a milestone exactly equal to the restored update is repeated; "
            "earlier milestones are skipped"
        ),
        "checkpoint_policy": (
            "milestone evaluations use the live model and serialize no model state"
        ),
    }


def _verified_hero_manifest_asset(
    record: Mapping[str, Any],
    *,
    label: str,
) -> Path:
    try:
        path = _require_workspace(record["path"], exists=True)
        expected_size = int(record["size_bytes"])
        expected_sha256 = str(record["sha256"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Malformed {label} record in hero manifest") from exc
    if path.stat().st_size != expected_size:
        raise ValueError(f"{label} size drift")
    if _sha256_file(path) != expected_sha256:
        raise ValueError(f"{label} checksum drift")
    return path


def _load_hero_milestone_resources(
    *,
    manifest_path: Path,
    arena_pairs: int,
) -> _HeroMilestoneResources:
    from research.arena import load_opening_pool
    from research.play_arena import load_opening_history_sidecar

    if arena_pairs < 1:
        raise ValueError("hero arena pair count must be positive")
    fast_batches, fast_pool = _load_hero_frozen_pool(
        manifest_path=manifest_path,
        pool_name="fast",
        batch_size=HERO_CONFIG.sigreg_example_count,
    )
    manifest_source = _require_workspace(manifest_path, exists=True)
    manifest = json.loads(manifest_source.read_text(encoding="utf-8"))
    try:
        paired = manifest["paired_arena"]
        pool_record = paired["opening_pool"]
        history_record = paired["opening_histories"]
        contract_record = paired["opening_contract"]
    except (KeyError, TypeError) as exc:
        raise ValueError("Hero manifest lacks the paired arena contract") from exc
    pool_path = _verified_hero_manifest_asset(
        pool_record,
        label="hero arena opening pool",
    )
    history_path = _verified_hero_manifest_asset(
        history_record,
        label="hero arena history sidecar",
    )
    contract_path = _verified_hero_manifest_asset(
        contract_record,
        label="hero arena opening contract",
    )
    opening_pool = load_opening_pool(pool_path)
    openings = opening_pool.get("openings")
    if not isinstance(openings, list) or len(openings) != int(
        paired["opening_count"]
    ):
        raise ValueError("Hero arena opening count drift")
    if arena_pairs > len(openings):
        raise ValueError(
            f"Requested {arena_pairs} arena pairs from {len(openings)} openings"
        )
    sidecar_payload = json.loads(history_path.read_text(encoding="utf-8"))
    expected_sidecar_manifest = str(
        sidecar_payload.get("manifest_sha256", "")
    )
    loaded_histories = load_opening_history_sidecar(
        history_path,
        opening_pool=opening_pool,
        expected_manifest_sha256=expected_sidecar_manifest,
    )
    return _HeroMilestoneResources(
        fast_batches=fast_batches,
        fast_pool=fast_pool,
        opening_pool=opening_pool,
        loaded_histories=loaded_histories,
        arena_provenance={
            "manifest_path": str(manifest_source),
            "manifest_sha256": _sha256_file(manifest_source),
            "opening_pool": {
                **dict(pool_record),
                "pool_sha256": opening_pool["pool_sha256"],
            },
            "opening_histories": {
                **dict(history_record),
                "manifest_sha256": loaded_histories.manifest_sha256,
                "pool_sha256": loaded_histories.pool_sha256,
            },
            "opening_contract": {
                **dict(contract_record),
                "verified_path": str(contract_path),
            },
            "opening_count": len(openings),
        },
    )


def _append_jsonl(path: Path, record: Mapping[str, Any]) -> None:
    destination = _require_workspace(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                dict(record),
                allow_nan=False,
                sort_keys=True,
            )
            + "\n"
        )
        handle.flush()
        os.fsync(handle.fileno())


def _run_hero_validation_milestone(
    model: JointModel,
    resources: _HeroMilestoneResources,
    *,
    output_dir: Path,
    update: int,
    batch_size: int,
    percentage: int,
    device: torch.device,
) -> dict[str, Any]:
    was_training = model.training
    try:
        metrics, evaluation_seconds = _evaluate_validation_pool(
            model,
            resources.fast_batches,
            count=resources.fast_batches.steps_per_epoch,
            seed=int(resources.fast_pool["pool_definition"]["seed"]),
            device=device,
        )
    finally:
        model.train(was_training)
    record = {
        "schema_version": "torch-hero-live-validation-milestone-v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "git_commit": _git_commit(),
        "update": update,
        "examples": update * batch_size,
        "target_percentage": percentage,
        "observed_fraction": (
            update * batch_size / _HERO_TRAIN_EXAMPLES
        ),
        "pool": resources.fast_pool,
        "evaluation_examples": int(
            resources.fast_batches.global_indices.size
        ),
        "evaluation_seconds": evaluation_seconds,
        "metrics": metrics,
        "serialized_model_state": False,
    }
    _append_jsonl(
        output_dir / "hero_validation_metrics.jsonl",
        record,
    )
    print(
        json.dumps(
            {
                "hero_validation_milestone": percentage,
                "update": update,
                "evaluation_seconds": evaluation_seconds,
                "metrics": metrics,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return record


def _arena_warmup_inputs(
    resources: _HeroMilestoneResources,
) -> tuple[list[chess.Board], list[tuple[chess.Board, ...]]]:
    opening = resources.opening_pool["openings"][0]
    board = chess.Board(str(opening["fen"]))
    history_fens = resources.loaded_histories.histories_by_opening_index[0]
    history = tuple(chess.Board(fen) for fen in history_fens)
    return [board], [history]


def _run_hero_arena_milestone(
    model: JointModel,
    resources: _HeroMilestoneResources,
    *,
    output_dir: Path,
    raw_bt4_path: Path,
    update: int,
    batch_size: int,
    percentage: int,
    arena_pairs: int,
    additional_ply_cap: int,
    inference_batch_size: int,
    device: torch.device,
) -> dict[str, Any]:
    from research.arena import (
        make_color_reversed_pairs,
        pair_aware_score_elo_interval,
        pair_score_for_model,
        pentanomial_stats,
    )
    from research.play_arena import (
        histories_for_pairs,
        play_arena_pairs,
    )

    candidate_id = f"hero-live-u{update:08d}"
    opponent_id = "raw-bt4"
    was_training = model.training
    model.eval()
    gc.collect()
    torch.cuda.empty_cache()
    opponent: JointModel | None = None
    candidate_policy: TorchHeroArenaPolicy | None = None
    opponent_policy: TorchHeroArenaPolicy | None = None
    try:
        opponent, opponent_initialization = load_raw_bt4_hero_model(
            device=device,
            raw_bt4_path=raw_bt4_path,
            config=HERO_CONFIG,
        )
        opponent.eval()
        candidate_policy = TorchHeroArenaPolicy(
            model=model,
            model_id=candidate_id,
            policy_mode="dfm",
            inference_batch_size=inference_batch_size,
            refinement_passes=HERO_CONFIG.horizon,
        )
        opponent_policy = TorchHeroArenaPolicy(
            model=opponent,
            model_id=opponent_id,
            policy_mode="raw_bt4",
            inference_batch_size=inference_batch_size,
            refinement_passes=HERO_CONFIG.horizon,
        )
        warm_boards, warm_histories = _arena_warmup_inputs(resources)
        warm_started = time.perf_counter()
        candidate_policy.select_actions(warm_boards, warm_histories)
        opponent_policy.select_actions(warm_boards, warm_histories)
        warmup_seconds = time.perf_counter() - warm_started

        fens = [
            str(opening["fen"])
            for opening in resources.opening_pool["openings"][:arena_pairs]
        ]
        pairs = make_color_reversed_pairs(
            fens,
            model_a=candidate_id,
            model_b=opponent_id,
        )
        opening_histories = histories_for_pairs(
            pairs,
            resources.loaded_histories,
        )
        arena_started = time.perf_counter()
        gameplay = play_arena_pairs(
            pairs,
            opening_histories=opening_histories,
            policies={
                candidate_id: candidate_policy,
                opponent_id: opponent_policy,
            },
            additional_ply_cap=additional_ply_cap,
            policy_timeout_seconds=30.0,
            policy_batch_size_cap=inference_batch_size,
        )
        arena_seconds = time.perf_counter() - arena_started
        outcomes = tuple(record.outcome for record in gameplay.records)
        pair_scores = [
            pair_score_for_model(
                outcomes[offset : offset + 2],
                model_id=candidate_id,
            )
            for offset in range(0, len(outcomes), 2)
        ]
        stats = pentanomial_stats(pair_scores)
        interval = pair_aware_score_elo_interval(stats)
        gameplay_payload = gameplay.as_dict()
        report = {
            "schema_version": "torch-hero-live-arena-milestone-v1",
            "created_utc": datetime.now(UTC).isoformat(),
            "git_commit": _git_commit(),
            "update": update,
            "examples": update * batch_size,
            "target_percentage": percentage,
            "observed_fraction": (
                update * batch_size / _HERO_TRAIN_EXAMPLES
            ),
            "candidate_model_id": candidate_id,
            "opponent_model_id": opponent_id,
            "action_codec": ACTION_CODEC_LC0_CANONICAL_1858,
            "refinement_passes": HERO_CONFIG.horizon,
            "inference_batch_size": inference_batch_size,
            "additional_ply_cap": additional_ply_cap,
            "pair_scores": pair_scores,
            "pentanomial": stats.as_dict(),
            "descriptive_logistic_elo_interval": dataclasses.asdict(interval),
            "warmup_seconds": warmup_seconds,
            "arena_seconds": arena_seconds,
            "gameplay": gameplay_payload,
            "arena_provenance": resources.arena_provenance,
            "opponent_initialization": {
                "combined_sha256": opponent_initialization[
                    "combined_sha256"
                ],
                "leaf_count": opponent_initialization["leaf_count"],
                "raw_asset": opponent_initialization["raw_asset"],
            },
            "serialized_model_state": False,
        }
        report_path = (
            output_dir
            / "hero_arena_milestones"
            / f"update{update:08d}.json"
        )
        _write_json(report_path, report)
        summary = {
            key: value
            for key, value in report.items()
            if key not in {"gameplay", "pair_scores", "arena_provenance"}
        }
        summary["report_path"] = str(report_path)
        summary["gameplay_payload_sha256"] = gameplay_payload[
            "payload_sha256"
        ]
        summary["gameplay_stats"] = gameplay_payload["stats"]
        _append_jsonl(
            output_dir / "hero_arena_metrics.jsonl",
            summary,
        )
        print(
            json.dumps(
                {
                    "hero_arena_milestone": percentage,
                    "update": update,
                    "pair_count": stats.pair_count,
                    "score": stats.score,
                    "elo": interval.elo,
                    "elo_lower": interval.elo_lower,
                    "elo_upper": interval.elo_upper,
                    "arena_seconds": arena_seconds,
                    "fault_counts": gameplay_payload["stats"][
                        "fault_counts"
                    ],
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return summary
    finally:
        candidate_policy = None
        opponent_policy = None
        if opponent is not None:
            del opponent
        gc.collect()
        torch.cuda.empty_cache()
        model.train(was_training)


def hero_milestone_smoke(args: argparse.Namespace) -> int:
    """Compile and exercise the exact live-evaluation shapes without training."""

    if not torch.cuda.is_available():
        raise RuntimeError("hero-milestone-smoke requires CUDA")
    if args.threads not in (1, 2):
        raise ValueError("--threads must be 1 or 2 under the resource guard")
    if args.gpu_monitor_interval_ms < 50:
        raise ValueError("--gpu-monitor-interval-ms must be at least 50")
    output_dir = _require_workspace(args.output_dir)
    if output_dir.exists():
        raise FileExistsError(
            f"Hero milestone smoke output already exists: {output_dir}"
        )
    resources = _load_hero_milestone_resources(
        manifest_path=args.eval_manifest,
        arena_pairs=1,
    )
    device = torch.device("cuda")
    torch.set_num_threads(args.threads)
    torch.set_float32_matmul_precision("high")
    model, initialization = load_raw_bt4_hero_model(
        device=device,
        raw_bt4_path=args.raw_bt4_path,
        config=HERO_CONFIG,
    )
    compiled_regions = _apply_compile_regions(model, "fresh")
    model.eval()
    output_dir.mkdir(parents=True)
    run_config = {
        "schema_version": "torch-hero-milestone-smoke-run-v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "git_commit": _git_commit(),
        "device": torch.cuda.get_device_name(device),
        "compute_dtype": "bfloat16",
        "compile_regions": compiled_regions,
        "compile_settings": {
            "backend": "inductor",
            "fullgraph": True,
            "dynamic": False,
            "mode": "default",
            "compile_threads": 1,
            "max_autotune": False,
        },
        "fast_validation_batches": 1,
        "fast_validation_batch_size": HERO_CONFIG.sigreg_example_count,
        "arena_pairs": 1,
        "arena_additional_ply_cap": 2,
        "arena_inference_batch_size": _HERO_ARENA_INFERENCE_BATCH_SIZE,
        "arena_refinement_passes": HERO_CONFIG.horizon,
        "raw_initialization": {
            "combined_sha256": initialization["combined_sha256"],
            "leaf_count": initialization["leaf_count"],
            "raw_asset": initialization["raw_asset"],
        },
        "resources": {
            "fast_pool": resources.fast_pool,
            "arena": resources.arena_provenance,
        },
    }
    _write_json(output_dir / "run_config.json", run_config)
    monitor = _start_gpu_monitor(
        output_dir,
        interval_ms=args.gpu_monitor_interval_ms,
    )
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    wall_started = time.perf_counter()
    try:
        validation_metrics, validation_seconds = _evaluate_validation_pool(
            model,
            resources.fast_batches,
            count=1,
            seed=int(resources.fast_pool["pool_definition"]["seed"]),
            device=device,
        )
        arena_summary = _run_hero_arena_milestone(
            model,
            resources,
            output_dir=output_dir,
            raw_bt4_path=args.raw_bt4_path,
            update=0,
            batch_size=1024,
            percentage=0,
            arena_pairs=1,
            additional_ply_cap=2,
            inference_batch_size=_HERO_ARENA_INFERENCE_BATCH_SIZE,
            device=device,
        )
    finally:
        _stop_gpu_monitor(monitor)
    residual_zero = bool(
        torch.count_nonzero(model.out_proj).item() == 0
        and torch.count_nonzero(model.out_bias).item() == 0
    )
    gate_checks = {
        "validation_metrics_finite": all(
            math.isfinite(value)
            for value in validation_metrics.values()
        ),
        "zero_initialized_dfm_residual": residual_zero,
        "arena_has_no_faults": not bool(
            arena_summary["gameplay_stats"]["fault_counts"]
        ),
        "arena_update_zero_score_is_half": math.isclose(
            float(arena_summary["pentanomial"]["score"]),
            0.5,
            abs_tol=1e-12,
        ),
    }
    report = {
        **run_config,
        "completed_utc": datetime.now(UTC).isoformat(),
        "wall_seconds": time.perf_counter() - wall_started,
        "validation_seconds": validation_seconds,
        "validation_metrics": validation_metrics,
        "arena": arena_summary,
        "gpu_monitor": _summarize_gpu_samples(
            output_dir / "gpu_samples.csv"
        ),
        "gpu_peak_memory_allocated_bytes": (
            torch.cuda.max_memory_allocated()
        ),
        "gpu_peak_memory_reserved_bytes": (
            torch.cuda.max_memory_reserved()
        ),
        "compile_counters": _compile_counter_snapshot(),
        "gate_checks": gate_checks,
        "gate_pass": all(gate_checks.values()),
    }
    _write_json(output_dir / "report.json", report)
    print(json.dumps(report, sort_keys=True), flush=True)
    if not report["gate_pass"]:
        raise RuntimeError("Hero live-milestone smoke gate failed")
    return 0


def evaluate_hero_pool(args: argparse.Namespace) -> int:
    """Evaluate clean hero initialization/checkpoint on one frozen index set."""

    if not torch.cuda.is_available():
        raise RuntimeError("hero-evaluate requires CUDA")
    if args.eval_batch_size != HERO_CONFIG.sigreg_example_count:
        raise ValueError(
            "Frozen hero evaluation requires batch "
            f"{HERO_CONFIG.sigreg_example_count}"
        )
    if args.threads not in (1, 2):
        raise ValueError("--threads must be 1 or 2 under the resource guard")
    if args.pool == "blind" and (
        args.checkpoint_dir is None or not args.allow_blind_terminal
    ):
        raise ValueError(
            "Blind evaluation requires a terminal checkpoint and "
            "--allow-blind-terminal"
        )
    output_dir = _require_workspace(args.output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Hero evaluation output exists: {output_dir}")

    batches, frozen_pool = _load_hero_frozen_pool(
        manifest_path=args.eval_manifest,
        pool_name=args.pool,
        batch_size=args.eval_batch_size,
    )
    device = torch.device("cuda")
    torch.set_num_threads(args.threads)
    torch.set_float32_matmul_precision("high")
    model, initialization = load_raw_bt4_hero_model(
        device=device,
        raw_bt4_path=args.raw_bt4_path,
        config=HERO_CONFIG,
    )
    checkpoint_manifest = None
    if args.checkpoint_dir is not None:
        checkpoint_manifest = load_checkpoint_model_for_evaluation(
            checkpoint_dir=args.checkpoint_dir,
            model=model,
        )
        if (
            checkpoint_manifest["source_mapping_sha256"]
            != initialization["combined_sha256"]
        ):
            raise ValueError("Hero checkpoint raw-BT4 mapping drift")
        state_label = (
            f"hero_checkpoint_u{checkpoint_manifest['optimizer_update']}"
        )
    else:
        state_label = "hero_initialization"
    output_dir.mkdir(parents=True)
    run_config = {
        "schema_version": "torch-hero-frozen-evaluation-run-v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "git_commit": _git_commit(),
        "state": state_label,
        "pool": frozen_pool,
        "eval_batch_size": args.eval_batch_size,
        "eval_batches": batches.steps_per_epoch,
        "evaluation_examples": int(batches.global_indices.size),
        "global_index_order": "ascending",
        "stochastic_choices": (
            "PCG64 keyed by the frozen pool seed and ascending batch index"
        ),
        "prefetch": {
            "depth": 1,
            "workers": 1,
            "deterministic_batch_order": True,
        },
        "compute_dtype": "bfloat16",
        "execution": "eager",
        "config": dataclasses.asdict(HERO_CONFIG),
        "raw_initialization": initialization,
        "checkpoint": checkpoint_manifest,
        "wdl_calibration": (
            "wdl_ece_15 is fixed-batch mean 15-bin confidence ECE; "
            "expected-value bias is also reported"
        ),
        "args": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
            if key != "handler"
        },
    }
    _write_json(output_dir / "run_config.json", run_config)
    monitor = (
        _start_gpu_monitor(
            output_dir,
            interval_ms=args.gpu_monitor_interval_ms,
        )
        if args.gpu_monitor_interval_ms > 0
        else None
    )
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    wall_started = time.perf_counter()
    pool_seed = int(frozen_pool["pool_definition"]["seed"])
    try:
        metrics, evaluation_seconds = _evaluate_validation_pool(
            model,
            batches,
            count=batches.steps_per_epoch,
            seed=pool_seed,
            device=device,
        )
    finally:
        _stop_gpu_monitor(monitor)
    residual_zero = bool(
        torch.count_nonzero(model.out_proj).item() == 0
        and torch.count_nonzero(model.out_bias).item() == 0
    )
    gate_checks = {
        "frozen_pool_count_reproduced": (
            batches.global_indices.size
            == int(frozen_pool["pool_definition"]["count"])
        ),
        "all_metrics_finite": all(math.isfinite(value) for value in metrics.values()),
        "initial_residual_is_zero": (
            checkpoint_manifest is not None or residual_zero
        ),
        "blind_terminal_authorized": (
            args.pool != "blind"
            or (
                checkpoint_manifest is not None
                and args.allow_blind_terminal
            )
        ),
    }
    report = {
        **run_config,
        "completed_utc": datetime.now(UTC).isoformat(),
        "wall_seconds": time.perf_counter() - wall_started,
        "evaluation_seconds": evaluation_seconds,
        "metrics": metrics,
        "gpu_monitor": _summarize_gpu_samples(output_dir / "gpu_samples.csv"),
        "gpu_peak_memory_allocated_bytes": torch.cuda.max_memory_allocated(),
        "gpu_peak_memory_reserved_bytes": torch.cuda.max_memory_reserved(),
        "gate_checks": gate_checks,
        "gate_pass": all(gate_checks.values()),
    }
    _write_json(output_dir / "report.json", report)
    print(
        json.dumps(
            {
                "state": state_label,
                "pool": args.pool,
                "examples": int(batches.global_indices.size),
                "evaluation_seconds": evaluation_seconds,
                "metrics": metrics,
                "gate_pass": report["gate_pass"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    if not report["gate_pass"]:
        raise RuntimeError("Frozen hero evaluation gate failed")
    return 0


def evaluate_checkpoint(args: argparse.Namespace) -> int:
    """Evaluate source and/or one sparse checkpoint on the frozen pools."""

    if not torch.cuda.is_available():
        raise RuntimeError("research/train_torch.py evaluate requires CUDA")
    if args.eval_batch_size != CONFIG.sigreg_example_count:
        raise ValueError(
            f"The frozen migration gate requires --eval-batch-size {CONFIG.sigreg_example_count}"
        )
    if args.eval_batches < 1:
        raise ValueError("--eval-batches must be positive")
    if len(set(args.eval_seeds)) != len(args.eval_seeds):
        raise ValueError("--eval-seeds must not contain duplicates")
    if args.threads not in (1, 2):
        raise ValueError("--threads must be 1 or 2 under the resource guard")
    if not args.include_source and args.checkpoint_dir is None:
        raise ValueError("Set --include-source and/or --checkpoint-dir")

    output_dir = _require_workspace(args.output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Evaluation output already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    device = torch.device("cuda")
    torch.set_num_threads(args.threads)
    torch.set_float32_matmul_precision("high")

    model, source_mapping = load_source_model(
        device=device,
        source_path=args.source_state,
    )
    from research.prepare import FixedTrajectoryBatches

    validation_batches = {
        seed: FixedTrajectoryBatches(
            _require_workspace(args.data_root) / "val",
            batch_size=args.eval_batch_size,
            horizon=CONFIG.horizon,
            seed=seed,
            shuffle_files=True,
            batch_schedule="global_permutation",
        )
        for seed in args.eval_seeds
    }
    run_config = {
        "schema_version": "torch-eager-validation-run-v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "git_commit": _git_commit(),
        "framework": "torch",
        "execution": "eager",
        "compute_dtype": "bfloat16",
        "full_horizon_evaluation": True,
        "deterministic_t": 0.0,
        "collapse_diagnostics": True,
        "eval_batch_size": args.eval_batch_size,
        "eval_batches": args.eval_batches,
        "eval_seeds": list(args.eval_seeds),
        "validation_examples_per_seed": args.eval_batch_size * args.eval_batches,
        "source_mapping_sha256": source_mapping["combined_sha256"],
        "checkpoint_dir": (
            None
            if args.checkpoint_dir is None
            else str(_require_workspace(args.checkpoint_dir, exists=True))
        ),
        "include_source": bool(args.include_source),
        "args": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
            if key != "handler"
        },
        "validation_schedule": (
            "one global permutation per seed; batch indexes "
            "[0, eval_batches) reused for every evaluated state"
        ),
        "stochastic_choices": ("framework-neutral PCG64 keyed by validation seed and batch index"),
        "val_data_by_seed": {
            str(seed): batches.provenance() for seed, batches in validation_batches.items()
        },
    }
    _write_json(output_dir / "run_config.json", run_config)

    records: list[dict[str, Any]] = []
    checkpoint_manifest = None
    monitor = (
        _start_gpu_monitor(
            output_dir,
            interval_ms=args.gpu_monitor_interval_ms,
        )
        if args.gpu_monitor_interval_ms > 0
        else None
    )
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    wall_started = time.perf_counter()
    try:
        if args.include_source:
            for seed, batches in validation_batches.items():
                metrics, seconds = _evaluate_validation_pool(
                    model,
                    batches,
                    count=args.eval_batches,
                    seed=seed,
                    device=device,
                )
                records.append(
                    {
                        "state": "source",
                        "validation_seed": seed,
                        "eval_batch_size": args.eval_batch_size,
                        "eval_batches": args.eval_batches,
                        "validation_examples": (args.eval_batch_size * args.eval_batches),
                        "validation_seconds": seconds,
                        "validation": metrics,
                        "checkpoint": None,
                    }
                )

        if args.checkpoint_dir is not None:
            checkpoint_manifest = load_model_checkpoint(
                checkpoint_dir=args.checkpoint_dir,
                model=model,
            )
            if checkpoint_manifest["source_mapping_sha256"] != source_mapping["combined_sha256"]:
                raise ValueError("Checkpoint source mapping differs from the restored source ABI")
            label = f"checkpoint_u{checkpoint_manifest['optimizer_update']}"
            for seed, batches in validation_batches.items():
                metrics, seconds = _evaluate_validation_pool(
                    model,
                    batches,
                    count=args.eval_batches,
                    seed=seed,
                    device=device,
                )
                records.append(
                    {
                        "state": label,
                        "validation_seed": seed,
                        "eval_batch_size": args.eval_batch_size,
                        "eval_batches": args.eval_batches,
                        "validation_examples": (args.eval_batch_size * args.eval_batches),
                        "validation_seconds": seconds,
                        "validation": metrics,
                        "checkpoint": checkpoint_manifest,
                    }
                )
    finally:
        _stop_gpu_monitor(monitor)

    metrics_path = output_dir / "validation_metrics.jsonl"
    with metrics_path.open("x", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    aggregate: dict[str, dict[str, float]] = {}
    for label in sorted({record["state"] for record in records}):
        matching = [record["validation"] for record in records if record["state"] == label]
        aggregate[label] = {
            key: float(np.mean([row[key] for row in matching])) for key in matching[0]
        }
    report = {
        **run_config,
        "completed_utc": datetime.now(UTC).isoformat(),
        "wall_seconds": time.perf_counter() - wall_started,
        "records": records,
        "two_pool_mean_by_state": aggregate,
        "checkpoint": checkpoint_manifest,
        "gpu_monitor": _summarize_gpu_samples(output_dir / "gpu_samples.csv"),
        "gpu_peak_memory_allocated_bytes": torch.cuda.max_memory_allocated(),
        "gpu_peak_memory_reserved_bytes": torch.cuda.max_memory_reserved(),
    }
    _write_json(output_dir / "report.json", report)
    print(json.dumps(report, sort_keys=True), flush=True)
    return 0


def inspect_source(args: argparse.Namespace) -> int:
    model, mapping = load_source_model(device=torch.device("cpu"), source_path=args.source_state)
    output = _require_workspace(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(mapping, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "output": str(output),
                "parameters": sum(parameter.numel() for parameter in model.parameters()),
                "mapping_sha256": mapping["combined_sha256"],
                "leaf_count": mapping["leaf_count"],
                "nbytes": mapping["nbytes"],
            },
            sort_keys=True,
        )
    )
    return 0


def cpu_smoke(args: argparse.Namespace) -> int:
    from research.prepare import FixedTrajectoryBatches

    torch.set_num_threads(args.threads)
    device = torch.device("cpu")
    model, mapping = load_source_model(device=device, source_path=args.source_state)
    model.train()
    batches = FixedTrajectoryBatches(
        _require_workspace(args.data_root) / "train",
        batch_size=args.batch_size,
        horizon=CONFIG.horizon,
        seed=args.seed,
        shuffle_files=True,
        batch_schedule="global_permutation",
    )
    batch = _torch_batch(batches.batch_at(args.data_step), device)
    choices = materialize_step_choices(
        seed=args.seed,
        update=args.update,
        batch_size=args.batch_size,
        config=CONFIG,
        device=device,
    )
    started = time.perf_counter()
    loss, aux = loss_and_aux(model, batch, choices, compute_dtype=torch.float32)
    if args.backward:
        loss.backward()
    elapsed = time.perf_counter() - started
    finite_gradients = (
        all(
            parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
            for parameter in model.parameters()
        )
        if args.backward
        else None
    )
    print(
        json.dumps(
            {
                "mode": "cpu_smoke",
                "backward": bool(args.backward),
                "seconds": elapsed,
                "finite_gradients": finite_gradients,
                "source_mapping_sha256": mapping["combined_sha256"],
                "metrics": _json_scalars(aux),
            },
            sort_keys=True,
        )
    )
    return 0


def verify_hero_init(args: argparse.Namespace) -> int:
    """Gate update-zero Torch policy actions against the pinned raw-BT4 oracle."""

    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if args.threads not in (1, 2):
        raise ValueError("--threads must be 1 or 2 under the resource guard")
    output = _require_workspace(args.output)
    if output.exists():
        raise FileExistsError(f"Verification output already exists: {output}")
    torch.set_num_threads(args.threads)
    torch.set_float32_matmul_precision("high")
    batches = CanonicalTrajectoryBatches(
        _require_workspace(args.data_root) / "val",
        batch_size=args.batch_size,
        horizon=HERO_CONFIG.horizon,
        seed=args.seed,
        shuffle_files=True,
        batch_schedule="global_permutation",
    )
    numpy_batch = batches.batch_at(0)

    import jax
    import jax.numpy as jnp

    from chess_dfm_jax.nnx_bt4 import jit_bt4_forward, make_bt4_model
    from chess_dfm_jax.policy import attention_policy_map
    from chess_dfm_jax.weights import load_pb_gz, map_bt4_weights

    source_path = _require_workspace(args.raw_bt4_path, exists=True)
    mapped = map_bt4_weights(
        load_pb_gz(str(source_path)),
        mapping_table=attention_policy_map(),
    )
    jax_model = make_bt4_model(mapped, dtype=jnp.bfloat16)
    jax_backend = jax.default_backend()
    jax_logits = np.asarray(
        jax.device_get(
            jax.block_until_ready(
                jit_bt4_forward(
                    jax_model,
                    jnp.asarray(numpy_batch["current_planes"], dtype=jnp.bfloat16),
                )[0]
            )
        ),
        dtype=np.float32,
    )
    del jax_model, mapped
    jax.clear_caches()
    gc.collect()

    if not torch.cuda.is_available():
        raise RuntimeError("verify-hero-init requires CUDA")
    device = torch.device("cuda")
    model, initialization = load_raw_bt4_hero_model(
        device=device,
        raw_bt4_path=args.raw_bt4_path,
        config=HERO_CONFIG,
    )
    model.eval()
    planes = torch.from_numpy(
        np.ascontiguousarray(numpy_batch["current_planes"])
    ).to(device)
    with torch.inference_mode():
        tokens = model.encoder.encode_current(
            planes,
            compute_dtype=torch.bfloat16,
            remat=False,
        )
        assert model.encoder.policy_head is not None
        torch_logits = (
            model.encoder.policy_head(tokens, torch.bfloat16)
            .float()
            .cpu()
            .numpy()
        )
    residual_zero = bool(
        torch.count_nonzero(model.out_proj).item() == 0
        and torch.count_nonzero(model.out_bias).item() == 0
    )

    legal_mask = np.zeros((args.batch_size, _VOCAB_SIZE), dtype=bool)
    legal_idx = np.asarray(numpy_batch["legal_idx"][:, 0], dtype=np.int64)
    legal_count = np.asarray(numpy_batch["legal_count"][:, 0], dtype=np.int64)
    for row, count in enumerate(legal_count):
        legal_mask[row, legal_idx[row, :count]] = True
    torch_actions = np.argmax(np.where(legal_mask, torch_logits, -np.inf), axis=-1)
    jax_actions = np.argmax(np.where(legal_mask, jax_logits, -np.inf), axis=-1)
    targets = np.asarray(numpy_batch["action_indices"][:, 0], dtype=np.int64)
    difference = torch_logits.astype(np.float64) - jax_logits.astype(np.float64)
    torch_norm = float(np.linalg.norm(torch_logits.astype(np.float64).reshape(-1)))
    jax_norm = float(np.linalg.norm(jax_logits.astype(np.float64).reshape(-1)))
    action_matches = torch_actions == jax_actions
    report = {
        "schema_version": "torch-hero-init-policy-parity-v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "git_commit": _git_commit(),
        "batch_size": args.batch_size,
        "seed": args.seed,
        "compute_dtype": "bfloat16",
        "jax_backend": jax_backend,
        "action_codec": HERO_CONFIG.action_codec,
        "raw_asset": initialization["raw_asset"],
        "raw_mapping_sha256": initialization["combined_sha256"],
        "zero_initialized_dfm_residual": residual_zero,
        "policy": {
            "max_absolute_error": float(np.max(np.abs(difference))),
            "relative_l2_error": float(
                np.linalg.norm(difference.reshape(-1)) / max(jax_norm, 1e-30)
            ),
            "cosine_similarity": float(
                np.vdot(
                    torch_logits.astype(np.float64).reshape(-1),
                    jax_logits.astype(np.float64).reshape(-1),
                )
                / max(torch_norm * jax_norm, 1e-30)
            ),
        },
        "legal_action_match_count": int(action_matches.sum()),
        "legal_action_match_fraction": float(action_matches.mean()),
        "mismatch_rows": np.flatnonzero(~action_matches).astype(int).tolist(),
        "torch_dataset_action_accuracy": float(np.mean(torch_actions == targets)),
        "jax_dataset_action_accuracy": float(np.mean(jax_actions == targets)),
        "gate_pass": bool(residual_zero and np.all(action_matches)),
    }
    _write_json(output, report)
    print(json.dumps(report, sort_keys=True), flush=True)
    if not report["gate_pass"]:
        raise RuntimeError("Update-zero raw-BT4 legal-action parity gate failed")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument(
        "--recipe",
        choices=("continuation", "hero"),
        default="continuation",
    )
    train_parser.add_argument(
        "--remat-mode",
        choices=(
            "all",
            "none",
            "bt4-only",
            "bt4-projector",
            "bt4-dfm",
            "heads-only",
        ),
        default="all",
    )
    train_parser.add_argument(
        "--attention-impl",
        choices=("manual", "sdpa-heads", "sdpa-all"),
        default="manual",
    )
    train_parser.add_argument(
        "--bt4-norm-impl",
        choices=(
            "eager",
            "eager-fused-backward",
            "native-fp32",
            "native-bf16",
        ),
        default="eager",
        help="BT4 LayerNorm runtime; eager is the frozen numerical reference.",
    )
    train_parser.add_argument(
        "--compile-regions",
        choices=(
            "none",
            "dfm-jepa",
            "fresh",
        ),
        default="none",
    )
    train_parser.add_argument("--source-state", type=Path, default=_SOURCE_STATE)
    train_parser.add_argument("--raw-bt4-path", type=Path, default=_RAW_BT4_PATH)
    train_parser.add_argument("--data-root", type=Path, default=_DATA_ROOT)
    train_parser.add_argument("--resume-checkpoint", type=Path)
    train_parser.add_argument("--output-dir", type=Path, required=True)
    train_parser.add_argument("--batch-size", type=int, default=64)
    train_parser.add_argument("--steps", type=int, default=1)
    train_parser.add_argument("--train-seconds", type=float, default=0.0)
    train_parser.add_argument("--data-start", type=int, default=0)
    train_parser.add_argument("--seed", type=int, default=0)
    train_parser.add_argument("--threads", type=int, default=2)
    train_parser.add_argument("--log-every", type=int, default=1)
    train_parser.add_argument("--prefetch-depth", type=int, default=1)
    train_parser.add_argument(
        "--prefetch-launch",
        choices=("step-start", "after-forward"),
        default="step-start",
        help=(
            "Schedule CPU preparation before the step or after forward so it "
            "overlaps the GPU-heavy backward pass."
        ),
    )
    train_parser.add_argument("--gpu-monitor-interval-ms", type=int, default=0)
    train_parser.add_argument(
        "--profile-update",
        type=int,
        default=0,
        help="Capture one bounded CPU/CUDA profiler window at this one-based update.",
    )
    train_parser.add_argument(
        "--hero-milestones",
        action="store_true",
        help=(
            "Enable the frozen one-epoch fast-validation and paired-Arena "
            "milestones without serializing intermediate model snapshots."
        ),
    )
    train_parser.add_argument(
        "--hero-eval-manifest",
        type=Path,
        default=_HERO_EVAL_MANIFEST,
    )
    train_parser.add_argument(
        "--hero-arena-pairs",
        type=int,
        default=_HERO_ARENA_PAIRS,
    )
    train_parser.add_argument(
        "--hero-arena-additional-ply-cap",
        type=int,
        default=_HERO_ARENA_ADDITIONAL_PLY_CAP,
    )
    train_parser.add_argument(
        "--hero-arena-inference-batch-size",
        type=int,
        default=_HERO_ARENA_INFERENCE_BATCH_SIZE,
    )
    train_parser.add_argument("--save-every", type=int, default=0)
    train_parser.add_argument("--save-updates", type=int, nargs="*", default=())
    train_parser.add_argument("--save-final", action="store_true")
    train_parser.add_argument("--max-checkpoints", type=int, default=1)
    train_parser.set_defaults(handler=train)

    lr_range_parser = subparsers.add_parser("lr-range")
    lr_range_parser.add_argument("--raw-bt4-path", type=Path, default=_RAW_BT4_PATH)
    lr_range_parser.add_argument("--data-root", type=Path, default=_DATA_ROOT)
    lr_range_parser.add_argument("--output-dir", type=Path, required=True)
    lr_range_parser.add_argument("--batch-size", type=int, default=1024)
    lr_range_parser.add_argument("--steps", type=int, default=128)
    lr_range_parser.add_argument("--data-start", type=int, default=0)
    lr_range_parser.add_argument("--seed", type=int, default=0)
    lr_range_parser.add_argument("--threads", type=int, default=2)
    lr_range_parser.add_argument("--log-every", type=int, default=8)
    lr_range_parser.add_argument("--prefetch-depth", type=int, default=1)
    lr_range_parser.add_argument(
        "--gpu-monitor-interval-ms",
        type=int,
        default=100,
    )
    lr_range_parser.add_argument(
        "--start-lr",
        dest="lr_range_start",
        type=float,
        default=1e-5,
    )
    lr_range_parser.add_argument(
        "--end-lr",
        dest="lr_range_end",
        type=float,
        default=1e-3,
    )
    lr_range_parser.set_defaults(
        handler=lr_range,
        recipe="hero",
        remat_mode="bt4-projector",
        attention_impl="sdpa-all",
        compile_regions="fresh",
        source_state=_SOURCE_STATE,
        resume_checkpoint=None,
        train_seconds=0.0,
        profile_update=0,
        save_every=0,
        save_updates=(),
        save_final=False,
        max_checkpoints=0,
    )

    hero_milestone_smoke_parser = subparsers.add_parser(
        "hero-milestone-smoke"
    )
    hero_milestone_smoke_parser.add_argument(
        "--raw-bt4-path",
        type=Path,
        default=_RAW_BT4_PATH,
    )
    hero_milestone_smoke_parser.add_argument(
        "--eval-manifest",
        type=Path,
        default=_HERO_EVAL_MANIFEST,
    )
    hero_milestone_smoke_parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )
    hero_milestone_smoke_parser.add_argument(
        "--threads",
        type=int,
        default=2,
    )
    hero_milestone_smoke_parser.add_argument(
        "--gpu-monitor-interval-ms",
        type=int,
        default=100,
    )
    hero_milestone_smoke_parser.set_defaults(
        handler=hero_milestone_smoke
    )

    hero_evaluate_parser = subparsers.add_parser("hero-evaluate")
    hero_evaluate_parser.add_argument(
        "--raw-bt4-path",
        type=Path,
        default=_RAW_BT4_PATH,
    )
    hero_evaluate_parser.add_argument(
        "--eval-manifest",
        type=Path,
        default=_HERO_EVAL_MANIFEST,
    )
    hero_evaluate_parser.add_argument(
        "--pool",
        choices=("fast", "primary", "blind"),
        required=True,
    )
    hero_evaluate_parser.add_argument("--checkpoint-dir", type=Path)
    hero_evaluate_parser.add_argument("--output-dir", type=Path, required=True)
    hero_evaluate_parser.add_argument("--eval-batch-size", type=int, default=64)
    hero_evaluate_parser.add_argument("--threads", type=int, default=2)
    hero_evaluate_parser.add_argument(
        "--gpu-monitor-interval-ms",
        type=int,
        default=100,
    )
    hero_evaluate_parser.add_argument(
        "--allow-blind-terminal",
        action="store_true",
    )
    hero_evaluate_parser.set_defaults(handler=evaluate_hero_pool)

    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("--source-state", type=Path, default=_SOURCE_STATE)
    evaluate_parser.add_argument("--data-root", type=Path, default=_DATA_ROOT)
    evaluate_parser.add_argument("--checkpoint-dir", type=Path)
    evaluate_parser.add_argument("--include-source", action="store_true")
    evaluate_parser.add_argument("--output-dir", type=Path, required=True)
    evaluate_parser.add_argument(
        "--eval-seeds",
        type=int,
        nargs="+",
        default=(10000, 20000),
    )
    evaluate_parser.add_argument("--eval-batch-size", type=int, default=64)
    evaluate_parser.add_argument("--eval-batches", type=int, default=64)
    evaluate_parser.add_argument("--threads", type=int, default=2)
    evaluate_parser.add_argument("--gpu-monitor-interval-ms", type=int, default=100)
    evaluate_parser.set_defaults(handler=evaluate_checkpoint)

    inspect_parser = subparsers.add_parser("inspect-source")
    inspect_parser.add_argument("--source-state", type=Path, default=_SOURCE_STATE)
    inspect_parser.add_argument(
        "--output",
        type=Path,
        default=_REPO_ROOT / "artifacts" / "pytorch" / "source_mapping.json",
    )
    inspect_parser.set_defaults(handler=inspect_source)

    smoke_parser = subparsers.add_parser("cpu-smoke")
    smoke_parser.add_argument("--source-state", type=Path, default=_SOURCE_STATE)
    smoke_parser.add_argument("--data-root", type=Path, default=_DATA_ROOT)
    smoke_parser.add_argument("--batch-size", type=int, default=1)
    smoke_parser.add_argument("--data-step", type=int, default=0)
    smoke_parser.add_argument("--update", type=int, default=0)
    smoke_parser.add_argument("--seed", type=int, default=0)
    smoke_parser.add_argument("--threads", type=int, default=2)
    smoke_parser.add_argument("--backward", action="store_true")
    smoke_parser.set_defaults(handler=cpu_smoke)

    hero_verify_parser = subparsers.add_parser("verify-hero-init")
    hero_verify_parser.add_argument("--raw-bt4-path", type=Path, default=_RAW_BT4_PATH)
    hero_verify_parser.add_argument("--data-root", type=Path, default=_DATA_ROOT)
    hero_verify_parser.add_argument("--batch-size", type=int, default=8)
    hero_verify_parser.add_argument("--seed", type=int, default=31_415)
    hero_verify_parser.add_argument("--threads", type=int, default=2)
    hero_verify_parser.add_argument(
        "--output",
        type=Path,
        default=_REPO_ROOT / "artifacts" / "pytorch" / "hero_init_policy_parity.json",
    )
    hero_verify_parser.set_defaults(handler=verify_hero_init)

    loss_audit_parser = subparsers.add_parser("loss-audit")
    loss_audit_parser.add_argument(
        "--raw-bt4-path",
        type=Path,
        default=_RAW_BT4_PATH,
    )
    loss_audit_parser.add_argument("--data-root", type=Path, default=_DATA_ROOT)
    loss_audit_parser.add_argument("--batch-size", type=int, default=512)
    loss_audit_parser.add_argument("--data-step", type=int, default=0)
    loss_audit_parser.add_argument("--seed", type=int, default=0)
    loss_audit_parser.add_argument("--threads", type=int, default=2)
    loss_audit_parser.add_argument(
        "--compile-regions",
        choices=("none", "dfm-jepa", "fresh"),
        default="fresh",
    )
    loss_audit_parser.add_argument(
        "--root-target-contribution",
        type=float,
        default=0.25,
    )
    loss_audit_parser.add_argument(
        "--root-calibration-metrics",
        type=Path,
        help=(
            "Optional matched update-zero metrics.jsonl used to avoid an "
            "extra calibration forward at memory-saturating batch sizes."
        ),
    )
    loss_audit_parser.add_argument(
        "--root-calibration-batch-size",
        type=int,
        default=1024,
    )
    loss_audit_parser.add_argument(
        "--output",
        type=Path,
        default=(
            _REPO_ROOT
            / "artifacts"
            / "profiles"
            / "hero_loss_gradient_audit_b512.json"
        ),
    )
    loss_audit_parser.set_defaults(handler=loss_gradient_audit)

    sigreg_sample_parser = subparsers.add_parser("sigreg-sample-audit")
    sigreg_sample_parser.add_argument(
        "--raw-bt4-path",
        type=Path,
        default=_RAW_BT4_PATH,
    )
    sigreg_sample_parser.add_argument(
        "--data-root",
        type=Path,
        default=_DATA_ROOT,
    )
    sigreg_sample_parser.add_argument("--checkpoint-dir", type=Path)
    sigreg_sample_parser.add_argument("--batch-size", type=int, default=1024)
    sigreg_sample_parser.add_argument(
        "--sample-counts",
        type=int,
        nargs="+",
        default=(64, 128, 256),
    )
    sigreg_sample_parser.add_argument("--replicates", type=int, default=8)
    sigreg_sample_parser.add_argument("--data-step", type=int, default=0)
    sigreg_sample_parser.add_argument("--update", type=int, default=0)
    sigreg_sample_parser.add_argument("--seed", type=int, default=0)
    sigreg_sample_parser.add_argument("--threads", type=int, default=2)
    sigreg_sample_parser.add_argument(
        "--compile-regions",
        choices=("none", "dfm-jepa", "fresh"),
        default="fresh",
    )
    sigreg_sample_parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )
    sigreg_sample_parser.set_defaults(handler=sigreg_sample_audit)

    runtime_parity_parser = subparsers.add_parser("runtime-parity")
    runtime_parity_parser.add_argument(
        "--raw-bt4-path",
        type=Path,
        default=_RAW_BT4_PATH,
    )
    runtime_parity_parser.add_argument("--data-root", type=Path, default=_DATA_ROOT)
    runtime_parity_parser.add_argument("--batch-size", type=int, default=512)
    runtime_parity_parser.add_argument("--data-step", type=int, default=0)
    runtime_parity_parser.add_argument("--seed", type=int, default=0)
    runtime_parity_parser.add_argument("--threads", type=int, default=2)
    runtime_parity_parser.add_argument(
        "--candidate-compile-regions",
        choices=(
            "fresh",
            "fresh-bt4-smolgen",
            "fresh-bt4-smolgen-eager-numerics",
            "fresh-bt4",
            "fresh-bt4-eager-numerics",
            "fresh-bt4-strict-numerics",
        ),
        default="fresh-bt4-strict-numerics",
    )
    runtime_parity_parser.add_argument(
        "--candidate-bt4-norm-impl",
        choices=(
            "eager",
            "eager-fused-backward",
            "native-fp32",
            "native-bf16",
        ),
        default="eager",
    )
    runtime_parity_parser.add_argument(
        "--output",
        type=Path,
        default=(
            _REPO_ROOT
            / "artifacts"
            / "profiles"
            / "hero_compiled_bt4_strict_numerics_runtime_parity_b512.json"
        ),
    )
    runtime_parity_parser.set_defaults(handler=runtime_parity)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
