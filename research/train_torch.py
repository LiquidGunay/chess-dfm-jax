#!/usr/bin/env python3
"""One-file eager-PyTorch BT4/DFM/JEPA autoresearch trainer.

This file intentionally keeps the production model, objective, optimizer, and
training loop together.  The initial migration uses ordinary eager PyTorch:
there is no ``torch.compile`` call in this file.  JAX remains the frozen
numerical/checkpoint/evaluation oracle until every migration gate passes.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import gc
import hashlib
import hmac
import json
import math
import os
import pickle
import stat
import subprocess
import sys
import time
import zipfile
from collections.abc import Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import IO, Any, NamedTuple

import ml_dtypes
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
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
HERO_CONFIG = dataclasses.replace(
    CONFIG,
    action_codec="lc0_canonical_1858",
    use_bt4_policy_residual=True,
    root_legal_ce_coeff=0.0,
    wdl_coeff=0.25,
    target_sigreg_coeff=2.0,
    pred_sigreg_coeff=2.0,
    loss_clip_value=0.0,
    learning_rate=3e-4,
    bt4_learning_rate=1e-5,
    weight_decay=1e-2,
    selective_weight_decay=True,
    lr_schedule_unit="examples",
    lr_warmup_examples=round(0.02 * _HERO_TRAIN_EXAMPLES),
    lr_total_examples=_HERO_TRAIN_EXAMPLES,
    lr_min_ratio=1e-3,
)


def _require_workspace(path: str | os.PathLike[str], *, exists: bool = False) -> Path:
    resolved = Path(path).expanduser().resolve(strict=exists)
    try:
        resolved.relative_to(_WORKSPACE_ROOT.resolve(strict=True))
    except ValueError as exc:
        raise ValueError(f"Path escapes {_WORKSPACE_ROOT}: {resolved}") from exc
    return resolved


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


class RawLayerNorm(nn.Module):
    def __init__(self, width: int, *, dtype: torch.dtype, eps: float = 1e-3):
        super().__init__()
        self.scale = _raw_parameter((width,), dtype)
        self.bias = _raw_parameter((width,), dtype)
        self.eps = float(eps)

    def forward(self, x: Tensor, compute_dtype: torch.dtype) -> Tensor:
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
    def __init__(self):
        super().__init__()
        dtype = torch.bfloat16
        self.preproc = RawLinear(768, 32768, dtype=dtype)
        self.proj = RawLinear(624, 1024, dtype=dtype)
        self.ln = RawLayerNorm(1024, dtype=dtype)
        self.mul_gate = _raw_parameter((64, 1024), dtype)
        self.add_gate = _raw_parameter((64, 1024), dtype)
        self.ffn1 = RawLinear(1024, 1536, dtype=dtype)
        self.ffn2 = RawLinear(1536, 1024, dtype=dtype)
        self.ffn_ln = RawLayerNorm(1024, dtype=dtype)

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
    def __init__(self):
        super().__init__()
        dtype = torch.bfloat16
        self.compress = RawLinear(1024, 32, dtype=dtype, bias=False)
        self.dense1 = RawLinear(2048, 256, dtype=dtype)
        self.ln1 = RawLayerNorm(256, dtype=dtype)
        self.dense2 = RawLinear(256, 8192, dtype=dtype)
        self.ln2 = RawLayerNorm(8192, dtype=dtype)
        self.shared_w = _raw_parameter((256, 4096), dtype)

    def forward(self, x: Tensor, compute_dtype: torch.dtype) -> Tensor:
        batch = x.shape[0]
        value = self.compress(x, compute_dtype).reshape(batch, 2048)
        value = self.ln1(F.silu(self.dense1(value, compute_dtype)), compute_dtype)
        value = self.ln2(F.silu(self.dense2(value, compute_dtype)), compute_dtype)
        value = value.reshape(batch, 32, 256)
        return (value @ self.shared_w.to(compute_dtype)).reshape(batch, 32, 64, 64)


class BT4EncoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        dtype = torch.bfloat16
        self.wq = _raw_parameter((1024, 1024), dtype)
        self.wq_b = _raw_parameter((1024,), dtype)
        self.wk = _raw_parameter((1024, 1024), dtype)
        self.wk_b = _raw_parameter((1024,), dtype)
        self.wv = _raw_parameter((1024, 1024), dtype)
        self.wv_b = _raw_parameter((1024,), dtype)
        self.wo = RawLinear(1024, 1024, dtype=dtype)
        self.ln_attn = RawLayerNorm(1024, dtype=dtype)
        self.ffn1 = RawLinear(1024, 1536, dtype=dtype)
        self.ffn2 = RawLinear(1536, 1024, dtype=dtype)
        self.ln_ffn = RawLayerNorm(1024, dtype=dtype)
        self.smolgen = BT4Smolgen()

    def forward(self, x: Tensor, alpha: float, compute_dtype: torch.dtype) -> Tensor:
        batch, sequence, _ = x.shape
        q = x @ self.wq.to(compute_dtype) + self.wq_b.to(compute_dtype)
        k = x @ self.wk.to(compute_dtype) + self.wk_b.to(compute_dtype)
        v = x @ self.wv.to(compute_dtype) + self.wv_b.to(compute_dtype)
        q = q.reshape(batch, sequence, 32, 32).transpose(1, 2)
        k = k.reshape(batch, sequence, 32, 32).transpose(1, 2)
        v = v.reshape(batch, sequence, 32, 32).transpose(1, 2)
        logits = (q @ k.transpose(-2, -1)) / math.sqrt(32.0)
        logits = logits + self.smolgen(x, compute_dtype)
        attention = F.softmax(logits, dim=-1)
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
    def __init__(self, *, include_policy_head: bool = False):
        super().__init__()
        self.embedding = BT4InputEmbedding()
        self.layers = nn.ModuleList(BT4EncoderLayer() for _ in range(15))
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
            remat=config.remat_blocks,
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
    def __init__(self, config: Config = CONFIG):
        super().__init__()
        self.config = config
        dtype = torch.float32
        self.encoder = BT4Encoder(
            include_policy_head=config.use_bt4_policy_residual,
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
            remat=config.remat_blocks,
        )
        self.dfm_out_norm = RawRMSNorm(config.token_dim, dtype=dtype)
        self.out_proj = _raw_parameter((config.token_dim, _VOCAB_SIZE), dtype)
        self.out_bias = _raw_parameter((_VOCAB_SIZE,), dtype)

    def encode_selected(
        self,
        current_planes: Tensor,
        selected_future_planes: Tensor,
        compute_dtype: torch.dtype,
    ) -> tuple[Tensor, Tensor, Tensor | None]:
        current = self.encoder.encode_current(
            current_planes,
            compute_dtype=compute_dtype,
            remat=self.config.remat_blocks,
        )
        future = self.encoder.encode_future_tail(
            selected_future_planes,
            compute_dtype=compute_dtype,
            trainable_tail_layers=self.config.future_trainable_tail_layers,
        )
        tokens = torch.stack((current, future), dim=1)
        batch = current.shape[0]
        z_all = self.state_projector(tokens.reshape(batch * 2, 64, 1024), compute_dtype).reshape(
            batch, 2, self.config.z_dim
        )
        z_dfm = self.dfm_state_projector(current, compute_dtype)
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
    )
    z_jepa = z_all[:, 0]
    target_z = z_all[:, 1:]

    t = choices.training_time
    is_masked = choices.mask_uniform < (1.0 - t).unsqueeze(1)
    noisy_actions = torch.where(is_masked, torch.full_like(actions, _MASK_TOKEN), actions)
    logits = model.planner(
        z_dfm,
        noisy_actions,
        t,
        compute_dtype,
        base_root_logits=base_policy_logits,
    )
    assert isinstance(logits, Tensor)
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
    pred_z = model.jepa_rollout(z_jepa, actions, clean_hidden, compute_dtype)
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

    pred_mean = (pred_f32 * latent_weight.unsqueeze(-1)).sum(dim=0) / latent_denom.unsqueeze(-1)
    target_mean = (target_f32 * latent_weight.unsqueeze(-1)).sum(dim=0) / latent_denom.unsqueeze(-1)
    pred_variance = (
        (pred_f32 - pred_mean.unsqueeze(0)).square() * latent_weight.unsqueeze(-1)
    ).sum(dim=0) / latent_denom.unsqueeze(-1)
    target_variance = (
        (target_f32 - target_mean.unsqueeze(0)).square() * latent_weight.unsqueeze(-1)
    ).sum(dim=0) / latent_denom.unsqueeze(-1)
    pred_feature_std = torch.sqrt(pred_variance.clamp_min(0.0) + 1e-12)
    target_feature_std = torch.sqrt(target_variance.clamp_min(0.0) + 1e-12)

    centered = (pred_f32 - pred_mean.unsqueeze(0)).permute(1, 0, 2)
    centered = centered * latent_weight.transpose(0, 1).sqrt().unsqueeze(-1)
    covariance_denom = (latent_denom - 1.0).clamp_min(1.0)
    gram = centered @ centered.transpose(-2, -1)
    gram = gram / covariance_denom[:, None, None]
    eigenvalues = torch.linalg.eigvalsh(gram).clamp_min(0.0)
    eigenvalue_sum = eigenvalues.sum(dim=-1, keepdim=True)
    spectrum = eigenvalues / eigenvalue_sum.clamp_min(1e-12)
    entropy = -torch.where(
        spectrum > 0.0,
        spectrum * torch.log(spectrum),
        torch.zeros_like(spectrum),
    ).sum(dim=-1)
    effective_rank = torch.where(
        eigenvalue_sum[:, 0] > 1e-12,
        torch.exp(entropy),
        torch.zeros_like(entropy),
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
        "pred_rms_by_horizon": torch.sqrt(horizon_mean(pred_f32.square().mean(dim=-1))),
        "target_rms_by_horizon": torch.sqrt(horizon_mean(target_f32.square().mean(dim=-1))),
        "pred_feature_std_mean_by_horizon": pred_feature_std.mean(dim=-1),
        "pred_feature_std_p05_by_horizon": torch.quantile(
            pred_feature_std,
            0.05,
            dim=-1,
        ),
        "target_feature_std_mean_by_horizon": target_feature_std.mean(dim=-1),
        "pred_effective_rank_by_horizon": effective_rank,
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
        peak = self.config.bt4_learning_rate if kind == "bt4" else self.config.learning_rate
        return peak * self.learning_rate_ratio()

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

    model = JointModel(config)
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
    }


_TRAJECTORY_METADATA_KEYS = frozenset(
    {"source", "game_id", "ply", "result", "fen_t", "input_format", "actions_uci"}
)


def _metadata_text(value: Any) -> str:
    scalar = np.asarray(value).item()
    return scalar.decode("utf-8") if isinstance(scalar, bytes) else str(scalar)


def canonicalize_trajectory_batch(
    batch: Mapping[str, Any],
) -> dict[str, Any]:
    """Re-encode actions and legal sets in the complete side-to-move LC0 frame."""

    import chess

    from chess_dfm_jax.policy import (
        LC0_CANONICAL_1858_INPUT_FORMAT,
        encode_lc0_canonical_1858,
        legal_mask_lc0_canonical_1858,
    )

    required = {"fen_t", "input_format", "actions_uci", "future_valid", "legal_idx"}
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
        board = chess.Board(_metadata_text(fens if fens.ndim == 0 else fens[row]))
        for offset in range(horizon):
            if future_valid[row, offset] <= 0.0:
                continue
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
            if legal.size > legal_capacity:
                raise ValueError(
                    f"Canonical legal set at row {row}, horizon {offset + 1} "
                    f"needs {legal.size} slots; shard capacity is {legal_capacity}"
                )
            if not np.any(legal == action):
                raise RuntimeError(
                    f"Canonical target {action} is absent from its legal set at "
                    f"row {row}, horizon {offset + 1}"
                )
            canonical_actions[row, offset] = action
            canonical_legal[row, offset, : legal.size] = legal
            canonical_count[row, offset] = legal.size
            canonical_valid[row, offset] = 1.0
            board.push(move)

    result["action_indices"] = canonical_actions
    result["action_idx"] = canonical_actions[:, 0]
    result["legal_idx"] = canonical_legal
    result["legal_count"] = canonical_count
    result["legal_masks_valid"] = canonical_valid
    return result


from research.prepare import FixedTrajectoryBatches as _FixedTrajectoryBatches


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


def _start_gpu_monitor(
    output_dir: Path,
    *,
    interval_ms: int,
) -> tuple[subprocess.Popen[str], IO[str], IO[str]]:
    samples = (output_dir / "gpu_samples.csv").open("w", encoding="utf-8")
    stderr = (output_dir / "gpu_monitor.stderr.log").open("w", encoding="utf-8")
    fields = (
        "timestamp,utilization.gpu,utilization.memory,memory.used,memory.total,"
        "power.draw,clocks.sm,clocks.mem"
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


def train(args: argparse.Namespace) -> int:
    config = HERO_CONFIG if args.recipe == "hero" else CONFIG
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
    if args.save_every != 0 or args.save_updates:
        raise ValueError(
            "Periodic/sparse checkpoints are disabled during migration; use at most --save-final"
        )
    if args.max_checkpoints not in (0, 1):
        raise ValueError("--max-checkpoints must be 0 or 1 for PyTorch migration")

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
    optimizer = MuonAdamW(
        model,
        config,
        examples_per_update=(
            args.batch_size if config.lr_schedule_unit == "examples" else None
        ),
    )
    restore_seconds = time.perf_counter() - restore_started
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
    run_config = {
        "schema_version": "torch-eager-train-run-v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "git_commit": _git_commit(),
        "framework": "torch",
        "execution": "eager",
        "torch_compile": False,
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
        "data": batches.provenance(),
        "prefetch": {
            "depth": args.prefetch_depth,
            "workers": 1 if args.prefetch_depth == 1 else 0,
            "deterministic_update_and_cursor_keys": True,
        },
        "restore_seconds": restore_seconds,
    }
    _write_json(output_dir / "run_config.json", run_config)

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    run_started = time.perf_counter()
    deadline = run_started + args.train_seconds if args.train_seconds > 0 else None
    records: list[dict[str, Any]] = []
    update = 0
    data_cursor = args.data_start
    monitor = (
        _start_gpu_monitor(
            output_dir,
            interval_ms=args.gpu_monitor_interval_ms,
        )
        if args.gpu_monitor_interval_ms > 0
        else None
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
            if prefetch_executor is not None and more_steps and before_deadline:
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

            transfer_started = time.perf_counter()
            batch = _torch_batch(compact_batch, device)
            choices = _choices_to_device(choices_cpu, device)
            torch.cuda.synchronize()
            transfer_seconds = time.perf_counter() - transfer_started

            step_started = time.perf_counter()
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
            )
            event_forward.record()
            loss.backward()
            event_backward.record()
            optimizer_metrics = optimizer.step()
            event_optimizer.record()
            torch.cuda.synchronize()
            step_seconds = time.perf_counter() - step_started
            forward_cuda_seconds = event_start.elapsed_time(event_forward) / 1000.0
            backward_cuda_seconds = event_forward.elapsed_time(event_backward) / 1000.0
            optimizer_cuda_seconds = event_backward.elapsed_time(event_optimizer) / 1000.0
            update += 1
            data_cursor += 1
            elapsed = time.perf_counter() - run_started
            record = {
                "schema_version": "torch-eager-train-metrics-v1",
                "update": update,
                "optimizer_update": int(optimizer_metrics["optimizer_update"]),
                "data_cursor": data_cursor,
                "examples": update * args.batch_size,
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
                "examples_per_second_end_to_end": (update * args.batch_size / elapsed),
                "gpu_memory_allocated_bytes": torch.cuda.memory_allocated(),
                "gpu_memory_reserved_bytes": torch.cuda.memory_reserved(),
                "gpu_peak_memory_allocated_bytes": (torch.cuda.max_memory_allocated()),
                "gpu_peak_memory_reserved_bytes": (torch.cuda.max_memory_reserved()),
                **_json_scalars(aux),
                **optimizer_metrics,
            }
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
    finally:
        if prepared_future is not None:
            prepared_future.cancel()
        if prefetch_executor is not None:
            prefetch_executor.shutdown(wait=True, cancel_futures=True)
        _stop_gpu_monitor(monitor)

    torch.cuda.synchronize()
    train_seconds = time.perf_counter() - run_started
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
    report = {
        "schema_version": "torch-eager-train-report-v1",
        "completed_utc": datetime.now(UTC).isoformat(),
        "updates": update,
        "examples": update * args.batch_size,
        "next_data_cursor": data_cursor,
        "train_seconds": train_seconds,
        "examples_per_second_end_to_end": (update * args.batch_size / max(train_seconds, 1e-12)),
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
        "gpu_peak_memory_allocated_bytes": torch.cuda.max_memory_allocated(),
        "gpu_peak_memory_reserved_bytes": torch.cuda.max_memory_reserved(),
        "last_metrics": records[-1],
        "loss_summary": loss_summary,
        "checkpoint": checkpoint_manifest,
    }
    _write_json(output_dir / "report.json", report)
    print(json.dumps(report, sort_keys=True), flush=True)
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
    with torch.inference_mode():
        for index in range(count):
            numpy_batch = batches.batch_at(index)
            batch = _torch_batch(numpy_batch, device)
            choices = materialize_step_choices(
                seed=seed,
                update=index,
                batch_size=batches.batch_size,
                config=CONFIG,
                device=device,
            )
            permutation_rng = np.random.Generator(
                np.random.PCG64(np.random.SeedSequence([int(seed), int(index), 0xC011A95E]))
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
                        f"Non-finite validation metric {key} at seed={seed}, batch={index}: {value}"
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
    return (
        {key: value / count for key, value in totals.items()},
        time.perf_counter() - started,
    )


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
    train_parser.add_argument("--source-state", type=Path, default=_SOURCE_STATE)
    train_parser.add_argument("--raw-bt4-path", type=Path, default=_RAW_BT4_PATH)
    train_parser.add_argument("--data-root", type=Path, default=_DATA_ROOT)
    train_parser.add_argument("--output-dir", type=Path, required=True)
    train_parser.add_argument("--batch-size", type=int, default=64)
    train_parser.add_argument("--steps", type=int, default=1)
    train_parser.add_argument("--train-seconds", type=float, default=0.0)
    train_parser.add_argument("--data-start", type=int, default=0)
    train_parser.add_argument("--seed", type=int, default=0)
    train_parser.add_argument("--threads", type=int, default=2)
    train_parser.add_argument("--log-every", type=int, default=1)
    train_parser.add_argument("--prefetch-depth", type=int, default=1)
    train_parser.add_argument("--gpu-monitor-interval-ms", type=int, default=0)
    train_parser.add_argument("--save-every", type=int, default=0)
    train_parser.add_argument("--save-updates", type=int, nargs="*", default=())
    train_parser.add_argument("--save-final", action="store_true")
    train_parser.add_argument("--max-checkpoints", type=int, default=1)
    train_parser.set_defaults(handler=train)

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
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
