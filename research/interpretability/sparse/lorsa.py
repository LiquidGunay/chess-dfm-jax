"""LoRSA dynamic sparse attention and a fail-closed conversion boundary."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from torch import Tensor, nn
from torch.nn import functional as F

from research.interpretability.artifacts import sha256_file
from research.interpretability.sparse.transcoder import positive_topk


CONVERTED_LORSA_SCHEMA = "bt4-lorsa-converted-safetensors-v1"


@dataclass(frozen=True)
class LoRSAConfig:
    d_model: int = 1024
    n_ctx: int = 64
    n_qk_heads: int = 128
    d_qk_head: int = 32
    n_ov_heads: int = 16_384
    top_k: int = 30
    attn_scale: float = math.sqrt(32)
    use_smolgen: bool = True
    smolgen_score_scale: float = 1.0
    use_learnable_attn_scale: bool = False
    use_decoder_bias: bool = True

    def validate(self) -> None:
        integer_values = (
            self.d_model,
            self.n_ctx,
            self.n_qk_heads,
            self.d_qk_head,
            self.n_ov_heads,
            self.top_k,
        )
        if any(type(value) is not int or value <= 0 for value in integer_values):
            raise ValueError("LoRSA dimensions must be positive integers")
        if self.n_ov_heads % self.n_qk_heads:
            raise ValueError("LoRSA OV heads must divide evenly across QK heads")
        if self.top_k > self.n_ov_heads:
            raise ValueError("LoRSA top_k exceeds the OV feature count")
        if self.attn_scale <= 0 or not math.isfinite(self.attn_scale):
            raise ValueError("LoRSA attention scale must be finite and positive")
        if not math.isfinite(self.smolgen_score_scale):
            raise ValueError("LoRSA SmolGen score scale must be finite")


class LoRSASmolgen(nn.Module):
    def __init__(
        self,
        config: LoRSAConfig,
        *,
        device: torch.device | str | None,
        dtype: torch.dtype,
    ):
        super().__init__()
        factory = {"device": device, "dtype": dtype}
        self.n_ctx = config.n_ctx
        self.n_qk_heads = config.n_qk_heads
        self.compress = nn.Linear(config.d_model, 32, bias=False, **factory)
        self.dense1 = nn.Linear(32 * config.n_ctx, 256, **factory)
        self.ln1 = nn.LayerNorm(256, eps=1e-5, **factory)
        self.dense2 = nn.Linear(256, config.n_qk_heads * 256, **factory)
        self.ln2 = nn.LayerNorm(config.n_qk_heads * 256, eps=1e-5, **factory)
        # Keep the published Leela-SAEs state-dict name. ``weight_generator``
        # remains a read-only compatibility alias below.
        self.smol_weight_gen = nn.Linear(256, config.n_ctx * config.n_ctx, bias=False, **factory)

    @property
    def weight_generator(self) -> nn.Linear:
        return self.smol_weight_gen

    def forward(self, states: Tensor) -> Tensor:
        batch = states.shape[0]
        value = self.compress(states).reshape(batch, 32 * self.n_ctx)
        value = self.ln1(F.silu(self.dense1(value)))
        value = self.ln2(F.silu(self.dense2(value)))
        value = value.reshape(batch * self.n_qk_heads, 256)
        return self.smol_weight_gen(value).reshape(
            batch,
            self.n_qk_heads,
            self.n_ctx,
            self.n_ctx,
        )


@dataclass(frozen=True)
class LoRSAOutput:
    query: Tensor
    key: Tensor
    value: Tensor
    qk_scores: Tensor
    smolgen_bias: Tensor | None
    pattern: Tensor
    hidden_pre: Tensor
    decoder_weighted_scores: Tensor
    features: Tensor
    reconstruction: Tensor


class LowRankSparseAttention(nn.Module):
    """Sparse dynamic Q/K/OV replacement for BT4's full attention branch."""

    def __init__(
        self,
        config: LoRSAConfig,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        config.validate()
        self.config = config
        factory = {"device": device, "dtype": dtype}
        self.W_Q = nn.Parameter(
            torch.empty((config.n_qk_heads, config.d_model, config.d_qk_head), **factory)
        )
        self.W_K = nn.Parameter(
            torch.empty((config.n_qk_heads, config.d_model, config.d_qk_head), **factory)
        )
        self.W_V = nn.Parameter(torch.empty((config.n_ov_heads, config.d_model), **factory))
        self.W_O = nn.Parameter(torch.empty((config.n_ov_heads, config.d_model), **factory))
        self.b_Q = nn.Parameter(torch.empty((config.n_qk_heads, config.d_qk_head), **factory))
        self.b_K = nn.Parameter(torch.empty((config.n_qk_heads, config.d_qk_head), **factory))
        self.b_V = nn.Parameter(torch.empty((config.n_ov_heads,), **factory))
        if config.use_decoder_bias:
            self.b_D = nn.Parameter(torch.empty((config.d_model,), **factory))
        else:
            self.register_parameter("b_D", None)
        if config.use_learnable_attn_scale:
            self._attn_scale_param = nn.Parameter(
                torch.tensor(config.attn_scale, device=device, dtype=dtype)
            )
        else:
            self.register_parameter("_attn_scale_param", None)
        if config.use_smolgen:
            self.smolgen = LoRSASmolgen(config, device=device, dtype=dtype)
            self.register_buffer(
                "smolgen_score_scale",
                torch.tensor(config.smolgen_score_scale, device=device, dtype=dtype),
            )
        else:
            self.smolgen = None

    @property
    def decoder_norm(self) -> Tensor:
        return self.W_O.norm(dim=1)

    @property
    def attention_scale(self) -> float | Tensor:
        if self._attn_scale_param is not None:
            return self._attn_scale_param
        return self.config.attn_scale

    @staticmethod
    def _replace(name: str, native: Tensor, override: Tensor | None) -> Tensor:
        if override is None:
            return native
        if override.shape != native.shape or override.dtype != native.dtype:
            raise ValueError(f"LoRSA {name} override shape or dtype differs from native")
        if override.device != native.device or not bool(torch.isfinite(override).all()):
            raise ValueError(f"LoRSA {name} override device differs or contains nonfinite values")
        return override

    def forward(
        self,
        states: Tensor,
        *,
        query_override: Tensor | None = None,
        key_override: Tensor | None = None,
        value_override: Tensor | None = None,
        pattern_override: Tensor | None = None,
        feature_override: Tensor | None = None,
    ) -> LoRSAOutput:
        cfg = self.config
        if states.ndim != 3 or states.shape[1:] != (cfg.n_ctx, cfg.d_model):
            raise ValueError("LoRSA requires the complete ordered [batch, n_ctx, d_model] sequence")
        query = torch.einsum("bsd,hdq->bshq", states, self.W_Q) + self.b_Q
        key = torch.einsum("bsd,hdk->bshk", states, self.W_K) + self.b_K
        value = torch.einsum("bsd,vd->bsv", states, self.W_V) + self.b_V
        query = self._replace("query", query, query_override)
        key = self._replace("key", key, key_override)
        value = self._replace("value", value, value_override)
        qk_scores = torch.einsum("bqhd,bkhd->bhqk", query, key) / self.attention_scale
        smolgen_bias = self.smolgen(states) if self.smolgen is not None else None
        scores = qk_scores
        if smolgen_bias is not None:
            scores = scores + self.smolgen_score_scale * smolgen_bias
        pattern = torch.softmax(scores, dim=-1)
        pattern = self._replace("pattern", pattern, pattern_override)
        if pattern_override is not None:
            valid_pattern = bool((pattern >= 0).all()) and bool(
                torch.allclose(
                    pattern.sum(dim=-1),
                    torch.ones_like(pattern[..., 0]),
                    rtol=1e-4,
                    atol=1e-5,
                )
            )
            if not valid_pattern:
                raise ValueError("LoRSA pattern override must be non-negative and row-normalized")
        group_count = cfg.n_ov_heads // cfg.n_qk_heads
        grouped_value = value.reshape(value.shape[0], cfg.n_ctx, cfg.n_qk_heads, group_count)
        hidden_grouped = torch.einsum("bhqk,bkhg->bqhg", pattern, grouped_value)
        hidden_pre = hidden_grouped.reshape(states.shape[0], cfg.n_ctx, cfg.n_ov_heads)
        decoder_norm = self.decoder_norm
        if bool((decoder_norm <= 0).any()):
            raise ValueError("LoRSA decoder has a zero-norm feature")
        weighted = hidden_pre * decoder_norm
        features = positive_topk(weighted, cfg.top_k, mode="upstream_threshold") / decoder_norm
        features = self._replace("features", features, feature_override)
        reconstruction = features @ self.W_O
        if self.b_D is not None:
            reconstruction = reconstruction + self.b_D
        return LoRSAOutput(
            query=query,
            key=key,
            value=value,
            qk_scores=qk_scores,
            smolgen_bias=smolgen_bias,
            pattern=pattern,
            hidden_pre=hidden_pre,
            decoder_weighted_scores=weighted,
            features=features,
            reconstruction=reconstruction,
        )


def initialize_lorsa(model: LowRankSparseAttention, *, seed: int) -> None:
    if next(model.parameters()).device.type == "meta":
        raise ValueError("Cannot initialize a meta-device LoRSA")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name == "_attn_scale_param":
                parameter.fill_(model.config.attn_scale)
                continue
            parameter.copy_(
                (0.02 * torch.randn(parameter.shape, generator=generator)).to(parameter)
            )
        model.W_O.div_(model.W_O.norm(dim=1, keepdim=True).clamp_min(1e-12))
        if model.b_D is not None:
            model.b_D.zero_()


def fold_lorsa_dataset_normalization(
    state: dict[str, Tensor],
    *,
    input_norm: float,
    output_norm: float,
    d_model: int,
) -> dict[str, Tensor]:
    """Apply the pinned upstream LoRSA dataset-normalization inference fold."""

    if input_norm <= 0 or output_norm <= 0 or not math.isfinite(input_norm + output_norm):
        raise ValueError("LoRSA dataset norms must be finite and positive")
    c_in = math.sqrt(d_model) / input_norm
    c_out = math.sqrt(d_model) / output_norm
    result = {name: value.clone() for name, value in state.items()}
    for name in ("W_Q", "W_K", "W_V"):
        result[name] = result[name] * c_in
    result["W_O"] = result["W_O"] / c_out
    if "b_D" in result:
        result["b_D"] = result["b_D"] / c_out
    return result


def inspect_untrusted_lorsa_dcp(directory: Path) -> dict[str, Any]:
    """Inventory a DCP directory without opening its pickle-based metadata."""

    root = directory.resolve(strict=True)
    if not root.is_dir():
        raise ValueError("LoRSA DCP path must be a directory")
    files = sorted(path for path in root.rglob("*") if path.is_file())
    if not files:
        raise ValueError("LoRSA DCP directory is empty")
    metadata = [path for path in files if path.name == ".metadata"]
    return {
        "schema_version": "bt4-untrusted-lorsa-dcp-inventory-v1",
        "path": str(root),
        "file_count": len(files),
        "files": {
            path.relative_to(root).as_posix(): {
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in files
        },
        "pickle_metadata_present": bool(metadata),
        "pickle_loaded": False,
        "direct_load_allowed": False,
        "required_next_step": (
            "Review and deserialize only in an isolated pinned conversion environment, then "
            "export safetensors plus a provenance manifest."
        ),
    }


@dataclass(frozen=True)
class LoadedConvertedLoRSA:
    module: LowRankSparseAttention
    manifest: dict[str, Any]


def load_converted_lorsa(manifest_path: Path, weights_path: Path) -> LoadedConvertedLoRSA:
    """Load a reviewed safetensors conversion; never load DCP in the main venv."""

    manifest_path = manifest_path.resolve(strict=True)
    weights_path = weights_path.resolve(strict=True)
    if weights_path.suffix != ".safetensors":
        raise ValueError("Converted LoRSA weights must use safetensors")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != CONVERTED_LORSA_SCHEMA:
        raise ValueError("Unsupported converted LoRSA schema")
    if manifest.get("weights_sha256") != sha256_file(weights_path):
        raise ValueError("Converted LoRSA weights digest mismatch")
    if manifest.get("main_environment_pickle_loaded") is not False:
        raise ValueError("Converted LoRSA manifest does not preserve the no-pickle boundary")
    config = LoRSAConfig(**manifest["architecture"])
    config.validate()
    model = LowRankSparseAttention(config, device="meta")
    with safe_open(weights_path, framework="pt", device="cpu") as handle:
        keys = tuple(sorted(handle.keys()))
        expected = tuple(sorted(model.state_dict()))
        if keys != expected:
            raise ValueError("Converted LoRSA tensor inventory differs from architecture")
        state = {key: handle.get_tensor(key) for key in keys}
    model.to_empty(device="cpu")
    model.load_state_dict(state, strict=True)
    model.eval()
    return LoadedConvertedLoRSA(
        module=model,
        manifest={
            **manifest,
            "manifest_path": str(manifest_path),
            "manifest_sha256": sha256_file(manifest_path),
            "weights_path": str(weights_path),
            "loaded_tensor_keys": list(keys),
            "main_environment_pickle_loaded": False,
        },
    )


def converted_lorsa_manifest_template(
    config: LoRSAConfig,
    weights_path: Path,
    source_inventory: dict[str, Any],
) -> dict[str, Any]:
    """Build the non-executable handoff contract for an isolated converter."""

    return {
        "schema_version": CONVERTED_LORSA_SCHEMA,
        "architecture": asdict(config),
        "weights_sha256": sha256_file(weights_path),
        "source_dcp_inventory": source_inventory,
        "source_pickle_deserialized_in_isolated_environment": True,
        "main_environment_pickle_loaded": False,
        "conversion_reviewed": False,
    }
