"""TopK MLP transcoders with a strict Leela-SAEs compatibility path."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch
from safetensors import safe_open
from torch import Tensor, nn
from torch.nn import functional as F

from research.interpretability.artifacts import sha256_file


LEELA_SAES_COMMIT = "f946a5736f40397f5c834104aeac6e8e6ff753bd"
TopKMode = Literal["upstream_threshold", "exact"]


def positive_topk(
    value: Tensor,
    k: int,
    *,
    mode: TopKMode = "upstream_threshold",
    tolerance: int = 1,
    max_iterations: int = 50,
) -> Tensor:
    """Keep high positive coordinates using released or exact TopK semantics.

    The pinned Leela-SAEs code binary-searches a non-negative threshold and
    may retain ``k +/- tolerance`` entries.  ``exact`` is a deterministic
    matched-refit alternative that retains at most exactly ``k`` positives.
    """

    if value.ndim < 1 or not torch.is_floating_point(value):
        raise ValueError("TopK input must be a floating tensor")
    width = int(value.shape[-1])
    if type(k) is not int or not 0 < k <= width:
        raise ValueError("k must lie in [1, feature_width]")
    if mode == "exact":
        top_values, top_indices = torch.topk(value, k, dim=-1, sorted=False)
        retained = torch.where(top_values > 0, top_values, torch.zeros_like(top_values))
        result = torch.zeros_like(value)
        return result.scatter(-1, top_indices, retained)
    if mode != "upstream_threshold":
        raise ValueError(f"Unknown TopK mode {mode!r}")
    if type(tolerance) is not int or tolerance < 0 or type(max_iterations) is not int:
        raise ValueError("Invalid upstream TopK search configuration")
    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive")
    constant_shape = value.shape[:-1]
    with torch.no_grad():
        low = torch.zeros(constant_shape, device=value.device, dtype=value.dtype)
        # This intentionally matches upstream's batch-global upper bound.
        high = torch.full_like(low, float(value.max()))
        threshold = 0.5 * (low + high)
        for _iteration in range(max_iterations):
            threshold = 0.5 * (low + high)
            count = (value > threshold.unsqueeze(-1)).sum(dim=-1)
            if bool(((count >= k - tolerance) & (count <= k + tolerance)).all()):
                break
            low = torch.where(count > k + tolerance, threshold, low)
            high = torch.where(count < k - tolerance, threshold, high)
            if bool(((high - low).abs() < 1e-6).all()):
                break
        mask = value >= threshold.unsqueeze(-1)
    return value * mask


@dataclass(frozen=True)
class TranscoderOutput:
    hidden_pre: Tensor
    decoder_weighted_scores: Tensor
    features: Tensor
    reconstruction: Tensor


class TopKTranscoder(nn.Module):
    """Sparse replacement for ``resid_mid_after_ln -> hook_mlp_out``."""

    def __init__(
        self,
        d_model: int,
        d_sae: int,
        *,
        k: int,
        decoder_bias: bool = True,
        topk_mode: TopKMode = "upstream_threshold",
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        if type(d_model) is not int or d_model <= 0:
            raise ValueError("d_model must be positive")
        if type(d_sae) is not int or d_sae <= 0:
            raise ValueError("d_sae must be positive")
        if not 0 < k <= d_sae:
            raise ValueError("k must lie in [1, d_sae]")
        self.d_model = d_model
        self.d_sae = d_sae
        self.k = k
        self.topk_mode = topk_mode
        factory = {"device": device, "dtype": dtype}
        self.W_E = nn.Parameter(torch.empty((d_model, d_sae), **factory))
        self.b_E = nn.Parameter(torch.empty((d_sae,), **factory))
        self.W_D = nn.Parameter(torch.empty((d_sae, d_model), **factory))
        if decoder_bias:
            self.b_D = nn.Parameter(torch.empty((d_model,), **factory))
        else:
            self.register_parameter("b_D", None)

    @property
    def decoder_norm(self) -> Tensor:
        return self.W_D.norm(dim=1)

    def encode(self, value: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        if value.shape[-1] != self.d_model or not torch.is_floating_point(value):
            raise ValueError("Transcoder input has an invalid final dimension or dtype")
        hidden_pre = value @ self.W_E + self.b_E
        decoder_norm = self.decoder_norm
        if bool((decoder_norm <= 0).any()) or not bool(torch.isfinite(decoder_norm).all()):
            raise ValueError("Transcoder decoder contains a zero or nonfinite row")
        weighted = hidden_pre * decoder_norm
        selected = positive_topk(weighted, self.k, mode=self.topk_mode)
        features = selected / decoder_norm
        return features, hidden_pre, weighted

    def decode(self, features: Tensor) -> Tensor:
        if features.shape[-1] != self.d_sae:
            raise ValueError("Transcoder feature width differs from d_sae")
        reconstruction = features @ self.W_D
        return reconstruction if self.b_D is None else reconstruction + self.b_D

    def forward(self, value: Tensor) -> TranscoderOutput:
        features, hidden_pre, weighted = self.encode(value)
        return TranscoderOutput(
            hidden_pre=hidden_pre,
            decoder_weighted_scores=weighted,
            features=features,
            reconstruction=self.decode(features),
        )


def initialize_transcoder(
    transcoder: TopKTranscoder,
    *,
    seed: int,
    decoder_bias: Tensor | None = None,
) -> None:
    """Deterministically initialize a matched-refit transcoder."""

    if next(transcoder.parameters()).device.type == "meta":
        raise ValueError("Cannot initialize a meta-device transcoder")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    with torch.no_grad():
        decoder = torch.randn(
            transcoder.W_D.shape,
            generator=generator,
            dtype=torch.float32,
        )
        decoder /= decoder.norm(dim=1, keepdim=True).clamp_min(1e-12)
        transcoder.W_D.copy_(decoder.to(transcoder.W_D))
        encoder = decoder.T + 0.01 * torch.randn(
            transcoder.W_E.shape,
            generator=generator,
            dtype=torch.float32,
        )
        transcoder.W_E.copy_(encoder.to(transcoder.W_E))
        transcoder.b_E.zero_()
        if transcoder.b_D is not None:
            if decoder_bias is None:
                transcoder.b_D.zero_()
            else:
                if decoder_bias.shape != transcoder.b_D.shape:
                    raise ValueError("decoder_bias shape differs from d_model")
                transcoder.b_D.copy_(decoder_bias.to(transcoder.b_D))


def fold_dataset_normalization(
    state: Mapping[str, Tensor],
    *,
    input_norm: float,
    output_norm: float,
    d_model: int,
) -> dict[str, Tensor]:
    """Fold Leela-SAEs dataset-wise input/output scales into TC parameters."""

    if input_norm <= 0 or output_norm <= 0 or not math.isfinite(input_norm + output_norm):
        raise ValueError("Dataset activation norms must be finite and positive")
    c_in = math.sqrt(d_model) / input_norm
    c_out = math.sqrt(d_model) / output_norm
    result = {name: value.clone() for name, value in state.items()}
    result["b_E"] = result["b_E"] / c_in
    result["W_D"] = result["W_D"] * (c_in / c_out)
    if "b_D" in result:
        result["b_D"] = result["b_D"] / c_out
    return result


def _read_safetensors(path: Path) -> tuple[dict[str, Tensor], dict[str, str], tuple[str, ...]]:
    if path.suffix != ".safetensors":
        raise ValueError("Published transcoder loader accepts safetensors only")
    tensors: dict[str, Tensor] = {}
    with safe_open(path, framework="pt", device="cpu") as handle:
        metadata = dict(handle.metadata() or {})
        keys = tuple(sorted(handle.keys()))
        for key in keys:
            tensors[key] = handle.get_tensor(key)
    return tensors, metadata, keys


@dataclass(frozen=True)
class LoadedTranscoder:
    module: TopKTranscoder
    manifest: dict[str, Any]


def load_published_transcoder(
    config_path: Path,
    weights_path: Path,
    *,
    fold_activation_scale: bool = True,
) -> LoadedTranscoder:
    """Strict-load a published transcoder without pickle or upstream imports."""

    config_path = config_path.resolve(strict=True)
    weights_path = weights_path.resolve(strict=True)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    required_config = {
        "d_model",
        "expansion_factor",
        "act_fn",
        "top_k",
        "hook_point_in",
        "hook_point_out",
        "sparsity_include_decoder_norm",
        "norm_activation",
    }
    if missing := sorted(required_config - set(config)):
        raise ValueError(f"Transcoder config is missing fields: {missing}")
    if config.get("sae_type", "sae") != "sae" or config["act_fn"] != "topk":
        raise ValueError("Published loader supports TopK SAE/transcoder configs only")
    d_model = int(config["d_model"])
    d_sae = int(round(d_model * float(config["expansion_factor"])))
    if d_model != 1024 or not config["sparsity_include_decoder_norm"]:
        raise ValueError("Published BT4 transcoder ABI requires d_model=1024 and decoder norms")
    if config.get("use_glu_encoder", False):
        raise ValueError("GLU transcoders are outside the published BT4 contract")
    layer_in = str(config["hook_point_in"])
    layer_out = str(config["hook_point_out"])
    if not layer_in.endswith(".resid_mid_after_ln") or not layer_out.endswith(".hook_mlp_out"):
        raise ValueError("Transcoder hook points do not match the BT4 MLP branch ABI")
    if layer_in.split(".")[:2] != layer_out.split(".")[:2]:
        raise ValueError("Transcoder input/output hooks refer to different layers")

    tensors, metadata, keys = _read_safetensors(weights_path)
    required_tensors = {"W_E", "b_E", "W_D"}
    if config.get("use_decoder_bias", True):
        required_tensors.add("b_D")
    norm_prefix = "dataset_average_activation_norm."
    norm_keys = {key for key in keys if key.startswith(norm_prefix)}
    allowed_optional = {"tokens_since_last_activation", "is_dead", "current_k"}
    unexpected = set(keys) - required_tensors - norm_keys - allowed_optional
    missing_tensors = required_tensors - set(keys)
    if missing_tensors or unexpected:
        raise ValueError(
            f"Transcoder tensor inventory mismatch: missing={sorted(missing_tensors)}, "
            f"unexpected={sorted(unexpected)}"
        )
    expected_shapes = {
        "W_E": (d_model, d_sae),
        "b_E": (d_sae,),
        "W_D": (d_sae, d_model),
        "b_D": (d_model,),
    }
    parameter_state: dict[str, Tensor] = {}
    for name in sorted(required_tensors):
        value = tensors[name]
        if tuple(value.shape) != expected_shapes[name] or value.dtype != torch.float32:
            raise ValueError(
                f"Published transcoder {name} expected FP32 {expected_shapes[name]}, "
                f"got {value.dtype} {tuple(value.shape)}"
            )
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"Published transcoder {name} contains nonfinite values")
        parameter_state[name] = value

    norms = {key[len(norm_prefix) :]: float(tensors[key]) for key in norm_keys}
    if set(norms) != {layer_in, layer_out}:
        raise ValueError("Transcoder dataset-norm keys do not exactly match its two hooks")
    if fold_activation_scale:
        if config["norm_activation"] != "dataset-wise":
            raise ValueError("Requested fold on a checkpoint not marked dataset-wise")
        parameter_state = fold_dataset_normalization(
            parameter_state,
            input_norm=norms[layer_in],
            output_norm=norms[layer_out],
            d_model=d_model,
        )

    module = TopKTranscoder(
        d_model,
        d_sae,
        k=int(config["top_k"]),
        decoder_bias="b_D" in parameter_state,
        device="meta",
    )
    module.W_E = nn.Parameter(parameter_state["W_E"], requires_grad=False)
    module.b_E = nn.Parameter(parameter_state["b_E"], requires_grad=False)
    module.W_D = nn.Parameter(parameter_state["W_D"], requires_grad=False)
    if "b_D" in parameter_state:
        module.b_D = nn.Parameter(parameter_state["b_D"], requires_grad=False)
    module.eval()
    config_digest = hashlib.sha256(config_path.read_bytes()).hexdigest()
    return LoadedTranscoder(
        module=module,
        manifest={
            "schema_version": "bt4-published-transcoder-load-v1",
            "upstream_code_commit": LEELA_SAES_COMMIT,
            "config_path": str(config_path),
            "config_sha256": config_digest,
            "weights_path": str(weights_path),
            "weights_sha256": sha256_file(weights_path),
            "safetensors_metadata": metadata,
            "tensor_keys": list(keys),
            "d_model": d_model,
            "d_sae": d_sae,
            "top_k": int(config["top_k"]),
            "hook_point_in": layer_in,
            "hook_point_out": layer_out,
            "dataset_average_activation_norm": norms,
            "dataset_normalization_folded": fold_activation_scale,
            "topk_semantics": "pinned upstream positive binary threshold, tolerance=1",
            "pickle_loaded": False,
        },
    )


def reconstruction_metrics(target: Tensor, reconstruction: Tensor) -> dict[str, float]:
    if target.shape != reconstruction.shape or target.numel() == 0:
        raise ValueError("Target and reconstruction must be aligned non-empty tensors")
    target_f64 = target.detach().double()
    reconstructed_f64 = reconstruction.detach().double()
    if not bool(torch.isfinite(target_f64).all() and torch.isfinite(reconstructed_f64).all()):
        raise ValueError("Reconstruction metrics received nonfinite values")
    error = reconstructed_f64 - target_f64
    sse = error.square().sum()
    target_energy = target_f64.square().sum().clamp_min(1e-24)
    centered = target_f64 - target_f64.mean(dim=0, keepdim=True)
    sst = centered.square().sum().clamp_min(1e-24)
    flat_target = target_f64.reshape(-1)
    flat_reconstruction = reconstructed_f64.reshape(-1)
    cosine = F.cosine_similarity(flat_target, flat_reconstruction, dim=0)
    return {
        "mse": float(error.square().mean()),
        "normalized_mse": float(sse / target_energy),
        "relative_l2": float(sse.sqrt() / target_energy.sqrt()),
        "cosine": float(cosine),
        "explained_variance": float(1.0 - sse / sst),
        "target_rms": float(target_f64.square().mean().sqrt()),
        "reconstruction_rms": float(reconstructed_f64.square().mean().sqrt()),
    }


def sparse_support_metrics(first: Tensor, second: Tensor) -> dict[str, float]:
    if first.shape != second.shape or first.ndim < 2:
        raise ValueError("Sparse feature tensors must have aligned shapes")
    first_flat = first.detach().double().reshape(-1, first.shape[-1])
    second_flat = second.detach().double().reshape(-1, second.shape[-1])
    first_support = first_flat > 0
    second_support = second_flat > 0
    union = (first_support | second_support).sum(dim=1)
    intersection = (first_support & second_support).sum(dim=1)
    jaccard = torch.where(union > 0, intersection / union, torch.ones_like(union))
    first_centered = first_flat - first_flat.mean(dim=0, keepdim=True)
    second_centered = second_flat - second_flat.mean(dim=0, keepdim=True)
    numerator = (first_centered * second_centered).sum(dim=0)
    denominator = (first_centered.square().sum(dim=0) * second_centered.square().sum(dim=0)).sqrt()
    resolved = denominator > 1e-20
    correlation = numerator[resolved] / denominator[resolved]
    return {
        "mean_support_jaccard": float(jaccard.mean()),
        "mean_first_l0": float(first_support.sum(dim=1).double().mean()),
        "mean_second_l0": float(second_support.sum(dim=1).double().mean()),
        "first_dead_feature_fraction": float((~first_support.any(dim=0)).double().mean()),
        "second_dead_feature_fraction": float((~second_support.any(dim=0)).double().mean()),
        "resolved_activation_correlations": int(resolved.sum()),
        "mean_resolved_activation_correlation": (
            float(correlation.mean()) if correlation.numel() else math.nan
        ),
    }


class SparseSupportAccumulator:
    """Exact bounded sufficient statistics for paired sparse feature transfer."""

    def __init__(self, feature_width: int) -> None:
        if type(feature_width) is not int or feature_width <= 0:
            raise ValueError("feature_width must be positive")
        self.feature_width = feature_width
        self.row_count = 0
        self.jaccard_sum = 0.0
        self.first_l0_sum = 0
        self.second_l0_sum = 0
        self.first_fired = torch.zeros(feature_width, dtype=torch.bool)
        self.second_fired = torch.zeros(feature_width, dtype=torch.bool)
        self.first_sum = torch.zeros(feature_width, dtype=torch.float64)
        self.second_sum = torch.zeros(feature_width, dtype=torch.float64)
        self.first_square_sum = torch.zeros(feature_width, dtype=torch.float64)
        self.second_square_sum = torch.zeros(feature_width, dtype=torch.float64)
        self.cross_sum = torch.zeros(feature_width, dtype=torch.float64)

    def update(self, first: Tensor, second: Tensor) -> None:
        if first.shape != second.shape or first.shape[-1] != self.feature_width:
            raise ValueError("Sparse accumulator inputs must align on feature_width")
        if first.ndim < 2 or not bool(torch.isfinite(first).all() and torch.isfinite(second).all()):
            raise ValueError("Sparse accumulator inputs must be finite with at least two dims")
        first_flat = first.detach().cpu().reshape(-1, self.feature_width)
        second_flat = second.detach().cpu().reshape(-1, self.feature_width)
        first_support = first_flat > 0
        second_support = second_flat > 0
        union = (first_support | second_support).sum(dim=1)
        intersection = (first_support & second_support).sum(dim=1)
        jaccard = torch.where(union > 0, intersection.double() / union, 1.0)
        self.jaccard_sum += float(jaccard.sum())
        self.first_l0_sum += int(first_support.sum())
        self.second_l0_sum += int(second_support.sum())
        self.first_fired |= first_support.any(dim=0)
        self.second_fired |= second_support.any(dim=0)
        first64 = first_flat.double()
        second64 = second_flat.double()
        self.first_sum += first64.sum(dim=0)
        self.second_sum += second64.sum(dim=0)
        self.first_square_sum += first64.square().sum(dim=0)
        self.second_square_sum += second64.square().sum(dim=0)
        self.cross_sum += (first64 * second64).sum(dim=0)
        self.row_count += first_flat.shape[0]

    def finalize(self) -> dict[str, float]:
        if self.row_count == 0:
            raise ValueError("Cannot finalize an empty sparse support accumulator")
        count = float(self.row_count)
        covariance = self.cross_sum - self.first_sum * self.second_sum / count
        first_ss = self.first_square_sum - self.first_sum.square() / count
        second_ss = self.second_square_sum - self.second_sum.square() / count
        denominator = (first_ss.clamp_min(0) * second_ss.clamp_min(0)).sqrt()
        resolved = denominator > 1e-20
        correlation = covariance[resolved] / denominator[resolved]
        return {
            "mean_support_jaccard": self.jaccard_sum / count,
            "mean_first_l0": self.first_l0_sum / count,
            "mean_second_l0": self.second_l0_sum / count,
            "first_dead_feature_fraction": float((~self.first_fired).double().mean()),
            "second_dead_feature_fraction": float((~self.second_fired).double().mean()),
            "resolved_activation_correlations": int(resolved.sum()),
            "mean_resolved_activation_correlation": (
                float(correlation.mean()) if correlation.numel() else math.nan
            ),
        }


@dataclass(frozen=True)
class FittedTranscoder:
    module: TopKTranscoder
    metadata: dict[str, Any]


def fit_transcoder(
    inputs: Tensor,
    targets: Tensor,
    *,
    d_sae: int,
    k: int,
    epochs: int,
    learning_rate: float,
    batch_size: int,
    seed: int,
    device: torch.device | str = "cpu",
) -> FittedTranscoder:
    """Fit a bounded matched-refit TC pilot; full published-scale fits run remotely."""

    value = inputs.detach().float().reshape(-1, inputs.shape[-1]).cpu()
    target = targets.detach().float().reshape(-1, targets.shape[-1]).cpu()
    if value.shape != target.shape or value.shape[1] <= 0:
        raise ValueError("Transcoder training input and target shapes must match")
    if not bool(torch.isfinite(value).all() and torch.isfinite(target).all()):
        raise ValueError("Transcoder training data contain nonfinite values")
    if epochs <= 0 or batch_size <= 0 or learning_rate <= 0:
        raise ValueError("Invalid transcoder training configuration")
    active_device = torch.device(device)
    module = TopKTranscoder(
        value.shape[1],
        d_sae,
        k=k,
        topk_mode="exact",
        device=active_device,
    )
    initialize_transcoder(module, seed=seed, decoder_bias=target.mean(dim=0))
    optimizer = torch.optim.Adam(module.parameters(), lr=learning_rate)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    losses: list[float] = []
    for _epoch in range(epochs):
        order = torch.randperm(value.shape[0], generator=generator)
        total = 0.0
        for start in range(0, value.shape[0], batch_size):
            rows = order[start : start + batch_size]
            batch_input = value[rows].to(active_device)
            batch_target = target[rows].to(active_device)
            output = module(batch_input)
            loss = F.mse_loss(output.reconstruction, batch_target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                norms = module.W_D.norm(dim=1, keepdim=True).clamp_min(1e-12)
                module.W_D.div_(norms)
                module.W_E.mul_(norms.T)
            total += float(loss.detach()) * rows.numel()
        losses.append(total / value.shape[0])
    module = module.cpu().eval()
    return FittedTranscoder(
        module=module,
        metadata={
            "schema_version": "bt4-matched-transcoder-pilot-v1",
            "d_model": int(value.shape[1]),
            "d_sae": d_sae,
            "k": k,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "seed": seed,
            "training_tokens": int(value.shape[0]),
            "topk_semantics": "exact positive TopK",
            "initial_loss": losses[0],
            "final_loss": losses[-1],
            "scope": "bounded pilot; not a published-scale 800M-token replication",
        },
    )
