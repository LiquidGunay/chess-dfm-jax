"""Streaming raw-to-Hero activation comparisons over the frozen hook ABI."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch

from research.interpretability.accumulators import (
    ProjectedPairAccumulator,
    StreamingPairedMoments,
    cluster_bootstrap_mean_ci,
    paired_position_metrics,
    rademacher_projection,
)
from research.train_torch import BT4EncoderCapture


HOOK_FIELDS: tuple[str, ...] = (
    "hook_attn_in",
    "hook_attn_out",
    "resid_mid_after_ln",
    "hook_mlp_out",
    "resid_post_after_ln",
)
CANONICAL_HOOK_NAMES = {
    "hook_attn_in": "blocks.{layer}.hook_attn_in",
    "hook_attn_out": "blocks.{layer}.hook_attn_out",
    "resid_mid_after_ln": "blocks.{layer}.resid_mid_after_ln",
    "hook_mlp_out": "blocks.{layer}.hook_mlp_out",
    "resid_post_after_ln": "blocks.{layer}.resid_post_after_ln",
}
POSITION_METRICS: tuple[str, ...] = (
    "relative_l2_to_raw",
    "symmetric_relative_l2",
    "cosine_similarity",
    "delta_rms",
)


def _seed_for(base_seed: int, label: str) -> int:
    digest = hashlib.sha256(label.encode("utf-8")).digest()
    return (base_seed ^ int.from_bytes(digest[:8], "big")) % (2**63 - 1)


def _linear_cka(raw: np.ndarray, hero: np.ndarray, cross: np.ndarray) -> float:
    denominator = float(np.linalg.norm(raw, ord="fro") * np.linalg.norm(hero, ord="fro"))
    if denominator <= 0.0:
        return 0.0
    return min(1.0, max(0.0, float(np.square(cross).sum()) / denominator))


class ActivationDiffAccumulator:
    """Bounded sufficient statistics for five hooks at selected layers.

    Exact scalar moments and one row of exact distance metrics per position are
    retained. Multivariate metrics use a frozen shared Rademacher projection.
    The full depth map keeps only its projected cross-products on the active
    device, independent of corpus size.
    """

    def __init__(
        self,
        *,
        layer_indices: Sequence[int],
        feature_width: int = 1024,
        sketch_width: int = 64,
        projection_seed: int = 20260807,
    ) -> None:
        layers = tuple(layer_indices)
        if not layers or tuple(sorted(set(layers))) != layers:
            raise ValueError("layer_indices must be unique and increasing")
        self.layer_indices = layers
        self.feature_width = feature_width
        self.sketch_width = sketch_width
        self.projection_seed = projection_seed
        self.projection, self.projection_sha256 = rademacher_projection(
            feature_width,
            sketch_width,
            seed=projection_seed,
        )
        self.exact = {hook: [StreamingPairedMoments() for _ in layers] for hook in HOOK_FIELDS}
        self.sketches = {
            hook: [ProjectedPairAccumulator(sketch_width) for _ in layers] for hook in HOOK_FIELDS
        }
        self._position_chunks: dict[str, dict[str, list[list[np.ndarray]]]] = {
            metric: {hook: [[] for _ in layers] for hook in HOOK_FIELDS}
            for metric in POSITION_METRICS
        }
        self.position_count = 0
        self.square_observation_count = 0
        self._layer_cross_gram: torch.Tensor | None = None
        self._projection_tensors: dict[str, torch.Tensor] = {}

    def _projection_for(self, device: torch.device) -> torch.Tensor:
        key = str(device)
        projection = self._projection_tensors.get(key)
        if projection is None:
            projection = torch.from_numpy(self.projection).to(device=device)
            self._projection_tensors[key] = projection
        return projection

    def update(
        self,
        raw_capture: BT4EncoderCapture,
        hero_capture: BT4EncoderCapture,
    ) -> None:
        if raw_capture.layer_indices != self.layer_indices:
            raise ValueError("Raw capture layers differ from accumulator layers")
        if hero_capture.layer_indices != self.layer_indices:
            raise ValueError("Hero capture layers differ from accumulator layers")
        batch_size: int | None = None
        for hook in HOOK_FIELDS:
            raw_tensor = getattr(raw_capture, hook)
            hero_tensor = getattr(hero_capture, hook)
            if raw_tensor.shape != hero_tensor.shape:
                raise ValueError(f"Raw/Hero {hook} shapes differ")
            expected_prefix = (len(self.layer_indices), raw_tensor.shape[1], 64)
            if (
                raw_tensor.ndim != 4
                or tuple(raw_tensor.shape[:3]) != expected_prefix
                or raw_tensor.shape[-1] != self.feature_width
            ):
                raise ValueError(f"Unexpected {hook} shape {tuple(raw_tensor.shape)}")
            if batch_size is None:
                batch_size = raw_tensor.shape[1]
            elif batch_size != raw_tensor.shape[1]:
                raise ValueError("Capture batch dimensions differ between hooks")
            if not torch.isfinite(raw_tensor).all() or not torch.isfinite(hero_tensor).all():
                raise ValueError(f"Nonfinite activation in {hook}")

            projection = self._projection_for(raw_tensor.device)
            raw_projected = raw_tensor.float() @ projection
            hero_projected = hero_tensor.float() @ projection
            raw_cpu = raw_tensor.detach().float().cpu().numpy()
            hero_cpu = hero_tensor.detach().float().cpu().numpy()
            raw_projected_cpu = raw_projected.detach().cpu().numpy()
            hero_projected_cpu = hero_projected.detach().cpu().numpy()
            for layer_offset in range(len(self.layer_indices)):
                raw_layer = raw_cpu[layer_offset]
                hero_layer = hero_cpu[layer_offset]
                self.exact[hook][layer_offset].update(raw_layer, hero_layer)
                position = paired_position_metrics(raw_layer, hero_layer)
                for metric in POSITION_METRICS:
                    self._position_chunks[metric][hook][layer_offset].append(position[metric])
                raw_rows = raw_projected_cpu[layer_offset].reshape(-1, self.sketch_width)
                hero_rows = hero_projected_cpu[layer_offset].reshape(-1, self.sketch_width)
                self.sketches[hook][layer_offset].update(raw_rows, hero_rows)

            if hook == "resid_post_after_ln":
                layer_count = len(self.layer_indices)
                flat_raw = raw_projected.permute(1, 2, 0, 3).reshape(
                    -1, layer_count * self.sketch_width
                )
                flat_hero = hero_projected.permute(1, 2, 0, 3).reshape(
                    -1, layer_count * self.sketch_width
                )
                cross = flat_raw.transpose(0, 1) @ flat_hero
                if self._layer_cross_gram is None:
                    self._layer_cross_gram = torch.zeros_like(cross)
                self._layer_cross_gram.add_(cross)
                self.square_observation_count += flat_raw.shape[0]
            del (
                raw_cpu,
                hero_cpu,
                raw_projected,
                hero_projected,
                raw_projected_cpu,
                hero_projected_cpu,
            )

        if batch_size is None:  # pragma: no cover - hook list is fixed non-empty
            raise AssertionError("Activation update saw no hooks")
        self.position_count += batch_size

    def position_metric_arrays(self) -> dict[str, np.ndarray]:
        """Return compact [position, hook, layer] exact metrics for re-bootstrap."""

        if self.position_count <= 0:
            raise ValueError("Cannot materialize empty activation metrics")
        result: dict[str, np.ndarray] = {}
        for metric in POSITION_METRICS:
            hooks: list[np.ndarray] = []
            for hook in HOOK_FIELDS:
                layers = [
                    np.concatenate(self._position_chunks[metric][hook][offset])
                    for offset in range(len(self.layer_indices))
                ]
                hooks.append(np.stack(layers, axis=1))
            result[metric] = np.stack(hooks, axis=1).astype(np.float64, copy=False)
        expected = (self.position_count, len(HOOK_FIELDS), len(self.layer_indices))
        if any(value.shape != expected for value in result.values()):
            raise AssertionError("Activation position-metric shape invariant failed")
        return result

    def _layer_correspondence(self) -> dict[str, Any]:
        if self._layer_cross_gram is None or self.square_observation_count <= 0:
            raise ValueError("Layer correspondence is empty")
        layer_count = len(self.layer_indices)
        width = self.sketch_width
        cross_gram = (
            self._layer_cross_gram.detach()
            .double()
            .cpu()
            .numpy()
            .reshape(
                layer_count,
                width,
                layer_count,
                width,
            )
        )
        cross_gram = cross_gram.transpose(0, 2, 1, 3)
        residual_sketches = self.sketches["resid_post_after_ln"]
        raw_scatters: list[np.ndarray] = []
        hero_scatters: list[np.ndarray] = []
        for sketch in residual_sketches:
            raw, hero, _cross = sketch.centered_scatters()
            raw_scatters.append(raw)
            hero_scatters.append(hero)
        widths = sorted({min(width, candidate) for candidate in (16, 32, width)})
        matrices: dict[str, list[list[float]]] = {}
        for nested_width in widths:
            matrix = np.zeros((layer_count, layer_count), dtype=np.float64)
            for raw_offset in range(layer_count):
                raw_sum = residual_sketches[raw_offset].raw_sum[:nested_width]
                for hero_offset in range(layer_count):
                    hero_sum = residual_sketches[hero_offset].hero_sum[:nested_width]
                    centered_cross = (
                        cross_gram[
                            raw_offset,
                            hero_offset,
                            :nested_width,
                            :nested_width,
                        ]
                        - np.outer(raw_sum, hero_sum) / self.square_observation_count
                    )
                    matrix[raw_offset, hero_offset] = _linear_cka(
                        raw_scatters[raw_offset][:nested_width, :nested_width],
                        hero_scatters[hero_offset][:nested_width, :nested_width],
                        centered_cross,
                    )
            matrices[str(nested_width)] = matrix.tolist()
        full = np.asarray(matrices[str(width)])
        diagonal_from_corresponding = np.asarray(
            [sketch.finalize()["linear_cka"] for sketch in residual_sketches]
        )
        sensitivity: dict[str, dict[str, float]] = {}
        for nested_width in widths[:-1]:
            nested = np.asarray(matrices[str(nested_width)])
            absolute = np.abs(nested - full)
            sensitivity[str(nested_width)] = {
                "mean_absolute_difference_from_full_width": float(absolute.mean()),
                "max_absolute_difference_from_full_width": float(absolute.max()),
            }
        best_offsets = np.argmax(full, axis=1)
        return {
            "hook": "blocks.{layer}.resid_post_after_ln",
            "metric": "linear CKA in one frozen shared Rademacher sketch",
            "observation_unit": "square token",
            "observation_count": self.square_observation_count,
            "layer_indices": list(self.layer_indices),
            "matrices_by_nested_sketch_width": matrices,
            "full_width": width,
            "nested_width_sensitivity": sensitivity,
            "diagonal_mean": float(np.diag(full).mean()),
            "diagonal_min": float(np.diag(full).min()),
            "diagonal_max": float(np.diag(full).max()),
            "diagonal_cross_accumulator_max_abs_error": float(
                np.max(np.abs(np.diag(full) - diagonal_from_corresponding))
            ),
            "best_hero_layer_for_each_raw_layer": [
                {
                    "raw_layer": self.layer_indices[raw_offset],
                    "hero_layer": self.layer_indices[int(hero_offset)],
                    "depth_shift": self.layer_indices[int(hero_offset)]
                    - self.layer_indices[raw_offset],
                    "linear_cka": float(full[raw_offset, hero_offset]),
                }
                for raw_offset, hero_offset in enumerate(best_offsets)
            ],
            "precision": (
                "projected per-layer covariance is accumulated in CPU FP64; "
                "the bounded all-layer cross-product is accumulated in device FP32"
            ),
        }

    def finalize(
        self,
        game_ids: np.ndarray,
        *,
        bootstrap_seed: int,
        bootstrap_replicates: int,
    ) -> dict[str, Any]:
        if len(game_ids) != self.position_count:
            raise ValueError("game_ids must align with accumulated positions")
        position_arrays = self.position_metric_arrays()
        hook_results: dict[str, Any] = {}
        for hook_offset, hook in enumerate(HOOK_FIELDS):
            layers: dict[str, Any] = {}
            for layer_offset, layer in enumerate(self.layer_indices):
                bootstrap = {
                    metric: cluster_bootstrap_mean_ci(
                        position_arrays[metric][:, hook_offset, layer_offset],
                        game_ids,
                        seed=_seed_for(
                            bootstrap_seed,
                            f"activation.{hook}.{layer}.{metric}",
                        ),
                        replicates=bootstrap_replicates,
                    )
                    for metric in POSITION_METRICS
                }
                layers[str(layer)] = {
                    "canonical_name": CANONICAL_HOOK_NAMES[hook].format(layer=layer),
                    "exact_elementwise": self.exact[hook][layer_offset].finalize(),
                    "projected_multivariate": self.sketches[hook][layer_offset].finalize(),
                    "position_level_game_cluster_bootstrap": bootstrap,
                }
            hook_results[hook] = {"layers": layers}

        residual_layers = hook_results["resid_post_after_ln"]["layers"]
        maximum_change = max(
            self.layer_indices,
            key=lambda layer: residual_layers[str(layer)]["exact_elementwise"][
                "symmetric_relative_l2"
            ],
        )
        anchors = (
            self.layer_indices[0],
            self.layer_indices[len(self.layer_indices) // 2],
            self.layer_indices[-1],
            maximum_change,
        )
        selected_layers = list(dict.fromkeys(anchors))
        return {
            "schema_version": "bt4-paired-activation-diff-v1",
            "position_count": self.position_count,
            "hook_fields": list(HOOK_FIELDS),
            "layer_indices": list(self.layer_indices),
            "tensor_contract": "captures [layer,batch,64,1024] before streaming",
            "exact_metrics": (
                "elementwise moments and per-position L2/cosine are exact after "
                "conversion of captured compute tensors to FP32"
            ),
            "projection": {
                "type": "shared Rademacher Johnson-Lindenstrauss",
                "input_width": self.feature_width,
                "output_width": self.sketch_width,
                "seed": self.projection_seed,
                "sha256": self.projection_sha256,
                "scope": "same matrix for both models, every hook, and every layer",
                "limitations": (
                    "CKA/SVCCA/PWCCA/Procrustes/principal-angle/effective-rank "
                    "metrics are pilot sketches, not full-width exact estimates"
                ),
            },
            "hooks": hook_results,
            "layer_correspondence": self._layer_correspondence(),
            "selected_layers_for_expensive_development_work": {
                "layers": selected_layers,
                "rule": (
                    "first, middle, final, and maximum residual-post symmetric "
                    "relative-L2 layer; duplicates removed"
                ),
                "selection_corpus": "128-position development pilot; not confirmatory",
                "maximum_change_layer": maximum_change,
            },
        }
