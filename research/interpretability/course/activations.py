"""Strict loader and small views for the real Raw/Hero activation gallery."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import chess
import numpy as np


SCHEMA_VERSION = "bt4-real-activation-gallery-v1"
DEFAULT_MANIFEST = Path(
    "curriculum/interpretability/artifacts/real_activation_gallery_v1.json"
)


@dataclass(frozen=True)
class ActivationGalleryDescriptor:
    """Integrity-checked manifest paths, whether or not tensors are local."""

    root: Path
    manifest_path: Path
    manifest: Mapping[str, Any]
    data_path: Path

    @property
    def is_materialized(self) -> bool:
        return self.data_path.is_file()


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def absolute_square_values(values: np.ndarray, fen: str) -> np.ndarray:
    """Map BT4 side-to-move token order back to absolute board squares."""

    source = np.asarray(values)
    if source.shape != (64,):
        raise ValueError("Square values must have shape [64]")
    board = chess.Board(fen)
    if board.turn == chess.WHITE:
        return source.copy()
    result = np.empty_like(source)
    for token in range(64):
        result[chess.square_mirror(token)] = source[token]
    return result


def _stable_softmax(values: np.ndarray) -> np.ndarray:
    logits = np.asarray(values, dtype=np.float64)
    shifted = logits - np.max(logits)
    weights = np.exp(shifted)
    return weights / weights.sum()


@dataclass(frozen=True)
class ActivationGallery:
    root: Path
    manifest_path: Path
    manifest: Mapping[str, Any]
    arrays: Mapping[str, np.ndarray]

    @property
    def position_count(self) -> int:
        return int(self.arrays["position_id"].size)

    @property
    def layer_indices(self) -> tuple[int, ...]:
        return tuple(int(value) for value in self.arrays["layer_indices_i16"])

    @property
    def run_id(self) -> str:
        return str(self.manifest["run_id"])

    @property
    def data_sha256(self) -> str:
        return str(self.manifest["data"]["sha256"])

    def position_options(self) -> dict[str, int]:
        return {
            (
                f"{self.arrays['puzzle_id'][index]} · "
                f"{str(self.arrays['themes'][index]).replace('_', ' ')[:54]}"
            ): index
            for index in range(self.position_count)
        }

    def position(self, index: int) -> dict[str, Any]:
        if not 0 <= index < self.position_count:
            raise IndexError(index)
        summary = self.manifest["summary"]["positions"][index]
        if summary["position_id"] != str(self.arrays["position_id"][index]):
            raise ValueError("Position summary and tensor artifact are misaligned")
        return dict(summary)

    def activation_view(
        self,
        *,
        position: int,
        layer: int,
        channel_rank: int = 0,
    ) -> dict[str, Any]:
        if layer not in self.layer_indices:
            raise ValueError(f"Layer {layer} is not in the gallery")
        if not 0 <= channel_rank < self.arrays["top_delta_channel_i16"].shape[1]:
            raise ValueError("channel_rank lies outside the stored top-channel list")
        record = self.position(position)
        layer_slot = self.layer_indices.index(layer)
        channel = int(self.arrays["top_delta_channel_i16"][layer_slot, channel_rank])
        raw = self.arrays["raw_resid_post_f16"][position, layer_slot].astype(np.float32)
        hero = self.arrays["hero_resid_post_f16"][position, layer_slot].astype(np.float32)
        delta = hero - raw
        raw_norm = np.sqrt(np.mean(raw * raw, axis=1))
        hero_norm = np.sqrt(np.mean(hero * hero, axis=1))
        delta_norm = np.sqrt(np.mean(delta * delta, axis=1))
        denominator = np.linalg.norm(raw, axis=1) * np.linalg.norm(hero, axis=1)
        cosine = np.divide(
            np.sum(raw * hero, axis=1),
            denominator,
            out=np.zeros(64, dtype=np.float32),
            where=denominator > 0.0,
        )
        fen = record["fen"]
        per_layer_delta = []
        per_layer_cosine = []
        for layer_index in range(len(self.layer_indices)):
            raw_layer = self.arrays["raw_resid_post_f16"][position, layer_index].astype(
                np.float32
            )
            hero_layer = self.arrays["hero_resid_post_f16"][position, layer_index].astype(
                np.float32
            )
            layer_delta = hero_layer - raw_layer
            layer_denominator = np.linalg.norm(raw_layer, axis=1) * np.linalg.norm(
                hero_layer, axis=1
            )
            layer_cosine = np.divide(
                np.sum(raw_layer * hero_layer, axis=1),
                layer_denominator,
                out=np.zeros(64, dtype=np.float32),
                where=layer_denominator > 0.0,
            )
            per_layer_delta.append(float(np.sqrt(np.mean(layer_delta * layer_delta))))
            per_layer_cosine.append(float(layer_cosine.mean()))
        return {
            "position": record,
            "layer": layer,
            "channel": channel,
            "channel_rank": channel_rank,
            "raw_channel": absolute_square_values(raw[:, channel], fen),
            "hero_channel": absolute_square_values(hero[:, channel], fen),
            "delta_channel": absolute_square_values(delta[:, channel], fen),
            "raw_square_rms": absolute_square_values(raw_norm, fen),
            "hero_square_rms": absolute_square_values(hero_norm, fen),
            "delta_square_rms": absolute_square_values(delta_norm, fen),
            "token_cosine": absolute_square_values(cosine, fen),
            "per_layer_delta_rms": per_layer_delta,
            "per_layer_mean_token_cosine": per_layer_cosine,
            "quantization": "float16 stored activation; calculations promoted to float32",
        }

    def activation_vector(
        self,
        *,
        position: int,
        layer: int,
        model: str,
        square: str,
    ) -> dict[str, Any]:
        if model not in {"raw", "hero"}:
            raise ValueError("model must be raw or hero")
        if layer not in self.layer_indices:
            raise ValueError(f"Layer {layer} is not in the gallery")
        absolute_square = chess.parse_square(square)
        record = self.position(position)
        board = chess.Board(record["fen"])
        token = (
            absolute_square
            if board.turn == chess.WHITE
            else chess.square_mirror(absolute_square)
        )
        slot = self.layer_indices.index(layer)
        vector = self.arrays[f"{model}_resid_post_f16"][position, slot, token].astype(
            np.float32
        )
        return {
            "position": record,
            "layer": layer,
            "model": model,
            "absolute_square": square,
            "token_index": int(token),
            "token_square": chess.square_name(token),
            "vector": vector,
            "width": int(vector.size),
            "rms": float(np.sqrt(np.mean(vector * vector))),
            "l2": float(np.linalg.norm(vector)),
            "minimum": float(vector.min()),
            "maximum": float(vector.max()),
            "quantization": "float16 stored activation promoted to float32",
        }

    def probe_view(
        self,
        *,
        position: int,
        layer: int,
        model: str,
        task: str,
    ) -> dict[str, Any]:
        if model not in {"raw", "hero"}:
            raise ValueError("model must be raw or hero")
        if task not in {"legal_destination", "opponent_attack_count", "piece_code"}:
            raise ValueError("Unknown gallery probe task")
        if layer not in self.layer_indices:
            raise ValueError(f"Layer {layer} is not in the gallery")
        record = self.position(position)
        slot = self.layer_indices.index(layer)
        key = f"{model}_probe_{task}_f16"
        prediction = self.arrays[key][position, slot].astype(np.float32)
        if task == "legal_destination":
            target = self.arrays["legal_destination_u8"][position].astype(np.float32)
            displayed = prediction
        elif task == "opponent_attack_count":
            target = self.arrays["attack_count_theirs_u8"][position].astype(np.float32)
            displayed = prediction
        else:
            target = self.arrays["piece_code_i8"][position].astype(np.float32)
            displayed = np.max(prediction, axis=-1)
        predicted_class = (
            np.argmax(prediction, axis=-1).astype(np.float32)
            if task == "piece_code"
            else None
        )
        error = (
            (predicted_class != target).astype(np.float32)
            if predicted_class is not None
            else displayed - target
        )
        return {
            "position": record,
            "layer": layer,
            "model": model,
            "task": task,
            "prediction": absolute_square_values(displayed, record["fen"]),
            "target": absolute_square_values(target, record["fen"]),
            "predicted_class": (
                absolute_square_values(predicted_class, record["fen"])
                if predicted_class is not None
                else None
            ),
            "error": absolute_square_values(error, record["fen"]),
            "error_semantics": (
                "1 denotes a misclassified piece code"
                if task == "piece_code"
                else "prediction minus target"
            ),
            "scope": "trained probe applied to a frozen gallery activation",
        }

    def policy_lens_view(
        self,
        *,
        position: int,
        layer: int,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        if layer not in self.layer_indices:
            raise ValueError(f"Layer {layer} is not in the gallery")
        if not 1 <= top_k <= 20:
            raise ValueError("top_k must lie in [1, 20]")
        slot = self.layer_indices.index(layer)
        legal_count = int(self.arrays["legal_count_u16"][position])
        legal = self.arrays["legal_idx_u16"][position, :legal_count].astype(np.int64)
        moves = self.arrays["legal_move_uci"][position, :legal_count].astype(str)
        target = int(self.arrays["target_action_u16"][position])
        rows: list[dict[str, Any]] = []
        for arm_index, arm in enumerate(self.arrays["arm_ids"].astype(str)):
            logits = self.arrays["lens_logits_f16"][arm_index, position, slot].astype(
                np.float64
            )
            legal_probability = _stable_softmax(logits[legal])
            full_probability = _stable_softmax(logits)
            order = np.argsort(-legal_probability, kind="stable")[:top_k]
            target_slot = np.flatnonzero(legal == target)
            if target_slot.size != 1:
                raise ValueError("Target action is not uniquely legal")
            rows.append(
                {
                    "arm": str(arm),
                    "top_moves": ", ".join(
                        f"{moves[index]} {legal_probability[index]:.1%}" for index in order
                    ),
                    "target_probability": float(legal_probability[target_slot[0]]),
                    "legal_mass_before_masking": float(full_probability[legal].sum()),
                    "entropy": float(
                        -np.sum(legal_probability * np.log(np.clip(legal_probability, 1e-300, None)))
                    ),
                }
            )
        return rows

    def target_gradient_view(
        self,
        *,
        position: int,
        layer: int,
        model: str,
    ) -> dict[str, Any]:
        if model not in {"raw", "hero"}:
            raise ValueError("model must be raw or hero")
        gradient_layers = tuple(
            int(value) for value in self.arrays["gradient_layer_indices_i16"]
        )
        if layer not in gradient_layers:
            raise ValueError(f"Gradient layer must be one of {gradient_layers}")
        slot = gradient_layers.index(layer)
        record = self.position(position)
        square_norm = self.arrays[f"{model}_target_gradient_square_norm_f32"][
            position, slot
        ]
        gradient_x_state = self.arrays[f"{model}_target_gradient_x_state_f32"][
            position, slot
        ]
        gradient_rms = float(
            self.arrays[f"{model}_target_gradient_rms_f32"][position, slot]
        )
        return {
            "position": record,
            "layer": layer,
            "model": model,
            "square_gradient_norm": absolute_square_values(square_norm, record["fen"]),
            "gradient_x_state": absolute_square_values(gradient_x_state, record["fen"]),
            "gradient_rms": gradient_rms,
            "objective": self.manifest["target_gradient"]["objective"],
            "interpretation": self.manifest["target_gradient"]["interpretation"],
        }

    def spectrum_view(self, *, layer: int, model: str) -> dict[str, Any]:
        if model not in {"raw", "hero"}:
            raise ValueError("model must be raw or hero")
        spectrum_layers = tuple(
            int(value) for value in self.arrays["spectrum_layer_indices_i16"]
        )
        if layer not in spectrum_layers:
            raise ValueError(f"Spectrum layer must be one of {spectrum_layers}")
        slot = spectrum_layers.index(layer)
        singular = self.arrays[f"{model}_centered_singular_values_f32"][slot].astype(
            np.float64
        )
        variance = singular * singular
        total = float(variance.sum())
        if total <= 0.0:
            raise ValueError("Activation spectrum has zero total variance")
        fraction = variance / total
        positive = fraction > 0.0
        participation_ratio = float(total * total / np.sum(variance * variance))
        entropy_rank = float(np.exp(-np.sum(fraction[positive] * np.log(fraction[positive]))))
        cumulative = np.cumsum(fraction)
        components_90 = int(np.searchsorted(cumulative, 0.90, side="left") + 1)
        components_95 = int(np.searchsorted(cumulative, 0.95, side="left") + 1)
        return {
            "layer": layer,
            "model": model,
            "singular_values": singular,
            "variance_fraction": fraction,
            "cumulative_variance": cumulative,
            "participation_ratio": participation_ratio,
            "entropy_effective_rank": entropy_rank,
            "components_for_90_percent": components_90,
            "components_for_95_percent": components_95,
            "observation_count": int(self.manifest["activation_spectrum"]["observations"]),
            "operation": self.manifest["activation_spectrum"]["operation"],
            "interpretation": self.manifest["activation_spectrum"]["interpretation"],
        }


def _validated_manifest_paths(
    root: Path,
    *,
    manifest_path: Path = DEFAULT_MANIFEST,
) -> tuple[Path, Mapping[str, Any], Path]:
    root = root.resolve()
    path = manifest_path if manifest_path.is_absolute() else root / manifest_path
    path = path.resolve(strict=True)
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError("Activation gallery manifest escapes the repository root") from exc
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported activation gallery schema")
    integrity = manifest.get("manifest_integrity", {})
    base = dict(manifest)
    base.pop("manifest_integrity", None)
    expected_integrity = hashlib.sha256(_canonical_bytes(base)).hexdigest()
    if integrity.get("sha256") != expected_integrity:
        raise ValueError("Activation gallery manifest integrity mismatch")
    data_record = manifest.get("data", {})
    data_path = (path.parent / str(data_record.get("path", ""))).resolve()
    try:
        data_path.relative_to(path.parent)
    except ValueError as exc:
        raise ValueError("Activation gallery data path escapes its artifact directory") from exc
    return path, manifest, data_path


def describe_activation_gallery(
    root: Path,
    *,
    manifest_path: Path = DEFAULT_MANIFEST,
) -> ActivationGalleryDescriptor:
    """Read and integrity-check the manifest without requiring the local NPZ."""

    selected_root = root.resolve()
    path, manifest, data_path = _validated_manifest_paths(
        selected_root,
        manifest_path=manifest_path,
    )
    return ActivationGalleryDescriptor(
        root=selected_root,
        manifest_path=path,
        manifest=manifest,
        data_path=data_path,
    )


def load_optional_activation_gallery(
    root: Path,
    *,
    manifest_path: Path = DEFAULT_MANIFEST,
    verify_sources: bool = False,
) -> ActivationGallery | None:
    """Load the local-only gallery, or return ``None`` when only its manifest exists."""

    descriptor = describe_activation_gallery(
        root,
        manifest_path=manifest_path,
    )
    if not descriptor.is_materialized:
        return None
    return load_activation_gallery(
        root,
        manifest_path=manifest_path,
        verify_sources=verify_sources,
    )


def load_activation_gallery(
    root: Path,
    *,
    manifest_path: Path = DEFAULT_MANIFEST,
    verify_sources: bool = False,
) -> ActivationGallery:
    descriptor = describe_activation_gallery(
        root,
        manifest_path=manifest_path,
    )
    path = descriptor.manifest_path
    manifest = descriptor.manifest
    data_path = descriptor.data_path
    if not data_path.is_file():
        raise FileNotFoundError(
            "The real activation gallery is a local, license-gated artifact; "
            f"materialize or rebuild {data_path} before requesting tensor views"
        )
    data_record = manifest["data"]
    if data_path.stat().st_size != int(data_record.get("size_bytes", -1)):
        raise ValueError("Activation gallery data size drift")
    if _sha256_file(data_path) != data_record.get("sha256"):
        raise ValueError("Activation gallery data SHA-256 drift")
    with np.load(data_path, allow_pickle=False) as payload:
        inventory = data_record.get("arrays", {})
        if set(payload.files) != set(inventory):
            raise ValueError("Activation gallery array inventory drift")
        arrays = {name: np.asarray(payload[name]).copy() for name in payload.files}
    for name, value in arrays.items():
        record = inventory[name]
        if value.dtype.hasobject:
            raise ValueError(f"Activation gallery array {name} has object dtype")
        if list(value.shape) != record["shape"] or str(value.dtype) != record["dtype"]:
            raise ValueError(f"Activation gallery array ABI drift for {name}")
        if value.dtype.kind == "f" and not np.isfinite(value).all():
            raise ValueError(f"Activation gallery array {name} has nonfinite values")
    if verify_sources:
        for source in manifest.get("sources", {}).values():
            source_path = (root / source["path"]).resolve(strict=True)
            try:
                source_path.relative_to(root)
            except ValueError as exc:
                raise ValueError("Activation gallery source path escapes root") from exc
            if _sha256_file(source_path) != source["sha256"]:
                raise ValueError(f"Activation gallery source drift: {source['path']}")
    gallery = ActivationGallery(
        root=root,
        manifest_path=path,
        manifest=manifest,
        arrays=arrays,
    )
    if gallery.position_count != int(manifest["selection"]["position_count"]):
        raise ValueError("Activation gallery position count drift")
    if gallery.layer_indices != tuple(range(15)):
        raise ValueError("Activation gallery must contain all 15 encoder layers")
    return gallery
