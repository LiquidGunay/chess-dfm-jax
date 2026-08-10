"""Deterministic exact/randomized matrix spectra for the raw-to-Hero update."""

from __future__ import annotations

import argparse
import dataclasses
import gc
import hashlib
import json
import math
import os
import platform
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, NamedTuple

import torch
from torch import Tensor

from research.interpretability.models import (
    BT4ComparisonModels,
    load_bt4_comparison_models,
)
from research.interpretability.parameter_diff import _git_commit


_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = _REPO_ROOT / "research/analysis/raw_hero_matrix_spectra_20260807.json"


class LowRankSVD(NamedTuple):
    singular_values: Tensor
    left_vectors: Tensor
    right_vectors_t: Tensor
    method: str
    sample_rank: int


@dataclasses.dataclass(frozen=True)
class SpectralConfig:
    rank: int = 32
    oversample: int = 8
    power_iterations: int = 1
    exact_max_min_dimension: int = 64
    seed: int = 20260807

    def validate(self) -> None:
        if self.rank <= 0:
            raise ValueError("rank must be positive")
        if self.oversample < 0:
            raise ValueError("oversample must be non-negative")
        if self.power_iterations < 0:
            raise ValueError("power_iterations must be non-negative")
        if self.exact_max_min_dimension <= 0:
            raise ValueError("exact threshold must be positive")


def _matrix_seed(name: str, base_seed: int, *, salt: str) -> int:
    digest = hashlib.sha256(f"{salt}\0{name}".encode("utf-8")).digest()
    return (base_seed ^ int.from_bytes(digest[:8], "big")) % (2**63 - 1)


def _thin_qr(value: Tensor) -> Tensor:
    return torch.linalg.qr(value, mode="reduced").Q


def low_rank_svd(
    matrix: Tensor,
    *,
    config: SpectralConfig,
    seed: int,
) -> LowRankSVD:
    """Return an exact small SVD or deterministic randomized leading subspace."""

    config.validate()
    if matrix.ndim != 2:
        raise ValueError(f"matrix must have rank 2, got {tuple(matrix.shape)}")
    if matrix.dtype != torch.float32:
        raise ValueError(f"matrix must be float32, got {matrix.dtype}")
    rows, columns = matrix.shape
    minimum_dimension = min(rows, columns)
    if minimum_dimension <= 0:
        raise ValueError("matrix dimensions must be positive")
    target_rank = min(config.rank, minimum_dimension)
    if minimum_dimension <= config.exact_max_min_dimension or target_rank == minimum_dimension:
        left, singular_values, right_t = torch.linalg.svd(
            matrix,
            full_matrices=False,
        )
        return LowRankSVD(
            singular_values=singular_values,
            left_vectors=left,
            right_vectors_t=right_t,
            method="exact",
            sample_rank=minimum_dimension,
        )

    sample_rank = min(
        minimum_dimension,
        target_rank + config.oversample,
    )
    generator = torch.Generator(device="cpu").manual_seed(seed)
    omega = torch.randn(
        (columns, sample_rank),
        generator=generator,
        dtype=torch.float32,
        device="cpu",
    ).to(matrix.device)
    left_basis = _thin_qr(matrix @ omega)
    del omega
    for _ in range(config.power_iterations):
        right_basis = _thin_qr(matrix.transpose(0, 1) @ left_basis)
        left_basis = _thin_qr(matrix @ right_basis)
        del right_basis
    compressed = left_basis.transpose(0, 1) @ matrix
    compressed_left, singular_values, right_t = torch.linalg.svd(
        compressed,
        full_matrices=False,
    )
    left = left_basis @ compressed_left
    del left_basis, compressed, compressed_left
    return LowRankSVD(
        singular_values=singular_values[:target_rank],
        left_vectors=left[:, :target_rank],
        right_vectors_t=right_t[:target_rank],
        method="randomized",
        sample_rank=sample_rank,
    )


def _energy_rank_bounds(
    singular_values: Tensor,
    *,
    frobenius_squared: float,
    minimum_dimension: int,
) -> dict[str, float]:
    if frobenius_squared <= 0.0:
        return {
            "participation_rank_lower": 0.0,
            "participation_rank_upper": 0.0,
            "entropy_effective_rank_lower": 0.0,
            "entropy_effective_rank_upper": 0.0,
        }
    energy = singular_values.detach().double().square().cpu()
    observed = torch.clamp(energy / frobenius_squared, min=0.0)
    observed_total = min(1.0, float(observed.sum().item()))
    tail = max(0.0, 1.0 - observed_total)
    remaining = max(0, minimum_dimension - observed.numel())

    observed_square_sum = float(observed.square().sum().item())
    denominator_lower = observed_square_sum + tail * tail
    denominator_upper = observed_square_sum + (tail * tail / remaining if remaining > 0 else 0.0)
    participation_lower = 1.0 / denominator_lower if denominator_lower > 0.0 else 0.0
    participation_upper = 1.0 / denominator_upper if denominator_upper > 0.0 else 0.0

    positive = observed[observed > 0]
    observed_entropy = float((-(positive * positive.log())).sum().item())
    tail_entropy_lower = -tail * math.log(tail) if tail > 0.0 else 0.0
    tail_entropy_upper = (
        -tail * math.log(tail / remaining) if tail > 0.0 and remaining > 0 else tail_entropy_lower
    )
    return {
        "participation_rank_lower": participation_lower,
        "participation_rank_upper": participation_upper,
        "entropy_effective_rank_lower": math.exp(observed_entropy + tail_entropy_lower),
        "entropy_effective_rank_upper": math.exp(observed_entropy + tail_entropy_upper),
    }


def spectrum_metrics(
    spectrum: LowRankSVD,
    *,
    frobenius_squared: float,
    minimum_dimension: int,
) -> dict[str, Any]:
    singular_values = spectrum.singular_values.detach().double().cpu()
    singular_squared = singular_values.square()
    captured = (
        min(1.0, float(singular_squared.sum().item()) / frobenius_squared)
        if frobenius_squared > 0.0
        else 0.0
    )
    residual_squared = max(0.0, frobenius_squared * (1.0 - captured))
    spectral_norm = float(singular_values[0].item()) if singular_values.numel() else 0.0
    stable_rank = (
        frobenius_squared / (spectral_norm * spectral_norm) if spectral_norm > 0.0 else 0.0
    )
    cumulative = (
        torch.cumsum(singular_squared, dim=0) / frobenius_squared
        if frobenius_squared > 0.0
        else torch.zeros_like(singular_squared)
    )
    ranks_at_energy: dict[str, int | None] = {}
    for threshold in (0.5, 0.9, 0.95, 0.99):
        matches = torch.nonzero(cumulative >= threshold, as_tuple=False)
        ranks_at_energy[f"{int(threshold * 100)}pct"] = (
            int(matches[0].item()) + 1 if matches.numel() else None
        )
    return {
        "method": spectrum.method,
        "returned_rank": int(singular_values.numel()),
        "sample_rank": spectrum.sample_rank,
        "minimum_dimension": minimum_dimension,
        "singular_values": singular_values.tolist(),
        "frobenius_l2": math.sqrt(max(0.0, frobenius_squared)),
        "spectral_norm": spectral_norm,
        "stable_rank": stable_rank,
        "captured_energy_fraction": captured,
        "relative_frobenius_residual": math.sqrt(max(0.0, 1.0 - captured)),
        "residual_frobenius_l2": math.sqrt(residual_squared),
        "rank_at_total_energy": ranks_at_energy,
        "energy_rank_bounds": _energy_rank_bounds(
            singular_values,
            frobenius_squared=frobenius_squared,
            minimum_dimension=minimum_dimension,
        ),
    }


def _subspace_overlap(first: Tensor, second: Tensor) -> float:
    rank = min(first.shape[1], second.shape[1])
    if rank <= 0:
        return 0.0
    value = first[:, :rank].transpose(0, 1) @ second[:, :rank]
    return float(value.float().square().sum().item() / rank)


def update_alignment(
    delta: Tensor,
    *,
    delta_svd: LowRankSVD,
    raw_svd: LowRankSVD,
    delta_frobenius_squared: float,
    max_rank: int | None = None,
) -> dict[str, float | int]:
    rank = min(
        delta_svd.left_vectors.shape[1],
        raw_svd.left_vectors.shape[1],
    )
    rank = min(rank, max_rank) if max_rank is not None else rank
    if rank <= 0 or delta_frobenius_squared <= 0.0:
        return {
            "rank": rank,
            "delta_energy_in_raw_left_subspace": 0.0,
            "delta_energy_in_raw_right_subspace": 0.0,
            "delta_energy_in_raw_bilinear_subspace": 0.0,
            "left_subspace_overlap": 0.0,
            "right_subspace_overlap": 0.0,
        }
    raw_left = raw_svd.left_vectors[:, :rank]
    raw_right_t = raw_svd.right_vectors_t[:rank]
    left_projected = raw_left.transpose(0, 1) @ delta
    right_projected = delta @ raw_right_t.transpose(0, 1)
    bilinear = left_projected @ raw_right_t.transpose(0, 1)
    denominator = delta_frobenius_squared
    return {
        "rank": rank,
        "delta_energy_in_raw_left_subspace": min(
            1.0,
            float(left_projected.float().square().sum().item()) / denominator,
        ),
        "delta_energy_in_raw_right_subspace": min(
            1.0,
            float(right_projected.float().square().sum().item()) / denominator,
        ),
        "delta_energy_in_raw_bilinear_subspace": min(
            1.0,
            float(bilinear.float().square().sum().item()) / denominator,
        ),
        "left_subspace_overlap": _subspace_overlap(
            raw_left,
            delta_svd.left_vectors[:, :rank],
        ),
        "right_subspace_overlap": _subspace_overlap(
            raw_right_t.transpose(0, 1),
            delta_svd.right_vectors_t[:rank].transpose(0, 1),
        ),
    }


def _matrix_candidates(
    models: BT4ComparisonModels,
) -> tuple[list[tuple[str, Tensor, Tensor, float]], float]:
    raw_named = dict(models.raw_encoder.named_parameters())
    hero_named = dict(models.hero_encoder.named_parameters())
    if raw_named.keys() != hero_named.keys():
        raise ValueError("Raw/Hero parameter names differ")
    candidates: list[tuple[str, Tensor, Tensor, float]] = []
    global_delta_squared = 0.0
    for name in raw_named:
        raw = raw_named[name]
        hero = hero_named[name]
        delta_squared = float(
            torch.sum(
                (hero.detach().cpu().float() - raw.detach().cpu().float()).square(),
                dtype=torch.float64,
            ).item()
        )
        global_delta_squared += delta_squared
        if raw.ndim >= 2:
            candidates.append((name, raw, hero, delta_squared))
    candidates.sort(key=lambda row: (-row[3], row[0]))
    return candidates, global_delta_squared


def matrix_delta_spectra(
    models: BT4ComparisonModels,
    *,
    config: SpectralConfig,
    device: torch.device,
    max_matrices: int | None = None,
) -> dict[str, Any]:
    """Analyze aligned matrix updates sequentially with bounded accelerator memory."""

    config.validate()
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    candidates, global_delta_squared = _matrix_candidates(models)
    total_matrix_count = len(candidates)
    if max_matrices is not None:
        if max_matrices <= 0:
            raise ValueError("max_matrices must be positive or None")
        candidates = candidates[:max_matrices]

    rows: list[dict[str, Any]] = []
    analyzed_delta_squared = 0.0
    for name, raw_parameter, hero_parameter, delta_squared in candidates:
        raw = raw_parameter.detach().reshape(raw_parameter.shape[0], -1)
        hero = hero_parameter.detach().reshape(hero_parameter.shape[0], -1)
        matrix = raw.cpu().float().to(device)
        delta = hero.cpu().float().to(device) - matrix
        observed_delta_squared = float(torch.sum(delta.double().square()).item())
        tolerance = max(1e-12, 1e-6 * max(delta_squared, observed_delta_squared))
        if abs(observed_delta_squared - delta_squared) > tolerance:
            raise RuntimeError(f"Delta energy changed while moving {name} to {device}")
        analyzed_delta_squared += delta_squared

        delta_svd = low_rank_svd(
            delta,
            config=config,
            seed=_matrix_seed(name, config.seed, salt="delta"),
        )
        raw_svd = low_rank_svd(
            matrix,
            config=config,
            seed=_matrix_seed(name, config.seed, salt="raw"),
        )
        raw_frobenius_squared = float(torch.sum(matrix.double().square()).item())
        delta_metrics = spectrum_metrics(
            delta_svd,
            frobenius_squared=delta_squared,
            minimum_dimension=min(delta.shape),
        )
        raw_metrics = spectrum_metrics(
            raw_svd,
            frobenius_squared=raw_frobenius_squared,
            minimum_dimension=min(matrix.shape),
        )
        rows.append(
            {
                "name": name,
                "shape": list(raw_parameter.shape),
                "matrix_shape": list(matrix.shape),
                "dtype": str(raw_parameter.dtype),
                "parameter_count": raw_parameter.numel(),
                "delta_energy_fraction_global": (
                    delta_squared / global_delta_squared if global_delta_squared > 0.0 else 0.0
                ),
                "delta": delta_metrics,
                "raw": raw_metrics,
                "update_alignment_with_raw_leading_subspace": update_alignment(
                    delta,
                    delta_svd=delta_svd,
                    raw_svd=raw_svd,
                    delta_frobenius_squared=delta_squared,
                    max_rank=config.rank,
                ),
            }
        )
        del matrix, delta, delta_svd, raw_svd
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    return {
        "schema_version": "bt4-raw-hero-matrix-spectra-v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "git_commit": _git_commit(),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "device": str(device),
            "device_name": (
                torch.cuda.get_device_name(device)
                if device.type == "cuda"
                else platform.processor() or "cpu"
            ),
        },
        "models": models.descriptor(),
        "config": dataclasses.asdict(config),
        "selection": {
            "order": "descending exact delta Frobenius energy, then name",
            "total_matrix_count": total_matrix_count,
            "analyzed_matrix_count": len(rows),
            "analyzed_delta_energy_fraction": (
                analyzed_delta_squared / global_delta_squared if global_delta_squared > 0.0 else 0.0
            ),
            "global_delta_frobenius_l2": math.sqrt(global_delta_squared),
        },
        "interpretation_contract": {
            "exact": (
                "method=exact contains the complete singular spectrum and exact energy ranks"
            ),
            "randomized": (
                "method=randomized contains only deterministic leading values; "
                "captured energy and residual are exact Frobenius accounting, "
                "while effective-rank values are explicit lower/upper bounds"
            ),
        },
        "matrices": rows,
    }


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    resolved = path.resolve()
    try:
        resolved.relative_to(_REPO_ROOT.resolve())
    except ValueError as exc:
        raise ValueError(f"Output must remain inside {_REPO_ROOT}: {resolved}") from exc
    resolved.parent.mkdir(parents=True, exist_ok=True)
    temporary = resolved.with_name(f".{resolved.name}.partial")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, resolved)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-bt4",
        type=Path,
        default=_REPO_ROOT / "models/source/extracted/BT4_exported.pb.gz",
    )
    parser.add_argument(
        "--hero-checkpoint",
        type=Path,
        default=_REPO_ROOT / "research/runs/torch_hero_epoch_v1/checkpoint",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--oversample", type=int, default=8)
    parser.add_argument("--power-iterations", type=int, default=1)
    parser.add_argument("--exact-max-min-dimension", type=int, default=64)
    parser.add_argument(
        "--max-matrices",
        type=int,
        default=0,
        help="0 analyzes every matrix; positive values keep top delta-energy matrices",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    torch.set_num_threads(2)
    if args.device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
    models = load_bt4_comparison_models(
        raw_bt4_path=args.raw_bt4,
        hero_checkpoint_dir=args.hero_checkpoint,
        device="cpu",
    )
    report = matrix_delta_spectra(
        models,
        config=SpectralConfig(
            rank=args.rank,
            oversample=args.oversample,
            power_iterations=args.power_iterations,
            exact_max_min_dimension=args.exact_max_min_dimension,
        ),
        device=torch.device(args.device),
        max_matrices=args.max_matrices or None,
    )
    _write_json_atomic(args.output, report)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
                "matrix_count": report["selection"]["analyzed_matrix_count"],
                "delta_energy_fraction": report["selection"]["analyzed_delta_energy_fraction"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
