"""Seal paired uncertainty diagnostics for a LoRSA Raw-to-Hero transfer run.

This is a CPU-only post-processing stage.  It consumes a checksum-verified
``lorsa_transfer_pilot`` bundle, performs a deterministic paired bootstrap over
unique corpus groups, and writes a new immutable, checksum-verified artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import shutil
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from research.interpretability.artifacts import (
    canonical_json_bytes,
    content_identity,
    sha256_file,
    verify_checksums,
    write_checksums,
    write_json_atomic,
    write_jsonl_atomic,
    write_text_atomic,
)


_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SEED = 20260808
DEFAULT_BOOTSTRAP_REPLICATES = 20_000
DEFAULT_BOOTSTRAP_CHUNK_SIZE = 1_024
DEFAULT_TAIL_COUNT = 10


def _output_staging(output: Path) -> tuple[Path, Path]:
    target = output.resolve()
    if target in {Path("/"), Path.home().resolve(), _REPO_ROOT.resolve()}:
        raise ValueError(f"Unsafe output path: {target}")
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite immutable run: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.with_name(f".{target.name}.{os.getpid()}.partial")
    if staging.exists():
        raise FileExistsError(f"Staging path already exists: {staging}")
    staging.mkdir()
    return target, staging


def _require_finite_number(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be a finite number")
    return result


def _require_bounded_number(
    value: object, *, field: str, minimum: float, maximum: float | None = None
) -> float:
    result = _require_finite_number(value, field=field)
    if result < minimum or (maximum is not None and result > maximum):
        interval = f"[{minimum}, {maximum}]" if maximum is not None else f"[{minimum}, inf)"
        raise ValueError(f"{field} must lie in {interval}")
    return result


def _nested(mapping: Mapping[str, Any], path: Sequence[str]) -> Any:
    value: Any = mapping
    for component in path:
        if not isinstance(value, Mapping) or component not in value:
            raise ValueError(f"Missing required input field: {'.'.join(path)}")
        value = value[component]
    return value


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _load_examples(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"examples.jsonl line {line_number} is not an object")
        rows.append(value)
    if not rows:
        raise ValueError("Input examples.jsonl is empty")
    return rows


def _require_passed_gate(metrics: Mapping[str, Any], field: str) -> None:
    gate = _nested(metrics, (field,))
    if not isinstance(gate, Mapping) or gate.get("passed") is not True:
        raise ValueError(f"Input {field} did not pass")
    criteria = gate.get("criteria")
    if not isinstance(criteria, Mapping) or not criteria:
        raise ValueError(f"Input {field} has no auditable criteria")
    failed = sorted(key for key, value in criteria.items() if value is not True)
    if failed:
        raise ValueError(f"Input {field} has failed criteria: {failed}")


def _validate_transfer_contract(
    manifest: Mapping[str, Any],
    metrics: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if manifest.get("schema_version") != "bt4-published-lorsa-transfer-run-v1":
        raise ValueError("Unsupported LoRSA transfer manifest schema")
    if metrics.get("schema_version") != "bt4-published-lorsa-transfer-metrics-v1":
        raise ValueError("Unsupported LoRSA transfer metrics schema")

    payloads = manifest.get("payload_identities")
    if not isinstance(payloads, dict):
        raise ValueError("Input manifest has no payload_identities object")
    legacy_fields = {
        "base_input_bundle",
        "hero",
        "lorsa",
        "lorsa_input_bundle",
        "metrics",
        "positions",
        "raw",
    }
    current_fields = legacy_fields | {"source_files"}
    if set(payloads) not in (legacy_fields, current_fields):
        raise ValueError("Input payload identity inventory violates transfer contract")
    metrics_identity = content_identity(metrics)
    if payloads["metrics"] != metrics_identity:
        raise ValueError("Input metrics identity does not match payload_identities")
    position_identity = hashlib.sha256(
        "\n".join(str(_nested(row, ("position_id",))) for row in rows).encode()
    ).hexdigest()
    if payloads["positions"] != position_identity:
        raise ValueError("Input ordered-position identity does not match payload_identities")

    source_files_bound = "source_files" in payloads
    if source_files_bound:
        source_files = manifest.get("source_files")
        if not isinstance(source_files, dict) or not source_files:
            raise ValueError("Input source_files payload has no manifest inventory")
        for path, digest in source_files.items():
            if (
                not isinstance(path, str)
                or not path
                or not isinstance(digest, str)
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise ValueError("Input source_files inventory is malformed")
        if payloads["source_files"] != content_identity(source_files):
            raise ValueError("Input source_files identity does not match payload_identities")

    expected_run_id = f"sha256:{content_identity(payloads)}"
    if manifest.get("run_id") != expected_run_id:
        raise ValueError("Input run_id does not match payload_identities")

    expected_count = metrics.get("position_count")
    if isinstance(expected_count, bool) or not isinstance(expected_count, int):
        raise ValueError("Input position_count must be an integer")
    if expected_count != len(rows):
        raise ValueError(
            f"Input row count {len(rows)} does not match position_count {expected_count}"
        )
    experiment = _nested(manifest, ("experiment",))
    if not isinstance(experiment, Mapping):
        raise ValueError("Input experiment contract is not an object")
    if experiment.get("position_count") != len(rows):
        raise ValueError("Input experiment position_count does not match retained rows")
    if experiment.get("sequential_source_gate") is not True:
        raise ValueError("Input did not enforce the sequential source gate")
    if experiment.get("confirmatory") is not False:
        raise ValueError("Input transfer run is not labeled exploratory")
    if _nested(manifest, ("public_corpus_manifest", "split")) != "development only":
        raise ValueError("Input transfer run is not restricted to development data")

    _require_passed_gate(metrics, "artifact_contract_gate")
    _require_passed_gate(metrics, "source_compatibility_gate")
    if _nested(metrics, ("source_stage", "status")) != "complete":
        raise ValueError("Input source stage is incomplete")
    if _nested(metrics, ("transfer_stage", "status")) != "complete":
        raise ValueError("Input Hero transfer stage is incomplete")
    if not isinstance(_nested(metrics, ("transfer_stage", "hero")), Mapping):
        raise ValueError("Input has no completed Hero transfer metrics")
    interpretation = _nested(metrics, ("interpretation_contract",))
    required_interpretation = {
        "development_split_only": True,
        "test_split_opened": False,
        "exploratory": True,
        "lorsa_frozen": True,
        "hero_transfer_interpretable": True,
    }
    if not isinstance(interpretation, Mapping) or any(
        interpretation.get(field) is not value
        for field, value in required_interpretation.items()
    ):
        raise ValueError("Input interpretation contract does not authorize Hero transfer")
    return {
        "metrics_identity": metrics_identity,
        "ordered_positions_identity": position_identity,
        "run_id_recomputed": expected_run_id,
        "source_files_bound_in_run_id": source_files_bound,
        "all_required_gates_passed": True,
    }


def _validate_input(
    input_dir: Path,
) -> tuple[
    dict[str, Any], dict[str, Any], list[dict[str, Any]], dict[str, str], dict[str, Any]
]:
    verified_files = verify_checksums(input_dir)
    expected_inventory = {"manifest.json", "metrics.json", "examples.jsonl", "run.log"}
    if set(verified_files) != expected_inventory:
        raise ValueError("Input artifact has an unexpected retained-file inventory")
    manifest = _load_json(input_dir / "manifest.json")
    metrics = _load_json(input_dir / "metrics.json")
    rows = _load_examples(input_dir / "examples.jsonl")
    transfer_contract = _validate_transfer_contract(manifest, metrics, rows)
    return manifest, metrics, rows, verified_files, transfer_contract


def _identity(rows: Sequence[Mapping[str, Any]], field: str) -> dict[str, Any]:
    values: list[str] = []
    for index, row in enumerate(rows):
        value = row.get(field)
        if not isinstance(value, str) or not value:
            raise ValueError(f"Row {index} has no non-empty {field}")
        values.append(value)
    unique = set(values)
    return {
        "field": field,
        "count": len(values),
        "unique_count": len(unique),
        "all_unique": len(unique) == len(values),
        "ordered_sha256": hashlib.sha256(canonical_json_bytes(values)).hexdigest(),
        "sorted_sha256": hashlib.sha256(canonical_json_bytes(sorted(unique))).hexdigest(),
    }


def paired_bootstrap_mean_ci(
    deltas: np.ndarray,
    *,
    replicates: int,
    seed: int,
    chunk_size: int = DEFAULT_BOOTSTRAP_CHUNK_SIZE,
) -> tuple[float, float]:
    """Return the deterministic percentile CI for the mean paired delta."""

    values = np.asarray(deltas, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("Paired deltas must be a non-empty finite vector")
    if replicates <= 0:
        raise ValueError("bootstrap replicates must be positive")
    if chunk_size <= 0:
        raise ValueError("bootstrap chunk size must be positive")
    generator = np.random.Generator(np.random.PCG64(seed))
    means = np.empty(replicates, dtype=np.float64)
    offset = 0
    while offset < replicates:
        count = min(chunk_size, replicates - offset)
        indices = generator.integers(0, values.size, size=(count, values.size))
        means[offset : offset + count] = values[indices].mean(axis=1)
        offset += count
    lower, upper = np.quantile(means, (0.025, 0.975), method="linear")
    return float(lower), float(upper)


def _metric_vector(
    rows: Sequence[Mapping[str, Any]], model: str, field: str
) -> np.ndarray:
    return np.asarray(
        [
            _require_finite_number(
                _nested(row, (model, field)), field=f"rows[{index}].{model}.{field}"
            )
            for index, row in enumerate(rows)
        ],
        dtype=np.float64,
    )


def _require_array_bounds(
    values: np.ndarray, *, field: str, minimum: float, maximum: float | None = None
) -> None:
    if np.any(values < minimum) or (maximum is not None and np.any(values > maximum)):
        interval = (
            f"[{minimum}, {maximum}]" if maximum is not None else f"[{minimum}, inf)"
        )
        raise ValueError(f"{field} values must lie in {interval}")


def _paired_summary(
    raw: np.ndarray,
    hero: np.ndarray,
    *,
    replicates: int,
    seed: int,
    unit: str = "position",
) -> dict[str, Any]:
    if raw.shape != hero.shape:
        raise ValueError("Paired metric vectors have different shapes")
    delta = hero - raw
    lower, upper = paired_bootstrap_mean_ci(
        delta, replicates=replicates, seed=seed
    )
    return {
        "estimand": f"unweighted mean across paired {unit}s",
        "bootstrap_seed": seed,
        "raw_mean": float(raw.mean()),
        "hero_mean": float(hero.mean()),
        "hero_minus_raw_mean": float(delta.mean()),
        "hero_minus_raw_median": float(np.median(delta)),
        "hero_greater_than_raw_fraction": float(np.mean(delta > 0.0)),
        "equal_fraction": float(np.mean(delta == 0.0)),
        "hero_minus_raw_mean_percentile_ci95": {
            "lower": lower,
            "upper": upper,
        },
        "delta_quantiles": {
            "minimum": float(np.min(delta)),
            "p05": float(np.quantile(delta, 0.05)),
            "p25": float(np.quantile(delta, 0.25)),
            "median": float(np.quantile(delta, 0.50)),
            "p75": float(np.quantile(delta, 0.75)),
            "p95": float(np.quantile(delta, 0.95)),
            "maximum": float(np.max(delta)),
        },
    }


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks


def _correlation(first: np.ndarray, second: np.ndarray) -> float | None:
    if first.size != second.size or first.size < 2:
        return None
    if np.ptp(first) == 0.0 or np.ptp(second) == 0.0:
        return None
    return float(np.corrcoef(first, second)[0, 1])


def _exact_two_sided_binomial_p(successes: int, trials: int) -> float:
    if trials == 0:
        return 1.0
    tail = min(successes, trials - successes)
    probability = sum(math.comb(trials, value) for value in range(tail + 1)) / (2**trials)
    return min(1.0, 2.0 * probability)


def _top1_contingency(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    counts = {
        "both_preserved": 0,
        "raw_only_preserved": 0,
        "hero_only_preserved": 0,
        "neither_preserved": 0,
    }
    for index, row in enumerate(rows):
        raw = _nested(row, ("raw", "policy_top1_preserved"))
        hero = _nested(row, ("hero", "policy_top1_preserved"))
        if not isinstance(raw, bool) or not isinstance(hero, bool):
            raise ValueError(f"Row {index} top-1 preservation fields must be booleans")
        if raw and hero:
            counts["both_preserved"] += 1
        elif raw:
            counts["raw_only_preserved"] += 1
        elif hero:
            counts["hero_only_preserved"] += 1
        else:
            counts["neither_preserved"] += 1
    total = len(rows)
    discordant = counts["raw_only_preserved"] + counts["hero_only_preserved"]
    exact_p = _exact_two_sided_binomial_p(counts["hero_only_preserved"], discordant)
    return {
        **counts,
        "total": total,
        "raw_preserved": counts["both_preserved"] + counts["raw_only_preserved"],
        "hero_preserved": counts["both_preserved"] + counts["hero_only_preserved"],
        "raw_preserved_fraction": (
            counts["both_preserved"] + counts["raw_only_preserved"]
        )
        / total,
        "hero_preserved_fraction": (
            counts["both_preserved"] + counts["hero_only_preserved"]
        )
        / total,
        "paired_exact_test": {
            "method": "two-sided exact conditional binomial test (exact McNemar)",
            "discordant_pairs": discordant,
            "hero_only_successes": counts["hero_only_preserved"],
            "null_probability": 0.5,
            "p_value": exact_p,
            "exploratory": True,
        },
        "interpretation": (
            "The off-diagonal cells identify changed failures even when aggregate "
            "Raw and Hero preservation rates are equal."
        ),
    }


def _tail_examples(
    rows: Sequence[Mapping[str, Any]], *, tail_count: int
) -> list[dict[str, Any]]:
    if tail_count <= 0:
        raise ValueError("tail_count must be positive")
    raw_policy = _metric_vector(rows, "raw", "policy_js")
    hero_policy = _metric_vector(rows, "hero", "policy_js")
    raw_nmse = _metric_vector(rows, "raw", "reconstruction_normalized_mse")
    hero_nmse = _metric_vector(rows, "hero", "reconstruction_normalized_mse")
    delta = hero_policy - raw_policy
    selected: list[tuple[str, int]] = []
    for index in np.argsort(delta, kind="stable")[-tail_count:][::-1]:
        selected.append(("largest_hero_policy_degradation", int(index)))
    for index in np.argsort(delta, kind="stable")[:tail_count]:
        selected.append(("largest_hero_policy_improvement", int(index)))
    output: list[dict[str, Any]] = []
    ranks_by_tail: dict[str, int] = {}
    for tail, index in selected:
        ranks_by_tail[tail] = ranks_by_tail.get(tail, 0) + 1
        row = rows[index]
        output.append(
            {
                "tail": tail,
                "rank_within_tail": ranks_by_tail[tail],
                "source_row_index": index,
                "source_row_sha256": hashlib.sha256(
                    canonical_json_bytes(dict(row))
                ).hexdigest(),
                "group_id": row["group_id"],
                "position_id": row["position_id"],
                "puzzle_id": row.get("puzzle_id"),
                "policy_js": {
                    "raw": float(raw_policy[index]),
                    "hero": float(hero_policy[index]),
                    "hero_minus_raw": float(delta[index]),
                },
                "reconstruction_normalized_mse": {
                    "raw": float(raw_nmse[index]),
                    "hero": float(hero_nmse[index]),
                    "hero_minus_raw": float(hero_nmse[index] - raw_nmse[index]),
                },
                "support_jaccard": _require_finite_number(
                    _nested(row, ("raw_hero_transfer", "support_jaccard")),
                    field=f"rows[{index}].raw_hero_transfer.support_jaccard",
                ),
                "policy_top1_preserved": {
                    "raw": row["raw"]["policy_top1_preserved"],
                    "hero": row["hero"]["policy_top1_preserved"],
                },
            }
        )
    return output


def analyze_rows(
    metrics: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    *,
    replicates: int,
    seed: int,
    tail_count: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    group_identity = _identity(rows, "group_id")
    position_identity = _identity(rows, "position_id")
    if not group_identity["all_unique"]:
        raise ValueError(
            "Paired bootstrap requires exactly one retained position per group_id; "
            "duplicate groups would require a cluster-weighting policy"
        )
    if not position_identity["all_unique"]:
        raise ValueError("Input position_id values must be unique")

    raw_nmse = _metric_vector(rows, "raw", "reconstruction_normalized_mse")
    hero_nmse = _metric_vector(rows, "hero", "reconstruction_normalized_mse")
    raw_cosine = _metric_vector(rows, "raw", "reconstruction_cosine")
    hero_cosine = _metric_vector(rows, "hero", "reconstruction_cosine")
    raw_policy = _metric_vector(rows, "raw", "policy_js")
    hero_policy = _metric_vector(rows, "hero", "policy_js")
    raw_pattern = _metric_vector(rows, "raw", "native_vs_lorsa_head_mean_pattern_js")
    hero_pattern = _metric_vector(rows, "hero", "native_vs_lorsa_head_mean_pattern_js")
    support = np.asarray(
        [
            _require_finite_number(
                _nested(row, ("raw_hero_transfer", "support_jaccard")),
                field=f"rows[{index}].raw_hero_transfer.support_jaccard",
            )
            for index, row in enumerate(rows)
        ],
        dtype=np.float64,
    )
    native_matched = np.asarray(
        [
            _require_finite_number(
                _nested(row, ("raw_hero_transfer", "native_matched_head_pattern_js")),
                field=f"rows[{index}].raw_hero_transfer.native_matched_head_pattern_js",
            )
            for index, row in enumerate(rows)
        ],
        dtype=np.float64,
    )
    lorsa_matched = np.asarray(
        [
            _require_finite_number(
                _nested(row, ("raw_hero_transfer", "lorsa_matched_head_pattern_js")),
                field=f"rows[{index}].raw_hero_transfer.lorsa_matched_head_pattern_js",
            )
            for index, row in enumerate(rows)
        ],
        dtype=np.float64,
    )

    maximum_js = math.log(2.0) + 1e-12
    for field, values, minimum, maximum in (
        ("raw reconstruction_normalized_mse", raw_nmse, 0.0, None),
        ("hero reconstruction_normalized_mse", hero_nmse, 0.0, None),
        ("raw reconstruction_cosine", raw_cosine, -1.0, 1.0),
        ("hero reconstruction_cosine", hero_cosine, -1.0, 1.0),
        ("raw policy_js", raw_policy, 0.0, maximum_js),
        ("hero policy_js", hero_policy, 0.0, maximum_js),
        ("raw native_vs_lorsa_head_mean_pattern_js", raw_pattern, 0.0, maximum_js),
        ("hero native_vs_lorsa_head_mean_pattern_js", hero_pattern, 0.0, maximum_js),
        ("support_jaccard", support, 0.0, 1.0),
        ("native_matched_head_pattern_js", native_matched, 0.0, maximum_js),
        ("lorsa_matched_head_pattern_js", lorsa_matched, 0.0, maximum_js),
    ):
        _require_array_bounds(values, field=field, minimum=minimum, maximum=maximum)

    source_raw_nmse = _require_bounded_number(
        _nested(metrics, ("source_stage", "reconstruction", "normalized_mse")),
        field="metrics.source_stage.reconstruction.normalized_mse",
        minimum=0.0,
    )
    source_hero_nmse = _require_bounded_number(
        _nested(metrics, ("transfer_stage", "hero", "reconstruction", "normalized_mse")),
        field="metrics.transfer_stage.hero.reconstruction.normalized_mse",
        minimum=0.0,
    )
    recorded_delta = _require_finite_number(
        _nested(
            metrics,
            (
                "transfer_stage",
                "raw_hero",
                "degradation",
                "normalized_mse_delta_hero_minus_raw",
            ),
        ),
        field="metrics.transfer_stage.raw_hero.degradation.normalized_mse_delta_hero_minus_raw",
    )
    if not math.isclose(source_hero_nmse - source_raw_nmse, recorded_delta, abs_tol=1e-12):
        raise ValueError("Input global NMSE delta is internally inconsistent")

    policy_delta = hero_policy - raw_policy
    analysis = {
        "schema_version": "bt4-published-lorsa-paired-bootstrap-metrics-v1",
        "position_count": len(rows),
        "group_count": group_identity["unique_count"],
        "bootstrap": {
            "method": "paired nonparametric percentile bootstrap over unique group_id",
            "replicates": replicates,
            "seed": seed,
            "rng": "numpy.random.Generator(PCG64)",
            "confidence_level": 0.95,
            "quantile_method": "linear",
            "confirmatory": False,
            "multiple_comparison_correction": None,
        },
        "reconstruction_normalized_mse": {
            "global_energy_weighted": {
                "estimand": (
                    "sum squared error divided by sum squared target activation over "
                    "all retained positions, tokens, and channels"
                ),
                "raw": source_raw_nmse,
                "hero": source_hero_nmse,
                "hero_minus_raw": recorded_delta,
                "hero_over_raw": source_hero_nmse / source_raw_nmse,
                "bootstrap_ci": None,
                "reason_no_bootstrap_ci": (
                    "The retained per-position rows contain normalized ratios, not the "
                    "per-position numerator and denominator needed to resample this ratio-of-sums."
                ),
            },
            "mean_per_position": _paired_summary(
                raw_nmse, hero_nmse, replicates=replicates, seed=seed
            ),
            "estimands_are_not_interchangeable": True,
        },
        "reconstruction_cosine_mean_per_position": _paired_summary(
            raw_cosine, hero_cosine, replicates=replicates, seed=seed + 1
        ),
        "policy_js_mean_per_position": _paired_summary(
            raw_policy, hero_policy, replicates=replicates, seed=seed + 2
        ),
        "native_vs_lorsa_head_mean_pattern_js_mean_per_position": _paired_summary(
            raw_pattern, hero_pattern, replicates=replicates, seed=seed + 3
        ),
        "raw_hero_matched_head_pattern_change": {
            "native_mean": float(native_matched.mean()),
            "lorsa_mean": float(lorsa_matched.mean()),
            "lorsa_minus_native": _paired_summary(
                native_matched,
                lorsa_matched,
                replicates=replicates,
                seed=seed + 4,
                unit="position",
            ),
        },
        "policy_top1_contingency": _top1_contingency(rows),
        "support_associations": {
            "exploratory": True,
            "support_jaccard_mean": float(support.mean()),
            "support_jaccard_vs_signed_policy_js_delta": {
                "pearson": _correlation(support, policy_delta),
                "spearman": _correlation(
                    _average_ranks(support), _average_ranks(policy_delta)
                ),
            },
            "support_jaccard_vs_absolute_policy_js_delta": {
                "pearson": _correlation(support, np.abs(policy_delta)),
                "spearman": _correlation(
                    _average_ranks(support), _average_ranks(np.abs(policy_delta))
                ),
            },
        },
        "tail_examples": {
            "selection_metric": "hero policy_js minus raw policy_js",
            "count_per_direction": min(tail_count, len(rows)),
            "retained_file": "examples.jsonl",
            "selection_is_post_hoc": True,
        },
        "interpretation": {
            "scope": "exploratory development split",
            "unit_of_resampling": "group_id (one retained position per unique group)",
            "test_split_opened": False,
            "causal_feature_claim": False,
            "warning": (
                "A confidence interval excluding zero describes this sampled replacement "
                "metric; it does not establish semantic feature equivalence or causal transfer."
            ),
        },
    }
    return analysis, _tail_examples(rows, tail_count=min(tail_count, len(rows))), {
        "groups": group_identity,
        "positions": position_identity,
        "ordered_group_position_pairs_sha256": hashlib.sha256(
            canonical_json_bytes(
                [[row["group_id"], row["position_id"]] for row in rows]
            )
        ).hexdigest(),
    }


def run_analysis(args: argparse.Namespace) -> dict[str, Any]:
    input_dir = args.input.resolve()
    if input_dir == args.output.resolve():
        raise ValueError("Input and output directories must differ")
    target, staging = _output_staging(args.output)
    started_utc = datetime.now(timezone.utc)
    try:
        manifest, source_metrics, rows, verified_files, transfer_contract = (
            _validate_input(input_dir)
        )
        metrics, tail_rows, identities = analyze_rows(
            source_metrics,
            rows,
            replicates=args.bootstrap_replicates,
            seed=args.seed,
            tail_count=args.tail_count,
        )
        source_files = {
            Path(__file__).resolve().relative_to(_REPO_ROOT).as_posix(): sha256_file(
                Path(__file__)
            ),
            "research/interpretability/artifacts.py": sha256_file(
                _REPO_ROOT / "research/interpretability/artifacts.py"
            ),
        }
        input_identity = {
            "run_id": manifest["run_id"],
            "checksum_ledger_sha256": sha256_file(input_dir / "checksums.sha256"),
            "verified_files": verified_files,
            "validated_transfer_contract": transfer_contract,
            "group_identity": identities["groups"],
            "position_identity": identities["positions"],
            "ordered_group_position_pairs_sha256": identities[
                "ordered_group_position_pairs_sha256"
            ],
        }
        config = {
            "bootstrap_replicates": args.bootstrap_replicates,
            "seed": args.seed,
            "tail_count": args.tail_count,
        }
        payload_identities = {
            "input": content_identity(input_identity),
            "config": content_identity(config),
            "metrics": content_identity(metrics),
            "examples": content_identity({"rows": tail_rows}),
            "source_files": content_identity(source_files),
        }
        run_id = content_identity(payload_identities)
        output_manifest = {
            "schema_version": "bt4-published-lorsa-paired-bootstrap-run-v1",
            "run_id": f"sha256:{run_id}",
            "started_utc": started_utc.isoformat(),
            "input_artifact": {
                "path": str(input_dir),
                **input_identity,
            },
            "group_identity": identities,
            "analysis_config": config,
            "environment": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "device": "CPU",
                "provider_dollars": 0.0,
            },
            "source_files": source_files,
            "payload_identities": payload_identities,
            "immutable_after_creation": True,
        }
        log_lines = [
            f"input_run_id {manifest['run_id']}",
            f"input_checksum_ledger_sha256 {input_identity['checksum_ledger_sha256']}",
            f"groups {identities['groups']['unique_count']}",
            f"bootstrap_replicates {args.bootstrap_replicates}",
            f"seed {args.seed}",
            f"run_id sha256:{run_id}",
        ]
        write_json_atomic(staging / "metrics.json", metrics)
        write_jsonl_atomic(staging / "examples.jsonl", tail_rows)
        write_json_atomic(staging / "manifest.json", output_manifest)
        write_text_atomic(staging / "run.log", "\n".join(log_lines) + "\n")
        write_checksums(staging)
        verify_checksums(staging)
        os.replace(staging, target)
        verify_checksums(target)
        return {
            "output": str(target),
            "run_id": f"sha256:{run_id}",
            "position_count": len(rows),
            "bootstrap_replicates": args.bootstrap_replicates,
        }
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--bootstrap-replicates", type=int, default=DEFAULT_BOOTSTRAP_REPLICATES
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--tail-count", type=int, default=DEFAULT_TAIL_COUNT)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    print(json.dumps(run_analysis(args), sort_keys=True))


if __name__ == "__main__":
    main()
