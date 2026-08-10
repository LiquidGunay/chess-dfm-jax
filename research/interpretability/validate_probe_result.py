"""Independent, prediction-level validation of Raw/Hero probe result bundles."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from safetensors import safe_open

from research.interpretability.artifacts import (
    content_identity,
    verify_checksums,
    write_json_atomic,
)
from research.interpretability.chessbench import load_public_corpus
from research.interpretability.probe_pilot import select_probe_rows
from research.interpretability.probes import (
    binary_metrics,
    multiclass_metrics,
    regression_metrics,
)


DEFAULT_RUN_DIR = Path("research/analysis/modal_probe_all_layers_dev_v1")
DEFAULT_CORPUS_MANIFEST = Path("research/eval/interpretability_public_v2/manifest.json")
DEFAULT_OUTPUT = Path("research/analysis/raw_hero_interpretability_report_v1/validation.json")


def interval_excludes_zero(interval: Mapping[str, Any]) -> bool:
    return float(interval["ci_low"]) > 0.0 or float(interval["ci_high"]) < 0.0


def _numeric_difference(observed: Mapping[str, Any], expected: Mapping[str, Any]) -> float:
    if set(observed) != set(expected):
        raise ValueError(
            f"Metric key mismatch: observed={sorted(observed)}, expected={sorted(expected)}"
        )
    maximum = 0.0
    for key in observed:
        left = observed[key]
        right = expected[key]
        if not isinstance(left, (int, float)) or not isinstance(right, (int, float)):
            raise TypeError(f"Metric {key!r} is not numeric")
        maximum = max(maximum, abs(float(left) - float(right)))
    return maximum


def _position_digest(puzzles: Mapping[str, np.ndarray], rows: np.ndarray) -> str:
    payload = "\n".join(str(puzzles["position_id"][row]) for row in rows).encode()
    return hashlib.sha256(payload).hexdigest()


def _paired_point(
    concept: str,
    labels: np.ndarray,
    prediction: np.ndarray,
) -> float:
    if concept == "piece_code":
        return float(np.mean(prediction.argmax(axis=1) == labels))
    if concept == "legal_destination":
        return float(np.mean((prediction >= 0.5) == labels))
    if concept == "opponent_attack_count":
        return -float(np.mean(np.abs(prediction - labels)))
    raise ValueError(f"Unknown concept {concept!r}")


def _all_finite(value: Any) -> bool:
    if isinstance(value, Mapping):
        return all(_all_finite(item) for item in value.values())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return all(_all_finite(item) for item in value)
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    return True


def _control_score(concept: str, metrics: Mapping[str, Any]) -> float:
    if concept == "piece_code":
        return float(metrics["accuracy"])
    if concept == "legal_destination":
        return float(metrics["auroc"])
    if concept == "opponent_attack_count":
        return float(metrics["r2"])
    raise ValueError(f"Unknown concept {concept!r}")


def validate_probe_result(run_dir: Path, corpus_manifest: Path) -> dict[str, Any]:
    run_path = run_dir.resolve(strict=True)
    checksums = verify_checksums(run_path)
    metrics = json.loads((run_path / "metrics.json").read_text(encoding="utf-8"))
    manifest = json.loads((run_path / "manifest.json").read_text(encoding="utf-8"))
    predictions = np.load(run_path / "predictions.npz", allow_pickle=False)
    corpus, loaded_corpus_manifest = load_public_corpus(corpus_manifest)
    puzzles = corpus["puzzles"]

    layers = tuple(int(layer) for layer in metrics["layers"])
    if layers != tuple(range(15)):
        raise ValueError(f"Expected all 15 BT4 layers, found {layers}")
    future_ply = int(metrics["future_ply_zero_indexed"])
    split_counts = metrics["split_counts"]
    rows = select_probe_rows(
        puzzles,
        future_ply=future_ply,
        fit_count=int(split_counts["fit"]),
        selection_count=int(split_counts["selection"]),
        evaluation_count=int(split_counts["evaluation"]),
    )
    evaluation_rows = rows["evaluation"]
    expected_ids = puzzles["position_id"][evaluation_rows]
    if not np.array_equal(predictions["evaluation_position_id"], expected_ids):
        raise RuntimeError("Saved evaluation IDs do not match deterministic row selection")
    if not np.all(puzzles["split_u8"][evaluation_rows] == 1):
        raise RuntimeError("Evaluation predictions contain non-development rows")
    if np.any(puzzles["split_u8"][evaluation_rows] == 2):
        raise RuntimeError("Test rows were opened")

    expected_split_hashes = {
        name: _position_digest(puzzles, selected) for name, selected in rows.items()
    }
    recorded_split_hashes = manifest["public_corpus_manifest"]["split_position_sha256"]
    if expected_split_hashes != recorded_split_hashes:
        raise RuntimeError("Manifest split identities differ from deterministic corpus selection")
    if (
        manifest["public_corpus_manifest"]["integrity_sha256"]
        != loaded_corpus_manifest["manifest_integrity"]["sha256"]
    ):
        raise RuntimeError("Run and local corpus manifests have different identities")

    weight_path = run_path / "probe_weights.safetensors"
    with safe_open(weight_path, framework="pt", device="cpu") as handle:
        weight_keys = list(handle.keys())
        weight_metadata = handle.metadata() or {}
    if weight_metadata.get("pickle") != "forbidden" or not weight_keys:
        raise RuntimeError("Probe weight bundle failed the pickle-free safetensors gate")

    concept_arrays = {
        "piece_code": puzzles["piece_code_i8"][evaluation_rows].reshape(-1),
        "legal_destination": puzzles["legal_destination_u8"][evaluation_rows].reshape(-1),
        "opponent_attack_count": puzzles["attack_count_theirs_u8"][evaluation_rows].reshape(-1),
    }
    metric_functions = {
        "piece_code": multiclass_metrics,
        "legal_destination": binary_metrics,
        "opponent_attack_count": regression_metrics,
    }
    layer_rows: list[dict[str, Any]] = []
    maximum_metric_difference = 0.0
    recomputed_metric_groups = 0
    concept_selectivity_passes = 0
    concept_selectivity_total = 0
    significant_concept_deltas: dict[str, list[int]] = {
        "piece_code": [],
        "legal_destination": [],
        "opponent_attack_count": [],
    }
    significant_lookahead_deltas: dict[str, list[int]] = {
        "destination_accuracy": [],
        "source_given_predicted_destination_accuracy": [],
        "joint_move_accuracy": [],
    }
    causal_policy_excludes_zero: list[dict[str, Any]] = []
    random_probe_objective_contains_zero = 0

    future_origin = puzzles["future_origin_token_root_i16"][evaluation_rows, future_ply]
    future_destination = puzzles["future_destination_token_root_i16"][evaluation_rows, future_ply]

    for layer in layers:
        output_row: dict[str, Any] = {"layer": layer}
        for concept, labels in concept_arrays.items():
            concept_report = metrics["concept_probes"][str(layer)][concept]
            model_points: dict[str, float] = {}
            for model_name in ("raw", "hero"):
                prediction = predictions[f"concept.{model_name}.L{layer}.{concept}"]
                recomputed = metric_functions[concept](labels, prediction)
                reported = concept_report[model_name]["evaluation"]
                difference = _numeric_difference(recomputed, reported)
                maximum_metric_difference = max(maximum_metric_difference, difference)
                recomputed_metric_groups += 1
                model_points[model_name] = _paired_point(concept, labels, prediction)
                primary = _control_score(concept, reported)
                permutation = _control_score(
                    concept,
                    concept_report[model_name]["label_permutation_control"],
                )
                baseline = _control_score(
                    concept,
                    concept_report["train_only_frequency_or_mean_baseline"],
                )
                concept_selectivity_passes += int(primary > max(permutation, baseline))
                concept_selectivity_total += 1
            paired = concept_report["paired_evaluation"]
            if not np.isclose(float(paired["raw"]), model_points["raw"], atol=1e-10):
                raise RuntimeError(f"Raw paired point mismatch: L{layer} {concept}")
            if not np.isclose(float(paired["hero"]), model_points["hero"], atol=1e-10):
                raise RuntimeError(f"Hero paired point mismatch: L{layer} {concept}")
            delta = model_points["hero"] - model_points["raw"]
            if not np.isclose(float(paired["delta_hero_minus_raw"]), delta, atol=1e-10):
                raise RuntimeError(f"Paired delta mismatch: L{layer} {concept}")
            if not float(paired["ci_low"]) <= delta <= float(paired["ci_high"]):
                raise RuntimeError(
                    f"Paired interval excludes its point estimate: L{layer} {concept}"
                )
            if interval_excludes_zero(paired):
                significant_concept_deltas[concept].append(layer)
            output_row[f"{concept}_raw"] = model_points["raw"]
            output_row[f"{concept}_hero"] = model_points["hero"]
            output_row[f"{concept}_delta"] = delta
            output_row[f"{concept}_ci_low"] = float(paired["ci_low"])
            output_row[f"{concept}_ci_high"] = float(paired["ci_high"])
            output_row[f"{concept}_raw_primary"] = _control_score(
                concept,
                concept_report["raw"]["evaluation"],
            )
            output_row[f"{concept}_hero_primary"] = _control_score(
                concept,
                concept_report["hero"]["evaluation"],
            )

        lookahead = metrics["lookahead"][str(layer)]
        lookahead_points: dict[str, dict[str, float]] = {}
        for model_name in ("raw", "hero"):
            destination_prediction = predictions[f"lookahead.{model_name}.L{layer}.destination"]
            source_prediction = predictions[f"lookahead.{model_name}.L{layer}.source"]
            points = {
                "destination_accuracy": float(
                    np.mean(destination_prediction == future_destination)
                ),
                "source_given_predicted_destination_accuracy": float(
                    np.mean(source_prediction == future_origin)
                ),
                "joint_move_accuracy": float(
                    np.mean(
                        (destination_prediction == future_destination)
                        & (source_prediction == future_origin)
                    )
                ),
            }
            reported_evaluation = lookahead[model_name]["evaluation"]
            reported_points = {
                "destination_accuracy": float(reported_evaluation["destination"]["accuracy"]),
                "source_given_predicted_destination_accuracy": float(
                    reported_evaluation["source_given_predicted_destination"]["accuracy"]
                ),
                "joint_move_accuracy": float(reported_evaluation["joint_move_accuracy"]),
            }
            maximum_metric_difference = max(
                maximum_metric_difference,
                max(abs(points[key] - reported_points[key]) for key in points),
            )
            recomputed_metric_groups += 1
            lookahead_points[model_name] = points
            permutation_accuracy = float(
                lookahead[model_name]["label_permutation_control"]["destination"]["accuracy"]
            )
            frequency_accuracy = float(
                lookahead[model_name]["destination_frequency_baseline"]["accuracy"]
            )
            concept_selectivity_passes += int(
                points["destination_accuracy"] > max(permutation_accuracy, frequency_accuracy)
            )
            concept_selectivity_total += 1

        for metric_name in significant_lookahead_deltas:
            paired = lookahead["paired_evaluation"][metric_name]
            raw = lookahead_points["raw"][metric_name]
            hero = lookahead_points["hero"][metric_name]
            delta = hero - raw
            if not (
                np.isclose(float(paired["raw"]), raw, atol=1e-10)
                and np.isclose(float(paired["hero"]), hero, atol=1e-10)
                and np.isclose(float(paired["delta_hero_minus_raw"]), delta, atol=1e-10)
            ):
                raise RuntimeError(f"Lookahead paired point mismatch: L{layer} {metric_name}")
            if not float(paired["ci_low"]) <= delta <= float(paired["ci_high"]):
                raise RuntimeError(
                    f"Lookahead interval excludes its point estimate: L{layer} {metric_name}"
                )
            if interval_excludes_zero(paired):
                significant_lookahead_deltas[metric_name].append(layer)
            output_row[f"lookahead_{metric_name}_raw"] = raw
            output_row[f"lookahead_{metric_name}_hero"] = hero
            output_row[f"lookahead_{metric_name}_delta"] = delta
            output_row[f"lookahead_{metric_name}_ci_low"] = float(paired["ci_low"])
            output_row[f"lookahead_{metric_name}_ci_high"] = float(paired["ci_high"])

        causal = metrics["causal_lookahead"][str(layer)]
        for model_name in ("raw", "hero"):
            report = causal[model_name]
            if not _all_finite(report):
                raise RuntimeError(f"Nonfinite causal metric: L{layer} {model_name}")
            objective = report["probe_direction_probe_objective_derivative"]
            if float(objective["lower"]) <= 0.0:
                raise RuntimeError(f"Probe objective causal gate failed: L{layer} {model_name}")
            random_objective = report["random_direction_probe_objective_derivative"]
            random_probe_objective_contains_zero += int(
                float(random_objective["lower"]) <= 0.0 <= float(random_objective["upper"])
            )
            policy = report["probe_direction_policy_derivative"]
            if float(policy["lower"]) > 0.0 or float(policy["upper"]) < 0.0:
                causal_policy_excludes_zero.append(
                    {
                        "layer": layer,
                        "model": model_name,
                        "mean": float(policy["mean"]),
                        "lower": float(policy["lower"]),
                        "upper": float(policy["upper"]),
                    }
                )
            output_row[f"causal_policy_{model_name}"] = float(policy["mean"])
            output_row[f"causal_policy_{model_name}_lower"] = float(policy["lower"])
            output_row[f"causal_policy_{model_name}_upper"] = float(policy["upper"])
        layer_rows.append(output_row)

    if maximum_metric_difference > 1e-8:
        raise RuntimeError(
            f"Prediction-level recomputation differs from metrics by {maximum_metric_difference}"
        )
    if not metrics["correctness_gates"]["passed"]:
        raise RuntimeError("Original run correctness gates did not pass")

    result: dict[str, Any] = {
        "schema_version": "bt4-probe-result-independent-validation-v1",
        "overall_assessment": "share with caveats",
        "run": {
            "run_id": manifest["run_id"],
            "metrics_sha256": checksums["metrics.json"],
            "input_bundle_sha256": manifest["cost"]["input_bundle_sha256"],
            "elapsed_seconds": manifest["elapsed_seconds"],
            "backend": manifest["cost"]["backend"],
            "layers": list(layers),
            "future_ply_zero_indexed": future_ply,
            "split_counts": split_counts,
        },
        "integrity": {
            "checksum_ledger_files": len(checksums),
            "corpus_integrity_match": True,
            "deterministic_split_hashes_match": True,
            "evaluation_ids_match": True,
            "evaluation_rows_all_development": True,
            "test_rows_evaluated": False,
            "safetensors_key_count": len(weight_keys),
            "pickle_free_weight_metadata": True,
            "all_causal_metrics_finite": True,
        },
        "recomputation": {
            "metric_groups_recomputed_from_saved_predictions": recomputed_metric_groups,
            "maximum_absolute_metric_difference": maximum_metric_difference,
            "paired_points_and_deltas_match": True,
            "paired_intervals_contain_point_estimates": True,
        },
        "controls": {
            "primary_beats_permutation_and_frequency_or_mean_control_count": (
                concept_selectivity_passes
            ),
            "primary_control_comparison_count": concept_selectivity_total,
            "probe_objective_lower_bound_positive_count": 2 * len(layers),
            "random_probe_objective_interval_contains_zero_count": (
                random_probe_objective_contains_zero
            ),
            "random_probe_objective_comparison_count": 2 * len(layers),
        },
        "paired_interval_excludes_zero_layers": {
            "concepts": significant_concept_deltas,
            "lookahead": significant_lookahead_deltas,
        },
        "causal_policy_intervals_excluding_zero": causal_policy_excludes_zero,
        "layer_metrics": layer_rows,
        "required_caveats": [
            "Exploratory nested-development analysis; the frozen test partition remains unopened.",
            "One training seed was used for learned probes; seed variance is not yet estimated.",
            "All-layer interval scans are uncorrected for multiple comparisons.",
            "Linear decodability does not by itself establish causal use or a search algorithm.",
            "Causal policy effects use only 16 development positions per model and layer.",
            "Third-ply labels are public puzzle continuations, not a newly standardized engine-PV benchmark.",
        ],
    }
    result["validation_sha256"] = content_identity(result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--corpus-manifest", type=Path, default=DEFAULT_CORPUS_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = validate_probe_result(args.run_dir, args.corpus_manifest)
    write_json_atomic(args.output, result)
    print(json.dumps({"output": str(args.output), **result["recomputation"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
