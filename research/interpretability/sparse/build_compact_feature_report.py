"""Build a sealed held-out Raw/Hero report from compact LoRSA summaries.

This builder intentionally supports only the narrow estimand retained by the
published transfer ledger: the square label at a reported top feature's peak
token.  It never promotes compact summaries to token-level feature semantics.
The source artifact and public corpus are verified before any analysis, the
Raw fit/evaluation split is group-disjoint, and evaluation never reselects.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from research.interpretability.artifacts import (
    content_identity,
    sha256_file,
    verify_checksums,
    write_checksums,
    write_json_atomic,
)
from research.interpretability.chessbench import load_public_corpus
from research.interpretability.sparse.compact_feature_semantics import (
    CompactPeakConceptSpec,
    build_compact_peak_feature_comparison,
    load_compact_examples,
)


REPORT_BUNDLE_SCHEMA = "bt4-lorsa-compact-peak-report-bundle-v1"
REPORT_MANIFEST_SCHEMA = "bt4-lorsa-compact-peak-report-manifest-v1"
SOURCE_MANIFEST_SCHEMA = "bt4-published-lorsa-transfer-run-v1"
SOURCE_METRICS_SCHEMA = "bt4-published-lorsa-transfer-metrics-v1"
REQUIRED_SOURCE_FILES = frozenset({"examples.jsonl", "manifest.json", "metrics.json", "run.log"})
DEFAULT_SOURCE_RUN = Path("research/analysis/published_lorsa_l14_transfer_dev512_v3")
DEFAULT_CORPUS_MANIFEST = Path("research/eval/interpretability_public_v2/manifest.json")
DEFAULT_OUTPUT_DIR = Path("research/analysis/published_lorsa_l14_compact_peak_semantics_dev512_v4")

ANALYSIS_CONFIGURATION: dict[str, Any] = {
    "fit_fraction": 0.5,
    "seed": 20260808,
    "top_per_class": 3,
    "minimum_fit_support": 16,
    "minimum_fit_joint_support": 5,
    "minimum_evaluation_support": 8,
    "smoothing": 0.5,
}

PIECE_CODE_NAMES = (
    "empty",
    "our_pawn",
    "our_knight",
    "our_bishop",
    "our_rook",
    "our_queen",
    "our_king",
    "their_pawn",
    "their_knight",
    "their_bishop",
    "their_rook",
    "their_queen",
    "their_king",
)

BINARY_CONCEPT_ARRAYS = (
    ("legal_origin", "legal_origin_u8"),
    ("legal_destination", "legal_destination_u8"),
    ("capture_destination", "capture_destination_u8"),
    ("checking_destination", "checking_destination_u8"),
    ("occupied_ours", "occupied_ours_u8"),
    ("occupied_theirs", "occupied_theirs_u8"),
    ("our_king_zone", "our_king_zone_u8"),
    ("their_king_zone", "their_king_zone_u8"),
    ("pinned_ours", "pinned_ours_u8"),
    ("pinned_theirs", "pinned_theirs_u8"),
    ("attacked_undefended_ours", "attacked_undefended_ours_u8"),
    ("attacked_undefended_theirs", "attacked_undefended_theirs_u8"),
)


def _load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _require_true(value: Any, *, label: str) -> None:
    if value is not True:
        raise ValueError(f"Required gate failed: {label}")


def _load_source_artifact(source_run: Path) -> dict[str, Any]:
    source = source_run.resolve(strict=True)
    if not source.is_dir():
        raise ValueError("Source LoRSA run is not a directory")
    checksums = verify_checksums(source)
    if set(checksums) != REQUIRED_SOURCE_FILES:
        raise ValueError("Source LoRSA file inventory is not the sealed four-file contract")

    manifest = _load_json_object(source / "manifest.json", label="Source manifest")
    metrics = _load_json_object(source / "metrics.json", label="Source metrics")
    if manifest.get("schema_version") != SOURCE_MANIFEST_SCHEMA:
        raise ValueError("Unsupported source LoRSA manifest schema")
    if metrics.get("schema_version") != SOURCE_METRICS_SCHEMA:
        raise ValueError("Unsupported source LoRSA metrics schema")
    payloads = manifest.get("payload_identities")
    if not isinstance(payloads, Mapping):
        raise ValueError("Source manifest has no payload identities")
    legacy_payload_names = {
        "metrics",
        "positions",
        "lorsa",
        "raw",
        "hero",
        "base_input_bundle",
        "lorsa_input_bundle",
    }
    current_payload_names = legacy_payload_names | {"source_files"}
    if set(payloads) not in (legacy_payload_names, current_payload_names):
        raise ValueError("Source payload identity inventory drifted")
    expected_run_id = "sha256:" + content_identity(payloads)
    source_files_bound_in_run_id = "source_files" in payloads
    if manifest.get("run_id") != expected_run_id:
        raise ValueError("Source manifest run ID differs from payload identities")
    if payloads["metrics"] != content_identity(metrics):
        raise ValueError("Source metrics logical identity drifted")
    _require_true(
        metrics.get("artifact_contract_gate", {}).get("passed"),
        label="artifact contract",
    )
    _require_true(
        metrics.get("source_compatibility_gate", {}).get("passed"),
        label="Raw source compatibility",
    )
    if metrics.get("source_stage", {}).get("status") != "complete":
        raise ValueError("Raw source stage is not complete")
    if metrics.get("transfer_stage", {}).get("status") != "complete":
        raise ValueError("Hero transfer stage is not complete")
    interpretation = metrics.get("interpretation_contract", {})
    for name in (
        "development_split_only",
        "exploratory",
        "hero_transfer_interpretable",
        "lorsa_frozen",
    ):
        _require_true(interpretation.get(name), label=f"interpretation.{name}")
    if interpretation.get("test_split_opened") is not False:
        raise ValueError("Source LoRSA artifact opened the frozen test split")

    examples = load_compact_examples(source / "examples.jsonl")
    experiment = manifest.get("experiment", {})
    position_count = experiment.get("position_count")
    if type(position_count) is not int or position_count != len(examples):
        raise ValueError("Source manifest position count differs from examples ledger")
    if metrics.get("position_count") != position_count:
        raise ValueError("Source metrics position count differs from manifest")
    if experiment.get("feature_summary_count", 0) <= 0:
        raise ValueError("Source artifact retained no compact feature summaries")
    if experiment.get("sequential_source_gate") is not True:
        raise ValueError("Source artifact did not gate Raw before Hero")
    if experiment.get("confirmatory") is not False:
        raise ValueError("Compact semantic report expects exploratory development data")
    n_features = manifest.get("lorsa", {}).get("architecture", {}).get("n_ov_heads")
    if type(n_features) is not int or n_features <= 0:
        raise ValueError("Source manifest has no valid LoRSA feature width")

    identifiers = [str(row.get("position_id", "")) for row in examples]
    if identifiers != sorted(identifiers) or len(set(identifiers)) != len(identifiers):
        raise ValueError("Source positions are not unique and monotonically selected")
    ordered_position_identity = hashlib.sha256("\n".join(identifiers).encode()).hexdigest()
    if payloads["positions"] != ordered_position_identity:
        raise ValueError("Source ordered-position identity drifted")
    source_files = manifest.get("source_files")
    if not isinstance(source_files, Mapping) or not source_files:
        raise ValueError("Source manifest has no source-file identities")
    if not all(
        isinstance(name, str)
        and isinstance(digest, str)
        and len(digest) == 64
        and all(character in "0123456789abcdef" for character in digest)
        for name, digest in source_files.items()
    ):
        raise ValueError("Source-file identity inventory is malformed")
    source_file_manifest_identity = content_identity(source_files)
    if source_files_bound_in_run_id and payloads["source_files"] != source_file_manifest_identity:
        raise ValueError("Source-file logical identity drifted")
    model_lattice = manifest.get("model_lattice", {})
    if payloads["raw"] != model_lattice.get("raw_asset_sha256"):
        raise ValueError("Source Raw model identity drifted")
    if payloads["hero"] != model_lattice.get("hero_state_sha256"):
        raise ValueError("Source Hero model identity drifted")
    lorsa = manifest.get("lorsa", {})
    if payloads["lorsa"] != lorsa.get("weights_sha256_before"):
        raise ValueError("Source LoRSA identity drifted")
    if lorsa.get("weights_sha256_after") != payloads["lorsa"]:
        raise ValueError("Source LoRSA weights changed during transfer")
    _require_true(lorsa.get("parameters_frozen"), label="LoRSA frozen parameters")
    groups = {str(row.get("group_id", "")) for row in examples}
    if "" in groups or len(groups) < 2:
        raise ValueError("Source examples cannot support a group-disjoint split")

    return {
        "path": source,
        "checksums": checksums,
        "checksum_ledger_sha256": sha256_file(source / "checksums.sha256"),
        "manifest": manifest,
        "metrics": metrics,
        "examples": examples,
        "position_count": position_count,
        "group_count": len(groups),
        "n_features": n_features,
        "metrics_identity_sha256": payloads["metrics"],
        "ordered_position_identity_sha256": ordered_position_identity,
        "source_file_manifest_identity_sha256": source_file_manifest_identity,
        "source_files_bound_in_run_id": source_files_bound_in_run_id,
        "payload_identity_sha256": content_identity(payloads),
    }


def _concepts(corpus_manifest: Mapping[str, Any]) -> tuple[CompactPeakConceptSpec, ...]:
    observed_piece_names = tuple(
        corpus_manifest.get("concept_labels", {}).get("piece_code_names", ())
    )
    if observed_piece_names != PIECE_CODE_NAMES:
        raise ValueError("Public corpus piece-code class contract drifted")
    return (
        CompactPeakConceptSpec("piece_code", "piece_code_i8", PIECE_CODE_NAMES),
        *(
            CompactPeakConceptSpec(name, array, ("no", "yes"))
            for name, array in BINARY_CONCEPT_ARRAYS
        ),
    )


def _validate_example_alignment(
    examples: Sequence[Mapping[str, Any]],
    puzzles: Mapping[str, np.ndarray],
) -> dict[str, str]:
    required = {"position_id", "group_id", "puzzle_id", "target_action_u16", "split_u8"}
    missing = sorted(required - set(puzzles))
    if missing:
        raise ValueError("Public corpus lacks identity arrays: " + ", ".join(missing))
    corpus_ids = np.asarray(puzzles["position_id"])
    if corpus_ids.ndim != 1 or len(set(map(str, corpus_ids))) != corpus_ids.size:
        raise ValueError("Public corpus position identities are not a unique vector")
    index_by_id = {str(value): index for index, value in enumerate(corpus_ids)}
    aligned_ids: list[str] = []
    aligned_groups: list[str] = []
    for example in examples:
        position_id = str(example["position_id"])
        if position_id not in index_by_id:
            raise ValueError(f"Source position {position_id!r} is absent from public corpus")
        index = index_by_id[position_id]
        expected_group = str(puzzles["group_id"][index])
        expected_puzzle = str(puzzles["puzzle_id"][index])
        if str(example["group_id"]) != expected_group:
            raise ValueError(f"Source group identity drift at {position_id}")
        if str(example["puzzle_id"]) != expected_puzzle:
            raise ValueError(f"Source puzzle identity drift at {position_id}")
        if int(example["target_action"]) != int(puzzles["target_action_u16"][index]):
            raise ValueError(f"Source target action drift at {position_id}")
        if int(puzzles["split_u8"][index]) != 1:
            raise ValueError("Compact report accepts development positions only")
        aligned_ids.append(position_id)
        aligned_groups.append(expected_group)
    return {
        "position_identity_sha256": content_identity({"position_id": aligned_ids}),
        "group_identity_sha256": content_identity({"group_id": sorted(set(aligned_groups))}),
    }


def _source_transfer_summary(metrics: Mapping[str, Any]) -> dict[str, Any]:
    raw = metrics["source_stage"]
    hero = metrics["transfer_stage"]["hero"]
    paired = metrics["transfer_stage"]["raw_hero"]
    return {
        "position_count": int(metrics["position_count"]),
        "raw": {
            "reconstruction_normalized_mse": float(raw["reconstruction"]["normalized_mse"]),
            "reconstruction_cosine": float(raw["reconstruction"]["cosine"]),
            "policy_js_mean": float(raw["policy_replacement"]["js_divergence"]["mean"]),
            "policy_js_maximum": float(
                raw["policy_replacement"]["js_divergence"]["maximum_absolute"]
            ),
            "policy_top1_true": int(raw["policy_replacement"]["top1_agreement"]["true_count"]),
        },
        "hero": {
            "reconstruction_normalized_mse": float(hero["reconstruction"]["normalized_mse"]),
            "reconstruction_cosine": float(hero["reconstruction"]["cosine"]),
            "policy_js_mean": float(hero["policy_replacement"]["js_divergence"]["mean"]),
            "policy_js_maximum": float(
                hero["policy_replacement"]["js_divergence"]["maximum_absolute"]
            ),
            "policy_top1_true": int(hero["policy_replacement"]["top1_agreement"]["true_count"]),
        },
        "hero_minus_raw": {
            "reconstruction_normalized_mse": float(
                paired["degradation"]["normalized_mse_delta_hero_minus_raw"]
            ),
            "policy_js_mean": float(paired["degradation"]["policy_js_delta_hero_minus_raw"]),
        },
        "raw_hero_sparse_transfer": {
            "mean_support_jaccard": float(paired["support_transfer"]["mean_support_jaccard"]),
            "mean_resolved_activation_correlation": float(
                paired["support_transfer"]["mean_resolved_activation_correlation"]
            ),
            "resolved_activation_correlations": int(
                paired["support_transfer"]["resolved_activation_correlations"]
            ),
        },
        "interpretation": (
            "Passing the source gate permits exploratory frozen-LoRSA transfer analysis; "
            "it does not establish feature semantics or head equivalence."
        ),
    }


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _row_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    raw_supported = [row for row in rows if row["raw_support_gate"]]
    hero_supported = [row for row in rows if row["hero_support_gate"]]
    paired = [row for row in rows if row["paired_support_gate"]]
    raw_replicated = sum(row["raw_direction_replicated_from_fit"] for row in raw_supported)
    hero_transferred = sum(row["hero_direction_transferred_from_raw_fit"] for row in hero_supported)
    agreement = sum(row["raw_hero_direction_agreement"] for row in paired)
    return {
        "selected": len(rows),
        "raw_supported": len(raw_supported),
        "hero_supported": len(hero_supported),
        "paired_supported": len(paired),
        "raw_fit_direction_replicated": raw_replicated,
        "raw_fit_direction_replication_rate": _rate(raw_replicated, len(raw_supported)),
        "hero_fit_direction_transferred": hero_transferred,
        "hero_fit_direction_transfer_rate": _rate(hero_transferred, len(hero_supported)),
        "raw_hero_direction_agreement": agreement,
        "raw_hero_direction_agreement_rate": _rate(agreement, len(paired)),
    }


def _notebook_summary(analysis: Mapping[str, Any]) -> dict[str, Any]:
    rows = list(analysis["pairs"])
    by_direction = {
        direction: _row_summary([row for row in rows if row["selection_direction"] == direction])
        for direction in ("positive", "negative")
    }
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["concept"])].append(row)
    by_concept = {concept: _row_summary(grouped[concept]) for concept in sorted(grouped)}
    paired = [row for row in rows if row["paired_support_gate"]]
    largest_deltas = sorted(
        paired,
        key=lambda row: (
            -abs(float(row["hero_minus_raw_log2_lift"])),
            str(row["concept"]),
            int(row["feature_index"]),
        ),
    )[:12]
    return {
        "status": analysis["status"],
        "claim_gate": analysis["claim_gate"],
        "split_contract": analysis["split_contract"],
        "selection_contract": analysis["selection_contract"],
        "coverage": analysis["coverage"],
        "overall": _row_summary(rows),
        "by_selection_direction": by_direction,
        "by_concept": by_concept,
        "negative_and_null_results": analysis["negative_and_null_results"],
        "largest_absolute_paired_deltas": [
            {
                "concept": row["concept"],
                "class": row["class_name"],
                "feature": row["feature_index"],
                "selection_direction": row["selection_direction"],
                "raw_evaluation_log2_lift": row["raw_evaluation_log2_lift"],
                "hero_evaluation_log2_lift": row["hero_evaluation_log2_lift"],
                "hero_minus_raw_log2_lift": row["hero_minus_raw_log2_lift"],
                "raw_support": row["raw_evaluation_feature_support"],
                "hero_support": row["hero_evaluation_feature_support"],
                "raw_hero_direction_agreement": row["raw_hero_direction_agreement"],
            }
            for row in largest_deltas
        ],
        "interpretation_contract": analysis["interpretation_contract"],
    }


def _validate_analysis_contract(analysis: Mapping[str, Any]) -> None:
    allowed_statuses = {
        "complete_descriptive",
        "complete_no_selected_pairs",
        "complete_no_paired_supported_pairs",
    }
    if analysis.get("status") not in allowed_statuses:
        raise ValueError("Compact semantic analysis did not complete")
    claim_gate = analysis.get("claim_gate", {})
    for forbidden in (
        "token_level_feature_semantics",
        "feature_causality",
        "raw_hero_semantic_equivalence",
    ):
        if claim_gate.get(forbidden) is not False:
            raise ValueError(f"Compact analysis weakened forbidden claim gate: {forbidden}")
    if analysis.get("split_contract", {}).get("group_disjoint") is not True:
        raise ValueError("Compact analysis did not establish group-disjoint evaluation")
    if analysis.get("selection_contract", {}).get("reselection_on_evaluation") is not False:
        raise ValueError("Compact analysis reselected on evaluation")


def validate_sealed_report(output_dir: Path) -> dict[str, Any]:
    output = output_dir.resolve(strict=True)
    checksums = verify_checksums(output)
    if set(checksums) != {"manifest.json", "report.json"}:
        raise ValueError("Compact report output inventory drifted")
    manifest = _load_json_object(output / "manifest.json", label="Report manifest")
    report = _load_json_object(output / "report.json", label="Compact report")
    if manifest.get("schema_version") != REPORT_MANIFEST_SCHEMA:
        raise ValueError("Unsupported compact report manifest schema")
    if report.get("schema_version") != REPORT_BUNDLE_SCHEMA:
        raise ValueError("Unsupported compact report bundle schema")
    if manifest.get("immutable_after_creation") is not True:
        raise ValueError("Compact report is not declared immutable")
    if manifest.get("report_identity_sha256") != content_identity(report):
        raise ValueError("Compact report logical identity drifted")
    _validate_analysis_contract(report["semantic_analysis"])
    return {
        "checksums": checksums,
        "manifest": manifest,
        "report": report,
    }


def build_sealed_report(
    source_run: Path,
    corpus_manifest_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    output = output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite immutable output: {output}")

    source = _load_source_artifact(source_run)
    corpus_manifest_path = corpus_manifest_path.resolve(strict=True)
    corpus, corpus_manifest = load_public_corpus(corpus_manifest_path)
    corpus_checksums = verify_checksums(corpus_manifest_path.parent)
    source_corpus = source["manifest"].get("public_corpus_manifest", {})
    corpus_integrity = corpus_manifest.get("manifest_integrity", {}).get("sha256")
    if not isinstance(corpus_integrity, str) or len(corpus_integrity) != 64:
        raise ValueError("Public corpus has no valid canonical integrity identity")
    if source_corpus.get("integrity_sha256") != corpus_integrity:
        raise ValueError("Source run and local public corpus identities differ")
    puzzles = corpus["puzzles"]
    alignment = _validate_example_alignment(source["examples"], puzzles)
    concepts = _concepts(corpus_manifest)
    if len(concepts) != 13:
        raise AssertionError("Exact compact concept contract must contain 13 concepts")

    analysis = build_compact_peak_feature_comparison(
        source["examples"],
        puzzles,
        n_features=source["n_features"],
        concepts=concepts,
        **ANALYSIS_CONFIGURATION,
    )
    _validate_analysis_contract(analysis)
    report = {
        "schema_version": REPORT_BUNDLE_SCHEMA,
        "source_transfer": _source_transfer_summary(source["metrics"]),
        "semantic_analysis": analysis,
        "notebook_summary": _notebook_summary(analysis),
    }

    builder_path = Path(__file__).resolve(strict=True)
    adapter_path = Path(
        __import__(
            "research.interpretability.sparse.compact_feature_semantics",
            fromlist=["__file__"],
        ).__file__
    ).resolve(strict=True)
    manifest = {
        "schema_version": REPORT_MANIFEST_SCHEMA,
        "immutable_after_creation": True,
        "report_identity_sha256": content_identity(report),
        "analysis_configuration": ANALYSIS_CONFIGURATION,
        "analysis_configuration_sha256": content_identity(ANALYSIS_CONFIGURATION),
        "concepts": [
            {
                "name": concept.name,
                "array": concept.array,
                "class_names": list(concept.class_names),
            }
            for concept in concepts
        ],
        "source_run": {
            "path": str(source["path"]),
            "schema_version": source["manifest"]["schema_version"],
            "run_id": source["manifest"].get("run_id"),
            "payload_identity_sha256": source["payload_identity_sha256"],
            "metrics_identity_sha256": source["metrics_identity_sha256"],
            "source_file_manifest_identity_sha256": source["source_file_manifest_identity_sha256"],
            "source_files_bound_in_run_id": source["source_files_bound_in_run_id"],
            "ordered_position_identity_sha256": source["ordered_position_identity_sha256"],
            "checksums": source["checksums"],
            "checksum_ledger_sha256": source["checksum_ledger_sha256"],
            "position_count": source["position_count"],
            "group_count": source["group_count"],
            "n_features": source["n_features"],
            "source_compatibility_gate_passed": True,
            "artifact_contract_gate_passed": True,
        },
        "public_corpus": {
            "manifest_path": str(corpus_manifest_path),
            "manifest_integrity_sha256": corpus_integrity,
            "checksums": corpus_checksums,
            "checksum_ledger_sha256": sha256_file(corpus_manifest_path.parent / "checksums.sha256"),
            **alignment,
        },
        "source_files": {
            "research/interpretability/sparse/build_compact_feature_report.py": sha256_file(
                builder_path
            ),
            "research/interpretability/sparse/compact_feature_semantics.py": sha256_file(
                adapter_path
            ),
        },
        "claim_gate": analysis["claim_gate"],
        "input_gates": {
            "source_checksum_inventory_verified": True,
            "source_run_id_verified": True,
            "source_metrics_identity_verified": True,
            "source_ordered_position_identity_verified": True,
            "source_file_manifest_identity_recorded": True,
            "source_files_bound_in_run_id": source["source_files_bound_in_run_id"],
            "source_artifact_contract_passed": True,
            "source_compatibility_gate_passed": True,
            "hero_transfer_complete": True,
            "public_corpus_checksum_inventory_verified": True,
            "public_corpus_identity_matches_source": True,
            "position_group_puzzle_action_alignment_verified": True,
            "development_split_only": True,
            "test_split_opened": False,
            "group_disjoint_fit_evaluation": analysis["split_contract"]["group_disjoint"],
            "evaluation_reselection": analysis["selection_contract"]["reselection_on_evaluation"],
        },
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.with_name(f".{output.name}.{os.getpid()}.partial")
    if staging.exists():
        raise FileExistsError(f"Staging path already exists: {staging}")
    staging.mkdir()
    try:
        write_json_atomic(staging / "report.json", report)
        write_json_atomic(staging / "manifest.json", manifest)
        write_checksums(staging)
        verify_checksums(staging)
        os.replace(staging, output)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return validate_sealed_report(output)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, default=DEFAULT_SOURCE_RUN)
    parser.add_argument("--corpus-manifest", type=Path, default=DEFAULT_CORPUS_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--verify-only",
        type=Path,
        help="Verify an existing compact report directory instead of building one",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.verify_only is not None:
        sealed = validate_sealed_report(args.verify_only)
        output = args.verify_only.resolve()
    else:
        sealed = build_sealed_report(
            args.source_run,
            args.corpus_manifest,
            args.output_dir,
        )
        output = args.output_dir.resolve()
    report = sealed["report"]
    print(
        json.dumps(
            {
                "output": str(output),
                "report_identity_sha256": sealed["manifest"]["report_identity_sha256"],
                "semantic_status": report["semantic_analysis"]["status"],
                "position_count": report["source_transfer"]["position_count"],
                "selected_pair_count": report["semantic_analysis"]["summary"][
                    "selected_pair_count"
                ],
                "claim_gate": report["semantic_analysis"]["claim_gate"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
