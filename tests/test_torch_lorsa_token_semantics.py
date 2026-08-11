from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from research.interpretability.artifacts import (
    content_identity,
    verify_checksums,
    write_checksums,
    write_json_atomic,
    write_text_atomic,
)
from research.interpretability.sparse.feature_semantics import (
    SparseConceptSpec,
    SparseFeatureConceptAccumulator,
)
from research.interpretability.sparse.lorsa_token_semantics import (
    SOURCE_TRANSFER_METRICS_SCHEMA,
    SOURCE_TRANSFER_RUN_SCHEMA,
    build_semantic_reports,
    exact_token_concepts,
    labels_for_rows,
    seal_semantic_run,
    select_evidence_gated_feature_concepts,
    select_group_disjoint_development_rows,
    validate_sealed_source_transfer,
)


def _concept_puzzles(count: int = 3) -> dict[str, np.ndarray]:
    puzzles: dict[str, np.ndarray] = {}
    for binding in exact_token_concepts():
        puzzles[binding.array] = np.zeros((count, 64), dtype=np.uint8)
    return puzzles


def test_exact_token_labels_include_preregistered_concepts_and_attack_bins() -> None:
    puzzles = _concept_puzzles()
    puzzles["piece_code_i8"][1, :5] = np.asarray([0, 1, 6, 7, 12])
    puzzles["legal_origin_u8"][1, 3] = 1
    puzzles["attack_count_ours_u8"][1, :6] = np.asarray([0, 1, 2, 3, 4, 9])
    puzzles["attack_count_theirs_u8"][1, :6] = np.asarray([9, 4, 3, 2, 1, 0])

    labels = labels_for_rows(puzzles, np.asarray([1], dtype=np.int64))

    assert set(labels) == {binding.spec.name for binding in exact_token_concepts()}
    assert labels["piece_code"][0, :5].tolist() == [0, 1, 6, 7, 12]
    assert labels["legal_origin"][0, 3].item() == 1
    assert labels["attack_count_ours"][0, :6].tolist() == [0, 1, 2, 3, 4, 4]
    assert labels["attack_count_theirs"][0, :6].tolist() == [4, 4, 3, 2, 1, 0]


def test_development_selection_is_deterministic_group_disjoint_and_test_closed() -> None:
    puzzles = {
        "split_u8": np.asarray([0, 1, 1, 1, 1, 1, 1, 1, 1, 2], dtype=np.uint8),
        "position_id": np.asarray([f"p{index}" for index in range(10)]),
        "group_id": np.asarray(
            ["train", "a", "a", "b", "c", "d", "e", "f", "g", "test"]
        ),
    }
    first = select_group_disjoint_development_rows(
        puzzles,
        position_count=6,
        fit_fraction=0.5,
        seed=17,
    )
    second = select_group_disjoint_development_rows(
        puzzles,
        position_count=6,
        fit_fraction=0.5,
        seed=17,
    )

    np.testing.assert_array_equal(first.fit_rows, second.fit_rows)
    np.testing.assert_array_equal(first.heldout_rows, second.heldout_rows)
    assert first.metadata == second.metadata
    fit_groups = set(puzzles["group_id"][first.fit_rows])
    heldout_groups = set(puzzles["group_id"][first.heldout_rows])
    assert fit_groups.isdisjoint(heldout_groups)
    assert bool((puzzles["split_u8"][first.fit_rows] == 1).all())
    assert bool((puzzles["split_u8"][first.heldout_rows] == 1).all())
    assert first.metadata["test_rows_selected"] == 0


def _semantic_batch(*, reverse: bool = False) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    labels = torch.tensor([[0, 1, 0, 1], [1, 0, 1, 0]], dtype=torch.int64)
    features = torch.zeros((2, 4, 3), dtype=torch.float32)
    if reverse:
        features[..., 0] = 1 - labels
        features[..., 1] = labels
    else:
        features[..., 0] = labels
        features[..., 1] = 1 - labels
    return features, {"occupied": labels}


def test_semantic_report_freezes_raw_fit_pairs_and_records_negative_hero_result() -> None:
    concepts = (SparseConceptSpec("occupied", ("empty", "occupied")),)
    raw_fit = SparseFeatureConceptAccumulator(3, concepts)
    raw_heldout = SparseFeatureConceptAccumulator(3, concepts)
    hero_heldout = SparseFeatureConceptAccumulator(3, concepts)
    features, labels = _semantic_batch()
    reverse, reverse_labels = _semantic_batch(reverse=True)
    raw_fit.update(features, labels)
    raw_heldout.update(features, labels)
    hero_heldout.update(reverse, reverse_labels)

    metrics, selection = build_semantic_reports(
        raw_fit=raw_fit,
        raw_heldout=raw_heldout,
        hero_heldout=hero_heldout,
        top_per_direction_per_class=1,
        minimum_fit_support=2,
        minimum_positive_fit_joint_support=2,
        minimum_negative_fit_expected_joint_support=2.0,
        minimum_evaluation_support=2,
        minimum_positive_evaluation_joint_support=2,
        minimum_negative_evaluation_expected_joint_support=2.0,
        smoothing=0.5,
    )

    assert selection["selection_source"] == "Raw fit only"
    assert selection["evaluation_reselection"] is False
    assert selection["pair_count"] == 4
    assert metrics["selection"]["selected_positive_fit_pair_count"] == 2
    assert metrics["selection"]["selected_negative_fit_pair_count"] == 2
    assert metrics["raw_heldout"]["direction_replicated_count"] == 4
    assert metrics["hero_heldout"]["direction_replicated_count"] == 0
    assert metrics["negative_and_null_results"][
        "hero_heldout_direction_nonreplicated_supported_count"
    ] == 2
    assert metrics["negative_and_null_results"][
        "hero_heldout_unsupported_pair_count"
    ] == 2
    assert metrics["claim_gates"]["feature_causality_permitted"] is False
    assert metrics["claim_gates"]["raw_hero_semantic_equivalence_permitted"] is False
    assert all("raw_heldout" in row and "hero_heldout" in row for row in selection["pairs"])


def test_evidence_selector_rejects_tiny_positive_and_negative_class_evidence() -> None:
    concepts = (SparseConceptSpec("rare", ("common", "rare")),)
    accumulator = SparseFeatureConceptAccumulator(2, concepts)
    labels = torch.zeros((1, 20), dtype=torch.int64)
    labels[0, 0] = 1
    features = torch.zeros((1, 20, 2), dtype=torch.float32)
    features[0, 0, 0] = 1.0  # Positive, but only one observed rare joint event.
    features[0, 1:11, 1] = 1.0  # Negative, but only 0.5 rare events expected.
    accumulator.update(features, {"rare": labels})

    selected = select_evidence_gated_feature_concepts(
        accumulator,
        top_per_direction_per_class=2,
        minimum_feature_support=1,
        minimum_positive_joint_support=2,
        minimum_negative_expected_joint_support=2.0,
        smoothing=0.5,
    )

    assert not any(
        row["class_name"] == "rare" and row["feature_index"] in {0, 1}
        for row in selected
    )


def _expected_identities() -> dict[str, str]:
    return {
        "corpus_integrity": "a" * 64,
        "lorsa_weights": "b" * 64,
        "lorsa_manifest": "c" * 64,
        "raw": "d" * 64,
        "hero": "e" * 64,
    }


def _write_source_transfer(
    root: Path,
    *,
    source_passed: bool = True,
    source_bound: bool = True,
) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    root.mkdir()
    identities = _expected_identities()
    examples = [
        {
            "position_id": "p1",
            "group_id": "g1",
            "raw": {},
            "hero": {},
            "raw_hero_transfer": {},
        },
        {
            "position_id": "p2",
            "group_id": "g2",
            "raw": {},
            "hero": {},
            "raw_hero_transfer": {},
        },
    ]
    metrics = {
        "schema_version": SOURCE_TRANSFER_METRICS_SCHEMA,
        "position_count": 2,
        "artifact_contract_gate": {"passed": True},
        "source_compatibility_gate": {"passed": source_passed},
        "source_stage": {"status": "complete"},
        "transfer_stage": {"status": "complete"},
        "interpretation_contract": {
            "development_split_only": True,
            "test_split_opened": False,
            "lorsa_frozen": True,
            "hero_transfer_interpretable": True,
        },
    }
    positions_identity = hashlib.sha256(b"p1\np2").hexdigest()
    payloads = {
        "metrics": content_identity(metrics),
        "positions": positions_identity,
        "lorsa": identities["lorsa_weights"],
        "raw": identities["raw"],
        "hero": identities["hero"],
        "base_input_bundle": None,
        "lorsa_input_bundle": None,
    }
    source_files = {"synthetic_transfer.py": "f" * 64}
    if source_bound:
        payloads["source_files"] = content_identity(source_files)
    manifest = {
        "schema_version": SOURCE_TRANSFER_RUN_SCHEMA,
        "run_id": f"sha256:{content_identity(payloads)}",
        "payload_identities": payloads,
        "public_corpus_manifest": {
            "integrity_sha256": identities["corpus_integrity"],
            "split": "development only",
        },
        "experiment": {"position_count": 2, "layer": 14},
        "model_lattice": {
            "raw_asset_sha256": identities["raw"],
            "hero_state_sha256": identities["hero"],
        },
        "lorsa": {
            "parameters_frozen": True,
            "weights_sha256": identities["lorsa_weights"],
            "weights_sha256_before": identities["lorsa_weights"],
            "weights_sha256_after": identities["lorsa_weights"],
            "manifest_sha256": identities["lorsa_manifest"],
            "main_environment_pickle_loaded": False,
            "architecture": {"n_ov_heads": 16_384, "n_ctx": 64},
        },
    }
    if source_bound:
        manifest["source_files"] = source_files
    write_json_atomic(root / "manifest.json", manifest)
    write_json_atomic(root / "metrics.json", metrics)
    lines = "".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
        for row in examples
    )
    write_text_atomic(root / "examples.jsonl", lines)
    write_text_atomic(root / "run.log", "synthetic sealed source transfer\n")
    write_checksums(root)
    puzzles = {
        "split_u8": np.asarray([1, 1, 2], dtype=np.uint8),
        "position_id": np.asarray(["p1", "p2", "reserved-test"]),
        "group_id": np.asarray(["g1", "g2", "reserved-test-group"]),
    }
    return puzzles, identities


def test_source_transfer_verification_is_sealed_exact_and_fail_closed(
    tmp_path: Path,
) -> None:
    puzzles, identities = _write_source_transfer(tmp_path / "passed")
    gate = validate_sealed_source_transfer(
        tmp_path / "passed",
        expected_identities=identities,
        puzzles=puzzles,
    )
    assert gate["passed"]
    assert all(gate["criteria"].values())

    legacy_puzzles, legacy_identities = _write_source_transfer(
        tmp_path / "legacy", source_bound=False
    )
    legacy = validate_sealed_source_transfer(
        tmp_path / "legacy",
        expected_identities=legacy_identities,
        puzzles=legacy_puzzles,
    )
    assert not legacy["passed"]
    assert legacy["source_files_bound_in_run_id"] is False
    assert legacy["criteria"]["source_files_bound_in_run_id"] is False
    assert legacy["criteria"]["source_files_manifest_well_formed"] is False
    assert legacy["criteria"]["source_files_identity_matches"] is False

    failed_puzzles, failed_identities = _write_source_transfer(
        tmp_path / "failed", source_passed=False
    )
    failed = validate_sealed_source_transfer(
        tmp_path / "failed",
        expected_identities=failed_identities,
        puzzles=failed_puzzles,
    )
    assert not failed["passed"]
    assert not failed["criteria"]["source_compatibility_passed"]

    (tmp_path / "passed" / "run.log").write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Checksum mismatch"):
        validate_sealed_source_transfer(
            tmp_path / "passed",
            expected_identities=identities,
            puzzles=puzzles,
        )


def test_source_transfer_current8_rejects_forged_source_file_identity(
    tmp_path: Path,
) -> None:
    transfer = tmp_path / "forged-current8"
    puzzles, identities = _write_source_transfer(transfer)
    manifest = json.loads((transfer / "manifest.json").read_text(encoding="utf-8"))
    manifest["source_files"]["synthetic_transfer.py"] = "0" * 64
    write_json_atomic(transfer / "manifest.json", manifest)
    write_checksums(transfer)

    gate = validate_sealed_source_transfer(
        transfer,
        expected_identities=identities,
        puzzles=puzzles,
    )
    assert not gate["passed"]
    assert gate["criteria"]["run_identity_matches"] is True
    assert gate["source_files_bound_in_run_id"] is True
    assert gate["criteria"]["source_files_bound_in_run_id"] is True
    assert gate["criteria"]["source_files_manifest_well_formed"] is True
    assert gate["criteria"]["source_files_identity_matches"] is False

def test_semantic_seal_is_storage_neutral_checksumed_and_immutable(tmp_path: Path) -> None:
    output = tmp_path / "semantic"
    metrics = {"status": "complete_descriptive"}
    selection = {"pair_count": 1}
    positions = {"positions": [{"position_id": "p1"}]}
    manifest = {"run_id": "sha256:" + "f" * 64}
    result = seal_semantic_run(
        output=output,
        metrics=metrics,
        selection=selection,
        positions=positions,
        manifest=manifest,
        logs=["features were streamed"],
    )

    checksums = verify_checksums(output)
    assert set(checksums) == {
        "manifest.json",
        "metrics.json",
        "positions.json",
        "run.log",
        "selection.json",
    }
    assert not any(
        path.suffix in {".pt", ".pth", ".npy", ".npz", ".safetensors"}
        for path in output.rglob("*")
        if path.is_file()
    )
    assert result["selected_pair_count"] == 1
    with pytest.raises(FileExistsError, match="immutable"):
        seal_semantic_run(
            output=output,
            metrics=metrics,
            selection=selection,
            positions=positions,
            manifest=manifest,
            logs=[],
        )
