from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

from research.interpretability.artifacts import (
    content_identity,
    write_checksums,
    write_json_atomic,
    write_jsonl_atomic,
    write_text_atomic,
)


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "research"
    / "interpretability"
    / "sparse"
    / "build_compact_feature_report.py"
)
SPEC = importlib.util.spec_from_file_location("staged_compact_feature_report", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
REPORT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = REPORT
SPEC.loader.exec_module(REPORT)


def _feature_summary(peak_token: int) -> dict[str, object]:
    return {
        "reported_feature_count": 1,
        "mean_token_l0": 1.0,
        "top_features": [
            {
                "feature": 0,
                "summed_activation": 2.0,
                "maximum_activation": 1.0,
                "active_token_count": 2,
                "peak_token": peak_token,
            }
        ],
    }


def _examples() -> list[dict[str, object]]:
    return [
        {
            "position_id": f"p{index}",
            "puzzle_id": f"z{index}",
            "group_id": f"g{index}",
            "target_action": index,
            "raw": {"features": _feature_summary(index)},
            "hero": {"features": _feature_summary(index)},
        }
        for index in range(2)
    ]


def _metrics() -> dict[str, object]:
    return {
        "schema_version": REPORT.SOURCE_METRICS_SCHEMA,
        "position_count": 2,
        "artifact_contract_gate": {"passed": True},
        "source_compatibility_gate": {"passed": True},
        "source_stage": {"status": "complete"},
        "transfer_stage": {"status": "complete"},
        "interpretation_contract": {
            "development_split_only": True,
            "exploratory": True,
            "hero_transfer_interpretable": True,
            "lorsa_frozen": True,
            "test_split_opened": False,
        },
    }


def _refresh_payloads(
    manifest: dict[str, object],
    metrics: dict[str, object],
    examples: list[dict[str, object]],
) -> None:
    payloads = manifest["payload_identities"]
    payloads["metrics"] = content_identity(metrics)
    if "source_files" in payloads:
        payloads["source_files"] = content_identity(manifest["source_files"])
    payloads["positions"] = hashlib.sha256(
        "\n".join(str(row["position_id"]) for row in examples).encode()
    ).hexdigest()
    manifest["run_id"] = "sha256:" + content_identity(payloads)


def _source_manifest(
    metrics: dict[str, object],
    examples: list[dict[str, object]],
) -> dict[str, object]:
    source_files = {"runner.py": "1" * 64}
    payloads = {
        "metrics": "",
        "positions": "",
        "lorsa": "2" * 64,
        "raw": "3" * 64,
        "hero": "4" * 64,
        "base_input_bundle": None,
        "lorsa_input_bundle": None,
    }
    manifest = {
        "schema_version": REPORT.SOURCE_MANIFEST_SCHEMA,
        "run_id": "",
        "payload_identities": payloads,
        "source_files": source_files,
        "model_lattice": {
            "raw_asset_sha256": payloads["raw"],
            "hero_state_sha256": payloads["hero"],
        },
        "lorsa": {
            "architecture": {"n_ov_heads": 4},
            "weights_sha256_before": payloads["lorsa"],
            "weights_sha256_after": payloads["lorsa"],
            "parameters_frozen": True,
        },
        "experiment": {
            "position_count": 2,
            "feature_summary_count": 1,
            "sequential_source_gate": True,
            "confirmatory": False,
        },
    }
    _refresh_payloads(manifest, metrics, examples)
    return manifest


def _seal_source(
    path: Path,
    mutation: str | None = None,
) -> Path:
    path.mkdir()
    examples = _examples()
    metrics = _metrics()
    manifest = _source_manifest(metrics, examples)
    if mutation in {"current8", "current8_source_files_identity"}:
        manifest["payload_identities"]["source_files"] = ""
        _refresh_payloads(manifest, metrics, examples)
    if mutation == "current8":
        pass
    elif mutation == "current8_source_files_identity":
        manifest["payload_identities"]["source_files"] = "0" * 64
        manifest["run_id"] = "sha256:" + content_identity(manifest["payload_identities"])
    elif mutation == "schema":
        metrics["schema_version"] = "wrong"
    elif mutation == "source_gate":
        metrics["source_compatibility_gate"]["passed"] = False
        _refresh_payloads(manifest, metrics, examples)
    elif mutation == "run_id":
        manifest["run_id"] = "sha256:" + "0" * 64
    elif mutation == "metrics_identity":
        manifest["payload_identities"]["metrics"] = "0" * 64
        manifest["run_id"] = "sha256:" + content_identity(manifest["payload_identities"])
    elif mutation == "position_identity":
        manifest["payload_identities"]["positions"] = "0" * 64
        manifest["run_id"] = "sha256:" + content_identity(manifest["payload_identities"])
    elif mutation is not None:
        raise AssertionError(f"Unknown test mutation: {mutation}")
    write_json_atomic(path / "metrics.json", metrics)
    write_json_atomic(path / "manifest.json", manifest)
    write_jsonl_atomic(path / "examples.jsonl", examples)
    write_text_atomic(path / "run.log", "synthetic source\n")
    write_checksums(path)
    return path


def test_source_contract_accepts_exact_payload_identities(tmp_path: Path) -> None:
    source = REPORT._load_source_artifact(_seal_source(tmp_path / "source"))

    assert source["position_count"] == 2
    assert source["group_count"] == 2
    assert source["manifest"]["run_id"] == ("sha256:" + source["payload_identity_sha256"])
    assert source["metrics_identity_sha256"] == content_identity(source["metrics"])
    assert source["source_files_bound_in_run_id"] is False


def test_source_contract_accepts_current_source_file_payload(tmp_path: Path) -> None:
    source = REPORT._load_source_artifact(_seal_source(tmp_path / "current8", "current8"))

    assert source["source_files_bound_in_run_id"] is True


@pytest.mark.parametrize(
    "mutation,error",
    [
        ("schema", "Unsupported source LoRSA metrics schema"),
        ("source_gate", "Raw source compatibility"),
        ("run_id", "run ID differs"),
        ("metrics_identity", "metrics logical identity drifted"),
        ("position_identity", "ordered-position identity drifted"),
        ("current8_source_files_identity", "Source-file logical identity drifted"),
    ],
)
def test_source_contract_fails_closed_on_tampering(
    tmp_path: Path,
    mutation: str,
    error: str,
) -> None:
    source = _seal_source(tmp_path / mutation, mutation)
    with pytest.raises(ValueError, match=error):
        REPORT._load_source_artifact(source)


def _seal_minimal_report(path: Path) -> Path:
    path.mkdir()
    analysis = {
        "status": "complete_no_selected_pairs",
        "summary": {"selected_pair_count": 0},
        "claim_gate": {
            "descriptive_peak_location_association": False,
            "token_level_feature_semantics": False,
            "feature_causality": False,
            "raw_hero_semantic_equivalence": False,
        },
        "split_contract": {"group_disjoint": True},
        "selection_contract": {"reselection_on_evaluation": False},
    }
    report = {
        "schema_version": REPORT.REPORT_BUNDLE_SCHEMA,
        "source_transfer": {"position_count": 0},
        "semantic_analysis": analysis,
    }
    manifest = {
        "schema_version": REPORT.REPORT_MANIFEST_SCHEMA,
        "immutable_after_creation": True,
        "report_identity_sha256": content_identity(report),
    }
    write_json_atomic(path / "report.json", report)
    write_json_atomic(path / "manifest.json", manifest)
    write_checksums(path)
    return path


def test_verify_only_reopens_sealed_report(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output = _seal_minimal_report(tmp_path / "sealed")

    sealed = REPORT.validate_sealed_report(output)
    assert sealed["report"]["semantic_analysis"]["status"] == ("complete_no_selected_pairs")
    assert REPORT.main(["--verify-only", str(output)]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["semantic_status"] == "complete_no_selected_pairs"


def test_build_refuses_existing_output_before_reading_inputs(tmp_path: Path) -> None:
    output = tmp_path / "immutable"
    output.mkdir()
    with pytest.raises(FileExistsError, match="Refusing to overwrite immutable output"):
        REPORT.build_sealed_report(
            tmp_path / "missing-source",
            tmp_path / "missing-corpus.json",
            output,
        )
