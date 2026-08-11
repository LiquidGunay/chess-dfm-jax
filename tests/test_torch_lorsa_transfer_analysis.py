from __future__ import annotations

import hashlib
import json
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest

from research.interpretability.artifacts import (
    content_identity,
    sha256_file,
    verify_checksums,
    write_checksums,
    write_json_atomic,
    write_jsonl_atomic,
    write_text_atomic,
)
from research.interpretability.sparse.lorsa_transfer_analysis import (
    paired_bootstrap_mean_ci,
    run_analysis,
)


def _source_row(
    index: int,
    *,
    raw_policy: float,
    hero_policy: float,
    raw_top1: bool,
    hero_top1: bool,
    group_id: str | None = None,
) -> dict[str, object]:
    return {
        "group_id": group_id or f"group-{index}",
        "position_id": f"position-{index}",
        "puzzle_id": f"puzzle-{index}",
        "raw": {
            "policy_js": raw_policy,
            "policy_top1_preserved": raw_top1,
            "reconstruction_normalized_mse": 0.10 * (index + 1),
            "reconstruction_cosine": 0.90 + 0.01 * index,
            "native_vs_lorsa_head_mean_pattern_js": 0.020 + 0.001 * index,
        },
        "hero": {
            "policy_js": hero_policy,
            "policy_top1_preserved": hero_top1,
            "reconstruction_normalized_mse": [0.20, 0.10, 0.35, 0.50][index],
            "reconstruction_cosine": 0.91 + 0.005 * index,
            "native_vs_lorsa_head_mean_pattern_js": 0.019 + 0.001 * index,
        },
        "raw_hero_transfer": {
            "support_jaccard": 0.50 + 0.10 * index,
            "native_matched_head_pattern_js": 0.004 + 0.001 * index,
            "lorsa_matched_head_pattern_js": 0.006 + 0.002 * index,
        },
    }


def _source_rows(*, duplicate_group: bool = False) -> list[dict[str, object]]:
    return [
        _source_row(
            0,
            raw_policy=0.01,
            hero_policy=0.02,
            raw_top1=True,
            hero_top1=True,
        ),
        _source_row(
            1,
            raw_policy=0.03,
            hero_policy=0.01,
            raw_top1=True,
            hero_top1=False,
            group_id="group-0" if duplicate_group else None,
        ),
        _source_row(
            2,
            raw_policy=0.01,
            hero_policy=0.04,
            raw_top1=False,
            hero_top1=True,
        ),
        _source_row(
            3,
            raw_policy=0.04,
            hero_policy=0.04,
            raw_top1=False,
            hero_top1=False,
        ),
    ]


def _source_metrics(position_count: int) -> dict[str, object]:
    return {
        "schema_version": "bt4-published-lorsa-transfer-metrics-v1",
        "position_count": position_count,
        "artifact_contract_gate": {
            "passed": True,
            "criteria": {"reviewed_conversion": True, "upstream_commit_pinned": True},
        },
        "source_compatibility_gate": {
            "passed": True,
            "criteria": {"artifact_contract_passed": True, "normalized_mse_le_0.25": True},
        },
        "interpretation_contract": {
            "development_split_only": True,
            "test_split_opened": False,
            "exploratory": True,
            "lorsa_frozen": True,
            "hero_transfer_interpretable": True,
        },
        "source_stage": {
            "status": "complete",
            "reconstruction": {"normalized_mse": 0.12},
        },
        "transfer_stage": {
            "status": "complete",
            "hero": {"reconstruction": {"normalized_mse": 0.15}},
            "raw_hero": {
                "degradation": {
                    "normalized_mse_delta_hero_minus_raw": 0.03,
                }
            },
        },
    }


def _source_manifest(
    metrics: dict[str, object], rows: list[dict[str, object]]
) -> dict[str, object]:
    source_files = {"synthetic_transfer.py": "b" * 64}
    payloads = {
        "base_input_bundle": None,
        "hero": "c" * 64,
        "lorsa": "d" * 64,
        "lorsa_input_bundle": None,
        "metrics": content_identity(metrics),
        "positions": hashlib.sha256(
            "\n".join(str(row["position_id"]) for row in rows).encode()
        ).hexdigest(),
        "raw": "e" * 64,
        "source_files": content_identity(source_files),
    }
    return {
        "schema_version": "bt4-published-lorsa-transfer-run-v1",
        "run_id": f"sha256:{content_identity(payloads)}",
        "payload_identities": payloads,
        "source_files": source_files,
        "experiment": {
            "position_count": len(rows),
            "sequential_source_gate": True,
            "confirmatory": False,
        },
        "public_corpus_manifest": {"split": "development only"},
    }


def _write_source_bundle(path: Path, *, duplicate_group: bool = False) -> None:
    path.mkdir()
    rows = _source_rows(duplicate_group=duplicate_group)
    metrics = _source_metrics(len(rows))
    manifest = _source_manifest(metrics, rows)
    write_json_atomic(path / "metrics.json", metrics)
    write_json_atomic(path / "manifest.json", manifest)
    write_jsonl_atomic(path / "examples.jsonl", rows)
    write_text_atomic(path / "run.log", "synthetic transfer fixture\n")
    write_checksums(path)


def _reseal_source_bundle(path: Path, *, recompute_run_id: bool = True) -> None:
    metrics = json.loads((path / "metrics.json").read_text(encoding="utf-8"))
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    rows = [
        json.loads(line)
        for line in (path / "examples.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    payloads = manifest["payload_identities"]
    payloads["metrics"] = content_identity(metrics)
    payloads["positions"] = hashlib.sha256(
        "\n".join(str(row["position_id"]) for row in rows).encode()
    ).hexdigest()
    if "source_files" in payloads:
        payloads["source_files"] = content_identity(manifest["source_files"])
    if recompute_run_id:
        manifest["run_id"] = f"sha256:{content_identity(payloads)}"
    write_json_atomic(path / "metrics.json", metrics)
    write_json_atomic(path / "manifest.json", manifest)
    write_checksums(path)


def _args(source: Path, output: Path) -> Namespace:
    return Namespace(
        input=source,
        output=output,
        bootstrap_replicates=400,
        seed=17,
        tail_count=1,
    )


def test_bootstrap_is_deterministic_across_chunk_sizes() -> None:
    deltas = np.asarray([-0.2, 0.1, 0.3, 0.5], dtype=np.float64)
    expected = paired_bootstrap_mean_ci(
        deltas, replicates=500, seed=9, chunk_size=500
    )
    assert paired_bootstrap_mean_ci(
        deltas, replicates=500, seed=9, chunk_size=7
    ) == expected
    with pytest.raises(ValueError, match="replicates"):
        paired_bootstrap_mean_ci(deltas, replicates=0, seed=9)


def test_analysis_seals_paired_estimands_identity_and_top1_contingency(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    _write_source_bundle(source)
    first_output = tmp_path / "first"
    first = run_analysis(_args(source, first_output))
    assert first["position_count"] == 4
    verified = verify_checksums(first_output)
    assert set(verified) == {"examples.jsonl", "manifest.json", "metrics.json", "run.log"}

    metrics = json.loads((first_output / "metrics.json").read_text(encoding="utf-8"))
    manifest = json.loads((first_output / "manifest.json").read_text(encoding="utf-8"))
    nmse = metrics["reconstruction_normalized_mse"]
    assert nmse["estimands_are_not_interchangeable"] is True
    assert nmse["global_energy_weighted"]["raw"] == 0.12
    assert nmse["global_energy_weighted"]["hero"] == 0.15
    assert nmse["global_energy_weighted"]["hero_minus_raw"] == 0.03
    assert nmse["global_energy_weighted"]["bootstrap_ci"] is None
    assert nmse["mean_per_position"]["raw_mean"] == pytest.approx(0.25)
    assert nmse["mean_per_position"]["hero_mean"] == pytest.approx(0.2875)
    assert nmse["mean_per_position"]["hero_minus_raw_mean"] == pytest.approx(0.0375)
    assert nmse["mean_per_position"]["bootstrap_seed"] == 17

    contingency = metrics["policy_top1_contingency"]
    assert contingency["both_preserved"] == 1
    assert contingency["raw_only_preserved"] == 1
    assert contingency["hero_only_preserved"] == 1
    assert contingency["neither_preserved"] == 1
    assert contingency["raw_preserved"] == contingency["hero_preserved"] == 2
    assert contingency["paired_exact_test"] == {
        "discordant_pairs": 2,
        "exploratory": True,
        "hero_only_successes": 1,
        "method": "two-sided exact conditional binomial test (exact McNemar)",
        "null_probability": 0.5,
        "p_value": 1.0,
    }

    assert manifest["input_artifact"]["run_id"].startswith("sha256:")
    assert manifest["input_artifact"]["checksum_ledger_sha256"] == sha256_file(
        source / "checksums.sha256"
    )
    validated = manifest["input_artifact"]["validated_transfer_contract"]
    assert validated["all_required_gates_passed"] is True
    assert validated["source_files_bound_in_run_id"] is True
    assert validated["run_id_recomputed"] == manifest["input_artifact"]["run_id"]
    assert manifest["group_identity"]["groups"]["all_unique"] is True
    tails = [
        json.loads(line)
        for line in (first_output / "examples.jsonl").read_text().splitlines()
    ]
    assert [row["tail"] for row in tails] == [
        "largest_hero_policy_degradation",
        "largest_hero_policy_improvement",
    ]
    assert all(row["rank_within_tail"] == 1 for row in tails)

    second_output = tmp_path / "second"
    second = run_analysis(_args(source, second_output))
    assert second["run_id"] == first["run_id"]
    assert (second_output / "metrics.json").read_bytes() == (
        first_output / "metrics.json"
    ).read_bytes()
    with pytest.raises(FileExistsError, match="immutable"):
        run_analysis(_args(source, first_output))


def test_analysis_rejects_duplicate_groups_and_checksum_tampering(tmp_path: Path) -> None:
    duplicate = tmp_path / "duplicate"
    _write_source_bundle(duplicate, duplicate_group=True)
    with pytest.raises(ValueError, match="duplicate groups"):
        run_analysis(_args(duplicate, tmp_path / "duplicate-output"))
    assert not list(tmp_path.glob(".duplicate-output.*.partial"))

    tampered = tmp_path / "tampered"
    _write_source_bundle(tampered)
    with (tampered / "examples.jsonl").open("a", encoding="utf-8") as handle:
        handle.write("{}\n")
    with pytest.raises(ValueError, match="Checksum mismatch"):
        run_analysis(_args(tampered, tmp_path / "tampered-output"))
    assert not list(tmp_path.glob(".tampered-output.*.partial"))


@pytest.mark.parametrize("forgery", ("run_id", "schema", "gate", "boundary"))
def test_analysis_rejects_resealed_forged_transfer_contracts(
    tmp_path: Path, forgery: str
) -> None:
    source = tmp_path / forgery
    _write_source_bundle(source)
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    metrics = json.loads((source / "metrics.json").read_text(encoding="utf-8"))
    rows = [
        json.loads(line)
        for line in (source / "examples.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    if forgery == "run_id":
        manifest["run_id"] = f"sha256:{'f' * 64}"
        write_json_atomic(source / "manifest.json", manifest)
        _reseal_source_bundle(source, recompute_run_id=False)
        expected = "run_id"
    elif forgery == "schema":
        manifest["schema_version"] = "forged-transfer-v1"
        write_json_atomic(source / "manifest.json", manifest)
        _reseal_source_bundle(source)
        expected = "manifest schema"
    elif forgery == "gate":
        metrics["artifact_contract_gate"]["passed"] = False
        metrics["artifact_contract_gate"]["criteria"]["reviewed_conversion"] = False
        write_json_atomic(source / "metrics.json", metrics)
        _reseal_source_bundle(source)
        expected = "did not pass"
    else:
        rows[0]["raw_hero_transfer"]["support_jaccard"] = 1.1
        write_jsonl_atomic(source / "examples.jsonl", rows)
        _reseal_source_bundle(source)
        expected = "support_jaccard"
    write_checksums(source)
    with pytest.raises(ValueError, match=expected):
        run_analysis(_args(source, tmp_path / f"{forgery}-output"))
