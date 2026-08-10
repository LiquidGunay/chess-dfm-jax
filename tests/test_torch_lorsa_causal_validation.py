from __future__ import annotations

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
from research.interpretability.sparse.lorsa_causal_validation import (
    CAUSAL_TARGETS,
    CalibrationAccumulator,
    build_decoder_interventions,
    choose_matched_random_controls,
    holm_adjust,
    paired_bootstrap_interval,
    paired_sign_flip_pvalue,
    seal_causal_run,
    validate_sealed_semantic_source,
)
from research.interpretability.sparse.lorsa_token_semantics import (
    TOKEN_SEMANTICS_METRICS_SCHEMA,
    TOKEN_SEMANTICS_POSITIONS_SCHEMA,
    TOKEN_SEMANTICS_RUN_SCHEMA,
    TOKEN_SEMANTICS_SELECTION_SCHEMA,
)


def test_decoder_interventions_apply_exact_feature_contributions() -> None:
    baseline = torch.tensor([[[10.0, 20.0], [30.0, 40.0]]])
    features = torch.zeros((1, 2, 4))
    features[0, 0, 0] = 2.0
    features[0, 0, 1] = 3.0
    decoder = torch.tensor(
        [[1.0, 2.0], [2.0, -1.0], [0.5, 0.5], [-1.0, 1.0]]
    )
    injection_mask = torch.tensor([[False, True]])

    names, values = build_decoder_interventions(
        baseline,
        features,
        decoder,
        target_feature=0,
        controls=(1,),
        injection_token_mask=injection_mask,
        reference_amplitudes={0: 4.0, 1: 5.0},
    )
    rows = dict(zip(names, values, strict=True))

    torch.testing.assert_close(rows["noop"], baseline)
    torch.testing.assert_close(
        rows["target_scale_0p0"][0, 0], torch.tensor([8.0, 16.0])
    )
    torch.testing.assert_close(
        rows["target_scale_2p0"][0, 0], torch.tensor([12.0, 24.0])
    )
    torch.testing.assert_close(
        rows["control_ablate_feature_1"][0, 0], torch.tensor([6.0, 22.0])
    )
    torch.testing.assert_close(
        rows["target_inject_1p0"][0, 1], torch.tensor([34.0, 48.0])
    )
    torch.testing.assert_close(
        rows["control_inject_feature_1"][0, 1], torch.tensor([38.0, 36.0])
    )


def test_matched_random_controls_are_seeded_active_and_exclude_targets() -> None:
    target_features = (0, 1)
    accumulator = CalibrationAccumulator.create(12, target_features)
    generator = torch.Generator().manual_seed(7)
    features = torch.rand((8, 4, 12), generator=generator)
    features = torch.where(features > 0.55, features, torch.zeros_like(features))
    features[..., 0] = torch.where(
        features[..., 2] > 0, features[..., 2] * 1.1, torch.zeros_like(features[..., 2])
    )
    features[..., 1] = torch.where(
        features[..., 3] > 0, features[..., 3] * 0.9, torch.zeros_like(features[..., 3])
    )
    accumulator.update(features, target_features)
    decoder_norm = np.linspace(0.8, 1.2, 12)

    first = choose_matched_random_controls(
        accumulator,
        decoder_norm,
        target_features=target_features,
        count=3,
        seed=19,
    )
    second = choose_matched_random_controls(
        accumulator,
        decoder_norm,
        target_features=target_features,
        count=3,
        seed=19,
    )

    assert first == second
    for rows in first.values():
        selected = [row["feature_index"] for row in rows]
        assert len(selected) == len(set(selected)) == 3
        assert not set(selected) & set(target_features)
        assert all(row["raw_fit_active_token_count"] > 0 for row in rows)


def test_paired_statistics_and_holm_are_deterministic() -> None:
    differences = [0.1, 0.2, 0.3, 0.4]
    first = paired_bootstrap_interval(differences, samples=2_000, seed=11)
    second = paired_bootstrap_interval(differences, samples=2_000, seed=11)
    assert first == second
    assert first["lower"] > 0
    assert paired_sign_flip_pvalue(differences, samples=5_000, seed=11) < 0.2

    adjusted = holm_adjust({"a": 0.01, "b": 0.03, "c": 0.2})
    assert adjusted == {"a": 0.03, "b": 0.06, "c": 0.2}


def _identities() -> dict[str, str]:
    return {
        "corpus_integrity": "a" * 64,
        "lorsa_weights": "b" * 64,
        "lorsa_manifest": "c" * 64,
        "raw": "d" * 64,
        "hero": "e" * 64,
    }


def _write_semantic_source(root: Path) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    root.mkdir()
    identities = _identities()
    position_rows = []
    for index in range(512):
        position_rows.append(
            {
                "position_id": f"p{index}",
                "group_id": f"g{index}",
                "public_split": "development",
                "role": "raw_fit" if index < 256 else "heldout_raw_and_hero",
            }
        )
    positions = {
        "schema_version": TOKEN_SEMANTICS_POSITIONS_SCHEMA,
        "positions": position_rows,
        "test_positions": [],
    }
    pairs = []
    for target in CAUSAL_TARGETS:
        concept, class_index, feature = target.pair_id.split(":")
        pairs.append(
            {
                "pair_id": target.pair_id,
                "concept": concept,
                "class_index": int(class_index),
                "class_name": f"class_{class_index}",
                "feature_index": int(feature.removeprefix("feature_")),
            }
        )
    selection = {
        "schema_version": TOKEN_SEMANTICS_SELECTION_SCHEMA,
        "selection_source": "Raw fit only",
        "evaluation_reselection": False,
        "pair_count": len(pairs),
        "pairs": pairs,
    }
    metrics = {
        "schema_version": TOKEN_SEMANTICS_METRICS_SCHEMA,
        "status": "complete_descriptive",
        "claim_gates": {
            "source_transfer_artifact_passed": True,
            "runtime_model_identities_match": True,
            "test_split_selected_or_evaluated": False,
        },
    }
    source_files = {"historical.py": "f" * 64}
    payloads = {
        "metrics": content_identity(metrics),
        "selection": content_identity(selection),
        "positions": content_identity(positions),
        "source_files": content_identity(source_files),
        "source_transfer_run": "sha256:" + "1" * 64,
    }
    manifest = {
        "schema_version": TOKEN_SEMANTICS_RUN_SCHEMA,
        "run_id": f"sha256:{content_identity(payloads)}",
        "payload_identities": payloads,
        "source_files": source_files,
        "source_identities": identities,
    }
    write_json_atomic(root / "manifest.json", manifest)
    write_json_atomic(root / "metrics.json", metrics)
    write_json_atomic(root / "selection.json", selection)
    write_json_atomic(root / "positions.json", positions)
    write_text_atomic(root / "run.log", "sealed semantic source\n")
    write_checksums(root)
    puzzles = {
        "split_u8": np.ones(512, dtype=np.uint8),
        "position_id": np.asarray([f"p{index}" for index in range(512)]),
        "group_id": np.asarray([f"g{index}" for index in range(512)]),
    }
    return puzzles, identities


def test_semantic_dependency_accepts_historical_source_map_and_rejects_tampering(
    tmp_path: Path,
) -> None:
    root = tmp_path / "semantic"
    puzzles, identities = _write_semantic_source(root)
    gate = validate_sealed_semantic_source(
        root, expected_identities=identities, puzzles=puzzles
    )
    assert gate["passed"]
    assert gate["current_source_equality_required"] is False

    (root / "run.log").write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Checksum mismatch"):
        validate_sealed_semantic_source(
            root, expected_identities=identities, puzzles=puzzles
        )


def test_causal_seal_is_checksumed_storage_neutral_and_immutable(tmp_path: Path) -> None:
    output = tmp_path / "causal"
    result = seal_causal_run(
        output=output,
        metrics={"status": "complete_exploratory_causal_surrogate"},
        design={"targets": []},
        positions={"positions": []},
        records=[{"position_id": "p1"}],
        manifest={"run_id": "sha256:" + "f" * 64},
        logs=["streamed and discarded tensors"],
    )
    assert result["record_count"] == 1
    assert set(verify_checksums(output)) == {
        "design.json",
        "interventions.jsonl",
        "manifest.json",
        "metrics.json",
        "positions.json",
        "run.log",
    }
    assert not any(
        path.suffix in {".npy", ".npz", ".pt", ".pth", ".safetensors"}
        for path in output.rglob("*")
        if path.is_file()
    )
    with pytest.raises(FileExistsError, match="immutable"):
        seal_causal_run(
            output=output,
            metrics={"status": "complete_exploratory_causal_surrogate"},
            design={},
            positions={},
            records=[],
            manifest={"run_id": "sha256:" + "f" * 64},
            logs=[],
        )
