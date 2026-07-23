#!/usr/bin/env python3
"""Diagnose constructor versus post-restore NNX state schemas without training."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import jax
from flax import nnx

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from chess_dfm_jax.nnx_bt4 import TrainableParam  # noqa: E402
from research import train  # noqa: E402
from research.prepare import REPO_ROOT, require_within_workspace  # noqa: E402


CONSTRUCTOR_MODEL_ABI = (
    "f9f9bde96785c9b9bb24a3b12ac10bde16ec5761f501f3783fc45733c0f1b467"
)
POST_RESTORE_MODEL_ABI = (
    "b53b21a8113b74655bb897e8171315503d31f4434d33eb2b579d83bb38614d13"
)
CONSTRUCTOR_OPTIMIZER_ABI = (
    "6d45c99b53bf22d638ecdfe9f9a3cfae5f1a50d8ecfa4c54a41bb8c8b84b707e"
)
EXPECTED_MODEL_LEAVES = 455
EXPECTED_MODEL_BYTES = 705_987_352
EXPECTED_OPTIMIZER_LEAVES = 2_743
EXPECTED_OPTIMIZER_BYTES = 1_145_636_465

RUN_ROOT = REPO_ROOT / "checkpoints" / "source" / "step0265000"
MODELS_DIR = REPO_ROOT / "models" / "source" / "extracted"
REFERENCE_MANIFEST = (
    REPO_ROOT
    / "research"
    / "runs"
    / "future-tail1-balanced-k1-cosine-b128-30m-v1"
    / "checkpoints"
    / "update00002072"
    / "manifest.json"
)


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def compact_abi(abi: dict[str, Any]) -> dict[str, Any]:
    """Drop the full schema while retaining its identity and storage totals."""

    return {
        "sha256": abi["sha256"],
        "leaf_count": int(abi["leaf_count"]),
        "nbytes": int(abi["nbytes"]),
    }


def leaf_schema(abi: dict[str, Any]) -> list[dict[str, Any]]:
    """Return only path/shape/dtype/byte records from a full state ABI."""

    return [record for record in abi["schema"] if record["kind"] == "leaf"]


def leaf_schema_summary(abi: dict[str, Any]) -> dict[str, Any]:
    leaves = leaf_schema(abi)
    return {
        "sha256": _sha256_json(leaves),
        "leaf_count": len(leaves),
        "nbytes": sum(int(record["nbytes"]) for record in leaves),
    }


def _path_key(record: dict[str, Any]) -> bytes:
    return _canonical_json_bytes(record["path"])


def compare_schemas(
    before_abi: dict[str, Any],
    after_abi: dict[str, Any],
) -> dict[str, Any]:
    """Return every path-level difference between two full ABI schemas."""

    before = {_path_key(record): record for record in before_abi["schema"]}
    after = {_path_key(record): record for record in after_abi["schema"]}
    missing_keys = sorted(set(before) - set(after))
    extra_keys = sorted(set(after) - set(before))
    changed_keys = sorted(
        key
        for key in set(before) & set(after)
        if before[key] != after[key]
    )
    missing = [before[key] for key in missing_keys]
    extra = [after[key] for key in extra_keys]
    changed = [
        {
            "path": before[key]["path"],
            "before": before[key],
            "after": after[key],
        }
        for key in changed_keys
    ]
    leaf_difference_count = sum(
        record["kind"] == "leaf" for record in missing
    ) + sum(record["kind"] == "leaf" for record in extra)
    leaf_difference_count += sum(
        item["before"]["kind"] == "leaf"
        or item["after"]["kind"] == "leaf"
        for item in changed
    )
    return {
        "missing_count": len(missing),
        "extra_count": len(extra),
        "changed_count": len(changed),
        "leaf_difference_count": leaf_difference_count,
        "missing": missing,
        "extra": extra,
        "changed": changed,
    }


def pure_leaf_identity(
    before: dict[str, Any],
    after: dict[str, Any],
) -> dict[str, Any]:
    """Compare pure-dictionary leaf paths and Python object identities."""

    before_path_leaves, before_tree = jax.tree_util.tree_flatten_with_path(before)
    after_path_leaves, after_tree = jax.tree_util.tree_flatten_with_path(after)
    if before_tree != after_tree:
        raise ValueError("Pure-dictionary tree definitions changed across round-trip")
    if len(before_path_leaves) != len(after_path_leaves):
        raise ValueError("Pure-dictionary leaf counts changed across round-trip")

    mismatches = []
    identical = 0
    for (before_path, before_leaf), (after_path, after_leaf) in zip(
        before_path_leaves,
        after_path_leaves,
        strict=True,
    ):
        before_key = jax.tree_util.keystr(before_path)
        after_key = jax.tree_util.keystr(after_path)
        if before_key != after_key:
            raise ValueError(
                "Pure-dictionary leaf paths changed across round-trip: "
                f"{before_key!r} != {after_key!r}"
            )
        if before_leaf is after_leaf:
            identical += 1
        else:
            mismatches.append(
                {
                    "path": before_key,
                    "before_type": type(before_leaf).__name__,
                    "after_type": type(after_leaf).__name__,
                }
            )
    return {
        "leaf_count": len(before_path_leaves),
        "identical_object_count": identical,
        "different_object_count": len(mismatches),
        "different_objects": mismatches,
    }


def validate_reference_manifest(
    manifest: dict[str, Any],
    *,
    model_config: dict[str, Any],
) -> dict[str, Any]:
    """Fail closed unless the retained accepted manifest is the fixed reference."""

    model_abi = manifest.get("model_abi")
    if not isinstance(model_abi, dict):
        raise ValueError("Reference manifest has no model_abi object")
    expected_model_abi = {
        "sha256": CONSTRUCTOR_MODEL_ABI,
        "leaf_count": EXPECTED_MODEL_LEAVES,
        "nbytes": EXPECTED_MODEL_BYTES,
    }
    if compact_abi(model_abi) != expected_model_abi:
        raise ValueError(
            "Reference checkpoint model ABI changed: "
            f"{compact_abi(model_abi)!r}"
        )
    resume_contract = manifest.get("resume_contract")
    if not isinstance(resume_contract, dict):
        raise ValueError("Reference manifest has no resume_contract object")
    if resume_contract.get("model_config") != model_config:
        raise ValueError("Current model config differs from accepted update-2,072")
    state = manifest.get("state")
    if not isinstance(state, dict) or state.get("filename") != "state.npz":
        raise ValueError("Reference manifest has an unexpected state contract")
    return model_abi


def fixed_training_args() -> argparse.Namespace:
    """Build the exact accepted graph configuration without data or execution."""

    return train.parse_args(
        [
            "--objective",
            "normalized",
            "--init",
            "model-only",
            "--batch-size",
            "128",
            "--eval-batch-size",
            "64",
            "--eval-batches",
            "0",
            "--steps",
            "0",
            "--compile-ahead",
            "--compile-only",
            "--learning-rate",
            "3e-5",
            "--bt4-learning-rate",
            "1e-6",
            "--target-sigreg-coeff",
            "5.76",
            "--pred-sigreg-coeff",
            "1.0",
            "--sigreg-estimator",
            "v_stat",
            "--sigreg-example-count",
            "64",
            "--jepa-norm-loss-coeff",
            "0.0",
            "--train-batch-schedule",
            "global_permutation",
            "--seed",
            "0",
            "--val-seed",
            "10000",
            "--save-every",
            "0",
            "--no-save-final",
            "--max-checkpoints",
            "2",
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    train.validate_environment()
    if jax.default_backend() != "gpu":
        raise RuntimeError(
            "compile ABI diagnostic requires guarded GPU construction, "
            f"found {jax.default_backend()!r}"
        )

    run_root = require_within_workspace(RUN_ROOT)
    models_dir = require_within_workspace(MODELS_DIR)
    manifest_path = require_within_workspace(REFERENCE_MANIFEST)
    output_dir = require_within_workspace(
        args.output_dir
        or REPO_ROOT / "research" / "runs" / args.run_id
    )
    if output_dir.exists():
        raise FileExistsError(f"Diagnostic output already exists: {output_dir}")

    training_args = fixed_training_args()
    config, _ = train.resolve_config(run_root)
    config = train.apply_experiment_overrides(config)
    config = train.apply_config_overrides(config, training_args)
    train.validate_no_inert_config_overrides(config)
    train.validate_objective_config(
        objective=training_args.objective,
        config=config,
    )
    serialized_config = train.serialized_model_config(config)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    reference_model_abi = validate_reference_manifest(
        manifest,
        model_config=serialized_config,
    )

    model_params = train.load_mapped_bt4_params(models_dir=models_dir)
    model, optimizer = train.create_joint_components(
        model_params,
        config,
        seed=training_args.seed,
    )
    del model_params
    gc.collect()

    optimizer_abi = train.research_state_abi(nnx.state(optimizer.opt_state))
    optimizer_summary = compact_abi(optimizer_abi)
    del optimizer_abi
    gc.collect()

    model_state = nnx.state(model, TrainableParam)
    before_abi = train.research_state_abi(model_state)
    before_pure = dict(nnx.to_pure_dict(model_state))

    nnx.replace_by_pure_dict(model_state, before_pure)
    nnx.update(model, model_state)

    after_state = nnx.state(model, TrainableParam)
    after_abi = train.research_state_abi(after_state)
    after_pure = dict(nnx.to_pure_dict(after_state))

    before_leaf_summary = leaf_schema_summary(before_abi)
    after_leaf_summary = leaf_schema_summary(after_abi)
    manifest_leaf_summary = leaf_schema_summary(reference_model_abi)
    differences = compare_schemas(before_abi, after_abi)
    identity = pure_leaf_identity(before_pure, after_pure)

    expected_optimizer = {
        "sha256": CONSTRUCTOR_OPTIMIZER_ABI,
        "leaf_count": EXPECTED_OPTIMIZER_LEAVES,
        "nbytes": EXPECTED_OPTIMIZER_BYTES,
    }
    container_metadata_only = all(
        (
            compact_abi(before_abi)
            == {
                "sha256": CONSTRUCTOR_MODEL_ABI,
                "leaf_count": EXPECTED_MODEL_LEAVES,
                "nbytes": EXPECTED_MODEL_BYTES,
            },
            compact_abi(after_abi)
            == {
                "sha256": POST_RESTORE_MODEL_ABI,
                "leaf_count": EXPECTED_MODEL_LEAVES,
                "nbytes": EXPECTED_MODEL_BYTES,
            },
            before_abi == reference_model_abi,
            before_leaf_summary == after_leaf_summary,
            before_leaf_summary == manifest_leaf_summary,
            differences["leaf_difference_count"] == 0,
            identity["different_object_count"] == 0,
            identity["identical_object_count"] == EXPECTED_MODEL_LEAVES,
            optimizer_summary == expected_optimizer,
        )
    )
    report = {
        "format": "chess-dfm-compile-abi-diagnostic-v1",
        "git_commit": train.git_commit(),
        "timestamp_utc": datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"),
        "run_id": args.run_id,
        "mode": "read_only_constructor_self_roundtrip",
        "reference_manifest": str(manifest_path),
        "source_checkpoint_opened": False,
        "training_batch_created": False,
        "training_graph_lowered_or_compiled": False,
        "model_or_optimizer_executed": False,
        "checkpoint_writes": 0,
        "model_config_matches_reference": True,
        "constructor_model_abi": compact_abi(before_abi),
        "self_roundtrip_model_abi": compact_abi(after_abi),
        "reference_checkpoint_model_abi": compact_abi(reference_model_abi),
        "constructor_leaf_schema": before_leaf_summary,
        "self_roundtrip_leaf_schema": after_leaf_summary,
        "reference_checkpoint_leaf_schema": manifest_leaf_summary,
        "optimizer_abi": optimizer_summary,
        "pure_leaf_identity": identity,
        "full_schema_differences": differences,
        "decision": (
            "container_metadata_only"
            if container_metadata_only
            else "blocked_by_signature_or_reference_difference"
        ),
        "gpu_memory": train.gpu_memory_stats(),
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    train.write_json(output_dir / "report.json", report)
    print(
        json.dumps(
            {
                "run_id": args.run_id,
                "output_dir": str(output_dir),
                "constructor_model_abi": report["constructor_model_abi"],
                "self_roundtrip_model_abi": report[
                    "self_roundtrip_model_abi"
                ],
                "optimizer_abi": report["optimizer_abi"],
                "pure_leaf_identity": report["pure_leaf_identity"],
                "schema_difference_counts": {
                    key: differences[key]
                    for key in (
                        "missing_count",
                        "extra_count",
                        "changed_count",
                        "leaf_difference_count",
                    )
                },
                "decision": report["decision"],
                "gpu_memory": report["gpu_memory"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if container_metadata_only else 2


if __name__ == "__main__":
    raise SystemExit(main())
