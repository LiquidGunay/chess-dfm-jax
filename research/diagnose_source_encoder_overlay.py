#!/usr/bin/env python3
"""Evaluate an accepted head before and after a source-encoder overlay."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import sys
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import jax
import numpy as np
from flax import nnx

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from chess_dfm_jax.nnx_bt4 import TrainableParam  # noqa: E402
from research import train  # noqa: E402
from research.prepare import (  # noqa: E402
    REPO_ROOT,
    FixedTrajectoryBatches,
    require_within_workspace,
    write_json,
)


SOURCE_STATE = (
    REPO_ROOT
    / "checkpoints"
    / "source"
    / "step0265000"
    / "checkpoints"
    / "step0265000"
    / "state.npz"
)
ACCEPTED_CHECKPOINT = (
    REPO_ROOT
    / "research"
    / "runs"
    / "future-tail1-balanced-k1-cosine-b128-30m-v1"
    / "checkpoints"
    / "update00002072"
)
MODELS_DIR = REPO_ROOT / "models" / "source" / "extracted"
DATA_ROOT = REPO_ROOT / "data" / "trajectory_v3"
SOURCE_STATE_SIZE = 1_851_704_172
SOURCE_STATE_SHA256 = (
    "16a3c7e77e411a8a7577ff04dac1ca5173ce24ecb343b5fa4938e2d2b5fb8906"
)
VALIDATION_SEEDS = (10_000, 20_000)
EVAL_BATCH_SIZE = 64
EVAL_BATCHES = 64
EXPECTED_CONTROL = {
    "dfm_ce_loss": 4.495009120553732,
    "dfm_ce_loss_by_horizon_h1": 2.8470487520098686,
    "accuracy": 0.1104583740234375,
    "first_legal_mass": 0.6488690404221416,
}
CONTROL_TOLERANCE = 1e-6
CE_DELTA_LIMIT = 2e-4
LEGAL_MASS_DELTA_LIMIT = 2e-4
ACCURACY_DELTA_LIMIT = 2 / 8192


def _sha256_file(path: Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _compact_abi(value: Any) -> dict[str, Any]:
    abi = train.research_state_abi(value)
    return {
        "sha256": abi["sha256"],
        "leaf_count": int(abi["leaf_count"]),
        "nbytes": int(abi["nbytes"]),
    }


def state_value_digest(value: Any) -> dict[str, Any]:
    """Hash paths, concrete dtypes/shapes, and raw bytes for an array tree."""

    path_leaves, treedef = jax.tree_util.tree_flatten_with_path(value)
    digest = hashlib.sha256()
    nbytes = 0
    for path, leaf in path_leaves:
        array = np.asarray(leaf)
        if array.dtype.hasobject:
            raise TypeError(
                "Object-valued state leaf at "
                f"{jax.tree_util.keystr(path)}"
            )
        metadata = json.dumps(
            {
                "path": jax.tree_util.keystr(path),
                "shape": list(array.shape),
                "dtype": array.dtype.name,
            },
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        digest.update(len(metadata).to_bytes(8, "little"))
        digest.update(metadata)
        contiguous = np.ascontiguousarray(array)
        raw = memoryview(contiguous).cast("B")
        digest.update(len(raw).to_bytes(8, "little"))
        digest.update(raw)
        nbytes += len(raw)
    treedef_text = str(treedef)
    return {
        "sha256": digest.hexdigest(),
        "leaf_count": len(path_leaves),
        "nbytes": int(nbytes),
        "treedef_sha256": hashlib.sha256(
            treedef_text.encode("utf-8")
        ).hexdigest(),
    }


def leaf_identity_summary(before: Any, after: Any) -> dict[str, Any]:
    """Require identical tree paths and report Python leaf identities."""

    before_leaves, before_tree = jax.tree_util.tree_flatten_with_path(before)
    after_leaves, after_tree = jax.tree_util.tree_flatten_with_path(after)
    if before_tree != after_tree:
        raise ValueError("Non-encoder tree definition changed during overlay")
    if len(before_leaves) != len(after_leaves):
        raise ValueError("Non-encoder leaf count changed during overlay")

    different: list[str] = []
    for (before_path, before_leaf), (after_path, after_leaf) in zip(
        before_leaves,
        after_leaves,
        strict=True,
    ):
        before_text = jax.tree_util.keystr(before_path)
        after_text = jax.tree_util.keystr(after_path)
        if before_text != after_text:
            raise ValueError(
                "Non-encoder leaf paths changed during overlay: "
                f"{before_text!r} != {after_text!r}"
            )
        if before_leaf is not after_leaf:
            different.append(before_text)
    return {
        "leaf_count": len(before_leaves),
        "identical_object_count": len(before_leaves) - len(different),
        "different_object_count": len(different),
        "different_paths": different,
    }


def overlay_source_encoder(
    model: train.JointLatentSASAModel,
    source_model_state: Mapping[str, Any],
) -> dict[str, Any]:
    """Strictly replace only ``model.encoder`` and prove head identity."""

    if "encoder" not in source_model_state:
        raise ValueError("Source model state has no encoder subtree")

    model_state_before = nnx.state(model, TrainableParam)
    pure_before = dict(nnx.to_pure_dict(model_state_before))
    if "encoder" not in pure_before:
        raise ValueError("Runtime model state has no encoder subtree")
    before_non_encoder = {
        key: value for key, value in pure_before.items() if key != "encoder"
    }
    before_non_encoder_digest = state_value_digest(before_non_encoder)
    before_model_abi = _compact_abi(pure_before)
    accepted_encoder_digest = state_value_digest(pure_before["encoder"])

    source_encoder = source_model_state["encoder"]
    train.assert_research_state_compatible(
        pure_before["encoder"],
        source_encoder,
        label="source encoder overlay",
    )
    source_encoder_digest = state_value_digest(source_encoder)
    if source_encoder_digest == accepted_encoder_digest:
        raise ValueError("Source encoder overlay is byte-identical to accepted encoder")

    encoder_state = nnx.state(model.encoder, TrainableParam)
    nnx.replace_by_pure_dict(encoder_state, source_encoder)
    nnx.update(model.encoder, encoder_state)
    jax.block_until_ready(nnx.state(model.encoder, TrainableParam))

    del model_state_before, pure_before, encoder_state
    gc.collect()

    model_state_after = nnx.state(model, TrainableParam)
    pure_after = dict(nnx.to_pure_dict(model_state_after))
    after_non_encoder = {
        key: value for key, value in pure_after.items() if key != "encoder"
    }
    identity = leaf_identity_summary(before_non_encoder, after_non_encoder)
    after_non_encoder_digest = state_value_digest(after_non_encoder)
    after_model_abi = _compact_abi(pure_after)
    overlaid_encoder_digest = state_value_digest(pure_after["encoder"])

    if identity["different_object_count"] != 0:
        raise ValueError("Source overlay replaced a non-encoder leaf object")
    if after_non_encoder_digest != before_non_encoder_digest:
        raise ValueError("Source overlay changed non-encoder values")
    if after_model_abi != before_model_abi:
        raise ValueError("Source overlay changed the model state ABI")
    if overlaid_encoder_digest != source_encoder_digest:
        raise ValueError("Runtime encoder values differ from source after overlay")

    return {
        "accepted_encoder": accepted_encoder_digest,
        "source_encoder": source_encoder_digest,
        "overlaid_encoder": overlaid_encoder_digest,
        "non_encoder_before": before_non_encoder_digest,
        "non_encoder_after": after_non_encoder_digest,
        "non_encoder_identity": identity,
        "model_abi_before": before_model_abi,
        "model_abi_after": after_model_abi,
        "encoder_only": True,
    }


def load_model_trainable(path: Path) -> dict[str, Any]:
    """Read only the model member of a research state archive."""

    with np.load(path, allow_pickle=True) as archive:
        expected = {"step", "model_trainable", "optimizer_state"}
        if set(archive.files) != expected:
            raise ValueError(
                "Source state archive keys differ: "
                f"expected {sorted(expected)}, found {sorted(archive.files)}"
            )
        raw = archive["model_trainable"]
        if raw.shape != () or raw.dtype != object:
            raise ValueError(
                "Source model_trainable must be an object scalar, found "
                f"shape={raw.shape}, dtype={raw.dtype}"
            )
        value = raw.item()
    if not isinstance(value, dict):
        raise TypeError("Source model_trainable payload must be a dictionary")
    return value


def aggregate_variant_records(
    records: list[dict[str, Any]],
) -> dict[str, dict[str, float]]:
    """Average common scalar metrics by variant."""

    grouped: dict[str, list[dict[str, float]]] = {}
    for record in records:
        variant = record.get("variant")
        metrics = record.get("validation")
        if not isinstance(variant, str) or not isinstance(metrics, dict):
            raise ValueError("Evaluation record is missing variant/validation")
        grouped.setdefault(variant, []).append(metrics)

    result: dict[str, dict[str, float]] = {}
    for variant, metrics_list in grouped.items():
        common = set.intersection(*(set(metrics) for metrics in metrics_list))
        result[variant] = {
            key: math.fsum(float(metrics[key]) for metrics in metrics_list)
            / len(metrics_list)
            for key in sorted(common)
            if all(
                isinstance(metrics[key], (int, float, np.number))
                and not isinstance(metrics[key], bool)
                for metrics in metrics_list
            )
        }
    return result


def metric_deltas(
    accepted: Mapping[str, float],
    overlay: Mapping[str, float],
) -> dict[str, float]:
    common = sorted(set(accepted) & set(overlay))
    return {
        key: float(overlay[key]) - float(accepted[key])
        for key in common
    }


def validate_control_metrics(metrics: Mapping[str, float]) -> dict[str, float]:
    deltas = {
        key: float(metrics[key]) - expected
        for key, expected in EXPECTED_CONTROL.items()
    }
    failures = {
        key: delta
        for key, delta in deltas.items()
        if abs(delta) > CONTROL_TOLERANCE
    }
    if failures:
        raise ValueError(
            "Accepted control does not reproduce retained metrics: "
            + ", ".join(
                f"{key} delta={delta:.9g}"
                for key, delta in sorted(failures.items())
            )
        )
    return deltas


def overlay_decision(
    *,
    records: list[dict[str, Any]],
    aggregate: Mapping[str, Mapping[str, float]],
) -> dict[str, Any]:
    """Apply the preregistered per-seed and pooled scalar thresholds."""

    by_variant_seed: dict[tuple[str, int], Mapping[str, float]] = {}
    for record in records:
        by_variant_seed[
            (str(record["variant"]), int(record["validation_seed"]))
        ] = record["validation"]

    comparisons: dict[str, Any] = {}
    failures: list[str] = []
    scopes: list[tuple[str, Mapping[str, float], Mapping[str, float]]] = [
        (
            "pooled",
            aggregate["accepted"],
            aggregate["source_encoder_overlay"],
        )
    ]
    for seed in VALIDATION_SEEDS:
        scopes.append(
            (
                f"seed_{seed}",
                by_variant_seed[("accepted", seed)],
                by_variant_seed[("source_encoder_overlay", seed)],
            )
        )

    limits = {
        "dfm_ce_loss": CE_DELTA_LIMIT,
        "dfm_ce_loss_by_horizon_h1": CE_DELTA_LIMIT,
        "first_legal_mass": LEGAL_MASS_DELTA_LIMIT,
        "accuracy": ACCURACY_DELTA_LIMIT,
    }
    for scope, accepted, overlay in scopes:
        deltas = metric_deltas(accepted, overlay)
        checks = {}
        for metric, limit in limits.items():
            delta = float(deltas[metric])
            passed = abs(delta) <= limit
            checks[metric] = {
                "accepted": float(accepted[metric]),
                "overlay": float(overlay[metric]),
                "delta": delta,
                "absolute_delta": abs(delta),
                "limit": float(limit),
                "passed": passed,
            }
            if not passed:
                failures.append(
                    f"{scope}:{metric} abs_delta={abs(delta):.9g} > {limit:.9g}"
                )
        comparisons[scope] = {
            "checks": checks,
            "all_common_metric_deltas": deltas,
        }
    return {
        "functionally_negligible_for_autoresearch_proxy": not failures,
        "comparisons": comparisons,
        "failures": failures,
    }


def cache_inventory(cache_dir: Path) -> dict[str, Any]:
    """Inventory executable cache files while ignoring mutable atime markers."""

    entries = {}
    for path in sorted(cache_dir.glob("*-cache")):
        if path.is_file():
            entries[path.name] = {
                "size_bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
    total_bytes = sum(
        path.stat().st_size
        for path in cache_dir.rglob("*")
        if path.is_file()
    )
    return {
        "directory": str(cache_dir),
        "executable_count": len(entries),
        "executables": entries,
        "total_bytes": int(total_bytes),
    }


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
            "source-encoder overlay diagnostic requires guarded GPU evaluation, "
            f"found {jax.default_backend()!r}"
        )

    source_state = require_within_workspace(SOURCE_STATE)
    accepted_checkpoint = require_within_workspace(ACCEPTED_CHECKPOINT)
    models_dir = require_within_workspace(MODELS_DIR)
    data_root = require_within_workspace(DATA_ROOT)
    cache_dir = require_within_workspace(
        Path(os.environ["JAX_COMPILATION_CACHE_DIR"])
    )
    output_dir = require_within_workspace(
        args.output_dir
        or REPO_ROOT / "research" / "runs" / args.run_id
    )
    if output_dir.exists():
        raise FileExistsError(f"Diagnostic output already exists: {output_dir}")
    if source_state.stat().st_size != SOURCE_STATE_SIZE:
        raise ValueError("Immutable source checkpoint size changed")
    if _sha256_file(source_state) != SOURCE_STATE_SHA256:
        raise ValueError("Immutable source checkpoint digest changed")

    config, objective, sigreg_reference_count, contract, contract_sha256 = (
        train.checkpoint_evaluation_contract([accepted_checkpoint])
    )
    if objective != "normalized":
        raise ValueError(f"Accepted checkpoint objective changed: {objective!r}")
    if config.jepa_target_semantics != "online":
        raise ValueError("Experiment 033 supports only the accepted online target")

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    cache_before = cache_inventory(cache_dir)
    run_config = {
        "format": "chess-dfm-source-encoder-overlay-v1",
        "mode": "read_only_source_encoder_overlay",
        "git_commit": train.git_commit(),
        "run_id": args.run_id,
        "timestamp_utc": timestamp,
        "accepted_checkpoint": str(accepted_checkpoint),
        "source_state": str(source_state),
        "source_state_size_bytes": SOURCE_STATE_SIZE,
        "source_state_sha256": SOURCE_STATE_SHA256,
        "objective": objective,
        "sigreg_reference_count": sigreg_reference_count,
        "model_config": train.serialized_model_config(config),
        "checkpoint_resume_contract": contract,
        "checkpoint_resume_contract_sha256": contract_sha256,
        "validation_seeds": list(VALIDATION_SEEDS),
        "eval_batch_size": EVAL_BATCH_SIZE,
        "eval_batches": EVAL_BATCHES,
        "validation_examples_per_seed": EVAL_BATCH_SIZE * EVAL_BATCHES,
        "deterministic_t": 0.0,
        "checkpoint_writes": 0,
        "optimizer_state_restored": False,
        "hybrid_checkpoint_written": False,
        "cache_before": cache_before,
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    write_json(output_dir / "run_config.json", run_config)

    validation_batches = {
        seed: FixedTrajectoryBatches(
            data_root / "val",
            batch_size=EVAL_BATCH_SIZE,
            horizon=config.horizon,
            seed=seed,
            shuffle_files=True,
            batch_schedule="global_permutation",
        )
        for seed in VALIDATION_SEEDS
    }
    model_params = train.load_mapped_bt4_params(models_dir=models_dir)
    encoder = train.make_bt4_model(
        model_params,
        dtype=train._parse_compute_dtype(config.encoder_dtype),
        train_encoder=config.unfreeze_bt4_encoder,
    )
    model = train.JointLatentSASAModel(
        encoder,
        config,
        rngs=nnx.Rngs(0),
    )
    del model_params, encoder
    gc.collect()

    restored = train.load_research_checkpoint_for_evaluation(
        accepted_checkpoint,
        model=model,
    )
    records: list[dict[str, Any]] = []
    metrics_path = output_dir / "metrics.jsonl"
    with metrics_path.open("x", encoding="utf-8") as metrics_log:
        for seed in VALIDATION_SEEDS:
            validation, seconds = train.evaluate(
                model,
                validation_batches[seed],
                count=EVAL_BATCHES,
                seed=seed,
                deterministic_t=0.0,
                objective=objective,
                sigreg_reference_count=sigreg_reference_count,
                collapse_diagnostics=False,
            )
            record = {
                "variant": "accepted",
                "validation_seed": seed,
                "validation_seconds": seconds,
                "validation": validation,
            }
            records.append(record)
            metrics_log.write(json.dumps(record, sort_keys=True) + "\n")
            metrics_log.flush()
            os.fsync(metrics_log.fileno())

        accepted_aggregate = aggregate_variant_records(records)["accepted"]
        control_deltas = validate_control_metrics(accepted_aggregate)

        source_model_state = load_model_trainable(source_state)
        overlay_evidence = overlay_source_encoder(model, source_model_state)
        del source_model_state
        gc.collect()

        for seed in VALIDATION_SEEDS:
            validation, seconds = train.evaluate(
                model,
                validation_batches[seed],
                count=EVAL_BATCHES,
                seed=seed,
                deterministic_t=0.0,
                objective=objective,
                sigreg_reference_count=sigreg_reference_count,
                collapse_diagnostics=False,
            )
            record = {
                "variant": "source_encoder_overlay",
                "validation_seed": seed,
                "validation_seconds": seconds,
                "validation": validation,
            }
            records.append(record)
            metrics_log.write(json.dumps(record, sort_keys=True) + "\n")
            metrics_log.flush()
            os.fsync(metrics_log.fileno())

    aggregate = aggregate_variant_records(records)
    decision = overlay_decision(records=records, aggregate=aggregate)
    cache_after = cache_inventory(cache_dir)
    if cache_after["executables"] != cache_before["executables"]:
        raise ValueError("Overlay diagnostic created or changed an executable cache entry")

    report = {
        **run_config,
        "accepted_checkpoint_state_sha256": restored["state"]["sha256"],
        "accepted_control_expected": EXPECTED_CONTROL,
        "accepted_control_tolerance": CONTROL_TOLERANCE,
        "accepted_control_deltas": control_deltas,
        "overlay_evidence": overlay_evidence,
        "aggregate_validation": aggregate,
        "decision": decision,
        "cache_after": cache_after,
        "cache_executables_unchanged": True,
        "gpu_memory": train.gpu_memory_stats(),
        "metrics_path": str(metrics_path),
        "completed": True,
        "checkpoint_writes": 0,
        "hybrid_checkpoint_written": False,
    }
    write_json(output_dir / "report.json", report)
    print(
        json.dumps(
            {
                "run_id": args.run_id,
                "output_dir": str(output_dir),
                "completed": True,
                "functionally_negligible_for_autoresearch_proxy": decision[
                    "functionally_negligible_for_autoresearch_proxy"
                ],
                "pooled_checks": decision["comparisons"]["pooled"]["checks"],
                "gpu_memory": report["gpu_memory"],
                "cache_executables_unchanged": True,
                "checkpoint_writes": 0,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
