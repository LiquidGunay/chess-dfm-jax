#!/usr/bin/env python3
"""Diff live model schemas around one verified legacy model-only restore."""

from __future__ import annotations

import argparse
import gc
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
from research import diagnose_compile_abi as schema_tools  # noqa: E402
from research import train  # noqa: E402
from research.prepare import REPO_ROOT, require_within_workspace  # noqa: E402


SOURCE_OPTIMIZER_ABI = (
    "3e794fe647c35f5b2d5e1f78c5625178c783cd19ec0a3fb1e9aa8f3d9e790b65"
)
SOURCE_CHECKPOINT = (
    REPO_ROOT
    / "checkpoints"
    / "source"
    / "step0265000"
    / "checkpoints"
    / "step0265000"
    / "state.npz"
)


def validate_import_invariants(
    *,
    import_result: Any,
    optimizer_before: dict[str, Any],
    optimizer_after: dict[str, Any],
    optimizer_step_before: int,
    optimizer_step_after: int,
) -> None:
    """Fail closed unless model-only import leaves the fresh optimizer untouched."""

    if import_result.model_abi_sha256 != schema_tools.CONSTRUCTOR_MODEL_ABI:
        raise ValueError(
            "Verified legacy model ABI changed: "
            f"{import_result.model_abi_sha256}"
        )
    if import_result.optimizer_abi_sha256 != SOURCE_OPTIMIZER_ABI:
        raise ValueError(
            "Verified legacy optimizer ABI changed: "
            f"{import_result.optimizer_abi_sha256}"
        )
    expected_optimizer = {
        "sha256": schema_tools.CONSTRUCTOR_OPTIMIZER_ABI,
        "leaf_count": schema_tools.EXPECTED_OPTIMIZER_LEAVES,
        "nbytes": schema_tools.EXPECTED_OPTIMIZER_BYTES,
    }
    if optimizer_before != expected_optimizer:
        raise ValueError(
            f"Constructor runtime optimizer ABI changed: {optimizer_before!r}"
        )
    if optimizer_after != optimizer_before:
        raise ValueError(
            "Model-only import changed the fresh runtime optimizer ABI: "
            f"{optimizer_before!r} != {optimizer_after!r}"
        )
    if optimizer_step_before != 0 or optimizer_step_after != 0:
        raise ValueError(
            "Model-only import changed the fresh optimizer step: "
            f"{optimizer_step_before} -> {optimizer_step_after}"
        )
    if import_result.optimizer_restored:
        raise ValueError("Model-only import unexpectedly restored optimizer values")
    if import_result.init_mode != "model-only":
        raise ValueError(
            f"Legacy importer returned unexpected mode {import_result.init_mode!r}"
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
            "legacy-restore ABI diagnostic requires guarded GPU construction, "
            f"found {jax.default_backend()!r}"
        )
    if jax.config.jax_enable_compilation_cache:
        raise RuntimeError(
            "legacy-restore ABI diagnostic requires persistent cache writes disabled"
        )

    run_root = require_within_workspace(schema_tools.RUN_ROOT)
    models_dir = require_within_workspace(schema_tools.MODELS_DIR)
    manifest_path = require_within_workspace(schema_tools.REFERENCE_MANIFEST)
    source_checkpoint = require_within_workspace(SOURCE_CHECKPOINT)
    output_dir = require_within_workspace(
        args.output_dir
        or REPO_ROOT / "research" / "runs" / args.run_id
    )
    if output_dir.exists():
        raise FileExistsError(f"Diagnostic output already exists: {output_dir}")

    training_args = schema_tools.fixed_training_args()
    config, metadata = train.resolve_config(run_root)
    config = train.apply_experiment_overrides(config)
    config = train.apply_config_overrides(config, training_args)
    train.validate_no_inert_config_overrides(config)
    train.validate_objective_config(
        objective=training_args.objective,
        config=config,
    )
    serialized_config = train.serialized_model_config(config)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    reference_model_abi = schema_tools.validate_reference_manifest(
        manifest,
        model_config=serialized_config,
    )

    checkpoint_step = int(metadata["latest_step"])
    if checkpoint_step != 265_000:
        raise ValueError(
            f"Pinned legacy checkpoint step changed: {checkpoint_step}"
        )
    asset = train.load_asset_manifest()["checkpoint_step_265000"]

    model_params = train.load_mapped_bt4_params(models_dir=models_dir)
    model, optimizer = train.create_joint_components(
        model_params,
        config,
        seed=training_args.seed,
    )
    del model_params
    gc.collect()

    before_model_abi = train.research_state_abi(
        nnx.state(model, TrainableParam)
    )
    if before_model_abi != reference_model_abi:
        raise ValueError(
            "Constructor model schema differs from the accepted checkpoint "
            f"manifest: {before_model_abi['sha256']} != "
            f"{reference_model_abi['sha256']}"
        )
    optimizer_before_full = train.research_state_abi(
        nnx.state(optimizer.opt_state)
    )
    optimizer_before = schema_tools.compact_abi(optimizer_before_full)
    del optimizer_before_full
    optimizer_step_before = int(optimizer.step[...])
    gc.collect()

    import_result = train.import_legacy_checkpoint(
        source_checkpoint,
        model=model,
        optimizer=optimizer,
        expected_size_bytes=int(asset["state_npz_size_bytes"]),
        expected_sha256=str(asset["state_npz_sha256"]),
        expected_step=checkpoint_step,
        init_mode="model-only",
    )
    gc.collect()

    after_model_abi = train.research_state_abi(
        nnx.state(model, TrainableParam)
    )
    optimizer_after_full = train.research_state_abi(
        nnx.state(optimizer.opt_state)
    )
    optimizer_after = schema_tools.compact_abi(optimizer_after_full)
    del optimizer_after_full
    optimizer_step_after = int(optimizer.step[...])
    validate_import_invariants(
        import_result=import_result,
        optimizer_before=optimizer_before,
        optimizer_after=optimizer_after,
        optimizer_step_before=optimizer_step_before,
        optimizer_step_after=optimizer_step_after,
    )

    before_leaf = schema_tools.leaf_schema_summary(before_model_abi)
    after_leaf = schema_tools.leaf_schema_summary(after_model_abi)
    differences = schema_tools.compare_schemas(
        before_model_abi,
        after_model_abi,
    )
    after_summary = schema_tools.compact_abi(after_model_abi)
    expected_after = {
        "sha256": schema_tools.POST_RESTORE_MODEL_ABI,
        "leaf_count": schema_tools.EXPECTED_MODEL_LEAVES,
        "nbytes": schema_tools.EXPECTED_MODEL_BYTES,
    }
    if after_summary != expected_after:
        decision = "unexpected_post_restore_model_abi"
    elif differences["leaf_difference_count"] == 0:
        decision = "container_metadata_only"
    else:
        decision = "leaf_signature_changed"

    report = {
        "format": "chess-dfm-legacy-restore-abi-diagnostic-v1",
        "git_commit": train.git_commit(),
        "timestamp_utc": datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"),
        "run_id": args.run_id,
        "mode": "read_only_verified_legacy_model_only_restore",
        "reference_manifest": str(manifest_path),
        "source_checkpoint": str(source_checkpoint),
        "source_checkpoint_opened": True,
        "source_checkpoint_size_bytes": import_result.source_size_bytes,
        "source_checkpoint_sha256": import_result.source_sha256,
        "source_checkpoint_step": import_result.source_step,
        "source_model_abi_sha256": import_result.model_abi_sha256,
        "source_optimizer_abi_sha256": import_result.optimizer_abi_sha256,
        "training_batch_created": False,
        "training_graph_lowered_or_compiled": False,
        "model_or_optimizer_executed": False,
        "checkpoint_writes": 0,
        "model_config_matches_reference": True,
        "constructor_model_abi": before_model_abi,
        "post_restore_model_abi": after_model_abi,
        "constructor_leaf_schema": before_leaf,
        "post_restore_leaf_schema": after_leaf,
        "optimizer_abi_before": optimizer_before,
        "optimizer_abi_after": optimizer_after,
        "optimizer_step_before": optimizer_step_before,
        "optimizer_step_after": optimizer_step_after,
        "full_schema_differences": differences,
        "decision": decision,
        "gpu_memory": train.gpu_memory_stats(),
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    train.write_json(output_dir / "report.json", report)
    print(
        json.dumps(
            {
                "run_id": args.run_id,
                "output_dir": str(output_dir),
                "constructor_model_abi": schema_tools.compact_abi(
                    before_model_abi
                ),
                "post_restore_model_abi": after_summary,
                "constructor_leaf_schema": before_leaf,
                "post_restore_leaf_schema": after_leaf,
                "optimizer_abi_before": optimizer_before,
                "optimizer_abi_after": optimizer_after,
                "optimizer_step_before": optimizer_step_before,
                "optimizer_step_after": optimizer_step_after,
                "schema_difference_counts": {
                    key: differences[key]
                    for key in (
                        "missing_count",
                        "extra_count",
                        "changed_count",
                        "leaf_difference_count",
                    )
                },
                "decision": decision,
                "gpu_memory": report["gpu_memory"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if after_summary == expected_after else 2


if __name__ == "__main__":
    raise SystemExit(main())
