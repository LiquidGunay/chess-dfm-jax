"""Budget-gated native-Torch checkpoint arena on Modal L4 workers."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from decimal import Decimal
from pathlib import Path
from typing import Any

import modal


APP_NAME = "chess-dfm-torch-arena"
BASE_INPUT_VOLUME_NAME = "chess-dfm-interpretability-inputs"
TRAINING_RESULT_VOLUME_NAME = "chess-dfm-training-results"
BASE_INPUT_MOUNT = "/workspace/base-inputs"
TRAINING_RESULT_MOUNT = "/workspace/training-results"
BASE_INPUT_BUNDLE_SHA256 = (
    "8be17957ad779e3756092306daa95ad9c1c7f5d482b6409bda7a16bc6d6c7a44"
)
RAW_BT4_RELATIVE = "models/raw/BT4_exported.pb.gz"
RAW_BT4_SIZE_BYTES = 335_916_563
RAW_BT4_SHA256 = (
    "61e43e98d2c4cb747498c6bc13a5bbb3b07e985d3f5d87f0351a4db855fa0651"
)
ARENA_SOURCE_TREE_PATHS = (
    "research/arena.py",
    "research/arena_history_trust.py",
    "research/evaluate_arena.py",
    "research/evaluate_torch_migration_parity.py",
    "research/inference.py",
    "research/local_policy.py",
    "research/play_arena.py",
    "research/prepare.py",
    "research/raw_bt4_policy.py",
    "research/train.py",
    "research/train_torch.py",
    "research/infra/modal_arena_app.py",
)
ARENA_ASSET_NAMES = (
    "promotion-ply12-n2048-v3.json",
    "promotion-ply12-n2048-histories-v3.json",
    "promotion-ply12-n2048-v3-contract.json",
)
ALLOWED_PAIR_COUNTS = (16, 128, 1_024)
ALLOWED_REFINEMENT_PASSES = (1, 8)
ALLOWED_POLICY_MODES = ("dfm", "policy_only")

app = modal.App(APP_NAME)
base_input_volume = modal.Volume.from_name(
    BASE_INPUT_VOLUME_NAME,
    create_if_missing=False,
)
training_result_volume = modal.Volume.from_name(
    TRAINING_RESULT_VOLUME_NAME,
    create_if_missing=False,
)

arena_image = (
    modal.Image.from_registry("pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime")
    .apt_install("gcc", "g++")
    .uv_pip_install(
        "ml-dtypes==0.5.3",
        "numpy==2.2.6",
        "protobuf==6.31.1",
        "python-chess==1.999",
        "safetensors==0.5.3",
    )
    .add_local_python_source("chess_dfm_jax", "research", copy=True)
    .add_local_file(
        "chess_dfm_jax/policy_moves.txt",
        "/root/chess_dfm_jax/policy_moves.txt",
        copy=True,
    )
    .add_local_file(
        "chess_dfm_jax/policy_attn_map.txt",
        "/root/chess_dfm_jax/policy_attn_map.txt",
        copy=True,
    )
)
for _asset_name in ARENA_ASSET_NAMES:
    arena_image = arena_image.add_local_file(
        f"research/assets/arena/{_asset_name}",
        f"/opt/chess-assets/{_asset_name}",
        copy=True,
    )


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(value: str, *, label: str) -> str:
    if (
        len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _require_git_commit(value: str) -> str:
    if (
        len(value) != 40
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError("Git commit must be a lowercase SHA-1")
    return value


def _require_result_label(value: str) -> str:
    safe = all(
        character.islower() or character.isdigit() or character in "-_"
        for character in value
    )
    if (
        not value.startswith("hero-training-")
        or not 1 <= len(value) <= 200
        or Path(value).name != value
        or not safe
    ):
        raise ValueError("Unsafe Hero training result label")
    return value


def _require_policy_mode(value: str, *, label: str) -> str:
    if value not in ALLOWED_POLICY_MODES:
        raise ValueError(f"{label} must be one of {ALLOWED_POLICY_MODES}")
    return value


def _arena_source_tree_sha256(repo_root: Path) -> str:
    candidates = [repo_root / relative for relative in ARENA_SOURCE_TREE_PATHS]
    candidates.extend(sorted((repo_root / "chess_dfm_jax").rglob("*.py")))
    candidates.extend(
        (
            repo_root / "chess_dfm_jax" / "policy_moves.txt",
            repo_root / "chess_dfm_jax" / "policy_attn_map.txt",
        )
    )
    paths = sorted({path.resolve() for path in candidates})
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Arena source identity is incomplete: {missing}")
    resolved_root = repo_root.resolve()
    digest = hashlib.sha256()
    for path in paths:
        relative = path.relative_to(resolved_root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(_sha256_file(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _validated_training_result(result_label: str) -> dict[str, Any]:
    label = _require_result_label(result_label)
    root = Path(TRAINING_RESULT_MOUNT) / "runs" / label
    paths = {
        "modal_run": root / "modal_run.json",
        "report": root / "report.json",
        "run_config": root / "run_config.json",
        "checkpoint_manifest": root / "checkpoint" / "manifest.json",
    }
    if any(not path.is_file() for path in paths.values()):
        raise RuntimeError(f"Arena requires a completed training result: {label}")
    modal_run = json.loads(paths["modal_run"].read_text(encoding="utf-8"))
    report = json.loads(paths["report"].read_text(encoding="utf-8"))
    checkpoint = json.loads(
        paths["checkpoint_manifest"].read_text(encoding="utf-8")
    )
    if (
        modal_run.get("result_label") != label
        or checkpoint.get("format") != "chess-dfm-torch-model-v1"
        or checkpoint.get("model_only") is not True
        or int(checkpoint.get("optimizer_update", -1))
        != int(report.get("updates", -2))
    ):
        raise RuntimeError(f"Training result provenance drift: {label}")
    state = checkpoint.get("state")
    if not isinstance(state, dict):
        raise RuntimeError(f"Training result has no checkpoint state: {label}")
    state_sha256 = _require_sha256(
        str(state.get("sha256", "")),
        label=f"{label} checkpoint state",
    )
    state_path = root / "checkpoint" / str(state.get("path", ""))
    if (
        not state_path.is_file()
        or state_path.stat().st_size != int(state.get("size_bytes", -1))
    ):
        raise RuntimeError(f"Training result checkpoint file drift: {label}")
    return {
        "label": label,
        "root": root,
        "state_sha256": state_sha256,
        "optimizer_update": int(checkpoint["optimizer_update"]),
        "arm": str(modal_run["arm"]),
        "profile": str(modal_run["profile"]),
    }


def _arena_identity(
    *,
    candidate_label: str,
    candidate_state_sha256: str,
    opponent_label: str,
    opponent_state_sha256: str,
    candidate_policy_mode: str,
    opponent_policy_mode: str,
    pair_count: int,
    candidate_refinement_passes: int,
    opponent_refinement_passes: int,
    opening_start_index: int,
    arena_source_tree_sha256: str,
) -> str:
    if pair_count not in ALLOWED_PAIR_COUNTS:
        raise ValueError(f"pair_count must be one of {ALLOWED_PAIR_COUNTS}")
    for label, passes in (
        ("candidate_refinement_passes", candidate_refinement_passes),
        ("opponent_refinement_passes", opponent_refinement_passes),
    ):
        if passes not in ALLOWED_REFINEMENT_PASSES:
            raise ValueError(
                f"{label} must be one of {ALLOWED_REFINEMENT_PASSES}"
            )
    if opening_start_index < 0:
        raise ValueError("opening_start_index must be non-negative")
    identity = hashlib.sha256(
        _canonical_json_bytes(
            {
                "schema_version": "chess-dfm-modal-torch-arena-identity-v3",
                "candidate_label": _require_result_label(candidate_label),
                "candidate_state_sha256": _require_sha256(
                    candidate_state_sha256,
                    label="candidate_state_sha256",
                ),
                "opponent_label": _require_result_label(opponent_label),
                "opponent_state_sha256": _require_sha256(
                    opponent_state_sha256,
                    label="opponent_state_sha256",
                ),
                "candidate_policy_mode": _require_policy_mode(
                    candidate_policy_mode,
                    label="candidate_policy_mode",
                ),
                "opponent_policy_mode": _require_policy_mode(
                    opponent_policy_mode,
                    label="opponent_policy_mode",
                ),
                "pair_count": pair_count,
                "opening_start_index": opening_start_index,
                "candidate_refinement_passes": candidate_refinement_passes,
                "opponent_refinement_passes": opponent_refinement_passes,
                "additional_ply_cap": 256,
                "block_pairs": 16,
                "policy_batch_size_cap": 16,
                "arena_source_tree_sha256": _require_sha256(
                    arena_source_tree_sha256,
                    label="arena_source_tree_sha256",
                ),
                "raw_bt4_sha256": RAW_BT4_SHA256,
            }
        )
    ).hexdigest()
    return (
        f"hero-arena-{pair_count}p-o{opening_start_index}-"
        f"cpass{candidate_refinement_passes}-"
        f"opass{opponent_refinement_passes}-{identity[:20]}"
    )


def _arena_command(
    *,
    candidate_root: Path,
    opponent_root: Path,
    models_dir: Path,
    output_dir: Path,
    candidate_policy_mode: str,
    opponent_policy_mode: str,
    pair_count: int,
    candidate_refinement_passes: int,
    opponent_refinement_passes: int,
    opening_start_index: int,
    resume: bool,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "research.evaluate_arena",
        "--candidate-torch-hero",
        str(candidate_root),
        "--opponent-torch-hero",
        str(opponent_root),
        "--candidate-torch-hero-policy-mode",
        _require_policy_mode(
            candidate_policy_mode,
            label="candidate_policy_mode",
        ),
        "--opponent-torch-hero-policy-mode",
        _require_policy_mode(
            opponent_policy_mode,
            label="opponent_policy_mode",
        ),
        "--tier",
        "hero_development",
        "--pair-count",
        str(pair_count),
        "--opening-start-index",
        str(opening_start_index),
        "--block-pairs",
        "16",
        "--additional-ply-cap",
        "256",
        "--refinement-passes",
        str(candidate_refinement_passes),
        "--candidate-refinement-passes",
        str(candidate_refinement_passes),
        "--opponent-refinement-passes",
        str(opponent_refinement_passes),
        "--policy-batch-size-cap",
        "16",
        "--policy-timeout-seconds",
        "30",
        "--models-dir",
        str(models_dir),
        "--output-dir",
        str(output_dir),
    ]
    if resume:
        command.append("--resume")
    return command


def _verify_raw_bt4_volume() -> Path:
    bundle_root = Path(BASE_INPUT_MOUNT) / "bundles" / BASE_INPUT_BUNDLE_SHA256
    marker_path = (
        Path(BASE_INPUT_MOUNT)
        / "bundles"
        / f"{BASE_INPUT_BUNDLE_SHA256}.stage.json"
    )
    if not marker_path.is_file():
        raise RuntimeError("Base input bundle is not staged on the Modal Volume")
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    record = marker.get("bundle", {}).get("files", {}).get(RAW_BT4_RELATIVE)
    if (
        not isinstance(record, dict)
        or int(record.get("size_bytes", -1)) != RAW_BT4_SIZE_BYTES
        or record.get("sha256") != RAW_BT4_SHA256
    ):
        raise RuntimeError("Raw BT4 bundle identity drift")
    raw_path = bundle_root / RAW_BT4_RELATIVE
    if not raw_path.is_file() or raw_path.stat().st_size != RAW_BT4_SIZE_BYTES:
        raise RuntimeError("Raw BT4 staged file drift")
    return raw_path


def _materialize_arena_assets() -> None:
    destination = Path("/root/research/assets/arena")
    destination.mkdir(parents=True, exist_ok=True)
    for name in ARENA_ASSET_NAMES:
        source = Path("/opt/chess-assets") / name
        target = destination / name
        if not target.is_file():
            shutil.copyfile(source, target)


def _completed_arena_summary(
    output_dir: Path,
    *,
    arena_label: str,
    candidate_state_sha256: str,
    opponent_state_sha256: str,
    candidate_policy_mode: str,
    opponent_policy_mode: str,
    pair_count: int,
    candidate_refinement_passes: int,
    opponent_refinement_passes: int,
    opening_start_index: int,
) -> dict[str, Any]:
    state_path = output_dir / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if (
        state.get("schema_version") != "chess-dfm-relative-arena-run-v3"
        or state.get("status") != "complete"
    ):
        raise RuntimeError("Arena state is not complete")
    contract = state.get("contract", {})
    models = contract.get("models", {})
    run = contract.get("run", {})
    candidate_model = models.get("candidate", {})
    opponent_model = models.get("opponent", {})
    candidate = candidate_model.get("state", {})
    opponent = opponent_model.get("state", {})
    aggregate = state.get("aggregate", {})
    if (
        candidate.get("sha256") != candidate_state_sha256
        or opponent.get("sha256") != opponent_state_sha256
        or candidate_model.get("arena_policy_mode") != candidate_policy_mode
        or opponent_model.get("arena_policy_mode") != opponent_policy_mode
        or int(aggregate.get("pair_count", -1)) != pair_count
        or int(run.get("candidate_refinement_passes", -1))
        != candidate_refinement_passes
        or int(run.get("opponent_refinement_passes", -1))
        != opponent_refinement_passes
        or int(run.get("opening_start_index", -1)) != opening_start_index
    ):
        raise RuntimeError("Arena terminal identity drift")
    pentanomial = aggregate.get("pentanomial")
    if not isinstance(pentanomial, dict):
        raise RuntimeError("Arena terminal state has no pentanomial result")
    return {
        "schema_version": "chess-dfm-modal-torch-arena-result-v2",
        "arena_label": arena_label,
        "pair_count": pair_count,
        "opening_start_index": opening_start_index,
        "candidate_refinement_passes": candidate_refinement_passes,
        "opponent_refinement_passes": opponent_refinement_passes,
        "candidate_state_sha256": candidate_state_sha256,
        "opponent_state_sha256": opponent_state_sha256,
        "candidate_policy_mode": candidate_policy_mode,
        "opponent_policy_mode": opponent_policy_mode,
        "pair_score": float(pentanomial["score"]),
        "pentanomial": pentanomial,
        "pair_aware_logistic_interval": aggregate[
            "pair_aware_logistic_interval"
        ],
        "cap_draw_rate": float(aggregate["cap_draw_rate"]),
        "fault_counts": aggregate["fault_counts"],
        "state_sha256": _sha256_file(state_path),
        "state_size_bytes": state_path.stat().st_size,
    }


def _archive_uninitialized_output(output_dir: Path) -> Path:
    attempt_root = output_dir.parent / ".attempts" / output_dir.name
    attempt_root.mkdir(parents=True, exist_ok=True)
    index = 0
    while (attempt_root / f"attempt{index:03d}").exists():
        index += 1
    target = attempt_root / f"attempt{index:03d}"
    os.replace(output_dir, target)
    return target


def _run_arena(
    *,
    candidate_label: str,
    opponent_label: str,
    candidate_policy_mode: str,
    opponent_policy_mode: str,
    pair_count: int,
    candidate_refinement_passes: int,
    opponent_refinement_passes: int,
    opening_start_index: int,
    expected_arena_source_tree_sha256: str,
    git_commit: str,
    declared_attempt_upper_bound_dollars: str,
) -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Modal arena function cannot see CUDA")
    device_name = str(torch.cuda.get_device_name(0))
    if "L4" not in device_name:
        raise RuntimeError(f"Requested L4 but Modal exposed {device_name!r}")
    expected_source = _require_sha256(
        expected_arena_source_tree_sha256,
        label="expected_arena_source_tree_sha256",
    )
    observed_source = _arena_source_tree_sha256(Path("/root"))
    if observed_source != expected_source:
        raise RuntimeError("Local and bundled arena source identities differ")
    git_commit = _require_git_commit(git_commit)
    raw_bt4_path = _verify_raw_bt4_volume()
    candidate = _validated_training_result(candidate_label)
    opponent = _validated_training_result(opponent_label)
    candidate_policy_mode = _require_policy_mode(
        candidate_policy_mode,
        label="candidate_policy_mode",
    )
    opponent_policy_mode = _require_policy_mode(
        opponent_policy_mode,
        label="opponent_policy_mode",
    )
    if (
        candidate["state_sha256"] == opponent["state_sha256"]
        and candidate_policy_mode == opponent_policy_mode
        and candidate_refinement_passes == opponent_refinement_passes
    ):
        raise ValueError(
            "Same-checkpoint arena sides must differ by policy mode or "
            "refinement passes"
        )
    arena_label = _arena_identity(
        candidate_label=candidate["label"],
        candidate_state_sha256=candidate["state_sha256"],
        opponent_label=opponent["label"],
        opponent_state_sha256=opponent["state_sha256"],
        candidate_policy_mode=candidate_policy_mode,
        opponent_policy_mode=opponent_policy_mode,
        pair_count=pair_count,
        candidate_refinement_passes=candidate_refinement_passes,
        opponent_refinement_passes=opponent_refinement_passes,
        opening_start_index=opening_start_index,
        arena_source_tree_sha256=observed_source,
    )
    output_dir = Path(TRAINING_RESULT_MOUNT) / "arenas" / arena_label
    state_path = output_dir / "state.json"
    if state_path.is_file():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("status") == "complete":
            summary = _completed_arena_summary(
                output_dir,
                arena_label=arena_label,
                candidate_state_sha256=candidate["state_sha256"],
                opponent_state_sha256=opponent["state_sha256"],
                candidate_policy_mode=candidate_policy_mode,
                opponent_policy_mode=opponent_policy_mode,
                pair_count=pair_count,
                candidate_refinement_passes=candidate_refinement_passes,
                opponent_refinement_passes=opponent_refinement_passes,
                opening_start_index=opening_start_index,
            )
            summary["reused_immutable_result"] = True
            return summary
        resume = state.get("status") == "running"
        if not resume:
            raise RuntimeError("Arena output has an unsupported terminal state")
    elif output_dir.exists():
        _archive_uninitialized_output(output_dir)
        training_result_volume.commit()
        resume = False
    else:
        resume = False

    _materialize_arena_assets()
    trusted_roots = os.pathsep.join(
        (BASE_INPUT_MOUNT, TRAINING_RESULT_MOUNT, "/root")
    )
    environment = os.environ.copy()
    environment.update(
        {
            "CHESS_DFM_WORKSPACE_ROOT": TRAINING_RESULT_MOUNT,
            "CHESS_DFM_TRUSTED_WORKSPACE_ROOTS": trusted_roots,
            "CHESS_DFM_GIT_COMMIT": git_commit,
            "PYTHONPYCACHEPREFIX": "/tmp/chess-dfm-arena-pycache",
            "OMP_NUM_THREADS": "4",
            "OPENBLAS_NUM_THREADS": "4",
            "MKL_NUM_THREADS": "4",
        }
    )
    command = _arena_command(
        candidate_root=candidate["root"],
        opponent_root=opponent["root"],
        models_dir=raw_bt4_path.parent,
        output_dir=output_dir,
        candidate_policy_mode=candidate_policy_mode,
        opponent_policy_mode=opponent_policy_mode,
        pair_count=pair_count,
        candidate_refinement_passes=candidate_refinement_passes,
        opponent_refinement_passes=opponent_refinement_passes,
        opening_start_index=opening_start_index,
        resume=resume,
    )
    started = time.perf_counter()
    stdout_path = Path("/tmp") / f"{arena_label}.stdout.log"
    with stdout_path.open("w", encoding="utf-8") as stdout_log:
        process = subprocess.Popen(
            command,
            cwd="/root",
            env=environment,
            text=True,
            stdout=stdout_log,
            stderr=subprocess.STDOUT,
        )
        while process.poll() is None:
            time.sleep(30)
            training_result_volume.commit()
        returncode = process.wait()
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(stdout_path, output_dir / "modal_stdout.log")
    if returncode != 0:
        stdout_tail = stdout_path.read_text(
            encoding="utf-8",
            errors="replace",
        )[-4_000:]
        (output_dir / "modal_failure.json").write_text(
            json.dumps(
                {
                    "schema_version": "chess-dfm-modal-torch-arena-failure-v1",
                    "arena_label": arena_label,
                    "returncode": returncode,
                    "arena_source_tree_sha256": observed_source,
                    "stdout_log_sha256": _sha256_file(stdout_path),
                    "stdout_log_size_bytes": stdout_path.stat().st_size,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        training_result_volume.commit()
        raise RuntimeError(
            f"Torch arena subprocess exited with status {returncode}; "
            f"terminal output:\n{stdout_tail}"
        )
    wrapper_record = {
        "schema_version": "chess-dfm-modal-torch-arena-wrapper-v1",
        "arena_label": arena_label,
        "candidate": candidate,
        "opponent": opponent,
        "candidate_policy_mode": candidate_policy_mode,
        "opponent_policy_mode": opponent_policy_mode,
        "pair_count": pair_count,
        "candidate_refinement_passes": candidate_refinement_passes,
        "opponent_refinement_passes": opponent_refinement_passes,
        "opening_start_index": opening_start_index,
        "arena_source_tree_sha256": observed_source,
        "git_commit": git_commit,
        "raw_bt4_sha256": RAW_BT4_SHA256,
        "declared_attempt_upper_bound_dollars": (
            declared_attempt_upper_bound_dollars
        ),
        "device_name": device_name,
        "remote_function_seconds": time.perf_counter() - started,
    }
    for record in (wrapper_record["candidate"], wrapper_record["opponent"]):
        record["root"] = str(record["root"])
    (output_dir / "modal_arena.json").write_text(
        json.dumps(wrapper_record, allow_nan=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    training_result_volume.commit()
    summary = _completed_arena_summary(
        output_dir,
        arena_label=arena_label,
        candidate_state_sha256=candidate["state_sha256"],
        opponent_state_sha256=opponent["state_sha256"],
        candidate_policy_mode=candidate_policy_mode,
        opponent_policy_mode=opponent_policy_mode,
        pair_count=pair_count,
        candidate_refinement_passes=candidate_refinement_passes,
        opponent_refinement_passes=opponent_refinement_passes,
        opening_start_index=opening_start_index,
    )
    summary["reused_immutable_result"] = False
    return summary


_ARENA_VOLUMES = {
    BASE_INPUT_MOUNT: base_input_volume.with_mount_options(read_only=True),
    TRAINING_RESULT_MOUNT: training_result_volume,
}


@app.function(
    image=arena_image,
    volumes=_ARENA_VOLUMES,
    gpu="L4",
    cpu=4.0,
    memory=16384,
    timeout=2400,
    startup_timeout=600,
    retries=0,
    max_containers=1,
    single_use_containers=True,
)
def arena_l4(**kwargs: Any) -> dict[str, Any]:
    return _run_arena(**kwargs)


def _arena_job_spec(
    pair_count: int,
    candidate_refinement_passes: int,
    opponent_refinement_passes: int,
) -> tuple[Any, Decimal]:
    from research.interpretability.remote_cost import ModalJobSpec

    if pair_count not in ALLOWED_PAIR_COUNTS:
        raise ValueError(f"pair_count must be one of {ALLOWED_PAIR_COUNTS}")
    for label, passes in (
        ("candidate_refinement_passes", candidate_refinement_passes),
        ("opponent_refinement_passes", opponent_refinement_passes),
    ):
        if passes not in ALLOWED_REFINEMENT_PASSES:
            raise ValueError(
                f"{label} must be one of {ALLOWED_REFINEMENT_PASSES}"
            )
    return (
        ModalJobSpec(
            name=(
                f"torch-hero-arena-{pair_count}p-"
                f"cpass{candidate_refinement_passes}-"
                f"opass{opponent_refinement_passes}"
            ),
            timeout_seconds=2400,
            startup_timeout_seconds=600,
            retries=0,
            cpu_cores=Decimal("4"),
            memory_gib=Decimal("16"),
            gpu="L4",
        ),
        Decimal("1.00"),
    )


def _local_git_commit(repo_root: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError("Cannot resolve local Git commit for arena provenance")
    return _require_git_commit(completed.stdout.strip())


@app.local_entrypoint()
def main(
    candidate_label: str,
    opponent_label: str,
    candidate_policy_mode: str = "dfm",
    opponent_policy_mode: str = "dfm",
    pair_count: int = 128,
    candidate_refinement_passes: int = 1,
    opponent_refinement_passes: int = 1,
    opening_start_index: int = 0,
    recorded_spend: str = "",
    per_job_cap: str = "",
) -> None:
    from research.interpretability.remote_cost import (
        enforce_modal_budget,
        estimate_modal_job,
        modal_month_spend,
    )

    if pair_count not in ALLOWED_PAIR_COUNTS:
        raise ValueError(f"pair_count must be one of {ALLOWED_PAIR_COUNTS}")
    for label, passes in (
        ("candidate_refinement_passes", candidate_refinement_passes),
        ("opponent_refinement_passes", opponent_refinement_passes),
    ):
        if passes not in ALLOWED_REFINEMENT_PASSES:
            raise ValueError(
                f"{label} must be one of {ALLOWED_REFINEMENT_PASSES}"
            )
    if opening_start_index < 0:
        raise ValueError("opening_start_index must be non-negative")
    candidate_label = _require_result_label(candidate_label)
    opponent_label = _require_result_label(opponent_label)
    candidate_policy_mode = _require_policy_mode(
        candidate_policy_mode,
        label="candidate_policy_mode",
    )
    opponent_policy_mode = _require_policy_mode(
        opponent_policy_mode,
        label="opponent_policy_mode",
    )
    if (
        candidate_label == opponent_label
        and candidate_policy_mode == opponent_policy_mode
        and candidate_refinement_passes == opponent_refinement_passes
    ):
        raise ValueError(
            "Same-result arena sides must differ by policy mode or "
            "refinement passes"
        )
    spec, default_cap = _arena_job_spec(
        pair_count,
        candidate_refinement_passes,
        opponent_refinement_passes,
    )
    estimate = estimate_modal_job(spec)
    # Invoking a second Modal CLI from inside `modal run` can deadlock on the
    # local profile lock. A supplied value is therefore authoritative and must
    # be refreshed immediately before launch; an omitted value keeps the
    # fail-closed live query.
    spend = (
        Decimal(recorded_spend) if recorded_spend else modal_month_spend()
    )
    decision = enforce_modal_budget(
        estimate,
        recorded_month_spend=spend,
        per_job_cap=Decimal(per_job_cap) if per_job_cap else default_cap,
    )
    repo_root = Path(__file__).resolve().parents[2]
    arena_source_tree_sha256 = _arena_source_tree_sha256(repo_root)
    git_commit = _local_git_commit(repo_root)
    print(
        json.dumps(
            {"launch_stage": "arena-l4", "cost_guard": decision},
            sort_keys=True,
        )
    )
    call = arena_l4.spawn(
        candidate_label=candidate_label,
        opponent_label=opponent_label,
        candidate_policy_mode=candidate_policy_mode,
        opponent_policy_mode=opponent_policy_mode,
        pair_count=pair_count,
        candidate_refinement_passes=candidate_refinement_passes,
        opponent_refinement_passes=opponent_refinement_passes,
        opening_start_index=opening_start_index,
        expected_arena_source_tree_sha256=arena_source_tree_sha256,
        git_commit=git_commit,
        declared_attempt_upper_bound_dollars=decision[
            "declared_attempt_upper_bound"
        ],
    )
    print(
        json.dumps(
            {
                "arena_submission": {
                    "function_call_id": call.object_id,
                    "mode": "asynchronous_spawn",
                }
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    raise SystemExit("Use `modal run research/infra/modal_arena_app.py ...`")
