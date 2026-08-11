"""Budget-gated Modal staging and training for the Hero v2 Torch program."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.request
from decimal import Decimal
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO

import modal


APP_NAME = "chess-dfm-training"
RUNNER_REVISION = "hero-v2-modal-training-v7"
BASE_GIT_COMMIT = "86ba0b37e3e188eab115b45a1460d079fac341f8"
BASE_INPUT_BUNDLE_SHA256 = "8be17957ad779e3756092306daa95ad9c1c7f5d482b6409bda7a16bc6d6c7a44"
BASE_INPUT_VOLUME_NAME = "chess-dfm-interpretability-inputs"
TRAINING_INPUT_VOLUME_NAME = "chess-dfm-training-inputs"
TRAINING_RESULT_VOLUME_NAME = "chess-dfm-training-results"

WORKSPACE_MOUNT = "/workspace"
BASE_INPUT_MOUNT = f"{WORKSPACE_MOUNT}/base-inputs"
TRAINING_INPUT_MOUNT = f"{WORKSPACE_MOUNT}/training-inputs"
TRAINING_RESULT_MOUNT = f"{WORKSPACE_MOUNT}/training-results"
EPHEMERAL_COMPILE_CACHE_ROOT = "/tmp/chess-dfm-training-compile-cache"
TRAINING_DATASET_LABEL = "trajectory-v3-lc0-test80-h8-sets1-3-20260430"
LC0_PROPOSAL_A_DATASET_LABEL = "lc0-sequential-test80-20240401-0117-pilot-v1"
LC0_PROPOSAL_A_STAGE_SCHEMA = "chess-dfm-modal-lc0-sequential-pilot-v1"
LC0_PROPOSAL_A_INVENTORY_SHA256 = (
    "4a3a640970aeac08fd152790e4b5dfe7b19ac600e6bec08d086eaf3510fe012e"
)
LC0_PROPOSAL_A_INVENTORY_SIZE_BYTES = 278_048_204
LC0_PROPOSAL_A_SOURCE_ARCHIVE_SHA256 = (
    "1c5e5d0d1d335bfeca9693700a1ad1415abfb190772bd051d1f00cb193eb3c2f"
)
LC0_PROPOSAL_A_FULL_EPOCH_EXAMPLES = 7_960_576
LC0_PROPOSAL_A_CHUNK_INDICES = (0, 32, 64, 96, 128, 160, 192, 224, 253)
HERO1_EXACT_PHASE1_RESULT_LABEL = (
    "hero-training-phase1-hero1_exact-l40s-fae7b0ccef5ca3668ed5"
)
HERO1_EXACT_PHASE1_STATE_SHA256 = (
    "e518083ee5476e7ff15e7c34caf9c0505f8fb7d330ea5a6882e4e9e5c1ef1912"
)

TRAJECTORY_DRIVE_FILE_ID = "14jgdmEZsBVMFtQ7Vvjm3PTWU2QlE45KV"
TRAJECTORY_TAR_SHA256 = "d8feffa259580563c8097fa4a604ca415469dd80da66dabda170ac9ce929b968"
TRAJECTORY_TAR_SIZE_BYTES = 11_531_386_880
TRAJECTORY_FILE_COUNT = 30_728
TRAJECTORY_FILE_BYTES = 11_507_849_049

RAW_BT4_RELATIVE = "models/raw/BT4_exported.pb.gz"
RAW_BT4_SIZE_BYTES = 335_916_563
RAW_BT4_SHA256 = "61e43e98d2c4cb747498c6bc13a5bbb3b07e985d3f5d87f0351a4db855fa0651"
SOURCE_TREE_PATHS = (
    "research/train_torch.py",
    "research/prepare.py",
    "research/arena.py",
    "research/arena_history_trust.py",
    "research/play_arena.py",
    "research/infra/modal_train_app.py",
    "research/stitch_training_segments.py",
    "research/interpretability/remote_cost.py",
)
COMPILE_SOURCE_TREE_PATHS = (
    "research/train_torch.py",
    "research/prepare.py",
)
MOVEMENT_SOURCE_TREE_PATHS = (
    "research/infra/modal_train_app.py",
    "research/train_torch.py",
    "research/interpretability/models.py",
    "research/interpretability/parameter_diff.py",
)

app = modal.App(APP_NAME)
base_input_volume = modal.Volume.from_name(BASE_INPUT_VOLUME_NAME, create_if_missing=False)
training_input_volume = modal.Volume.from_name(TRAINING_INPUT_VOLUME_NAME, create_if_missing=True)
training_result_volume = modal.Volume.from_name(TRAINING_RESULT_VOLUME_NAME, create_if_missing=True)

control_image = (
    modal.Image.debian_slim(python_version="3.12")
    .add_local_file(
        "research/eval/hero_epoch_v1/manifest.json",
        "/opt/chess-assets/hero_eval_manifest.json",
        copy=True,
    )
    .add_local_file(
        "research/eval/hero_epoch_v1/position_indices.npz",
        "/opt/chess-assets/position_indices.npz",
        copy=True,
    )
    .add_local_file(
        "research/assets/arena/promotion-ply12-n2048-v3.json",
        "/opt/chess-assets/promotion-ply12-n2048-v3.json",
        copy=True,
    )
    .add_local_file(
        "research/assets/arena/promotion-ply12-n2048-histories-v3.json",
        "/opt/chess-assets/promotion-ply12-n2048-histories-v3.json",
        copy=True,
    )
    .add_local_file(
        "research/assets/arena/promotion-ply12-n2048-v3-contract.json",
        "/opt/chess-assets/promotion-ply12-n2048-v3-contract.json",
        copy=True,
    )
)

gpu_image = (
    modal.Image.from_registry("pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime")
    .apt_install("gcc", "g++")
    .uv_pip_install(
        "ml-dtypes==0.5.3",
        "numpy==2.2.6",
        "protobuf==6.31.1",
        "python-chess==1.999",
        "safetensors==0.5.3",
        "zstandard==0.23.0",
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


@dataclasses.dataclass(frozen=True)
class TrainingArm:
    recipe: str
    data_format: str = "trajectory_v3"
    dataset_label: str = TRAINING_DATASET_LABEL
    main_lr_multiplier: float | None = None
    encoder_lr_ratio: float | None = None
    lr_total_examples: int | None = None
    optimizer_precision: str | None = None
    weight_decay_multiplier: float | None = None
    weight_decay_mode: str | None = None
    legality_coeff: float | None = None
    policy_distill_coeff: float | None = None
    policy_distill_teacher_result_label: str | None = None
    policy_distill_teacher_state_sha256: str | None = None
    wdl_include_current_state: bool | None = None
    dfm_closed_loop_mode: str | None = None


@dataclasses.dataclass(frozen=True)
class TrainingProfile:
    steps: int
    validation_updates: tuple[int, ...]
    save_recovery: bool
    save_model: bool
    log_every: int
    recovery_updates: tuple[int, ...] = ()
    resume_from_profile: str | None = None


TRAINING_ARMS = {
    "hero1_exact": TrainingArm(recipe="hero"),
    "v2_bf16_1_30": TrainingArm(
        recipe="hero_v2",
        main_lr_multiplier=1.0,
        encoder_lr_ratio=1.0 / 30.0,
        optimizer_precision="parameter",
        weight_decay_mode="decoupled",
    ),
    "v2_fp32_1_30": TrainingArm(
        recipe="hero_v2",
        main_lr_multiplier=1.0,
        encoder_lr_ratio=1.0 / 30.0,
        optimizer_precision="fp32_master",
        weight_decay_mode="decoupled",
    ),
    "v2_fp32_1_12": TrainingArm(
        recipe="hero_v2",
        main_lr_multiplier=1.0,
        encoder_lr_ratio=1.0 / 12.0,
        optimizer_precision="fp32_master",
        weight_decay_mode="decoupled",
    ),
    "v2_fp32_1_10": TrainingArm(
        recipe="hero_v2",
        main_lr_multiplier=1.0,
        encoder_lr_ratio=1.0 / 10.0,
        optimizer_precision="fp32_master",
        weight_decay_mode="decoupled",
    ),
    "v2_fp32_1_3": TrainingArm(
        recipe="hero_v2",
        main_lr_multiplier=1.0,
        encoder_lr_ratio=1.0 / 3.0,
        optimizer_precision="fp32_master",
        weight_decay_mode="decoupled",
    ),
    "v2_encoder_fp32_1_30": TrainingArm(
        recipe="hero_v2",
        main_lr_multiplier=1.0,
        encoder_lr_ratio=1.0 / 30.0,
        optimizer_precision="encoder_fp32_master",
        weight_decay_mode="decoupled",
    ),
    "v2_encoder_fp32_1_12": TrainingArm(
        recipe="hero_v2",
        main_lr_multiplier=1.0,
        encoder_lr_ratio=1.0 / 12.0,
        optimizer_precision="encoder_fp32_master",
        weight_decay_mode="decoupled",
    ),
    "v2_trunk_fp32_1_12": TrainingArm(
        recipe="hero_v2",
        main_lr_multiplier=1.0,
        encoder_lr_ratio=1.0 / 12.0,
        optimizer_precision="encoder_trunk_fp32_master",
        weight_decay_mode="decoupled",
    ),
    "v2_main_fp32_1_30": TrainingArm(
        recipe="hero_v2",
        main_lr_multiplier=1.0,
        encoder_lr_ratio=1.0 / 30.0,
        optimizer_precision="main_fp32_master",
        weight_decay_mode="decoupled",
    ),
    "v2_fp32_1_12_legality0": TrainingArm(
        recipe="hero_v2",
        main_lr_multiplier=1.0,
        encoder_lr_ratio=1.0 / 12.0,
        optimizer_precision="fp32_master",
        weight_decay_mode="decoupled",
        legality_coeff=0.0,
    ),
    "v2_fp32_1_12_legality0_distill1": TrainingArm(
        recipe="hero_v2",
        main_lr_multiplier=1.0,
        encoder_lr_ratio=1.0 / 12.0,
        optimizer_precision="fp32_master",
        weight_decay_mode="decoupled",
        legality_coeff=0.0,
        policy_distill_coeff=1.0,
    ),
    "v2_fp32_1_12_legality0_hero1distill1": TrainingArm(
        recipe="hero_v2",
        main_lr_multiplier=1.0,
        encoder_lr_ratio=1.0 / 12.0,
        optimizer_precision="fp32_master",
        weight_decay_mode="decoupled",
        legality_coeff=0.0,
        policy_distill_coeff=1.0,
        policy_distill_teacher_result_label=HERO1_EXACT_PHASE1_RESULT_LABEL,
        policy_distill_teacher_state_sha256=HERO1_EXACT_PHASE1_STATE_SHA256,
    ),
    "v2_fp32_1_12_legality0_hero1distill1_closedloop": TrainingArm(
        recipe="hero_v2",
        main_lr_multiplier=1.0,
        encoder_lr_ratio=1.0 / 12.0,
        optimizer_precision="fp32_master",
        weight_decay_mode="decoupled",
        legality_coeff=0.0,
        policy_distill_coeff=1.0,
        policy_distill_teacher_result_label=HERO1_EXACT_PHASE1_RESULT_LABEL,
        policy_distill_teacher_state_sha256=HERO1_EXACT_PHASE1_STATE_SHA256,
        dfm_closed_loop_mode="predicted_jepa_tokens",
    ),
}
for _ratio_label, _ratio in (
    ("1_30", 1.0 / 30.0),
    ("1_12", 1.0 / 12.0),
    ("1_10", 1.0 / 10.0),
    ("1_3", 1.0 / 3.0),
):
    for _lr_label, _lr_multiplier in (
        ("lr075", 0.75),
        ("lr125", 1.25),
        ("lr150", 1.5),
    ):
        TRAINING_ARMS[f"v2_fp32_{_ratio_label}_{_lr_label}"] = TrainingArm(
            recipe="hero_v2",
            main_lr_multiplier=_lr_multiplier,
            encoder_lr_ratio=_ratio,
            optimizer_precision="fp32_master",
            weight_decay_mode="decoupled",
        )
    TRAINING_ARMS[f"v2_fp32_{_ratio_label}_cautious"] = TrainingArm(
        recipe="hero_v2",
        main_lr_multiplier=1.0,
        encoder_lr_ratio=_ratio,
        optimizer_precision="fp32_master",
        weight_decay_mode="cautious",
    )
for _lr_label, _lr_multiplier in (
    ("5em5", 0.1),
    ("1em4", 0.2),
    ("2em4", 0.4),
    ("4em4", 0.8),
):
    _proposal_base = TrainingArm(
        recipe="hero_v2",
        data_format="lc0_sequential",
        dataset_label=LC0_PROPOSAL_A_DATASET_LABEL,
        main_lr_multiplier=_lr_multiplier,
        encoder_lr_ratio=1.0,
        lr_total_examples=LC0_PROPOSAL_A_FULL_EPOCH_EXAMPLES,
        optimizer_precision="fp32_master",
        weight_decay_mode="decoupled",
        legality_coeff=0.0,
        wdl_include_current_state=True,
    )
    TRAINING_ARMS[f"proposal_a_equal_lr_{_lr_label}_wd0"] = dataclasses.replace(
        _proposal_base,
        weight_decay_multiplier=0.0,
    )
    TRAINING_ARMS[f"proposal_a_equal_lr_{_lr_label}_wd1"] = dataclasses.replace(
        _proposal_base,
        weight_decay_multiplier=1.0,
    )
    TRAINING_ARMS[
        f"proposal_a_equal_lr_{_lr_label}_wd1_cautious"
    ] = dataclasses.replace(
        _proposal_base,
        weight_decay_multiplier=1.0,
        weight_decay_mode="cautious",
    )
TRAINING_PROFILES = {
    "smoke": TrainingProfile(
        steps=1,
        validation_updates=(),
        save_recovery=False,
        save_model=False,
        log_every=1,
    ),
    "benchmark": TrainingProfile(
        steps=100,
        validation_updates=(),
        save_recovery=False,
        save_model=False,
        log_every=10,
    ),
    "diagnostic": TrainingProfile(
        steps=10,
        validation_updates=(),
        save_recovery=False,
        save_model=False,
        log_every=1,
    ),
    "proposal_screen": TrainingProfile(
        steps=252,
        validation_updates=(252,),
        save_recovery=False,
        save_model=False,
        log_every=10,
    ),
    "phase1": TrainingProfile(
        steps=554,
        validation_updates=(554,),
        save_recovery=True,
        save_model=True,
        log_every=20,
        recovery_updates=(100, 200, 300, 400, 500, 554),
    ),
    "u1024": TrainingProfile(
        steps=1_024,
        validation_updates=(554, 1_024),
        save_recovery=True,
        save_model=True,
        log_every=20,
        recovery_updates=(200, 400, 554, 700, 850, 1_024),
    ),
    "phase2": TrainingProfile(
        steps=2_768,
        validation_updates=(1_384, 2_768),
        save_recovery=True,
        save_model=True,
        log_every=20,
        recovery_updates=(954, 1_354, 1_754, 2_154, 2_554, 2_768),
        resume_from_profile="phase1",
    ),
}


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(value: str, *, label: str) -> str:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _require_git_commit(value: str) -> str:
    if len(value) != 40 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError("Git commit must be a lowercase SHA-1")
    return value


def _drive_download_url(file_id: str) -> str:
    if not file_id or any(not (character.isalnum() or character in "-_") for character in file_id):
        raise ValueError("Unsafe Google Drive file id")
    return (
        "https://drive.usercontent.google.com/download"
        f"?id={file_id}&export=download&confirm=t"
    )


def _safe_tar_relative(name: str) -> Path | None:
    logical = PurePosixPath(name)
    if logical.is_absolute():
        raise ValueError(f"Absolute tar member is forbidden: {name!r}")
    parts = tuple(part for part in logical.parts if part not in {"", "."})
    if any(part == ".." for part in parts):
        raise ValueError(f"Traversing tar member is forbidden: {name!r}")
    if not parts:
        return None
    return Path(*parts)


class _HashingReader:
    def __init__(self, source: BinaryIO):
        self.source = source
        self.digest = hashlib.sha256()
        self.size_bytes = 0

    def read(self, size: int = -1) -> bytes:
        payload = self.source.read(size)
        self.digest.update(payload)
        self.size_bytes += len(payload)
        return payload

    def hexdigest(self) -> str:
        return self.digest.hexdigest()


def _extract_streaming_tar(
    source: BinaryIO,
    target: Path,
) -> dict[str, int | str]:
    reader = _HashingReader(source)
    file_count = 0
    file_bytes = 0
    with tarfile.open(fileobj=reader, mode="r|") as archive:
        for member in archive:
            relative = _safe_tar_relative(member.name)
            if relative is None:
                continue
            destination = target / relative
            if member.isdir():
                destination.mkdir(parents=True, exist_ok=True)
                continue
            if not member.isfile():
                raise ValueError(f"Non-regular tar member is forbidden: {member.name!r}")
            if destination.exists():
                raise ValueError(f"Duplicate tar member: {member.name!r}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            extracted = archive.extractfile(member)
            if extracted is None:
                raise RuntimeError(f"Cannot read tar member: {member.name!r}")
            temporary = destination.with_name(f".{destination.name}.partial")
            written = 0
            with temporary.open("xb") as output:
                while chunk := extracted.read(8 * 1024 * 1024):
                    output.write(chunk)
                    written += len(chunk)
            if written != member.size:
                raise RuntimeError(f"Tar member size mismatch: {member.name!r}")
            os.replace(temporary, destination)
            file_count += 1
            file_bytes += written
    while reader.read(8 * 1024 * 1024):
        pass
    return {
        "archive_sha256": reader.hexdigest(),
        "archive_size_bytes": reader.size_bytes,
        "file_count": file_count,
        "file_bytes": file_bytes,
    }


def _retarget_hero_eval_manifest(
    template: dict[str, Any],
    *,
    dataset_root: Path,
) -> dict[str, Any]:
    trajectory_root = dataset_root / "trajectory_v3"
    contract_root = dataset_root / "contracts" / "hero_epoch_v1"
    arena_root = contract_root / "arena"
    template["dataset"]["manifest"] = str(trajectory_root / "manifest.json")
    for split in ("train", "val", "test"):
        if split in template["dataset"]:
            template["dataset"][split]["path"] = str(trajectory_root / split)
    template["position_indices"]["path"] = str(contract_root / "position_indices.npz")
    paired = template["paired_arena"]
    paired["opening_pool"]["path"] = str(arena_root / "promotion-ply12-n2048-v3.json")
    paired["opening_histories"]["path"] = str(
        arena_root / "promotion-ply12-n2048-histories-v3.json"
    )
    paired["opening_contract"]["path"] = str(
        arena_root / "promotion-ply12-n2048-v3-contract.json"
    )
    return template


def _copy_contract_assets(
    dataset_root: Path,
    *,
    logical_dataset_root: Path | None = None,
) -> Path:
    contract_root = dataset_root / "contracts" / "hero_epoch_v1"
    arena_root = contract_root / "arena"
    arena_root.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(
        "/opt/chess-assets/position_indices.npz",
        contract_root / "position_indices.npz",
    )
    for name in (
        "promotion-ply12-n2048-v3.json",
        "promotion-ply12-n2048-histories-v3.json",
        "promotion-ply12-n2048-v3-contract.json",
    ):
        shutil.copyfile(f"/opt/chess-assets/{name}", arena_root / name)

    template = json.loads(
        Path("/opt/chess-assets/hero_eval_manifest.json").read_text(encoding="utf-8")
    )
    template = _retarget_hero_eval_manifest(
        template,
        dataset_root=logical_dataset_root or dataset_root,
    )
    manifest_path = contract_root / "manifest.json"
    manifest_path.write_text(
        json.dumps(template, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def _training_dataset_root(
    dataset_label: str = TRAINING_DATASET_LABEL,
) -> Path:
    if Path(dataset_label).name != dataset_label:
        raise ValueError("Unsafe training dataset label")
    return Path(TRAINING_INPUT_MOUNT) / dataset_label


def _validate_training_stage() -> dict[str, Any]:
    root = _training_dataset_root()
    marker_path = root / ".stage.json"
    if not root.is_dir() or not marker_path.is_file():
        raise RuntimeError("Training dataset has not been staged")
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    expected = {
        "schema_version": "chess-dfm-modal-training-stage-v1",
        "dataset_label": TRAINING_DATASET_LABEL,
        "drive_file_id": TRAJECTORY_DRIVE_FILE_ID,
        "archive_sha256": TRAJECTORY_TAR_SHA256,
        "archive_size_bytes": TRAJECTORY_TAR_SIZE_BYTES,
        "file_count": TRAJECTORY_FILE_COUNT,
        "file_bytes": TRAJECTORY_FILE_BYTES,
    }
    if any(marker.get(key) != value for key, value in expected.items()):
        raise RuntimeError("Training dataset stage marker drift")
    required = (
        root / "trajectory_v3" / "manifest.json",
        root / "trajectory_v3" / "train",
        root / "trajectory_v3" / "val",
        root / "contracts" / "hero_epoch_v1" / "manifest.json",
        root / "contracts" / "hero_epoch_v1" / "position_indices.npz",
    )
    if any(not path.exists() for path in required):
        raise RuntimeError("Training dataset stage inventory is incomplete")
    manifest_path = root / "contracts" / "hero_epoch_v1" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_manifest = _retarget_hero_eval_manifest(
        json.loads(json.dumps(manifest)), dataset_root=root
    )
    if manifest != expected_manifest:
        raise RuntimeError("Training evaluation contract path drift")
    return marker


def _validate_lc0_proposal_a_stage() -> dict[str, Any]:
    root = _training_dataset_root(LC0_PROPOSAL_A_DATASET_LABEL)
    marker_path = root / ".stage.json"
    if not root.is_dir() or not marker_path.is_file():
        raise RuntimeError("Proposal A LC0 pilot has not been staged")
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    expected = {
        "schema_version": LC0_PROPOSAL_A_STAGE_SCHEMA,
        "dataset_label": LC0_PROPOSAL_A_DATASET_LABEL,
        "source_archive_sha256": LC0_PROPOSAL_A_SOURCE_ARCHIVE_SHA256,
        "selected_chunk_indices": list(LC0_PROPOSAL_A_CHUNK_INDICES),
        "full_batch_1024_epoch_examples": LC0_PROPOSAL_A_FULL_EPOCH_EXAMPLES,
        "inventory_file_count": 400,
        "inventory_size_bytes": LC0_PROPOSAL_A_INVENTORY_SIZE_BYTES,
        "inventory_sha256": LC0_PROPOSAL_A_INVENTORY_SHA256,
        "split_totals": {
            "train": {
                "positions": 264_105,
                "trainable_starts": 261_701,
                "full_batches": 252,
            },
            "validation": {
                "positions": 1_473,
                "trainable_starts": 1_458,
                "full_batches": 18,
            },
            "test": {
                "positions": 2_201,
                "trainable_starts": 2_181,
                "full_batches": 31,
            },
        },
    }
    if any(marker.get(key) != value for key, value in expected.items()):
        raise RuntimeError("Proposal A LC0 pilot stage marker drift")

    sequential_root = root / "lc0_sequential"
    source_manifest = sequential_root / "source_dataset_manifest.json"
    train_manifests = tuple(sorted(sequential_root.glob("chunks/*/train/manifest.json")))
    validation_manifests = tuple(
        sorted(sequential_root.glob("chunks/*/validation/manifest.json"))
    )
    if (
        not source_manifest.is_file()
        or len(train_manifests) != len(LC0_PROPOSAL_A_CHUNK_INDICES)
        or len(validation_manifests) != 8
    ):
        raise RuntimeError("Proposal A LC0 pilot inventory is incomplete")
    return marker


@app.function(
    image=control_image,
    volumes={TRAINING_INPUT_MOUNT: training_input_volume},
    cpu=2.0,
    memory=4096,
    timeout=7200,
    startup_timeout=300,
    retries=0,
    max_containers=1,
    single_use_containers=True,
)
def stage_trajectory_dataset() -> dict[str, Any]:
    """Stream, hash, safely extract, and atomically commit the public Drive corpus."""

    target = _training_dataset_root()
    if target.exists():
        _copy_contract_assets(target, logical_dataset_root=target)
        marker = _validate_training_stage()
        training_input_volume.commit()
        return {**marker, "cache_hit": True, "network_download_bytes": 0}

    partial = target.with_name(f".{target.name}.partial")
    if partial.exists():
        shutil.rmtree(partial)
    trajectory_root = partial / "trajectory_v3"
    trajectory_root.mkdir(parents=True)

    request = urllib.request.Request(
        _drive_download_url(TRAJECTORY_DRIVE_FILE_ID),
        headers={"User-Agent": "chess-dfm-modal-stage/1"},
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            content_length = response.headers.get("Content-Length")
            if content_length is not None and int(content_length) != TRAJECTORY_TAR_SIZE_BYTES:
                raise RuntimeError("Google Drive trajectory size header drift")
            extracted = _extract_streaming_tar(response, trajectory_root)
        expected = {
            "archive_sha256": TRAJECTORY_TAR_SHA256,
            "archive_size_bytes": TRAJECTORY_TAR_SIZE_BYTES,
            "file_count": TRAJECTORY_FILE_COUNT,
            "file_bytes": TRAJECTORY_FILE_BYTES,
        }
        if any(extracted[key] != value for key, value in expected.items()):
            raise RuntimeError(f"Extracted trajectory identity drift: {extracted}")
        eval_manifest = _copy_contract_assets(
            partial,
            logical_dataset_root=target,
        )
        marker = {
            "schema_version": "chess-dfm-modal-training-stage-v1",
            "dataset_label": TRAINING_DATASET_LABEL,
            "drive_file_id": TRAJECTORY_DRIVE_FILE_ID,
            **extracted,
            "eval_manifest": str(
                target
                / eval_manifest.relative_to(partial)
            ),
            "stage_seconds": time.perf_counter() - started,
            "gpu_active": False,
        }
        (partial / ".stage.json").write_text(
            json.dumps(marker, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(partial, target)
        training_input_volume.commit()
        return {**marker, "cache_hit": False, "network_download_bytes": extracted["archive_size_bytes"]}
    except BaseException:
        shutil.rmtree(partial, ignore_errors=True)
        raise


def _verify_raw_bt4_volume() -> Path:
    bundle_root = Path(BASE_INPUT_MOUNT) / "bundles" / BASE_INPUT_BUNDLE_SHA256
    marker_path = Path(BASE_INPUT_MOUNT) / "bundles" / f"{BASE_INPUT_BUNDLE_SHA256}.stage.json"
    if not marker_path.is_file():
        raise RuntimeError("Base input bundle is not staged on the Modal Volume")
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    bundle = marker.get("bundle")
    if not isinstance(bundle, dict) or bundle.get("bundle_sha256") != BASE_INPUT_BUNDLE_SHA256:
        raise RuntimeError("Base input bundle marker drift")
    record = bundle.get("files", {}).get(RAW_BT4_RELATIVE)
    if not isinstance(record, dict):
        raise RuntimeError("Base input bundle has no raw BT4 record")
    if (
        int(record.get("size_bytes", -1)) != RAW_BT4_SIZE_BYTES
        or record.get("sha256") != RAW_BT4_SHA256
    ):
        raise RuntimeError("Raw BT4 bundle identity drift")
    raw_path = bundle_root / RAW_BT4_RELATIVE
    if not raw_path.is_file() or raw_path.stat().st_size != RAW_BT4_SIZE_BYTES:
        raise RuntimeError("Raw BT4 staged file drift")
    return raw_path


def _tree_sha256(
    repo_root: Path,
    *,
    relative_paths: tuple[str, ...],
) -> str:
    candidates = [repo_root / relative for relative in relative_paths]
    candidates.extend(sorted((repo_root / "chess_dfm_jax").rglob("*.py")))
    candidates.extend(
        (
            repo_root / "chess_dfm_jax" / "policy_moves.txt",
            repo_root / "chess_dfm_jax" / "policy_attn_map.txt",
        )
    )
    unique = sorted({path.resolve() for path in candidates})
    if any(not path.is_file() for path in unique):
        missing = [str(path) for path in unique if not path.is_file()]
        raise FileNotFoundError(f"Training source identity is incomplete: {missing}")
    digest = hashlib.sha256()
    resolved_root = repo_root.resolve()
    for path in unique:
        relative = path.relative_to(resolved_root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(_sha256_file(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _source_tree_sha256(repo_root: Path) -> str:
    return _tree_sha256(repo_root, relative_paths=SOURCE_TREE_PATHS)


def _compile_source_tree_sha256(repo_root: Path) -> str:
    return _tree_sha256(repo_root, relative_paths=COMPILE_SOURCE_TREE_PATHS)


def _movement_source_tree_sha256(repo_root: Path) -> str:
    return _tree_sha256(repo_root, relative_paths=MOVEMENT_SOURCE_TREE_PATHS)


def _run_identity(
    *,
    profile_name: str,
    arm_name: str,
    gpu: str,
    source_tree_sha256: str,
) -> str:
    profile = TRAINING_PROFILES[profile_name]
    arm = TRAINING_ARMS[arm_name]
    dataset_identity_sha256 = (
        TRAJECTORY_TAR_SHA256
        if arm.data_format == "trajectory_v3"
        else LC0_PROPOSAL_A_INVENTORY_SHA256
    )
    identity = hashlib.sha256(
        _canonical_json_bytes(
            {
                "runner_revision": RUNNER_REVISION,
                "profile": dataclasses.asdict(profile),
                "arm": dataclasses.asdict(arm),
                "gpu": gpu,
                "source_tree_sha256": _require_sha256(
                    source_tree_sha256,
                    label="source_tree_sha256",
                ),
                "dataset": {
                    "format": arm.data_format,
                    "label": arm.dataset_label,
                    "identity_sha256": dataset_identity_sha256,
                },
                "raw_bt4_sha256": RAW_BT4_SHA256,
            }
        )
    ).hexdigest()
    return f"hero-training-{profile_name}-{arm_name}-{gpu.lower()}-{identity[:20]}"


def _training_command(
    *,
    profile_name: str,
    arm_name: str,
    output_dir: Path,
    raw_bt4_path: Path,
    data_root: Path,
    eval_manifest: Path,
    policy_distill_teacher_checkpoint_dir: Path | None = None,
    resume_checkpoint: Path | None = None,
    resume_update: int = 0,
) -> list[str]:
    profile = TRAINING_PROFILES[profile_name]
    arm = TRAINING_ARMS[arm_name]
    if not 0 <= resume_update < profile.steps:
        raise ValueError("Resume update must be in [0, profile.steps)")
    if profile.save_recovery and (
        not profile.recovery_updates
        or profile.recovery_updates[-1] != profile.steps
    ):
        raise RuntimeError("Recovery profile must checkpoint its terminal update")
    save_updates = tuple(
        update
        for update in profile.recovery_updates
        if update > resume_update
    )
    command = [
        sys.executable,
        "-m",
        "research.train_torch",
        "train",
        "--recipe",
        arm.recipe,
        "--remat-mode",
        "bt4-projector",
        "--attention-impl",
        "sdpa-all",
        "--compile-regions",
        "fresh",
        "--raw-bt4-path",
        str(raw_bt4_path),
        "--data-root",
        str(data_root),
        "--data-format",
        arm.data_format,
        "--output-dir",
        str(output_dir),
        "--batch-size",
        "1024",
        "--steps",
        str(profile.steps),
        "--train-seconds",
        "0",
        "--data-start",
        "0",
        "--seed",
        "0",
        "--threads",
        "2",
        "--log-every",
        str(profile.log_every),
        "--prefetch-depth",
        "1",
        "--gpu-monitor-interval-ms",
        "100",
        "--profile-update",
        "0",
        "--hero-eval-manifest",
        str(eval_manifest),
        "--save-every",
        "0",
        "--max-checkpoints",
        str(len(save_updates) + int(profile.save_model)),
    ]
    if profile.validation_updates:
        command.append("--validation-updates")
        command.extend(str(update) for update in profile.validation_updates)
    if resume_checkpoint is not None:
        command.extend(("--resume-checkpoint", str(resume_checkpoint)))
    if save_updates:
        command.append("--save-updates")
        command.extend(str(update) for update in save_updates)
    if profile.save_model:
        command.append("--save-final")
    if arm.recipe == "hero_v2":
        assert arm.main_lr_multiplier is not None
        assert arm.encoder_lr_ratio is not None
        assert arm.optimizer_precision is not None
        assert arm.weight_decay_mode is not None
        command.extend(
            (
                "--main-lr-multiplier",
                repr(arm.main_lr_multiplier),
                "--encoder-lr-ratio",
                repr(arm.encoder_lr_ratio),
                "--lr-schedule-kind",
                "warmup_stable_linear_decay",
                "--wsd-decay-fraction",
                "0.2",
                "--optimizer-precision",
                arm.optimizer_precision,
                "--weight-decay-mode",
                arm.weight_decay_mode,
            )
        )
        if arm.lr_total_examples is not None:
            command.extend(
                ("--lr-total-examples", str(arm.lr_total_examples))
            )
        if arm.weight_decay_multiplier is not None:
            command.extend(
                (
                    "--weight-decay-multiplier",
                    repr(arm.weight_decay_multiplier),
                )
            )
        if arm.legality_coeff is not None:
            command.extend(("--legality-coeff", repr(arm.legality_coeff)))
        if arm.policy_distill_coeff is not None:
            command.extend(
                (
                    "--policy-distill-coeff",
                    repr(arm.policy_distill_coeff),
                )
            )
            if arm.policy_distill_teacher_result_label is None:
                if arm.policy_distill_teacher_state_sha256 is not None:
                    raise RuntimeError("Online policy teacher cannot bind a checkpoint SHA")
                if policy_distill_teacher_checkpoint_dir is not None:
                    raise RuntimeError("Online policy teacher cannot use a checkpoint path")
                command.extend(("--policy-distill-teacher", "online"))
            else:
                if arm.policy_distill_teacher_state_sha256 is None:
                    raise RuntimeError("Checkpoint policy teacher has no state SHA")
                if policy_distill_teacher_checkpoint_dir is None:
                    raise RuntimeError("Checkpoint policy teacher has no checkpoint path")
                command.extend(
                    (
                        "--policy-distill-teacher",
                        "checkpoint",
                        "--policy-distill-teacher-checkpoint-dir",
                        str(policy_distill_teacher_checkpoint_dir),
                        "--policy-distill-teacher-state-sha256",
                        arm.policy_distill_teacher_state_sha256,
                    )
                )
    if arm.wdl_include_current_state is not None:
        command.append(
            "--wdl-include-current-state"
            if arm.wdl_include_current_state
            else "--no-wdl-include-current-state"
        )
    if arm.dfm_closed_loop_mode is not None:
        command.extend(
            (
                "--dfm-closed-loop-mode",
                arm.dfm_closed_loop_mode,
            )
        )
    return command


def _run_file_hashes(output_dir: Path) -> dict[str, str]:
    names = (
        "run_config.json",
        "optimizer_partition.json",
        "loss_summary.json",
        "report.json",
        "modal_run.json",
    )
    optional_names = (
        "metrics.jsonl",
        "validation_metrics.jsonl",
        "stitch_manifest.json",
    )
    retained = names + tuple(
        name for name in optional_names if (output_dir / name).is_file()
    )
    return {name: _sha256_file(output_dir / name) for name in retained}


_METRIC_SEGMENT_FILES = (
    "metrics.jsonl",
    "run_config.json",
    "optimizer_partition.json",
)


def _training_metric_segments(
    *,
    continuation_predecessor_root: Path | None,
    attempt_roots: tuple[Path, ...],
    output_dir: Path,
) -> tuple[tuple[str, Path], ...]:
    segments: list[tuple[str, Path]] = []
    if continuation_predecessor_root is not None:
        if not all(
            (continuation_predecessor_root / name).is_file()
            for name in _METRIC_SEGMENT_FILES
        ):
            raise RuntimeError(
                "Continuation predecessor is missing metric provenance"
            )
        segments.append(("predecessor", continuation_predecessor_root))
    segments.extend(
        (path.name, path)
        for path in attempt_roots
        if all((path / name).is_file() for name in _METRIC_SEGMENT_FILES)
    )
    if not all((output_dir / name).is_file() for name in _METRIC_SEGMENT_FILES):
        raise RuntimeError("Terminal training segment is incomplete")
    segments.append(("terminal", output_dir))
    names = [name for name, _ in segments]
    if len(names) != len(set(names)):
        raise RuntimeError("Training metric segment names are not unique")
    return tuple(segments)


def _recovery_checkpoint_candidate(
    checkpoint_dir: Path,
) -> tuple[int, Path] | None:
    manifest_path = checkpoint_dir / "manifest.json"
    if not manifest_path.is_file():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        update = int(manifest.get("optimizer_update", -1))
        state = manifest.get("state")
        state_size = int(state.get("size_bytes", -1)) if isinstance(state, dict) else -1
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return None
    if (
        manifest.get("format")
        not in {
            "chess-dfm-torch-training-v1",
            "chess-dfm-torch-training-v2",
        }
        or manifest.get("model_only") is not False
        or manifest.get("optimizer_resume_supported") is not True
        or not isinstance(state, dict)
    ):
        return None
    relative_state = state.get("path")
    if (
        not isinstance(relative_state, str)
        or Path(relative_state).name != relative_state
    ):
        return None
    state_path = checkpoint_dir / relative_state
    if (
        update < 0
        or checkpoint_dir.name != f"update{update:08d}"
        or not state_path.is_file()
        or state_path.stat().st_size != state_size
    ):
        return None
    return update, checkpoint_dir


def _latest_recovery_checkpoint(
    roots: tuple[Path, ...],
    before_update: int | None = None,
) -> tuple[int, Path] | None:
    candidates: list[tuple[int, Path]] = []
    for root in roots:
        checkpoint_root = root / "checkpoints"
        if not checkpoint_root.is_dir():
            continue
        for checkpoint_dir in checkpoint_root.glob("update*"):
            candidate = _recovery_checkpoint_candidate(checkpoint_dir)
            if candidate is not None and (
                before_update is None
                or candidate[0] < before_update
            ):
                candidates.append(candidate)
    if not candidates:
        return None
    return max(candidates, key=lambda candidate: candidate[0])


def _archive_partial_result(output_dir: Path) -> tuple[Path, Path]:
    attempt_root = output_dir.parent / ".attempts" / output_dir.name
    attempt_root.mkdir(parents=True, exist_ok=True)
    attempt_index = 0
    while True:
        archive_dir = attempt_root / f"attempt{attempt_index:03d}"
        if not archive_dir.exists():
            break
        attempt_index += 1
    output_dir.rename(archive_dir)
    return attempt_root, archive_dir


def _completed_run_summary(output_dir: Path) -> dict[str, Any]:
    report = json.loads((output_dir / "report.json").read_text(encoding="utf-8"))
    modal_run = json.loads((output_dir / "modal_run.json").read_text(encoding="utf-8"))
    loss_summary = json.loads(
        (output_dir / "loss_summary.json").read_text(encoding="utf-8")
    )
    stitched = (output_dir / "stitch_manifest.json").is_file()
    resumed = int(modal_run.get("resume_update", 0) or 0) > 0
    return {
        "schema_version": "chess-dfm-modal-training-result-v1",
        "result_label": output_dir.name,
        "profile": modal_run["profile"],
        "arm": modal_run["arm"],
        "gpu": modal_run["gpu"],
        "device_name": modal_run["device_name"],
        "source_tree_sha256": modal_run["source_tree_sha256"],
        "training_source_tree_sha256": modal_run["compile_source_tree_sha256"],
        "reused_immutable_result": True,
        "updates": report["updates"],
        "examples": report["examples"],
        "train_seconds": report["train_seconds"],
        "train_wall_seconds": report["train_wall_seconds"],
        "examples_per_second_end_to_end": report["examples_per_second_end_to_end"],
        "gpu_peak_memory_allocated_bytes": report["gpu_peak_memory_allocated_bytes"],
        "gpu_peak_memory_reserved_bytes": report["gpu_peak_memory_reserved_bytes"],
        "loss_summary": loss_summary,
        "training_curve_scope": (
            "stitched_full_run"
            if stitched
            else "terminal_segment_only" if resumed else "full_run"
        ),
        "timing_scope": (
            "terminal_segment_only"
            if stitched or resumed
            else "full_run"
        ),
        "file_sha256": _run_file_hashes(output_dir),
    }


def _run_training(
    *,
    profile_name: str,
    arm_name: str,
    gpu: str,
    expected_source_tree_sha256: str,
    expected_compile_source_tree_sha256: str,
    git_commit: str,
    declared_attempt_upper_bound_dollars: str,
) -> dict[str, Any]:
    import torch

    if profile_name not in TRAINING_PROFILES:
        raise ValueError(f"Unknown training profile {profile_name!r}")
    if arm_name not in TRAINING_ARMS:
        raise ValueError(f"Unknown training arm {arm_name!r}")
    arm = TRAINING_ARMS[arm_name]
    expected_source_tree_sha256 = _require_sha256(
        expected_source_tree_sha256,
        label="expected_source_tree_sha256",
    )
    expected_compile_source_tree_sha256 = _require_sha256(
        expected_compile_source_tree_sha256,
        label="expected_compile_source_tree_sha256",
    )
    git_commit = _require_git_commit(git_commit)
    if not torch.cuda.is_available():
        raise RuntimeError("Modal training function cannot see CUDA")
    device_name = str(torch.cuda.get_device_name(0))
    if gpu not in device_name:
        raise RuntimeError(f"Requested {gpu} but Modal exposed {device_name!r}")

    if (
        arm.data_format == "trajectory_v3"
        and arm.dataset_label == TRAINING_DATASET_LABEL
    ):
        stage = _validate_training_stage()
    elif (
        arm.data_format == "lc0_sequential"
        and arm.dataset_label == LC0_PROPOSAL_A_DATASET_LABEL
    ):
        stage = _validate_lc0_proposal_a_stage()
    else:
        raise RuntimeError("Training arm has an unsupported dataset contract")
    raw_bt4_path = _verify_raw_bt4_volume()
    source_tree_sha256 = _source_tree_sha256(Path("/root"))
    compile_source_tree_sha256 = _compile_source_tree_sha256(Path("/root"))
    if source_tree_sha256 != expected_source_tree_sha256:
        raise RuntimeError("Local and bundled training source identities differ")
    if compile_source_tree_sha256 != expected_compile_source_tree_sha256:
        raise RuntimeError("Local and bundled scientific source identities differ")
    policy_distill_teacher_checkpoint_dir = (
        _policy_distill_teacher_checkpoint_dir(
            arm_name=arm_name,
            result_mount=Path(TRAINING_RESULT_MOUNT),
        )
    )

    result_label = _run_identity(
        profile_name=profile_name,
        arm_name=arm_name,
        gpu=gpu,
        source_tree_sha256=compile_source_tree_sha256,
    )
    output_dir = Path(TRAINING_RESULT_MOUNT) / "runs" / result_label
    attempt_root = output_dir.parent / ".attempts" / output_dir.name
    if output_dir.exists():
        if (output_dir / "report.json").is_file() and (output_dir / "modal_run.json").is_file():
            return _completed_run_summary(output_dir)
        attempt_root, _ = _archive_partial_result(output_dir)
        training_result_volume.commit()
    attempt_roots = tuple(
        sorted(
            path
            for path in attempt_root.glob("attempt*")
            if path.is_dir()
        )
    )

    profile = TRAINING_PROFILES[profile_name]
    resume_result_label = None
    resume_checkpoint = None
    resume_update = 0
    continuation_predecessor_root: Path | None = None
    if profile.resume_from_profile is not None:
        predecessor_profile = TRAINING_PROFILES[profile.resume_from_profile]
        if predecessor_profile.steps >= profile.steps:
            raise RuntimeError("Continuation profile must extend its predecessor")
        resume_result_label = _run_identity(
            profile_name=profile.resume_from_profile,
            arm_name=arm_name,
            gpu=gpu,
            source_tree_sha256=compile_source_tree_sha256,
        )
        predecessor_root = (
            Path(TRAINING_RESULT_MOUNT) / "runs" / resume_result_label
        )
        continuation_predecessor_root = predecessor_root
        predecessor_run_path = predecessor_root / "modal_run.json"
        predecessor_report_path = predecessor_root / "report.json"
        if (
            not predecessor_run_path.is_file()
            or not predecessor_report_path.is_file()
        ):
            raise RuntimeError(
                "Continuation predecessor is incomplete or missing: "
                f"{resume_result_label}"
            )
        predecessor_run = json.loads(
            predecessor_run_path.read_text(encoding="utf-8")
        )
        if (
            predecessor_run.get("profile") != profile.resume_from_profile
            or predecessor_run.get("arm") != arm_name
            or predecessor_run.get("gpu") != gpu
            or predecessor_run.get("compile_source_tree_sha256")
            != compile_source_tree_sha256
        ):
            raise RuntimeError("Continuation predecessor provenance drift")
        resume_checkpoint = (
            predecessor_root
            / "checkpoints"
            / f"update{predecessor_profile.steps:08d}"
        )
        resume_manifest_path = resume_checkpoint / "manifest.json"
        if not resume_manifest_path.is_file():
            raise RuntimeError("Continuation predecessor has no recovery checkpoint")
        resume_manifest = json.loads(
            resume_manifest_path.read_text(encoding="utf-8")
        )
        if (
            resume_manifest.get("optimizer_resume_supported") is not True
            or int(resume_manifest.get("optimizer_update", -1))
            != predecessor_profile.steps
        ):
            raise RuntimeError("Continuation recovery checkpoint contract drift")
        resume_update = predecessor_profile.steps

    attempt_checkpoint = _latest_recovery_checkpoint(
        attempt_roots,
        before_update=profile.steps,
    )
    if attempt_checkpoint is not None and attempt_checkpoint[0] > resume_update:
        resume_update, resume_checkpoint = attempt_checkpoint
        resume_result_label = result_label

    dataset_root = _training_dataset_root(arm.dataset_label)
    data_root = dataset_root / (
        "trajectory_v3" if arm.data_format == "trajectory_v3" else "lc0_sequential"
    )
    eval_manifest = (
        _training_dataset_root(TRAINING_DATASET_LABEL)
        / "contracts"
        / "hero_epoch_v1"
        / "manifest.json"
    )
    # TorchInductor emits many small shared objects. Reusing those directly from a
    # network Volume was slower than a cold local compile for this model, so each
    # single-use GPU container gets a fresh cache on its ephemeral filesystem.
    cache_root = (
        Path(EPHEMERAL_COMPILE_CACHE_ROOT)
        / gpu.lower()
        / compile_source_tree_sha256
    )
    cache_root.mkdir(parents=True, exist_ok=True)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    command = _training_command(
        profile_name=profile_name,
        arm_name=arm_name,
        output_dir=output_dir,
        raw_bt4_path=raw_bt4_path,
        data_root=data_root,
        eval_manifest=eval_manifest,
        policy_distill_teacher_checkpoint_dir=(
            policy_distill_teacher_checkpoint_dir
        ),
        resume_checkpoint=resume_checkpoint,
        resume_update=resume_update,
    )
    trusted_workspace_roots = os.pathsep.join(
        str(Path(root).resolve(strict=True))
        for root in (
            BASE_INPUT_MOUNT,
            TRAINING_INPUT_MOUNT,
            TRAINING_RESULT_MOUNT,
        )
    )
    environment = os.environ.copy()
    environment.update(
        {
            "CHESS_DFM_WORKSPACE_ROOT": WORKSPACE_MOUNT,
            "CHESS_DFM_TRUSTED_WORKSPACE_ROOTS": trusted_workspace_roots,
            "CHESS_DFM_GIT_COMMIT": git_commit,
            "CHESS_DFM_SOURCE_TREE_SHA256": compile_source_tree_sha256,
            "TORCHINDUCTOR_CACHE_DIR": str(cache_root / "torchinductor"),
            "TRITON_CACHE_DIR": str(cache_root / "triton"),
            "PYTHONPYCACHEPREFIX": str(cache_root / "pycache"),
            "OMP_NUM_THREADS": "2",
            "OPENBLAS_NUM_THREADS": "2",
            "MKL_NUM_THREADS": "2",
            "NUMEXPR_MAX_THREADS": "2",
        }
    )
    started = time.perf_counter()
    stdout_path = Path("/tmp") / f"{result_label}.stdout.log"
    with stdout_path.open("x", encoding="utf-8") as stdout_log:
        process = subprocess.Popen(
            command,
            cwd="/root",
            env=environment,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            stdout_log.write(line)
            stdout_log.flush()
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            recovery_path = record.get("recovery_checkpoint_path")
            if isinstance(recovery_path, str):
                candidate = _recovery_checkpoint_candidate(Path(recovery_path))
                if (
                    candidate is not None
                    and candidate[1].parent == (output_dir / "checkpoints").resolve()
                ):
                    training_result_volume.commit()
                    print(
                        json.dumps(
                            {
                                "checkpoint_volume_commit": {
                                    "path": str(candidate[1]),
                                    "update": candidate[0],
                                }
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
        returncode = process.wait()
    if output_dir.exists():
        shutil.copyfile(stdout_path, output_dir / "modal_stdout.log")
    if returncode != 0:
        if output_dir.exists():
            (output_dir / "modal_failure.json").write_text(
                json.dumps(
                    {
                        "schema_version": "chess-dfm-modal-training-failure-v1",
                        "returncode": returncode,
                        "profile": profile_name,
                        "arm": arm_name,
                        "gpu": gpu,
                        "device_name": device_name,
                        "source_tree_sha256": source_tree_sha256,
                        "compile_source_tree_sha256": compile_source_tree_sha256,
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            training_result_volume.commit()
        raise RuntimeError(f"Hero training subprocess exited with status {returncode}")

    metric_segments = _training_metric_segments(
        continuation_predecessor_root=continuation_predecessor_root,
        attempt_roots=attempt_roots,
        output_dir=output_dir,
    )
    modal_run = {
        "schema_version": "chess-dfm-modal-training-run-v1",
        "runner_revision": RUNNER_REVISION,
        "result_label": result_label,
        "profile": profile_name,
        "profile_config": dataclasses.asdict(TRAINING_PROFILES[profile_name]),
        "arm": arm_name,
        "arm_config": dataclasses.asdict(TRAINING_ARMS[arm_name]),
        "gpu": gpu,
        "device_name": device_name,
        "device_total_memory_bytes": int(torch.cuda.get_device_properties(0).total_memory),
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "source_tree_sha256": source_tree_sha256,
        "compile_source_tree_sha256": compile_source_tree_sha256,
        "git_commit": git_commit,
        "dataset_stage": stage,
        "dataset": {
            "format": arm.data_format,
            "label": arm.dataset_label,
            "data_root": str(data_root),
        },
        "raw_bt4_sha256": RAW_BT4_SHA256,
        "declared_attempt_upper_bound_dollars": declared_attempt_upper_bound_dollars,
        "resumed_from_result_label": resume_result_label,
        "continuation_predecessor_result_label": (
            None
            if continuation_predecessor_root is None
            else continuation_predecessor_root.name
        ),
        "resume_checkpoint": None if resume_checkpoint is None else str(resume_checkpoint),
        "policy_distill_teacher_checkpoint": (
            None
            if policy_distill_teacher_checkpoint_dir is None
            else str(policy_distill_teacher_checkpoint_dir)
        ),
        "resume_update": resume_update,
        "archived_attempts": [str(path) for path in attempt_roots],
        "metric_segments": [str(path) for _, path in metric_segments],
        "remote_function_seconds": time.perf_counter() - started,
    }
    (output_dir / "modal_run.json").write_text(
        json.dumps(modal_run, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if len(metric_segments) > 1:
        from research.stitch_training_segments import stitch_training_segments

        stitch_training_segments(
            metric_segments,
            terminal_segment="terminal",
            output_dir=output_dir,
            continuation_predecessor_segment=(
                "predecessor"
                if continuation_predecessor_root is not None
                else None
            ),
        )
    training_result_volume.commit()
    summary = _completed_run_summary(output_dir)
    summary["reused_immutable_result"] = False
    return summary



def _require_result_label(value: str) -> str:
    safe_characters = all(
        character.islower() or character.isdigit() or character in "-_"
        for character in value
    )
    if (
        not value.startswith("hero-training-")
        or not 1 <= len(value) <= 200
        or Path(value).name != value
        or not safe_characters
    ):
        raise ValueError("Unsafe Hero training result label")
    return value


def _policy_distill_teacher_checkpoint_dir(
    *,
    arm_name: str,
    result_mount: Path,
) -> Path | None:
    """Resolve and validate an immutable Phase-1 policy teacher."""

    arm = TRAINING_ARMS[arm_name]
    result_label = arm.policy_distill_teacher_result_label
    expected_state_sha256 = arm.policy_distill_teacher_state_sha256
    if result_label is None and expected_state_sha256 is None:
        return None
    if (
        result_label is None
        or expected_state_sha256 is None
        or arm.policy_distill_coeff is None
        or arm.policy_distill_coeff <= 0.0
    ):
        raise RuntimeError("Checkpoint policy teacher arm is incomplete")
    result_label = _require_result_label(result_label)
    expected_state_sha256 = _require_sha256(
        expected_state_sha256,
        label="policy_distill_teacher_state_sha256",
    )
    result_dir = result_mount / "runs" / result_label
    checkpoint_dir = result_dir / "checkpoint"
    paths = {
        "modal_run": result_dir / "modal_run.json",
        "report": result_dir / "report.json",
        "run_config": result_dir / "run_config.json",
        "manifest": checkpoint_dir / "manifest.json",
    }
    if any(not path.is_file() for path in paths.values()):
        raise RuntimeError("Policy teacher requires a completed immutable training result")
    modal_run = json.loads(paths["modal_run"].read_text(encoding="utf-8"))
    report = json.loads(paths["report"].read_text(encoding="utf-8"))
    run_config = json.loads(paths["run_config"].read_text(encoding="utf-8"))
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    state = manifest.get("state", {})
    if (
        modal_run.get("result_label") != result_label
        or modal_run.get("profile") != "phase1"
        or modal_run.get("arm") != "hero1_exact"
        or modal_run.get("gpu") != "L40S"
        or int(report.get("updates", -1)) != TRAINING_PROFILES["phase1"].steps
        or run_config.get("framework") != "torch"
        or not isinstance(run_config.get("config"), dict)
        or manifest.get("format") != "chess-dfm-torch-model-v1"
        or manifest.get("model_only") is not True
        or int(manifest.get("optimizer_update", -1))
        != TRAINING_PROFILES["phase1"].steps
        or state.get("sha256") != expected_state_sha256
    ):
        raise RuntimeError("Policy teacher immutable result contract drift")
    state_relative = state.get("path")
    if not isinstance(state_relative, str):
        raise RuntimeError("Policy teacher checkpoint has no state path")
    logical_state = PurePosixPath(state_relative)
    if logical_state.is_absolute() or ".." in logical_state.parts:
        raise RuntimeError("Policy teacher checkpoint state path is unsafe")
    state_path = checkpoint_dir.joinpath(*logical_state.parts)
    if (
        not state_path.is_file()
        or state_path.stat().st_size != int(state.get("size_bytes", -1))
    ):
        raise RuntimeError("Policy teacher checkpoint state inventory drift")
    return checkpoint_dir


def _parameter_pair_movement(
    initial_parameter: Any,
    trained_parameter: Any,
) -> dict[str, Any]:
    import torch

    if (
        tuple(initial_parameter.shape) != tuple(trained_parameter.shape)
        or initial_parameter.dtype != trained_parameter.dtype
    ):
        raise ValueError("Movement parameter ABI mismatch")
    initial = initial_parameter.detach().cpu().float()
    trained = trained_parameter.detach().cpu().float()
    delta = trained - initial
    initial_l2 = float(
        torch.sqrt(torch.sum(initial.square(), dtype=torch.float64)).item()
    )
    trained_l2 = float(
        torch.sqrt(torch.sum(trained.square(), dtype=torch.float64)).item()
    )
    delta_l2 = float(
        torch.sqrt(torch.sum(delta.square(), dtype=torch.float64)).item()
    )
    denominator = initial_l2 * trained_l2
    count = int(initial.numel())
    unchanged_count = int(torch.count_nonzero(initial == trained).item())
    return {
        "parameter_count": count,
        "initial_l2": initial_l2,
        "trained_l2": trained_l2,
        "delta_l2": delta_l2,
        "relative_delta_l2": (
            delta_l2 / initial_l2 if initial_l2 > 0.0 else None
        ),
        "cosine": (
            float(
                torch.sum(initial * trained, dtype=torch.float64).item()
            )
            / denominator
            if denominator > 0.0
            else None
        ),
        "delta_rms": delta_l2 / (count**0.5),
        "delta_absolute_max": float(delta.abs().max().item()),
        "unchanged_count": unchanged_count,
        "changed_fraction": 1.0 - unchanged_count / count,
    }


def _closed_loop_adapter_movement(
    *,
    checkpoint_dir: Path,
    model_config: Any,
    expected_state_sha256: str,
) -> dict[str, Any]:
    import gc

    from research.train_torch import (
        JointModel,
        initialize_fresh_modules,
        load_model_checkpoint,
    )

    mode = str(getattr(model_config, "dfm_closed_loop_mode", "none"))
    if mode != "predicted_jepa_tokens":
        return {
            "schema_version": "chess-dfm-closed-loop-adapter-movement-v1",
            "status": "absent_by_config",
            "dfm_closed_loop_mode": mode,
            "parameters": {},
        }

    initial_model = JointModel(model_config)
    initialization = initialize_fresh_modules(
        initial_model,
        seed=int(model_config.init_seed),
    )
    initial_named = dict(initial_model.named_parameters())
    selected_paths = sorted(
        name
        for name in initial_named
        if name.startswith("dfm_jepa_rollout_adapter.")
    )
    if selected_paths != ["dfm_jepa_rollout_adapter.w"]:
        raise RuntimeError(
            "Closed-loop adapter parameter inventory drift: "
            f"{selected_paths}"
        )
    initial_selected = {
        name: initial_named[name].detach().cpu().clone()
        for name in selected_paths
    }
    del initial_named, initial_model
    gc.collect()

    trained_model = JointModel(model_config)
    trained_manifest = load_model_checkpoint(
        checkpoint_dir=checkpoint_dir,
        model=trained_model,
    )
    observed_state_sha256 = str(
        trained_manifest.get("state", {}).get("sha256", "")
    )
    if observed_state_sha256 != expected_state_sha256:
        raise RuntimeError("Adapter audit loaded a different checkpoint state")
    trained_named = dict(trained_model.named_parameters())
    metrics = {
        name: _parameter_pair_movement(
            initial_selected[name],
            trained_named[name],
        )
        for name in selected_paths
    }
    del trained_named, trained_model, initial_selected
    gc.collect()
    return {
        "schema_version": "chess-dfm-closed-loop-adapter-movement-v1",
        "status": "measured",
        "dfm_closed_loop_mode": mode,
        "initialization_schema_version": initialization["schema_version"],
        "initialization_seed": initialization["seed"],
        "parameters": metrics,
    }


def _completed_movement_summary(output_dir: Path) -> dict[str, Any]:
    report_path = output_dir / "parameter_diff.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    remote = report["remote_audit"]
    return {
        "schema_version": "chess-dfm-modal-movement-result-v2",
        "audit_label": output_dir.name,
        "result_label": remote["result_label"],
        "movement_source_tree_sha256": remote["movement_source_tree_sha256"],
        "checkpoint_state_sha256": remote["checkpoint_state_sha256"],
        "group_metrics": {
            name: report["group_metrics"][name]
            for name in ("all", "trunk", "embedding", "layers", "policy_head")
        },
        "closed_loop_adapter_movement": report[
            "closed_loop_adapter_movement"
        ],
        "parameter_diff_sha256": _sha256_file(report_path),
        "reused_immutable_result": True,
    }


def _run_movement_audit(
    *,
    result_label: str,
    expected_movement_source_tree_sha256: str,
    git_commit: str,
    declared_attempt_upper_bound_dollars: str,
) -> dict[str, Any]:
    import torch

    result_label = _require_result_label(result_label)
    expected_movement_source_tree_sha256 = _require_sha256(
        expected_movement_source_tree_sha256,
        label="expected_movement_source_tree_sha256",
    )
    git_commit = _require_git_commit(git_commit)
    movement_source_tree_sha256 = _movement_source_tree_sha256(Path("/root"))
    if movement_source_tree_sha256 != expected_movement_source_tree_sha256:
        raise RuntimeError("Local and bundled movement-audit source identities differ")

    raw_bt4_path = _verify_raw_bt4_volume()
    result_dir = Path(TRAINING_RESULT_MOUNT) / "runs" / result_label
    modal_run_path = result_dir / "modal_run.json"
    train_report_path = result_dir / "report.json"
    run_config_path = result_dir / "run_config.json"
    checkpoint_dir = result_dir / "checkpoint"
    checkpoint_manifest_path = checkpoint_dir / "manifest.json"
    required_paths = (
        modal_run_path,
        train_report_path,
        run_config_path,
        checkpoint_manifest_path,
    )
    if any(not path.is_file() for path in required_paths):
        raise RuntimeError(
            "Movement audit requires a completed immutable training result"
        )
    modal_run = json.loads(modal_run_path.read_text(encoding="utf-8"))
    train_report = json.loads(train_report_path.read_text(encoding="utf-8"))
    run_config = json.loads(run_config_path.read_text(encoding="utf-8"))
    checkpoint_manifest = json.loads(
        checkpoint_manifest_path.read_text(encoding="utf-8")
    )
    profile_name = str(modal_run.get("profile", ""))
    if profile_name not in {"phase1", "u1024"}:
        raise RuntimeError(
            f"Unsupported movement-audit training profile: {profile_name!r}"
        )
    expected_updates = TRAINING_PROFILES[profile_name].steps
    model_config_payload = run_config.get("config")
    if not isinstance(model_config_payload, dict):
        raise RuntimeError("Movement-audit run has no recorded model config")
    model_config_sha256 = hashlib.sha256(
        _canonical_json_bytes(model_config_payload)
    ).hexdigest()
    if (
        modal_run.get("result_label") != result_label
        or int(train_report.get("updates", -1)) != expected_updates
        or checkpoint_manifest.get("format")
        != "chess-dfm-torch-model-v1"
        or checkpoint_manifest.get("model_only") is not True
        or int(checkpoint_manifest.get("optimizer_update", -1))
        != expected_updates
    ):
        raise RuntimeError("Movement-audit training provenance drift")
    checkpoint_state_sha256 = _require_sha256(
        str(checkpoint_manifest.get("state", {}).get("sha256", "")),
        label="checkpoint_state_sha256",
    )
    audit_identity = hashlib.sha256(
        _canonical_json_bytes(
            {
                "schema_version": "chess-dfm-modal-movement-audit-v2",
                "result_label": result_label,
                "training_profile": profile_name,
                "model_config_sha256": model_config_sha256,
                "checkpoint_state_sha256": checkpoint_state_sha256,
                "movement_source_tree_sha256": movement_source_tree_sha256,
                "raw_bt4_sha256": RAW_BT4_SHA256,
            }
        )
    ).hexdigest()
    output_dir = (
        Path(TRAINING_RESULT_MOUNT)
        / "audits"
        / f"{result_label}-movement-{audit_identity[:20]}"
    )
    if output_dir.exists():
        if (output_dir / "parameter_diff.json").is_file():
            return _completed_movement_summary(output_dir)
        _archive_partial_result(output_dir)
        training_result_volume.commit()

    trusted_workspace_roots = os.pathsep.join(
        str(Path(root).resolve(strict=True))
        for root in (BASE_INPUT_MOUNT, TRAINING_RESULT_MOUNT)
    )
    os.environ.update(
        {
            "CHESS_DFM_WORKSPACE_ROOT": WORKSPACE_MOUNT,
            "CHESS_DFM_TRUSTED_WORKSPACE_ROOTS": trusted_workspace_roots,
        }
    )
    from research.interpretability.models import load_bt4_comparison_models
    from research.interpretability.parameter_diff import parameter_diff
    from research.train_torch import Config

    model_config = Config(**model_config_payload)
    if dataclasses.asdict(model_config) != model_config_payload:
        raise RuntimeError("Recorded movement-audit model config drift")
    torch.set_num_threads(4)
    started = time.perf_counter()
    models = load_bt4_comparison_models(
        raw_bt4_path=raw_bt4_path,
        hero_checkpoint_dir=checkpoint_dir,
        device=torch.device("cpu"),
        model_config=model_config,
    )
    report = parameter_diff(models)
    observed_checkpoint_sha256 = str(report["models"]["hero_state_sha256"])
    if observed_checkpoint_sha256 != checkpoint_state_sha256:
        raise RuntimeError("Movement audit loaded a different checkpoint state")
    del models
    report["closed_loop_adapter_movement"] = (
        _closed_loop_adapter_movement(
            checkpoint_dir=checkpoint_dir,
            model_config=model_config,
            expected_state_sha256=checkpoint_state_sha256,
        )
    )
    report["remote_audit"] = {
        "schema_version": "chess-dfm-modal-movement-audit-v2",
        "runner_revision": RUNNER_REVISION,
        "result_label": result_label,
        "training_profile": profile_name,
        "training_updates": expected_updates,
        "model_config_sha256": model_config_sha256,
        "training_arm": modal_run["arm"],
        "training_compile_source_tree_sha256": modal_run[
            "compile_source_tree_sha256"
        ],
        "checkpoint_state_sha256": checkpoint_state_sha256,
        "movement_source_tree_sha256": movement_source_tree_sha256,
        "git_commit": git_commit,
        "declared_attempt_upper_bound_dollars": declared_attempt_upper_bound_dollars,
        "remote_function_seconds": time.perf_counter() - started,
        "device": "cpu",
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    report_path = output_dir / "parameter_diff.json"
    temporary = output_dir / ".parameter_diff.json.partial"
    temporary.write_text(
        json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, report_path)
    training_result_volume.commit()
    summary = _completed_movement_summary(output_dir)
    summary["reused_immutable_result"] = False
    return summary


_TRAINING_VOLUMES = {
    BASE_INPUT_MOUNT: base_input_volume.with_mount_options(read_only=True),
    TRAINING_INPUT_MOUNT: training_input_volume.with_mount_options(read_only=True),
    TRAINING_RESULT_MOUNT: training_result_volume,
}
_MOVEMENT_VOLUMES = {
    BASE_INPUT_MOUNT: base_input_volume.with_mount_options(read_only=True),
    TRAINING_RESULT_MOUNT: training_result_volume,
}


@app.function(
    image=gpu_image,
    volumes=_MOVEMENT_VOLUMES,
    cpu=4.0,
    memory=16384,
    timeout=1800,
    startup_timeout=300,
    retries=0,
    max_containers=1,
    single_use_containers=True,
)
def audit_training_movement(**kwargs: str) -> dict[str, Any]:
    return _run_movement_audit(**kwargs)


@app.function(
    image=gpu_image,
    volumes=_TRAINING_VOLUMES,
    gpu="L40S",
    cpu=4.0,
    memory=32768,
    timeout=1200,
    startup_timeout=900,
    retries=0,
    max_containers=1,
    single_use_containers=True,
)
def smoke_l40s(**kwargs: str) -> dict[str, Any]:
    return _run_training(gpu="L40S", **kwargs)


@app.function(
    image=gpu_image,
    volumes=_TRAINING_VOLUMES,
    gpu="A100-40GB",
    cpu=4.0,
    memory=32768,
    timeout=1200,
    startup_timeout=900,
    retries=0,
    max_containers=1,
    single_use_containers=True,
)
def smoke_a100(**kwargs: str) -> dict[str, Any]:
    return _run_training(gpu="A100", **kwargs)


@app.function(
    image=gpu_image,
    volumes=_TRAINING_VOLUMES,
    gpu="L40S",
    cpu=4.0,
    memory=32768,
    timeout=1500,
    startup_timeout=600,
    retries=0,
    max_containers=1,
    single_use_containers=True,
)
def proposal_screen_l40s(**kwargs: str) -> dict[str, Any]:
    return _run_training(gpu="L40S", **kwargs)


@app.function(
    image=gpu_image,
    volumes=_TRAINING_VOLUMES,
    gpu="A100-40GB",
    cpu=4.0,
    memory=32768,
    timeout=1500,
    startup_timeout=600,
    retries=0,
    max_containers=1,
    single_use_containers=True,
)
def proposal_screen_a100(**kwargs: str) -> dict[str, Any]:
    return _run_training(gpu="A100", **kwargs)


@app.function(
    image=gpu_image,
    volumes=_TRAINING_VOLUMES,
    gpu="L40S",
    cpu=4.0,
    memory=32768,
    timeout=5400,
    startup_timeout=900,
    retries=0,
    max_containers=1,
    single_use_containers=True,
)
def phase1_l40s(**kwargs: str) -> dict[str, Any]:
    return _run_training(gpu="L40S", **kwargs)


@app.function(
    image=gpu_image,
    volumes=_TRAINING_VOLUMES,
    gpu="A100-40GB",
    cpu=4.0,
    memory=32768,
    timeout=5400,
    startup_timeout=900,
    retries=0,
    max_containers=1,
    single_use_containers=True,
)
def phase1_a100(**kwargs: str) -> dict[str, Any]:
    return _run_training(gpu="A100", **kwargs)


@app.function(
    image=gpu_image,
    volumes=_TRAINING_VOLUMES,
    gpu="L40S",
    cpu=4.0,
    memory=32768,
    timeout=9000,
    startup_timeout=900,
    retries=0,
    max_containers=1,
    single_use_containers=True,
)
def u1024_l40s(**kwargs: str) -> dict[str, Any]:
    return _run_training(gpu="L40S", **kwargs)


@app.function(
    image=gpu_image,
    volumes=_TRAINING_VOLUMES,
    gpu="A100-40GB",
    cpu=4.0,
    memory=32768,
    timeout=9000,
    startup_timeout=900,
    retries=0,
    max_containers=1,
    single_use_containers=True,
)
def u1024_a100(**kwargs: str) -> dict[str, Any]:
    return _run_training(gpu="A100", **kwargs)


@app.function(
    image=gpu_image,
    volumes=_TRAINING_VOLUMES,
    gpu="L40S",
    cpu=4.0,
    memory=32768,
    timeout=14400,
    startup_timeout=900,
    retries=0,
    max_containers=1,
    single_use_containers=True,
)
def phase2_l40s(**kwargs: str) -> dict[str, Any]:
    return _run_training(gpu="L40S", **kwargs)


@app.function(
    image=gpu_image,
    volumes=_TRAINING_VOLUMES,
    gpu="A100-40GB",
    cpu=4.0,
    memory=32768,
    timeout=14400,
    startup_timeout=900,
    retries=0,
    max_containers=1,
    single_use_containers=True,
)
def phase2_a100(**kwargs: str) -> dict[str, Any]:
    return _run_training(gpu="A100", **kwargs)


def _spec(stage: str) -> tuple[Any, Decimal]:
    from research.interpretability.remote_cost import ModalJobSpec

    if stage == "stage-data":
        return (
            ModalJobSpec(
                name="cpu-stage-public-trajectory-v3",
                timeout_seconds=7200,
                startup_timeout_seconds=300,
                retries=0,
                cpu_cores=Decimal("2"),
                memory_gib=Decimal("4"),
            ),
            Decimal("0.30"),
        )
    if stage == "movement":
        return (
            ModalJobSpec(
                name="cpu-phase1-parameter-movement",
                timeout_seconds=1800,
                startup_timeout_seconds=300,
                retries=0,
                cpu_cores=Decimal("4"),
                memory_gib=Decimal("16"),
            ),
            Decimal("0.30"),
        )
    resources = {
        "smoke-l40s": (1200, 900, "L40S", Decimal("1.40")),
        "smoke-a100": (1200, 900, "A100-40GB", Decimal("1.50")),
        "proposal_screen-l40s": (1500, 600, "L40S", Decimal("1.40")),
        "proposal_screen-a100": (1500, 600, "A100-40GB", Decimal("1.49")),
        "phase1-l40s": (5400, 900, "L40S", Decimal("4.20")),
        "phase1-a100": (5400, 900, "A100-40GB", Decimal("4.50")),
        "u1024-l40s": (9000, 900, "L40S", Decimal("7.00")),
        "u1024-a100": (9000, 900, "A100-40GB", Decimal("7.35")),
        "phase2-l40s": (14400, 900, "L40S", Decimal("10.20")),
        "phase2-a100": (14400, 900, "A100-40GB", Decimal("10.85")),
    }
    if stage not in resources:
        raise ValueError(f"Unknown training cost stage {stage!r}")
    timeout, startup_timeout, gpu, cap = resources[stage]
    return (
        ModalJobSpec(
            name=f"{stage}-hero-training",
            timeout_seconds=timeout,
            startup_timeout_seconds=startup_timeout,
            retries=0,
            cpu_cores=Decimal("4"),
            memory_gib=Decimal("32"),
            gpu=gpu,
        ),
        cap,
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
        raise RuntimeError("Cannot resolve local Git commit for training provenance")
    return _require_git_commit(completed.stdout.strip())


@app.local_entrypoint()
def main(
    mode: str = "stage-data",
    arm: str = "v2_fp32_1_10",
    gpu: str = "L40S",
    result_label: str = "",
    recorded_spend: str = "",
    per_job_cap: str = "",
) -> None:
    from research.interpretability.remote_cost import (
        enforce_modal_budget,
        estimate_modal_job,
        modal_month_spend,
    )

    if mode == "stage-data":
        stage = "stage-data"
        spec, default_cap = _spec(stage)
        estimate = estimate_modal_job(spec)
        spend = max(
            modal_month_spend(),
            Decimal(recorded_spend) if recorded_spend else Decimal("0"),
        )
        decision = enforce_modal_budget(
            estimate,
            recorded_month_spend=spend,
            per_job_cap=Decimal(per_job_cap) if per_job_cap else default_cap,
        )
        print(json.dumps({"launch_stage": stage, "cost_guard": decision}, sort_keys=True))
        call = stage_trajectory_dataset.spawn()
        print(
            json.dumps(
                {
                    "stage_submission": {
                        "function_call_id": call.object_id,
                        "mode": "asynchronous_spawn",
                    }
                },
                sort_keys=True,
            )
        )
        return

    if mode == "movement":
        result_label = _require_result_label(result_label)
        spec, default_cap = _spec("movement")
        estimate = estimate_modal_job(spec)
        spend = max(
            modal_month_spend(),
            Decimal(recorded_spend) if recorded_spend else Decimal("0"),
        )
        decision = enforce_modal_budget(
            estimate,
            recorded_month_spend=spend,
            per_job_cap=Decimal(per_job_cap) if per_job_cap else default_cap,
        )
        repo_root = Path(__file__).resolve().parents[2]
        movement_source_tree_sha256 = _movement_source_tree_sha256(repo_root)
        git_commit = _local_git_commit(repo_root)
        print(
            json.dumps(
                {"launch_stage": "movement", "cost_guard": decision},
                sort_keys=True,
            )
        )
        call = audit_training_movement.spawn(
            result_label=result_label,
            expected_movement_source_tree_sha256=movement_source_tree_sha256,
            git_commit=git_commit,
            declared_attempt_upper_bound_dollars=decision[
                "declared_attempt_upper_bound"
            ],
        )
        print(
            json.dumps(
                {
                    "movement_submission": {
                        "function_call_id": call.object_id,
                        "mode": "asynchronous_spawn",
                    }
                },
                sort_keys=True,
            )
        )
        return

    if mode not in TRAINING_PROFILES:
        raise ValueError(f"Mode must be stage-data or one of {sorted(TRAINING_PROFILES)}")
    if arm not in TRAINING_ARMS:
        raise ValueError(f"Unknown arm {arm!r}; choose from {sorted(TRAINING_ARMS)}")
    if gpu not in {"L40S", "A100"}:
        raise ValueError("GPU must be L40S or A100")

    stage_profile = "smoke" if mode in {"smoke", "benchmark", "diagnostic"} else mode
    stage = f"{stage_profile}-{gpu.lower()}"
    spec, default_cap = _spec(stage)
    estimate = estimate_modal_job(spec)
    spend = max(
        modal_month_spend(),
        Decimal(recorded_spend) if recorded_spend else Decimal("0"),
    )
    decision = enforce_modal_budget(
        estimate,
        recorded_month_spend=spend,
        per_job_cap=Decimal(per_job_cap) if per_job_cap else default_cap,
    )
    repo_root = Path(__file__).resolve().parents[2]
    source_tree_sha256 = _source_tree_sha256(repo_root)
    compile_source_tree_sha256 = _compile_source_tree_sha256(repo_root)
    git_commit = _local_git_commit(repo_root)
    print(json.dumps({"launch_stage": stage, "cost_guard": decision}, sort_keys=True))
    functions = {
        ("smoke", "L40S"): smoke_l40s,
        ("smoke", "A100"): smoke_a100,
        ("proposal_screen", "L40S"): proposal_screen_l40s,
        ("proposal_screen", "A100"): proposal_screen_a100,
        ("phase1", "L40S"): phase1_l40s,
        ("phase1", "A100"): phase1_a100,
        ("u1024", "L40S"): u1024_l40s,
        ("u1024", "A100"): u1024_a100,
        ("phase2", "L40S"): phase2_l40s,
        ("phase2", "A100"): phase2_a100,
    }
    function = functions[(stage_profile, gpu)]
    call = function.spawn(
        profile_name=mode,
        arm_name=arm,
        expected_source_tree_sha256=source_tree_sha256,
        expected_compile_source_tree_sha256=compile_source_tree_sha256,
        git_commit=git_commit,
        declared_attempt_upper_bound_dollars=decision[
            "declared_attempt_upper_bound"
        ],
    )
    print(
        json.dumps(
            {
                "training_submission": {
                    "function_call_id": call.object_id,
                    "mode": "asynchronous_spawn",
                }
            },
            sort_keys=True,
        )
    )
