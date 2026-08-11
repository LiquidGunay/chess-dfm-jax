"""Cost-gated Modal pipelines for cached, reproducible chess-DFM experiments."""

from __future__ import annotations

import hashlib
import json
import os
from argparse import Namespace
from decimal import Decimal
from pathlib import Path, PurePosixPath
from typing import Any

import modal


APP_NAME = "chess-dfm-interpretability"
INPUT_BUNDLE_SHA256 = "8be17957ad779e3756092306daa95ad9c1c7f5d482b6409bda7a16bc6d6c7a44"
OBJECT_PREFIX = "chess-dfm/v1/bundles/sha256"
RAILWAY_SECRET_NAME = "chess-dfm-railway-s3"
INPUT_VOLUME_NAME = "chess-dfm-interpretability-inputs"
RESULT_VOLUME_NAME = "chess-dfm-interpretability-results"
INPUT_VOLUME_MOUNT = "/vol/inputs"
RESULT_VOLUME_MOUNT = "/vol/results"
PROBE_RUNNER_REVISION = "probe-modal-v4"
LORSA_RUNNER_REVISION = "lorsa-transfer-modal-v2"
DEFAULT_LORSA_MANIFEST_RELATIVE = "converted/manifest.json"
DEFAULT_LORSA_WEIGHTS_RELATIVE = "converted/lorsa.safetensors"


app = modal.App(APP_NAME)
railway_secret = modal.Secret.from_name(RAILWAY_SECRET_NAME)
input_volume = modal.Volume.from_name(INPUT_VOLUME_NAME, create_if_missing=True)
result_volume = modal.Volume.from_name(RESULT_VOLUME_NAME, create_if_missing=True)

control_image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(
        "boto3==1.40.0",
        "botocore==1.40.76",
        "jmespath==1.1.0",
        "python-dateutil==2.9.0.post0",
        "s3transfer==0.13.1",
        "six==1.17.0",
        "urllib3==2.7.0",
    )
    .add_local_python_source("research.interpretability", copy=True)
)

gpu_image = (
    modal.Image.from_registry("pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime")
    .uv_pip_install(
        "boto3==1.40.0",
        "botocore==1.40.76",
        "jmespath==1.1.0",
        "ml-dtypes==0.5.3",
        "numpy==2.2.6",
        "protobuf==6.31.1",
        "python-chess==1.999",
        "python-dateutil==2.9.0.post0",
        "s3transfer==0.13.1",
        "safetensors==0.5.3",
        "six==1.17.0",
        "urllib3==2.7.0",
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


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _require_bundle_sha256(value: str) -> str:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError("bundle_sha256 must be a lowercase SHA256 digest")
    return value


def _require_result_label(value: str) -> str:
    if not value or len(value) > 160:
        raise ValueError("Result label must contain 1-160 characters")
    if any(not (character.isalnum() or character in "-_.") for character in value):
        raise ValueError("Result label contains unsafe characters")
    return value



def _bundle_file(bundle_root: Path, relative: str) -> Path:
    logical = PurePosixPath(relative)
    if logical.is_absolute() or any(part in {"", ".", ".."} for part in logical.parts):
        raise ValueError(f"Unsafe bundle-relative path: {relative!r}")
    root = bundle_root.resolve(strict=True)
    target = (root / logical.as_posix()).resolve(strict=True)
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Bundle-relative path escapes bundle: {relative!r}") from exc
    if not target.is_file():
        raise ValueError(f"Bundle-relative artifact is not a file: {relative!r}")
    return target

def _validate_bundle_manifest(bundle_sha256: str, manifest: dict[str, Any]) -> None:
    _require_bundle_sha256(bundle_sha256)
    if manifest.get("bundle_sha256") != bundle_sha256:
        raise RuntimeError("Bundle manifest identity mismatch")
    identity = {key: manifest[key] for key in ("schema_version", "kind", "files", "metadata")}
    observed = hashlib.sha256(_canonical_json_bytes(identity)).hexdigest()
    if observed != bundle_sha256:
        raise RuntimeError("Bundle canonical identity mismatch")


def _verify_staged_bundle(
    bundle_sha256: str,
    target: Path,
    marker: Path,
    *,
    hash_files: bool,
) -> dict[str, Any]:
    from research.interpretability.artifacts import sha256_file

    if not target.is_dir() or not marker.is_file():
        raise RuntimeError("Staged bundle directory or verification marker is missing")
    marker_payload = json.loads(marker.read_text(encoding="utf-8"))
    if marker_payload.get("schema_version") != "chess-dfm-modal-volume-stage-v1":
        raise RuntimeError("Unknown Modal Volume stage marker")
    manifest = marker_payload["bundle"]
    _validate_bundle_manifest(bundle_sha256, manifest)
    expected_names = set(manifest["files"])
    actual_names = {
        path.relative_to(target).as_posix() for path in target.rglob("*") if path.is_file()
    }
    if actual_names != expected_names:
        raise RuntimeError("Staged bundle inventory differs from its immutable manifest")
    verified_bytes = 0
    for relative, record in manifest["files"].items():
        logical = PurePosixPath(relative)
        if logical.is_absolute() or any(part in {"", ".", ".."} for part in logical.parts):
            raise RuntimeError(f"Unsafe staged bundle path: {relative!r}")
        path = target / logical.as_posix()
        if path.stat().st_size != int(record["size_bytes"]):
            raise RuntimeError(f"Staged bundle size mismatch: {relative}")
        if hash_files and sha256_file(path) != record["sha256"]:
            raise RuntimeError(f"Staged bundle checksum mismatch: {relative}")
        verified_bytes += int(record["size_bytes"])
    return {
        "bundle_sha256": bundle_sha256,
        "kind": manifest["kind"],
        "verified_files": len(manifest["files"]),
        "verified_bytes": verified_bytes,
        "full_file_hashes_verified": hash_files,
    }


def _verify_volume_workspace_guard(target: Path) -> bool:
    workspace = Path(INPUT_VOLUME_MOUNT).resolve(strict=True)
    raw_model = (target / "models/raw/BT4_exported.pb.gz").resolve(strict=True)
    raw_model.relative_to(workspace)
    return True


def _probe_config(profile: str) -> dict[str, Any]:
    common: dict[str, Any] = {
        "profile": profile,
        "future_ply": 2,
        "l2_values": [0.0, 1e-5],
        "probe_learning_rate": 0.03,
        "probe_batch_size": 4096,
        "lookahead_epochs": 5,
        "causal_dose": 1.0,
        "seed": 20260808,
        "cpu_threads": 4,
    }
    if profile == "smoke":
        return {
            **common,
            "layers": [14],
            "fit_count": 16,
            "selection_count": 8,
            "evaluation_count": 8,
            "capture_batch_size": 2,
            "probe_epochs": 2,
            "causal_count": 2,
            "bootstrap_samples": 100,
        }
    if profile == "all-layers":
        return {
            **common,
            "layers": list(range(15)),
            "fit_count": 1024,
            "selection_count": 256,
            "evaluation_count": 512,
            "capture_batch_size": 8,
            "probe_epochs": 20,
            "causal_count": 16,
            "bootstrap_samples": 2000,
        }
    raise ValueError(f"Unknown probe profile {profile!r}")


def _probe_result_label(bundle_sha256: str, profile: str) -> str:
    config = _probe_config(profile)
    identity = hashlib.sha256(
        _canonical_json_bytes(
            {
                "schema_version": PROBE_RUNNER_REVISION,
                "input_bundle_sha256": _require_bundle_sha256(bundle_sha256),
                "config": config,
            }
        )
    ).hexdigest()
    return f"raw-hero-probe-{profile}-{identity[:20]}"



def _lorsa_config(profile: str) -> dict[str, Any]:
    common: dict[str, Any] = {
        "profile": profile,
        "layer": 14,
        "batch_size": 1,
        "feature_summary_count": 16,
        "seed": 20260808,
        "cpu_threads": 4,
    }
    if profile == "smoke":
        return {**common, "position_count": 8}
    if profile == "development":
        return {**common, "position_count": 64}
    raise ValueError(f"Unknown LoRSA profile {profile!r}")


def _lorsa_result_label(
    base_bundle_sha256: str,
    lorsa_bundle_sha256: str,
    *,
    profile: str,
    gpu: str,
    manifest_relative: str,
    weights_relative: str,
) -> str:
    if gpu not in {"T4", "L4"}:
        raise ValueError("LoRSA benchmark GPU must be T4 or L4")
    config = _lorsa_config(profile)
    identity = hashlib.sha256(
        _canonical_json_bytes(
            {
                "schema_version": LORSA_RUNNER_REVISION,
                "base_bundle_sha256": _require_bundle_sha256(base_bundle_sha256),
                "lorsa_bundle_sha256": _require_bundle_sha256(lorsa_bundle_sha256),
                "manifest_relative": manifest_relative,
                "weights_relative": weights_relative,
                "gpu": gpu,
                "config": config,
            }
        )
    ).hexdigest()
    return f"raw-hero-lorsa-{profile}-{gpu.lower()}-{identity[:20]}"

@app.function(
    image=control_image,
    secrets=[railway_secret],
    cpu=0.25,
    memory=512,
    timeout=60,
    startup_timeout=120,
    retries=1,
    max_containers=1,
    single_use_containers=True,
)
def railway_control_plane_smoke(bundle_sha256: str) -> dict[str, Any]:
    """Authenticate, verify the bundle manifest, and HEAD every object."""

    import boto3
    from botocore.config import Config

    bundle_sha256 = _require_bundle_sha256(bundle_sha256)
    style = os.environ.get("AWS_S3_URL_STYLE", "virtual")
    if style in {"virtual-host", "virtual-hosted"}:
        style = "virtual"
    client = boto3.client(
        "s3",
        endpoint_url=os.environ["AWS_ENDPOINT_URL"],
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
        region_name=os.environ.get("AWS_DEFAULT_REGION", "auto"),
        config=Config(
            retries={"max_attempts": 5, "mode": "adaptive"},
            s3={"addressing_style": style},
            connect_timeout=10,
            read_timeout=30,
            tcp_keepalive=True,
        ),
    )
    bucket = os.environ["AWS_S3_BUCKET_NAME"]
    root = f"{OBJECT_PREFIX}/{bundle_sha256}"
    manifest_response = client.get_object(Bucket=bucket, Key=f"{root}/bundle.json")
    manifest_bytes = manifest_response["Body"].read()
    manifest = json.loads(manifest_bytes)
    _validate_bundle_manifest(bundle_sha256, manifest)
    verified_bytes = 0
    for relative, record in manifest["files"].items():
        head = client.head_object(Bucket=bucket, Key=f"{root}/{relative}")
        metadata = {key.lower(): value for key, value in head["Metadata"].items()}
        if int(head["ContentLength"]) != int(record["size_bytes"]):
            raise RuntimeError(f"Remote size mismatch: {relative}")
        if metadata.get("sha256") != record["sha256"]:
            raise RuntimeError(f"Remote checksum metadata mismatch: {relative}")
        verified_bytes += int(record["size_bytes"])
    return {
        "schema_version": "chess-dfm-modal-storage-smoke-v1",
        "bundle_sha256": bundle_sha256,
        "kind": manifest["kind"],
        "verified_files": len(manifest["files"]),
        "verified_bytes": verified_bytes,
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "credentials_exposed": False,
        "gpu_active": False,
    }


@app.function(
    image=control_image,
    secrets=[railway_secret],
    volumes={INPUT_VOLUME_MOUNT: input_volume},
    cpu=1.0,
    memory=2048,
    timeout=1800,
    startup_timeout=300,
    retries=1,
    max_containers=1,
    single_use_containers=True,
)
def stage_input_bundle(
    bundle_sha256: str,
    require_workspace_guard: bool = True,
) -> dict[str, Any]:
    """Download and fully verify inputs using CPU-only compute, then commit the cache."""

    from research.interpretability.artifact_store import (
        DEFAULT_PREFIX,
        create_s3_client,
        create_transfer_config,
        load_credentials,
        pull_bundle,
    )
    from research.interpretability.artifacts import write_json_atomic

    bundle_sha256 = _require_bundle_sha256(bundle_sha256)
    root = Path(INPUT_VOLUME_MOUNT) / "bundles"
    target = root / bundle_sha256
    marker = root / f"{bundle_sha256}.stage.json"
    if target.exists() or marker.exists():
        verification = _verify_staged_bundle(
            bundle_sha256,
            target,
            marker,
            hash_files=True,
        )
        return {
            "schema_version": "chess-dfm-modal-input-stage-result-v1",
            **verification,
            "cache_hit": True,
            "network_download_bytes": 0,
            "workspace_path_guard_required": require_workspace_guard,
            "workspace_path_guard_ready": (
                _verify_volume_workspace_guard(target) if require_workspace_guard else None
            ),
            "gpu_active": False,
        }

    credentials = load_credentials()
    client = create_s3_client(credentials)
    pulled = pull_bundle(
        bundle_sha256,
        target,
        client=client,
        bucket=credentials.bucket_name,
        prefix=DEFAULT_PREFIX,
        transfer_config=create_transfer_config(),
    )
    manifest_key = f"{DEFAULT_PREFIX}/{bundle_sha256}/bundle.json"
    response = client.get_object(Bucket=credentials.bucket_name, Key=manifest_key)
    manifest = json.loads(response["Body"].read())
    _validate_bundle_manifest(bundle_sha256, manifest)
    write_json_atomic(
        marker,
        {
            "schema_version": "chess-dfm-modal-volume-stage-v1",
            "bundle": manifest,
            "source": "Railway private S3-compatible bucket",
            "full_file_hashes_verified_before_commit": True,
        },
    )
    verification = _verify_staged_bundle(
        bundle_sha256,
        target,
        marker,
        hash_files=True,
    )
    input_volume.commit()
    return {
        "schema_version": "chess-dfm-modal-input-stage-result-v1",
        **verification,
        "cache_hit": False,
        "network_download_bytes": pulled["total_size_bytes"],
        "workspace_path_guard_required": require_workspace_guard,
        "workspace_path_guard_ready": (
            _verify_volume_workspace_guard(target) if require_workspace_guard else None
        ),
        "gpu_active": False,
    }


@app.function(
    image=gpu_image,
    gpu="T4",
    cpu=1.0,
    memory=2048,
    timeout=60,
    startup_timeout=300,
    retries=0,
    max_containers=1,
    single_use_containers=True,
)
def gpu_smoke() -> dict[str, Any]:
    """Minimal paid validation of the pinned Torch/CUDA execution surface."""

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Modal T4 function cannot see CUDA")
    torch.manual_seed(20260807)
    torch.backends.cuda.matmul.allow_tf32 = False
    first = torch.arange(4096, device="cuda", dtype=torch.float32).reshape(64, 64)
    second = first.transpose(0, 1)
    result = first @ second
    if not torch.isfinite(result).all():
        raise RuntimeError("Modal GPU smoke produced nonfinite values")
    return {
        "schema_version": "chess-dfm-modal-gpu-smoke-v1",
        "torch": str(torch.__version__),
        "cuda_runtime": str(torch.version.cuda),
        "device_name": str(torch.cuda.get_device_name(0)),
        "compute_capability": [int(value) for value in torch.cuda.get_device_capability(0)],
        "result_sha256": hashlib.sha256(result.cpu().numpy().tobytes()).hexdigest(),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
    }


def _run_probe_job(
    bundle_sha256: str,
    *,
    profile: str,
    declared_attempt_upper_bound_dollars: str,
) -> dict[str, Any]:
    from research.interpretability.artifacts import verify_checksums

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Modal probe job cannot see CUDA")
    bundle_sha256 = _require_bundle_sha256(bundle_sha256)
    input_root = Path(INPUT_VOLUME_MOUNT) / "bundles"
    input_dir = input_root / bundle_sha256
    marker = input_root / f"{bundle_sha256}.stage.json"
    staged = _verify_staged_bundle(
        bundle_sha256,
        input_dir,
        marker,
        hash_files=False,
    )
    config = _probe_config(profile)
    result_label = _probe_result_label(bundle_sha256, profile)
    output = Path(RESULT_VOLUME_MOUNT) / "runs" / result_label
    if output.exists():
        checksums = verify_checksums(output)
        manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
        return {
            "schema_version": "chess-dfm-modal-probe-result-v1",
            "result_label": result_label,
            "run_id": manifest["run_id"],
            "metrics_sha256": checksums["metrics.json"],
            "gates_passed": metrics["correctness_gates"]["passed"],
            "profile": profile,
            "input_stage": staged,
            "reused_immutable_result": True,
        }

    os.environ["CHESS_DFM_WORKSPACE_ROOT"] = INPUT_VOLUME_MOUNT
    from research.interpretability.probe_pilot import run_probe_pilot

    args = Namespace(
        raw_bt4=input_dir / "models/raw/BT4_exported.pb.gz",
        hero_checkpoint=input_dir / "models/hero",
        corpus_manifest=input_dir / "corpus/manifest.json",
        output=output,
        device="cuda",
        layers=config["layers"],
        future_ply=config["future_ply"],
        fit_count=config["fit_count"],
        selection_count=config["selection_count"],
        evaluation_count=config["evaluation_count"],
        capture_batch_size=config["capture_batch_size"],
        l2_values=config["l2_values"],
        probe_epochs=config["probe_epochs"],
        probe_learning_rate=config["probe_learning_rate"],
        probe_batch_size=config["probe_batch_size"],
        lookahead_epochs=config["lookahead_epochs"],
        causal_count=config["causal_count"],
        causal_dose=config["causal_dose"],
        bootstrap_samples=config["bootstrap_samples"],
        seed=config["seed"],
        cpu_threads=config["cpu_threads"],
        backend="Modal T4 with CPU-staged immutable Volume inputs",
        provider_dollars=None,
        declared_attempt_upper_bound_dollars=float(declared_attempt_upper_bound_dollars),
        input_bundle_sha256=bundle_sha256,
    )
    result = run_probe_pilot(args)
    result_volume.commit()
    return {
        "schema_version": "chess-dfm-modal-probe-result-v1",
        "result_label": result_label,
        "profile": profile,
        "input_stage": staged,
        "reused_immutable_result": False,
        **result,
    }


_PROBE_VOLUMES = {
    INPUT_VOLUME_MOUNT: input_volume.with_mount_options(read_only=True),
    RESULT_VOLUME_MOUNT: result_volume,
}


@app.function(
    image=gpu_image,
    volumes=_PROBE_VOLUMES,
    gpu="T4",
    cpu=4.0,
    memory=16384,
    timeout=1200,
    startup_timeout=600,
    retries=0,
    max_containers=1,
    single_use_containers=True,
)
def probe_smoke_job(
    bundle_sha256: str,
    declared_attempt_upper_bound_dollars: str,
) -> dict[str, Any]:
    return _run_probe_job(
        bundle_sha256,
        profile="smoke",
        declared_attempt_upper_bound_dollars=declared_attempt_upper_bound_dollars,
    )


@app.function(
    image=gpu_image,
    volumes=_PROBE_VOLUMES,
    gpu="T4",
    cpu=4.0,
    memory=24576,
    timeout=7200,
    startup_timeout=900,
    retries=0,
    max_containers=1,
    single_use_containers=True,
)
def probe_all_layers_job(
    bundle_sha256: str,
    declared_attempt_upper_bound_dollars: str,
) -> dict[str, Any]:
    return _run_probe_job(
        bundle_sha256,
        profile="all-layers",
        declared_attempt_upper_bound_dollars=declared_attempt_upper_bound_dollars,
    )



def _run_lorsa_job(
    base_bundle_sha256: str,
    lorsa_bundle_sha256: str,
    *,
    profile: str,
    gpu: str,
    declared_attempt_upper_bound_dollars: str,
    manifest_relative: str,
    weights_relative: str,
) -> dict[str, Any]:
    import subprocess
    import sys

    from research.interpretability.artifacts import verify_checksums

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Modal LoRSA job cannot see CUDA")
    device_name = str(torch.cuda.get_device_name(0))
    if gpu not in device_name:
        raise RuntimeError(f"Requested {gpu} but Modal exposed {device_name!r}")
    base_bundle_sha256 = _require_bundle_sha256(base_bundle_sha256)
    lorsa_bundle_sha256 = _require_bundle_sha256(lorsa_bundle_sha256)
    input_root = Path(INPUT_VOLUME_MOUNT) / "bundles"
    base_dir = input_root / base_bundle_sha256
    base_marker = input_root / f"{base_bundle_sha256}.stage.json"
    lorsa_dir = input_root / lorsa_bundle_sha256
    lorsa_marker = input_root / f"{lorsa_bundle_sha256}.stage.json"
    staged_base = _verify_staged_bundle(
        base_bundle_sha256,
        base_dir,
        base_marker,
        hash_files=False,
    )
    staged_lorsa = _verify_staged_bundle(
        lorsa_bundle_sha256,
        lorsa_dir,
        lorsa_marker,
        hash_files=False,
    )
    lorsa_manifest = _bundle_file(lorsa_dir, manifest_relative)
    lorsa_weights = _bundle_file(lorsa_dir, weights_relative)
    config = _lorsa_config(profile)
    result_label = _lorsa_result_label(
        base_bundle_sha256,
        lorsa_bundle_sha256,
        profile=profile,
        gpu=gpu,
        manifest_relative=manifest_relative,
        weights_relative=weights_relative,
    )
    output = Path(RESULT_VOLUME_MOUNT) / "runs" / result_label
    if output.exists():
        checksums = verify_checksums(output)
        manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
        return {
            "schema_version": "chess-dfm-modal-lorsa-result-v1",
            "result_label": result_label,
            "run_id": manifest["run_id"],
            "metrics_sha256": checksums["metrics.json"],
            "source_gate_passed": metrics["source_compatibility_gate"]["passed"],
            "transfer_status": metrics["transfer_stage"]["status"],
            "profile": profile,
            "gpu": gpu,
            "resource_usage": manifest["resource_usage"],
            "input_stage": {"base": staged_base, "lorsa": staged_lorsa},
            "reused_immutable_result": True,
        }

    os.environ["CHESS_DFM_WORKSPACE_ROOT"] = INPUT_VOLUME_MOUNT
    command = [
        sys.executable,
        "-m",
        "research.interpretability.sparse.lorsa_transfer_pilot",
        "--raw-bt4",
        str(base_dir / "models/raw/BT4_exported.pb.gz"),
        "--hero-checkpoint",
        str(base_dir / "models/hero"),
        "--corpus-manifest",
        str(base_dir / "corpus/manifest.json"),
        "--lorsa-manifest",
        str(lorsa_manifest),
        "--lorsa-weights",
        str(lorsa_weights),
        "--output",
        str(output),
        "--device",
        "cuda",
        "--layer",
        str(config["layer"]),
        "--position-count",
        str(config["position_count"]),
        "--batch-size",
        str(config["batch_size"]),
        "--feature-summary-count",
        str(config["feature_summary_count"]),
        "--seed",
        str(config["seed"]),
        "--cpu-threads",
        str(config["cpu_threads"]),
        "--backend",
        f"Modal {gpu} with two CPU-staged immutable Volume bundles",
        "--declared-attempt-upper-bound-dollars",
        declared_attempt_upper_bound_dollars,
        "--base-input-bundle-sha256",
        base_bundle_sha256,
        "--lorsa-input-bundle-sha256",
        lorsa_bundle_sha256,
    ]
    completed = subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    print(completed.stdout, end="", flush=True)
    if completed.returncode != 0:
        raise RuntimeError(f"LoRSA module runner exited with status {completed.returncode}")
    output_lines = [line for line in completed.stdout.splitlines() if line.strip()]
    try:
        result = json.loads(output_lines[-1])
    except (IndexError, json.JSONDecodeError) as exc:
        raise RuntimeError("LoRSA module runner did not emit a final JSON result") from exc
    checksums = verify_checksums(output)
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    result_volume.commit()
    return {
        "schema_version": "chess-dfm-modal-lorsa-result-v1",
        "result_label": result_label,
        "profile": profile,
        "gpu": gpu,
        "resource_usage": manifest["resource_usage"],
        "input_stage": {"base": staged_base, "lorsa": staged_lorsa},
        "reused_immutable_result": False,
        "metrics_sha256": checksums["metrics.json"],
        **result,
    }


@app.function(
    image=gpu_image,
    volumes=_PROBE_VOLUMES,
    gpu="T4",
    cpu=4.0,
    memory=24576,
    timeout=1800,
    startup_timeout=600,
    retries=0,
    max_containers=1,
    single_use_containers=True,
)
def lorsa_smoke_t4_job(
    base_bundle_sha256: str,
    lorsa_bundle_sha256: str,
    declared_attempt_upper_bound_dollars: str,
    manifest_relative: str,
    weights_relative: str,
) -> dict[str, Any]:
    return _run_lorsa_job(
        base_bundle_sha256,
        lorsa_bundle_sha256,
        profile="smoke",
        gpu="T4",
        declared_attempt_upper_bound_dollars=declared_attempt_upper_bound_dollars,
        manifest_relative=manifest_relative,
        weights_relative=weights_relative,
    )


@app.function(
    image=gpu_image,
    volumes=_PROBE_VOLUMES,
    gpu="L4",
    cpu=4.0,
    memory=24576,
    timeout=1800,
    startup_timeout=600,
    retries=0,
    max_containers=1,
    single_use_containers=True,
)
def lorsa_smoke_l4_job(
    base_bundle_sha256: str,
    lorsa_bundle_sha256: str,
    declared_attempt_upper_bound_dollars: str,
    manifest_relative: str,
    weights_relative: str,
) -> dict[str, Any]:
    return _run_lorsa_job(
        base_bundle_sha256,
        lorsa_bundle_sha256,
        profile="smoke",
        gpu="L4",
        declared_attempt_upper_bound_dollars=declared_attempt_upper_bound_dollars,
        manifest_relative=manifest_relative,
        weights_relative=weights_relative,
    )


@app.function(
    image=gpu_image,
    volumes=_PROBE_VOLUMES,
    gpu="T4",
    cpu=4.0,
    memory=24576,
    timeout=5400,
    startup_timeout=900,
    retries=0,
    max_containers=1,
    single_use_containers=True,
)
def lorsa_development_t4_job(
    base_bundle_sha256: str,
    lorsa_bundle_sha256: str,
    declared_attempt_upper_bound_dollars: str,
    manifest_relative: str,
    weights_relative: str,
) -> dict[str, Any]:
    return _run_lorsa_job(
        base_bundle_sha256,
        lorsa_bundle_sha256,
        profile="development",
        gpu="T4",
        declared_attempt_upper_bound_dollars=declared_attempt_upper_bound_dollars,
        manifest_relative=manifest_relative,
        weights_relative=weights_relative,
    )


@app.function(
    image=gpu_image,
    volumes=_PROBE_VOLUMES,
    gpu="L4",
    cpu=4.0,
    memory=24576,
    timeout=5400,
    startup_timeout=900,
    retries=0,
    max_containers=1,
    single_use_containers=True,
)
def lorsa_development_l4_job(
    base_bundle_sha256: str,
    lorsa_bundle_sha256: str,
    declared_attempt_upper_bound_dollars: str,
    manifest_relative: str,
    weights_relative: str,
) -> dict[str, Any]:
    return _run_lorsa_job(
        base_bundle_sha256,
        lorsa_bundle_sha256,
        profile="development",
        gpu="L4",
        declared_attempt_upper_bound_dollars=declared_attempt_upper_bound_dollars,
        manifest_relative=manifest_relative,
        weights_relative=weights_relative,
    )

@app.function(
    image=control_image,
    secrets=[railway_secret],
    volumes={RESULT_VOLUME_MOUNT: result_volume.with_mount_options(read_only=True)},
    cpu=0.5,
    memory=1024,
    timeout=900,
    startup_timeout=300,
    retries=1,
    max_containers=1,
    single_use_containers=True,
)
def publish_probe_result(result_label: str) -> dict[str, Any]:
    """Publish a committed result Volume directory to Railway without a GPU."""

    from research.interpretability.artifact_store import (
        build_run_bundle,
        create_s3_client,
        create_transfer_config,
        load_credentials,
        push_bundle,
    )
    from research.interpretability.artifacts import verify_checksums

    result_label = _require_result_label(result_label)
    run_dir = Path(RESULT_VOLUME_MOUNT) / "runs" / result_label
    checksums = verify_checksums(run_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    credentials = load_credentials()
    bundle = build_run_bundle(run_dir)
    pushed = push_bundle(
        bundle,
        client=create_s3_client(credentials),
        bucket=credentials.bucket_name,
        transfer_config=create_transfer_config(),
    )
    return {
        "schema_version": "chess-dfm-modal-result-publish-v1",
        "result_label": result_label,
        "run_id": manifest["run_id"],
        "metrics_sha256": checksums["metrics.json"],
        "gpu_active": False,
        "railway": pushed,
    }


def _spec(stage: str) -> tuple[Any, Decimal]:
    from research.interpretability.remote_cost import ModalJobSpec

    if stage == "control-plane":
        return (
            ModalJobSpec(
                name="railway-control-plane-smoke",
                timeout_seconds=60,
                startup_timeout_seconds=120,
                retries=1,
                cpu_cores=Decimal("0.25"),
                memory_gib=Decimal("0.5"),
            ),
            Decimal("0.01"),
        )
    if stage == "stage-input":
        return (
            ModalJobSpec(
                name="cpu-stage-railway-input-bundle",
                timeout_seconds=1800,
                startup_timeout_seconds=300,
                retries=1,
                cpu_cores=Decimal("1"),
                memory_gib=Decimal("2"),
            ),
            Decimal("0.10"),
        )
    if stage == "gpu-smoke":
        return (
            ModalJobSpec(
                name="t4-gpu-smoke",
                timeout_seconds=60,
                startup_timeout_seconds=300,
                retries=0,
                cpu_cores=Decimal("1"),
                memory_gib=Decimal("2"),
                gpu="T4",
            ),
            Decimal("0.10"),
        )
    if stage == "probe-smoke-gpu":
        return (
            ModalJobSpec(
                name="t4-probe-pipeline-smoke",
                timeout_seconds=1200,
                startup_timeout_seconds=600,
                retries=0,
                cpu_cores=Decimal("4"),
                memory_gib=Decimal("16"),
                gpu="T4",
            ),
            Decimal("0.50"),
        )
    if stage == "probe-all-layers-gpu":
        return (
            ModalJobSpec(
                name="t4-probe-all-layers-development",
                timeout_seconds=7200,
                startup_timeout_seconds=900,
                retries=0,
                cpu_cores=Decimal("4"),
                memory_gib=Decimal("24"),
                gpu="T4",
            ),
            Decimal("2.25"),
        )
    lorsa_stages = {
        "lorsa-smoke-t4-gpu": ("smoke", "T4", 1800, 600, Decimal("0.70")),
        "lorsa-smoke-l4-gpu": ("smoke", "L4", 1800, 600, Decimal("0.85")),
        "lorsa-development-t4-gpu": (
            "development", "T4", 5400, 900, Decimal("1.80")
        ),
        "lorsa-development-l4-gpu": (
            "development", "L4", 5400, 900, Decimal("2.20")
        ),
    }
    if stage in lorsa_stages:
        profile, gpu, timeout, startup_timeout, cap = lorsa_stages[stage]
        return (
            ModalJobSpec(
                name=f"{gpu.lower()}-lorsa-{profile}-transfer",
                timeout_seconds=timeout,
                startup_timeout_seconds=startup_timeout,
                retries=0,
                cpu_cores=Decimal("4"),
                memory_gib=Decimal("24"),
                gpu=gpu,
            ),
            cap,
        )
    if stage == "publish-result":
        return (
            ModalJobSpec(
                name="cpu-publish-probe-result",
                timeout_seconds=900,
                startup_timeout_seconds=300,
                retries=1,
                cpu_cores=Decimal("0.5"),
                memory_gib=Decimal("1"),
            ),
            Decimal("0.03"),
        )
    raise ValueError(f"Unknown cost stage {stage!r}")


@app.local_entrypoint()
def main(
    mode: str = "control-plane",
    bundle_sha256: str = INPUT_BUNDLE_SHA256,
    result_label: str = "",
    lorsa_bundle_sha256: str = "",
    lorsa_manifest_relative: str = DEFAULT_LORSA_MANIFEST_RELATIVE,
    lorsa_weights_relative: str = DEFAULT_LORSA_WEIGHTS_RELATIVE,
    recorded_spend: str = "",
    per_job_cap: str = "",
) -> None:
    from research.interpretability.remote_cost import (
        enforce_modal_budget,
        estimate_modal_job,
        modal_month_spend,
    )

    bundle_sha256 = _require_bundle_sha256(bundle_sha256)
    spend_floor = Decimal(recorded_spend) if recorded_spend else Decimal("0")

    def guard(stage: str, *, cap_override: str = "") -> tuple[dict[str, Any], Decimal]:
        nonlocal spend_floor
        spec, default_cap = _spec(stage)
        estimate = estimate_modal_job(spec)
        measured = modal_month_spend()
        spend = max(measured, spend_floor)
        cap = Decimal(cap_override) if cap_override else default_cap
        decision = enforce_modal_budget(
            estimate,
            recorded_month_spend=spend,
            per_job_cap=cap,
        )
        spend_floor = Decimal(decision["projected_month_spend"])
        print(json.dumps({"launch_stage": stage, "cost_guard": decision}, sort_keys=True))
        return decision, spend_floor

    if mode == "control-plane":
        guard("control-plane", cap_override=per_job_cap)
        result = railway_control_plane_smoke.remote(bundle_sha256)
        print(json.dumps({"result": result}, sort_keys=True))
        return
    if mode == "gpu-smoke":
        guard("gpu-smoke", cap_override=per_job_cap)
        result = gpu_smoke.remote()
        print(json.dumps({"result": result}, sort_keys=True))
        return
    if mode == "stage-input":
        guard("stage-input", cap_override=per_job_cap)
        staged = stage_input_bundle.remote(bundle_sha256)
        print(json.dumps({"stage_result": staged}, sort_keys=True))
        return
    if mode == "publish-result":
        if not result_label:
            raise ValueError("--result-label is required for publish-result")
        guard("publish-result", cap_override=per_job_cap)
        published = publish_probe_result.remote(result_label)
        print(json.dumps({"publish_result": published}, sort_keys=True))
        return
    lorsa_modes = {
        "lorsa-smoke-t4": (
            "lorsa-smoke-t4-gpu",
            lorsa_smoke_t4_job,
        ),
        "lorsa-smoke-l4": (
            "lorsa-smoke-l4-gpu",
            lorsa_smoke_l4_job,
        ),
        "lorsa-development-t4": (
            "lorsa-development-t4-gpu",
            lorsa_development_t4_job,
        ),
        "lorsa-development-l4": (
            "lorsa-development-l4-gpu",
            lorsa_development_l4_job,
        ),
    }
    if mode in lorsa_modes:
        if not lorsa_bundle_sha256:
            raise ValueError("--lorsa-bundle-sha256 is required for LoRSA modes")
        lorsa_bundle_sha256 = _require_bundle_sha256(lorsa_bundle_sha256)
        for relative in (lorsa_manifest_relative, lorsa_weights_relative):
            logical = PurePosixPath(relative)
            if logical.is_absolute() or any(
                part in {"", ".", ".."} for part in logical.parts
            ):
                raise ValueError(f"Unsafe LoRSA bundle-relative path: {relative!r}")

        guard("stage-input")
        staged_base = stage_input_bundle.remote(bundle_sha256, True)
        print(json.dumps({"base_stage_result": staged_base}, sort_keys=True))
        guard("stage-input")
        staged_lorsa = stage_input_bundle.remote(lorsa_bundle_sha256, False)
        print(json.dumps({"lorsa_stage_result": staged_lorsa}, sort_keys=True))

        gpu_stage, job = lorsa_modes[mode]
        gpu_decision, _ = guard(gpu_stage, cap_override=per_job_cap)
        declared = gpu_decision["declared_attempt_upper_bound"]
        gpu_result = job.remote(
            bundle_sha256,
            lorsa_bundle_sha256,
            declared,
            lorsa_manifest_relative,
            lorsa_weights_relative,
        )
        print(json.dumps({"gpu_result": gpu_result}, sort_keys=True))

        guard("publish-result")
        published = publish_probe_result.remote(gpu_result["result_label"])
        print(
            json.dumps(
                {
                    "pipeline_result": {
                        "stage": {
                            "base": staged_base,
                            "lorsa": staged_lorsa,
                        },
                        "gpu": gpu_result,
                        "publish": published,
                    }
                },
                sort_keys=True,
            )
        )
        return

    if mode not in {"probe-smoke", "probe-all-layers"}:
        raise ValueError(f"Unknown mode {mode!r}")

    guard("stage-input")
    staged = stage_input_bundle.remote(bundle_sha256)
    print(json.dumps({"stage_result": staged}, sort_keys=True))

    gpu_stage = "probe-smoke-gpu" if mode == "probe-smoke" else "probe-all-layers-gpu"
    gpu_decision, _ = guard(gpu_stage, cap_override=per_job_cap)
    declared = gpu_decision["declared_attempt_upper_bound"]
    gpu_result = (
        probe_smoke_job.remote(bundle_sha256, declared)
        if mode == "probe-smoke"
        else probe_all_layers_job.remote(bundle_sha256, declared)
    )
    print(json.dumps({"gpu_result": gpu_result}, sort_keys=True))

    guard("publish-result")
    published = publish_probe_result.remote(gpu_result["result_label"])
    print(
        json.dumps(
            {
                "pipeline_result": {
                    "stage": staged,
                    "gpu": gpu_result,
                    "publish": published,
                }
            },
            sort_keys=True,
        )
    )
