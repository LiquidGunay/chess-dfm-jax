"""Stage, launch, and externally monitor the promoted Proposal A epoch.

The control plane deliberately separates storage staging from GPU creation.
Inputs are uploaded to a Runpod network volume through its S3 API before a Pod
exists.  The Pod receives no Railway credential and can only consume the
content-bound launch manifest already on the volume.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime, timedelta
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
import time
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PREREGISTER = (
    REPO_ROOT / "research" / "analysis" / "proposal_a_one_epoch_preregister_20260810.json"
)
DEFAULT_RUNPODCTL = REPO_ROOT / ".local" / "bin" / "runpodctl"
DEFAULT_PREFIX = "chess-dfm/proposal-a/v1"
SOURCE_TAR = (
    REPO_ROOT / "data" / "raw" / "lc0" / "test80" / "training-run1-test80-20240401-0117.tar"
)
RAW_BT4 = REPO_ROOT / "models" / "source" / "extracted" / "BT4_exported.pb.gz"
BOOTSTRAP = Path(__file__).with_name("runpod_epoch_bootstrap.py")


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial-{os.getpid()}")
    temporary.write_bytes(_json_bytes(payload))
    os.replace(temporary, path)
    path.chmod(0o600)


def validate_preregister(
    path: Path = DEFAULT_PREREGISTER,
    *,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    payload = _load_json(path)
    if payload.get("schema_version") != "proposal-a-one-epoch-preregister-v1":
        raise ValueError("Unsupported Proposal A preregistration schema")
    if payload["launch_binding"]["status"] != "prepared_not_launched":
        raise ValueError("Preregistration launch status drift")
    if payload["model_contract"] != {
        "architecture": (
            "Proposal A open-loop z0 -> DFM(a0..a7) -> "
            "action-conditioned JEPA(z1..z8), with current and future WDL heads"
        ),
        "compute_dtype": "bfloat16",
        "encoder_learning_rate_ratio": 1.0,
        "horizon": 8,
        "main_learning_rate": 5e-5,
        "optimizer_precision": "fp32_master",
        "recipe": "hero_v2",
        "weight_decay": 0.05622851404690516,
        "weight_decay_mode": "decoupled",
        "wdl_include_current_state": True,
    }:
        raise ValueError("Promoted model contract drift")
    forbidden = payload["forbidden_training_features"]
    if forbidden != {
        "dfm_closed_loop_mode": "none",
        "future_jepa_teacher_forcing": False,
        "jepa_inference_feedback": False,
        "legality_auxiliary_coefficient": 0.0,
        "policy_teacher_checkpoint": None,
        "teacher_kl_coefficient": 0.0,
    }:
        raise ValueError("Forbidden training feature contract drift")
    schedule = payload["schedule_contract"]
    if (
        schedule["steps"] != 7774
        or schedule["total_examples"] != 7_960_576
        or schedule["batch_size"] != 1024
        or schedule["lr_schedule"] != "warmup_stable_linear_decay"
    ):
        raise ValueError("One-epoch schedule drift")
    if payload["execution_contract"]["expected_gpu"] != "NVIDIA L40S":
        raise ValueError("Promoted GPU contract drift")
    if payload["execution_contract"]["cloud"] != "SECURE":
        raise ValueError("Promoted cloud contract drift")
    for section_name in ("learning_rate_analysis", "decay_analysis"):
        record = payload["selection_evidence"][section_name]
        referenced = repo_root / record["path"]
        if _sha256_file(referenced) != record["sha256"]:
            raise ValueError(f"Selection evidence checksum drift: {section_name}")
    for section_name in ("audit",):
        record = payload["data_contract"][section_name]
        referenced = repo_root / record["path"]
        if _sha256_file(referenced) != record["sha256"]:
            raise ValueError(f"Data evidence checksum drift: {section_name}")
    for section_name in ("source_tar", "raw_bt4"):
        record = payload["data_contract"][section_name]
        receipt = repo_root / record["receipt_path"]
        if _sha256_file(receipt) != record["receipt_sha256"]:
            raise ValueError(f"Data receipt checksum drift: {section_name}")
    command = payload["training_command"]
    joined = "\0".join(command)
    required_fragments = (
        "--policy-distill-coeff\x000.0",
        "--dfm-closed-loop-mode\x00none",
        "--legality-coeff\x000.0",
        "--validation-example-count\x008192",
        "--encoder-lr-ratio\x001.0",
        "--weight-decay-mode\x00decoupled",
    )
    if any(fragment not in joined for fragment in required_fragments):
        raise ValueError("Training command no longer pins the promoted recipe")
    return payload


def _run_git(repo_root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ("git", *arguments),
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def resolve_clean_pushed_git_binding(
    *,
    repo_root: Path = REPO_ROOT,
    branch: str = "research/local-gpu-autoresearch",
) -> dict[str, str]:
    if _run_git(repo_root, "status", "--porcelain"):
        raise RuntimeError("Refusing to stage from a dirty checkout")
    current_branch = _run_git(repo_root, "branch", "--show-current")
    if current_branch != branch:
        raise RuntimeError(f"Expected branch {branch!r}, got {current_branch!r}")
    commit = _run_git(repo_root, "rev-parse", "HEAD")
    upstream = f"origin/{branch}"
    remote_commit = _run_git(repo_root, "rev-parse", upstream)
    if commit != remote_commit:
        raise RuntimeError(f"HEAD is not exactly pushed to {upstream}")
    return {
        "branch": branch,
        "commit": commit,
        "remote": "https://github.com/LiquidGunay/chess-dfm-jax.git",
        "upstream": upstream,
    }


def _verified_local_object(
    path: Path,
    *,
    expected_size: int,
    expected_sha256: str,
) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    size = path.stat().st_size
    if size != expected_size:
        raise ValueError(f"Size drift for {path}: {size} != {expected_size}")
    digest = _sha256_file(path)
    if digest != expected_sha256:
        raise ValueError(f"SHA-256 drift for {path}")
    return {
        "local_path": str(path),
        "size_bytes": size,
        "sha256": digest,
    }


def build_launch_manifest(
    *,
    preregister: Mapping[str, Any],
    preregister_sha256: str,
    git_binding: Mapping[str, str],
    volume_id: str,
    s3_endpoint: str,
    prefix: str = DEFAULT_PREFIX,
    created_utc: str | None = None,
) -> dict[str, Any]:
    if not volume_id or "/" in volume_id:
        raise ValueError("Runpod network volume ID must be non-empty and path-free")
    if not s3_endpoint.startswith("https://s3api-"):
        raise ValueError("Expected an HTTPS Runpod S3 endpoint")
    mount_root = f"/workspace/{prefix}"
    objects = {
        "source_tar": {
            "key": f"{prefix}/inputs/training-run1-test80-20240401-0117.tar",
            **preregister["data_contract"]["source_tar"],
        },
        "raw_bt4": {
            "key": f"{prefix}/inputs/BT4_exported.pb.gz",
            **preregister["data_contract"]["raw_bt4"],
        },
        "bootstrap": {
            "key": f"{prefix}/control/runpod_epoch_bootstrap.py",
            "size_bytes": BOOTSTRAP.stat().st_size,
            "sha256": _sha256_file(BOOTSTRAP),
        },
    }
    return {
        "schema_version": "proposal-a-runpod-launch-v1",
        "created_utc": created_utc or datetime.now(UTC).isoformat(),
        "preregister": dict(preregister),
        "preregister_sha256": _sha256_bytes(_json_bytes(preregister)),
        "preregister_file_sha256": preregister_sha256,
        "git": dict(git_binding),
        "network_volume": {
            "id": volume_id,
            "mount": "/workspace",
            "prefix": prefix,
            "s3_endpoint": s3_endpoint,
        },
        "objects": objects,
        "paths": {
            "source_tar": f"{mount_root}/inputs/training-run1-test80-20240401-0117.tar",
            "raw_bt4": f"{mount_root}/inputs/BT4_exported.pb.gz",
            "dataset": f"{mount_root}/data/lc0_sequential",
            "run_root": f"{mount_root}/run",
            "control": f"{mount_root}/control",
        },
    }


def _runpod_s3_client(endpoint: str) -> Any:
    try:
        import boto3
        from botocore.config import Config
    except ImportError as exc:
        raise RuntimeError("Install research/requirements_infra.txt first") from exc
    access_key = os.environ.get("RUNPOD_S3_ACCESS_KEY_ID")
    secret_key = os.environ.get("RUNPOD_S3_SECRET_ACCESS_KEY")
    if not access_key or not secret_key:
        raise RuntimeError("Set RUNPOD_S3_ACCESS_KEY_ID and RUNPOD_S3_SECRET_ACCESS_KEY")
    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        config=Config(
            retries={"max_attempts": 8, "mode": "adaptive"},
            signature_version="s3v4",
        ),
    )


def _stage_file(
    client: Any,
    *,
    volume_id: str,
    key: str,
    path: Path,
    sha256: str,
) -> dict[str, Any]:
    from boto3.s3.transfer import TransferConfig

    client.upload_file(
        str(path),
        volume_id,
        key,
        ExtraArgs={"Metadata": {"sha256": sha256}},
        Config=TransferConfig(
            multipart_threshold=256 * 1024 * 1024,
            multipart_chunksize=64 * 1024 * 1024,
            max_concurrency=2,
            use_threads=True,
        ),
    )
    head = client.head_object(Bucket=volume_id, Key=key)
    if int(head["ContentLength"]) != path.stat().st_size:
        raise RuntimeError(f"Remote size verification failed for {key}")
    if head.get("Metadata", {}).get("sha256") != sha256:
        raise RuntimeError(f"Remote SHA-256 metadata verification failed for {key}")
    return {
        "key": key,
        "size_bytes": path.stat().st_size,
        "sha256": sha256,
        "etag": str(head.get("ETag", "")).strip('"'),
    }


def stage(
    *,
    volume_id: str,
    s3_endpoint: str,
    output: Path,
    preregister_path: Path = DEFAULT_PREREGISTER,
    prefix: str = DEFAULT_PREFIX,
) -> dict[str, Any]:
    preregister = validate_preregister(preregister_path)
    preregister_sha256 = _sha256_file(preregister_path)
    git_binding = resolve_clean_pushed_git_binding(
        branch=preregister["launch_binding"]["git_branch"]
    )
    source = _verified_local_object(
        SOURCE_TAR,
        expected_size=preregister["data_contract"]["source_tar"]["size_bytes"],
        expected_sha256=preregister["data_contract"]["source_tar"]["sha256"],
    )
    raw = _verified_local_object(
        RAW_BT4,
        expected_size=preregister["data_contract"]["raw_bt4"]["size_bytes"],
        expected_sha256=preregister["data_contract"]["raw_bt4"]["sha256"],
    )
    manifest = build_launch_manifest(
        preregister=preregister,
        preregister_sha256=preregister_sha256,
        git_binding=git_binding,
        volume_id=volume_id,
        s3_endpoint=s3_endpoint,
        prefix=prefix,
    )
    client = _runpod_s3_client(s3_endpoint)
    staged: dict[str, Any] = {}
    for name, local in (
        ("source_tar", Path(source["local_path"])),
        ("raw_bt4", Path(raw["local_path"])),
        ("bootstrap", BOOTSTRAP),
    ):
        record = manifest["objects"][name]
        staged[name] = _stage_file(
            client,
            volume_id=volume_id,
            key=record["key"],
            path=local,
            sha256=record["sha256"],
        )
    with tempfile.TemporaryDirectory(prefix="proposal-a-runpod-stage-") as temporary:
        manifest_path = Path(temporary) / "launch_manifest.json"
        manifest_path.write_bytes(_json_bytes(manifest))
        manifest_key = f"{prefix}/control/launch_manifest.json"
        staged["launch_manifest"] = _stage_file(
            client,
            volume_id=volume_id,
            key=manifest_key,
            path=manifest_path,
            sha256=_sha256_file(manifest_path),
        )
    receipt = {
        "schema_version": "proposal-a-runpod-stage-receipt-v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "volume_id": volume_id,
        "s3_endpoint": s3_endpoint,
        "prefix": prefix,
        "git": git_binding,
        "preregister_sha256": preregister_sha256,
        "launch_manifest_sha256": staged["launch_manifest"]["sha256"],
        "objects": staged,
        "gpu_started": False,
    }
    _write_json_atomic(output, receipt)
    return receipt


def build_pod_create_command(
    *,
    stage_receipt: Mapping[str, Any],
    preregister: Mapping[str, Any],
    runpodctl: Path = DEFAULT_RUNPODCTL,
    now: datetime | None = None,
) -> list[str]:
    execution = preregister["execution_contract"]
    if stage_receipt.get("gpu_started") is not False:
        raise ValueError("Stage receipt does not prove zero-GPU staging")
    launched_at = now or datetime.now(UTC)
    if launched_at.tzinfo is None:
        raise ValueError("Launch time must be timezone-aware")
    terminate_at = launched_at + timedelta(hours=int(execution["absolute_auto_terminate_hours"]))
    prefix = stage_receipt["prefix"]
    bootstrap = f"/workspace/{prefix}/control/runpod_epoch_bootstrap.py"
    manifest = f"/workspace/{prefix}/control/launch_manifest.json"
    docker_args = f"python {shlex.quote(bootstrap)} {shlex.quote(manifest)}"
    return [
        str(runpodctl),
        "pod",
        "create",
        "--cloud-type",
        execution["cloud"],
        "--image",
        execution["image"],
        "--gpu-id",
        execution["expected_gpu"],
        "--gpu-count",
        str(execution["gpu_count"]),
        "--container-disk-in-gb",
        str(execution["container_disk_gb"]),
        "--network-volume-id",
        stage_receipt["volume_id"],
        "--volume-mount-path",
        execution["network_volume_mount"],
        "--name",
        "chess-dfm-proposal-a-epoch-v1",
        "--docker-args",
        docker_args,
        "--terminate-after",
        terminate_at.isoformat().replace("+00:00", "Z"),
    ]


def _runpodctl_json(
    runpodctl: Path,
    arguments: Sequence[str],
) -> dict[str, Any]:
    completed = subprocess.run(
        (str(runpodctl), *arguments),
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    if not isinstance(payload, dict):
        raise RuntimeError("Runpod CLI returned a non-object response")
    return payload


def _delete_pod(runpodctl: Path, pod_id: str) -> dict[str, Any]:
    return _runpodctl_json(runpodctl, ("pod", "delete", pod_id))


def _pod_gpu_name(payload: Mapping[str, Any]) -> str:
    gpu = payload.get("gpu")
    if isinstance(gpu, Mapping) and gpu.get("displayName"):
        return str(gpu["displayName"])
    machine = payload.get("machine")
    if isinstance(machine, Mapping):
        gpu_type = machine.get("gpuType")
        if isinstance(gpu_type, Mapping) and gpu_type.get("displayName"):
            return str(gpu_type["displayName"])
    return ""


def _pod_hourly_rate(payload: Mapping[str, Any]) -> float:
    for key in ("adjustedCostPerHr", "costPerHr"):
        value = payload.get(key)
        if value is not None:
            return float(value)
    raise RuntimeError("Runpod Pod response omitted hourly cost")


def launch(
    *,
    stage_receipt_path: Path,
    output: Path,
    execute: bool,
    runpodctl: Path = DEFAULT_RUNPODCTL,
    preregister_path: Path = DEFAULT_PREREGISTER,
) -> dict[str, Any]:
    preregister = validate_preregister(preregister_path)
    stage_receipt = _load_json(stage_receipt_path)
    command = build_pod_create_command(
        stage_receipt=stage_receipt,
        preregister=preregister,
        runpodctl=runpodctl,
    )
    receipt: dict[str, Any] = {
        "schema_version": "proposal-a-runpod-launch-receipt-v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "executed": execute,
        "command": command,
        "stage_receipt_sha256": _sha256_file(stage_receipt_path),
    }
    if execute:
        created = _runpodctl_json(runpodctl, tuple(command[1:]))
        pod_id = str(created.get("id") or created.get("pod", {}).get("id") or "")
        if not pod_id:
            raise RuntimeError("Runpod create response omitted Pod ID")
        try:
            details = _runpodctl_json(runpodctl, ("pod", "get", pod_id))
            gpu_name = _pod_gpu_name(details)
            rate = _pod_hourly_rate(details)
            maximum_rate = float(
                preregister["execution_contract"]["maximum_accepted_gpu_rate_dollars_per_hour"]
            )
            failures = []
            if "L40S" not in gpu_name.upper():
                failures.append(f"unexpected GPU {gpu_name!r}")
            if rate > maximum_rate:
                failures.append(f"hourly rate {rate} exceeds {maximum_rate}")
            if failures:
                raise RuntimeError("; ".join(failures))
        except BaseException:
            try:
                _delete_pod(runpodctl, pod_id)
            finally:
                raise
        network_volume = details.get("networkVolume")
        safe_pod = {
            key: details.get(key)
            for key in (
                "id",
                "name",
                "desiredStatus",
                "image",
                "machineId",
                "memoryInGb",
                "vcpuCount",
            )
        }
        if isinstance(network_volume, Mapping):
            safe_pod["networkVolume"] = {
                key: network_volume.get(key) for key in ("id", "name", "size", "dataCenterId")
            }
        receipt.update(
            {
                "pod_id": pod_id,
                "pod": safe_pod,
                "verified_gpu": gpu_name,
                "verified_hourly_rate": rate,
            }
        )
    _write_json_atomic(output, receipt)
    return receipt


def _read_s3_json(client: Any, *, volume_id: str, key: str) -> dict[str, Any] | None:
    from botocore.exceptions import ClientError

    try:
        response = client.get_object(Bucket=volume_id, Key=key)
    except ClientError as exc:
        if exc.response.get("Error", {}).get("Code") in {
            "404",
            "NoSuchKey",
            "NotFound",
        }:
            return None
        raise
    payload = json.loads(response["Body"].read())
    if not isinstance(payload, dict):
        raise RuntimeError(f"Runpod control object is not a JSON object: {key}")
    return payload


def monitor_once(
    *,
    launch_receipt: Mapping[str, Any],
    s3_endpoint: str,
    runpodctl: Path = DEFAULT_RUNPODCTL,
    now: datetime | None = None,
) -> dict[str, Any]:
    pod_id = str(launch_receipt["pod_id"])
    stage_path = Path(launch_receipt["stage_receipt_path"])
    stage_receipt = _load_json(stage_path)
    preregister = validate_preregister()
    prefix = stage_receipt["prefix"]
    client = _runpod_s3_client(s3_endpoint)
    heartbeat = _read_s3_json(
        client,
        volume_id=stage_receipt["volume_id"],
        key=f"{prefix}/run/heartbeat.json",
    )
    status = _read_s3_json(
        client,
        volume_id=stage_receipt["volume_id"],
        key=f"{prefix}/run/status.json",
    )
    observed_at = now or datetime.now(UTC)
    reasons: list[str] = []
    if heartbeat is not None:
        heartbeat_at = datetime.fromisoformat(str(heartbeat["created_utc"]).replace("Z", "+00:00"))
        stale_seconds = (observed_at - heartbeat_at).total_seconds()
        if stale_seconds > int(
            preregister["execution_contract"]["stale_heartbeat_terminate_seconds"]
        ):
            reasons.append(f"heartbeat stale for {stale_seconds:.1f}s")
    elif (
        observed_at
        - datetime.fromisoformat(str(launch_receipt["created_utc"]).replace("Z", "+00:00"))
    ).total_seconds() > 1_200:
        reasons.append("no heartbeat within the 20-minute startup grace")
    terminal = status is not None and status.get("state") in {"succeeded", "failed"}
    try:
        details = _runpodctl_json(runpodctl, ("pod", "get", pod_id))
    except subprocess.CalledProcessError:
        if not terminal:
            raise
        return {
            "schema_version": "proposal-a-runpod-monitor-observation-v1",
            "created_utc": observed_at.isoformat(),
            "pod_id": pod_id,
            "gpu": launch_receipt.get("verified_gpu"),
            "hourly_rate": launch_receipt.get("verified_hourly_rate"),
            "heartbeat": heartbeat,
            "status": status,
            "termination_reasons": [],
            "terminated": True,
            "provider_state": "pod_absent_after_terminal_worker_status",
        }
    gpu_name = _pod_gpu_name(details)
    try:
        rate = _pod_hourly_rate(details)
    except (TypeError, ValueError, RuntimeError):
        rate = None
        reasons.append("hourly rate unavailable")
    if "L40S" not in gpu_name.upper():
        reasons.append(f"unexpected GPU {gpu_name!r}")
    if rate is not None and rate > float(
        preregister["execution_contract"]["maximum_accepted_gpu_rate_dollars_per_hour"]
    ):
        reasons.append(f"hourly rate {rate} exceeds cap")
    if terminal and str(details.get("desiredStatus", "")).upper() == "RUNNING":
        reasons.append(f"worker is terminal: {status['state']}")
    terminated = False
    if reasons:
        _delete_pod(runpodctl, pod_id)
        terminated = True
    return {
        "schema_version": "proposal-a-runpod-monitor-observation-v1",
        "created_utc": observed_at.isoformat(),
        "pod_id": pod_id,
        "gpu": gpu_name,
        "hourly_rate": rate,
        "heartbeat": heartbeat,
        "status": status,
        "termination_reasons": reasons,
        "terminated": terminated,
    }


def monitor(
    *,
    launch_receipt_path: Path,
    s3_endpoint: str,
    output: Path,
    runpodctl: Path = DEFAULT_RUNPODCTL,
    interval_seconds: int = 60,
) -> dict[str, Any]:
    if not 30 <= interval_seconds <= 300:
        raise ValueError("Monitor interval must be in [30, 300] seconds")
    launch_receipt = _load_json(launch_receipt_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    observations = 0
    while True:
        observation = monitor_once(
            launch_receipt=launch_receipt,
            s3_endpoint=s3_endpoint,
            runpodctl=runpodctl,
        )
        with output.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(observation, allow_nan=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        observations += 1
        status = observation.get("status")
        terminal = isinstance(status, Mapping) and status.get("state") in {"succeeded", "failed"}
        if observation["terminated"] or terminal:
            return {
                "schema_version": "proposal-a-runpod-monitor-summary-v1",
                "observations": observations,
                "last_observation": observation,
            }
        time.sleep(interval_seconds)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("--preregister", type=Path, default=DEFAULT_PREREGISTER)

    stage_parser = subparsers.add_parser("stage")
    stage_parser.add_argument("--volume-id", required=True)
    stage_parser.add_argument("--s3-endpoint", required=True)
    stage_parser.add_argument("--output", type=Path, required=True)
    stage_parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    stage_parser.add_argument("--preregister", type=Path, default=DEFAULT_PREREGISTER)

    launch_parser = subparsers.add_parser("launch")
    launch_parser.add_argument("--stage-receipt", type=Path, required=True)
    launch_parser.add_argument("--output", type=Path, required=True)
    launch_parser.add_argument("--runpodctl", type=Path, default=DEFAULT_RUNPODCTL)
    launch_parser.add_argument("--execute", action="store_true")
    launch_parser.add_argument("--preregister", type=Path, default=DEFAULT_PREREGISTER)

    monitor_parser = subparsers.add_parser("monitor-once")
    monitor_parser.add_argument("--launch-receipt", type=Path, required=True)
    monitor_parser.add_argument("--s3-endpoint", required=True)
    monitor_parser.add_argument("--runpodctl", type=Path, default=DEFAULT_RUNPODCTL)
    monitor_parser.add_argument("--output", type=Path)

    continuous_parser = subparsers.add_parser("monitor")
    continuous_parser.add_argument("--launch-receipt", type=Path, required=True)
    continuous_parser.add_argument("--s3-endpoint", required=True)
    continuous_parser.add_argument("--runpodctl", type=Path, default=DEFAULT_RUNPODCTL)
    continuous_parser.add_argument("--output", type=Path, required=True)
    continuous_parser.add_argument("--interval-seconds", type=int, default=60)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.command == "validate":
        payload = validate_preregister(args.preregister)
        result = {
            "schema_version": payload["schema_version"],
            "preregister_sha256": _sha256_file(args.preregister),
            "status": "valid",
        }
    elif args.command == "stage":
        result = stage(
            volume_id=args.volume_id,
            s3_endpoint=args.s3_endpoint,
            output=args.output,
            preregister_path=args.preregister,
            prefix=args.prefix,
        )
    elif args.command == "launch":
        result = launch(
            stage_receipt_path=args.stage_receipt,
            output=args.output,
            execute=args.execute,
            runpodctl=args.runpodctl,
            preregister_path=args.preregister,
        )
        if args.execute:
            result["stage_receipt_path"] = str(args.stage_receipt.resolve())
            _write_json_atomic(args.output, result)
    elif args.command == "monitor-once":
        launch_receipt = _load_json(args.launch_receipt)
        result = monitor_once(
            launch_receipt=launch_receipt,
            s3_endpoint=args.s3_endpoint,
            runpodctl=args.runpodctl,
        )
        if args.output is not None:
            _write_json_atomic(args.output, result)
    elif args.command == "monitor":
        result = monitor(
            launch_receipt_path=args.launch_receipt,
            s3_endpoint=args.s3_endpoint,
            output=args.output,
            runpodctl=args.runpodctl,
            interval_seconds=args.interval_seconds,
        )
    else:
        raise AssertionError(args.command)
    print(json.dumps(result, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
