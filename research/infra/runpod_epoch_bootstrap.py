"""Standalone, self-terminating Runpod bootstrap for the Proposal A epoch.

This file is uploaded to the network volume before a GPU Pod is created.  It
uses only the Python standard library until the exact pushed repository commit
has been cloned and its Torch-only dependencies installed.
"""

from __future__ import annotations

from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
from typing import Any, Mapping, Sequence
import urllib.request


_STAGE = "bootstrap_start"
_STAGE_LOCK = threading.Lock()


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial-{os.getpid()}")
    temporary.write_bytes(_json_bytes(payload))
    os.replace(temporary, path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _set_stage(value: str) -> None:
    global _STAGE
    with _STAGE_LOCK:
        _STAGE = value


def _get_stage() -> str:
    with _STAGE_LOCK:
        return _STAGE


def _run_stream(
    command: Sequence[str],
    *,
    cwd: Path | None = None,
    env: Mapping[str, str] | None = None,
    log_path: Path,
) -> None:
    printable = " ".join(command)
    print(json.dumps({"bootstrap_command": printable}), flush=True)
    with log_path.open("a", encoding="utf-8") as log:
        process = subprocess.Popen(
            tuple(command),
            cwd=cwd,
            env=None if env is None else dict(env),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log.write(line)
            log.flush()
        returncode = process.wait()
    if returncode != 0:
        raise subprocess.CalledProcessError(returncode, tuple(command))


def _heartbeat_loop(
    *,
    heartbeat_path: Path,
    status_path: Path,
    stop: threading.Event,
    interval_seconds: int,
) -> None:
    while not stop.is_set():
        status = None
        if status_path.is_file():
            try:
                status = json.loads(status_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                status = None
        _atomic_json(
            heartbeat_path,
            {
                "schema_version": "proposal-a-runpod-heartbeat-v1",
                "created_utc": datetime.now(UTC).isoformat(),
                "pod_id": os.environ.get("RUNPOD_POD_ID"),
                "stage": _get_stage(),
                "worker_status": status,
            },
        )
        stop.wait(interval_seconds)


def _delete_self() -> None:
    pod_id = os.environ.get("RUNPOD_POD_ID")
    api_key = os.environ.get("RUNPOD_API_KEY")
    if not pod_id or not api_key:
        print(
            json.dumps(
                {
                    "self_termination": "unavailable",
                    "reason": "Runpod Pod ID or Pod-scoped API key missing",
                }
            ),
            flush=True,
        )
        return
    request = urllib.request.Request(
        f"https://rest.runpod.io/v1/pods/{pod_id}",
        method="DELETE",
        headers={"Authorization": f"Bearer {api_key}"},
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            response.read()
        print(json.dumps({"self_termination": "requested", "pod_id": pod_id}), flush=True)
    except Exception as exc:  # The absolute provider deadline remains the backstop.
        print(
            json.dumps(
                {
                    "self_termination": "failed",
                    "error_type": type(exc).__name__,
                }
            ),
            flush=True,
        )


def _load_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "proposal-a-runpod-launch-v1":
        raise ValueError("Unsupported Runpod launch manifest")
    preregister = payload.get("preregister")
    if not isinstance(preregister, dict):
        raise ValueError("Launch manifest omitted the preregistration")
    preregister_sha = hashlib.sha256(_json_bytes(preregister)).hexdigest()
    if preregister_sha != payload.get("preregister_sha256"):
        raise ValueError("Embedded preregistration checksum drift")
    return payload


def _verify_volume_binding(manifest: Mapping[str, Any]) -> None:
    expected = str(manifest["network_volume"]["id"])
    observed = os.environ.get("RUNPOD_VOLUME_ID")
    if observed != expected:
        raise RuntimeError(f"Network volume mismatch: {observed!r} != {expected!r}")
    if os.environ.get("RUNPOD_GPU_COUNT") not in {None, "1"}:
        raise RuntimeError("Proposal A epoch requires exactly one GPU")


def _ensure_git(log_path: Path) -> None:
    if shutil.which("git") is not None:
        return
    _set_stage("install_git")
    _run_stream(("apt-get", "update"), log_path=log_path)
    _run_stream(
        ("apt-get", "install", "--yes", "--no-install-recommends", "git"),
        log_path=log_path,
    )


def _clone_exact_commit(
    manifest: Mapping[str, Any],
    *,
    repository: Path,
    log_path: Path,
) -> None:
    git = manifest["git"]
    _set_stage("clone_repository")
    if repository.exists():
        shutil.rmtree(repository)
    _run_stream(
        (
            "git",
            "clone",
            "--single-branch",
            "--branch",
            str(git["branch"]),
            str(git["remote"]),
            str(repository),
        ),
        log_path=log_path,
    )
    _run_stream(
        ("git", "checkout", "--detach", str(git["commit"])),
        cwd=repository,
        log_path=log_path,
    )
    commit = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if commit != git["commit"]:
        raise RuntimeError("Cloned Git commit does not match the launch manifest")
    dirty = subprocess.run(
        ("git", "status", "--porcelain"),
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if dirty:
        raise RuntimeError("Cloned repository is unexpectedly dirty")


def _install_environment(
    *,
    repository: Path,
    log_path: Path,
) -> None:
    _set_stage("install_torch_only_environment")
    _run_stream(
        (
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-cache-dir",
            "--requirement",
            str(repository / "research" / "requirements_torch_local.txt"),
        ),
        log_path=log_path,
    )
    _run_stream(
        (
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--editable",
            str(repository),
        ),
        log_path=log_path,
    )


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit("usage: runpod_epoch_bootstrap.py LAUNCH_MANIFEST")
    manifest_path = Path(sys.argv[1]).resolve()
    manifest = _load_manifest(manifest_path)
    _verify_volume_binding(manifest)
    run_root = Path(manifest["paths"]["run_root"])
    control_root = Path(manifest["paths"]["control"])
    run_root.mkdir(parents=True, exist_ok=True)
    control_root.mkdir(parents=True, exist_ok=True)
    log_path = run_root / "bootstrap.log"
    status_path = run_root / "status.json"
    heartbeat_path = run_root / "heartbeat.json"
    heartbeat_stop = threading.Event()
    heartbeat = threading.Thread(
        target=_heartbeat_loop,
        kwargs={
            "heartbeat_path": heartbeat_path,
            "status_path": status_path,
            "stop": heartbeat_stop,
            "interval_seconds": int(
                manifest["preregister"]["execution_contract"]["heartbeat_interval_seconds"]
            ),
        },
        name="proposal-a-runpod-heartbeat",
        daemon=True,
    )
    heartbeat.start()
    returncode = 1
    try:
        bootstrap_record = manifest["objects"]["bootstrap"]
        if Path(__file__).stat().st_size != int(bootstrap_record["size_bytes"]):
            raise RuntimeError("Bootstrap size drift")
        if _sha256_file(Path(__file__)) != bootstrap_record["sha256"]:
            raise RuntimeError("Bootstrap checksum drift")
        _atomic_json(
            status_path,
            {
                "schema_version": "proposal-a-runpod-worker-status-v1",
                "created_utc": datetime.now(UTC).isoformat(),
                "state": "bootstrapping",
                "stage": _get_stage(),
                "pod_id": os.environ.get("RUNPOD_POD_ID"),
            },
        )
        repository = Path("/opt/chess-dfm-jax")
        _ensure_git(log_path)
        _clone_exact_commit(manifest, repository=repository, log_path=log_path)
        _install_environment(repository=repository, log_path=log_path)
        environment = dict(os.environ)
        environment.update(
            {
                "CHESS_DFM_TRUSTED_WORKSPACE_ROOTS": "/workspace:/opt",
                "CUDA_CACHE_PATH": "/opt/chess-dfm-cache/cuda",
                "TORCHINDUCTOR_CACHE_DIR": "/opt/chess-dfm-cache/torchinductor",
                "XDG_CACHE_HOME": "/opt/chess-dfm-cache/xdg",
                "OMP_NUM_THREADS": "2",
                "PYTHONPATH": str(repository),
            }
        )
        _set_stage("epoch_worker")
        _run_stream(
            (
                sys.executable,
                str(repository / "research" / "infra" / "runpod_proposal_a_worker.py"),
                "--manifest",
                str(manifest_path),
            ),
            cwd=repository,
            env=environment,
            log_path=log_path,
        )
        returncode = 0
    except BaseException as exc:
        existing_status = None
        if status_path.is_file():
            try:
                existing_status = json.loads(status_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                existing_status = None
        if not isinstance(existing_status, dict) or existing_status.get("state") not in {
            "succeeded",
            "failed",
        }:
            _atomic_json(
                status_path,
                {
                    "schema_version": "proposal-a-runpod-worker-status-v1",
                    "created_utc": datetime.now(UTC).isoformat(),
                    "state": "failed",
                    "stage": _get_stage(),
                    "error_type": type(exc).__name__,
                    "pod_id": os.environ.get("RUNPOD_POD_ID"),
                },
            )
        print(
            json.dumps(
                {
                    "bootstrap_failed": True,
                    "error_type": type(exc).__name__,
                    "stage": _get_stage(),
                }
            ),
            flush=True,
        )
    finally:
        _set_stage("terminal_sync")
        heartbeat_stop.set()
        heartbeat.join(timeout=5)
        os.sync()
        _delete_self()
    return returncode


if __name__ == "__main__":
    raise SystemExit(main())
