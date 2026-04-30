"""Utilities for launching resumable Spot TPU JEPA jobs from a local controller."""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import tarfile
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from chess_dfm_jax.paths import project_root


def _import_gcp_clients():
    try:
        from google.cloud import storage
        from google.cloud import tpu_v2alpha1
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "google-cloud-storage and google-cloud-tpu are required for TPU job control."
        ) from exc
    return storage, tpu_v2alpha1


def _run_cli(args: list[str], *, input_text: str | None = None) -> str:
    proc = subprocess.run(
        args,
        input=input_text,
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"Command failed ({proc.returncode}): {' '.join(args)}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )
    return proc.stdout


def _has_adc() -> bool:
    credentials_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if credentials_path and Path(credentials_path).exists():
        return True
    adc_path = Path.home() / ".config" / "gcloud" / "application_default_credentials.json"
    return adc_path.exists()


def _queued_resource_id(name_or_id: str) -> str:
    if "/queuedResources/" in name_or_id:
        return name_or_id.rsplit("/", 1)[-1]
    return name_or_id


def _queued_resource_zone(name_or_id: str) -> str:
    if "/locations/" not in name_or_id:
        raise ValueError(f"Cannot infer zone from queued resource name: {name_or_id!r}")
    return name_or_id.split("/locations/", 1)[1].split("/", 1)[0]


def _queued_resource_project(name_or_id: str) -> str | None:
    if not name_or_id.startswith("projects/"):
        return None
    return name_or_id.split("/", 2)[1]


def parse_gs_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("gs://"):
        raise ValueError(f"Expected gs:// URI, got {uri!r}")
    body = uri[5:]
    if "/" not in body:
        return body, ""
    bucket, blob = body.split("/", 1)
    return bucket, blob


def zone_region(zone: str) -> str:
    return zone.rsplit("-", 1)[0]


STALE_SOURCE_MARKERS = (
    "verified_source_v27_dfm.tar.gz",
    "lc0jax-human",
    "your-bucket",
    "example",
)


def validate_launch_spec(spec: "TPUJobSpec", *, override_source_uri: str | None = None) -> None:
    """Fail closed before creating TPU queued resources."""
    if not spec.project_id or spec.project_id == "your-gcp-project":
        raise ValueError("project_id must be set to a real GCP project before launch.")
    if not spec.run_id:
        raise ValueError("run_id must be set before launch.")
    if not spec.zone_order:
        raise ValueError("zone_order must include at least one TPU zone before launch.")
    if not spec.service_account or "your-gcp-project" in spec.service_account:
        raise ValueError("service_account must be set to a real service account before launch.")
    for region, bucket in spec.bucket_by_region.items():
        if not bucket.startswith("gs://") or "your-bucket" in bucket:
            raise ValueError(f"bucket_by_region[{region!r}] must be a real gs:// bucket URI.")
    if override_source_uri:
        if not override_source_uri.startswith("gs://"):
            raise ValueError("override_source_uri must be a gs:// URI.")
        if any(marker in override_source_uri for marker in STALE_SOURCE_MARKERS):
            raise ValueError(f"Refusing stale or placeholder source snapshot: {override_source_uri}")
    if spec.entry_command and "verified_source_v27_dfm" in spec.entry_command:
        raise ValueError("Refusing stale entry_command that references verified_source_v27_dfm.")
    if "WANDB_API_KEY" in spec.env:
        raise ValueError(
            "Do not put WANDB_API_KEY in TPUJobSpec.env; startup metadata is visible in TPU metadata. "
            "Use TPUJobSpec.secret_env or an SSH-scoped env file instead."
        )
    if spec.cache_disk_mode not in {"read-write", "read-only"}:
        raise ValueError("cache_disk_mode must be either 'read-write' or 'read-only'.")


@dataclass
class TPUJobSpec:
    project_id: str
    run_id: str
    zone_order: list[str]
    bucket_by_region: dict[str, str]
    run_name: str | None = None
    run_family: str = "jepa"
    wandb_entity: str | None = None
    wandb_project: str = "chess_dfm_jax-jepa"
    wandb_group: str = "bt4-token-jepa"
    wandb_run_id: str | None = None
    accelerator_type: str = "v5litepod-8"
    runtime_version: str = "tpu-ubuntu2204-base"
    service_account: str | None = None
    network: str | None = None
    subnetwork_by_zone: dict[str, str] = field(default_factory=dict)
    enable_external_ips: bool = False
    autocheckpoint_enabled: bool = True
    allocation_timeout_s: int = 1800
    poll_interval_s: int = 30
    workdir: str = "/tmp/chess_dfm_jax"
    cache_disk_by_zone: dict[str, str] = field(default_factory=dict)
    cache_disk_mode: str = "read-write"
    cache_mount_point: str = "/mnt/chess-dfm-cache"
    models_uri: str | None = None
    models_uri_by_region: dict[str, str] = field(default_factory=dict)
    chunk_data_uri: str | None = None
    chunk_data_uri_by_region: dict[str, str] = field(default_factory=dict)
    entry_command: str | None = None
    train_args: dict[str, Any] = field(default_factory=dict)
    env: dict[str, str] = field(default_factory=dict)
    secret_env: dict[str, str] = field(default_factory=dict)
    spot: bool = True

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TPUJobSpec":
        return cls(**payload)

    @classmethod
    def from_json(cls, path: str | Path) -> "TPUJobSpec":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def effective_run_name(self) -> str:
        return self.run_name or self.run_id

    def bucket_for_zone(self, zone: str) -> str:
        region = zone_region(zone)
        if region not in self.bucket_by_region:
            raise KeyError(f"No bucket configured for region {region!r}.")
        return self.bucket_by_region[region].rstrip("/")

    def run_root_uri(self, zone: str) -> str:
        return f"{self.bucket_for_zone(zone)}/runs/{self.run_family}/{self.run_id}"

    def checkpoint_uri(self, zone: str) -> str:
        return f"{self.run_root_uri(zone)}/checkpoints"

    def status_uri(self, zone: str) -> str:
        return f"{self.run_root_uri(zone)}/status.json"

    def source_uri(self, zone: str, stamp: str) -> str:
        return f"{self.bucket_for_zone(zone)}/source_snapshots/{self.run_id}/{stamp}.tar.gz"

    def models_uri_for_zone(self, zone: str) -> str | None:
        region = zone_region(zone)
        return self.models_uri_by_region.get(region, self.models_uri)

    def chunk_data_uri_for_zone(self, zone: str) -> str | None:
        region = zone_region(zone)
        return self.chunk_data_uri_by_region.get(region, self.chunk_data_uri)

    def cache_disk_for_zone(self, zone: str) -> str | None:
        return self.cache_disk_by_zone.get(zone)

    def cache_disk_source_for_zone(self, zone: str) -> str | None:
        disk = self.cache_disk_for_zone(zone)
        if not disk:
            return None
        if disk.startswith("projects/"):
            return disk
        return f"projects/{self.project_id}/zones/{zone}/disks/{disk}"


def create_source_snapshot(repo_root: str | Path, output_path: str | Path) -> Path:
    root = Path(repo_root).resolve()
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    excluded_top_level = {
        ".git",
        ".venv",
        "data",
        "models",
        "runs",
        "wandb",
        "artifacts",
        ".pytest_cache",
        ".ruff_cache",
    }
    excluded_anywhere = {"__pycache__"}
    with tarfile.open(out_path, "w:gz") as archive:
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            rel = path.relative_to(root)
            if rel.parts[0] in excluded_top_level:
                continue
            if set(rel.parts) & excluded_anywhere:
                continue
            if rel.name == ".env" or rel.name.endswith(".local.json"):
                continue
            archive.add(path, arcname=str(rel))
    return out_path


def upload_file(local_path: str | Path, gs_uri: str) -> str:
    if _has_adc():
        storage, _ = _import_gcp_clients()
        client = storage.Client()
        bucket_name, blob_name = parse_gs_uri(gs_uri)
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        blob.upload_from_filename(str(local_path))
        return gs_uri
    _run_cli(["gcloud", "storage", "cp", str(local_path), gs_uri])
    return gs_uri


def upload_json(payload: dict[str, Any], gs_uri: str) -> str:
    text = json.dumps(payload, indent=2)
    if _has_adc():
        storage, _ = _import_gcp_clients()
        client = storage.Client()
        bucket_name, blob_name = parse_gs_uri(gs_uri)
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        blob.upload_from_string(text, content_type="application/json")
        return gs_uri
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as handle:
        handle.write(text)
        tmp_name = handle.name
    try:
        _run_cli(["gcloud", "storage", "cp", tmp_name, gs_uri])
    finally:
        Path(tmp_name).unlink(missing_ok=True)
    return gs_uri


def read_json(gs_uri: str) -> dict[str, Any] | None:
    if _has_adc():
        storage, _ = _import_gcp_clients()
        client = storage.Client()
        bucket_name, blob_name = parse_gs_uri(gs_uri)
        blob = client.bucket(bucket_name).blob(blob_name)
        if not blob.exists():
            return None
        return json.loads(blob.download_as_text())
    proc = subprocess.run(
        ["gcloud", "storage", "cat", gs_uri],
        text=True,
        capture_output=True,
        check=False,
    )
    stderr = proc.stderr or ""
    if proc.returncode != 0:
        missing_markers = [
            "matched no objects",
            "No URLs matched",
            "One or more URLs matched no objects",
            "No such object",
            "not found",
            "403",
        ]
        if any(marker in stderr for marker in missing_markers):
            return None
        raise RuntimeError(
            f"Command failed ({proc.returncode}): gcloud storage cat {gs_uri}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{stderr}"
        )
    return json.loads(proc.stdout)


def render_train_command(spec: TPUJobSpec, zone: str) -> str:
    args = {
        "run-id": spec.run_id,
        "wandb-project": spec.wandb_project,
        "wandb-group": spec.wandb_group,
        "checkpoint-uri": spec.checkpoint_uri(zone),
        "resume": True,
        **spec.train_args,
    }
    if spec.wandb_entity:
        args["wandb-entity"] = spec.wandb_entity
    parts = ["python3", "scripts/train_jepa.py"]
    for key, value in args.items():
        flag = f"--{key}"
        if isinstance(value, bool):
            if value:
                parts.append(flag)
            continue
        parts.extend([flag, str(value)])
    return " ".join(shlex.quote(part) for part in parts)


def render_entry_command(spec: TPUJobSpec, zone: str) -> str:
    if spec.entry_command:
        # Older sweep specs used the pre-rename workdir. Keep those specs usable
        # while the startup script downloads chunks into the canonical workdir.
        command = spec.entry_command.replace("/tmp/lc0jaxhuman/chunks", f"{spec.workdir}/chunks")
    else:
        command = render_train_command(spec, zone)
    if spec.cache_disk_for_zone(zone):
        command = command.replace(
            f"{spec.workdir}/gcs_cache",
            f"{spec.cache_mount_point.rstrip('/')}/gcs_cache",
        )
    return command


def render_cache_disk_mount_script(spec: TPUJobSpec, zone: str) -> str:
    if not spec.cache_disk_for_zone(zone):
        return ""
    mount_point = spec.cache_mount_point.rstrip("/")
    return f"""
CACHE_MOUNT={shlex.quote(mount_point)}
CACHE_DEVICE=""
for candidate in /dev/disk/by-id/google-persistent-disk-1 /dev/disk/by-id/scsi-0Google_PersistentDisk_persistent-disk-1; do
  if [ -e "$candidate" ]; then
    CACHE_DEVICE="$candidate"
    break
  fi
done
if [ -z "$CACHE_DEVICE" ]; then
  echo "Cache disk configured but no data-disk device was found." >&2
  ls -l /dev/disk/by-id || true
  exit 2
fi
if ! blkid "$CACHE_DEVICE" >/dev/null 2>&1; then
  mkfs.ext4 -F -m 0 "$CACHE_DEVICE"
fi
mkdir -p "$CACHE_MOUNT"
if ! mountpoint -q "$CACHE_MOUNT"; then
  mount -o discard,defaults "$CACHE_DEVICE" "$CACHE_MOUNT"
fi
chmod 777 "$CACHE_MOUNT"
mkdir -p "$CACHE_MOUNT/gcs_cache/train" "$CACHE_MOUNT/gcs_cache/val"
df -h "$CACHE_MOUNT"
"""


def render_startup_script(spec: TPUJobSpec, zone: str, source_uri: str) -> str:
    env_vars = dict(spec.env)
    if "WANDB_API_KEY" in env_vars:
        raise ValueError(
            "Do not put WANDB_API_KEY in TPUJobSpec.env; startup metadata is visible in TPU metadata. "
            "Use a local SSH-scoped env file or Secret Manager instead."
        )
    env_exports = "\n".join(
        f"export {name}={shlex.quote(value)}" for name, value in sorted(env_vars.items())
    )
    secret_exports = "\n".join(
        (
            f"export {name}="
            f"$(/snap/google-cloud-cli/current/bin/gcloud secrets versions access latest "
            f"--secret={shlex.quote(secret_name)})"
        )
        for name, secret_name in sorted(spec.secret_env.items())
    )
    cache_disk_mount = render_cache_disk_mount_script(spec, zone)
    model_sync = ""
    models_uri = spec.models_uri_for_zone(zone)
    if models_uri:
        model_sync = (
            f'mkdir -p "$WORKDIR/repo/models"\n'
            f'/snap/google-cloud-cli/current/bin/gcloud storage cp --recursive {shlex.quote(models_uri.rstrip("/") + "/*")} "$WORKDIR/repo/models/"\n'
        )
    chunk_sync = ""
    chunk_data_uri = spec.chunk_data_uri_for_zone(zone)
    if chunk_data_uri:
        chunk_sync = (
            f'mkdir -p "$WORKDIR/chunks"\n'
            f'/snap/google-cloud-cli/current/bin/gcloud storage cp --recursive {shlex.quote(chunk_data_uri.rstrip("/") + "/*")} "$WORKDIR/chunks/"\n'
            f'mkdir -p /tmp/lc0jaxhuman\n'
            f'ln -sfn "$WORKDIR/chunks" /tmp/lc0jaxhuman/chunks\n'
        )
    entry_cmd = render_entry_command(spec, zone)
    status_uri = spec.status_uri(zone)
    artifacts_uri = f"{spec.run_root_uri(zone)}/artifacts"
    return f"""#!/bin/bash
set -euo pipefail
export HOME=/root
LOG_FILE=/var/log/chess_dfm_jax-startup.log
exec > >(tee -a "$LOG_FILE") 2>&1

WORKDIR={shlex.quote(spec.workdir)}
SOURCE_URI={shlex.quote(source_uri)}
STATUS_URI={shlex.quote(status_uri)}
ARTIFACTS_URI={shlex.quote(artifacts_uri)}
mkdir -p "$WORKDIR"

cat <<'JSON' >/tmp/chess_dfm_jax_status.json
{{"state": "booting", "run_id": "{spec.run_id}", "zone": "{zone}", "timestamp": "{time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}"}}
JSON
/snap/google-cloud-cli/current/bin/gcloud storage cp /tmp/chess_dfm_jax_status.json "$STATUS_URI"

/snap/google-cloud-cli/current/bin/gcloud storage cp "$SOURCE_URI" "$WORKDIR/source.tar.gz"
rm -rf "$WORKDIR/repo"
mkdir -p "$WORKDIR/repo"
tar -xzf "$WORKDIR/source.tar.gz" -C "$WORKDIR/repo"

cd "$WORKDIR/repo"
mkdir -p "$WORKDIR/artifacts"
export DEBIAN_FRONTEND=noninteractive
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="/root/.cargo/bin:/root/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/snap/bin:$PATH"
uv venv /tmp/venv --python 3.11
uv pip install --python /tmp/venv "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
uv pip install --python /tmp/venv -e .
{env_exports}
{secret_exports}
{cache_disk_mount}
{model_sync}{chunk_sync}
cat <<'JSON' >/tmp/chess_dfm_jax_status.json
{{"state": "running", "run_id": "{spec.run_id}", "zone": "{zone}"}}
JSON
/snap/google-cloud-cli/current/bin/gcloud storage cp /tmp/chess_dfm_jax_status.json "$STATUS_URI"

set +e
{entry_cmd}
EXIT_CODE=$?
set -e

if [ -d "$WORKDIR/artifacts" ]; then
  /snap/google-cloud-cli/current/bin/gcloud storage cp --recursive "$WORKDIR/artifacts" "$ARTIFACTS_URI" || true
fi

/snap/google-cloud-cli/current/bin/gcloud storage cp "$LOG_FILE" "$ARTIFACTS_URI/startup.log" || true

cat <<JSON >/tmp/chess_dfm_jax_status.json
{{"state": "$( [ "$EXIT_CODE" -eq 0 ] && echo completed || echo failed )", "exit_code": $EXIT_CODE, "run_id": "{spec.run_id}", "zone": "{zone}"}}
JSON
/snap/google-cloud-cli/current/bin/gcloud storage cp /tmp/chess_dfm_jax_status.json "$STATUS_URI"
exit "$EXIT_CODE"
"""


def request_spot_tpu(spec: TPUJobSpec, zone: str, startup_script: str, attempt: int) -> str:
    queued_resource_id = f"{spec.run_id}-{attempt:03d}"
    node_id = f"{spec.run_id}-{attempt:03d}"
    with tempfile.NamedTemporaryFile("w", suffix=".sh", delete=False, encoding="utf-8") as handle:
        handle.write(startup_script)
        startup_path = handle.name
    try:
        cmd = [
            "gcloud",
            "compute",
            "tpus",
            "queued-resources",
            "create",
            queued_resource_id,
            f"--project={spec.project_id}",
            f"--zone={zone}",
            f"--accelerator-type={spec.accelerator_type}",
            f"--runtime-version={spec.runtime_version}",
            f"--node-id={node_id}",
        ]
        if spec.spot:
            cmd.append("--spot")
        cmd.extend([
            f"--metadata-from-file=startup-script={startup_path}",
            f"--labels=run_id={spec.run_id},controller=chess_dfm_jax",
        ])
        if spec.service_account:
            cmd.append(f"--service-account={spec.service_account}")
        if spec.network:
            cmd.append(f"--network={spec.network}")
        if zone in spec.subnetwork_by_zone:
            cmd.append(f"--subnetwork={spec.subnetwork_by_zone[zone]}")
        if not spec.enable_external_ips:
            cmd.append("--internal-ips")
        cache_disk_source = spec.cache_disk_source_for_zone(zone)
        if cache_disk_source:
            cmd.append(f"--data-disk=source={cache_disk_source},mode={spec.cache_disk_mode}")
        _run_cli(cmd)
    finally:
        Path(startup_path).unlink(missing_ok=True)
    return f"projects/{spec.project_id}/locations/{zone}/queuedResources/{queued_resource_id}"


def get_queued_resource(name: str):
    zone = _queued_resource_zone(name)
    queued_resource_id = _queued_resource_id(name)
    cmd = [
        "gcloud",
        "compute",
        "tpus",
        "queued-resources",
        "describe",
        queued_resource_id,
        f"--zone={zone}",
        "--format=json",
    ]
    project = _queued_resource_project(name)
    if project:
        cmd.append(f"--project={project}")
    try:
        output = _run_cli(cmd)
    except RuntimeError as exc:
        if "NOT_FOUND" in str(exc):
            return {"state": "CREATING"}
        raise
    return json.loads(output)


def delete_queued_resource(name: str) -> None:
    zone = _queued_resource_zone(name)
    queued_resource_id = _queued_resource_id(name)
    cmd = [
        "gcloud",
        "compute",
        "tpus",
        "queued-resources",
        "delete",
        queued_resource_id,
        f"--zone={zone}",
        "--quiet",
    ]
    project = _queued_resource_project(name)
    if project:
        cmd.append(f"--project={project}")
    
    for attempt in range(5):
        try:
            _run_cli(cmd)
            break
        except RuntimeError as exc:
            if "DeleteQueuedResource is not supported when state is PROVISIONING" in str(exc) and attempt < 4:
                time.sleep(10)
            else:
                force_cmd = [
                    "gcloud",
                    "compute",
                    "tpus",
                    "queued-resources",
                    "delete",
                    queued_resource_id,
                    f"--zone={zone}",
                    "--quiet",
                    "--force",
                ]
                if project:
                    force_cmd.append(f"--project={project}")
                try:
                    _run_cli(force_cmd)
                except Exception:
                    pass
                break


def queued_resource_state_name(resource) -> str:
    if isinstance(resource, dict):
        state = resource.get("state")
        if isinstance(state, dict):
            return state.get("state") or state.get("name") or str(state)
        return str(state)
    state = getattr(resource, "state", None)
    return getattr(state, "name", str(state))


def run_spot_controller(spec: TPUJobSpec, *, repo_root: str | Path | None = None, override_source_uri: str | None = None) -> dict[str, Any]:
    print("Starting run_spot_controller")
    validate_launch_spec(spec, override_source_uri=override_source_uri)
    root = Path(repo_root) if repo_root is not None else project_root()
    for zone in spec.zone_order:
        print(f"Checking status for zone: {zone}")
        status = read_json(spec.status_uri(zone))
        print(f"Status for {zone}: {status}")
        if status and status.get("state") == "completed":
            return {"status": "already_completed", "zone": zone, "details": status}

    attempt = 0
    while True:
        request_failures: list[dict[str, str]] = []
        for zone in spec.zone_order:
            attempt += 1
            stamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
            
            if override_source_uri:
                source_uri = override_source_uri
            else:
                with tempfile.TemporaryDirectory(prefix="chess_dfm_jax_tpu_") as tmp_dir:
                    archive_path = create_source_snapshot(root, Path(tmp_dir) / "source.tar.gz")
                    source_uri = spec.source_uri(zone, stamp)
                    upload_file(archive_path, source_uri)
            
            upload_json(spec.to_dict(), f"{spec.run_root_uri(zone)}/job_spec.json")
            startup_script = render_startup_script(spec, zone, source_uri)
            queued_resource_id = f"{spec.run_id}-{attempt:03d}"
            print(f"Requesting spot TPU {queued_resource_id} in {zone}...")
            try:
                resource_name = request_spot_tpu(spec, zone, startup_script, attempt)
            except RuntimeError as exc:
                print(f"Failed to request spot TPU in {zone}: {exc}")
                request_failures.append({"zone": zone, "error": str(exc)})
                if (
                    "does not have permission" in str(exc)
                    or "permission" in str(exc).lower()
                    or "Insufficient capacity" in str(exc)
                ):
                    continue
                if len(spec.zone_order) == 1:
                    return {"status": "request_failed", "zone": zone, "error": str(exc)}
                time.sleep(30)
                continue
            
            print(f"Created queued resource: {resource_name}")
            started = time.monotonic()
            last_state = None
            last_status_state = None
            while True:
                resource = get_queued_resource(resource_name)
                state_name = queued_resource_state_name(resource)
                status = read_json(spec.status_uri(zone))
                
                if state_name != last_state:
                    print(f"Resource {resource_name} state: {state_name}")
                    last_state = state_name
                
                status_state = status.get("state") if status else None
                if status_state != last_status_state:
                    print(f"Job status in {zone}: {status_state}")
                    last_status_state = status_state

                if status and status.get("state") == "completed":
                    print(f"Job completed successfully in {zone}.")
                    delete_queued_resource(resource_name)
                    return {
                        "status": "completed",
                        "zone": zone,
                        "resource": resource_name,
                        "details": status,
                    }
                if state_name in {"FAILED", "SUSPENDED"}:
                    print(f"Resource {resource_name} failed or suspended (state={state_name}). Deleting and retrying...")
                    delete_queued_resource(resource_name)
                    return {"status": "preempted", "zone": zone}
                if status and status.get("state") == "failed":
                    print(f"Job failed on resource {resource_name} (state={state_name}). Deleting and retrying...")
                    delete_queued_resource(resource_name)
                    return {"status": "job_failed", "zone": zone}
                if state_name in {"ACCEPTED", "WAITING_FOR_RESOURCES", "CREATING"}:
                    if time.monotonic() - started > spec.allocation_timeout_s:
                        print(f"Resource {resource_name} timed out in {state_name}. Deleting and retrying...")
                        delete_queued_resource(resource_name)
                        return {"status": "timeout", "zone": zone}
                time.sleep(spec.poll_interval_s)
        if request_failures:
            return {"status": "request_failed", "failures": request_failures}


__all__ = [
    "TPUJobSpec",
    "create_source_snapshot",
    "parse_gs_uri",
    "read_json",
    "render_startup_script",
    "render_entry_command",
    "run_spot_controller",
    "upload_file",
    "upload_json",
    "validate_launch_spec",
    "zone_region",
]
