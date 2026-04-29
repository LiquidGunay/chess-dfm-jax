#!/usr/bin/env python3
"""Launch a small Compute Engine VM to run long CPU data-building commands."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chess_dfm_jax.training.tpu_jobs import create_source_snapshot, upload_file  # noqa: E402


def run(args: list[str]) -> None:
    subprocess.run(args, check=True)


def startup_script(*, source_uri: str, command: str, log_uri: str) -> str:
    return f"""#!/bin/bash
set -euo pipefail
export HOME=/root
LOG_FILE=/var/log/chess_dfm_jax-data-vm.log
exec > >(tee -a "$LOG_FILE") 2>&1

WORKDIR=/opt/chess_dfm_jax
SOURCE_URI={shlex.quote(source_uri)}
COMMAND={shlex.quote(command)}
export COMMAND
LOG_URI={shlex.quote(log_uri.rstrip('/'))}
GCLOUD="$(command -v gcloud || true)"
if [ -z "$GCLOUD" ] && [ -x /snap/google-cloud-cli/current/bin/gcloud ]; then
  GCLOUD=/snap/google-cloud-cli/current/bin/gcloud
fi
if [ -z "$GCLOUD" ]; then
  snap install google-cloud-cli --classic
  GCLOUD=/snap/google-cloud-cli/current/bin/gcloud
fi

cat <<JSON >/tmp/data_vm_status.json
{{"state":"booting","timestamp":"{time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}"}}
JSON
"$GCLOUD" storage cp /tmp/data_vm_status.json "$LOG_URI/status.json" || true

apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y python3 python3-venv python3-pip curl build-essential
python3 -m pip install --upgrade pip
python3 -m pip install uv

rm -rf "$WORKDIR"
mkdir -p "$WORKDIR"
"$GCLOUD" storage cp "$SOURCE_URI" /tmp/source.tar.gz
tar -xzf /tmp/source.tar.gz -C "$WORKDIR"
cd "$WORKDIR"
uv sync

cat <<JSON >/tmp/data_vm_status.json
{{"state":"running","timestamp":"{time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}","command":$(
python3 - <<'PY'
import json, os
print(json.dumps(os.environ["COMMAND"]))
PY
)}}
JSON
"$GCLOUD" storage cp /tmp/data_vm_status.json "$LOG_URI/status.json" || true

set +e
bash -lc "$COMMAND"
STATUS=$?
set -e

"$GCLOUD" storage cp "$LOG_FILE" "$LOG_URI/startup.log" || true
cat <<JSON >/tmp/data_vm_status.json
{{"state":"$([ "$STATUS" -eq 0 ] && echo completed || echo failed)","exit_code":$STATUS,"timestamp":"$(date -u +%Y-%m-%dT%H:%M:%SZ)"}}
JSON
"$GCLOUD" storage cp /tmp/data_vm_status.json "$LOG_URI/status.json" || true
exit "$STATUS"
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--project-id", required=True)
    parser.add_argument("--zone", default="us-central1-a")
    parser.add_argument("--machine-type", default="e2-standard-8")
    parser.add_argument("--boot-disk-size", default="200GB")
    parser.add_argument("--service-account", required=True)
    parser.add_argument("--bucket-uri", required=True)
    parser.add_argument("--command", required=True)
    parser.add_argument("--source-uri", default="", help="Optional pre-uploaded source tarball.")
    parser.add_argument("--allow-launch", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.allow_launch:
        raise SystemExit("Refusing to create a VM without --allow-launch.")
    bucket = args.bucket_uri.rstrip("/")
    stamp = time.strftime("%Y%m%d-%H%M%S")
    source_uri = args.source_uri
    if not source_uri:
        with tempfile.TemporaryDirectory(prefix="chess-dfm-source-") as tmp:
            archive = create_source_snapshot(ROOT, Path(tmp) / "source.tar.gz")
            source_uri = f"{bucket}/source_snapshots/data-vm/{args.name}-{stamp}.tar.gz"
            upload_file(archive, source_uri)
    log_uri = f"{bucket}/data_vm_logs/{args.name}-{stamp}"
    with tempfile.NamedTemporaryFile("w", suffix=".sh", delete=False, encoding="utf-8") as handle:
        handle.write(startup_script(source_uri=source_uri, command=args.command, log_uri=log_uri))
        startup_path = handle.name
    try:
        run(
            [
                "gcloud",
                "compute",
                "instances",
                "create",
                args.name,
                f"--project={args.project_id}",
                f"--zone={args.zone}",
                f"--machine-type={args.machine_type}",
                "--image-family=ubuntu-2204-lts",
                "--image-project=ubuntu-os-cloud",
                f"--boot-disk-size={args.boot_disk_size}",
                f"--service-account={args.service_account}",
                "--scopes=https://www.googleapis.com/auth/cloud-platform",
                f"--metadata-from-file=startup-script={startup_path}",
            ]
        )
    finally:
        Path(startup_path).unlink(missing_ok=True)
    print(json.dumps({"name": args.name, "zone": args.zone, "source_uri": source_uri, "log_uri": log_uri}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
