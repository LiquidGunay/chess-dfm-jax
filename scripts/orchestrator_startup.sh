#!/bin/bash
export HOME=/root
cd /root

# Install dependencies
apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y curl wget tar python3-pip python3-venv python-is-python3

# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="/root/.cargo/bin:$PATH"

# Download the code snapshot and sweep config
SOURCE_SNAPSHOT_URI="${SOURCE_SNAPSHOT_URI:-gs://your-us-central2-bucket/fixed_snapshots/verified_source_example.tar.gz}"
SWEEP_CONFIG_URI="${SWEEP_CONFIG_URI:-gs://your-us-central2-bucket/configs/hparam_sweep.jsonl}"

gcloud storage cp "$SOURCE_SNAPSHOT_URI" .
gcloud storage cp "$SWEEP_CONFIG_URI" .

ARCHIVE_NAME="$(basename "$SOURCE_SNAPSHOT_URI")"

mkdir chess-dfm-jax
tar -xzf "$ARCHIVE_NAME" -C chess-dfm-jax

cd chess-dfm-jax
mv ../hparam_sweep.jsonl .

# Create venv and install
uv venv .venv --python 3.11
uv pip install "jax[cpu]"
uv pip install -e .

export PYTHONPATH=$PYTHONPATH:$(pwd)

# Run the orchestrator
nohup .venv/bin/python -u scripts/sweep_manager.py --experiments hparam_sweep.jsonl > /root/sweep_hparam.log 2>&1 &
