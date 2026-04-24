#!/bin/bash
export HOME=/root
cd /root

# Install dependencies
apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y curl wget tar python3-pip python3-venv python-is-python3

# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="/root/.cargo/bin:$PATH"

# Download the code snapshot and sweep config
gcloud storage cp gs://gunay-chess-experiments-us-central2/fixed_snapshots/verified_source_v22_dfm.tar.gz .
gcloud storage cp gs://gunay-chess-experiments-us-central2/configs/hparam_sweep.jsonl .

mkdir lc0jax-human
tar -xzf verified_source_v22_dfm.tar.gz -C lc0jax-human

cd lc0jax-human
mv ../hparam_sweep.jsonl .

# Create venv and install
uv venv .venv --python 3.11
uv pip install "jax[cpu]"
uv pip install -e .

export PYTHONPATH=$PYTHONPATH:$(pwd)

# Run the orchestrator
nohup .venv/bin/python -u scripts/sweep_manager.py --experiments hparam_sweep.jsonl > /root/sweep_hparam.log 2>&1 &
