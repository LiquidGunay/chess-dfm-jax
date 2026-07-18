#!/usr/bin/env bash

# Source this file before every local-GPU research command:
#
#   source research/env.sh
#
# It intentionally keeps temporary files, package caches, compiler caches, and
# experiment output inside /mountpoint/.exp.

_research_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export CHESS_DFM_REPO_ROOT="$(cd -- "${_research_dir}/.." && pwd)"

case "${CHESS_DFM_REPO_ROOT}" in
  /mountpoint/.exp/*) ;;
  *)
    echo "Refusing to configure research paths outside /mountpoint/.exp: ${CHESS_DFM_REPO_ROOT}" >&2
    return 1 2>/dev/null || exit 1
    ;;
esac

export CHESS_DFM_LOCAL_ROOT="${CHESS_DFM_REPO_ROOT}/.local"
export CHESS_DFM_DATA_ROOT="${CHESS_DFM_REPO_ROOT}/data"
export CHESS_DFM_CHECKPOINT_ROOT="${CHESS_DFM_REPO_ROOT}/checkpoints"
export CHESS_DFM_ARTIFACT_ROOT="${CHESS_DFM_REPO_ROOT}/artifacts"

export TMPDIR="${CHESS_DFM_LOCAL_ROOT}/tmp"
export TEMP="${TMPDIR}"
export TMP="${TMPDIR}"

export XDG_CACHE_HOME="${CHESS_DFM_LOCAL_ROOT}/cache/xdg"
export UV_CACHE_DIR="${CHESS_DFM_LOCAL_ROOT}/cache/uv"
export PIP_CACHE_DIR="${CHESS_DFM_LOCAL_ROOT}/cache/pip"
export HF_HOME="${CHESS_DFM_LOCAL_ROOT}/cache/huggingface"
export JAX_COMPILATION_CACHE_DIR="${CHESS_DFM_LOCAL_ROOT}/cache/jax"
export CUDA_CACHE_PATH="${CHESS_DFM_LOCAL_ROOT}/cache/nvidia/ComputeCache"
export NUMBA_CACHE_DIR="${CHESS_DFM_LOCAL_ROOT}/cache/numba"
export MPLCONFIGDIR="${CHESS_DFM_LOCAL_ROOT}/cache/matplotlib"
export PYTHONPYCACHEPREFIX="${CHESS_DFM_LOCAL_ROOT}/cache/pycache"

export WANDB_DIR="${CHESS_DFM_ARTIFACT_ROOT}/wandb"
export WANDB_CACHE_DIR="${CHESS_DFM_LOCAL_ROOT}/cache/wandb"
export WANDB_CONFIG_DIR="${CHESS_DFM_LOCAL_ROOT}/config/wandb"
export WANDB_MODE="${WANDB_MODE:-offline}"

export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS="${JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS:-1}"
export JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES="${JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES:--1}"

mkdir -p \
  "${TMPDIR}" \
  "${XDG_CACHE_HOME}" \
  "${UV_CACHE_DIR}" \
  "${PIP_CACHE_DIR}" \
  "${HF_HOME}" \
  "${JAX_COMPILATION_CACHE_DIR}" \
  "${CUDA_CACHE_PATH}" \
  "${NUMBA_CACHE_DIR}" \
  "${MPLCONFIGDIR}" \
  "${PYTHONPYCACHEPREFIX}" \
  "${WANDB_DIR}" \
  "${WANDB_CACHE_DIR}" \
  "${WANDB_CONFIG_DIR}" \
  "${CHESS_DFM_DATA_ROOT}" \
  "${CHESS_DFM_CHECKPOINT_ROOT}/source" \
  "${CHESS_DFM_CHECKPOINT_ROOT}/recovered" \
  "${CHESS_DFM_ARTIFACT_ROOT}/baselines" \
  "${CHESS_DFM_ARTIFACT_ROOT}/profiles" \
  "${CHESS_DFM_ARTIFACT_ROOT}/arena" \
  "${CHESS_DFM_REPO_ROOT}/research/runs"

unset _research_dir
