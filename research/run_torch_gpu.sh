#!/usr/bin/env bash
set -euo pipefail

_torch_research_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
_torch_repo_root="$(cd -- "${_torch_research_dir}/.." && pwd)"
_torch_venv_python="${_torch_repo_root}/.venv/bin/python"

if [[ ! -x "${_torch_venv_python}" ]]; then
  echo "Missing Torch environment: ${_torch_venv_python}" >&2
  echo "Follow the Torch experiment quickstart in README.md." >&2
  exit 2
fi

export CHESS_DFM_REPO_ROOT="${_torch_repo_root}"
export CHESS_DFM_WORKSPACE_ROOT="${_torch_repo_root}"
export CHESS_DFM_LOCAL_ROOT="${_torch_repo_root}/.local"
export CHESS_DFM_DATA_ROOT="${_torch_repo_root}/data"
export CHESS_DFM_MODEL_ROOT="${_torch_repo_root}/models"
export CHESS_DFM_CHECKPOINT_ROOT="${_torch_repo_root}/checkpoints"
export CHESS_DFM_ARTIFACT_ROOT="${_torch_repo_root}/artifacts"

export TMPDIR="${CHESS_DFM_LOCAL_ROOT}/tmp"
export TEMP="${TMPDIR}"
export TMP="${TMPDIR}"
export XDG_CACHE_HOME="${CHESS_DFM_LOCAL_ROOT}/cache/xdg"
export XDG_CONFIG_HOME="${CHESS_DFM_LOCAL_ROOT}/config/xdg"
export XDG_DATA_HOME="${CHESS_DFM_LOCAL_ROOT}/data/xdg"
export UV_CACHE_DIR="${CHESS_DFM_LOCAL_ROOT}/cache/uv"
export PIP_CACHE_DIR="${CHESS_DFM_LOCAL_ROOT}/cache/pip"
export HF_HOME="${CHESS_DFM_LOCAL_ROOT}/cache/huggingface"
export CUDA_CACHE_PATH="${CHESS_DFM_LOCAL_ROOT}/cache/nvidia/ComputeCache"
export TRITON_CACHE_DIR="${CHESS_DFM_LOCAL_ROOT}/cache/triton"
export TORCH_HOME="${CHESS_DFM_LOCAL_ROOT}/cache/torch"
export TORCH_EXTENSIONS_DIR="${CHESS_DFM_LOCAL_ROOT}/cache/torch-extensions"
export MPLCONFIGDIR="${CHESS_DFM_LOCAL_ROOT}/cache/matplotlib"
export PYTHONPYCACHEPREFIX="${CHESS_DFM_LOCAL_ROOT}/cache/pycache"
export WANDB_DIR="${CHESS_DFM_ARTIFACT_ROOT}/wandb"
export WANDB_CACHE_DIR="${CHESS_DFM_LOCAL_ROOT}/cache/wandb"
export WANDB_CONFIG_DIR="${CHESS_DFM_LOCAL_ROOT}/config/wandb"
export WANDB_DATA_DIR="${CHESS_DFM_LOCAL_ROOT}/data/wandb"
export WANDB_MODE="${WANDB_MODE:-offline}"

export CHESS_DFM_GUARD_CPU_COUNT="${CHESS_DFM_GUARD_CPU_COUNT:-2}"
export CHESS_DFM_GUARD_MIN_START_AVAILABLE_BYTES="${CHESS_DFM_GUARD_MIN_START_AVAILABLE_BYTES:-4294967296}"
export CHESS_DFM_GUARD_MIN_RUNTIME_AVAILABLE_BYTES="${CHESS_DFM_GUARD_MIN_RUNTIME_AVAILABLE_BYTES:-1610612736}"
export CHESS_DFM_GUARD_MAX_PROCESS_RSS_BYTES="${CHESS_DFM_GUARD_MAX_PROCESS_RSS_BYTES:-5368709120}"
export CHESS_DFM_GUARD_MIN_DISK_RESERVE_BYTES="${CHESS_DFM_GUARD_MIN_DISK_RESERVE_BYTES:-5368709120}"
export CHESS_DFM_GUARD_MAX_WORKSPACE_BYTES="${CHESS_DFM_GUARD_MAX_WORKSPACE_BYTES:-32212254720}"
export CHESS_DFM_GUARD_MIN_WORKSPACE_RESERVE_BYTES="${CHESS_DFM_GUARD_MIN_WORKSPACE_RESERVE_BYTES:-5368709120}"
export CHESS_DFM_GUARD_POLL_SECONDS="${CHESS_DFM_GUARD_POLL_SECONDS:-0.05}"
export OMP_NUM_THREADS="${CHESS_DFM_GUARD_CPU_COUNT}"
export OPENBLAS_NUM_THREADS="${CHESS_DFM_GUARD_CPU_COUNT}"
export MKL_NUM_THREADS="${CHESS_DFM_GUARD_CPU_COUNT}"
export NUMEXPR_MAX_THREADS="${CHESS_DFM_GUARD_CPU_COUNT}"
export RAYON_NUM_THREADS="${CHESS_DFM_GUARD_CPU_COUNT}"

if [[ -x /usr/lib/wsl/lib/nvidia-smi ]]; then
  export PATH="/usr/lib/wsl/lib:${PATH}"
fi

mkdir -p \
  "${TMPDIR}" \
  "${XDG_CACHE_HOME}" \
  "${XDG_CONFIG_HOME}" \
  "${XDG_DATA_HOME}" \
  "${UV_CACHE_DIR}" \
  "${PIP_CACHE_DIR}" \
  "${HF_HOME}" \
  "${CUDA_CACHE_PATH}" \
  "${TRITON_CACHE_DIR}" \
  "${TORCH_HOME}" \
  "${TORCH_EXTENSIONS_DIR}" \
  "${MPLCONFIGDIR}" \
  "${PYTHONPYCACHEPREFIX}" \
  "${WANDB_DIR}" \
  "${WANDB_CACHE_DIR}" \
  "${WANDB_CONFIG_DIR}" \
  "${WANDB_DATA_DIR}" \
  "${CHESS_DFM_LOCAL_ROOT}/locks"

exec 9>"${CHESS_DFM_LOCAL_ROOT}/locks/gpu-workload.lock"
if ! flock --exclusive --nonblock 9; then
  echo "RESOURCE_GUARD_BUSY: another local GPU workload owns the lock" >&2
  exit 73
fi

"${_torch_venv_python}" -c \
  'import torch; assert torch.cuda.is_available(), "Torch cannot see CUDA"'

if [[ "$#" -eq 0 ]]; then
  set -- "${_torch_venv_python}" -c \
    'import torch; print(torch.__version__, torch.cuda.get_device_name(0))'
fi

exec "${_torch_venv_python}" \
  "${_torch_research_dir}/resource_guard.py" \
  -- \
  "$@"
