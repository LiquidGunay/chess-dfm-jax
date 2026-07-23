#!/usr/bin/env bash
set -euo pipefail

_research_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${_research_dir}/env.sh"

export JAX_PLATFORMS="${JAX_PLATFORMS:-cuda}"
export CHESS_DFM_GUARD_CPU_COUNT="${CHESS_DFM_GUARD_CPU_COUNT:-2}"

# These cap library thread pools; resource_guard.py additionally applies a
# kernel-enforced CPU affinity mask, so compilation cannot consume all host
# CPUs even if a library ignores its thread-count environment variable.
export OMP_NUM_THREADS="${CHESS_DFM_GUARD_CPU_COUNT}"
export OPENBLAS_NUM_THREADS="${CHESS_DFM_GUARD_CPU_COUNT}"
export MKL_NUM_THREADS="${CHESS_DFM_GUARD_CPU_COUNT}"
export NUMEXPR_MAX_THREADS="${CHESS_DFM_GUARD_CPU_COUNT}"
export RAYON_NUM_THREADS="${CHESS_DFM_GUARD_CPU_COUNT}"
export TF_NUM_INTRAOP_THREADS="${CHESS_DFM_GUARD_CPU_COUNT}"
export TF_NUM_INTEROP_THREADS="1"
export JAX_NUM_THREADS="${CHESS_DFM_GUARD_CPU_COUNT}"

# Hold one host-visible lock for the full prepare/compile/run lifetime. This is
# deliberately independent of process namespaces: a disconnected tool session
# cannot make a second JAX compilation appear safe.
_lock_dir="${CHESS_DFM_LOCAL_ROOT}/locks"
mkdir -p "${_lock_dir}"
exec 9>"${_lock_dir}/gpu-workload.lock"
if ! flock --exclusive --nonblock 9; then
  echo "RESOURCE_GUARD_BUSY: another local GPU workload owns ${_lock_dir}/gpu-workload.lock" >&2
  exit 73
fi

"${CHESS_DFM_REPO_ROOT}/.venv/bin/python" \
  "${CHESS_DFM_REPO_ROOT}/research/prepare.py" \
  runtime-check >/dev/null

if [[ "$#" -eq 0 ]]; then
  set -- "${CHESS_DFM_REPO_ROOT}/.venv/bin/python" \
    "${CHESS_DFM_REPO_ROOT}/research/prepare.py" \
    system-info \
    --require-gpu
fi

exec "${CHESS_DFM_REPO_ROOT}/.venv/bin/python" \
  "${CHESS_DFM_REPO_ROOT}/research/resource_guard.py" \
  -- \
  "$@"
