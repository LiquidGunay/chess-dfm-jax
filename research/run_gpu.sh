#!/usr/bin/env bash
set -euo pipefail

_research_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${_research_dir}/env.sh"

export JAX_PLATFORMS="${JAX_PLATFORMS:-cuda}"

if [[ "$#" -eq 0 ]]; then
  set -- "${CHESS_DFM_REPO_ROOT}/.venv/bin/python" \
    "${CHESS_DFM_REPO_ROOT}/research/prepare.py" \
    system-info \
    --require-gpu
fi

exec "$@"
