#!/usr/bin/env python3
"""Run the local controller loop for Spot TPU JEPA training."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chess_dfm_jax.training.tpu_jobs import TPUJobSpec, run_spot_controller


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-spec", required=True, type=str)
    parser.add_argument("--override-source-uri", type=str, default=None, help="Force workers to use this code snapshot.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    spec = TPUJobSpec.from_json(args.job_spec)
    result = run_spot_controller(spec, override_source_uri=args.override_source_uri)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
