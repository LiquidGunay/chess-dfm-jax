#!/usr/bin/env python3
"""Sweep token-level JEPA profiling points and write a CSV summary."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chess_dfm_jax.analysis.jepa_theory import estimate_jepa_theory
from chess_dfm_jax.analysis.profile_targets import make_jepa_train_target
from chess_dfm_jax.analysis.roofline import measure_roofline_point, write_points_csv


def parse_int_list(text: str) -> list[int]:
    return [int(part) for part in text.split(",") if part.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-sizes", default="64,128,256")
    parser.add_argument("--token-dims", default="256,384,512")
    parser.add_argument("--mlp-dims", default="1024,1536,2048")
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--compute-dtype", default="bfloat16")
    parser.add_argument("--head-param-dtype", default="float32")
    parser.add_argument("--head-compute-dtype", default="bfloat16")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--out-json", default=None)
    args = parser.parse_args()

    batch_sizes = parse_int_list(args.batch_sizes)
    token_dims = parse_int_list(args.token_dims)
    mlp_dims = parse_int_list(args.mlp_dims)
    if len(token_dims) != len(mlp_dims):
        raise SystemExit("--token-dims and --mlp-dims must have the same length")

    points = []
    rows = []
    for batch_size in batch_sizes:
        for token_dim, mlp_dim in zip(token_dims, mlp_dims):
            target = make_jepa_train_target(
                batch_size=batch_size,
                compute_dtype=args.compute_dtype,
                token_dim=token_dim,
                num_layers=args.num_layers,
                num_heads=args.num_heads,
                mlp_dim=mlp_dim,
                head_param_dtype=args.head_param_dtype,
                head_compute_dtype=args.head_compute_dtype,
            )
            point = measure_roofline_point(
                target["name"],
                target["step_fn"],
                *target.get("args", ()),
                warmup=args.warmup,
                repeat=args.repeat,
                **target.get("kwargs", {}),
            )
            theory = estimate_jepa_theory(
                batch_size=batch_size,
                token_dim=token_dim,
                num_layers=args.num_layers,
                num_heads=args.num_heads,
                mlp_dim=mlp_dim,
            )
            points.append(point)
            rows.append(
                {
                    "name": point.name,
                    "batch_size": batch_size,
                    "token_dim": token_dim,
                    "mlp_dim": mlp_dim,
                    "num_layers": args.num_layers,
                    "num_heads": args.num_heads,
                    "seconds_per_step": point.seconds_per_step,
                    "flops_per_step": point.flops_per_step,
                    "bytes_per_step": point.bytes_per_step,
                    "arithmetic_intensity": point.arithmetic_intensity,
                    "achieved_tflops": point.achieved_tflops,
                    "achieved_gbps": point.achieved_gbps,
                    "theory_trainable_params": theory.trainable_parameter_count,
                    "theory_forward_flops": theory.forward_flops,
                    "theory_train_step_flops": theory.train_step_flops_rule_of_thumb,
                }
            )
            print(
                json.dumps(
                    {
                        "batch_size": batch_size,
                        "token_dim": token_dim,
                        "mlp_dim": mlp_dim,
                        "seconds_per_step": point.seconds_per_step,
                        "arithmetic_intensity": point.arithmetic_intensity,
                        "achieved_tflops": point.achieved_tflops,
                    }
                )
            )

    write_points_csv(points, args.out_csv)
    if args.out_json:
        Path(args.out_json).write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
