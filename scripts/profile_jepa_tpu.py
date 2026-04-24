#!/usr/bin/env python3
"""Run a TPU-oriented JEPA profiling bundle and save artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import jax

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chess_dfm_jax.analysis.jepa_theory import estimate_jepa_theory
from chess_dfm_jax.analysis.profile_targets import make_jepa_train_target
from chess_dfm_jax.analysis.roofline import (
    compile_and_analyze,
    format_point,
    measure_roofline_point,
    trace_compiled,
    write_points_csv,
)


def parse_int_list(text: str) -> list[int]:
    return [int(part) for part in text.split(",") if part.strip()]


def main() -> int:
    jax.distributed.initialize()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--models-dir", default=None)
    parser.add_argument("--compute-dtype", default="bfloat16")
    parser.add_argument("--head-param-dtype", default="float32")
    parser.add_argument("--head-compute-dtype", default="bfloat16")
    parser.add_argument("--trace-batch-size", type=int, default=128)
    parser.add_argument("--trace-token-dim", type=int, default=512)
    parser.add_argument("--trace-num-layers", type=int, default=4)
    parser.add_argument("--trace-num-heads", type=int, default=8)
    parser.add_argument("--trace-mlp-dim", type=int, default=2048)
    parser.add_argument("--sweep-batch-sizes", default="64,128,256")
    parser.add_argument("--sweep-token-dims", default="256,512,768")
    parser.add_argument("--sweep-mlp-dims", default="1024,2048,3072")
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=5)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    trace_dir = out_dir / "tb_trace"

    device_info = {
        "backend": jax.default_backend(),
        "process_count": jax.process_count(),
        "process_index": jax.process_index(),
        "local_device_count": jax.local_device_count(),
        "device_count": jax.device_count(),
        "devices": [str(device) for device in jax.devices()],
    }
    (out_dir / "device_info.json").write_text(json.dumps(device_info, indent=2), encoding="utf-8")

    trace_target = make_jepa_train_target(
        batch_size=args.trace_batch_size,
        models_dir=args.models_dir,
        compute_dtype=args.compute_dtype,
        token_dim=args.trace_token_dim,
        num_layers=args.trace_num_layers,
        num_heads=args.trace_num_heads,
        mlp_dim=args.trace_mlp_dim,
        head_param_dtype=args.head_param_dtype,
        head_compute_dtype=args.head_compute_dtype,
    )
    trace_point = measure_roofline_point(
        trace_target["name"],
        trace_target["step_fn"],
        *trace_target.get("args", ()),
        warmup=args.warmup,
        repeat=args.repeat,
        **trace_target.get("kwargs", {}),
    )
    trace_theory = estimate_jepa_theory(
        batch_size=args.trace_batch_size,
        token_dim=args.trace_token_dim,
        num_layers=args.trace_num_layers,
        num_heads=args.trace_num_heads,
        mlp_dim=args.trace_mlp_dim,
    )
    compiled, _ = compile_and_analyze(
        trace_target["step_fn"],
        *trace_target.get("args", ()),
        **trace_target.get("kwargs", {}),
    )
    trace_compiled(
        trace_dir,
        compiled,
        *trace_target.get("args", ()),
        repeat=args.repeat,
        **trace_target.get("kwargs", {}),
    )
    write_points_csv([trace_point], out_dir / "single_point.csv")
    (out_dir / "single_point.json").write_text(
        json.dumps(
            {
                "device_info": device_info,
                "point": trace_point.as_dict(),
                "theory": trace_theory.__dict__,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    batch_sizes = parse_int_list(args.sweep_batch_sizes)
    token_dims = parse_int_list(args.sweep_token_dims)
    mlp_dims = parse_int_list(args.sweep_mlp_dims)
    if len(token_dims) != len(mlp_dims):
        raise SystemExit("--sweep-token-dims and --sweep-mlp-dims must have the same length")

    sweep_rows = []
    sweep_points = []
    for batch_size in batch_sizes:
        for token_dim, mlp_dim in zip(token_dims, mlp_dims):
            target = make_jepa_train_target(
                batch_size=batch_size,
                models_dir=args.models_dir,
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
            sweep_points.append(point)
            sweep_rows.append(
                {
                    "batch_size": batch_size,
                    "token_dim": token_dim,
                    "mlp_dim": mlp_dim,
                    "num_layers": args.num_layers,
                    "num_heads": args.num_heads,
                    "point": point.as_dict(),
                    "theory": theory.__dict__,
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
            sys.stdout.flush()

    write_points_csv(sweep_points, out_dir / "sweep_points.csv")
    (out_dir / "sweep_points.json").write_text(json.dumps(sweep_rows, indent=2), encoding="utf-8")

    summary_lines = [
        "TPU JEPA profile bundle",
        json.dumps(device_info),
        "",
        "Single point:",
        format_point(trace_point),
        "",
        "Sweep count: " + str(len(sweep_rows)),
        "Trace directory: " + str(trace_dir),
    ]
    (out_dir / "SUMMARY.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
