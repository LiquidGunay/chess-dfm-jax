#!/usr/bin/env python3
"""Measure a roofline point for the BT4 reference forward or a custom JAX step."""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path
import sys

import jax

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chess_dfm_jax.analysis.roofline import (
    format_point,
    measure_roofline_point,
    roofline_bound,
    trace_compiled,
    write_points_csv,
    compile_and_analyze,
)
from chess_dfm_jax.analysis.profile_targets import (
    load_mapped_bt4_params,
    make_jepa_train_target,
    make_nnx_encoder_backward_target,
    make_nnx_encoder_forward_target,
    make_reference_forward_target,
)


def load_callable(spec: str):
    module_name, attr = spec.split(":", 1)
    module = importlib.import_module(module_name)
    return getattr(module, attr)


def build_reference_target(args: argparse.Namespace):
    if args.forward_fn == "chess_dfm_jax.reference_bt4:bt4_forward":
        return make_reference_forward_target(
            batch_size=args.batch_size,
            models_dir=args.models_dir,
            pb=args.pb,
            input_format=args.input_format,
        )

    forward_fn = load_callable(args.forward_fn)
    base = make_reference_forward_target(
        batch_size=args.batch_size,
        models_dir=args.models_dir,
        pb=args.pb,
        input_format=args.input_format,
    )
    params = load_mapped_bt4_params(models_dir=args.models_dir, pb=args.pb)
    planes = base["args"][0]

    def step(batch_planes):
        return forward_fn(params, batch_planes)

    return {"name": f"reference_forward_bs{args.batch_size}", "step_fn": step, "args": (planes,), "kwargs": {}}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target",
        choices=["reference", "nnx_encoder_forward", "nnx_encoder_backward", "jepa_train", "factory"],
        default="reference",
    )
    parser.add_argument("--factory", default=None, help="Dotted path module:function returning a target dict.")
    parser.add_argument("--forward-fn", default="chess_dfm_jax.reference_bt4:bt4_forward")
    parser.add_argument("--models-dir", default=None)
    parser.add_argument("--pb", default=None)
    parser.add_argument("--input-format", default="INPUT_CLASSICAL_112_PLANE")
    parser.add_argument("--compute-dtype", default="float32")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--token-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--mlp-dim", type=int, default=1024)
    parser.add_argument("--head-param-dtype", default="float32")
    parser.add_argument("--head-compute-dtype", default=None)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--peak-tflops", type=float, default=None)
    parser.add_argument("--peak-gbps", type=float, default=None)
    parser.add_argument("--trace-dir", default=None)
    parser.add_argument("--out-csv", default=None)
    args = parser.parse_args()

    if args.target == "reference":
        target = build_reference_target(args)
    elif args.target == "nnx_encoder_forward":
        target = make_nnx_encoder_forward_target(
            batch_size=args.batch_size,
            models_dir=args.models_dir,
            pb=args.pb,
            input_format=args.input_format,
            compute_dtype=args.compute_dtype,
        )
    elif args.target == "nnx_encoder_backward":
        target = make_nnx_encoder_backward_target(
            batch_size=args.batch_size,
            models_dir=args.models_dir,
            pb=args.pb,
            input_format=args.input_format,
            compute_dtype=args.compute_dtype,
        )
    elif args.target == "jepa_train":
        target = make_jepa_train_target(
            batch_size=args.batch_size,
            models_dir=args.models_dir,
            pb=args.pb,
            input_format=args.input_format,
            compute_dtype=args.compute_dtype,
            token_dim=args.token_dim,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            mlp_dim=args.mlp_dim,
            head_param_dtype=args.head_param_dtype,
            head_compute_dtype=args.head_compute_dtype,
        )
    else:
        if args.factory is None:
            raise SystemExit("--factory is required when --target factory")
        target = load_callable(args.factory)()

    point = measure_roofline_point(
        target["name"],
        target["step_fn"],
        *target.get("args", ()),
        warmup=args.warmup,
        repeat=args.repeat,
        **target.get("kwargs", {}),
    )
    print(f"device: {jax.default_backend()} | {jax.devices()[0]}")
    print(format_point(point))

    if args.peak_tflops is not None and args.peak_gbps is not None and point.arithmetic_intensity is not None:
        bound = roofline_bound(
            point.arithmetic_intensity,
            peak_tflops=args.peak_tflops,
            peak_gbps=args.peak_gbps,
        )
        print(f"roofline_bound_tflops: {bound:.3f}")
        if point.achieved_tflops is not None:
            print(f"roofline_efficiency: {point.achieved_tflops / max(bound, 1e-12):.3f}")

    if args.trace_dir:
        compiled, _analysis = compile_and_analyze(target["step_fn"], *target.get("args", ()), **target.get("kwargs", {}))
        trace_compiled(args.trace_dir, compiled, *target.get("args", ()), repeat=args.repeat, **target.get("kwargs", {}))
        print(f"trace written to: {args.trace_dir}")

    if args.out_csv:
        write_points_csv([point], args.out_csv)
        print(f"csv written to: {args.out_csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
