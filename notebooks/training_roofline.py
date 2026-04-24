import marimo

__generated_with = "0.20.1"
app = marimo.App(width="full")


@app.cell
def _():
    import jax
    import pandas as pd

    from chess_dfm_jax.analysis.bt4_theory import estimate_bt4_theory
    from chess_dfm_jax.analysis.profile_targets import (
        load_mapped_bt4_params,
        make_nnx_encoder_backward_target,
        make_nnx_encoder_forward_target,
        make_reference_forward_target,
    )
    from chess_dfm_jax.analysis.roofline import measure_roofline_point, roofline_bound
    from chess_dfm_jax.paths import default_bt4_paths

    return (
        default_bt4_paths,
        estimate_bt4_theory,
        jax,
        load_mapped_bt4_params,
        make_nnx_encoder_backward_target,
        make_nnx_encoder_forward_target,
        make_reference_forward_target,
        measure_roofline_point,
        pd,
        roofline_bound,
    )


@app.cell
def _(marimo):
    marimo.md(
        "# Training Roofline Scaffold\n\n"
        "This notebook measures three points against the same BT4 weights and synthetic FEN batch:\n\n"
        "- trusted reference full forward\n"
        "- `flax.nnx` encoder forward\n"
        "- `flax.nnx` encoder forward + backward on a simple encoder-energy loss\n"
    )
    return


@app.cell
def _(default_bt4_paths):
    MODEL_PATHS = default_bt4_paths()
    BATCH_SIZE = 8
    COMPUTE_DTYPE = "float16"
    WARMUP = 2
    REPEAT = 10
    PEAK_TFLOPS = None
    PEAK_GBPS = None
    return BATCH_SIZE, COMPUTE_DTYPE, MODEL_PATHS, PEAK_GBPS, PEAK_TFLOPS, REPEAT, WARMUP


@app.cell
def _(BATCH_SIZE, COMPUTE_DTYPE, MODEL_PATHS, PEAK_GBPS, PEAK_TFLOPS, REPEAT, WARMUP, marimo):
    marimo.md(
        f"""
## Profiling config

- pb: `{MODEL_PATHS['exported_pb']}`
- batch size: `{BATCH_SIZE}`
- nnx compute dtype: `{COMPUTE_DTYPE}`
- warmup: `{WARMUP}`
- repeat: `{REPEAT}`
- peak TFLOPs: `{PEAK_TFLOPS}`
- peak GB/s: `{PEAK_GBPS}`
"""
    )
    return


@app.cell
def _(
    BATCH_SIZE,
    COMPUTE_DTYPE,
    MODEL_PATHS,
    REPEAT,
    WARMUP,
    estimate_bt4_theory,
    load_mapped_bt4_params,
    make_nnx_encoder_backward_target,
    make_nnx_encoder_forward_target,
    make_reference_forward_target,
    measure_roofline_point,
):
    target_specs = [
        make_reference_forward_target(batch_size=BATCH_SIZE, models_dir=MODEL_PATHS["models_dir"]),
        make_nnx_encoder_forward_target(
            batch_size=BATCH_SIZE,
            models_dir=MODEL_PATHS["models_dir"],
            compute_dtype=COMPUTE_DTYPE,
        ),
        make_nnx_encoder_backward_target(
            batch_size=BATCH_SIZE,
            models_dir=MODEL_PATHS["models_dir"],
            compute_dtype=COMPUTE_DTYPE,
        ),
    ]
    points = [
        measure_roofline_point(
            spec["name"],
            spec["step_fn"],
            *spec.get("args", ()),
            warmup=WARMUP,
            repeat=REPEAT,
            **spec.get("kwargs", {}),
        )
        for spec in target_specs
    ]
    params = load_mapped_bt4_params(models_dir=MODEL_PATHS["models_dir"])
    theory = estimate_bt4_theory(params, batch_size=BATCH_SIZE)
    return points, theory


@app.cell
def _(jax, marimo):
    marimo.md(f"## Runtime\n\nDevice: `{jax.default_backend()}` | `{jax.devices()[0]}`")
    return


@app.cell
def _(pd, points):
    summary = pd.DataFrame(
        [
            {
                "name": point.name,
                "seconds_per_step": point.seconds_per_step,
                "steps_per_second": point.steps_per_second,
                "flops_per_step": point.flops_per_step,
                "bytes_per_step": point.bytes_per_step,
                "arithmetic_intensity": point.arithmetic_intensity,
                "achieved_tflops": point.achieved_tflops,
                "achieved_gbps": point.achieved_gbps,
            }
            for point in points
        ]
    )
    summary
    return (summary,)


@app.cell
def _(pd, theory):
    theory_summary = pd.DataFrame(
        [
            {"metric": "parameter_count", "value": theory.parameter_count},
            {"metric": "parameter_bytes_fp16", "value": theory.parameter_bytes_fp16},
            {"metric": "parameter_bytes_fp32", "value": theory.parameter_bytes_fp32},
            {"metric": "encoder_forward_flops", "value": theory.encoder_forward_flops},
            {"metric": "full_forward_flops", "value": theory.full_forward_flops},
            {
                "metric": "encoder_backward_flops_rule_of_thumb",
                "value": theory.encoder_backward_flops_rule_of_thumb,
            },
        ]
    )
    theory_summary
    return


@app.cell
def _(PEAK_GBPS, PEAK_TFLOPS, pd, points, roofline_bound):
    if PEAK_GBPS is None or PEAK_TFLOPS is None:
        peak_summary = pd.DataFrame(
            [{"note": "Set PEAK_TFLOPS and PEAK_GBPS in the config cell to compute roofline efficiency."}]
        )
    else:
        peak_summary = pd.DataFrame(
            [
                {
                    "name": point.name,
                    "roofline_bound_tflops": roofline_bound(
                        point.arithmetic_intensity,
                        peak_tflops=PEAK_TFLOPS,
                        peak_gbps=PEAK_GBPS,
                    )
                    if point.arithmetic_intensity is not None
                    else None,
                    "efficiency": (
                        point.achieved_tflops
                        / roofline_bound(
                            point.arithmetic_intensity,
                            peak_tflops=PEAK_TFLOPS,
                            peak_gbps=PEAK_GBPS,
                        )
                    )
                    if point.arithmetic_intensity is not None and point.achieved_tflops is not None
                    else None,
                }
                for point in points
            ]
        )
    peak_summary
    return


@app.cell
def _(marimo):
    marimo.md(
        "## Notes\n\n"
        "- The backward point uses a simple encoder-energy loss `mean(square(tokens))` to force gradients through the full encoder.\n"
        "- This is a profiling surrogate, not the final JEPA loss.\n"
        "- Once the JEPA objective exists, swap the backward target to your real train step and reuse the same notebook structure.\n"
    )
    return


if __name__ == "__main__":
    app.run()
