#!/usr/bin/env python3
"""Cross-framework parity gates for the eager-PyTorch migration.

The harness restores the same checksum-pinned model tree into both runtimes,
materializes stochastic choices once, and compares named intermediates and
loss components. CPU comparison runs both models in one process. Production
BF16 comparison uses two sequential guarded GPU processes and a small NPZ
exchange artifact so the full PyTorch and JAX runtimes never coexist on GPU.
The harness never creates an optimizer or checkpoint.
"""

from __future__ import annotations

import argparse
import dataclasses
import gc
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping


def _requested_mode() -> str:
    for argument in sys.argv[1:]:
        if argument.startswith("--mode="):
            return argument.split("=", 1)[1]
    try:
        return sys.argv[sys.argv.index("--mode") + 1]
    except (ValueError, IndexError):
        return "cpu-compare"


if _requested_mode() in {"cpu-compare", "torch-export"}:
    os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from flax import nnx  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from chess_dfm_jax.analysis.profile_targets import load_mapped_bt4_params  # noqa: E402
from chess_dfm_jax.nnx_bt4 import TrainableParam, make_bt4_model  # noqa: E402
from research.prepare import FixedTrajectoryBatches, require_within_workspace  # noqa: E402
from research.train import (  # noqa: E402
    DEFAULT_MODELS_DIR,
    DEFAULT_RUN_ROOT,
    JointLatentSASAConfig,
    JointLatentSASAModel,
    resolve_config,
)
from research.train_torch import (  # noqa: E402
    CONFIG,
    JointModel,
    StepChoices,
    _legal_mass,
    _sigreg_v_stat,
    _torch_batch,
    bind_source_model,
    load_source_model,
    load_verified_source_model,
    materialize_step_choices,
)


RELATIVE_L2_MAX = 7.5e-4
MAX_ABSOLUTE_MAX = 2e-2
SCALAR_RELATIVE_MAX = 2e-3
SCALAR_ABSOLUTE_MAX = 2e-4
BF16_RELATIVE_L2_MAX = 3e-2
BF16_COSINE_MIN = 0.999
BF16_SCALAR_RELATIVE_MAX = 2e-2
BF16_SCALAR_ABSOLUTE_MAX = 1e-2
SOURCE_MAPPING_SHA256 = (
    "697c0944786a208eb22a290d01c9889c9c02564d43068d1b67cb30d8ae565261"
)
_TENSOR_NAMES = (
    "current_tokens",
    "future_tokens",
    "z_all",
    "z_dfm",
    "noisy_logits",
    "clean_logits",
    "clean_action_hidden",
    "pred_z",
)


def _jax_config(compute_dtype: str) -> JointLatentSASAConfig:
    config, _ = resolve_config(DEFAULT_RUN_ROOT)
    values = dataclasses.asdict(config)
    values.update(
        {
            "encoder_dtype": compute_dtype,
            "compute_dtype": compute_dtype,
            "param_dtype": "float32",
            "learning_rate": CONFIG.learning_rate,
            "bt4_learning_rate": CONFIG.bt4_learning_rate,
            "first_legality_coeff": CONFIG.legality_coeff,
            "jepa_norm_loss_coeff": 0.0,
            "jepa_sigreg_coeff": CONFIG.target_sigreg_coeff,
            "jepa_pred_sigreg_coeff": CONFIG.pred_sigreg_coeff,
            "jepa_sigreg_estimator": "v_stat",
            "jepa_sigreg_example_count": CONFIG.sigreg_example_count,
            "jepa_target_sample_count": CONFIG.target_sample_count,
            "jepa_target_sampling_unit": "example_balanced",
            "bt4_encode_chunk_size": 0,
            "bt4_future_target_stop_gradient": True,
            "bt4_future_target_trainable_tail_layers": (
                CONFIG.future_trainable_tail_layers
            ),
            "lr_warmup_steps": 0,
            "lr_decay_start_steps": CONFIG.lr_decay_start,
            "lr_decay_steps": CONFIG.lr_decay_steps,
            "lr_min_ratio": CONFIG.lr_min_ratio,
        }
    )
    return JointLatentSASAConfig(**values)


def _build_jax_model(
    source: Mapping[str, Any],
    *,
    models_dir: Path,
    compute_dtype: str,
) -> JointLatentSASAModel:
    mapped = load_mapped_bt4_params(models_dir=str(models_dir))
    config = _jax_config(compute_dtype)
    encoder = make_bt4_model(
        mapped,
        dtype=(
            jnp.float32
            if compute_dtype == "float32"
            else jnp.bfloat16
        ),
        attention_impl="manual",
        train_encoder=True,
    )
    del mapped
    gc.collect()
    model = JointLatentSASAModel(encoder, config, rngs=nnx.Rngs(0))
    state = nnx.state(model, TrainableParam)
    nnx.replace_by_pure_dict(state, source)
    nnx.update(model, state)
    return model


def _numpy_choices(choices: StepChoices) -> dict[str, np.ndarray]:
    return {
        field: np.asarray(tensor.detach().cpu())
        for field, tensor in zip(StepChoices._fields, choices, strict=True)
    }


def _torch_intermediates(
    model: JointModel,
    batch: Mapping[str, torch.Tensor],
    choices: StepChoices,
    *,
    compute_dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    dtype = compute_dtype
    actions = batch["action_indices"][:, : CONFIG.horizon].long()
    rows = torch.arange(actions.shape[0])
    selected_planes = batch["future_planes"][rows, choices.target_horizon]
    current_tokens = model.encoder.encode_current(
        batch["current_planes"], compute_dtype=dtype, remat=False
    )
    future_tokens = model.encoder.encode_future_tail(
        selected_planes,
        compute_dtype=dtype,
        trainable_tail_layers=CONFIG.future_trainable_tail_layers,
    )
    all_tokens = torch.stack((current_tokens, future_tokens), dim=1)
    z_all = model.state_projector(
        all_tokens.reshape(actions.shape[0] * 2, 64, 1024), dtype
    ).reshape(actions.shape[0], 2, CONFIG.z_dim)
    z_dfm = model.dfm_state_projector(current_tokens, dtype)
    t = choices.training_time
    masked = choices.mask_uniform < (1.0 - t).unsqueeze(1)
    noisy = torch.where(masked, torch.full_like(actions, 1858), actions)
    logits = model.planner(z_dfm, noisy, t, dtype)
    assert isinstance(logits, torch.Tensor)
    clean = model.planner(
        z_dfm,
        actions,
        torch.ones_like(t),
        dtype,
        return_hidden=True,
    )
    assert isinstance(clean, tuple)
    clean_logits, clean_hidden = clean
    pred_z = model.jepa_rollout(z_all[:, 0], actions, clean_hidden, dtype)
    return {
        "current_tokens": current_tokens,
        "future_tokens": future_tokens,
        "z_all": z_all,
        "z_dfm": z_dfm,
        "noisy_logits": logits,
        "clean_logits": clean_logits,
        "clean_action_hidden": clean_hidden,
        "pred_z": pred_z,
        "is_masked": masked,
    }


def _jax_intermediates(
    model: JointLatentSASAModel,
    batch: Mapping[str, np.ndarray],
    choices: Mapping[str, np.ndarray],
) -> dict[str, jax.Array]:
    actions = jnp.asarray(batch["action_indices"][:, : CONFIG.horizon])
    rows = jnp.arange(actions.shape[0])
    selected = jnp.asarray(choices["target_horizon"])
    selected_planes = jnp.asarray(batch["future_planes"])[rows, selected]
    current_tokens = model.encode_bt4_tokens(jnp.asarray(batch["current_planes"]))
    future_tokens = model.encode_future_bt4_tokens(selected_planes)
    all_tokens = jnp.stack((current_tokens, future_tokens), axis=1)
    z_all = model.state_projector(
        all_tokens.reshape((actions.shape[0] * 2, 64, 1024))
    ).reshape((actions.shape[0], 2, CONFIG.z_dim))
    z_dfm = model.dfm_latents(current_tokens)
    t = jnp.asarray(choices["training_time"])
    masked = jnp.asarray(choices["mask_uniform"]) < (1.0 - t[:, None])
    noisy = jnp.where(masked, 1858, actions)
    logits = model.planner_from_latents(z_dfm, noisy, t)
    clean_logits, clean_hidden = model.planner_from_latents(
        z_dfm,
        actions,
        jnp.ones_like(t),
        return_hidden=True,
    )
    pred_z = model.jepa_predictions_from_latents(
        z_all[:, 0],
        actions,
        clean_hidden["action_tokens"],
        z0_normalized=True,
    )
    return {
        "current_tokens": current_tokens,
        "future_tokens": future_tokens,
        "z_all": z_all,
        "z_dfm": z_dfm,
        "noisy_logits": logits,
        "clean_logits": clean_logits,
        "clean_action_hidden": clean_hidden["action_tokens"],
        "pred_z": pred_z,
        "is_masked": masked,
    }


def _torch_loss_components(
    values: Mapping[str, torch.Tensor],
    batch: Mapping[str, torch.Tensor],
    choices: StepChoices,
) -> dict[str, torch.Tensor]:
    actions = batch["action_indices"][:, : CONFIG.horizon].long()
    valid = batch["valid"].float()
    future_valid = batch["future_valid"][:, : CONFIG.horizon].float()
    rows = torch.arange(actions.shape[0])
    selected = choices.target_horizon
    selected_valid = future_valid[rows, selected].unsqueeze(1)
    logits = values["noisy_logits"]
    masked = values["is_masked"]
    log_probabilities = torch.log_softmax(logits, dim=-1)
    ce = -torch.gather(
        log_probabilities, -1, actions.unsqueeze(-1)
    ).squeeze(-1)
    ce_weight = masked.float() * valid.unsqueeze(1)
    denominator = ce_weight.sum(dim=0)
    ce_horizon = (ce * ce_weight).sum(dim=0) / denominator.clamp_min(1.0)
    active = (denominator > 0).float()
    dfm_ce = (ce_horizon * active).sum() / active.sum().clamp_min(1.0)
    legal_mass = _legal_mass(
        torch.softmax(logits[:, 0].float(), dim=-1),
        batch["legal_idx"][:, 0],
        batch["legal_count"][:, 0],
    )
    legal_gate = valid * masked[:, 0].float()
    if "legal_masks_valid" in batch:
        legal_gate = legal_gate * batch["legal_masks_valid"][:, 0].float()
    legality = ((1.0 - legal_mass) * legal_gate).sum() / legal_gate.sum().clamp_min(
        1.0
    )
    target_z = values["z_all"][:, 1:]
    pred_for_loss = values["pred_z"][rows, selected].unsqueeze(1)
    raw_mse = (pred_for_loss.float() - target_z.float()).square().mean(dim=-1)
    positive_weight = valid.unsqueeze(1) * selected_valid
    positive = (raw_mse * positive_weight).sum() / positive_weight.sum().clamp_min(
        1.0
    )
    selected_rows = choices.sigreg_indices
    sigreg_valid = valid[selected_rows]
    target_weight = torch.cat(
        (
            sigreg_valid.unsqueeze(1),
            sigreg_valid.unsqueeze(1)
            * selected_valid[selected_rows]
            * float(CONFIG.horizon),
        ),
        dim=1,
    ).reshape(-1)
    target_sigreg, _ = _sigreg_v_stat(
        values["z_all"][selected_rows].reshape(-1, CONFIG.z_dim),
        target_weight,
        choices.sigreg_directions,
        reference_count=1.0,
    )
    pred_weight = (
        future_valid[selected_rows] * sigreg_valid.unsqueeze(1)
    ).reshape(-1)
    pred_sigreg, _ = _sigreg_v_stat(
        values["pred_z"][selected_rows].reshape(-1, CONFIG.z_dim),
        pred_weight,
        choices.sigreg_directions,
        reference_count=1.0,
    )
    total = (
        dfm_ce
        + CONFIG.legality_coeff * legality
        + positive
        + CONFIG.target_sigreg_coeff * target_sigreg
        + CONFIG.pred_sigreg_coeff * pred_sigreg
    )
    return {
        "dfm_ce": dfm_ce,
        "legality": legality,
        "jepa_positive": positive,
        "target_sigreg": target_sigreg,
        "pred_sigreg": pred_sigreg,
        "unclipped_total": total,
    }


def _jax_sigreg(
    z: jax.Array,
    sample_weight: jax.Array,
    directions: jax.Array,
) -> jax.Array:
    projected = jnp.asarray(z, jnp.float32) @ jnp.asarray(directions, jnp.float32)
    weight = jnp.maximum(jnp.asarray(sample_weight, jnp.float32), 0.0)
    t = jnp.linspace(0.0, 3.0, 17, dtype=jnp.float32)
    dt = jnp.asarray(3.0 / 16.0, dtype=jnp.float32)
    quadrature = jnp.full((17,), 2.0 * dt, dtype=jnp.float32)
    quadrature = quadrature.at[0].set(dt).at[-1].set(dt)
    phi = jnp.exp(-0.5 * jnp.square(t))
    quadrature = quadrature * phi
    xt = projected[:, :, None] * t
    denominator = jnp.maximum(jnp.sum(weight), 1.0)
    cosine = jnp.sum(jnp.cos(xt) * weight[:, None, None], axis=0) / denominator
    sine = jnp.sum(jnp.sin(xt) * weight[:, None, None], axis=0) / denominator
    error = jnp.square(cosine - phi[None, :]) + jnp.square(sine)
    return jnp.mean(error @ quadrature)


def _jax_loss_components(
    values: Mapping[str, jax.Array],
    batch: Mapping[str, np.ndarray],
    choices: Mapping[str, np.ndarray],
) -> dict[str, jax.Array]:
    actions = jnp.asarray(batch["action_indices"][:, : CONFIG.horizon])
    valid = jnp.asarray(batch["valid"], jnp.float32)
    future_valid = jnp.asarray(
        batch["future_valid"][:, : CONFIG.horizon], jnp.float32
    )
    rows = jnp.arange(actions.shape[0])
    selected = jnp.asarray(choices["target_horizon"])
    selected_valid = future_valid[rows, selected][:, None]
    logits = values["noisy_logits"]
    masked = values["is_masked"]
    log_probabilities = jax.nn.log_softmax(logits, axis=-1)
    ce = -jnp.take_along_axis(
        log_probabilities, actions[..., None], axis=-1
    )[..., 0]
    ce_weight = masked.astype(jnp.float32) * valid[:, None]
    denominator = jnp.sum(ce_weight, axis=0)
    ce_horizon = jnp.sum(ce * ce_weight, axis=0) / jnp.maximum(denominator, 1.0)
    active = (denominator > 0).astype(jnp.float32)
    dfm_ce = jnp.sum(ce_horizon * active) / jnp.maximum(jnp.sum(active), 1.0)
    safe = jnp.clip(jnp.asarray(batch["legal_idx"][:, 0]), 0, _VOCAB_SIZE - 1)
    probabilities = jax.nn.softmax(jnp.asarray(logits[:, 0], jnp.float32), axis=-1)
    gathered = jnp.take_along_axis(probabilities, safe, axis=-1)
    slots = jnp.arange(safe.shape[-1])
    legal_slots = slots < jnp.asarray(batch["legal_count"][:, 0])[:, None]
    legal_mass = jnp.clip(jnp.sum(jnp.where(legal_slots, gathered, 0.0), axis=-1), 0, 1)
    legal_gate = valid * masked[:, 0].astype(jnp.float32)
    if "legal_masks_valid" in batch:
        legal_gate = legal_gate * jnp.asarray(
            batch["legal_masks_valid"][:, 0], jnp.float32
        )
    legality = jnp.sum((1.0 - legal_mass) * legal_gate) / jnp.maximum(
        jnp.sum(legal_gate), 1.0
    )
    target_z = values["z_all"][:, 1:]
    pred_for_loss = values["pred_z"][rows, selected][:, None]
    raw_mse = jnp.mean(
        jnp.square(
            jnp.asarray(pred_for_loss, jnp.float32)
            - jnp.asarray(target_z, jnp.float32)
        ),
        axis=-1,
    )
    positive_weight = valid[:, None] * selected_valid
    positive = jnp.sum(raw_mse * positive_weight) / jnp.maximum(
        jnp.sum(positive_weight), 1.0
    )
    selected_rows = jnp.asarray(choices["sigreg_indices"])
    sigreg_valid = valid[selected_rows]
    target_weight = jnp.concatenate(
        (
            sigreg_valid[:, None],
            sigreg_valid[:, None]
            * selected_valid[selected_rows]
            * float(CONFIG.horizon),
        ),
        axis=1,
    ).reshape(-1)
    target_sigreg = _jax_sigreg(
        values["z_all"][selected_rows].reshape((-1, CONFIG.z_dim)),
        target_weight,
        jnp.asarray(choices["sigreg_directions"]),
    )
    pred_weight = (
        future_valid[selected_rows] * sigreg_valid[:, None]
    ).reshape(-1)
    pred_sigreg = _jax_sigreg(
        values["pred_z"][selected_rows].reshape((-1, CONFIG.z_dim)),
        pred_weight,
        jnp.asarray(choices["sigreg_directions"]),
    )
    total = (
        dfm_ce
        + CONFIG.legality_coeff * legality
        + positive
        + CONFIG.target_sigreg_coeff * target_sigreg
        + CONFIG.pred_sigreg_coeff * pred_sigreg
    )
    return {
        "dfm_ce": dfm_ce,
        "legality": legality,
        "jepa_positive": positive,
        "target_sigreg": target_sigreg,
        "pred_sigreg": pred_sigreg,
        "unclipped_total": total,
    }


_VOCAB_SIZE = 1858


def _array_metrics(
    reference: Any,
    candidate: Any,
    *,
    compute_dtype: str,
) -> dict[str, float | bool]:
    left = np.asarray(reference, dtype=np.float64)
    right = np.asarray(candidate, dtype=np.float64)
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch: {left.shape} != {right.shape}")
    difference = right - left
    left_flat = left.reshape(-1)
    right_flat = right.reshape(-1)
    left_norm = float(np.linalg.norm(left_flat))
    right_norm = float(np.linalg.norm(right_flat))
    relative_l2 = float(
        np.linalg.norm(difference.reshape(-1)) / max(left_norm, 1e-30)
    )
    maximum = float(np.max(np.abs(difference)))
    cosine = float(
        np.dot(left_flat, right_flat) / max(left_norm * right_norm, 1e-30)
    )
    if compute_dtype == "bfloat16":
        passed = (
            relative_l2 <= BF16_RELATIVE_L2_MAX
            and cosine >= BF16_COSINE_MIN
        )
    else:
        passed = (
            relative_l2 <= RELATIVE_L2_MAX
            and maximum <= MAX_ABSOLUTE_MAX
        )
    return {
        "relative_l2_error": relative_l2,
        "max_absolute_error": maximum,
        "cosine_similarity": cosine,
        "pass": bool(passed),
    }


def _scalar_metrics(
    reference: Any,
    candidate: Any,
    *,
    compute_dtype: str,
) -> dict[str, float | bool]:
    left = float(np.asarray(reference))
    right = float(np.asarray(candidate))
    absolute = abs(right - left)
    relative = absolute / max(abs(left), 1e-12)
    relative_max = (
        BF16_SCALAR_RELATIVE_MAX
        if compute_dtype == "bfloat16"
        else SCALAR_RELATIVE_MAX
    )
    absolute_max = (
        BF16_SCALAR_ABSOLUTE_MAX
        if compute_dtype == "bfloat16"
        else SCALAR_ABSOLUTE_MAX
    )
    return {
        "jax": left,
        "torch": right,
        "absolute_error": absolute,
        "relative_error": relative,
        "pass": bool(absolute <= absolute_max or relative <= relative_max),
    }


def _inputs(
    args: argparse.Namespace,
) -> tuple[dict[str, np.ndarray], StepChoices, dict[str, np.ndarray]]:
    torch.set_num_threads(args.threads)
    batches = FixedTrajectoryBatches(
        require_within_workspace(args.data_root / "train"),
        batch_size=args.batch_size,
        horizon=CONFIG.horizon,
        seed=args.seed,
        shuffle_files=True,
        batch_schedule="global_permutation",
    )
    numpy_batch = batches.batch_at(args.data_step)
    choices = materialize_step_choices(
        seed=args.seed,
        update=args.update,
        batch_size=args.batch_size,
        config=CONFIG,
        device=torch.device("cpu"),
    )
    numpy_choices = _numpy_choices(choices)
    return numpy_batch, choices, numpy_choices


def _choices_on_device(
    choices: StepChoices,
    device: torch.device,
) -> StepChoices:
    return StepChoices(*(value.to(device) for value in choices))


def _tolerances(compute_dtype: str) -> dict[str, float]:
    if compute_dtype == "bfloat16":
        return {
            "tensor_relative_l2_max": BF16_RELATIVE_L2_MAX,
            "tensor_cosine_min": BF16_COSINE_MIN,
            "scalar_relative_max": BF16_SCALAR_RELATIVE_MAX,
            "scalar_absolute_max": BF16_SCALAR_ABSOLUTE_MAX,
        }
    return {
        "tensor_relative_l2_max": RELATIVE_L2_MAX,
        "tensor_max_absolute_max": MAX_ABSOLUTE_MAX,
        "scalar_relative_max": SCALAR_RELATIVE_MAX,
        "scalar_absolute_max": SCALAR_ABSOLUTE_MAX,
    }


def _comparison_result(
    args: argparse.Namespace,
    *,
    jax_values: Mapping[str, Any],
    jax_losses: Mapping[str, Any],
    torch_values: Mapping[str, Any],
    torch_losses: Mapping[str, Any],
    source_mapping_sha256: str,
) -> dict[str, Any]:
    tensors: dict[str, dict[str, float | bool]] = {}
    for name in _TENSOR_NAMES:
        candidate = torch_values[name]
        if isinstance(candidate, torch.Tensor):
            candidate = candidate.detach().float().cpu().numpy()
        tensors[name] = _array_metrics(
            jax_values[name],
            candidate,
            compute_dtype=args.compute_dtype,
        )
    scalars: dict[str, dict[str, float | bool]] = {}
    for name in jax_losses:
        candidate = torch_losses[name]
        if isinstance(candidate, torch.Tensor):
            candidate = candidate.detach().float().cpu().numpy()
        scalars[name] = _scalar_metrics(
            jax_losses[name],
            candidate,
            compute_dtype=args.compute_dtype,
        )
    gate_pass = all(row["pass"] for row in tensors.values()) and all(
        row["pass"] for row in scalars.values()
    )
    return {
        "schema_version": "torch-jax-source-parity-v2",
        "gate_pass": gate_pass,
        "compute_dtype": args.compute_dtype,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "data_step": args.data_step,
        "update": args.update,
        "source_mapping_sha256": source_mapping_sha256,
        "tolerances": _tolerances(args.compute_dtype),
        "tensors": tensors,
        "loss_components": scalars,
    }


def _write_result(args: argparse.Namespace, result: Mapping[str, Any]) -> int:
    output = require_within_workspace(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(output), **result}, sort_keys=True))
    return 0 if result["gate_pass"] else 1


def _run_cpu_compare(args: argparse.Namespace) -> int:
    if jax.default_backend() != "cpu":
        raise RuntimeError(f"CPU parity found JAX backend {jax.default_backend()}")
    numpy_batch, choices, numpy_choices = _inputs(args)
    torch_batch = _torch_batch(numpy_batch, torch.device("cpu"))

    source = load_verified_source_model(args.source_state)
    torch_model = JointModel(CONFIG)
    source_mapping = bind_source_model(torch_model, source)
    torch_model.eval()
    jax_model = _build_jax_model(
        source,
        models_dir=args.models_dir,
        compute_dtype=args.compute_dtype,
    )
    del source
    gc.collect()

    with torch.no_grad():
        torch_values = _torch_intermediates(
            torch_model,
            torch_batch,
            choices,
            compute_dtype=(
                torch.float32
                if args.compute_dtype == "float32"
                else torch.bfloat16
            ),
        )
        torch_losses = _torch_loss_components(torch_values, torch_batch, choices)
    jax_values = _jax_intermediates(jax_model, numpy_batch, numpy_choices)
    jax_losses = _jax_loss_components(jax_values, numpy_batch, numpy_choices)
    jax_values, jax_losses = jax.device_get(
        jax.block_until_ready((jax_values, jax_losses))
    )
    result = _comparison_result(
        args,
        jax_values=jax_values,
        jax_losses=jax_losses,
        torch_values=torch_values,
        torch_losses=torch_losses,
        source_mapping_sha256=source_mapping["combined_sha256"],
    )
    return _write_result(args, result)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(4 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _run_torch_export(args: argparse.Namespace) -> int:
    if args.compute_dtype != "bfloat16":
        raise ValueError("Sequential GPU parity requires --compute-dtype bfloat16")
    if not torch.cuda.is_available():
        raise RuntimeError("Torch BF16 export requires CUDA")
    torch.set_num_threads(args.threads)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    numpy_batch, choices_cpu, _ = _inputs(args)
    torch_batch = _torch_batch(numpy_batch, device)
    choices = _choices_on_device(choices_cpu, device)
    model, source_mapping = load_source_model(
        device=device,
        source_path=args.source_state,
    )
    model.eval()
    with torch.no_grad():
        torch_values = _torch_intermediates(
            model,
            torch_batch,
            choices,
            compute_dtype=torch.bfloat16,
        )
        torch_losses = _torch_loss_components(
            torch_values,
            torch_batch,
            choices,
        )
    torch.cuda.synchronize()

    metadata = {
        "schema_version": "torch-bf16-parity-exchange-v1",
        "compute_dtype": args.compute_dtype,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "data_step": args.data_step,
        "update": args.update,
        "source_mapping_sha256": source_mapping["combined_sha256"],
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "device": torch.cuda.get_device_name(device),
        "tolerances": _tolerances(args.compute_dtype),
    }
    payload: dict[str, Any] = {
        "metadata_json": np.asarray(json.dumps(metadata, sort_keys=True)),
    }
    for name in _TENSOR_NAMES:
        payload[f"tensor__{name}"] = (
            torch_values[name].detach().float().cpu().numpy()
        )
    for name, value in torch_losses.items():
        payload[f"loss__{name}"] = value.detach().float().cpu().numpy()

    exchange = require_within_workspace(args.exchange)
    if exchange.exists():
        raise FileExistsError(f"Exchange artifact already exists: {exchange}")
    exchange.parent.mkdir(parents=True, exist_ok=True)
    temporary = exchange.with_name(f".{exchange.name}.partial")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **payload)
    os.replace(temporary, exchange)
    result = {
        **metadata,
        "exchange": str(exchange),
        "exchange_sha256": _sha256(exchange),
        "exchange_size_bytes": exchange.stat().st_size,
    }
    print(json.dumps(result, sort_keys=True))
    return 0


def _load_exchange(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, np.ndarray]]:
    exchange = require_within_workspace(args.exchange)
    with np.load(exchange, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata_json"].item()))
        torch_values = {
            name: np.asarray(archive[f"tensor__{name}"]) for name in _TENSOR_NAMES
        }
        loss_names = (
            "dfm_ce",
            "legality",
            "jepa_positive",
            "target_sigreg",
            "pred_sigreg",
            "unclipped_total",
        )
        torch_losses = {
            name: np.asarray(archive[f"loss__{name}"]) for name in loss_names
        }
    expected = {
        "compute_dtype": args.compute_dtype,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "data_step": args.data_step,
        "update": args.update,
        "source_mapping_sha256": SOURCE_MAPPING_SHA256,
        "tolerances": _tolerances(args.compute_dtype),
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(
                f"Exchange metadata mismatch for {key}: "
                f"{metadata.get(key)!r} != {value!r}"
            )
    return metadata, torch_values, torch_losses


def _run_jax_compare(args: argparse.Namespace) -> int:
    if args.compute_dtype != "bfloat16":
        raise ValueError("Sequential GPU parity requires --compute-dtype bfloat16")
    if jax.default_backend() != "gpu":
        raise RuntimeError(f"JAX BF16 comparison found {jax.default_backend()}")
    metadata, torch_values, torch_losses = _load_exchange(args)
    numpy_batch, _, numpy_choices = _inputs(args)
    source = load_verified_source_model(args.source_state)
    jax_model = _build_jax_model(
        source,
        models_dir=args.models_dir,
        compute_dtype=args.compute_dtype,
    )
    del source
    gc.collect()
    jax_values = _jax_intermediates(jax_model, numpy_batch, numpy_choices)
    jax_losses = _jax_loss_components(jax_values, numpy_batch, numpy_choices)
    jax_values, jax_losses = jax.device_get(
        jax.block_until_ready((jax_values, jax_losses))
    )
    result = _comparison_result(
        args,
        jax_values=jax_values,
        jax_losses=jax_losses,
        torch_values=torch_values,
        torch_losses=torch_losses,
        source_mapping_sha256=metadata["source_mapping_sha256"],
    )
    result["jax_backend"] = jax.default_backend()
    result["exchange"] = str(require_within_workspace(args.exchange))
    result["exchange_sha256"] = _sha256(require_within_workspace(args.exchange))
    return _write_result(args, result)


def run(args: argparse.Namespace) -> int:
    if args.mode == "cpu-compare":
        return _run_cpu_compare(args)
    if args.mode == "torch-export":
        return _run_torch_export(args)
    if args.mode == "jax-compare":
        return _run_jax_compare(args)
    raise ValueError(f"Unsupported mode: {args.mode}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("cpu-compare", "torch-export", "jax-compare"),
        default="cpu-compare",
    )
    parser.add_argument(
        "--source-state",
        type=Path,
        default=(
            _REPO_ROOT
            / "checkpoints/source/step0265000/checkpoints/step0265000/state.npz"
        ),
    )
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    parser.add_argument(
        "--data-root", type=Path, default=_REPO_ROOT / "data/trajectory_v3"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_REPO_ROOT / "artifacts/pytorch/source_fp32_parity.json",
    )
    parser.add_argument(
        "--exchange",
        type=Path,
        default=_REPO_ROOT / "artifacts/pytorch/source_gpu_bf16_exchange.npz",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--data-step", type=int, default=0)
    parser.add_argument("--update", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument(
        "--compute-dtype",
        choices=("float32", "bfloat16"),
        default="float32",
    )
    return parser


if __name__ == "__main__":
    raise SystemExit(run(build_parser().parse_args()))
