#!/usr/bin/env python3
"""CPU FP32 parity gate for the eager-PyTorch migration.

The harness restores the same checksum-pinned model tree into both runtimes,
materializes stochastic choices once, and compares named intermediates and
loss components. It never creates an optimizer or checkpoint.
"""

from __future__ import annotations

import argparse
import dataclasses
import gc
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import torch
from flax import nnx

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
    load_verified_source_model,
    materialize_step_choices,
)


RELATIVE_L2_MAX = 7.5e-4
MAX_ABSOLUTE_MAX = 2e-2
SCALAR_RELATIVE_MAX = 2e-3
SCALAR_ABSOLUTE_MAX = 2e-4


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


def _array_metrics(reference: Any, candidate: Any) -> dict[str, float | bool]:
    left = np.asarray(reference, dtype=np.float64)
    right = np.asarray(candidate, dtype=np.float64)
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch: {left.shape} != {right.shape}")
    difference = right - left
    left_norm = float(np.linalg.norm(left.reshape(-1)))
    relative_l2 = float(
        np.linalg.norm(difference.reshape(-1)) / max(left_norm, 1e-30)
    )
    maximum = float(np.max(np.abs(difference)))
    return {
        "relative_l2_error": relative_l2,
        "max_absolute_error": maximum,
        "pass": bool(
            relative_l2 <= RELATIVE_L2_MAX and maximum <= MAX_ABSOLUTE_MAX
        ),
    }


def _scalar_metrics(reference: Any, candidate: Any) -> dict[str, float | bool]:
    left = float(np.asarray(reference))
    right = float(np.asarray(candidate))
    absolute = abs(right - left)
    relative = absolute / max(abs(left), 1e-12)
    return {
        "jax": left,
        "torch": right,
        "absolute_error": absolute,
        "relative_error": relative,
        "pass": bool(
            absolute <= SCALAR_ABSOLUTE_MAX or relative <= SCALAR_RELATIVE_MAX
        ),
    }


def run(args: argparse.Namespace) -> int:
    if jax.default_backend() != "cpu":
        raise RuntimeError(f"Parity requires CPU JAX, found {jax.default_backend()}")
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
    torch_batch = _torch_batch(numpy_batch, torch.device("cpu"))
    choices = materialize_step_choices(
        seed=args.seed,
        update=args.update,
        batch_size=args.batch_size,
        config=CONFIG,
        device=torch.device("cpu"),
    )
    numpy_choices = _numpy_choices(choices)

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

    tensors: dict[str, dict[str, float | bool]] = {}
    for name in (
        "current_tokens",
        "future_tokens",
        "z_all",
        "z_dfm",
        "noisy_logits",
        "clean_logits",
        "clean_action_hidden",
        "pred_z",
    ):
        tensors[name] = _array_metrics(
            jax_values[name],
            torch_values[name].detach().float().numpy(),
        )
    scalars = {
        name: _scalar_metrics(jax_losses[name], torch_losses[name].detach().numpy())
        for name in jax_losses
    }
    gate_pass = all(row["pass"] for row in tensors.values()) and all(
        row["pass"] for row in scalars.values()
    )
    result = {
        "schema_version": "torch-jax-source-fp32-parity-v1",
        "gate_pass": gate_pass,
        "compute_dtype": args.compute_dtype,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "data_step": args.data_step,
        "update": args.update,
        "source_mapping_sha256": source_mapping["combined_sha256"],
        "tolerances": {
            "tensor_relative_l2_max": RELATIVE_L2_MAX,
            "tensor_max_absolute_max": MAX_ABSOLUTE_MAX,
            "scalar_relative_max": SCALAR_RELATIVE_MAX,
            "scalar_absolute_max": SCALAR_ABSOLUTE_MAX,
        },
        "tensors": tensors,
        "loss_components": scalars,
    }
    output = require_within_workspace(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(output), **result}, sort_keys=True))
    return 0 if gate_pass else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
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
