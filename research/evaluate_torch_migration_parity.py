#!/usr/bin/env python3
"""Cross-framework parity gates for the eager-PyTorch migration.

The harness restores the same checksum-pinned model tree into both runtimes,
materializes stochastic choices once, and compares named intermediates and
loss components. CPU comparison runs both models in one process. Production
BF16 comparison uses two sequential guarded GPU processes and a small NPZ
exchange artifact so the full PyTorch and JAX runtimes never coexist on GPU.
The hero round-trip mode performs an exact CPU-side state materialization
audit, including the trainable BT4 policy residual. The harness never creates
an optimizer or checkpoint.
"""

from __future__ import annotations

import argparse
import dataclasses
import gc
import hashlib
import json
import math
import os
import sys
import time
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


if __name__ == "__main__" and _requested_mode() in {
    "cpu-compare",
    "torch-export",
    "hero-checkpoint-roundtrip",
}:
    os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
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
    evaluate,
    resolve_config,
)
from research.train_torch import (  # noqa: E402
    CONFIG,
    HERO_CONFIG,
    JointModel,
    StepChoices,
    _legal_mass,
    _sigreg_v_stat,
    _torch_batch,
    bind_source_model,
    load_checkpoint_numpy_tree_for_evaluation,
    load_model_checkpoint,
    load_model_checkpoint_numpy_tree,
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
EXPECTED_MODEL_LEAVES = 455
EXPECTED_MODEL_BYTES = 705_987_352
HERO_SOURCE_MAPPING_SHA256 = (
    "c7a22e25a74e959357f294dfdda4a8f47372d79c96ada49131967bec75b2d93c"
)
HERO_EXPECTED_MODEL_LEAVES = 462
HERO_EXPECTED_MODEL_BYTES = 712_293_144
ACTION_VOCAB_SIZE = 1858
_CORE_TENSOR_NAMES = (
    "current_tokens",
    "future_tokens",
    "z_all",
    "z_dfm",
    "noisy_logits",
    "clean_logits",
    "clean_action_hidden",
    "pred_z",
)
_TRACE_TENSOR_NAMES = (
    "encoder_trace",
    "last_layer_attention_out",
    "last_layer_resid_mid",
    "last_layer_mlp_out",
)


def _tensor_names(*, encoder_trace: bool) -> tuple[str, ...]:
    if encoder_trace:
        return (*_TRACE_TENSOR_NAMES, *_CORE_TENSOR_NAMES)
    return _CORE_TENSOR_NAMES


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


def _flatten_state_structure(
    value: Any,
    path: tuple[str | int, ...] = (),
) -> dict[tuple[str | int, ...], Any]:
    if isinstance(value, dict):
        flattened: dict[tuple[str | int, ...], Any] = {}
        for key, child in value.items():
            flattened.update(_flatten_state_structure(child, (*path, key)))
        return flattened
    if isinstance(value, (list, tuple)):
        flattened = {}
        for index, child in enumerate(value):
            flattened.update(_flatten_state_structure(child, (*path, index)))
        return flattened
    return {path: value}


def _checkpoint_tree(
    checkpoint_dir: Path,
) -> tuple[
    dict[str | int, Any],
    dict[str, Any],
    dict[str, Any],
    tuple[str, ...],
]:
    with torch.device("meta"):
        template = JointModel(CONFIG)
    ordered_names = tuple(name for name, _ in template.named_parameters())
    tree, manifest, summary = load_model_checkpoint_numpy_tree(
        checkpoint_dir=checkpoint_dir,
        model=template,
    )
    del template
    if manifest["source_mapping_sha256"] != SOURCE_MAPPING_SHA256:
        raise ValueError(
            "Checkpoint source mapping drift: "
            f"{manifest['source_mapping_sha256']} != {SOURCE_MAPPING_SHA256}"
        )
    if summary["leaf_count"] != EXPECTED_MODEL_LEAVES:
        raise ValueError(
            f"Checkpoint leaf count drift: {summary['leaf_count']} != "
            f"{EXPECTED_MODEL_LEAVES}"
        )
    if summary["nbytes"] != EXPECTED_MODEL_BYTES:
        raise ValueError(
            f"Checkpoint model bytes drift: {summary['nbytes']} != "
            f"{EXPECTED_MODEL_BYTES}"
        )
    return tree, manifest, summary, ordered_names


def _hero_checkpoint_tree(
    checkpoint_dir: Path,
) -> tuple[
    dict[str | int, Any],
    dict[str, Any],
    dict[str, Any],
    tuple[str, ...],
]:
    with torch.device("meta"):
        template = JointModel(HERO_CONFIG)
    ordered_names = tuple(name for name, _ in template.named_parameters())
    tree, manifest, summary = load_checkpoint_numpy_tree_for_evaluation(
        checkpoint_dir=checkpoint_dir,
        model=template,
    )
    del template
    if manifest["source_mapping_sha256"] != HERO_SOURCE_MAPPING_SHA256:
        raise ValueError(
            "Hero checkpoint source mapping drift: "
            f"{manifest['source_mapping_sha256']} != "
            f"{HERO_SOURCE_MAPPING_SHA256}"
        )
    if summary["leaf_count"] != HERO_EXPECTED_MODEL_LEAVES:
        raise ValueError(
            f"Hero checkpoint leaf count drift: {summary['leaf_count']} != "
            f"{HERO_EXPECTED_MODEL_LEAVES}"
        )
    if summary["nbytes"] != HERO_EXPECTED_MODEL_BYTES:
        raise ValueError(
            f"Hero checkpoint model bytes drift: {summary['nbytes']} != "
            f"{HERO_EXPECTED_MODEL_BYTES}"
        )
    return tree, manifest, summary, ordered_names


def _split_hero_checkpoint_tree(
    tree: Mapping[str | int, Any],
) -> tuple[dict[str | int, Any], dict[str | int, Any]]:
    core_tree = dict(tree)
    encoder = core_tree.get("encoder")
    if not isinstance(encoder, dict):
        raise ValueError("Hero checkpoint tree has no encoder object")
    core_encoder = dict(encoder)
    policy_tree = core_encoder.pop("policy_head", None)
    if not isinstance(policy_tree, dict):
        raise ValueError("Hero checkpoint tree has no BT4 policy head")
    core_tree["encoder"] = core_encoder
    policy_leaf_count = len(_flatten_state_structure(policy_tree))
    if policy_leaf_count != (
        HERO_EXPECTED_MODEL_LEAVES - EXPECTED_MODEL_LEAVES
    ):
        raise ValueError(
            "Hero checkpoint policy-head leaf count mismatch: "
            f"{policy_leaf_count}"
        )
    return core_tree, dict(policy_tree)


def _apply_and_verify_jax_hero_checkpoint(
    model: JointLatentSASAModel,
    tree: Mapping[str | int, Any],
    *,
    ordered_names: tuple[str, ...],
    expected_summary: Mapping[str, Any],
) -> dict[str, Any]:
    core_tree, policy_tree = _split_hero_checkpoint_tree(tree)
    core_state = nnx.state(model, TrainableParam)
    nnx.replace_by_pure_dict(core_state, core_tree)
    nnx.update(model, core_state)

    policy_state = nnx.state(model.encoder.policy_head, nnx.Param)
    nnx.replace_by_pure_dict(policy_state, policy_tree)
    nnx.update(model.encoder.policy_head, policy_state)

    expected_flat = _flatten_state_structure(tree)
    observed_flat = _flatten_state_structure(
        dict(nnx.to_pure_dict(nnx.state(model, TrainableParam)))
    )
    policy_flat = _flatten_state_structure(
        dict(nnx.to_pure_dict(nnx.state(model.encoder.policy_head, nnx.Param)))
    )
    policy_prefix = ("encoder", "policy_head")
    for path, value in policy_flat.items():
        full_path = (*policy_prefix, *path)
        if full_path in expected_flat:
            observed_flat[full_path] = value
    if set(observed_flat) != set(expected_flat):
        raise ValueError(
            "JAX hero checkpoint state leaf mismatch after update: "
            f"missing={list(set(expected_flat) - set(observed_flat))[:10]}, "
            f"extra={list(set(observed_flat) - set(expected_flat))[:10]}"
        )

    combined = hashlib.sha256()
    total_bytes = 0
    for name in ordered_names:
        parts: tuple[str | int, ...] = tuple(
            int(part) if part.isdigit() else part for part in name.split(".")
        )
        expected = np.asarray(expected_flat[parts])
        observed = np.asarray(jax.device_get(observed_flat[parts]))
        if observed.shape != expected.shape or observed.dtype != expected.dtype:
            raise ValueError(
                f"JAX hero checkpoint ABI mismatch at {name}: "
                f"{observed.shape}/{observed.dtype} != "
                f"{expected.shape}/{expected.dtype}"
            )
        expected_bytes = expected.tobytes(order="C")
        observed_bytes = observed.tobytes(order="C")
        if observed_bytes != expected_bytes:
            raise ValueError(f"JAX hero checkpoint value mismatch at {name}")
        leaf_digest = hashlib.sha256(observed_bytes).hexdigest()
        combined.update(name.encode("utf-8"))
        combined.update(b"\0")
        combined.update(leaf_digest.encode("ascii"))
        total_bytes += int(observed.nbytes)
    observed_summary = {
        "schema_version": "torch-hero-to-jax-model-roundtrip-v1",
        "leaf_count": len(ordered_names),
        "core_trainable_leaf_count": EXPECTED_MODEL_LEAVES,
        "bt4_policy_head_leaf_count": (
            HERO_EXPECTED_MODEL_LEAVES - EXPECTED_MODEL_LEAVES
        ),
        "nbytes": total_bytes,
        "combined_state_sha256": combined.hexdigest(),
        "exact_shape_dtype_and_value_match": True,
    }
    if (
        observed_summary["combined_state_sha256"]
        != expected_summary["combined_state_sha256"]
    ):
        raise ValueError(
            "JAX hero checkpoint combined state checksum mismatch: "
            f"{observed_summary['combined_state_sha256']} != "
            f"{expected_summary['combined_state_sha256']}"
        )
    return observed_summary


def _apply_and_verify_jax_checkpoint(
    model: JointLatentSASAModel,
    tree: Mapping[str | int, Any],
    *,
    ordered_names: tuple[str, ...],
    expected_summary: Mapping[str, Any],
) -> dict[str, Any]:
    state = nnx.state(model, TrainableParam)
    nnx.replace_by_pure_dict(state, tree)
    nnx.update(model, state)

    observed_state = dict(nnx.to_pure_dict(nnx.state(model, TrainableParam)))
    expected_flat = _flatten_state_structure(tree)
    observed_flat = _flatten_state_structure(observed_state)
    if set(observed_flat) != set(expected_flat):
        raise ValueError(
            "JAX checkpoint state leaf mismatch after update: "
            f"missing={list(set(expected_flat) - set(observed_flat))[:10]}, "
            f"extra={list(set(observed_flat) - set(expected_flat))[:10]}"
        )
    combined = hashlib.sha256()
    total_bytes = 0
    for name in ordered_names:
        parts: tuple[str | int, ...] = tuple(
            int(part) if part.isdigit() else part for part in name.split(".")
        )
        expected = np.asarray(expected_flat[parts])
        observed = np.asarray(jax.device_get(observed_flat[parts]))
        if observed.shape != expected.shape or observed.dtype != expected.dtype:
            raise ValueError(
                f"JAX checkpoint ABI mismatch at {name}: "
                f"{observed.shape}/{observed.dtype} != "
                f"{expected.shape}/{expected.dtype}"
            )
        expected_bytes = expected.tobytes(order="C")
        observed_bytes = observed.tobytes(order="C")
        if observed_bytes != expected_bytes:
            raise ValueError(f"JAX checkpoint value mismatch at {name}")
        leaf_digest = hashlib.sha256(observed_bytes).hexdigest()
        combined.update(name.encode("utf-8"))
        combined.update(b"\0")
        combined.update(leaf_digest.encode("ascii"))
        total_bytes += int(observed.nbytes)
    observed_summary = {
        "schema_version": "torch-to-jax-model-roundtrip-v1",
        "leaf_count": len(ordered_names),
        "nbytes": total_bytes,
        "combined_state_sha256": combined.hexdigest(),
        "exact_shape_dtype_and_value_match": True,
    }
    if (
        observed_summary["combined_state_sha256"]
        != expected_summary["combined_state_sha256"]
    ):
        raise ValueError(
            "JAX checkpoint combined state checksum mismatch: "
            f"{observed_summary['combined_state_sha256']} != "
            f"{expected_summary['combined_state_sha256']}"
        )
    return observed_summary


def _numpy_choices(choices: StepChoices) -> dict[str, np.ndarray]:
    return {
        field: np.asarray(tensor.detach().cpu())
        for field, tensor in zip(StepChoices._fields, choices, strict=True)
    }


def _torch_bt4_layer_capture(
    layer: torch.nn.Module,
    x: torch.Tensor,
    *,
    alpha: float,
    compute_dtype: torch.dtype,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Mirror one production BT4 layer and expose the JAX capture boundaries."""

    batch, sequence, _ = x.shape
    q = x @ layer.wq.to(compute_dtype) + layer.wq_b.to(compute_dtype)
    k = x @ layer.wk.to(compute_dtype) + layer.wk_b.to(compute_dtype)
    v = x @ layer.wv.to(compute_dtype) + layer.wv_b.to(compute_dtype)
    q = q.reshape(batch, sequence, 32, 32).transpose(1, 2)
    k = k.reshape(batch, sequence, 32, 32).transpose(1, 2)
    v = v.reshape(batch, sequence, 32, 32).transpose(1, 2)
    logits = (q @ k.transpose(-2, -1)) / math.sqrt(32.0)
    logits = logits + layer.smolgen(x, compute_dtype)
    attention = F.softmax(logits, dim=-1)
    attention_out = attention @ v
    attention_out = attention_out.transpose(1, 2).reshape(
        batch * sequence, 1024
    )
    attention_out = layer.wo(attention_out, compute_dtype).reshape(
        batch, sequence, 1024
    )
    resid_mid = layer.ln_attn(attention_out * alpha + x, compute_dtype)
    flat = resid_mid.reshape(batch * sequence, 1024)
    mlp_out = layer.ffn2(
        F.mish(layer.ffn1(flat, compute_dtype)),
        compute_dtype,
    ).reshape(batch, sequence, 1024)
    resid_post = layer.ln_ffn(mlp_out * alpha + resid_mid, compute_dtype)
    return resid_post, {
        "last_layer_attention_out": attention_out,
        "last_layer_resid_mid": resid_mid,
        "last_layer_mlp_out": mlp_out,
    }


def _torch_intermediates(
    model: JointModel,
    batch: Mapping[str, torch.Tensor],
    choices: StepChoices,
    *,
    compute_dtype: torch.dtype,
    encoder_trace: bool = False,
    final_bt4_fp32: bool = False,
) -> dict[str, torch.Tensor]:
    if final_bt4_fp32 and not encoder_trace:
        raise ValueError("Final-BT4-FP32 diagnostic requires encoder tracing")
    dtype = compute_dtype
    actions = batch["action_indices"][:, : CONFIG.horizon].long()
    rows = torch.arange(actions.shape[0])
    selected_planes = batch["future_planes"][rows, choices.target_horizon]
    diagnostics: dict[str, torch.Tensor] = {}
    if encoder_trace:
        current_tokens = model.encoder.embedding(
            batch["current_planes"],
            model.encoder.alpha,
            dtype,
        )
        trace_values = [current_tokens]
        last_layer_capture: dict[str, torch.Tensor] | None = None
        for index, layer in enumerate(model.encoder.layers):
            if index == len(model.encoder.layers) - 1:
                layer_dtype = torch.float32 if final_bt4_fp32 else dtype
                layer_input = current_tokens.to(layer_dtype)
                captured_output, last_layer_capture = _torch_bt4_layer_capture(
                    layer,
                    layer_input,
                    alpha=model.encoder.alpha,
                    compute_dtype=layer_dtype,
                )
                current_tokens = layer(
                    layer_input,
                    model.encoder.alpha,
                    layer_dtype,
                )
                torch.testing.assert_close(
                    current_tokens,
                    captured_output,
                    rtol=0.0,
                    atol=0.0,
                )
            else:
                current_tokens = layer(
                    current_tokens,
                    model.encoder.alpha,
                    dtype,
                )
            trace_values.append(current_tokens)
        assert last_layer_capture is not None
        diagnostics = {
            "encoder_trace": torch.stack(trace_values),
            **last_layer_capture,
        }
    else:
        current_tokens = model.encoder.encode_current(
            batch["current_planes"],
            compute_dtype=dtype,
            remat=False,
        )

    if encoder_trace and final_bt4_fp32:
        future_tokens = model.encoder.embedding(
            selected_planes,
            model.encoder.alpha,
            dtype,
        )
        for layer in model.encoder.layers[:-1]:
            future_tokens = layer(
                future_tokens,
                model.encoder.alpha,
                dtype,
            )
        future_tokens = model.encoder.layers[-1](
            future_tokens.float(),
            model.encoder.alpha,
            torch.float32,
        )
    else:
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
        **diagnostics,
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
    *,
    encoder_trace: bool = False,
) -> dict[str, jax.Array]:
    actions = jnp.asarray(batch["action_indices"][:, : CONFIG.horizon])
    rows = jnp.arange(actions.shape[0])
    selected = jnp.asarray(choices["target_horizon"])
    selected_planes = jnp.asarray(batch["future_planes"])[rows, selected]
    diagnostics: dict[str, jax.Array] = {}
    if encoder_trace:
        alpha = float((2.0 * len(model.encoder.layers)) ** -0.25)
        current_tokens, current_batch = model.encoder.embedding(
            jnp.asarray(batch["current_planes"]),
            alpha,
        )
        current_tokens = current_tokens.reshape(
            (current_batch, 64, model.encoder_dim)
        )
        trace_values = [current_tokens]
        last_layer_capture = None
        for index, layer in enumerate(model.encoder.layers):
            if index == len(model.encoder.layers) - 1:
                current_tokens, last_layer_capture = layer.forward_with_capture(
                    current_tokens,
                    alpha,
                )
            else:
                current_tokens = layer(current_tokens, alpha)
            trace_values.append(current_tokens)
        assert last_layer_capture is not None
        diagnostics = {
            "encoder_trace": jnp.stack(trace_values),
            "last_layer_attention_out": last_layer_capture.hook_attn_out,
            "last_layer_resid_mid": last_layer_capture.resid_mid_after_ln,
            "last_layer_mlp_out": last_layer_capture.hook_mlp_out,
        }
    else:
        current_tokens = model.encode_bt4_tokens(
            jnp.asarray(batch["current_planes"])
        )
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
        **diagnostics,
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
    for name in _tensor_names(encoder_trace=args.encoder_trace):
        candidate = torch_values[name]
        if isinstance(candidate, torch.Tensor):
            candidate = candidate.detach().float().cpu().numpy()
        tensors[name] = _array_metrics(
            jax_values[name],
            candidate,
            compute_dtype=args.compute_dtype,
        )
    trace_stages: list[dict[str, Any]] = []
    if args.encoder_trace:
        trace_reference = np.asarray(jax_values["encoder_trace"])
        trace_candidate = torch_values["encoder_trace"]
        if isinstance(trace_candidate, torch.Tensor):
            trace_candidate = trace_candidate.detach().float().cpu().numpy()
        trace_stages = [
            {
                "stage": (
                    "embedding" if index == 0 else f"layer_{index - 1:02d}"
                ),
                **_array_metrics(
                    trace_reference[index],
                    trace_candidate[index],
                    compute_dtype=args.compute_dtype,
                ),
            }
            for index in range(trace_reference.shape[0])
        ]
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
    result = {
        "schema_version": "torch-jax-source-parity-v2",
        "gate_pass": gate_pass,
        "compute_dtype": args.compute_dtype,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "data_step": args.data_step,
        "update": args.update,
        "encoder_trace": args.encoder_trace,
        "torch_final_bt4_fp32": args.torch_final_bt4_fp32,
        "source_mapping_sha256": source_mapping_sha256,
        "tolerances": _tolerances(args.compute_dtype),
        "tensors": tensors,
        "loss_components": scalars,
    }
    if args.encoder_trace:
        result["encoder_trace_stages"] = trace_stages
    return result


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
            encoder_trace=args.encoder_trace,
            final_bt4_fp32=args.torch_final_bt4_fp32,
        )
        torch_losses = _torch_loss_components(torch_values, torch_batch, choices)
    jax_values = _jax_intermediates(
        jax_model,
        numpy_batch,
        numpy_choices,
        encoder_trace=args.encoder_trace,
    )
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
    checkpoint_manifest = None
    if args.checkpoint_dir is not None:
        checkpoint_manifest = load_model_checkpoint(
            checkpoint_dir=args.checkpoint_dir,
            model=model,
        )
        if (
            checkpoint_manifest["source_mapping_sha256"]
            != source_mapping["combined_sha256"]
        ):
            raise ValueError("Checkpoint/source mapping mismatch")
    model.eval()
    with torch.no_grad():
        torch_values = _torch_intermediates(
            model,
            torch_batch,
            choices,
            compute_dtype=torch.bfloat16,
            encoder_trace=args.encoder_trace,
            final_bt4_fp32=args.torch_final_bt4_fp32,
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
        "encoder_trace": args.encoder_trace,
        "torch_final_bt4_fp32": args.torch_final_bt4_fp32,
        "source_mapping_sha256": source_mapping["combined_sha256"],
        "checkpoint_state_sha256": (
            None
            if checkpoint_manifest is None
            else checkpoint_manifest["state"]["sha256"]
        ),
        "checkpoint_optimizer_update": (
            None
            if checkpoint_manifest is None
            else checkpoint_manifest["optimizer_update"]
        ),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "device": torch.cuda.get_device_name(device),
        "tolerances": _tolerances(args.compute_dtype),
    }
    payload: dict[str, Any] = {
        "metadata_json": np.asarray(json.dumps(metadata, sort_keys=True)),
    }
    for name in _tensor_names(encoder_trace=args.encoder_trace):
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
            name: np.asarray(archive[f"tensor__{name}"])
            for name in _tensor_names(encoder_trace=args.encoder_trace)
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
        "encoder_trace": args.encoder_trace,
        "torch_final_bt4_fp32": args.torch_final_bt4_fp32,
        "source_mapping_sha256": SOURCE_MAPPING_SHA256,
        "tolerances": _tolerances(args.compute_dtype),
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(
                f"Exchange metadata mismatch for {key}: "
                f"{metadata.get(key)!r} != {value!r}"
            )
    checkpoint_exported = metadata.get("checkpoint_state_sha256") is not None
    if checkpoint_exported != (args.checkpoint_dir is not None):
        raise ValueError(
            "Exchange checkpoint presence differs from --checkpoint-dir: "
            f"{checkpoint_exported} != {args.checkpoint_dir is not None}"
        )
    return metadata, torch_values, torch_losses


def _run_jax_compare(args: argparse.Namespace) -> int:
    if args.compute_dtype != "bfloat16":
        raise ValueError("Sequential GPU parity requires --compute-dtype bfloat16")
    if jax.default_backend() != "gpu":
        raise RuntimeError(f"JAX BF16 comparison found {jax.default_backend()}")
    metadata, torch_values, torch_losses = _load_exchange(args)
    numpy_batch, _, numpy_choices = _inputs(args)
    checkpoint_manifest = None
    checkpoint_tree_summary = None
    roundtrip_summary = None
    if args.checkpoint_dir is None:
        source = load_verified_source_model(args.source_state)
        jax_model = _build_jax_model(
            source,
            models_dir=args.models_dir,
            compute_dtype=args.compute_dtype,
        )
        del source
    else:
        tree, checkpoint_manifest, checkpoint_tree_summary, ordered_names = (
            _checkpoint_tree(args.checkpoint_dir)
        )
        if (
            metadata["checkpoint_state_sha256"]
            != checkpoint_manifest["state"]["sha256"]
        ):
            raise ValueError("Torch exchange/checkpoint state checksum mismatch")
        if (
            metadata["checkpoint_optimizer_update"]
            != checkpoint_manifest["optimizer_update"]
        ):
            raise ValueError("Torch exchange/checkpoint update mismatch")
        jax_model = _build_jax_model(
            tree,
            models_dir=args.models_dir,
            compute_dtype=args.compute_dtype,
        )
        roundtrip_summary = _apply_and_verify_jax_checkpoint(
            jax_model,
            tree,
            ordered_names=ordered_names,
            expected_summary=checkpoint_tree_summary,
        )
        del tree
    gc.collect()
    jax_values = _jax_intermediates(
        jax_model,
        numpy_batch,
        numpy_choices,
        encoder_trace=args.encoder_trace,
    )
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
    result["checkpoint_manifest"] = checkpoint_manifest
    result["checkpoint_tree"] = checkpoint_tree_summary
    result["jax_roundtrip"] = roundtrip_summary
    return _write_result(args, result)


def _root_legal_mask(batch: Mapping[str, Any]) -> np.ndarray:
    legal_idx = np.asarray(batch["legal_idx"])[:, 0]
    legal_count = np.asarray(batch["legal_count"])[:, 0].astype(np.int64)
    batch_size = legal_idx.shape[0]
    if "legal_masks_valid" in batch and not np.all(
        np.asarray(batch["legal_masks_valid"])[:, 0]
    ):
        raise ValueError("Frozen inference batch contains invalid root legal metadata")
    if np.any(legal_count <= 0) or np.any(legal_count > legal_idx.shape[1]):
        raise ValueError("Frozen inference batch has invalid root legal counts")
    mask = np.zeros((batch_size, ACTION_VOCAB_SIZE), dtype=np.bool_)
    slots = np.arange(legal_idx.shape[1])[None, :] < legal_count[:, None]
    rows = np.broadcast_to(np.arange(batch_size)[:, None], legal_idx.shape)
    selected = legal_idx.astype(np.int64)
    if np.any((selected[slots] < 0) | (selected[slots] >= ACTION_VOCAB_SIZE)):
        raise ValueError("Frozen inference batch contains out-of-range legal actions")
    mask[rows[slots], selected[slots]] = True
    return mask


def _jax_inference_gate(
    model: JointLatentSASAModel,
    batch: Mapping[str, Any],
) -> dict[str, Any]:
    from chess_dfm_jax.policy import ACTION_CODEC_LEGACY_ABSOLUTE_1858
    from research.inference import infer_dfm_from_current

    legal_mask = _root_legal_mask(batch)
    started = time.perf_counter()
    result = infer_dfm_from_current(
        model,
        np.asarray(batch["current_planes"]),
        legal_mask,
        refinement_passes=8,
        trace_top_k=8,
        action_codec_id=ACTION_CODEC_LEGACY_ABSOLUTE_1858,
    )
    actions, trace = jax.device_get(jax.block_until_ready(result))
    seconds = time.perf_counter() - started
    actions = np.asarray(actions, dtype=np.int32)
    root_actions = actions[:, 0]
    root_in_range = (root_actions >= 0) & (root_actions < ACTION_VOCAB_SIZE)
    safe_root_actions = np.clip(root_actions, 0, ACTION_VOCAB_SIZE - 1)
    root_legal = root_in_range & legal_mask[
        np.arange(actions.shape[0]),
        safe_root_actions,
    ]
    all_filled = bool(
        np.all((actions >= 0) & (actions < ACTION_VOCAB_SIZE))
    )
    gate_pass = bool(np.all(root_legal) and all_filled)
    return {
        "gate_pass": gate_pass,
        "batch_size": int(actions.shape[0]),
        "refinement_passes": 8,
        "action_codec_id": ACTION_CODEC_LEGACY_ABSOLUTE_1858,
        "compile_and_first_call_seconds": seconds,
        "all_final_root_actions_legal": bool(np.all(root_legal)),
        "all_final_action_slots_filled": all_filled,
        "final_actions_sha256": hashlib.sha256(
            actions.tobytes(order="C")
        ).hexdigest(),
        "first_root_actions": root_actions[:16].tolist(),
        "trace": {
            "times": np.asarray(trace.times).tolist(),
            "mean_root_raw_entropy_by_pass": np.asarray(
                trace.root_raw_entropy
            ).mean(axis=1).tolist(),
            "mean_root_raw_legal_mass_by_pass": np.asarray(
                trace.root_raw_legal_mass
            ).mean(axis=1).tolist(),
            "mean_root_legal_entropy_by_pass": np.asarray(
                trace.root_legal_entropy
            ).mean(axis=1).tolist(),
        },
    }


def _jax_validation_records(
    model: JointLatentSASAModel,
    validation_batches: Mapping[int, FixedTrajectoryBatches],
    *,
    label: str,
    eval_batches: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for seed, batches in validation_batches.items():
        metrics, seconds = evaluate(
            model,
            batches,
            count=eval_batches,
            seed=seed,
            deterministic_t=0.0,
            objective="normalized",
            sigreg_reference_count=1.0,
            collapse_diagnostics=True,
        )
        if not all(math.isfinite(value) for value in metrics.values()):
            raise FloatingPointError(
                f"Non-finite frozen JAX validation metric for {label}/seed {seed}"
            )
        records.append(
            {
                "state": label,
                "validation_seed": seed,
                "eval_batch_size": batches.batch_size,
                "eval_batches": eval_batches,
                "validation_examples": batches.batch_size * eval_batches,
                "validation_seconds": seconds,
                "validation": metrics,
            }
        )
    inference = _jax_inference_gate(
        model,
        validation_batches[next(iter(validation_batches))].batch_at(0),
    )
    return records, inference


def _run_jax_checkpoint_eval(args: argparse.Namespace) -> int:
    if jax.default_backend() != "gpu":
        raise RuntimeError(
            f"JAX checkpoint evaluation found backend {jax.default_backend()}"
        )
    if args.checkpoint_dir is None:
        raise ValueError("--mode jax-checkpoint-eval requires --checkpoint-dir")
    if args.eval_batch_size != CONFIG.sigreg_example_count:
        raise ValueError(
            "Frozen JAX checkpoint evaluation requires --eval-batch-size "
            f"{CONFIG.sigreg_example_count}"
        )
    if args.eval_batches < 1:
        raise ValueError("--eval-batches must be positive")
    if len(set(args.eval_seeds)) != len(args.eval_seeds):
        raise ValueError("--eval-seeds must not contain duplicates")

    validation_batches = {
        seed: FixedTrajectoryBatches(
            require_within_workspace(args.data_root / "val"),
            batch_size=args.eval_batch_size,
            horizon=CONFIG.horizon,
            seed=seed,
            shuffle_files=True,
            batch_schedule="global_permutation",
        )
        for seed in args.eval_seeds
    }
    records: list[dict[str, Any]] = []
    inference: dict[str, Any] = {}
    if args.include_source:
        source = load_verified_source_model(args.source_state)
        model = _build_jax_model(
            source,
            models_dir=args.models_dir,
            compute_dtype="bfloat16",
        )
        del source
        gc.collect()
        source_records, source_inference = _jax_validation_records(
            model,
            validation_batches,
            label="source",
            eval_batches=args.eval_batches,
        )
        records.extend(source_records)
        inference["source"] = source_inference
    else:
        model = None

    tree, manifest, tree_summary, ordered_names = _checkpoint_tree(
        args.checkpoint_dir
    )
    if model is None:
        model = _build_jax_model(
            tree,
            models_dir=args.models_dir,
            compute_dtype="bfloat16",
        )
    roundtrip = _apply_and_verify_jax_checkpoint(
        model,
        tree,
        ordered_names=ordered_names,
        expected_summary=tree_summary,
    )
    del tree
    gc.collect()
    label = f"checkpoint_u{manifest['optimizer_update']}"
    checkpoint_records, checkpoint_inference = _jax_validation_records(
        model,
        validation_batches,
        label=label,
        eval_batches=args.eval_batches,
    )
    records.extend(checkpoint_records)
    inference[label] = checkpoint_inference

    aggregate: dict[str, dict[str, float]] = {}
    for state_label in sorted({record["state"] for record in records}):
        matching = [
            record["validation"]
            for record in records
            if record["state"] == state_label
        ]
        aggregate[state_label] = {
            key: float(np.mean([row[key] for row in matching]))
            for key in matching[0]
        }
    delta_from_source = None
    if args.include_source:
        delta_from_source = {
            key: aggregate[label][key] - aggregate["source"][key]
            for key in aggregate[label]
        }
    result = {
        "schema_version": "torch-checkpoint-frozen-jax-evaluation-v1",
        "gate_pass": bool(
            roundtrip["exact_shape_dtype_and_value_match"]
            and all(row["gate_pass"] for row in inference.values())
        ),
        "jax_backend": jax.default_backend(),
        "jax_device": jax.devices()[0].device_kind,
        "checkpoint_dir": str(require_within_workspace(args.checkpoint_dir)),
        "checkpoint_manifest": manifest,
        "checkpoint_tree": tree_summary,
        "jax_roundtrip": roundtrip,
        "include_source": bool(args.include_source),
        "eval_batch_size": args.eval_batch_size,
        "eval_batches": args.eval_batches,
        "eval_seeds": list(args.eval_seeds),
        "validation_schedule": (
            "one global permutation per seed; batch indexes "
            "[0, eval_batches) reused for every evaluated state"
        ),
        "records": records,
        "two_pool_mean_by_state": aggregate,
        "checkpoint_minus_source": delta_from_source,
        "eight_pass_inference": inference,
        "scientific_promotion_decision": (
            "not evaluated by the migration bridge"
        ),
    }
    return _write_result(args, result)


def _run_hero_checkpoint_roundtrip(args: argparse.Namespace) -> int:
    if jax.default_backend() != "cpu":
        raise RuntimeError(
            "Hero checkpoint round-trip requires the JAX CPU backend"
        )
    if args.checkpoint_dir is None:
        raise ValueError(
            "--mode hero-checkpoint-roundtrip requires --checkpoint-dir"
        )
    tree, manifest, tree_summary, ordered_names = _hero_checkpoint_tree(
        args.checkpoint_dir
    )
    core_tree, _ = _split_hero_checkpoint_tree(tree)
    model = _build_jax_model(
        core_tree,
        models_dir=args.models_dir,
        compute_dtype="float32",
    )
    roundtrip = _apply_and_verify_jax_hero_checkpoint(
        model,
        tree,
        ordered_names=ordered_names,
        expected_summary=tree_summary,
    )
    result = {
        "schema_version": "torch-hero-checkpoint-jax-roundtrip-audit-v1",
        "gate_pass": bool(roundtrip["exact_shape_dtype_and_value_match"]),
        "jax_backend": jax.default_backend(),
        "checkpoint_dir": str(require_within_workspace(args.checkpoint_dir)),
        "checkpoint_format": manifest["format"],
        "checkpoint_optimizer_update": manifest["optimizer_update"],
        "checkpoint_state_sha256": manifest["state"]["sha256"],
        "checkpoint_tree": tree_summary,
        "jax_roundtrip": roundtrip,
        "compatibility": {
            "jax_inference_state_materialization_supported": True,
            "jax_hero_forward_parity_evaluated": False,
            "jax_hero_training_resume_supported": False,
            "limitations": [
                (
                    "The current JAX model stores the seven BT4 policy-head "
                    "leaves as fixed Param variables."
                ),
                (
                    "The current JAX planner does not add the trained BT4 "
                    "policy logits to the canonical root logits."
                ),
                (
                    "Optimizer state is intentionally not materialized by "
                    "this model-only audit."
                ),
            ],
        },
    }
    return _write_result(args, result)


def run(args: argparse.Namespace) -> int:
    if args.torch_final_bt4_fp32 and not args.encoder_trace:
        raise ValueError("--torch-final-bt4-fp32 requires --encoder-trace")
    if args.mode == "cpu-compare":
        return _run_cpu_compare(args)
    if args.mode == "torch-export":
        return _run_torch_export(args)
    if args.mode == "jax-compare":
        return _run_jax_compare(args)
    if args.mode == "jax-checkpoint-eval":
        return _run_jax_checkpoint_eval(args)
    if args.mode == "hero-checkpoint-roundtrip":
        return _run_hero_checkpoint_roundtrip(args)
    raise ValueError(f"Unsupported mode: {args.mode}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=(
            "cpu-compare",
            "torch-export",
            "jax-compare",
            "jax-checkpoint-eval",
            "hero-checkpoint-roundtrip",
        ),
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
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--include-source", action="store_true")
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
        "--eval-seeds",
        type=int,
        nargs="+",
        default=(10000, 20000),
    )
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--eval-batches", type=int, default=64)
    parser.add_argument(
        "--compute-dtype",
        choices=("float32", "bfloat16"),
        default="float32",
    )
    parser.add_argument("--encoder-trace", action="store_true")
    parser.add_argument("--torch-final-bt4-fp32", action="store_true")
    return parser


if __name__ == "__main__":
    raise SystemExit(run(build_parser().parse_args()))
