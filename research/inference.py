"""Fast, checked inference for the local JointLatentSASAModel.

This module deliberately depends only on the localized model's small inference
interface.  It does not construct a model, restore a checkpoint, encode chess
boards, remap actions, or import the legacy joint-training implementation.

The recovered DFM was trained with ``legacy_absolute_1858`` action indices.
Callers must build root legal masks in that same codec and state the codec ID
explicitly.  Empty or malformed masks fail closed before the compiled inference
kernel runs.
"""

from __future__ import annotations

import dataclasses
import time
from typing import Any, NamedTuple, Protocol

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from chess_dfm_jax.policy import (
    ACTION_CODEC_LEGACY_ABSOLUTE_1858,
    ACTION_VOCAB_SIZE,
)


class _DFMConfig(Protocol):
    horizon: int
    action_vocab_size: int
    jepa_feedback_mode: str


class LocalDFMInferenceModel(Protocol):
    """Structural interface used from the localized research model."""

    config: _DFMConfig

    def encode_bt4_tokens(self, planes: jax.Array) -> jax.Array: ...

    def dfm_latents(self, bt4_tokens: jax.Array) -> jax.Array: ...

    def jepa_latents(self, bt4_tokens: jax.Array) -> jax.Array: ...

    def jepa_step_from_latents(
        self,
        z0_jepa: jax.Array,
        action: jax.Array,
        action_hidden: jax.Array,
        *,
        z0_normalized: bool = False,
    ) -> jax.Array: ...

    def dfm_latents_with_jepa_feedback(
        self,
        z_dfm: jax.Array,
        z0_jepa: jax.Array,
        z1_jepa: jax.Array,
        feedback_gate: jax.Array,
    ) -> Any: ...

    def planner_from_latents(
        self,
        z_dfm: jax.Array,
        action_tokens: jax.Array,
        t: jax.Array,
        *,
        return_hidden: bool = False,
    ) -> jax.Array: ...


class DFMRefinementTrace(NamedTuple):
    """Compact per-pass state for policy/refinement diagnostics.

    ``root_raw_*`` describes the model distribution before legality masking.
    ``root_legal_*`` describes the legal-conditioned decision distribution
    actually used by inference.  Top-k entries beyond the number of legal moves
    have ``root_legal_topk_valid=False`` and probability zero.
    """

    times: jax.Array
    actions_before: jax.Array
    actions_after: jax.Array
    root_raw_entropy: jax.Array
    root_raw_legal_mass: jax.Array
    root_legal_entropy: jax.Array
    root_legal_topk_indices: jax.Array
    root_legal_topk_probabilities: jax.Array
    root_legal_topk_valid: jax.Array


class DFMInferenceResult(NamedTuple):
    actions: jax.Array
    trace: DFMRefinementTrace


@dataclasses.dataclass(frozen=True)
class DFMInferenceBenchmark:
    """Wall-clock summary from one fixed batch and refinement count."""

    batch_size: int
    refinement_passes: int
    trace_top_k: int
    compile_and_first_call_seconds: float
    steady_latency_p50_seconds: float
    steady_latency_p95_seconds: float
    steady_examples_per_second: float
    warmup_calls: int
    measured_calls: int


def _model_dimensions(model: LocalDFMInferenceModel) -> tuple[int, int]:
    try:
        horizon = int(model.config.horizon)
        action_vocab_size = int(model.config.action_vocab_size)
    except (AttributeError, TypeError, ValueError) as exc:
        raise TypeError("model.config must expose integer horizon and action_vocab_size") from exc
    if horizon < 1:
        raise ValueError(f"model.config.horizon must be >= 1, got {horizon}")
    if action_vocab_size != ACTION_VOCAB_SIZE:
        raise ValueError(
            "legacy_absolute_1858 inference requires "
            f"action_vocab_size={ACTION_VOCAB_SIZE}, got {action_vocab_size}"
        )
    feedback_mode = getattr(model.config, "jepa_feedback_mode", "none")
    if feedback_mode not in ("none", "final_pass_adjoint"):
        raise ValueError(
            "Unsupported jepa_feedback_mode for local inference: "
            f"{feedback_mode!r}"
        )
    if feedback_mode == "final_pass_adjoint" and horizon != 8:
        raise ValueError(
            "jepa_feedback_mode='final_pass_adjoint' requires horizon=8, "
            f"got {horizon}"
        )
    return horizon, action_vocab_size


def _feedback_mode(model: LocalDFMInferenceModel) -> str:
    return str(getattr(model.config, "jepa_feedback_mode", "none"))


def _validate_feedback_refinement_count(
    model: LocalDFMInferenceModel,
    *,
    refinement_passes: int,
) -> None:
    if (
        _feedback_mode(model) == "final_pass_adjoint"
        and refinement_passes != 8
    ):
        raise ValueError(
            "jepa_feedback_mode='final_pass_adjoint' requires exactly "
            f"8 refinement passes, got {refinement_passes}"
        )


def _validate_refinement_options(
    *,
    refinement_passes: int,
    trace_top_k: int,
    action_vocab_size: int,
) -> None:
    if isinstance(refinement_passes, bool) or not isinstance(refinement_passes, int):
        raise TypeError("refinement_passes must be an integer")
    if refinement_passes < 1:
        raise ValueError(f"refinement_passes must be >= 1, got {refinement_passes}")
    if isinstance(trace_top_k, bool) or not isinstance(trace_top_k, int):
        raise TypeError("trace_top_k must be an integer")
    if not 1 <= trace_top_k <= action_vocab_size:
        raise ValueError(f"trace_top_k must be in [1, {action_vocab_size}], got {trace_top_k}")


def validate_root_legal_mask(
    root_legal_mask: Any,
    *,
    batch_size: int,
    action_vocab_size: int,
    action_codec_id: str,
) -> jax.Array:
    """Validate a root mask and return it as a boolean JAX array.

    This is intentionally a host-side, fail-closed boundary.  It is called once
    per inference batch before entering the compiled kernel.  No all-legal
    fallback, codec remap, or replacement action is ever synthesized.
    """

    if action_codec_id != ACTION_CODEC_LEGACY_ABSOLUTE_1858:
        raise ValueError(
            "recovered DFM inference requires action_codec_id="
            f"{ACTION_CODEC_LEGACY_ABSOLUTE_1858!r}, got {action_codec_id!r}"
        )

    mask = np.asarray(jax.device_get(root_legal_mask))
    expected_shape = (batch_size, action_vocab_size)
    if mask.shape != expected_shape:
        raise ValueError(f"root_legal_mask must have shape {expected_shape}, got {mask.shape}")
    if mask.dtype != np.dtype(np.bool_):
        raise TypeError(
            "root_legal_mask must have boolean dtype; "
            f"got {mask.dtype} (numeric masks are not coerced)"
        )
    empty_rows = np.flatnonzero(~np.any(mask, axis=1))
    if empty_rows.size:
        raise ValueError(
            f"root_legal_mask has no legal actions for batch rows {empty_rows.tolist()}"
        )
    return jnp.asarray(mask, dtype=jnp.bool_)


def _selection_distributions(
    logits: jax.Array,
    root_legal_mask: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Return selection log-probs plus root diagnostics in FP32."""

    logits_fp32 = jnp.asarray(logits, dtype=jnp.float32)
    root_logits = logits_fp32[:, 0, :]
    root_raw_log_probs = jax.nn.log_softmax(root_logits, axis=-1)
    root_raw_probs = jnp.exp(root_raw_log_probs)
    root_raw_entropy = -jnp.sum(
        root_raw_probs * root_raw_log_probs,
        axis=-1,
    )
    root_raw_legal_mass = jnp.sum(
        root_raw_probs * root_legal_mask,
        axis=-1,
    )

    root_legal_logits = jnp.where(root_legal_mask, root_logits, -jnp.inf)
    root_legal_log_probs = jax.nn.log_softmax(root_legal_logits, axis=-1)
    root_legal_probs = jnp.exp(root_legal_log_probs)
    root_legal_entropy = -jnp.sum(
        jnp.where(
            root_legal_mask,
            root_legal_probs * root_legal_log_probs,
            0.0,
        ),
        axis=-1,
    )

    if logits_fp32.shape[1] == 1:
        selection_log_probs = root_legal_log_probs[:, None, :]
    else:
        later_log_probs = jax.nn.log_softmax(
            logits_fp32[:, 1:, :],
            axis=-1,
        )
        selection_log_probs = jnp.concatenate(
            (root_legal_log_probs[:, None, :], later_log_probs),
            axis=1,
        )
    return (
        selection_log_probs,
        root_raw_entropy,
        root_raw_legal_mass,
        root_legal_entropy,
        root_legal_log_probs,
    )


def _proposal_feedback_latents(
    model: LocalDFMInferenceModel,
    z_dfm: jax.Array,
    z_jepa: jax.Array,
    root_action_before_update: jax.Array,
    root_action_after_update: jax.Array,
    root_prediction: jax.Array,
    root_legal_log_probs: jax.Array,
    root_action_hidden: jax.Array,
) -> jax.Array:
    """Return DFM latents conditioned on pass-7's legal root proposal."""

    mask_token = int(model.config.action_vocab_size)
    proposal = jnp.where(
        root_action_after_update == mask_token,
        root_prediction,
        root_action_after_update,
    )
    proposal_confidence = jnp.exp(
        jnp.max(root_legal_log_probs, axis=-1)
    )
    feedback_gate = jnp.where(
        root_action_before_update == mask_token,
        proposal_confidence,
        1.0,
    )
    proposal_z = model.jepa_step_from_latents(
        z_jepa,
        jax.lax.stop_gradient(proposal),
        root_action_hidden,
        z0_normalized=True,
    )
    feedback = model.dfm_latents_with_jepa_feedback(
        z_dfm,
        z_jepa,
        proposal_z,
        jax.lax.stop_gradient(feedback_gate),
    )
    return feedback.latents


def _refine_dfm_from_latents_impl(
    model: LocalDFMInferenceModel,
    z_dfm: jax.Array,
    root_legal_mask: jax.Array,
    *,
    refinement_passes: int,
    trace_top_k: int,
    z_jepa: jax.Array | None = None,
) -> DFMInferenceResult:
    """Compiled-kernel implementation; inputs must already be validated."""

    batch_size = z_dfm.shape[0]
    horizon = int(model.config.horizon)
    mask_token = int(model.config.action_vocab_size)
    feedback_active = _feedback_mode(model) == "final_pass_adjoint"
    if feedback_active:
        if refinement_passes != 8:
            raise ValueError(
                "jepa_feedback_mode='final_pass_adjoint' requires exactly "
                f"8 refinement passes, got {refinement_passes}"
            )
        if z_jepa is None:
            raise ValueError(
                "Closed-loop refinement requires current JEPA latents."
            )
    action_tokens = jnp.full(
        (batch_size, horizon),
        mask_token,
        dtype=jnp.int32,
    )
    planner_z_dfm = z_dfm

    times: list[jax.Array] = []
    actions_before: list[jax.Array] = []
    actions_after: list[jax.Array] = []
    raw_entropies: list[jax.Array] = []
    raw_legal_masses: list[jax.Array] = []
    legal_entropies: list[jax.Array] = []
    legal_topk_indices: list[jax.Array] = []
    legal_topk_probabilities: list[jax.Array] = []
    legal_topk_valid: list[jax.Array] = []

    for pass_index in range(refinement_passes):
        t_scalar = jnp.asarray(
            pass_index / refinement_passes,
            dtype=jnp.float32,
        )
        t = jnp.full((batch_size,), t_scalar, dtype=jnp.float32)
        feedback_source_pass = (
            feedback_active and pass_index == refinement_passes - 2
        )
        if feedback_source_pass:
            logits, hidden = model.planner_from_latents(
                planner_z_dfm,
                action_tokens,
                t,
                return_hidden=True,
            )
        else:
            logits = model.planner_from_latents(
                planner_z_dfm,
                action_tokens,
                t,
            )
            hidden = None
        (
            selection_log_probs,
            root_raw_entropy,
            root_raw_legal_mass,
            root_legal_entropy,
            root_legal_log_probs,
        ) = _selection_distributions(logits, root_legal_mask)

        topk_log_probs, topk_indices = jax.lax.top_k(
            root_legal_log_probs,
            trace_top_k,
        )
        topk_valid = jnp.take_along_axis(
            root_legal_mask,
            topk_indices,
            axis=-1,
        )
        topk_probabilities = jnp.where(
            topk_valid,
            jnp.exp(topk_log_probs),
            0.0,
        )

        selection_probs = jnp.exp(selection_log_probs)
        predictions = jnp.argmax(selection_log_probs, axis=-1).astype(jnp.int32)
        confidence = jnp.max(selection_probs, axis=-1)
        confidence = jnp.where(action_tokens == mask_token, confidence, jnp.inf)

        # Preserve the DFM's iterative-update behavior while choosing exactly
        # the scheduled number of positions. Stable positional ranks
        # intentionally remove the legacy threshold sampler's over-unmask
        # ambiguity when confidences tie, and also support more passes than the
        # action horizon.
        target_unmasked = (horizon * (pass_index + 1)) // refinement_passes
        position_order = jnp.argsort(
            -confidence,
            axis=-1,
            stable=True,
        )
        position_ranks = jnp.argsort(
            position_order,
            axis=-1,
            stable=True,
        )
        update_position = position_ranks < target_unmasked

        before = action_tokens
        action_tokens = jnp.where(
            update_position,
            predictions,
            action_tokens,
        )
        if feedback_source_pass:
            assert hidden is not None
            assert z_jepa is not None
            planner_z_dfm = _proposal_feedback_latents(
                model,
                z_dfm,
                z_jepa,
                before[:, 0],
                action_tokens[:, 0],
                predictions[:, 0],
                root_legal_log_probs,
                hidden["action_tokens"][:, 0, :],
            )

        times.append(t_scalar)
        actions_before.append(before)
        actions_after.append(action_tokens)
        raw_entropies.append(root_raw_entropy)
        raw_legal_masses.append(root_raw_legal_mass)
        legal_entropies.append(root_legal_entropy)
        legal_topk_indices.append(topk_indices)
        legal_topk_probabilities.append(topk_probabilities)
        legal_topk_valid.append(topk_valid)

    trace = DFMRefinementTrace(
        times=jnp.stack(times),
        actions_before=jnp.stack(actions_before),
        actions_after=jnp.stack(actions_after),
        root_raw_entropy=jnp.stack(raw_entropies),
        root_raw_legal_mass=jnp.stack(raw_legal_masses),
        root_legal_entropy=jnp.stack(legal_entropies),
        root_legal_topk_indices=jnp.stack(legal_topk_indices),
        root_legal_topk_probabilities=jnp.stack(legal_topk_probabilities),
        root_legal_topk_valid=jnp.stack(legal_topk_valid),
    )
    return DFMInferenceResult(actions=action_tokens, trace=trace)


def _refine_dfm_actions_from_latents_impl(
    model: LocalDFMInferenceModel,
    z_dfm: jax.Array,
    root_legal_mask: jax.Array,
    *,
    refinement_passes: int,
    z_jepa: jax.Array | None = None,
) -> jax.Array:
    """Return only final actions while preserving the checked refinement rule.

    Arena gameplay consumes only the first final action.  Keeping this kernel
    separate from the diagnostic path avoids constructing and transferring
    entropy, top-k, and per-pass trace arrays for every played position.
    """

    batch_size = z_dfm.shape[0]
    horizon = int(model.config.horizon)
    mask_token = int(model.config.action_vocab_size)
    feedback_active = _feedback_mode(model) == "final_pass_adjoint"
    if feedback_active:
        if refinement_passes != 8:
            raise ValueError(
                "jepa_feedback_mode='final_pass_adjoint' requires exactly "
                f"8 refinement passes, got {refinement_passes}"
            )
        if z_jepa is None:
            raise ValueError(
                "Closed-loop refinement requires current JEPA latents."
            )
    action_tokens = jnp.full(
        (batch_size, horizon),
        mask_token,
        dtype=jnp.int32,
    )
    planner_z_dfm = z_dfm

    for pass_index in range(refinement_passes):
        t = jnp.full(
            (batch_size,),
            jnp.asarray(pass_index / refinement_passes, dtype=jnp.float32),
            dtype=jnp.float32,
        )
        feedback_source_pass = (
            feedback_active and pass_index == refinement_passes - 2
        )
        if feedback_source_pass:
            logits, hidden = model.planner_from_latents(
                planner_z_dfm,
                action_tokens,
                t,
                return_hidden=True,
            )
            logits = jnp.asarray(logits, dtype=jnp.float32)
        else:
            logits = jnp.asarray(
                model.planner_from_latents(
                    planner_z_dfm,
                    action_tokens,
                    t,
                ),
                dtype=jnp.float32,
            )
            hidden = None
        root_logits = jnp.where(root_legal_mask, logits[:, 0, :], -jnp.inf)
        if horizon == 1:
            selection_logits = root_logits[:, None, :]
        else:
            selection_logits = jnp.concatenate(
                (root_logits[:, None, :], logits[:, 1:, :]),
                axis=1,
            )
        selection_log_probs = jax.nn.log_softmax(
            selection_logits,
            axis=-1,
        )
        predictions = jnp.argmax(selection_log_probs, axis=-1).astype(jnp.int32)
        confidence = jnp.exp(jnp.max(selection_log_probs, axis=-1))
        confidence = jnp.where(
            action_tokens == mask_token,
            confidence,
            jnp.inf,
        )

        target_unmasked = (horizon * (pass_index + 1)) // refinement_passes
        position_order = jnp.argsort(
            -confidence,
            axis=-1,
            stable=True,
        )
        position_ranks = jnp.argsort(
            position_order,
            axis=-1,
            stable=True,
        )
        before = action_tokens
        action_tokens = jnp.where(
            position_ranks < target_unmasked,
            predictions,
            action_tokens,
        )
        if feedback_source_pass:
            assert hidden is not None
            assert z_jepa is not None
            planner_z_dfm = _proposal_feedback_latents(
                model,
                z_dfm,
                z_jepa,
                before[:, 0],
                action_tokens[:, 0],
                predictions[:, 0],
                jax.nn.log_softmax(root_logits, axis=-1),
                hidden["action_tokens"][:, 0, :],
            )
    return action_tokens


def _infer_dfm_from_current_impl(
    model: LocalDFMInferenceModel,
    current_planes: jax.Array,
    root_legal_mask: jax.Array,
    *,
    refinement_passes: int,
    trace_top_k: int,
) -> DFMInferenceResult:
    """Encode BT4 exactly once, then reuse its DFM latents for every pass."""

    bt4_tokens = model.encode_bt4_tokens(current_planes)
    z_dfm = model.dfm_latents(bt4_tokens)
    z_jepa = (
        model.jepa_latents(bt4_tokens)
        if _feedback_mode(model) == "final_pass_adjoint"
        else None
    )
    return _refine_dfm_from_latents_impl(
        model,
        z_dfm,
        root_legal_mask,
        refinement_passes=refinement_passes,
        trace_top_k=trace_top_k,
        z_jepa=z_jepa,
    )


def _infer_dfm_actions_from_current_impl(
    model: LocalDFMInferenceModel,
    current_planes: jax.Array,
    root_legal_mask: jax.Array,
    *,
    refinement_passes: int,
) -> jax.Array:
    bt4_tokens = model.encode_bt4_tokens(current_planes)
    z_dfm = model.dfm_latents(bt4_tokens)
    z_jepa = (
        model.jepa_latents(bt4_tokens)
        if _feedback_mode(model) == "final_pass_adjoint"
        else None
    )
    return _refine_dfm_actions_from_latents_impl(
        model,
        z_dfm,
        root_legal_mask,
        refinement_passes=refinement_passes,
        z_jepa=z_jepa,
    )


_compiled_infer_dfm_from_current = nnx.jit(
    _infer_dfm_from_current_impl,
    static_argnames=("refinement_passes", "trace_top_k"),
)

_compiled_infer_dfm_actions_from_current = nnx.jit(
    _infer_dfm_actions_from_current_impl,
    static_argnames=("refinement_passes",),
)

_compiled_refine_dfm_from_latents = nnx.jit(
    _refine_dfm_from_latents_impl,
    static_argnames=("refinement_passes", "trace_top_k"),
)


def _validated_current_inputs(
    model: LocalDFMInferenceModel,
    current_planes: Any,
    root_legal_mask: Any,
    *,
    refinement_passes: int,
    trace_top_k: int,
    action_codec_id: str,
) -> tuple[jax.Array, jax.Array]:
    _, action_vocab_size = _model_dimensions(model)
    _validate_refinement_options(
        refinement_passes=refinement_passes,
        trace_top_k=trace_top_k,
        action_vocab_size=action_vocab_size,
    )
    _validate_feedback_refinement_count(
        model,
        refinement_passes=refinement_passes,
    )
    planes = jnp.asarray(current_planes)
    if planes.ndim != 4:
        raise ValueError(
            f"current_planes must have shape [batch, channels, height, width], got {planes.shape}"
        )
    if planes.shape[0] < 1:
        raise ValueError("current_planes batch must not be empty")
    legal_mask = validate_root_legal_mask(
        root_legal_mask,
        batch_size=planes.shape[0],
        action_vocab_size=action_vocab_size,
        action_codec_id=action_codec_id,
    )
    return planes, legal_mask


def infer_dfm_from_current(
    model: LocalDFMInferenceModel,
    current_planes: Any,
    root_legal_mask: Any,
    *,
    refinement_passes: int,
    trace_top_k: int,
    action_codec_id: str,
) -> DFMInferenceResult:
    """Run checked, compiled, batched DFM inference from board planes.

    ``refinement_passes`` and ``trace_top_k`` are compilation-static.  Repeated
    calls with the same model shapes and values reuse the NNX/JAX compilation
    cache.  The BT4 encoder runs once per call, outside the refinement loop.
    """

    planes, legal_mask = _validated_current_inputs(
        model,
        current_planes,
        root_legal_mask,
        refinement_passes=refinement_passes,
        trace_top_k=trace_top_k,
        action_codec_id=action_codec_id,
    )
    return _compiled_infer_dfm_from_current(
        model,
        planes,
        legal_mask,
        refinement_passes=refinement_passes,
        trace_top_k=trace_top_k,
    )


def infer_dfm_actions_from_current(
    model: LocalDFMInferenceModel,
    current_planes: Any,
    root_legal_mask: Any,
    *,
    refinement_passes: int,
    action_codec_id: str,
) -> jax.Array:
    """Run the checked action-only kernel used by lean arena policies."""

    planes, legal_mask = _validated_current_inputs(
        model,
        current_planes,
        root_legal_mask,
        refinement_passes=refinement_passes,
        trace_top_k=1,
        action_codec_id=action_codec_id,
    )
    return _compiled_infer_dfm_actions_from_current(
        model,
        planes,
        legal_mask,
        refinement_passes=refinement_passes,
    )


def refine_dfm_from_latents(
    model: LocalDFMInferenceModel,
    z_dfm: Any,
    root_legal_mask: Any,
    *,
    refinement_passes: int,
    trace_top_k: int,
    action_codec_id: str,
) -> DFMInferenceResult:
    """Run checked refinement when the caller already owns cached DFM latents."""

    _, action_vocab_size = _model_dimensions(model)
    _validate_refinement_options(
        refinement_passes=refinement_passes,
        trace_top_k=trace_top_k,
        action_vocab_size=action_vocab_size,
    )
    _validate_feedback_refinement_count(
        model,
        refinement_passes=refinement_passes,
    )
    if _feedback_mode(model) == "final_pass_adjoint":
        raise ValueError(
            "refine_dfm_from_latents cannot run "
            "jepa_feedback_mode='final_pass_adjoint' because current JEPA "
            "latents are not part of this API; use current-board inference."
        )
    latents = jnp.asarray(z_dfm)
    if latents.ndim != 3:
        raise ValueError(f"z_dfm must have shape [batch, tokens, width], got {latents.shape}")
    if latents.shape[0] < 1:
        raise ValueError("z_dfm batch must not be empty")
    legal_mask = validate_root_legal_mask(
        root_legal_mask,
        batch_size=latents.shape[0],
        action_vocab_size=action_vocab_size,
        action_codec_id=action_codec_id,
    )
    return _compiled_refine_dfm_from_latents(
        model,
        latents,
        legal_mask,
        refinement_passes=refinement_passes,
        trace_top_k=trace_top_k,
    )


def root_topk_turnover(trace: DFMRefinementTrace) -> jax.Array:
    """Return per-transition, per-sample legal top-k turnover in ``[0, 1]``."""

    indices = trace.root_legal_topk_indices
    valid = trace.root_legal_topk_valid
    if indices.shape[0] < 2:
        return jnp.empty((0, indices.shape[1]), dtype=jnp.float32)
    previous_indices = indices[:-1]
    current_indices = indices[1:]
    previous_valid = valid[:-1]
    current_valid = valid[1:]
    matches = (previous_indices[..., :, None] == current_indices[..., None, :]) & current_valid[
        ..., None, :
    ]
    overlap = jnp.sum(
        previous_valid & jnp.any(matches, axis=-1),
        axis=-1,
        dtype=jnp.float32,
    )
    denominator = jnp.sum(previous_valid, axis=-1, dtype=jnp.float32)
    return 1.0 - overlap / jnp.maximum(denominator, 1.0)


def benchmark_dfm_inference(
    model: LocalDFMInferenceModel,
    current_planes: Any,
    root_legal_mask: Any,
    *,
    refinement_passes: int,
    trace_top_k: int,
    action_codec_id: str,
    warmup_calls: int = 2,
    measured_calls: int = 20,
) -> DFMInferenceBenchmark:
    """Benchmark one fixed, already-materialized inference batch.

    The first-call field includes compilation and execution because those costs
    cannot be separated robustly across persistent-cache states.  Validation is
    performed once before timing.  Subsequent calls invoke the same checked
    compiled kernel directly.
    """

    if isinstance(warmup_calls, bool) or not isinstance(warmup_calls, int):
        raise TypeError("warmup_calls must be an integer")
    if warmup_calls < 1:
        raise ValueError("warmup_calls must be >= 1")
    if isinstance(measured_calls, bool) or not isinstance(measured_calls, int):
        raise TypeError("measured_calls must be an integer")
    if measured_calls < 1:
        raise ValueError("measured_calls must be >= 1")

    planes, legal_mask = _validated_current_inputs(
        model,
        current_planes,
        root_legal_mask,
        refinement_passes=refinement_passes,
        trace_top_k=trace_top_k,
        action_codec_id=action_codec_id,
    )

    def run_once() -> DFMInferenceResult:
        result = _compiled_infer_dfm_from_current(
            model,
            planes,
            legal_mask,
            refinement_passes=refinement_passes,
            trace_top_k=trace_top_k,
        )
        jax.block_until_ready(result)
        return result

    start = time.perf_counter()
    run_once()
    first_call_seconds = time.perf_counter() - start
    for _ in range(warmup_calls - 1):
        run_once()

    latencies = np.empty((measured_calls,), dtype=np.float64)
    for index in range(measured_calls):
        start = time.perf_counter()
        run_once()
        latencies[index] = time.perf_counter() - start

    p50 = float(np.percentile(latencies, 50))
    p95 = float(np.percentile(latencies, 95))
    return DFMInferenceBenchmark(
        batch_size=int(planes.shape[0]),
        refinement_passes=refinement_passes,
        trace_top_k=trace_top_k,
        compile_and_first_call_seconds=first_call_seconds,
        steady_latency_p50_seconds=p50,
        steady_latency_p95_seconds=p95,
        steady_examples_per_second=float(planes.shape[0]) / p50,
        warmup_calls=warmup_calls,
        measured_calls=measured_calls,
    )


__all__ = [
    "ACTION_CODEC_LEGACY_ABSOLUTE_1858",
    "DFMInferenceBenchmark",
    "DFMInferenceResult",
    "DFMRefinementTrace",
    "LocalDFMInferenceModel",
    "benchmark_dfm_inference",
    "infer_dfm_actions_from_current",
    "infer_dfm_from_current",
    "refine_dfm_from_latents",
    "root_topk_turnover",
    "validate_root_legal_mask",
]
