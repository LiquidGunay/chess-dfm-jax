"""Action-specific attribution methods and chess-aware perturbation utilities."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

import chess
import numpy as np
import torch
from torch import Tensor

from chess_dfm_jax.encoding import encode_board
from research.interpretability.patching import legal_log_probabilities, target_log_odds


ObjectiveKind = Literal["target_logit", "target_log_probability", "target_log_odds"]


@dataclass(frozen=True)
class GradientAttribution:
    method: str
    attribution: Tensor
    square_signed: Tensor
    square_magnitude: Tensor
    objective: Tensor


@dataclass(frozen=True)
class IntegratedGradientsResult:
    method: str
    attribution: Tensor
    per_baseline_attribution: Tensor
    square_signed: Tensor
    square_magnitude: Tensor
    input_objective: Tensor
    baseline_objective: Tensor
    completeness_residual: Tensor
    relative_completeness_error: Tensor
    steps: int


@dataclass(frozen=True)
class PieceRemovalVariants:
    planes: np.ndarray
    eligible: np.ndarray
    target_legal: np.ndarray
    square_names: tuple[str, ...]
    contract: str


def _validate_input(inputs: Tensor) -> None:
    if inputs.ndim != 4:
        raise ValueError(f"inputs must have shape [batch, channel, 8, 8], got {inputs.shape}")
    if tuple(inputs.shape[-2:]) != (8, 8):
        raise ValueError("Attribution inputs must use an 8 by 8 board")
    if not torch.is_floating_point(inputs):
        raise TypeError("Attribution inputs must be floating point")
    if not bool(torch.isfinite(inputs).all()):
        raise ValueError("Attribution inputs contain nonfinite values")


def action_objective(
    logits: Tensor,
    targets: Tensor,
    *,
    kind: ObjectiveKind,
    legal_indices: Tensor | None = None,
    legal_counts: Tensor | None = None,
) -> Tensor:
    """Return one differentiable, action-specific scalar per position."""

    if logits.ndim != 2 or not torch.is_floating_point(logits):
        raise ValueError("logits must be a floating [batch, action] tensor")
    if targets.shape != (logits.shape[0],) or targets.device != logits.device:
        raise ValueError("targets must have shape [batch] on the logits device")
    target_indices = targets.to(dtype=torch.long)
    if bool(((target_indices < 0) | (target_indices >= logits.shape[1])).any()):
        raise ValueError("A target lies outside the logit vocabulary")
    if kind == "target_logit":
        return logits.gather(1, target_indices.unsqueeze(1)).squeeze(1)
    if legal_indices is None or legal_counts is None:
        raise ValueError(f"{kind} requires legal_indices and legal_counts")
    if kind == "target_log_odds":
        return target_log_odds(logits, legal_indices, legal_counts, targets)
    if kind == "target_log_probability":
        log_probability, mask = legal_log_probabilities(
            logits,
            legal_indices,
            legal_counts,
            targets=targets,
        )
        matches = mask & (legal_indices.to(dtype=torch.long) == target_indices.unsqueeze(1))
        return torch.where(matches, log_probability, 0.0).sum(dim=1)
    raise ValueError(f"Unknown action objective {kind!r}")


def _square_views(attribution: Tensor) -> tuple[Tensor, Tensor]:
    return attribution.sum(dim=1), attribution.abs().sum(dim=1)


def input_gradient_attribution(
    forward: Callable[[Tensor], Tensor],
    inputs: Tensor,
    targets: Tensor,
    *,
    kind: ObjectiveKind = "target_log_odds",
    legal_indices: Tensor | None = None,
    legal_counts: Tensor | None = None,
    multiply_by_input: bool = False,
    reference: Tensor | None = None,
) -> GradientAttribution:
    """Compute input gradients or gradient times displacement from a reference."""

    _validate_input(inputs)
    value = inputs.detach().clone().requires_grad_(True)
    logits = forward(value)
    objective = action_objective(
        logits,
        targets,
        kind=kind,
        legal_indices=legal_indices,
        legal_counts=legal_counts,
    )
    gradient = torch.autograd.grad(objective.sum(), value, create_graph=False)[0]
    if multiply_by_input:
        baseline = torch.zeros_like(value) if reference is None else reference
        if baseline.shape != value.shape or baseline.device != value.device:
            raise ValueError("reference must match inputs")
        attribution = gradient * (value - baseline)
        method = "gradient_x_displacement"
    else:
        attribution = gradient
        method = "input_gradient"
    signed, magnitude = _square_views(attribution)
    return GradientAttribution(
        method=method,
        attribution=attribution.detach(),
        square_signed=signed.detach(),
        square_magnitude=magnitude.detach(),
        objective=objective.detach(),
    )


def integrated_gradients(
    forward: Callable[[Tensor], Tensor],
    inputs: Tensor,
    targets: Tensor,
    baselines: Tensor,
    *,
    kind: ObjectiveKind = "target_log_odds",
    legal_indices: Tensor | None = None,
    legal_counts: Tensor | None = None,
    steps: int = 64,
    quadrature: Literal["trapezoidal", "gauss_legendre"] = "trapezoidal",
) -> IntegratedGradientsResult:
    """Integrated gradients with explicit quadrature and completeness audit.

    baselines has shape [baseline, batch, channel, 8, 8]. Each path is
    integrated independently; the returned attribution is their arithmetic
    mean and completeness is retained per baseline and position. Gauss-Legendre
    uses ``steps`` interior nodes and is useful when uniform trapezoids miss
    narrow gradient structure along an off-manifold chess-board path.
    """

    _validate_input(inputs)
    if baselines.ndim != 5 or tuple(baselines.shape[1:]) != tuple(inputs.shape):
        raise ValueError("baselines must have shape [baseline, *inputs.shape]")
    if baselines.device != inputs.device or baselines.dtype != inputs.dtype:
        raise ValueError("baselines must share input dtype and device")
    if not bool(torch.isfinite(baselines).all()):
        raise ValueError("baselines contain nonfinite values")
    if type(steps) is not int or steps < 2:
        raise ValueError("steps must be an integer of at least two")
    if quadrature not in {"trapezoidal", "gauss_legendre"}:
        raise ValueError("quadrature must be 'trapezoidal' or 'gauss_legendre'")

    with torch.no_grad():
        input_objective = action_objective(
            forward(inputs),
            targets,
            kind=kind,
            legal_indices=legal_indices,
            legal_counts=legal_counts,
        )
    baseline_attributions: list[Tensor] = []
    baseline_objectives: list[Tensor] = []
    residuals: list[Tensor] = []
    relative_errors: list[Tensor] = []
    if quadrature == "trapezoidal":
        alphas = torch.linspace(0.0, 1.0, steps + 1, dtype=torch.float64)
        weights = torch.full((steps + 1,), 1.0 / steps, dtype=torch.float64)
        weights[[0, -1]] *= 0.5
    else:
        nodes, numpy_weights = np.polynomial.legendre.leggauss(steps)
        alphas = torch.from_numpy((nodes + 1.0) * 0.5)
        weights = torch.from_numpy(numpy_weights * 0.5)
    for baseline in baselines:
        with torch.no_grad():
            baseline_objective = action_objective(
                forward(baseline),
                targets,
                kind=kind,
                legal_indices=legal_indices,
                legal_counts=legal_counts,
            )
        gradient_sum = torch.zeros_like(inputs)
        for alpha, weight in zip(alphas, weights, strict=True):
            alpha = alpha.to(device=inputs.device, dtype=inputs.dtype)
            interpolated = (baseline + alpha * (inputs - baseline)).detach().requires_grad_(True)
            objective = action_objective(
                forward(interpolated),
                targets,
                kind=kind,
                legal_indices=legal_indices,
                legal_counts=legal_counts,
            )
            gradient = torch.autograd.grad(objective.sum(), interpolated)[0]
            gradient_sum.add_(gradient.detach(), alpha=float(weight))
        attribution = (inputs - baseline) * gradient_sum
        explained = attribution.flatten(1).sum(dim=1)
        objective_delta = input_objective - baseline_objective
        residual = objective_delta - explained
        relative = residual.abs() / objective_delta.abs().clamp_min(1e-8)
        baseline_attributions.append(attribution.detach())
        baseline_objectives.append(baseline_objective.detach())
        residuals.append(residual.detach())
        relative_errors.append(relative.detach())

    stacked = torch.stack(baseline_attributions)
    mean_attribution = stacked.mean(dim=0)
    signed, magnitude = _square_views(mean_attribution)
    return IntegratedGradientsResult(
        method=f"integrated_gradients_{quadrature}",
        attribution=mean_attribution,
        per_baseline_attribution=stacked,
        square_signed=signed,
        square_magnitude=magnitude,
        input_objective=input_objective.detach(),
        baseline_objective=torch.stack(baseline_objectives),
        completeness_residual=torch.stack(residuals),
        relative_completeness_error=torch.stack(relative_errors),
        steps=steps,
    )


def smoothgrad(
    forward: Callable[[Tensor], Tensor],
    inputs: Tensor,
    targets: Tensor,
    *,
    kind: ObjectiveKind = "target_log_odds",
    legal_indices: Tensor | None = None,
    legal_counts: Tensor | None = None,
    samples: int = 32,
    noise_std: float = 0.1,
    seed: int = 20260808,
    multiply_by_input: bool = False,
) -> GradientAttribution:
    """Average gradients over seeded Gaussian input noise."""

    _validate_input(inputs)
    if type(samples) is not int or samples <= 0:
        raise ValueError("samples must be positive")
    if noise_std < 0.0 or not math.isfinite(noise_std):
        raise ValueError("noise_std must be finite and non-negative")
    generator = torch.Generator(device=inputs.device).manual_seed(seed)
    total = torch.zeros_like(inputs)
    objective_total = torch.zeros(inputs.shape[0], device=inputs.device, dtype=inputs.dtype)
    for _ in range(samples):
        noise = torch.randn(
            inputs.shape,
            dtype=inputs.dtype,
            device=inputs.device,
            generator=generator,
        )
        noisy = (inputs + noise_std * noise).detach().requires_grad_(True)
        objective = action_objective(
            forward(noisy),
            targets,
            kind=kind,
            legal_indices=legal_indices,
            legal_counts=legal_counts,
        )
        gradient = torch.autograd.grad(objective.sum(), noisy)[0]
        total.add_(gradient.detach() * noisy.detach() if multiply_by_input else gradient.detach())
        objective_total.add_(objective.detach())
    attribution = total / samples
    signed, magnitude = _square_views(attribution)
    return GradientAttribution(
        method="smoothgrad_x_input" if multiply_by_input else "smoothgrad",
        attribution=attribution,
        square_signed=signed,
        square_magnitude=magnitude,
        objective=objective_total / samples,
    )


def gradcam_square_map(
    activations: Tensor,
    gradients: Tensor,
    *,
    relu: bool = True,
) -> Tensor:
    """Grad-CAM over square tokens, retained only as a historical baseline."""

    if activations.shape != gradients.shape or activations.ndim != 3:
        raise ValueError("activations and gradients must match [batch, square, feature]")
    if activations.shape[1] != 64:
        raise ValueError("Grad-CAM expects 64 square tokens")
    channel_weight = gradients.mean(dim=1, keepdim=True)
    score = (activations * channel_weight).sum(dim=2)
    if relu:
        score = score.relu()
    return score.reshape(-1, 8, 8)


def sarfa_scores(
    original_logits: Tensor,
    perturbed_logits: Tensor,
    legal_indices: Tensor,
    legal_counts: Tensor,
    targets: Tensor,
    *,
    eligible: Tensor | None = None,
    target_legal: Tensor | None = None,
) -> dict[str, Tensor]:
    """Exact SARFA equation on a fixed original legal-action support.

    perturbed_logits is [batch, perturbation, action]. Negative specificity is
    clamped to zero, matching the authors' released implementation. If a
    supplied target_legal mask is false, saliency is one, matching their chess
    handling for perturbations that invalidate the explained move.
    """

    if original_logits.ndim != 2 or perturbed_logits.ndim != 3:
        raise ValueError("Expected original [B,A] and perturbed [B,P,A] logits")
    batch, action_count = original_logits.shape
    if perturbed_logits.shape[0] != batch or perturbed_logits.shape[2] != action_count:
        raise ValueError("Original and perturbed logits are not aligned")
    perturbation_count = int(perturbed_logits.shape[1])
    original_logp, mask = legal_log_probabilities(
        original_logits,
        legal_indices,
        legal_counts,
        targets=targets,
    )
    repeated_legal = (
        legal_indices[:, None, :]
        .expand(-1, perturbation_count, -1)
        .reshape(
            batch * perturbation_count,
            -1,
        )
    )
    repeated_counts = legal_counts[:, None].expand(-1, perturbation_count).reshape(-1)
    repeated_targets = targets[:, None].expand(-1, perturbation_count).reshape(-1)
    perturbed_logp, repeated_mask = legal_log_probabilities(
        perturbed_logits.reshape(batch * perturbation_count, action_count),
        repeated_legal,
        repeated_counts,
        targets=repeated_targets,
    )
    max_legal = legal_indices.shape[1]
    perturbed_logp = perturbed_logp.reshape(batch, perturbation_count, max_legal)
    repeated_mask = repeated_mask.reshape(batch, perturbation_count, max_legal)
    target_matches = mask & (
        legal_indices.to(dtype=torch.long) == targets.to(dtype=torch.long).unsqueeze(1)
    )
    original_target_probability = torch.where(
        target_matches,
        original_logp.exp(),
        0.0,
    ).sum(dim=1)
    perturbed_target_probability = torch.where(
        target_matches[:, None, :],
        perturbed_logp.exp(),
        0.0,
    ).sum(dim=2)
    specificity = original_target_probability[:, None] - perturbed_target_probability

    other_mask = mask & ~target_matches
    original_other_logits = original_logp.masked_fill(~other_mask, -torch.inf)
    original_other_logp = torch.log_softmax(original_other_logits, dim=1)
    perturbed_other_logits = perturbed_logp.masked_fill(
        ~other_mask[:, None, :],
        -torch.inf,
    )
    perturbed_other_logp = torch.log_softmax(perturbed_other_logits, dim=2)
    original_other_probability = original_other_logp.exp().masked_fill(~other_mask, 0.0)
    kl = torch.where(
        repeated_mask & other_mask[:, None, :],
        original_other_probability[:, None, :]
        * (original_other_logp[:, None, :] - perturbed_other_logp),
        0.0,
    ).sum(dim=2)
    relevance = 1.0 / (1.0 + kl.clamp_min(0.0))
    positive_specificity = specificity.clamp_min(0.0)
    saliency = torch.where(
        positive_specificity > 0.0,
        2.0
        * positive_specificity
        * relevance
        / (positive_specificity + relevance).clamp_min(torch.finfo(relevance.dtype).tiny),
        0.0,
    )
    if eligible is not None:
        if eligible.shape != saliency.shape or eligible.device != saliency.device:
            raise ValueError("eligible must match [batch, perturbation]")
        saliency = saliency.masked_fill(~eligible.bool(), 0.0)
    if target_legal is not None:
        if target_legal.shape != saliency.shape or target_legal.device != saliency.device:
            raise ValueError("target_legal must match [batch, perturbation]")
        forced = ~target_legal.bool()
        if eligible is not None:
            forced &= eligible.bool()
        saliency = torch.where(forced, torch.ones_like(saliency), saliency)
    return {
        "saliency": saliency,
        "specificity_delta_probability": specificity,
        "relevance": relevance,
        "other_action_kl_original_to_perturbed": kl,
        "original_target_probability": original_target_probability,
        "perturbed_target_probability": perturbed_target_probability,
    }


def legal_piece_removal_variants(
    board: chess.Board,
    history: Sequence[chess.Board],
    *,
    target_move: chess.Move | None = None,
    input_format: str = "INPUT_CLASSICAL_112_PLANE",
) -> PieceRemovalVariants:
    """Re-encode every legal non-king piece removal with full available history."""

    if not isinstance(board, chess.Board):
        raise TypeError("board must be a python-chess Board")
    historical = [value.copy(stack=False) for value in history]
    if not historical or historical[-1].fen() != board.fen():
        historical.append(board.copy(stack=False))
    base = encode_board(board, historical, input_format=input_format)
    variants = np.repeat(base[None, ...], 64, axis=0)
    eligible = np.zeros(64, dtype=np.bool_)
    target_legal = np.ones(64, dtype=np.bool_)
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is None or piece.piece_type == chess.KING:
            continue
        perturbed = board.copy(stack=False)
        perturbed.remove_piece_at(square)
        perturbed.castling_rights = perturbed.clean_castling_rights()
        if not perturbed.is_valid():
            continue
        perturbed_history = [*historical[:-1], perturbed]
        variants[square] = encode_board(
            perturbed,
            perturbed_history,
            input_format=input_format,
        )
        eligible[square] = True
        if target_move is not None:
            target_legal[square] = perturbed.is_legal(target_move)
    return PieceRemovalVariants(
        planes=variants,
        eligible=eligible,
        target_legal=target_legal,
        square_names=tuple(chess.square_name(square) for square in chess.SQUARES),
        contract=(
            "Non-king occupied squares only; full supplied history is preserved, "
            "current board and auxiliary planes are re-encoded, invalid boards are excluded"
        ),
    )


def pairwise_occlusion_interaction(
    original_score: Tensor,
    single_scores: Tensor,
    pair_scores: Tensor,
) -> Tensor:
    """Second-order inclusion-exclusion interaction for selected perturbations."""

    if original_score.ndim != 1 or single_scores.ndim != 2 or pair_scores.ndim != 3:
        raise ValueError("Expected original [B], single [B,S], pair [B,S,S]")
    if (
        single_scores.shape[0] != original_score.shape[0]
        or pair_scores.shape[:2] != single_scores.shape
        or pair_scores.shape[2] != single_scores.shape[1]
    ):
        raise ValueError("Occlusion score tensors are not aligned")
    return (
        original_score[:, None, None]
        - single_scores[:, :, None]
        - single_scores[:, None, :]
        + pair_scores
    )
