"""Future-move probes and causal diagnostics for BT4 square-token states.

The primary replication target is Jenner et al. (NeurIPS 2024): a rank-32
bilinear probe predicts a later move destination conditioned on the first move
destination, then predicts the later source conditioned on the predicted
destination.  This module preserves that exact readout while adding group-aware
training, all-seven-ply labels, stronger metrics, and causal probe gradients.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F


LEELA_INTERP_COMMIT = "da19a5905375570995dace848c94c794bef70347"
Conditioning = Literal["first_destination", "previous_destination", "true_destination"]


def _board_states(value: np.ndarray | Tensor) -> Tensor:
    states = torch.as_tensor(value).detach().to(dtype=torch.float32, device="cpu")
    if states.ndim != 3 or states.shape[1] != 64 or states.shape[2] <= 0:
        raise ValueError("Lookahead activations must have shape [position, 64, feature]")
    if not bool(torch.isfinite(states).all()):
        raise ValueError("Lookahead activations contain nonfinite values")
    return states


def _squares(value: Sequence[int] | np.ndarray | Tensor, count: int, name: str) -> Tensor:
    squares = torch.as_tensor(value, dtype=torch.long, device="cpu")
    if squares.shape != (count,):
        raise ValueError(f"{name} must have shape [position]")
    if bool(((squares < 0) | (squares >= 64)).any()):
        raise ValueError(f"{name} contains a square outside [0, 63]")
    return squares


class BilinearSquareProbe(nn.Module):
    """Low-rank conditional square classifier used in learned-lookahead work."""

    def __init__(
        self,
        feature_width: int,
        *,
        rank: int = 32,
        seed: int = 0,
        per_square_bias: bool = False,
    ):
        super().__init__()
        if type(feature_width) is not int or feature_width <= 0:
            raise ValueError("feature_width must be positive")
        if type(rank) is not int or rank <= 0:
            raise ValueError("rank must be positive")
        self.feature_width = feature_width
        self.rank = rank
        self.per_square_bias = bool(per_square_bias)
        generator = torch.Generator(device="cpu").manual_seed(seed)
        self.candidate_weight = nn.Parameter(
            torch.randn((rank, feature_width), generator=generator) / math.sqrt(feature_width)
        )
        self.condition_weight = nn.Parameter(
            torch.randn((rank, feature_width), generator=generator) / math.sqrt(feature_width)
        )
        bias_width = 64 if self.per_square_bias else 1
        self.bias = nn.Parameter(torch.randn((bias_width,), generator=generator))

    def forward(self, states: Tensor, condition_states: Tensor) -> Tensor:
        if states.ndim != 3 or states.shape[1:] != (64, self.feature_width):
            raise ValueError("Probe states have an invalid shape")
        if condition_states.shape != (states.shape[0], self.feature_width):
            raise ValueError("Probe condition states have an invalid shape")
        candidate = torch.einsum("kd,bsd->bsk", self.candidate_weight, states)
        condition = torch.einsum("kd,bd->bk", self.condition_weight, condition_states)
        logits = (candidate * condition[:, None, :]).sum(dim=-1) / math.sqrt(self.rank)
        return logits + self.bias

    def logits_from_squares(self, states: Tensor, condition_squares: Tensor) -> Tensor:
        if condition_squares.shape != (states.shape[0],):
            raise ValueError("condition_squares must align with states")
        rows = torch.arange(states.shape[0], device=states.device)
        return self(states, states[rows, condition_squares.to(states.device)])


@dataclass
class FittedBilinearSquareProbe:
    module: BilinearSquareProbe
    metadata: dict[str, Any]

    def logits(
        self,
        states: np.ndarray | Tensor,
        condition_squares: Sequence[int] | np.ndarray | Tensor,
        *,
        batch_size: int = 1024,
    ) -> Tensor:
        value = _board_states(states)
        condition = _squares(condition_squares, value.shape[0], "condition_squares")
        if type(batch_size) is not int or batch_size <= 0:
            raise ValueError("batch_size must be positive")
        chunks: list[Tensor] = []
        self.module.eval()
        with torch.inference_mode():
            for start in range(0, value.shape[0], batch_size):
                batch = value[start : start + batch_size]
                chunks.append(
                    self.module.logits_from_squares(
                        batch,
                        condition[start : start + batch_size],
                    ).cpu()
                )
        return torch.cat(chunks)

    def predict(
        self,
        states: np.ndarray | Tensor,
        condition_squares: Sequence[int] | np.ndarray | Tensor,
    ) -> Tensor:
        return self.logits(states, condition_squares).argmax(dim=1)


def _group_weights(group_ids: Sequence[object] | None, count: int) -> Tensor:
    if group_ids is None:
        return torch.ones(count, dtype=torch.float32)
    groups = np.asarray(group_ids).astype(str)
    if groups.shape != (count,):
        raise ValueError("group_ids must align with probe examples")
    _unique, inverse, frequencies = np.unique(groups, return_inverse=True, return_counts=True)
    weights = torch.from_numpy(1.0 / frequencies[inverse]).to(torch.float32)
    return weights * (count / weights.sum())


def fit_bilinear_square_probe(
    states: np.ndarray | Tensor,
    labels: Sequence[int] | np.ndarray | Tensor,
    condition_squares: Sequence[int] | np.ndarray | Tensor,
    *,
    rank: int = 32,
    epochs: int = 5,
    learning_rate: float = 1e-2,
    batch_size: int = 64,
    weight_decay: float = 0.0,
    seed: int = 0,
    group_ids: Sequence[object] | None = None,
    shuffle: bool = True,
    per_square_bias: bool = False,
    device: torch.device | str = "cpu",
) -> FittedBilinearSquareProbe:
    """Fit the published bilinear readout with optional group-balanced loss.

    ``rank=32``, five epochs, Adam at 1e-2, batch size 64, and a scalar bias
    reproduce the released probe architecture.  Shuffling and group weights
    are recorded extensions; set ``shuffle=False`` and ``group_ids=None`` for
    the closest released-training behavior.
    """

    value = _board_states(states)
    target = _squares(labels, value.shape[0], "labels")
    condition = _squares(condition_squares, value.shape[0], "condition_squares")
    if type(epochs) is not int or epochs <= 0:
        raise ValueError("epochs must be positive")
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if learning_rate <= 0.0 or not math.isfinite(learning_rate):
        raise ValueError("learning_rate must be finite and positive")
    if weight_decay < 0.0 or not math.isfinite(weight_decay):
        raise ValueError("weight_decay must be finite and non-negative")
    weights = _group_weights(group_ids, value.shape[0])
    active_device = torch.device(device)
    module = BilinearSquareProbe(
        value.shape[2],
        rank=rank,
        seed=seed,
        per_square_bias=per_square_bias,
    ).to(active_device)
    optimizer = torch.optim.Adam(
        module.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    generator = torch.Generator(device="cpu").manual_seed(seed)
    losses: list[float] = []
    for _epoch in range(epochs):
        order = (
            torch.randperm(value.shape[0], generator=generator)
            if shuffle
            else torch.arange(value.shape[0])
        )
        total_loss = 0.0
        total_weight = 0.0
        for start in range(0, value.shape[0], batch_size):
            rows = order[start : start + batch_size]
            batch_states = value[rows].to(active_device)
            batch_condition = condition[rows].to(active_device)
            batch_target = target[rows].to(active_device)
            batch_weight = weights[rows].to(active_device)
            logits = module.logits_from_squares(batch_states, batch_condition)
            element_loss = F.cross_entropy(logits, batch_target, reduction="none")
            loss = (element_loss * batch_weight).sum() / batch_weight.sum()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += float((element_loss.detach() * batch_weight).sum())
            total_weight += float(batch_weight.sum())
        losses.append(total_loss / total_weight)
    module = module.cpu().eval()
    return FittedBilinearSquareProbe(
        module=module,
        metadata={
            "schema_version": "bt4-bilinear-lookahead-probe-v1",
            "replication": {
                "paper": "Jenner et al., Evidence of Learned Look-Ahead (NeurIPS 2024)",
                "official_code_commit": LEELA_INTERP_COMMIT,
                "architecture_match": (rank == 32 and not per_square_bias and epochs == 5),
            },
            "feature_width": int(value.shape[2]),
            "rank": rank,
            "per_square_bias": per_square_bias,
            "training_examples": int(value.shape[0]),
            "epochs": epochs,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "weight_decay": weight_decay,
            "seed": seed,
            "shuffle": bool(shuffle),
            "group_balanced_loss": group_ids is not None,
            "initial_loss": losses[0],
            "final_loss": losses[-1],
        },
    )


def square_prediction_metrics(logits: Tensor, labels: Sequence[int] | Tensor) -> dict[str, float]:
    target = _squares(labels, logits.shape[0], "labels").to(logits.device)
    if logits.shape != (target.shape[0], 64) or not bool(torch.isfinite(logits).all()):
        raise ValueError("Square logits must be finite with shape [position, 64]")
    ranks = 1 + (logits > logits.gather(1, target[:, None])).sum(dim=1)
    log_probability = torch.log_softmax(logits, dim=1)
    probabilities = log_probability.exp()
    return {
        "accuracy": float((ranks == 1).float().mean()),
        "top3_accuracy": float((ranks <= 3).float().mean()),
        "top5_accuracy": float((ranks <= 5).float().mean()),
        "mean_reciprocal_rank": float((1.0 / ranks.float()).mean()),
        "nll": float(-log_probability.gather(1, target[:, None]).mean()),
        "entropy": float((-(probabilities * log_probability).sum(dim=1)).mean()),
        "count": int(target.shape[0]),
    }


def select_future_ply(
    states: np.ndarray | Tensor,
    future_origins_root: np.ndarray | Tensor,
    future_destinations_root: np.ndarray | Tensor,
    future_valid: np.ndarray | Tensor,
    *,
    ply: int,
    conditioning: Conditioning = "first_destination",
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Select valid examples for one 0-indexed future ply in the root frame."""

    value = _board_states(states)
    origins = torch.as_tensor(future_origins_root, dtype=torch.long)
    destinations = torch.as_tensor(future_destinations_root, dtype=torch.long)
    valid = torch.as_tensor(future_valid, dtype=torch.bool)
    if origins.shape != destinations.shape or valid.shape != origins.shape:
        raise ValueError("Future move arrays must be aligned")
    if origins.ndim != 2 or origins.shape[0] != value.shape[0]:
        raise ValueError("Future move arrays must have shape [position, ply]")
    if type(ply) is not int or not 0 <= ply < origins.shape[1]:
        raise ValueError("ply is outside the available future line")
    mask = valid[:, ply]
    if conditioning == "first_destination":
        mask &= valid[:, 0]
        condition = destinations[:, 0]
    elif conditioning == "previous_destination":
        if ply == 0:
            raise ValueError("ply zero has no previous destination")
        mask &= valid[:, ply - 1]
        condition = destinations[:, ply - 1]
    elif conditioning == "true_destination":
        condition = destinations[:, ply]
    else:
        raise ValueError(f"Unknown conditioning scheme {conditioning!r}")
    if not bool(mask.any()):
        raise ValueError(f"No valid examples for future ply {ply}")
    selected_origins = origins[mask, ply]
    selected_destinations = destinations[mask, ply]
    selected_condition = condition[mask]
    for name, selected in (
        ("origins", selected_origins),
        ("destinations", selected_destinations),
        ("conditioning", selected_condition),
    ):
        if bool(((selected < 0) | (selected >= 64)).any()):
            raise ValueError(f"Valid future {name} contain an invalid root-frame square")
    return value[mask], selected_origins, selected_destinations, selected_condition


def evaluate_future_move_pair(
    destination_probe: FittedBilinearSquareProbe,
    source_probe: FittedBilinearSquareProbe,
    states: np.ndarray | Tensor,
    source_labels: Sequence[int] | np.ndarray | Tensor,
    destination_labels: Sequence[int] | np.ndarray | Tensor,
    first_destination_squares: Sequence[int] | np.ndarray | Tensor,
) -> dict[str, Any]:
    """Evaluate the published two-stage destination-then-source decoder."""

    value = _board_states(states)
    source = _squares(source_labels, value.shape[0], "source_labels")
    destination = _squares(destination_labels, value.shape[0], "destination_labels")
    anchor = _squares(first_destination_squares, value.shape[0], "first_destination_squares")
    destination_logits = destination_probe.logits(value, anchor)
    predicted_destination = destination_logits.argmax(dim=1)
    source_logits_sequential = source_probe.logits(value, predicted_destination)
    source_logits_oracle = source_probe.logits(value, destination)
    predicted_source = source_logits_sequential.argmax(dim=1)
    return {
        "destination": square_prediction_metrics(destination_logits, destination),
        "source_given_predicted_destination": square_prediction_metrics(
            source_logits_sequential,
            source,
        ),
        "source_given_true_destination": square_prediction_metrics(source_logits_oracle, source),
        "joint_move_accuracy": float(
            ((predicted_destination == destination) & (predicted_source == source)).float().mean()
        ),
        "destination_logits": destination_logits,
        "source_logits_sequential": source_logits_sequential,
        "predicted_destination": predicted_destination,
        "predicted_source": predicted_source,
    }


def normalized_probe_gradient(
    probe: FittedBilinearSquareProbe,
    states: np.ndarray | Tensor,
    condition_squares: Sequence[int] | np.ndarray | Tensor,
    target_squares: Sequence[int] | np.ndarray | Tensor,
) -> tuple[Tensor, Tensor]:
    """Gradient of target log probability, normalized per position for steering."""

    value = _board_states(states).requires_grad_(True)
    condition = _squares(condition_squares, value.shape[0], "condition_squares")
    target = _squares(target_squares, value.shape[0], "target_squares")
    probe.module.eval()
    logits = probe.module.logits_from_squares(value, condition)
    objective = torch.log_softmax(logits, dim=1).gather(1, target[:, None]).squeeze(1)
    gradient = torch.autograd.grad(objective.sum(), value)[0]
    norm = gradient.flatten(1).norm(dim=1, keepdim=True).clamp_min(1e-12)
    normalized = gradient / norm[:, None]
    return normalized.detach(), objective.detach()


def steer_with_probe_gradient(
    states: Tensor, normalized_gradient: Tensor, *, dose: float
) -> Tensor:
    """Add a dimensionless RMS-matched probe-gradient dose to residual states."""

    if states.shape != normalized_gradient.shape or states.ndim != 3:
        raise ValueError("States and probe gradient must share [position, square, feature] shape")
    if states.device != normalized_gradient.device or states.dtype != normalized_gradient.dtype:
        raise ValueError("States and probe gradient must share dtype and device")
    if not math.isfinite(dose):
        raise ValueError("Probe steering dose must be finite")
    state_norm = states.float().flatten(1).norm(dim=1, keepdim=True).clamp_min(1e-12)
    scale = dose * state_norm / math.sqrt(states.shape[1] * states.shape[2])
    return states + scale[:, None].to(states.dtype) * normalized_gradient


def frequency_baseline_metrics(
    train_labels: Sequence[int] | np.ndarray | Tensor,
    test_labels: Sequence[int] | np.ndarray | Tensor,
) -> dict[str, float]:
    """Train-only square-frequency baseline for probe selectivity."""

    train = torch.as_tensor(train_labels, dtype=torch.long)
    test = torch.as_tensor(test_labels, dtype=torch.long)
    if train.ndim != 1 or test.ndim != 1 or train.numel() == 0 or test.numel() == 0:
        raise ValueError("Frequency baseline requires non-empty label vectors")
    if bool(((train < 0) | (train >= 64)).any()) or bool(((test < 0) | (test >= 64)).any()):
        raise ValueError("Frequency labels must lie in [0, 63]")
    counts = torch.bincount(train, minlength=64).to(torch.float64) + 1.0
    log_probability = (counts / counts.sum()).log().expand(test.shape[0], -1)
    return square_prediction_metrics(log_probability, test)
