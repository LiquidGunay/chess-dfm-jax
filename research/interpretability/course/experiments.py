"""Deterministic constructed experiments with known mechanisms."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .stats import first_illegal_mask, masked_softmax


@dataclass(frozen=True)
class PolicyExperiment:
    probabilities: dict[str, np.ndarray]
    targets: np.ndarray
    legal: np.ndarray
    groups: np.ndarray


def toy_paired_outcomes(seed: int = 7) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    sizes = np.asarray([4, 7, 11, 5, 13, 6, 9, 8])
    groups = np.repeat(np.arange(sizes.size), sizes)
    group_effect = rng.normal(0.03, 0.08, sizes.size)
    baseline = rng.normal(0.0, 0.15, groups.size)
    first = baseline + group_effect[groups] + rng.normal(0.0, 0.03, groups.size)
    second = baseline + rng.normal(0.0, 0.03, groups.size)
    return first, second, groups


def toy_policy_experiment(seed: int = 11) -> PolicyExperiment:
    rng = np.random.default_rng(seed)
    positions, actions, games = 48, 12, 8
    groups = np.repeat(np.arange(games), positions // games)
    state = rng.normal(size=(positions, 5))
    policy_map = rng.normal(scale=0.7, size=(5, actions))
    raw_logits = state @ policy_map + rng.normal(scale=0.15, size=(positions, actions))
    encoder_delta = 0.75 * np.outer(np.tanh(state[:, 0]), rng.normal(size=actions))
    head_delta = rng.normal(scale=0.025, size=(positions, actions))
    interaction = 0.08 * np.outer(np.tanh(state[:, 1]), rng.normal(size=actions))
    logits = {
        "RR": raw_logits,
        "RH": raw_logits + head_delta,
        "HR": raw_logits + encoder_delta,
        "HH": raw_logits + encoder_delta + head_delta + interaction,
    }
    legal = rng.random((positions, actions)) > 0.28
    legal[:, 0] = True
    targets = np.argmax(np.where(legal, logits["HH"] + rng.normal(scale=0.8, size=logits["HH"].shape), -np.inf), axis=1)
    probabilities = {arm: masked_softmax(value, legal) for arm, value in logits.items()}
    return PolicyExperiment(probabilities, targets, legal, groups)


def toy_representations(seed: int = 13) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    latent = rng.normal(size=(160, 6))
    first = latent @ rng.normal(size=(6, 16)) + rng.normal(scale=0.05, size=(160, 16))
    second = first + 0.2 * np.outer(np.tanh(latent[:, 0]), rng.normal(size=16))
    return first, second


def transform_representation(
    values: np.ndarray,
    kind: Literal["identity", "rotate", "permute", "scale", "noise", "duplicate"],
    *,
    strength: float,
    seed: int = 101,
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    if kind == "identity":
        return values.copy()
    if kind == "rotate":
        q, _ = np.linalg.qr(rng.normal(size=(values.shape[1], values.shape[1])))
        mixed = values @ q
        return (1.0 - strength) * values + strength * mixed
    if kind == "permute":
        permuted = values[:, rng.permutation(values.shape[1])]
        return (1.0 - strength) * values + strength * permuted
    if kind == "scale":
        scales = np.exp(strength * rng.normal(size=values.shape[1]))
        return values * scales
    if kind == "noise":
        return values + strength * values.std() * rng.normal(size=values.shape)
    if kind == "duplicate":
        duplicated = values.copy()
        duplicated[:, values.shape[1] // 2 :] = values[:, : values.shape[1] // 2]
        return (1.0 - strength) * values + strength * duplicated
    raise ValueError(kind)


@dataclass(frozen=True)
class ProbeExperiment:
    features: np.ndarray
    board_baseline: np.ndarray
    labels: np.ndarray
    games: np.ndarray
    row_fit: np.ndarray
    row_selection: np.ndarray
    row_eval: np.ndarray
    group_fit: np.ndarray
    group_selection: np.ndarray
    group_eval: np.ndarray


def toy_probe_experiment(seed: int = 17) -> ProbeExperiment:
    rng = np.random.default_rng(seed)
    games, per_game = 24, 8
    game_ids = np.repeat(np.arange(games), per_game)
    game_label = rng.integers(0, 2, games)
    local_signal = rng.normal(size=games * per_game)
    logits = 0.55 * local_signal + 1.3 * (game_label[game_ids] * 2 - 1)
    labels = (logits + rng.normal(scale=1.0, size=logits.size) > 0.0).astype(np.int64)
    board = np.column_stack(
        [local_signal, rng.normal(size=(logits.size, 3))]
    )
    fingerprints = np.eye(games)[game_ids]
    features = np.column_stack(
        [board, fingerprints, rng.normal(scale=0.25, size=(logits.size, 4))]
    )
    order = rng.permutation(logits.size)
    row_fit = np.zeros(logits.size, dtype=bool)
    row_selection = np.zeros(logits.size, dtype=bool)
    fit_end = int(0.6 * logits.size)
    selection_end = int(0.8 * logits.size)
    row_fit[order[:fit_end]] = True
    row_selection[order[fit_end:selection_end]] = True
    row_eval = ~(row_fit | row_selection)
    group_fit = game_ids < 12
    group_selection = (game_ids >= 12) & (game_ids < 18)
    group_eval = game_ids >= 18
    return ProbeExperiment(
        features,
        board,
        labels,
        game_ids,
        row_fit,
        row_selection,
        row_eval,
        group_fit,
        group_selection,
        group_eval,
    )


def toy_attribution(
    *,
    saturation: float,
    seed: int = 19,
) -> dict[str, np.ndarray | float]:
    rng = np.random.default_rng(seed)
    features = rng.normal(scale=0.55, size=64)
    features[[9, 18, 27, 36]] += np.asarray([1.4, -1.2, 1.7, -1.5])
    weights = rng.normal(scale=0.2, size=64)
    weights[[9, 18, 27, 36]] += np.asarray([1.4, -1.1, 1.7, -1.6])

    def output(x: np.ndarray) -> float:
        return float(np.tanh(saturation * np.dot(x, weights)))

    score = output(features)
    gradient = saturation * (1.0 - score**2) * weights
    input_gradient = features * gradient
    steps = 64
    alphas = np.linspace(0.0, 1.0, steps + 1)[1:]
    path_gradients = []
    for alpha in alphas:
        path_score = output(alpha * features)
        path_gradients.append(saturation * (1.0 - path_score**2) * weights)
    integrated = features * np.mean(path_gradients, axis=0)
    occlusion = np.asarray(
        [score - output(np.where(np.arange(64) == index, 0.0, features)) for index in range(64)]
    )
    completeness_error = abs(integrated.sum() - (score - output(np.zeros_like(features))))
    return {
        "features": features,
        "weights": weights,
        "gradient": gradient,
        "input_gradient": input_gradient,
        "integrated_gradients": integrated,
        "occlusion": occlusion,
        "output": score,
        "completeness_error": float(completeness_error),
    }


def toy_intervention(
    doses: np.ndarray,
    *,
    seed: int = 23,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    hidden = rng.normal(size=(96, 12))
    # Construct an orthonormal concept basis so the controls have known
    # semantics and known causal loadings, rather than post-hoc names.
    concept_basis, _ = np.linalg.qr(rng.normal(size=(12, 12)))
    candidate = concept_basis[:, 0]  # constructed pin signal
    placebo = concept_basis[:, 1]  # constructed own-queen signal
    random = concept_basis[:, 2]  # prespecified matched random direction
    nuisance = concept_basis[:, 3]
    output_direction = 0.8 * candidate + 0.6 * nuisance
    output_direction /= np.linalg.norm(output_direction)
    positive_control = output_direction.copy()  # known endpoint gradient
    baseline = hidden @ output_direction
    results = {}
    for name, direction in {
        "candidate": candidate,
        "semantic_placebo": placebo,
        "random_matched": random,
        "positive_control": positive_control,
    }.items():
        curves = []
        for dose in np.asarray(doses, dtype=np.float64):
            changed = (hidden + dose * direction) @ output_direction
            curves.append(np.mean(changed - baseline))
        results[name] = np.asarray(curves)
    density_distance = np.linalg.norm(np.asarray(doses)[:, None] * candidate, axis=1)
    results["density_distance"] = density_distance
    return results


def toy_lookahead_rollout(
    *,
    examples: int = 96,
    horizon: int = 8,
    predicted_error_rate: float = 0.18,
    seed: int = 29,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    teacher_actions = rng.integers(0, 12, size=(examples, horizon))
    predicted_actions = teacher_actions.copy()
    errors = rng.random((examples, horizon)) < predicted_error_rate
    predicted_actions[errors] = rng.integers(0, 12, size=np.count_nonzero(errors))
    legal = rng.random((examples, horizon)) > (0.02 + 0.08 * np.arange(horizon) / horizon)
    legal &= predicted_actions != 11
    valid_prefix, first_illegal = first_illegal_mask(legal)
    state_steps = np.cumsum(teacher_actions[..., None] * rng.normal(size=(1, 1, 6)), axis=1)
    teacher_pred = state_steps + rng.normal(scale=0.2, size=state_steps.shape)
    exposure = np.cumsum(predicted_actions != teacher_actions, axis=1)[..., None]
    predicted_pred = state_steps + rng.normal(scale=0.2, size=state_steps.shape) + 0.35 * exposure
    identity = np.repeat(np.zeros((examples, 1, 6)), horizon, axis=1)
    return {
        "teacher_actions": teacher_actions,
        "predicted_actions": predicted_actions,
        "legal": legal,
        "valid_prefix": valid_prefix,
        "first_illegal": first_illegal,
        "targets": state_steps,
        "teacher_prediction": teacher_pred,
        "predicted_prediction": predicted_pred,
        "identity_prediction": identity,
    }


def toy_sparse_frontier(
    k_values: np.ndarray,
    *,
    seed: int = 31,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    samples, latent_width, observed_width = 256, 24, 12
    codes = np.zeros((samples, latent_width))
    for row in range(samples):
        active = rng.choice(latent_width, size=3, replace=False)
        codes[row, active] = rng.lognormal(mean=-0.2, sigma=0.55, size=3)
    decoder = rng.normal(size=(latent_width, observed_width))
    decoder /= np.linalg.norm(decoder, axis=1, keepdims=True)
    observed = codes @ decoder + rng.normal(scale=0.03, size=(samples, observed_width))
    # Match the nonnegative code derived in the lesson: approximate linear
    # encoding, ReLU, then top-k over nonnegative magnitudes.
    encoded = np.maximum(observed @ np.linalg.pinv(decoder), 0.0)
    nmse, active_fraction, dead_fraction = [], [], []
    for k in np.asarray(k_values, dtype=np.int64):
        keep = np.argsort(encoded, axis=1)[:, -int(k) :]
        sparse = np.zeros_like(encoded)
        np.put_along_axis(sparse, keep, np.take_along_axis(encoded, keep, axis=1), axis=1)
        reconstructed = sparse @ decoder
        nmse.append(np.mean((reconstructed - observed) ** 2) / np.var(observed))
        active_fraction.append(np.count_nonzero(sparse) / sparse.size)
        dead_fraction.append(np.mean(np.count_nonzero(sparse, axis=0) == 0))
    return {
        "k": np.asarray(k_values),
        "normalized_mse": np.asarray(nmse),
        "active_fraction": np.asarray(active_fraction),
        "dead_fraction": np.asarray(dead_fraction),
        "minimum_ground_truth_code": np.asarray(codes.min()),
        "minimum_encoded_coefficient": np.asarray(encoded.min()),
    }


def toy_circuit_truth_table() -> list[dict[str, int]]:
    rows = []
    for attack in (0, 1):
        for defender in (0, 1):
            for king_exposed in (0, 1):
                threat = attack & (1 - defender)
                redundant_copy = threat
                output = (threat | redundant_copy) & king_exposed
                rows.append(
                    {
                        "attack": attack,
                        "defender": defender,
                        "king_exposed": king_exposed,
                        "threat": threat,
                        "redundant_copy": redundant_copy,
                        "output": output,
                    }
                )
    return rows
