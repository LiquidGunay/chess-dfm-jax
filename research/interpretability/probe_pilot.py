"""Group-disjoint concept and learned-lookahead probes for Raw BT4 versus Hero."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import shutil
import subprocess
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from safetensors.torch import save_file as save_safetensors
from torch import Tensor

from research.interpretability.artifacts import (
    content_identity,
    sha256_file,
    verify_checksums,
    write_checksums,
    write_json_atomic,
    write_text_atomic,
)
from research.interpretability.chessbench import DEFAULT_OUTPUT_DIR as DEFAULT_CORPUS_DIR
from research.interpretability.chessbench import load_public_corpus
from research.interpretability.literature_pilot import bootstrap_mean_interval
from research.interpretability.lookahead import (
    FittedBilinearSquareProbe,
    evaluate_future_move_pair,
    fit_bilinear_square_probe,
    frequency_baseline_metrics,
    normalized_probe_gradient,
    select_future_ply,
    steer_with_probe_gradient,
)
from research.interpretability.models import load_bt4_comparison_models
from research.interpretability.pilot_corpus import write_deterministic_npz
from research.interpretability.probes import (
    FittedProbe,
    binary_metrics,
    evaluate_probe,
    fit_probe,
    fit_probe_grid,
    group_block_permutation,
    multiclass_metrics,
    paired_group_bootstrap_metric,
    regression_metrics,
    validate_group_disjoint_splits,
)
from research.interpretability.patching import target_log_odds


_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_BT4 = _REPO_ROOT / "models/source/extracted/BT4_exported.pb.gz"
DEFAULT_HERO_CHECKPOINT = _REPO_ROOT / "research/runs/torch_hero_epoch_v1/checkpoint"
DEFAULT_OUTPUT = _REPO_ROOT / "research/analysis/raw_hero_probe_lookahead_pilot_v1"
DEFAULT_LAYERS = (0, 7, 14)
ModelName = Literal["raw", "hero"]


@dataclass(frozen=True)
class ConceptSpec:
    name: str
    array: str
    task: Literal["binary", "multiclass", "regression"]


CONCEPT_SPECS = (
    ConceptSpec("piece_code", "piece_code_i8", "multiclass"),
    ConceptSpec("legal_destination", "legal_destination_u8", "binary"),
    ConceptSpec("opponent_attack_count", "attack_count_theirs_u8", "regression"),
)


def select_probe_rows(
    puzzles: Mapping[str, np.ndarray],
    *,
    future_ply: int,
    fit_count: int,
    selection_count: int,
    evaluation_count: int,
) -> dict[str, np.ndarray]:
    """Select deterministic train-fit/train-selection/dev rows without test access."""

    if min(fit_count, selection_count, evaluation_count) <= 0:
        raise ValueError("All probe split counts must be positive")
    valid = puzzles["future_valid_u8"][:, future_ply].astype(bool)

    def ordered(split: int) -> np.ndarray:
        rows = np.flatnonzero((puzzles["split_u8"] == split) & valid)
        return rows[np.argsort(puzzles["position_id"][rows], kind="stable")]

    training = ordered(0)
    development = ordered(1)
    required_train = fit_count + selection_count
    if training.size < required_train or development.size < evaluation_count:
        raise ValueError("Public corpus has insufficient valid rows for requested probe splits")
    result = {
        "fit": training[:fit_count],
        "selection": training[fit_count:required_train],
        "evaluation": development[:evaluation_count],
    }
    groups = np.concatenate([puzzles["group_id"][result[name]] for name in result])
    split_codes = np.concatenate(
        [np.full(result[name].size, index, dtype=np.uint8) for index, name in enumerate(result)]
    )
    validate_group_disjoint_splits(groups, split_codes)
    return result


def _git_record() -> dict[str, Any]:
    def run(*arguments: str) -> str:
        try:
            result = subprocess.run(
                ["git", *arguments],
                cwd=_REPO_ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        except FileNotFoundError:
            return ""
        return result.stdout.strip()

    status = run("status", "--short")
    return {
        "git_available": shutil.which("git") is not None,
        "commit": run("rev-parse", "HEAD") or None,
        "branch": run("branch", "--show-current") or None,
        "worktree_clean": not bool(status),
        "status_sha256": hashlib.sha256(status.encode()).hexdigest(),
        "status": status.splitlines(),
    }


def _environment(device: torch.device) -> dict[str, Any]:
    result: dict[str, Any] = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cuda_runtime": torch.version.cuda,
        "device": str(device),
    }
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        result["gpu"] = {
            "name": properties.name,
            "total_memory_bytes": properties.total_memory,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        }
    return result


def _output_staging(output: Path) -> tuple[Path, Path]:
    target = output.resolve()
    if target in {Path("/"), Path.home().resolve(), _REPO_ROOT.resolve()}:
        raise ValueError(f"Unsafe output path: {target}")
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite immutable run: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.with_name(f".{target.name}.{os.getpid()}.partial")
    if staging.exists():
        raise FileExistsError(f"Staging path already exists: {staging}")
    staging.mkdir()
    return target, staging


def _capture_states(
    models: Any,
    puzzles: Mapping[str, np.ndarray],
    rows: np.ndarray,
    layers: tuple[int, ...],
    *,
    batch_size: int,
    device: torch.device,
    log: Any,
) -> dict[ModelName, dict[int, Tensor]]:
    count = int(rows.size)
    states: dict[ModelName, dict[int, Tensor]] = {
        name: {layer: torch.empty((count, 64, 1024), dtype=torch.float16) for layer in layers}
        for name in ("raw", "hero")
    }
    for start in range(0, count, batch_size):
        selected = rows[start : start + batch_size]
        planes = torch.from_numpy(puzzles["current_planes_u8"][selected]).to(device)
        with torch.inference_mode():
            lattice = models.lattice_logits(
                planes,
                compute_dtype=torch.float32,
                capture_layers=layers,
            )
        if lattice.raw_captures is None or lattice.hero_captures is None:
            raise AssertionError("Probe capture pass lost representations")
        stop = start + selected.size
        for name, capture in (
            ("raw", lattice.raw_captures),
            ("hero", lattice.hero_captures),
        ):
            for offset, layer in enumerate(layers):
                states[name][layer][start:stop].copy_(
                    capture.resid_post_after_ln[offset]
                    .detach()
                    .to(
                        device="cpu",
                        dtype=torch.float16,
                    )
                )
        del lattice, planes
        if stop == count or stop % max(batch_size, 64) == 0:
            log(f"captured {stop}/{count} positions")
    return states


def _baseline_metrics(
    task: str,
    train_labels: np.ndarray,
    evaluation_labels: np.ndarray,
) -> dict[str, float]:
    if task == "binary":
        probability = float(np.mean(train_labels))
        prediction = np.full(evaluation_labels.shape, probability, dtype=np.float64)
        return binary_metrics(evaluation_labels, prediction)
    if task == "multiclass":
        classes = np.unique(train_labels)
        if not np.array_equal(classes, np.arange(classes[-1] + 1)):
            raise ValueError("Multiclass frequency baseline requires contiguous labels")
        counts = np.bincount(train_labels, minlength=classes.size).astype(np.float64) + 1.0
        probability = counts / counts.sum()
        prediction = np.repeat(probability[None, :], evaluation_labels.size, axis=0)
        return multiclass_metrics(evaluation_labels, prediction)
    prediction = np.full(evaluation_labels.shape, np.mean(train_labels), dtype=np.float64)
    return regression_metrics(evaluation_labels, prediction)


def _comparison_metric(
    task: str,
    classes: np.ndarray | None = None,
) -> Any:
    if task == "binary":
        return lambda labels, predictions: float(
            np.mean((np.asarray(predictions) >= 0.5) == np.asarray(labels))
        )
    if task == "multiclass":
        if classes is None:
            raise ValueError("Multiclass paired comparison needs a class map")
        return lambda labels, predictions: float(
            np.mean(classes[np.asarray(predictions).argmax(axis=1)] == np.asarray(labels))
        )
    return lambda labels, predictions: -float(
        np.mean(np.abs(np.asarray(predictions) - np.asarray(labels)))
    )


def _store_probe(prefix: str, probe: FittedProbe, tensors: dict[str, Tensor]) -> None:
    for name, value in probe.module.state_dict().items():
        tensors[f"{prefix}.module.{name}"] = value.detach().cpu().contiguous()
    tensors[f"{prefix}.feature_mean"] = probe.feature_mean.detach().cpu().contiguous()
    tensors[f"{prefix}.feature_scale"] = probe.feature_scale.detach().cpu().contiguous()
    tensors[f"{prefix}.target_mean"] = torch.tensor([probe.target_mean], dtype=torch.float32)
    tensors[f"{prefix}.target_scale"] = torch.tensor([probe.target_scale], dtype=torch.float32)
    if probe.classes is not None:
        tensors[f"{prefix}.classes"] = probe.classes.detach().cpu().contiguous()


def _store_bilinear(
    prefix: str,
    probe: FittedBilinearSquareProbe,
    tensors: dict[str, Tensor],
) -> None:
    for name, value in probe.module.state_dict().items():
        tensors[f"{prefix}.{name}"] = value.detach().cpu().contiguous()


def _strip_lookahead_tensors(report: Mapping[str, Any]) -> dict[str, Any]:
    return {name: value for name, value in report.items() if not isinstance(value, Tensor)}


def _future_arrays(
    states: Tensor,
    puzzles: Mapping[str, np.ndarray],
    rows: np.ndarray,
    *,
    ply: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    return select_future_ply(
        states,
        puzzles["future_origin_token_root_i16"][rows],
        puzzles["future_destination_token_root_i16"][rows],
        puzzles["future_valid_u8"][rows],
        ply=ply,
        conditioning="first_destination",
    )


def _causal_probe_validation(
    models: Any,
    probe: FittedBilinearSquareProbe,
    *,
    model_name: ModelName,
    layer: int,
    puzzles: Mapping[str, np.ndarray],
    rows: np.ndarray,
    future_ply: int,
    dose: float,
    seed: int,
    device: torch.device,
    bootstrap_samples: int,
) -> dict[str, Any]:
    arm = "RR" if model_name == "raw" else "HH"
    planes = torch.from_numpy(puzzles["current_planes_u8"][rows]).to(device)
    legal = torch.from_numpy(puzzles["legal_idx_u16"][rows]).to(device)
    legal_count = torch.from_numpy(puzzles["legal_count_u16"][rows]).to(device)
    targets = torch.from_numpy(puzzles["target_action_u16"][rows]).to(device)
    anchors = torch.from_numpy(puzzles["future_destination_token_root_i16"][rows, 0])
    destinations = torch.from_numpy(
        puzzles["future_destination_token_root_i16"][rows, future_ply]
    ).long()
    with torch.inference_mode():
        baseline = models.policy_logits_with_captures(
            planes,
            arm=arm,
            compute_dtype=torch.float32,
            capture_layers=(layer,),
        )
    if baseline.captures is None:
        raise AssertionError("Causal lookahead validation lost its native state")
    native_states = baseline.captures.resid_post_after_ln[0].detach().cpu().float()
    probe_gradient, _objective = normalized_probe_gradient(
        probe,
        native_states,
        anchors,
        destinations,
    )
    generator = torch.Generator(device="cpu").manual_seed(seed)
    random_gradient = torch.randn(native_states.shape, generator=generator)
    random_gradient /= random_gradient.flatten(1).norm(dim=1)[:, None, None].clamp_min(1e-12)
    baseline_odds = target_log_odds(baseline.logits, legal, legal_count, targets).cpu()
    rows_index = torch.arange(rows.size)

    def evaluate_direction(direction: Tensor) -> tuple[Tensor, Tensor]:
        policy_odds: dict[int, Tensor] = {}
        probe_logp: dict[int, Tensor] = {}
        for sign in (-1, 1):
            steered = steer_with_probe_gradient(
                native_states,
                direction,
                dose=sign * dose,
            )
            with torch.inference_mode():
                output = models.policy_logits_with_captures(
                    planes,
                    arm=arm,
                    compute_dtype=torch.float32,
                    capture_layers=(layer,),
                    resid_post_overrides={layer: steered.to(device)},
                )
                policy_odds[sign] = target_log_odds(
                    output.logits,
                    legal,
                    legal_count,
                    targets,
                ).cpu()
                logits = probe.logits(steered, anchors)
                probe_logp[sign] = torch.log_softmax(logits, dim=1)[
                    rows_index,
                    destinations,
                ]
        return (
            (policy_odds[1] - policy_odds[-1]) / (2.0 * dose),
            (probe_logp[1] - probe_logp[-1]) / (2.0 * dose),
        )

    probe_policy, probe_objective = evaluate_direction(probe_gradient)
    random_policy, random_objective = evaluate_direction(random_gradient)
    return {
        "position_count": int(rows.size),
        "dose": dose,
        "policy_target_log_odds_baseline": bootstrap_mean_interval(
            baseline_odds,
            samples=bootstrap_samples,
            seed=seed,
        ),
        "probe_direction_policy_derivative": bootstrap_mean_interval(
            probe_policy,
            samples=bootstrap_samples,
            seed=seed + 1,
        ),
        "random_direction_policy_derivative": bootstrap_mean_interval(
            random_policy,
            samples=bootstrap_samples,
            seed=seed + 2,
        ),
        "probe_direction_probe_objective_derivative": bootstrap_mean_interval(
            probe_objective,
            samples=bootstrap_samples,
            seed=seed + 3,
        ),
        "random_direction_probe_objective_derivative": bootstrap_mean_interval(
            random_objective,
            samples=bootstrap_samples,
            seed=seed + 4,
        ),
        "probe_objective_positive_for_every_position": bool((probe_objective > 0).all()),
    }


def run_probe_pilot(args: argparse.Namespace) -> dict[str, Any]:
    layers = tuple(args.layers)
    if not layers or tuple(sorted(set(layers))) != layers or layers[0] < 0 or layers[-1] >= 15:
        raise ValueError("layers must be unique, increasing BT4 layer indices")
    if args.future_ply < 0 or args.future_ply >= 7:
        raise ValueError("future_ply must lie in [0, 6]")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    target, staging = _output_staging(args.output)
    started = time.monotonic()
    started_utc = datetime.now(UTC)
    log_lines: list[str] = []

    def log(message: str) -> None:
        line = f"{time.monotonic() - started:10.3f}s {message}"
        print(line, flush=True)
        log_lines.append(line)

    try:
        torch.set_num_threads(args.cpu_threads)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        if device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.cuda.reset_peak_memory_stats(device)

        log("strict-loading public corpus and selecting nested development splits")
        corpus, corpus_manifest = load_public_corpus(args.corpus_manifest)
        puzzles = corpus["puzzles"]
        row_splits = select_probe_rows(
            puzzles,
            future_ply=args.future_ply,
            fit_count=args.fit_count,
            selection_count=args.selection_count,
            evaluation_count=args.evaluation_count,
        )
        split_order = ("fit", "selection", "evaluation")
        all_rows = np.concatenate([row_splits[name] for name in split_order])
        offsets: dict[str, slice] = {}
        start = 0
        for name in split_order:
            stop = start + row_splits[name].size
            offsets[name] = slice(start, stop)
            start = stop

        log("strict-loading Raw/Hero lattice and streaming selected residual states")
        models = load_bt4_comparison_models(
            raw_bt4_path=args.raw_bt4,
            hero_checkpoint_dir=args.hero_checkpoint,
            device=device,
        )
        model_descriptor = models.descriptor()
        states = _capture_states(
            models,
            puzzles,
            all_rows,
            layers,
            batch_size=args.capture_batch_size,
            device=device,
            log=log,
        )
        weight_tensors: dict[str, Tensor] = {}
        weight_metadata: dict[str, Any] = {}
        prediction_arrays: dict[str, np.ndarray] = {
            "evaluation_position_id": puzzles["position_id"][row_splits["evaluation"]],
            "evaluation_puzzle_id": puzzles["puzzle_id"][row_splits["evaluation"]],
        }

        concept_metrics: dict[str, Any] = {}
        log("fitting nested-split linear concept probes and label-permutation controls")
        for layer in layers:
            concept_metrics[str(layer)] = {}
            for spec_index, spec in enumerate(CONCEPT_SPECS):
                labels = puzzles[spec.array][all_rows]
                flattened_labels = labels.reshape(-1)
                repeated_groups = np.repeat(puzzles["group_id"][all_rows], 64)
                split_labels = {
                    name: flattened_labels[offsets[name].start * 64 : offsets[name].stop * 64]
                    for name in split_order
                }
                split_groups = {
                    name: repeated_groups[offsets[name].start * 64 : offsets[name].stop * 64]
                    for name in split_order
                }
                per_model: dict[str, Any] = {}
                predictions: dict[str, np.ndarray] = {}
                class_map: np.ndarray | None = None
                for model_index, model_name in enumerate(("raw", "hero")):
                    layer_states = states[model_name][layer]
                    split_features = {
                        name: layer_states[offsets[name]].float().reshape(-1, 1024)
                        for name in split_order
                    }
                    grid = fit_probe_grid(
                        split_features["fit"],
                        split_labels["fit"],
                        split_features["selection"],
                        split_labels["selection"],
                        task=spec.task,
                        l2_values=args.l2_values,
                        hidden_widths=(0,),
                        fit_kwargs={
                            "epochs": args.probe_epochs,
                            "learning_rate": args.probe_learning_rate,
                            "batch_size": args.probe_batch_size,
                            "seed": args.seed + 100 * layer + 10 * spec_index + model_index,
                            "group_ids": split_groups["fit"],
                            "device": device,
                        },
                    )
                    selected = grid.candidates[grid.best_index]
                    combined_features = torch.cat(
                        (split_features["fit"], split_features["selection"])
                    )
                    combined_labels = np.concatenate(
                        (split_labels["fit"], split_labels["selection"])
                    )
                    combined_groups = np.concatenate(
                        (split_groups["fit"], split_groups["selection"])
                    )
                    final_probe = fit_probe(
                        combined_features,
                        combined_labels,
                        task=spec.task,
                        l2=float(selected["l2"]),
                        hidden_width=0,
                        epochs=args.probe_epochs,
                        learning_rate=args.probe_learning_rate,
                        batch_size=args.probe_batch_size,
                        seed=args.seed + 10_000 + 100 * layer + 10 * spec_index + model_index,
                        group_ids=combined_groups,
                        device=device,
                    )
                    permuted = group_block_permutation(
                        combined_labels,
                        combined_groups,
                        seed=args.seed + 20_000 + 100 * layer + 10 * spec_index + model_index,
                    )
                    control_probe = fit_probe(
                        combined_features,
                        permuted,
                        task=spec.task,
                        l2=float(selected["l2"]),
                        hidden_width=0,
                        epochs=args.probe_epochs,
                        learning_rate=args.probe_learning_rate,
                        batch_size=args.probe_batch_size,
                        seed=args.seed + 30_000 + 100 * layer + 10 * spec_index + model_index,
                        group_ids=combined_groups,
                        device=device,
                    )
                    evaluation_prediction = final_probe.predict(
                        split_features["evaluation"]
                    ).numpy()
                    control_metrics = evaluate_probe(
                        control_probe,
                        split_features["evaluation"],
                        split_labels["evaluation"],
                    )
                    primary_metrics = evaluate_probe(
                        final_probe,
                        split_features["evaluation"],
                        split_labels["evaluation"],
                    )
                    per_model[model_name] = {
                        "grid": {
                            "selection_metric": grid.selection_metric,
                            "best_index": grid.best_index,
                            "candidates": list(grid.candidates),
                        },
                        "selected_l2": float(selected["l2"]),
                        "evaluation": primary_metrics,
                        "label_permutation_control": control_metrics,
                        "training_metadata": final_probe.metadata,
                    }
                    predictions[model_name] = evaluation_prediction
                    prefix = f"concept.{model_name}.L{layer}.{spec.name}"
                    prediction_arrays[prefix] = evaluation_prediction
                    _store_probe(prefix, final_probe, weight_tensors)
                    weight_metadata[prefix] = {
                        "probe": final_probe.metadata,
                        "classes": (
                            None if final_probe.classes is None else final_probe.classes.tolist()
                        ),
                    }
                    if final_probe.classes is not None:
                        observed_classes = final_probe.classes.numpy()
                        if class_map is None:
                            class_map = observed_classes
                        elif not np.array_equal(class_map, observed_classes):
                            raise ValueError("Raw/Hero concept class maps differ")
                    del split_features, combined_features, final_probe, control_probe
                    gc.collect()
                evaluation_labels = split_labels["evaluation"]
                train_labels = np.concatenate((split_labels["fit"], split_labels["selection"]))
                paired = paired_group_bootstrap_metric(
                    evaluation_labels,
                    predictions["raw"],
                    predictions["hero"],
                    split_groups["evaluation"],
                    _comparison_metric(spec.task, class_map),
                    seed=args.seed + 40_000 + 100 * layer + spec_index,
                    replicates=args.bootstrap_samples,
                )
                concept_metrics[str(layer)][spec.name] = {
                    "task": spec.task,
                    "array": spec.array,
                    "raw": per_model["raw"],
                    "hero": per_model["hero"],
                    "train_only_frequency_or_mean_baseline": _baseline_metrics(
                        spec.task,
                        train_labels,
                        evaluation_labels,
                    ),
                    "paired_evaluation": paired,
                }
                log(f"concept probes complete layer={layer} concept={spec.name}")

        lookahead_metrics: dict[str, Any] = {}
        lookahead_probes: dict[tuple[ModelName, int], FittedBilinearSquareProbe] = {}
        lookahead_predictions: dict[tuple[ModelName, int], dict[str, np.ndarray]] = {}
        log("fitting published rank-32 third-ply lookahead probes and controls")
        combined_rows = np.concatenate((row_splits["fit"], row_splits["selection"]))
        combined_slice = slice(0, offsets["selection"].stop)
        evaluation_groups = puzzles["group_id"][row_splits["evaluation"]]
        combined_groups = puzzles["group_id"][combined_rows]
        for layer in layers:
            lookahead_metrics[str(layer)] = {}
            for model_index, model_name in enumerate(("raw", "hero")):
                train_state, train_origin, train_destination, train_anchor = _future_arrays(
                    states[model_name][layer][combined_slice],
                    puzzles,
                    combined_rows,
                    ply=args.future_ply,
                )
                eval_state, eval_origin, eval_destination, eval_anchor = _future_arrays(
                    states[model_name][layer][offsets["evaluation"]],
                    puzzles,
                    row_splits["evaluation"],
                    ply=args.future_ply,
                )
                seed = args.seed + 50_000 + 100 * layer + model_index
                destination_probe = fit_bilinear_square_probe(
                    train_state,
                    train_destination,
                    train_anchor,
                    rank=32,
                    epochs=args.lookahead_epochs,
                    learning_rate=1e-2,
                    batch_size=64,
                    seed=seed,
                    group_ids=combined_groups,
                    device=device,
                )
                source_probe = fit_bilinear_square_probe(
                    train_state,
                    train_origin,
                    train_destination,
                    rank=32,
                    epochs=args.lookahead_epochs,
                    learning_rate=1e-2,
                    batch_size=64,
                    seed=seed + 1,
                    group_ids=combined_groups,
                    device=device,
                )
                permuted_destination = group_block_permutation(
                    train_destination.numpy(),
                    combined_groups,
                    seed=seed + 2,
                )
                permuted_origin = group_block_permutation(
                    train_origin.numpy(),
                    combined_groups,
                    seed=seed + 3,
                )
                destination_control = fit_bilinear_square_probe(
                    train_state,
                    permuted_destination,
                    train_anchor,
                    rank=32,
                    epochs=args.lookahead_epochs,
                    learning_rate=1e-2,
                    batch_size=64,
                    seed=seed + 4,
                    group_ids=combined_groups,
                    device=device,
                )
                source_control = fit_bilinear_square_probe(
                    train_state,
                    permuted_origin,
                    train_destination,
                    rank=32,
                    epochs=args.lookahead_epochs,
                    learning_rate=1e-2,
                    batch_size=64,
                    seed=seed + 5,
                    group_ids=combined_groups,
                    device=device,
                )
                report = evaluate_future_move_pair(
                    destination_probe,
                    source_probe,
                    eval_state,
                    eval_origin,
                    eval_destination,
                    eval_anchor,
                )
                control = evaluate_future_move_pair(
                    destination_control,
                    source_control,
                    eval_state,
                    eval_origin,
                    eval_destination,
                    eval_anchor,
                )
                lookahead_metrics[str(layer)][model_name] = {
                    "evaluation": _strip_lookahead_tensors(report),
                    "label_permutation_control": _strip_lookahead_tensors(control),
                    "destination_frequency_baseline": frequency_baseline_metrics(
                        train_destination,
                        eval_destination,
                    ),
                    "source_frequency_baseline": frequency_baseline_metrics(
                        train_origin,
                        eval_origin,
                    ),
                    "destination_training": destination_probe.metadata,
                    "source_training": source_probe.metadata,
                }
                predictions = {
                    "destination": report["predicted_destination"].numpy(),
                    "source": report["predicted_source"].numpy(),
                }
                lookahead_predictions[(model_name, layer)] = predictions
                lookahead_probes[(model_name, layer)] = destination_probe
                prefix = f"lookahead.{model_name}.L{layer}"
                prediction_arrays[f"{prefix}.destination"] = predictions["destination"]
                prediction_arrays[f"{prefix}.source"] = predictions["source"]
                _store_bilinear(f"{prefix}.destination", destination_probe, weight_tensors)
                _store_bilinear(f"{prefix}.source", source_probe, weight_tensors)
                weight_metadata[prefix] = {
                    "destination": destination_probe.metadata,
                    "source": source_probe.metadata,
                }
                log(f"lookahead probes complete layer={layer} model={model_name}")

            eval_origin = puzzles["future_origin_token_root_i16"][
                row_splits["evaluation"], args.future_ply
            ]
            eval_destination = puzzles["future_destination_token_root_i16"][
                row_splits["evaluation"], args.future_ply
            ]
            raw_prediction = lookahead_predictions[("raw", layer)]
            hero_prediction = lookahead_predictions[("hero", layer)]
            lookahead_metrics[str(layer)]["paired_evaluation"] = {
                "destination_accuracy": paired_group_bootstrap_metric(
                    eval_destination,
                    raw_prediction["destination"],
                    hero_prediction["destination"],
                    evaluation_groups,
                    lambda labels, prediction: float(np.mean(labels == prediction)),
                    seed=args.seed + 60_000 + layer,
                    replicates=args.bootstrap_samples,
                ),
                "source_given_predicted_destination_accuracy": paired_group_bootstrap_metric(
                    eval_origin,
                    raw_prediction["source"],
                    hero_prediction["source"],
                    evaluation_groups,
                    lambda labels, prediction: float(np.mean(labels == prediction)),
                    seed=args.seed + 61_000 + layer,
                    replicates=args.bootstrap_samples,
                ),
                "joint_move_accuracy": paired_group_bootstrap_metric(
                    np.stack((eval_origin, eval_destination), axis=1),
                    np.stack(
                        (raw_prediction["source"], raw_prediction["destination"]),
                        axis=1,
                    ),
                    np.stack(
                        (hero_prediction["source"], hero_prediction["destination"]),
                        axis=1,
                    ),
                    evaluation_groups,
                    lambda labels, prediction: float(np.mean(np.all(labels == prediction, axis=1))),
                    seed=args.seed + 62_000 + layer,
                    replicates=args.bootstrap_samples,
                ),
            }

        log("running matched causal probe-gradient versus random-direction interventions")
        causal_rows = row_splits["evaluation"][: args.causal_count]
        causal_metrics: dict[str, Any] = {}
        for layer in layers:
            causal_metrics[str(layer)] = {}
            for model_index, model_name in enumerate(("raw", "hero")):
                report = _causal_probe_validation(
                    models,
                    lookahead_probes[(model_name, layer)],
                    model_name=model_name,
                    layer=layer,
                    puzzles=puzzles,
                    rows=causal_rows,
                    future_ply=args.future_ply,
                    dose=args.causal_dose,
                    seed=args.seed + 70_000 + 100 * layer + model_index,
                    device=device,
                    bootstrap_samples=args.bootstrap_samples,
                )
                causal_metrics[str(layer)][model_name] = report
                log(f"causal lookahead validation complete layer={layer} model={model_name}")

        write_deterministic_npz(staging / "predictions.npz", prediction_arrays)
        save_safetensors(
            weight_tensors,
            str(staging / "probe_weights.safetensors"),
            metadata={
                "schema_version": "bt4-raw-hero-probe-weights-v1",
                "pickle": "forbidden",
            },
        )
        write_json_atomic(
            staging / "probe_metadata.json",
            {
                "schema_version": "bt4-raw-hero-probe-metadata-v1",
                "probes": weight_metadata,
            },
        )
        probe_objective_gate = all(
            causal_metrics[str(layer)][model_name]["probe_direction_probe_objective_derivative"][
                "lower"
            ]
            > 0.0
            for layer in layers
            for model_name in ("raw", "hero")
        )
        metrics = {
            "schema_version": "bt4-raw-hero-probe-lookahead-metrics-v1",
            "scope": "exploratory nested-development pilot",
            "layers": list(layers),
            "future_ply_zero_indexed": args.future_ply,
            "split_counts": {name: int(rows.size) for name, rows in row_splits.items()},
            "concept_probes": concept_metrics,
            "lookahead": lookahead_metrics,
            "causal_lookahead": causal_metrics,
            "correctness_gates": {
                "group_disjoint_nested_splits": True,
                "test_split_evaluated": False,
                "published_lookahead_architecture_match": all(
                    lookahead_metrics[str(layer)][model_name]["destination_training"][
                        "replication"
                    ]["architecture_match"]
                    for layer in layers
                    for model_name in ("raw", "hero")
                ),
                "probe_gradient_bootstrap_lower_bound_positive": probe_objective_gate,
                "pickle_free_weights": True,
            },
        }
        metrics["correctness_gates"]["passed"] = (
            all(
                value is True
                for name, value in metrics["correctness_gates"].items()
                if name != "test_split_evaluated"
            )
            and metrics["correctness_gates"]["test_split_evaluated"] is False
        )
        write_json_atomic(staging / "metrics.json", metrics)
        elapsed = time.monotonic() - started
        split_identity = {
            name: hashlib.sha256(
                "\n".join(str(puzzles["position_id"][row]) for row in rows).encode()
            ).hexdigest()
            for name, rows in row_splits.items()
        }
        payloads = {
            "metrics": content_identity(metrics),
            "splits": split_identity,
            "weights": sha256_file(staging / "probe_weights.safetensors"),
            "predictions": sha256_file(staging / "predictions.npz"),
            "raw": models.raw_manifest["raw_asset"]["sha256"],
            "hero": models.hero_manifest["state"]["sha256"],
        }
        run_id = content_identity(payloads)
        manifest = {
            "schema_version": "bt4-raw-hero-probe-lookahead-run-v1",
            "run_id": f"sha256:{run_id}",
            "started_utc": started_utc.isoformat(),
            "elapsed_seconds": elapsed,
            "git": _git_record(),
            "environment": _environment(device),
            "model_lattice": model_descriptor,
            "public_corpus_manifest": {
                "path": str(args.corpus_manifest.resolve()),
                "integrity_sha256": corpus_manifest["manifest_integrity"]["sha256"],
                "split_policy": "train fit + train selection; development evaluation; test untouched",
                "split_position_sha256": split_identity,
            },
            "experiment": {
                "layers": list(layers),
                "activation_boundary": "resid_post_after_ln",
                "activation_storage_dtype": "float16 ephemeral CPU only",
                "concepts": [spec.name for spec in CONCEPT_SPECS],
                "linear_probe_l2_grid": args.l2_values,
                "probe_epochs": args.probe_epochs,
                "lookahead_replication": "rank-32, five epochs, Adam 1e-2, scalar bias",
                "future_ply_zero_indexed": args.future_ply,
                "causal_dose": args.causal_dose,
                "bootstrap_samples": args.bootstrap_samples,
                "seed": args.seed,
                "confirmatory": False,
                "test_split_opened": False,
            },
            "cost": {
                "backend": args.backend,
                "provider_dollars": args.provider_dollars,
                "declared_attempt_upper_bound_dollars": (args.declared_attempt_upper_bound_dollars),
                "input_bundle_sha256": args.input_bundle_sha256,
            },
            "payload_identities": payloads,
            "source_files": {
                path.relative_to(_REPO_ROOT).as_posix(): sha256_file(path)
                for path in (
                    Path(__file__),
                    Path(__file__).with_name("lookahead.py"),
                    Path(__file__).with_name("probes.py"),
                    _REPO_ROOT / "research/train_torch.py",
                )
            },
        }
        write_json_atomic(staging / "manifest.json", manifest)
        log(f"correctness gates passed={metrics['correctness_gates']['passed']}")
        log_lines.append(f"run_id sha256:{run_id}")
        write_text_atomic(staging / "run.log", "\n".join(log_lines) + "\n")
        write_checksums(staging)
        verify_checksums(staging)
        os.replace(staging, target)
        checksums = verify_checksums(target)
        return {
            "output": str(target),
            "run_id": f"sha256:{run_id}",
            "metrics_sha256": checksums["metrics.json"],
            "gates_passed": metrics["correctness_gates"]["passed"],
            "elapsed_seconds": elapsed,
        }
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    finally:
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-bt4", type=Path, default=DEFAULT_RAW_BT4)
    parser.add_argument("--hero-checkpoint", type=Path, default=DEFAULT_HERO_CHECKPOINT)
    parser.add_argument(
        "--corpus-manifest",
        type=Path,
        default=DEFAULT_CORPUS_DIR / "manifest.json",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--layers", type=int, nargs="+", default=DEFAULT_LAYERS)
    parser.add_argument("--future-ply", type=int, default=2)
    parser.add_argument("--fit-count", type=int, default=256)
    parser.add_argument("--selection-count", type=int, default=64)
    parser.add_argument("--evaluation-count", type=int, default=128)
    parser.add_argument("--capture-batch-size", type=int, default=4)
    parser.add_argument("--l2-values", type=float, nargs="+", default=(0.0, 1e-5))
    parser.add_argument("--probe-epochs", type=int, default=20)
    parser.add_argument("--probe-learning-rate", type=float, default=0.03)
    parser.add_argument("--probe-batch-size", type=int, default=4096)
    parser.add_argument("--lookahead-epochs", type=int, default=5)
    parser.add_argument("--causal-count", type=int, default=8)
    parser.add_argument("--causal-dose", type=float, default=1.0)
    parser.add_argument("--bootstrap-samples", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=20260808)
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--backend", default="local GTX 1660 Ti")
    parser.add_argument("--provider-dollars", type=float, default=0.0)
    parser.add_argument("--declared-attempt-upper-bound-dollars", type=float, default=0.0)
    parser.add_argument("--input-bundle-sha256")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    result = run_probe_pilot(build_parser().parse_args(argv))
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
