"""Inference-aligned policy CE on shared, content-bound validation positions."""
import gc
import json
from pathlib import Path
import time

import numpy as np

from research.evaluate_common_policy import input_planes, sha256


def validate_arrays(data, count):
    x, y, legal = (data[k] for k in ("planes", "actions", "legal"))
    if x.shape != (count, 112, 8, 8) or x.dtype != np.uint8:
        raise ValueError("Invalid input planes")
    if y.shape != (count,) or not np.issubdtype(y.dtype, np.integer):
        raise ValueError("Invalid played-move targets")
    if legal.shape != (count, 1858) or legal.dtype != np.bool_:
        raise ValueError("Invalid legal masks")
    if np.any(y < 0) or np.any(y >= 1858) or not np.all(legal[np.arange(count), y]):
        raise ValueError("Illegal target")
    if not np.any(x[:, 13:104]):
        raise ValueError("Full-history pool contains no history")
    return {"positions": count, "history_nonzero_positions": int(np.any(x[:, 13:104], axis=(1, 2, 3)).sum()),
            "legal_count_min": int(legal.sum(1).min()), "legal_count_max": int(legal.sum(1).max())}


def prepare_hero(root, manifest_sha256, output):
    from research.hero_history_batches import HistoryTrajectoryBatches
    from research.train_torch import canonicalize_trajectory_batch
    batches = HistoryTrajectoryBatches(root, split="validation", batch_size=64, horizon=8,
        manifest_sha256=manifest_sha256, canonicalize=canonicalize_trajectory_batch)
    if batches.available_steps != 127:
        raise ValueError("Expected exactly the common 8128-position Hero validation pool")
    pieces = {k: [] for k in ("planes", "actions", "legal")}
    for i in range(127):
        batch = batches.batch_at(i)  # Every consumed shard is size/hash verified.
        x = batch["current_planes"]
        if not np.array_equal(x, x.astype(np.uint8)):
            raise ValueError("Planes cannot be losslessly stored as uint8")
        legal = np.zeros((64, 1858), dtype=np.bool_)
        for row in range(64):
            count = int(batch["legal_count"][row, 0])
            legal[row, batch["legal_idx"][row, 0, :count]] = True
        pieces["planes"].append(x.astype(np.uint8))
        pieces["actions"].append(batch["action_indices"][:, 0])
        pieces["legal"].append(legal)
    data = {k: np.concatenate(v) for k, v in pieces.items()}
    checks = validate_arrays(data, 8128)
    with Path(output).open("xb") as stream:
        np.savez_compressed(stream, **data)
    return {**checks, "file_sha256": sha256(Path(output)), "size_bytes": Path(output).stat().st_size,
            "data": batches.provenance()}


def metric_vectors(logits, legal, targets):
    import torch
    import torch.nn.functional as F
    if not torch.isfinite(logits).all() or not legal.gather(1, targets[:, None]).all():
        raise ValueError("Nonfinite policy or illegal target")
    masked = logits.float().masked_fill(~legal, -torch.inf)
    return (F.cross_entropy(masked, targets, reduction="none").cpu().numpy(),
            (masked.argmax(-1) == targets).cpu().numpy())


def refinement_logits(model, latents, base, legal, *, check_one_pass=False):
    """Observe the actual arena helper, rather than reimplementing its schedule."""
    import torch
    from research.train_torch import _torch_refine_dfm_actions
    original = model.planner
    captured = []
    def observe(*args, **kwargs):
        result = original(*args, **kwargs)
        if not isinstance(result, torch.Tensor):
            raise ValueError("This evaluator is for the frozen feedback-off recipe")
        captured.append(result[:, 0].detach())
        return result
    model.planner = observe
    try:
        selected = _torch_refine_dfm_actions(model, latents, base, legal,
            refinement_passes=8, compute_dtype=torch.bfloat16)
    finally:
        model.planner = original
    if len(captured) != 8:
        raise ValueError("Wrong refinement call count")
    first, last = captured[0], captured[-1]
    if not torch.equal(last.float().masked_fill(~legal, -torch.inf).argmax(-1), selected):
        raise ValueError("Final-pass logits do not reproduce arena actions")
    if check_one_pass:
        one = _torch_refine_dfm_actions(model, latents, base, legal,
            refinement_passes=1, compute_dtype=torch.bfloat16)
        if not torch.equal(first.float().masked_fill(~legal, -torch.inf).argmax(-1), one):
            raise ValueError("First-pass logits do not reproduce one-pass arena actions")
    return {"dfm_one_pass": first, "dfm_eight_pass": last}


def evaluate(pools, models, raw, output):
    import torch
    from research.train_torch import Config, HERO_CONFIG, load_raw_bt4_hero_model, load_model_checkpoint
    from research.input_contract import torch_hero_plane_history_mode
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("high")
    device, dtype = torch.device("cuda"), torch.bfloat16
    rows, vectors, parity = [], {}, []
    start_time = time.monotonic()
    for name, record in models.items():
        root = record.get("root")
        if root:
            config_record = json.loads((Path(root) / "run_config.json").read_text())
            config = Config(**config_record["config"])
            if torch_hero_plane_history_mode(config_record) != record["plane_history_mode"]:
                raise ValueError("Training history contract differs")
        else:
            config = HERO_CONFIG
        if config.dfm_state_source != "trunk" or any((config.dfm_condition_on_current_jepa_state,
                config.dfm_jepa_fusion_mode != "none", config.dfm_closed_loop_mode != "none",
                config.jepa_feedback_mode != "none")):
            raise ValueError("Unsupported feedback architecture")
        model, _ = load_raw_bt4_hero_model(device=device, raw_bt4_path=raw, config=config)
        if root:
            loaded = load_model_checkpoint(checkpoint_dir=Path(root) / "checkpoint", model=model)
            if loaded["state"]["sha256"] != record["state_sha256"]:
                raise ValueError("Restored checkpoint differs from frozen identity")
        model.eval()
        for pool_name, pool in pools.items():
            with np.load(pool["path"], allow_pickle=False) as archive:
                data = {k: archive[k] for k in archive.files}
            n = pool["positions"]
            validate_arrays(data, n)
            modes = ["native_policy", "dfm_one_pass", "dfm_eight_pass"] if root else ["native_policy"]
            samples = {k: {"ce": [], "correct": []} for k in modes}
            with torch.inference_mode():
                for start in range(0, n, 16):
                    x = input_planes(data["planes"][start:start+16], record["plane_history_mode"])
                    planes = torch.from_numpy(x).to(device)
                    target = torch.from_numpy(data["actions"][start:start+16]).long().to(device)
                    legal = torch.from_numpy(data["legal"][start:start+16]).to(device)
                    tokens = model.encoder.encode_current(planes, compute_dtype=dtype, remat=False)
                    base, _ = model.policy_outputs(tokens, dtype)
                    outputs = {"native_policy": base if base.ndim == 2 else base[:, 0]}
                    if root:
                        z = model.dfm_state_projector(tokens, dtype)
                        z = model.condition_dfm_latents(z, None, dtype).latents
                        outputs.update(refinement_logits(model, z, base, legal, check_one_pass=(start == 0)))
                    for mode, logits in outputs.items():
                        ce, correct = metric_vectors(logits, legal, target)
                        samples[mode]["ce"].append(ce); samples[mode]["correct"].append(correct)
                    if start % 2048 == 0:
                        print(json.dumps({"model": name, "pool": pool_name, "completed": start+16, "total": n}), flush=True)
            for mode, metrics in samples.items():
                ce = np.concatenate(metrics["ce"]); correct = np.concatenate(metrics["correct"])
                key = f"{pool_name}__{name}__{mode}"
                vectors[key+"__ce"] = ce; vectors[key+"__correct"] = correct
                row = {"pool": pool_name, "model": name, "mode": mode, "positions": n,
                       "plane_history_mode": record["plane_history_mode"], "state_sha256": record["state_sha256"],
                       "legal_conditional_ce": float(ce.astype(np.float64).mean()),
                       "legal_top1_accuracy": float(correct.mean())}
                rows.append(row); print(json.dumps({"result_row": row}), flush=True)
            parity.append({"pool": pool_name, "model": name, "checked_one_pass_actions": min(16, n) if root else 0,
                           "checked_eight_pass_actions": n if root else 0})
        del model, tokens, base, outputs
        gc.collect(); torch.cuda.empty_cache()
    with (Path(output) / "position_metrics.npz").open("xb") as stream:
        np.savez_compressed(stream, **vectors)
    return {"experiment_id": "EV-007", "rows": rows, "arena_action_parity": parity,
            "position_metrics_sha256": sha256(Path(output) / "position_metrics.npz"),
            "compute_dtype": "bfloat16", "physical_batch_size": 16,
            "metric": "Mean negative log probability of the played next move, normalized over its complete legal set, in nats.",
            "inference": "Native logits; DFM first and eighth root logits captured from the unchanged arena refinement helper, without teacher forcing.",
            "seconds": time.monotonic()-start_time}
