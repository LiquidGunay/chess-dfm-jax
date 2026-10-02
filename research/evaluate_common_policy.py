"""Matched root-policy validation; never compare unlike aggregate objectives."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def input_planes(planes: np.ndarray, mode: str) -> np.ndarray:
    result = planes.astype(np.float32, copy=True)
    if mode == "current_only_as_preprocessed":
        result[:, 12:104] = 0
    elif mode != "lc0_full_history_as_preprocessed":
        raise ValueError("Unknown input-history mode")
    return result


def legal_metrics(logits, legal, target) -> tuple[float, int]:
    import torch
    import torch.nn.functional as F

    if not torch.isfinite(logits).all() or not legal.gather(1, target[:, None]).all():
        raise ValueError("Nonfinite logits or illegal validation target")
    values = logits.float().masked_fill(~legal, -torch.inf)
    return F.cross_entropy(values, target, reduction="sum").item(), int((values.argmax(-1) == target).sum())


def prepare(dataset: Path, output: Path, manifest: Path) -> dict:
    from chess_dfm_jax.data.lc0_sequential import SequentialBatches

    batches = SequentialBatches(dataset, split="validation", batch_size=64,
                                horizon=8, seed=20000, shuffle_batches=True)
    provenance = batches.provenance()
    if provenance["file_manifest_sha256"] != "9240e014a2d9126e9a937cb2c908af68bf1aab28f51e692bc2a645111d859629":
        raise ValueError("Frozen one-epoch validation source changed")
    slots = [list(batches._slot_for_step(i)) for i in range(128)]
    slot_sha = hashlib.sha256(json.dumps(slots, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()
    if slot_sha != "629bb68af7d0ec4bb4546320083664547bc023ebe5917ee4c1263e9bd8558333":
        raise ValueError("Frozen validation positions changed")
    planes, actions, masks = [], [], []
    for i in range(128):
        batch = batches.batch_at(i)
        x = batch["current_planes"]
        if not np.array_equal(x, x.astype(np.uint8)):
            raise ValueError("Input planes are not losslessly uint8 encodable")
        mask = np.zeros((64, 1858), dtype=np.bool_)
        for row in range(64):
            count = int(batch["legal_count"][row, 0])
            mask[row, batch["legal_idx"][row, 0, :count]] = True
        target = batch["action_indices"][:, 0]
        if not np.all(mask[np.arange(64), target]):
            raise ValueError("Validation target is not legal")
        planes.append(x.astype(np.uint8)); actions.append(target); masks.append(mask)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as handle:
        np.savez_compressed(handle, planes=np.concatenate(planes), actions=np.concatenate(actions), legal=np.concatenate(masks))
    record = {
        "schema_version": "chess-dfm-common-policy-validation-input-v1", "positions": 8192,
        "file_sha256": sha256(output), "size_bytes": output.stat().st_size,
        "selected_slots_sha256": slot_sha, "data": provenance,
        "endpoint": "Played-move root legal-conditional cross-entropy and top-1, at t=0/all action slots masked; native-policy and one-pass DFM logits.",
        "not_comparable_to": "Saved composite training validation losses or randomly masked root metrics.",
        "input_modes": ["current_only_as_preprocessed", "lc0_full_history_as_preprocessed"],
        "current_only_transform": "Zero planes 12:104, including the current repetition plane; all other current and auxiliary planes retained.",
    }
    with manifest.open("x") as handle:
        json.dump(record, handle, indent=2, sort_keys=True); handle.write("\n")
    return record


def evaluate(inputs: Path, checkpoints: dict[str, Path], raw_model: Path) -> dict:
    import torch
    from research.train_torch import Config, load_raw_bt4_hero_model, load_checkpoint_model_for_evaluation

    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("high")
    with np.load(inputs, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    device = torch.device("cuda")
    dtype = torch.bfloat16
    output = {"schema_version": "chess-dfm-common-policy-validation-v1", "positions": len(data["planes"]),
              "inputs_sha256": sha256(inputs), "rows": [], "compute_dtype": "bfloat16", "batch_size": 16,
              "target": "actual played next move", "dfm_time": 0.0, "dfm_input": "all eight action tokens masked"}
    for name, root in checkpoints.items():
        config = Config(**json.loads((root / "run_config.json").read_text())["config"])
        if config.dfm_state_source != "trunk" or any([config.dfm_condition_on_current_jepa_state,
                config.dfm_jepa_fusion_mode != "none", config.dfm_closed_loop_mode != "none", config.jepa_feedback_mode != "none"]):
            raise ValueError("Common-policy evaluator only supports the frozen feedback-off architecture")
        model, _ = load_raw_bt4_hero_model(device=device, raw_bt4_path=raw_model, config=config)
        manifest = load_checkpoint_model_for_evaluation(checkpoint_dir=root / "checkpoint", model=model)
        model.eval()
        for mode in ["current_only_as_preprocessed", "lc0_full_history_as_preprocessed"]:
            sums = {head: {"ce": 0.0, "correct": 0} for head in ["native_policy", "dfm_one_pass"]}
            with torch.inference_mode():
                for start in range(0, output["positions"], 16):
                    x = input_planes(data["planes"][start:start + 16], mode)
                    planes = torch.from_numpy(x).to(device)
                    target = torch.from_numpy(data["actions"][start:start + 16]).long().to(device)
                    legal = torch.from_numpy(data["legal"][start:start + 16]).to(device)
                    tokens = model.encoder.encode_current(planes, compute_dtype=dtype, remat=False)
                    base, _ = model.policy_outputs(tokens, dtype)
                    z = model.dfm_state_projector(tokens, dtype)
                    masked = torch.full((len(x), config.horizon), 1858, device=device, dtype=torch.long)
                    logits = model.planner(z, masked, torch.zeros(len(x), device=device), dtype, base_root_logits=base)
                    for head, values in [("native_policy", base), ("dfm_one_pass", logits[:, 0])]:
                        values = values[:, 0] if values.ndim == 3 else values
                        ce, correct = legal_metrics(values, legal, target)
                        sums[head]["ce"] += ce
                        sums[head]["correct"] += correct
            row = {"model": name, "state_sha256": manifest["state"]["sha256"], "plane_history_mode": mode,
                   "metrics": {head: {"legal_conditional_ce": s["ce"] / output["positions"],
                                      "legal_top1_accuracy": s["correct"] / output["positions"]} for head, s in sums.items()}}
            output["rows"].append(row)
            print(json.dumps(row), flush=True)
        del model, tokens, base, z, logits
        gc.collect(); torch.cuda.empty_cache()
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.dataset_root, args.output, args.manifest)))


if __name__ == "__main__":
    main()
