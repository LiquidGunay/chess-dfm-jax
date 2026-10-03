"""Evaluate retained no-history Hero checkpoints on HH-002's common pool."""
import argparse
import json
from pathlib import Path

import torch

from research import train_torch as train
from research.hero_history_batches import HistoryTrajectoryBatches
from research.hero_history_data import sha256, verify_dataset
from research.input_contract import torch_hero_plane_history_mode, CURRENT_ONLY


class CurrentOnlyBatches:
    def __init__(self, original):
        self.original, self.batch_size = original, original.batch_size

    def batch_at(self, step):
        result = dict(self.original.batch_at(step))
        for key in ("current_planes", "future_planes"):
            result[key] = result[key].copy()
            result[key][...,12:104,:,:] = 0
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight",type=Path,required=True)
    parser.add_argument("--raw",type=Path,required=True)
    parser.add_argument("--baseline",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    receipt = json.loads(args.preflight.read_text())
    root = Path(receipt["dataset_root"])
    if sha256(root/"manifest.json") != receipt["manifest_sha256"]:
        raise ValueError("Dataset manifest drift")
    verify_dataset(root)
    config_record = json.loads((args.baseline/"run_config.json").read_text())
    if torch_hero_plane_history_mode(config_record) != CURRENT_ONLY:
        raise ValueError("Baseline must retain its historical no-history contract")
    config = train.Config(**config_record["config"])
    batches = CurrentOnlyBatches(HistoryTrajectoryBatches(root,split="validation",batch_size=64,horizon=8,
        manifest_sha256=receipt["manifest_sha256"],canonicalize=train.canonicalize_trajectory_batch))
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    model, _ = train.load_raw_bt4_hero_model(device=device,raw_bt4_path=args.raw,config=config)
    train._apply_compile_regions(model,"fresh")
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open("x") as stream:
        for checkpoint in receipt["baseline_checkpoints"]:
            manifest = train.load_checkpoint_model_for_evaluation(checkpoint_dir=Path(checkpoint["path"]),model=model)
            if manifest["state"]["sha256"] != checkpoint["state_sha256"]:
                raise ValueError("Baseline checkpoint identity drift")
            count = 127 if checkpoint["update"] == 554 else 16
            metrics, seconds = train._evaluate_validation_pool(model,batches,count=count,seed=2026072501,device=device)
            record = {"update":checkpoint["update"],"examples":checkpoint["update"]*1024,
                      "evaluation_examples":count*64,"metrics":metrics,"evaluation_seconds":seconds,
                      "state_sha256":checkpoint["state_sha256"],"history_mode":CURRENT_ONLY,
                      "dataset_manifest_sha256":receipt["manifest_sha256"],
                      "scope":"same validation rows and RNG; historical non-matched training data"}
            stream.write(json.dumps(record,sort_keys=True)+"\n");stream.flush()
            print(json.dumps({"baseline_validation":record}),flush=True)


if __name__ == "__main__":
    main()
