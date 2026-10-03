"""Content-bound batches for a history-reconstructed original Hero selection."""
from bisect import bisect_right
from collections import OrderedDict
import json
from pathlib import Path

import numpy as np

from chess_dfm_jax.data.trajectory_v3 import trajectory_v3_to_batch
from research.hero_history_data import SCHEMA, sha256


class HistoryTrajectoryBatches:
    def __init__(self, root, *, split, batch_size, horizon, manifest_sha256, canonicalize):
        self.root = Path(root)
        if split not in {"train", "validation"}:
            raise ValueError("Unknown history data split")
        manifest_path = self.root/"manifest.json"
        if len(manifest_sha256) != 64 or sha256(manifest_path) != manifest_sha256:
            raise ValueError("Full-history manifest checksum mismatch")
        self.manifest = json.loads(manifest_path.read_text())
        if self.manifest.get("schema_version") != SCHEMA or not self.manifest.get("passed"):
            raise ValueError("History dataset has no successful reconstruction contract")
        self.split, self.batch_size, self.horizon = split, int(batch_size), int(horizon)
        self.manifest_sha256 = manifest_sha256
        self.records = self.manifest[split]
        self.offsets = np.concatenate(([0], np.cumsum([r["examples"] for r in self.records])))
        if self.offsets[-1] != self.manifest[f"{split}_examples"] or self.offsets[-1] % self.batch_size:
            raise ValueError("History dataset row count is not a complete batch multiple")
        if horizon != 8:
            raise ValueError("History reconstruction is bound to the Hero eight-ply horizon")
        self.available_steps = int(self.offsets[-1])//self.batch_size
        # Preserve the original epoch's meaning, though only its short prefix is
        # materialized. Never cycle this short selection and call it an epoch.
        self.steps_per_epoch = 27679 if split == "train" else self.available_steps
        self.canonicalize = canonicalize
        self.cache = OrderedDict()

    def _shard(self, index):
        if index in self.cache:
            self.cache.move_to_end(index)
            return self.cache[index]
        record = self.records[index]
        name = record["file"]
        if Path(name).name != name:
            raise ValueError("Unsafe history dataset member")
        path = self.root/self.split/name
        if path.stat().st_size != record["size_bytes"] or sha256(path) != record["sha256"]:
            raise ValueError("Reconstructed history shard checksum mismatch")
        with np.load(path, allow_pickle=False) as payload:
            if int(payload["batch_size"]) != record["examples"]:
                raise ValueError("Reconstructed shard row count drift")
            decoded = trajectory_v3_to_batch(payload, view="joint_latent_sasa", horizon=8,
                                             include_metadata=True)
        decoded = self.canonicalize(decoded)
        self.cache[index] = decoded
        while len(self.cache) > 2:
            self.cache.popitem(last=False)
        return decoded

    def batch_at(self, step):
        if not 0 <= step < self.available_steps:
            raise IndexError("Requested step is outside the frozen original-data selection")
        start, stop = step*self.batch_size, (step+1)*self.batch_size
        pieces = []
        while start < stop:
            index = bisect_right(self.offsets, start)-1
            end = min(stop, int(self.offsets[index+1]))
            lo, hi = start-int(self.offsets[index]), end-int(self.offsets[index])
            pieces.append({key: value[lo:hi] for key,value in self._shard(index).items()})
            start = end
        if len(pieces) == 1:
            return pieces[0]
        return {key: np.concatenate([piece[key] for piece in pieces]) for key in pieces[0]}

    def provenance(self):
        return {"schema_version": SCHEMA, "split": self.split, "split_dir": str(self.root/self.split),
                "manifest_sha256": self.manifest_sha256, "batch_size": self.batch_size,
                "steps_per_epoch": self.steps_per_epoch, "available_steps": self.available_steps,
                "examples": int(self.offsets[-1]), "seed": 0,
                "selection_contract": self.manifest.get("selection_contract", "hh001-exact"),
                "batch_schedule": ("contracted_original_order_fixed_batches" if self.manifest.get("selection_contract") else
                                   "original_Hero_seed0_permutation_prefix" if self.split == "train" else "original_fast_pool_ascending_global_indices"),
                "original_training_inventory_sha256": self.manifest["original_training_inventory_sha256"]}
