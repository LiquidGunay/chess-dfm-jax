"""HH-002: a frozen, explicitly contracted full-history Hero dataset."""
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

from research.hero_history_data import (
    SCHEMA, TRAIN_MANIFEST_SHA256, training_order, sha256, load_npz,
    reconstruct_shard, write_npz_once, write_json_once,
)

SELECTION = "hh002-contracted-v1"
AUDIT_RELATIVE = "research/analysis/hh001_history_20261003/contracted_selection_audit.json"
INDICES_RELATIVE = "research/eval/hero_epoch_v1/position_indices.npz"
TRAIN_EXAMPLES = 553 * 1024
VAL_EXAMPLES = 127 * 64


def selection_jobs(source, output, audit, validation):
    exclusions = {(r["split"], r["file"]): r for r in audit["records"]}
    jobs, tails = [], {}
    for split, indices, limit in (("train", training_order(), TRAIN_EXAMPLES),
                                  ("validation", np.unique(validation//1024), VAL_EXAMPLES)):
        available = []
        original_split = "val" if split == "validation" else "train"
        for slot, index in enumerate(indices):
            name = f"chunk_{index:06d}.npz"
            rows = list(range(1024)) if split == "train" else (validation[validation//1024 == index] % 1024).tolist()
            dropped = set(exclusions.get((original_split, name), {}).get("dropped_rows", []))
            rows = [int(row) for row in rows if row not in dropped]
            available.append((slot, int(index), name, rows))
        expected = 566954 if split == "train" else 8184
        if sum(len(r[3]) for r in available) != expected:
            raise ValueError("Frozen contraction count drift")
        remaining, tail = limit, []
        for slot, index, name, rows in available:
            kept, held = rows[:remaining], rows[remaining:]
            remaining -= len(kept)
            if held:
                tail.append({"source": name, "rows": held})
            if kept:
                output_name = f"batch_{slot:06d}.npz" if split == "train" else name
                jobs.append((Path(source)/original_split/name, Path(output)/split/output_name, kept))
        if remaining:
            raise ValueError("Insufficient retained data")
        tails[split] = tail
    return jobs, tails


def _build_subset_one(job):
    source, output, rows, cache_root, expected_original_sha = job
    if expected_original_sha and sha256(source) != expected_original_sha:
        raise ValueError("Original affected shard differs from contraction audit")
    cached = Path(cache_root)/output.parent.name/output.name
    sidecar = Path(str(cached)+".audit.json")
    record = json.loads(sidecar.read_text()) if sidecar.is_file() else None
    if record is not None and record["rows"] == rows:
        if record["source"] != source.name or sha256(cached) != record["sha256"]:
            raise ValueError("Prior reconstruction content drift")
        for item in record["inputs"]:
            p = source.with_name(item["name"])
            if sha256(p) != item["sha256"]:
                raise ValueError("Prior reconstruction source drift")
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists():
            raise FileExistsError(output)
        shutil.copyfile(cached, output)
    else:
        payload, record = reconstruct_shard(source, rows=rows)
        write_npz_once(output, payload)
        record.update(file=output.name, sha256=sha256(output), size_bytes=output.stat().st_size)
    record = {**record, "original_split": source.parent.name}
    write_json_once(str(output)+".audit.json", record)
    return output.parent.name, output.name, record


def build_subset(source_root, output_root, cache_root, audit_path, indices_path, *, workers=4):
    source_root, output_root = Path(source_root), Path(output_root)
    if output_root.exists():
        raise FileExistsError("Dataset output is immutable")
    for split, count, expected in (("train", 27679, TRAIN_MANIFEST_SHA256),
                                  ("val", 1524, "c02ecfd114fa1dbf59e0d7189cc32b0aedd5a468c6e24145b131e0e58745b281")):
        files = sorted((source_root/split).glob("*.npz"))
        inventory = "\n".join(f"{p.name}\t{p.stat().st_size}" for p in files)
        if len(files) != count or hashlib.sha256(inventory.encode()).hexdigest() != expected:
            raise ValueError("Original corpus inventory drift")
    if sha256(indices_path) != "c20102b49358dda5c9d6828e703cf133b196b3bf3b018cb143290aa8c1dc3e96":
        raise ValueError("Original validation selection drift")
    audit = json.loads(Path(audit_path).read_text())
    with np.load(indices_path, allow_pickle=False) as data:
        validation = np.sort(data["fast_val_global_index"])
    jobs, tails = selection_jobs(source_root, output_root, audit, validation)
    source_hashes = {(r["split"], r["file"]): r["source_sha256"] for r in audit["records"]}
    records = {"train": {}, "validation": {}}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_build_subset_one, (*job, cache_root,
                    source_hashes.get((job[0].parent.name, job[0].name)))) for job in jobs]
        for index, future in enumerate(as_completed(futures), 1):
            split, name, record = future.result()
            records[split][name] = record
            if index % 100 == 0:
                print(json.dumps({"subset_staging": index, "total": len(jobs)}), flush=True)
    manifest = {
        "schema_version": SCHEMA, "passed": True, "selection_contract": SELECTION,
        "history_mode": "lc0_full_history_as_preprocessed",
        "original_training_inventory_sha256": TRAIN_MANIFEST_SHA256,
        "original_training_shards_in_order": training_order(),
        "original_validation_indices_sha256": sha256(indices_path),
        "selection_audit_sha256": sha256(audit_path), "batching_tail_exclusions": tails,
        "train_examples": TRAIN_EXAMPLES, "validation_examples": VAL_EXAMPLES,
        "missing_history_exclusions": {"train":342, "validation":8},
        "batching_tail_exclusion_counts": {"train":682, "validation":56},
        "changed_plane_slice": [12,104],
    }
    for split in records:
        manifest[split] = [records[split][job[1].name] for job in jobs if job[1].parent.name == split]
    manifest["validation_global_indices"] = [int(r["source"][6:12])*1024+row
        for r in manifest["validation"] for row in r["rows"]]
    write_json_once(output_root/"selection_audit.json", audit)
    write_npz_once(output_root/"original_indices.npz", {"fast_val_global_index":validation})
    manifest["original_indices_file_sha256"] = sha256(output_root/"original_indices.npz")
    verify_subset(output_root, manifest)
    write_json_once(output_root/"manifest.json", manifest)
    return manifest


def verify_subset(root, manifest):
    root = Path(root)
    if (manifest.get("schema_version") != SCHEMA or manifest.get("passed") is not True
            or manifest.get("selection_contract") != SELECTION
            or manifest.get("history_mode") != "lc0_full_history_as_preprocessed"
            or manifest.get("original_training_inventory_sha256") != TRAIN_MANIFEST_SHA256
            or manifest.get("original_training_shards_in_order") != training_order()):
        raise ValueError("Invalid contracted history contract")
    if sha256(root/"selection_audit.json") != manifest["selection_audit_sha256"]:
        raise ValueError("Contraction audit checksum drift")
    audit = json.loads((root/"selection_audit.json").read_text())
    if sha256(root/"original_indices.npz") != manifest["original_indices_file_sha256"]:
        raise ValueError("Original validation indices checksum drift")
    with np.load(root/"original_indices.npz", allow_pickle=False) as data:
        indices = data["fast_val_global_index"]
    if len(indices) != 8192 or np.any(np.diff(indices) <= 0):
        raise ValueError("Original validation indices drift")
    jobs, tails = selection_jobs(Path("original"), root, audit, indices)
    if tails != manifest["batching_tail_exclusions"]:
        raise ValueError("Batching tail selection drift")
    for split, expected in (("train", TRAIN_EXAMPLES), ("validation", VAL_EXAMPLES)):
        selected = [job for job in jobs if job[1].parent.name == split]
        records = manifest[split]
        if (len(selected) != len(records) or manifest[f"{split}_examples"] != expected
                or sum(r["examples"] for r in records) != expected):
            raise ValueError("Contracted row count drift")
        for (source, output, rows), record in zip(selected, records, strict=True):
            if (record["file"] != output.name or record["source"] != source.name
                    or record["rows"] != rows or record["examples"] != len(rows)):
                raise ValueError("Contracted row identity/order drift")
            if output.stat().st_size != record["size_bytes"] or sha256(output) != record["sha256"]:
                raise ValueError("Contracted dataset content drift")
    return manifest
