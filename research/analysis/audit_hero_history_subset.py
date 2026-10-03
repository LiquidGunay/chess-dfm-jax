"""Audit an explicitly contracted Hero selection, never guess omitted moves.

The exact HH-001 builder remains fail-closed. This separate diagnostic removes
only requested rows whose current/future encodes cross an unresolved move. It
re-runs the original legal-replay, label and reference-parity checks on every
retained row. No GPU, original-data mutation, or training manifest is produced.
"""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from research.hero_history_data import (
    UnresolvedHistory, load_npz, reconstruct_shard, sha256, write_json_once,
)


def verified_subset(path, *, rows=None):
    source = load_npz(path)
    selected = np.arange(int(source["batch_size"])) if rows is None else np.asarray(rows, dtype=np.int64)
    requested = selected.tolist()
    exclusions = []
    while len(selected):
        try:
            rebuilt, record = reconstruct_shard(path, rows=selected)
            break
        except UnresolvedHistory as exc:
            affected = set(exc.affected_rows)
            if not affected or not affected.issubset(set(selected.tolist())):
                raise ValueError("Invalid history exclusion; refusing to continue") from exc
            exclusions.append({"rows": sorted(affected), "missing_from_ply": exc.missing_ply,
                               "reason": str(exc)})
            selected = selected[~np.isin(selected, list(affected))]
    else:
        rebuilt, record = None, {"examples": 0, "rows": [], "reference_parity_positions": 0}
    dropped = sorted(set(requested)-set(selected.tolist()))
    record.update(source_sha256=sha256(path), requested_rows=requested,
                  dropped_rows=dropped, exclusions=exclusions,
                  selection_rule="retain_only_complete_verified_prefix_through_last_future_target")
    record["dropped_ply"] = source["ply"][dropped].tolist()
    record["dropped_outcomes"] = dict(Counter(map(str, source["result"][dropped])))
    return rebuilt, record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--failures", type=Path, required=True)
    parser.add_argument("--indices", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with np.load(args.indices, allow_pickle=False) as data:
        validation = np.sort(data["fast_val_global_index"])
    records = []
    for failure in json.loads(args.failures.read_text()):
        split = "train" if failure["split"] == "train" else "val"
        path = args.fixtures/split/Path(failure["source"]).name
        index = int(path.stem.split("_")[-1])
        rows = None if split == "train" else validation[validation//1024 == index] % 1024
        _, audit = verified_subset(path, rows=rows)
        record = {"split": split, "file": path.name, **audit}
        records.append(record)
        print(json.dumps({"file": str(path), "requested": len(audit["requested_rows"]),
                          "kept": audit["examples"], "dropped": len(audit["dropped_rows"])}), flush=True)
    totals = {}
    for split, original in (("train", 567296), ("val", 8192)):
        dropped = sum(len(r["dropped_rows"]) for r in records if r["split"] == split)
        totals[split] = {"original_examples": original, "dropped_examples": dropped,
                         "retained_examples": original-dropped, "dropped_fraction": dropped/original}
    report = {"schema_version": "hero-history-contraction-audit-v1", "passed": True,
              "scope": "All 18 shards rejected by HH-001; other shards retain their prior verification",
              "failure_list_sha256": sha256(args.failures), "indices_sha256": sha256(args.indices),
              "totals": totals, "records": records,
              "caveat": "Not missing at random: omitted promotion spans affect late-game positions. This is not an exact data-matched ablation."}
    write_json_once(args.output, report)
    print(json.dumps(totals, indent=2))


if __name__ == "__main__":
    main()
