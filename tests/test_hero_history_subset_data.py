import json
from pathlib import Path

import numpy as np

from research.hero_history_subset_data import selection_jobs, AUDIT_RELATIVE, INDICES_RELATIVE


def test_frozen_selection_counts_and_row_order():
    root = Path(__file__).resolve().parents[1]
    audit = json.loads((root/AUDIT_RELATIVE).read_text())
    with np.load(root/INDICES_RELATIVE) as data:
        indices = np.sort(data["fast_val_global_index"])
    jobs, tails = selection_jobs("original", "output", audit, indices)
    counts = {split:sum(len(rows) for _, out, rows in jobs if out.parent.name == split)
              for split in ("train","validation")}
    assert counts == {"train":566272,"validation":8128}
    assert {s:sum(len(r["rows"]) for r in rows) for s,rows in tails.items()} == {"train":682,"validation":56}
    exclusions = {(r["split"],r["file"]):set(r["dropped_rows"]) for r in audit["records"]}
    for source, _, rows in jobs:
        assert rows == sorted(set(rows))
        assert not set(rows) & exclusions.get((source.parent.name,source.name),set())
    assert jobs[0][0].name == "chunk_024436.npz"
    assert jobs[0][2] == list(range(1024))
