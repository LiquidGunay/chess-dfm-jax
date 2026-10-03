"""Compare hash-verified HH-002 and historical Hero1 parameter-movement audits."""
import argparse
import hashlib
import json
import math
from pathlib import Path


def compare_group(old, new):
    if old["parameter_count"] != new["parameter_count"] or not math.isclose(
            old["raw_l2"], new["raw_l2"], rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError("Movement groups do not have the same initial parameter norm/count")
    ratio = new["delta_l2"] / old["delta_l2"] if old["delta_l2"] else None
    return {
        "parameter_count": old["parameter_count"],
        "initial_l2": old["raw_l2"],
        "hero1_delta_l2": old["delta_l2"],
        "history_delta_l2": new["delta_l2"],
        "hero1_relative_delta_l2": old["relative_delta_l2"],
        "history_relative_delta_l2": new["relative_delta_l2"],
        "history_to_hero1_movement_ratio": ratio,
        "movement_change_percent": 100 * (ratio - 1) if ratio is not None else None,
        "hero1_stored_changed_fraction": 1 - old["unchanged_fraction"],
        "history_stored_changed_fraction": 1 - new["unchanged_fraction"],
    }


def summarize(directory):
    directory = Path(directory)
    completion = json.loads((directory / "completion.json").read_text())
    if not completion["passed"] or completion["gpu_active"]:
        raise ValueError("Expected completed CPU audit")
    reports = {}
    for label, entry in completion["reports"].items():
        raw = (directory / entry["path"]).read_bytes()
        if hashlib.sha256(raw).hexdigest() != entry["sha256"]:
            raise ValueError("Parameter audit checksum mismatch")
        report = json.loads(raw)
        if report["training_identity"]["state_sha256"] != completion["input_preflight"]["models"][label]["state_sha256"]:
            raise ValueError("Checkpoint identity mismatch")
        reports[report["training_identity"]["plane_history_mode"]] = report
    old = reports["current_only_as_preprocessed"]
    new = reports["lc0_full_history_as_preprocessed"]
    compared = {}
    for output_key, get_groups in (
        ("bt4_groups", lambda r: r["group_metrics"]),
        ("added_groups", lambda r: r["added_heads_vs_recorded_fresh_initialization"]["groups"]),
    ):
        a, b = get_groups(old), get_groups(new)
        if a.keys() != b.keys():
            raise ValueError("Group identities differ")
        compared[output_key] = {key: compare_group(a[key], b[key]) for key in a}
    # These disjoint groups also explain where the change in squared trunk
    # movement comes from; do not sum overlapping attention/layer summaries.
    for report in (old, new):
        groups = report["group_metrics"]
        if not math.isclose(groups["trunk"]["delta_l2"] ** 2,
                            sum(groups[g]["delta_l2"] ** 2 for g in ("embedding", "layers")),
                            rel_tol=1e-10):
            raise ValueError("Trunk does not decompose into embedding and transformer layers")
    return {
        "schema_version": "hh002-weight-movement-summary-v1",
        "hero1": old["training_identity"], "full_history": new["training_identity"],
        "source_reports": completion["reports"],
        "definition": "relative_delta_l2 = ||stored trained weights - matched initial weights||_2 / ||matched initial weights||_2",
        **compared,
        "limitations": [
            "Pretrained encoder weights are compared at their stored BF16 precision, not optimizer-master precision; fresh modules are reported separately.",
            "Shared raw BT4 and recorded fresh initializations were verified; zero-initialized tensors have undefined relative movement, represented by null.",
            "Single seed, 553 versus 554 updates and slightly different retained training examples; not a pure causal history effect.",
            "Greater parameter movement does not establish better representations or stronger play.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.directory)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({k: result["bt4_groups"][k] for k in ("trunk", "embedding", "layers", "policy_head")}))


if __name__ == "__main__":
    main()
