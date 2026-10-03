"""Read-only progress for the frozen six-match VP-001 tournament."""
import argparse
import itertools
import json
from pathlib import Path

import modal
from research.infra import modal_arena_app as arena


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    preflight = json.loads((args.directory / "eval-preflight-result.json").read_text())
    models = preflight["models"]
    source = arena._arena_source_tree_sha256(Path.cwd())
    volume = modal.Volume.from_name("chess-dfm-training-results", environment_name="chess-dfm-research")
    for left, right in itertools.combinations(("native", "jepa_residual", "hero1", "raw_bt4"), 2):
        label = arena._arena_identity(candidate_label=models[left]["label"],
            candidate_state_sha256=models[left]["state_sha256"], opponent_label=models[right]["label"],
            opponent_state_sha256=models[right]["state_sha256"], candidate_policy_mode="dfm",
            opponent_policy_mode="policy_only" if right == "raw_bt4" else "dfm",
            pair_count=128, candidate_refinement_passes=1, opponent_refinement_passes=1,
            opening_start_index=1664, arena_source_tree_sha256=source)
        try:
            state = json.loads(b"".join(volume.read_file("arenas/" + label + "/state.json")))
        except FileNotFoundError:
            print(json.dumps({"left": left, "right": right, "status": "not_started", "arena_label": label}), flush=True)
            continue
        aggregate = state["aggregate"]
        record = {"left": left, "right": right, "status": state["status"],
            "completed_pairs": aggregate["pair_count"], "arena_label": label}
        if state["status"] == "complete":
            model_id = state["contract"]["models"]["candidate"]["model_id"]
            record.update(left_wdl={k: aggregate["models"][model_id][k] for k in ("wins", "draws", "losses")},
                score=aggregate["pentanomial"]["score"], faults=aggregate["fault_counts"],
                cap_draws=aggregate["cap_draw_count"])
        print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
