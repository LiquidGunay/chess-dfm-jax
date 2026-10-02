"""Recompute the frozen EV-004 metrics from immutable arena and validation evidence."""

import hashlib
import json
from dataclasses import asdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DIRECTORY = ROOT / "research/analysis/ev004_terminal_hero_20261002"
PRIOR = ROOT / "research/analysis/ev003_history_corrected_20261002"


def read(path):
    return json.loads(path.read_text())


def main():
    from research.train_torch import Config

    protocol = read(DIRECTORY / "preregister.json")
    rows = []
    for edge in protocol["development_edges"]:
        path = DIRECTORY / "results" / edge["id"] / "state.json"
        state = read(path)
        summary = read(path.with_name("summary.json"))
        if hashlib.sha256(path.read_bytes()).hexdigest() != summary["state_sha256"]:
            raise ValueError("Arena evidence digest mismatch")
        for side in ("candidate", "opponent"):
            if summary[f"{side}_state_sha256"] != protocol[edge[side]]["state_sha256"]:
                raise ValueError("Arena model identity differs from frozen protocol")
            if summary[f"{side}_policy_mode"] != edge[f"{side}_mode"]:
                raise ValueError("Arena policy differs from frozen protocol")
            if summary[f"{side}_refinement_passes"] != edge[f"{side}_passes"]:
                raise ValueError("Arena pass count differs from frozen protocol")
        if summary["opening_start_index"] != protocol["openings"]["development"]["start"]:
            raise ValueError("Arena opening allocation differs from frozen protocol")
        aggregate = state["aggregate"]
        candidate, = (v for k, v in aggregate["models"].items() if k.startswith("candidate-"))
        wins, draws, losses = (candidate[k] for k in ("wins", "draws", "losses"))
        games = wins + draws + losses
        score = (wins + 0.5 * draws) / games
        if (state["status"] != "complete" or games != 256
                or abs(score - summary["pair_score"]) > 1e-12
                or abs(score - sum(aggregate["pair_scores"]) / games) > 1e-12):
            raise ValueError("Arena totals fail independent reconciliation")
        rows.append({**edge, "wins": wins, "draws": draws, "losses": losses,
                     "games": games, "score": score,
                     "interval": aggregate["pair_aware_logistic_interval"],
                     "cap_draws": aggregate["cap_draw_count"],
                     "faults": aggregate["fault_counts"],
                     "incomplete_coverage_positions": sum(v["coverage"]["incomplete_coverage_positions"] for v in aggregate["models"].values()),
                     "candidate_seconds_per_position": candidate["coverage"]["call_seconds"] / candidate["positions_evaluated"],
                     "state_sha256": summary["state_sha256"]})
    validation = read(DIRECTORY / "common_validation_result.json")
    previous = read(PRIOR / "common_validation_result.json")
    if validation["inputs_sha256"] != previous["inputs_sha256"]:
        raise ValueError("Validation positions differ")
    controls = [r for r in validation["rows"] if r["model"] == "hero1"]
    prior_controls = [r for r in previous["rows"] if r["model"] == "hero1"]
    if controls != prior_controls:
        raise ValueError("Repeated Hero1 validation did not reproduce exactly")
    terminal_config = read(DIRECTORY / "terminal_run_config.json")
    short_config = read(ROOT / "research/analysis/hero_v2_phase1/hero1_exact/run_config.json")
    recipe_equal = asdict(Config(**terminal_config["config"])) == asdict(Config(**short_config["config"]))
    data_order_equal = all(terminal_config["data"][k] == short_config["data"][k]
                           for k in ("file_manifest_sha256", "seed", "batch_size", "batch_schedule"))
    if not recipe_equal or not data_order_equal:
        raise ValueError("Claimed matched legacy recipe or data order differs")
    movement = {}
    for name, relative in (
        ("hero1", "research/analysis/hero_v2_phase1/hero1_exact/parameter_diff.json"),
        ("terminal_hero", "research/analysis/raw_hero_parameter_diff_20260807.json"),
    ):
        source = read(ROOT / relative)
        movement[name] = {"source": relative, "state_sha256": source["models"]["hero_state_sha256"],
                          "groups": {k: source["group_metrics"][k] for k in ("all", "trunk", "policy_head")}}
    report = {
        "schema_version": "chess-dfm-terminal-hero-comparison-v1",
        "experiment_id": "EV-004", "arena_rows": rows,
        "validation_positions": validation["positions"],
        "validation_inputs_sha256": validation["inputs_sha256"],
        "hero1_control_reproduced_exactly": True,
        "legacy_normalized_config_equal": recipe_equal,
        "legacy_data_order_equal": data_order_equal,
        "raw_bt4_parameter_movement": movement,
        "validation_rows": validation["rows"] + [r for r in previous["rows"] if r["model"] == "one_epoch"],
        "limitations": ["All arenas use exposed development openings, not confirmation.",
                        "The raw BT4 arena anchor uses current-only inputs.",
                        "Legacy terminal and Hero1 share the recipe and data order but are separate executions.",
                        "Played-move CE is not game strength or the composite training loss."],
    }
    output = DIRECTORY / "development_summary.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(output), "arena_rows": rows}, indent=2))


if __name__ == "__main__":
    main()
