"""Freeze and reconcile EV-005; no training and no automatic GPU retries."""
from __future__ import annotations

import argparse
from datetime import UTC, datetime
from decimal import Decimal
import hashlib
import json
import math
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT / "research/analysis/ev005_full_history_bt4_20261003"
HERO = "hero-training-phase1-hero1_exact-l40s-fae7b0ccef5ca3668ed5"
HERO_SHA = "e518083ee5476e7ff15e7c34caf9c0505f8fb7d330ea5a6882e4e9e5c1ef1912"
CURRENT = "current_only_as_preprocessed"
FULL = "lc0_full_history_as_preprocessed"


def write_once(path, value):
    raw = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != raw:
            raise RuntimeError(f"Immutable evidence already exists: {path}")
        return
    with path.open("xb") as stream:
        stream.write(raw)


def billing():
    from research.infra.modal_train_app import _modal_launch_spend
    # Keep the last known settled total as a conservative floor if the live
    # provider meter revises downward. No old completed-job reservation is active.
    return max(_modal_launch_spend(), Decimal("2.66285855"))


def preflight():
    from research.infra.modal_arena_app import _arena_job_spec
    from research.interpretability.remote_cost import ModalJobSpec, enforce_modal_budget, estimate_modal_job
    if (DEST / "input_preflight.json").exists():
        raise RuntimeError("Preflight already recorded; do not submit a duplicate")
    spend = billing()
    spec = ModalJobSpec(name="arena-input-preflight", timeout_seconds=300,
                        startup_timeout_seconds=120, retries=0,
                        cpu_cores=Decimal("2"), memory_gib=Decimal("4"), gpu=None)
    decision = enforce_modal_budget(estimate_modal_job(spec), recorded_month_spend=spend,
                                    per_job_cap=Decimal("0.02"))
    write_once(DEST / "preflight_reservation.json", {
        "created_utc": datetime.now(UTC).isoformat(), "cost_guard": decision,
        "scope": "CPU-only complete checkpoint, raw weights and opening input verification",
    })
    command = ["/root/.local/bin/modal", "run", "--env", "chess-dfm-research",
               "research/infra/modal_arena_app.py", "--candidate-label", HERO,
               "--opponent-label", "raw-bt4", "--preflight-only", "--recorded-spend", str(spend)]
    result = subprocess.run(command, cwd=ROOT, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, timeout=600)
    (DEST / "preflight.log").write_text(result.stdout)
    print(result.stdout, flush=True)
    if result.returncode:
        raise RuntimeError("CPU input preflight failed; no GPU launch permitted")
    records = [json.loads(line) for line in result.stdout.splitlines()
               if line.startswith('{"input_preflight":')]
    if len(records) != 1 or not records[0]["input_preflight"]["passed"]:
        raise RuntimeError("Missing complete CPU preflight result")
    write_once(DEST / "input_preflight.json", records[0]["input_preflight"])


def freeze():
    from research.infra.modal_arena_app import _arena_source_tree_sha256, RAW_BT4_SHA256
    verified = json.loads((DEST / "input_preflight.json").read_text())
    source = _arena_source_tree_sha256(ROOT)
    if verified["arena_source_tree_sha256"] != source or verified["models"][HERO]["state_sha256"] != HERO_SHA:
        raise RuntimeError("Preflight source/checkpoint drift")
    # Audit retained arena contracts, not just recent evaluation summaries.
    exposure = []
    for path in sorted((ROOT / "research/analysis").rglob("state.json")):
        if DEST in path.parents:
            continue
        record = json.loads(path.read_text())
        run = record.get("contract", {}).get("run", {})
        if "pair_count" in run:
            start, count = int(run.get("opening_start_index", 0)), int(run["pair_count"])
            exposure.append({"path": str(path.relative_to(ROOT)), "start": start, "pairs": count,
                             "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    for record in exposure:
        if record["start"] < 2048 and record["start"] + record["pairs"] > 1904:
            raise RuntimeError("Proposed opening slice overlaps retained arena evidence")
    write_once(DEST / "opening_exposure_audit.json", {
        "retained_contracts": exposure,
        "limitation": "No overlap in retained state contracts; historical unretained exposure cannot be excluded. Development-only, not confirmation.",
        "smoke": [1904, 1920], "development": [1920, 2048],
    })
    def edge(name, opponent, mode, role):
        return dict(id=name, candidate="hero1", opponent=opponent, candidate_mode=mode,
                    opponent_mode="policy_only", candidate_passes=1, opponent_passes=1, role=role)
    prior = json.loads((ROOT / "research/analysis/ev003_history_corrected_20261002/preregister.json").read_text())
    protocol = {
        "experiment_id": "EV-005", "created_utc": datetime.now(UTC).isoformat(),
        "question": "Does Hero1 beat raw BT4 with its full historical inputs, and how much did missing history affect the anchor?",
        "hero1": {"label": HERO, "state_sha256": HERO_SHA, "plane_history_mode": CURRENT},
        "raw_full": {"label": "raw-bt4", "state_sha256": RAW_BT4_SHA256, "plane_history_mode": FULL},
        "raw_current": {"label": "raw-bt4", "state_sha256": RAW_BT4_SHA256, "plane_history_mode": CURRENT},
        "openings": {**{k: v for k, v in prior["openings"].items() if k not in {"smoke", "development"}},
                     "smoke": {"start": 1904, "pairs": 16}, "development": {"start": 1920, "pairs": 128}},
        "smoke_edges": [edge("smoke_p1_full", "raw_full", "dfm", "plumbing"),
                        edge("smoke_policy_full", "raw_full", "policy_only", "plumbing")],
        "development_edges": [edge("p1_full", "raw_full", "dfm", "primary"),
                              edge("policy_full", "raw_full", "policy_only", "secondary_native_policy"),
                              edge("p1_current", "raw_current", "dfm", "history_attribution_control")],
        "game_contract": prior["game_contract"],
        "analysis": {"unit": "complete color-reversed opening pair", "primary": "Hero1 p1 versus raw full-history BT4",
                     "secondary": "Hero1 native policy versus raw full-history BT4",
                     "attribution": "Change in Hero1 p1 pair score when only raw opponent history is restored, on the same openings",
                     "rule": "Report all three fixed-size comparisons and paired 95% intervals; no promotion, absolute Elo, or SOTA claim. No optional stopping.",
                     "scope": "Hero1 retains training-matched current-only inputs. Raw BT4 differs only in input history across controls. No search or new training.",
                     "parity_limit": "Uses previously audited Torch weight conversion and LC0-record plane parity, not a new native-LC0 forward-parity comparison."},
        "budget": {"launch_month_usage_dollars": str(billing() + Decimal("0.02")),
                   "declared_attempt_maximum_dollars": "0.92976000", "maximum_simultaneous_attempts": 3,
                   "smoke_plus_development_attempts": 5, "per_attempt_cap_dollars": "1.00",
                   "total_declared_maximum_dollars": "4.6688", "workspace_stop_dollars": "34.00",
                   "rule": "Live usage plus full reservations; zero retries. CPU preflight before GPU."},
        "tooling": {"arena_source_tree_sha256": source, "input_preflight_file": "input_preflight.json",
                    "input_preflight_sha256": hashlib.sha256((DEST / "input_preflight.json").read_bytes()).hexdigest()},
    }
    write_once(DEST / "preregister.json", protocol)
    print(json.dumps({"protocol": str(DEST / "preregister.json"), "declared_maximum_dollars": "4.6688"}))


def report():
    protocol = json.loads((DEST / "preregister.json").read_text())
    rows = []
    pair_scores = {}
    for edge in protocol["development_edges"]:
        path = DEST / "results" / edge["id"] / "state.json"
        summary = json.loads(path.with_name("summary.json").read_text())
        if hashlib.sha256(path.read_bytes()).hexdigest() != summary["state_sha256"]:
            raise RuntimeError("Result checksum drift")
        state = json.loads(path.read_text())
        if (summary["pair_count"] != protocol["openings"]["development"]["pairs"]
                or summary["opening_start_index"] != protocol["openings"]["development"]["start"]):
            raise RuntimeError("Result opening allocation drift")
        for side in ("candidate", "opponent"):
            identity = protocol[edge[side]]
            if (summary[f"{side}_state_sha256"] != identity["state_sha256"]
                    or summary[f"{side}_plane_history_mode"] != identity["plane_history_mode"]
                    or summary[f"{side}_policy_mode"] != edge[f"{side}_mode"]
                    or summary[f"{side}_refinement_passes"] != edge[f"{side}_passes"]):
                raise RuntimeError("Result input identity drift")
        models = state["aggregate"]["models"]
        candidate, = [v for k, v in models.items() if k.startswith("candidate-")]
        w, d, l = (candidate[key] for key in ("wins", "draws", "losses"))
        if state["status"] != "complete" or w + d + l != 256 or (w + .5*d)/256 != summary["pair_score"]:
            raise RuntimeError("Game total reconciliation failed")
        scores = state["aggregate"]["pair_scores"]
        if len(scores) != 128 or sum(scores) / 256 != summary["pair_score"]:
            raise RuntimeError("Complete-pair scores fail reconciliation")
        pair_scores[edge["id"]] = [score / 2 for score in scores]
        rows.append({**edge, "wins": w, "draws": d, "losses": l, "score": summary["pair_score"],
                     "interval": summary["pair_aware_logistic_interval"], "faults": summary["fault_counts"],
                     "cap_draw_rate": summary["cap_draw_rate"],
                     "incomplete_coverage_positions": sum(v["coverage"]["incomplete_coverage_positions"] for v in models.values()),
                     "seconds_per_candidate_position": candidate["coverage"]["call_seconds"] / candidate["positions_evaluated"],
                     "state_sha256": summary["state_sha256"]})
    changes = [a-b for a, b in zip(pair_scores["p1_full"], pair_scores["p1_current"], strict=True)]
    delta = sum(changes) / len(changes)
    radius = 2 * math.sqrt(math.log(40) / (2 * len(changes)))
    result = {"experiment_id": "EV-005", "rows": rows, "scope": protocol["analysis"],
              "hero1_score_change_after_restoring_raw_history": delta,
              "history_effect_interval": {"method": "hoeffding_paired_difference_range_minus1_plus1",
                                          "confidence": .95, "lower": max(-1., delta-radius),
                                          "upper": min(1., delta+radius), "pairs": len(changes)}}
    write_once(DEST / "development_summary.json", result)
    print(json.dumps(result, indent=2))


def closeout():
    from research.interpretability.remote_cost import budget_usage_from_summary
    receipt = json.loads((DEST / "runs.json").read_text())
    if len(receipt["runs"]) != 5 or any(r["status"] != "complete" for r in receipt["runs"].values()):
        raise RuntimeError("All five frozen runs must complete before closeout")
    def cli(*args):
        completed = subprocess.run(["/root/.local/bin/modal", *args], text=True,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        return json.loads(completed.stdout)
    usage = cli("billing", "summary", "--for", "this month", "--json")
    apps = cli("app", "list", "--env", "chess-dfm-research", "--json")
    our_ids = {run["app_id"] for run in receipt["runs"].values()}
    active = [app for app in apps if app["app_id"] in our_ids and app["state"] != "stopped"]
    if active:
        raise RuntimeError("Completed experiment still has active apps")
    write_once(DEST / "closeout_usage.json", {
        "collected_utc": datetime.now(UTC).isoformat(), "provider_summary": usage,
        "budget_usage_before_credits_excluding_waived_storage": str(budget_usage_from_summary(usage)),
        "apps": [app for app in apps if app["app_id"] in our_ids], "active_experiment_apps": 0,
        "note": "Provider meter can lag or revise. Launch ledger retained a conservative historical floor and full attempt reservations.",
    })
    results = json.loads((DEST / "development_summary.json").read_text())
    def verdict(row):
        ci = row["interval"]
        return "inferior_on_development" if ci["score_upper"] < .5 else (
            "superior_on_development" if ci["score_lower"] > .5 else "inconclusive")
    write_once(DEST / "decision.json", {
        "experiment_id": "EV-005", "status": "complete",
        "primary": verdict(results["rows"][0]), "native_policy": verdict(results["rows"][1]),
        "input_plane_parity_positions": 3225, "targeted_tests_passed": 92,
        "evaluation_source_commit": "50b37e9", "post_launch_training_guard_commit": "5df1516",
        "source_note": "All GPU jobs used the frozen source hash. Current-position training-audit hardening was committed only after all jobs were submitted; it changed no arena inputs or inference.",
        "legacy_jax_suite": "Not run: Torch-only environment lacks JAX; maintained Torch and data/gameplay tests passed.",
        "scope": "Fixed-size development comparisons, no absolute Elo or SOTA claim; no native-LC0 forward-parity test was added.",
        "future_training": "Full-history sequential inputs by default; legacy zero-history training requires explicit reproduction override. Existing weights and historical evidence unchanged.",
    })
    print(json.dumps({"status": "complete", "usage": usage, "active_experiment_apps": 0}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["preflight", "freeze", "report", "closeout"])
    {"preflight": preflight, "freeze": freeze, "report": report, "closeout": closeout}[parser.parse_args().action]()
