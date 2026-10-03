"""Independently reconcile VP-001 game records and per-position loss vectors.

Read-only with respect to source evidence; creates an immutable local summary.
Does not use the arena's aggregation or interval implementation.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import itertools
import json
import math
from pathlib import Path

import chess
import numpy as np


PLAYERS = ("native", "jepa_residual", "hero1", "raw_bt4")


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(a, b, message):
    require(math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-10), message)


def score_interval(points, pairs):
    score = points / (2 * pairs)
    radius = math.sqrt(math.log(40) / (2 * pairs))
    lower, upper = max(0, score - radius), min(1, score + radius)
    def elo(p):
        return 400 * math.log10(p / (1 - p)) if 0 < p < 1 else None
    return {"score": score, "score_lower": lower, "score_upper": upper,
            "elo": elo(score), "elo_lower": elo(lower), "elo_upper": elo(upper)}


def opening_board(history, cache):
    fens = tuple(history["fens"])
    if fens not in cache:
        require(fens[0] == chess.STARTING_FEN, "Opening does not begin at standard initial board")
        board = chess.Board()
        for wanted in fens[1:]:
            matches = []
            for move in list(board.legal_moves):
                board.push(move)
                if board.fen(en_passant="legal") == wanted:
                    matches.append(move)
                board.pop()
            require(len(matches) == 1, "Opening history is not a unique legal sequence")
            board.push(matches[0])
        cache[fens] = board
    return cache[fens].copy(stack=True)


def verify_match(directory, left, right, models, cache):
    root = directory / "matches" / (left + "-" + right)
    state = read(root / "state.json")
    receipt = read(directory / ("match-" + left + "-" + right + ".json"))
    require(sha(root / "state.json") == receipt["summary"]["state_sha256"], "State checksum mismatch")
    require(state["status"] == "complete", "Incomplete match")
    contract, aggregate = state["contract"], state["aggregate"]
    run = contract["run"]
    for key, expected in {"pair_count": 128, "opening_start_index": 1664,
            "candidate_refinement_passes": 1, "opponent_refinement_passes": 1,
            "additional_ply_cap": 256, "policy_batch_size_cap": 16}.items():
        require(run[key] == expected, "Tournament protocol drift: " + key)
    ids = {}
    for side, name in (("candidate", left), ("opponent", right)):
        descriptor = contract["models"][side]
        identity = descriptor.get("state", descriptor.get("bt4_checkpoint", {}))
        require(identity["sha256"] == models[name]["state_sha256"], "Wrong checkpoint")
        require(descriptor["plane_history_mode"] == models[name]["plane_history_mode"], "Wrong history")
        ids[descriptor["model_id"]] = name
    left_id = contract["models"]["candidate"]["model_id"]
    counts = {name: Counter(games=0, wins=0, draws=0, losses=0, points=0) for name in (left, right)}
    color_counts = {name: {color: Counter() for color in ("white", "black")} for name in (left, right)}
    games, histories, terminations, normal_terms, faults = [], {}, Counter(), Counter(), Counter()
    next_pair = 0
    for reference in state["blocks"]:
        path = root / "blocks" / Path(reference["path"]).name
        require(sha(path) == reference["sha256"], "Game block checksum mismatch")
        block = read(path)
        require(block["contract_sha256"] == state["contract_sha256"], "Wrong game contract")
        require(block["start_pair"] == next_pair, "Noncontiguous game blocks")
        require(block["pair_count"] == 16, "Unexpected block size")
        next_pair += block["pair_count"]
        games.extend(block["gameplay"]["games"])
        for history in block["gameplay"]["opening_histories"]:
            require(history["pair_id"] not in histories, "Duplicate opening pair")
            histories[history["pair_id"]] = history
    require(next_pair == 128 and len(games) == 256 and len(histories) == 128, "Incomplete game evidence")
    pairs = defaultdict(list)
    replay_coverage = {model_id: Counter() for model_id in ids}
    legal_moves_replayed = 0
    for game in games:
        board = opening_board(histories[game["pair_id"]], cache)
        require(board.fen(en_passant="legal") == game["fen"], "Game opening mismatch")
        require(len(game["moves_uci"]) == game["ply_count"], "Wrong ply count")
        for uci in game["moves_uci"]:
            move = chess.Move.from_uci(uci)
            legal = tuple(board.legal_moves)
            require(move in legal, "Illegal move in game record")
            model_id = game["white_model" if board.turn == chess.WHITE else "black_model"]
            replay_coverage[model_id].update(positions=1, legal_moves=len(legal))
            board.push(move)
            legal_moves_replayed += 1
        require(board.fen(en_passant="legal") == game["final_fen"], "Final board mismatch")
        termination = game["termination"]
        terminations[termination] += 1
        if termination == "normal":
            outcome = board.outcome(claim_draw=True)
            require(outcome is not None and outcome.result() == game["result"], "Incorrect game result")
            require(outcome.termination.name.lower() == game["normal_termination"], "Incorrect draw/mate reason")
            normal_terms[game["normal_termination"]] += 1
        elif termination == "ply_cap":
            require(game["ply_count"] == 256 and game["result"] == "1/2-1/2", "Incorrect cap draw")
        else:
            faults[game["fault_kind"]] += 1
        require({game["white_model"], game["black_model"]} == set(ids), "Wrong game players")
        white_points = {"1-0": 1.0, "0-1": 0.0, "1/2-1/2": 0.5}[game["result"]]
        for color, points in (("white", white_points), ("black", 1 - white_points)):
            name = ids[game[color + "_model"]]
            item = {"games": 1, "points": points, "wins": int(points == 1),
                    "draws": int(points == .5), "losses": int(points == 0)}
            counts[name].update(item)
            color_counts[name][color].update(item)
        left_points = white_points if game["white_model"] == left_id else 1 - white_points
        pairs[game["pair_id"]].append((game, left_points))
    pair_scores, opening_signature = [], []
    for pair_id, pair in sorted(pairs.items(), key=lambda item: item[1][0][0]["opening_index"]):
        require(len(pair) == 2, "Incomplete opening pair")
        a, b = sorted(pair, key=lambda item: item[0]["game_in_pair"])
        ga, gb = a[0], b[0]
        require((ga["game_in_pair"], gb["game_in_pair"]) == (0, 1), "Duplicate color game")
        require(ga["white_model"] == gb["black_model"] and ga["black_model"] == gb["white_model"], "Colors not reversed")
        require(ga["fen"] == gb["fen"] and ga["opening_index"] == gb["opening_index"], "Unmatched opening")
        opening_signature.append((ga["opening_index"], histories[pair_id]["fens"]))
        pair_scores.append(a[1] + b[1])
    require([x[0] for x in opening_signature] == list(range(1664, 1792)), "Unexpected opening range")
    require(pair_scores == aggregate["pair_scores"], "Recorded paired scores disagree")
    bins = [pair_scores.count(i / 2) for i in range(5)]
    require(bins == aggregate["pentanomial"]["counts"], "Pentanomial disagreement")
    for model_id, name in ids.items():
        for key, value in counts[name].items():
            close(value, aggregate["models"][model_id][key], "WDL/count disagreement")
        if not faults:
            recorded = aggregate["models"][model_id]["coverage"]
            for key, value in replay_coverage[model_id].items():
                require(value == recorded[key], "Position/legal-move count disagreement")
            require(recorded["legal_moves"] == recorded["representable_legal_actions"], "Incomplete legal coverage")
            require(recorded["incomplete_coverage_positions"] == 0, "Coverage failures")
    interval = score_interval(counts[left]["points"], 128)
    for key, value in interval.items():
        if value is not None:
            close(value, aggregate["pair_aware_logistic_interval"][key], "Interval disagreement")
    require(dict(terminations) == aggregate["termination_counts"], "Termination count disagreement")
    require(dict(faults) == aggregate["fault_counts"], "Fault count disagreement")
    require(terminations["ply_cap"] == aggregate["cap_draw_count"], "Cap count disagreement")
    return {"left": left, "right": right, "games": 256, "pairs": 128,
        "left_wdl": dict(counts[left]), "right_wdl": dict(counts[right]),
        "by_color": color_counts, "paired_score_95_interval": interval,
        "pentanomial_counts": bins, "normal_termination_counts": dict(normal_terms),
        "termination_counts": dict(terminations), "fault_counts": dict(faults),
        "legal_moves_replayed": legal_moves_replayed,
        "independently_recomputed_coverage": replay_coverage,
        "ply_count_mean": math.fsum(g["ply_count"] for g in games) / len(games),
        "ply_count_min": min(g["ply_count"] for g in games),
        "ply_count_max": max(g["ply_count"] for g in games),
        "recorded_runtime_and_coverage": aggregate["models"],
        "gameplay_wall_seconds": aggregate["gameplay_wall_seconds"]}, opening_signature


def verify_losses(directory, models):
    report = read(directory / "policy_losses.json")
    path = directory / "position_metrics.npz"
    require(sha(path) == report["position_metrics_sha256"], "Loss-vector checksum mismatch")
    expected = {(name, mode) for name in PLAYERS for mode in
        (["native_policy"] if name == "raw_bt4" else ["native_policy", "dfm_one_pass", "dfm_eight_pass"])}
    require(len(report["rows"]) == len(expected), "Unexpected loss row count")
    seen, rows = set(), []
    with np.load(path, allow_pickle=False) as vectors:
        for row in report["rows"]:
            name, mode = row["model"], row["mode"]
            require((name, mode) in expected - seen, "Duplicate/unexpected loss row")
            seen.add((name, mode))
            require(row["pool"] == "sequential" and row["positions"] == 8192, "Wrong policy validation pool")
            require(row["state_sha256"] == models[name]["state_sha256"], "Loss checkpoint mismatch")
            require(row["plane_history_mode"] == models[name]["plane_history_mode"], "Loss history mismatch")
            key = "sequential__" + name + "__" + mode
            ce, correct = vectors[key + "__ce"], vectors[key + "__correct"]
            require(ce.shape == correct.shape == (8192,), "Incorrect vector length")
            require(np.isfinite(ce).all() and (ce >= 0).all(), "Invalid CE")
            require(np.isin(correct, [0, 1]).all(), "Invalid accuracy vector")
            mean = math.fsum(float(x) for x in ce) / 8192
            accuracy = sum(int(x) for x in correct) / 8192
            close(mean, row["legal_conditional_ce"], "CE aggregation mismatch")
            close(accuracy, row["legal_top1_accuracy"], "Accuracy aggregation mismatch")
            rows.append({**row, "ce_p50": float(np.median(ce)), "ce_p95": float(np.quantile(ce, .95))})
        require(len(vectors.files) == 2 * len(expected), "Unexpected metric vectors")
    return rows


def verify_training(directory):
    preflight = read(directory / "preflight.json")
    reports, initial_hashes = {}, []
    for arm in PLAYERS[:2]:
        root = directory / arm
        report = read(root / "report.json")
        metrics = [json.loads(line) for line in (root / "metrics.jsonl").read_text().splitlines()]
        require(report["status"] == "complete" and report["updates"] == len(metrics) == 1000, "Incomplete training")
        for i, row in enumerate(metrics):
            require(row["update"] == i + 1 and row["examples"] == (i + 1) * 1024, "Wrong train counters")
            require(row["input_sha256"] == preflight["training_hashes"][str(i)], "Unmatched train input/RNG")
            require(row["optimizer_skipped_nonfinite"] == 0 and math.isfinite(row["loss"]), "Invalid train update")
        close(math.fsum(row["loss"] for row in metrics[-50:]) / 50,
              report["mean_last50"]["loss"], "Train mean mismatch")
        initial_hashes.append(report["common_initial_parameters_sha256"])
        validation = [json.loads(line) for line in (root / "validation_metrics.jsonl").read_text().splitlines()]
        require([row["update"] for row in validation] == [0, 250, 554, 1000], "Missing validation endpoint")
        reports[arm] = {"updates": 1000, "examples": 1024000, "wall_seconds": report["wall_seconds"],
            "train_last50": report["mean_last50"], "validation_curve": validation,
            "final_validation_8192": report["validation"], "model_sha256": report["checkpoint"]["state"]["sha256"]}
    require(len(set(initial_hashes)) == 1, "Unmatched common initialization")
    return reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    directory = args.directory / "collected"
    completion = read(directory / "evaluation_complete.json")
    require(completion["status"] == "complete" and len(completion["matches"]) == 6, "Evaluation incomplete")
    models = read(directory / "evaluation_preflight.json")["models"]
    cache, matches, signature = {}, [], None
    standings = {name: Counter() for name in PLAYERS}
    for left, right in itertools.combinations(PLAYERS, 2):
        match, current = verify_match(directory, left, right, models, cache)
        if signature is None:
            signature = current
        require(signature == current, "Opening histories differ between matches")
        matches.append(match)
        standings[left].update(match["left_wdl"])
        standings[right].update(match["right_wdl"])
    for item in standings.values():
        require(item["games"] == 768, "Incomplete player schedule")
        item["score"] = item["points"] / item["games"]
    summary = {"experiment_id": "VP-001", "status": "complete_independently_reconciled",
        "scope": "Single-seed searchless development screening; no search/absolute Elo/held-out confirmation claim.",
        "interval_note": "Marginal 95% Hoeffding intervals on complete opening pairs; not multiplicity-adjusted. Elo is match-relative logistic only.",
        "loss_note": "Policy CE in nats over full legal set, same 8192 played-move targets. Native policy/DFM1/DFM8 separated. Composite train and validation objectives are different; intermediate validation uses 2048, final uses 8192.",
        "models": models, "standings": standings, "matches": matches,
        "policy_losses": verify_losses(directory, models), "training": verify_training(directory),
        "verification": {"games_legally_replayed_with_opening_history": 1536,
            "policy_aggregations_recomputed": 10, "matched_training_batches_per_arm": 1000,
            "opening_pairs_shared_across_all_matches": 128,
            "sources": {str(path.relative_to(args.directory)): sha(path) for path in sorted(directory.rglob("*")) if path.is_file()}}}
    path = args.directory / "summary_verified.json"
    payload = json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if path.exists():
        require(path.read_text() == payload, "Immutable summary differs")
    else:
        with path.open("x") as stream:
            stream.write(payload)
    print(json.dumps({"status": summary["status"], "standings": standings, "policy_losses": summary["policy_losses"]}, indent=2))


if __name__ == "__main__":
    main()
