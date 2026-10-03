"""CPU-only, fail-closed feasibility audit for restoring legacy Hero history.

Rows are ordered trajectory windows, not independent game IDs: the retained
PGNs often have identical placeholder headers. This audits ordered shards and
their immediate predecessor context without treating those headers as keys.
It does not emit a training dataset or authorize a GPU launch.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import chess
import numpy as np

from chess_dfm_jax.encoding import encode_board


def _without_history(planes):
    result = planes.copy()
    result[12:104] = 0
    return result


def profile_payload(data):
    plies = np.asarray(data["ply"])
    roots = np.flatnonzero(plies == 0)
    gaps = np.flatnonzero((plies[1:] != 0) & (plies[1:] != plies[:-1] + 1)) + 1
    return {
        "rows": int(data["batch_size"]),
        "distinct_header_game_ids": len(set(map(str, data["game_id"]))),
        "root_rows": len(roots),
        "nonstandard_root_rows": [
            {"row": int(row), "fen": str(data["fen_t"][row])}
            for row in roots if str(data["fen_t"][row]) != chess.STARTING_FEN
        ],
        "within_shard_ply_gaps": [
            {"row": int(row), "previous_ply": int(plies[row-1]), "ply": int(plies[row])}
            for row in gaps
        ],
    }


def audit_payloads(payloads, *, context_count=0):
    """Audit all target rows; leading incomplete context is never a target.

    Require complete, uninterrupted prefixes from the standard initial board.
    A new game begins only at ply zero. Overlapping future windows must agree.
    Stored current/future planes must equal replay with only history removed.
    """
    if not 0 <= context_count < len(payloads):
        raise ValueError("Expected at least one target shard")
    segments = []
    segment = None
    headers = set()
    ignored_context = 0
    for shard, data in enumerate(payloads):
        if str(data["schema_version"]) != "trajectory-v3":
            raise ValueError("Expected legacy trajectory-v3")
        count, horizon = data["actions_uci"].shape
        if count != int(data["batch_size"]):
            raise ValueError("Shard row count mismatch")
        if not np.all(data["future_valid_u8"] == 1):
            raise ValueError("Expected complete Hero future windows")
        for row in range(count):
            ply = int(data["ply"][row])
            headers.add(str(data["game_id"][row]))
            if ply == 0:
                segment = []
                segments.append(segment)
            if segment is None:
                if shard >= context_count:
                    raise ValueError("Missing initial game prefix for target row")
                ignored_context += 1
                continue
            if str(data["input_format"][row]) != "INPUT_CLASSICAL_112_PLANE":
                raise ValueError("Unsupported input format")
            segment.append((shard, row, ply, horizon))

    targets = futures = games = repetitions = 0
    recovered_digest = hashlib.sha256()
    for segment in segments:
        if not any(shard >= context_count for shard, *_ in segment):
            continue
        games += 1
        moves = {}
        by_ply = {}
        previous_ply = -1
        for shard, row, ply, horizon in segment:
            if ply != previous_ply + 1:
                raise ValueError(f"Missing/nonconsecutive ply in target game: {previous_ply} -> {ply}, shard {shard} row {row}")
            previous_ply = ply
            data = payloads[shard]
            by_ply[ply] = (shard, row)
            for offset, uci in enumerate(data["actions_uci"][row]):
                key, value = ply + offset, str(uci)
                if key in moves and moves[key] != value:
                    raise ValueError("Conflicting overlapping future moves")
                moves[key] = value
        board = chess.Board()
        history = [board.copy(stack=False)]
        encoded = []
        results = set()
        for ply in range(max(moves) + 2):
            if ply in by_ply:
                shard, row = by_ply[ply]
                data = payloads[shard]
                if board.fen() != str(data["fen_t"][row]):
                    raise ValueError(f"Replayed FEN mismatch (or unsupported setup game), shard {shard} row {row} ply {ply}")
                results.add(str(data["result"][row]))
            planes = encode_board(board, history)
            encoded.append(planes.astype(np.uint8))
            if ply <= max(moves):
                if ply not in moves:
                    raise ValueError("Missing prefix move")
                move = chess.Move.from_uci(moves[ply])
                if move not in board.legal_moves:
                    raise ValueError("Illegal replay move")
                board.push(move)
                history.append(board.copy(stack=False))
        if len(results) != 1:
            raise ValueError("Result changed inside a reconstructed game")
        for shard, row, ply, horizon in segment:
            data = payloads[shard]
            current = encoded[ply]
            future = np.stack(encoded[ply + 1:ply + horizon + 1])
            np.testing.assert_array_equal(
                data["planes_t_u8"][row], _without_history(current),
                err_msg="Non-history current input differs from replay",
            )
            for offset, item in enumerate(future):
                np.testing.assert_array_equal(
                    data["planes_future_u8"][row, offset], _without_history(item),
                    err_msg="Non-history future input differs from replay",
                )
            if shard < context_count:
                continue
            if ply and not np.any(current[13:104]):
                raise ValueError("Reconstruction has missing current history")
            if any(not np.any(item[13:104]) for item in future):
                raise ValueError("Reconstruction has missing future history")
            targets += 1
            futures += horizon
            repetitions += int(bool(np.any(current[12])))
            recovered_digest.update(current.tobytes())
            recovered_digest.update(future.tobytes())
    expected = sum(int(data["batch_size"]) for data in payloads[context_count:])
    if targets != expected:
        raise ValueError(f"Incomplete target coverage: {targets} != {expected}")
    return {
        "passed": True, "target_rows": targets, "future_targets": futures,
        "reconstructed_games_touching_targets": games,
        "distinct_header_game_ids": len(headers),
        "target_rows_with_current_repetition": repetitions,
        "ignored_unrooted_context_rows": ignored_context,
        "reconstructed_planes_sha256": recovered_digest.hexdigest(),
        "scope": "Supplied ordered shards only; not full-corpus certification",
        "training_started": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard", type=Path, action="append", required=True)
    parser.add_argument("--context-count", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    records, payloads = [], []
    for path in args.shard:
        with np.load(path, allow_pickle=False) as source:
            payloads.append({key: source[key] for key in source.files})
        records.append({"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "source_uri": str(payloads[-1].get("source_uri", "")),
                        "profile": profile_payload(payloads[-1])})
    try:
        findings = audit_payloads(payloads, context_count=args.context_count)
    except (ValueError, AssertionError) as exc:
        findings = {"passed": False, "error": str(exc), "training_started": False}
    result = {"schema_version": "hero-history-recovery-audit-v1", "shards": records,
              "context_count": args.context_count, **findings}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
