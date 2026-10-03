"""Reproduce full-history arena-plane parity against retained official LC0 records."""
import argparse
import hashlib
import json
from pathlib import Path

import chess
import numpy as np

from chess_dfm_jax.data.lc0_sequential import SequentialShard, decode_classical_planes
from chess_dfm_jax.encoding import encode_board
from chess_dfm_jax.policy import ACTION_CODEC_LC0_CANONICAL_1858, decode_action
from research.analysis.run_full_history_eval import write_once
from research.input_contract import audit_full_history_batch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = sorted(args.dataset_root.glob("chunks/chunk-*/train"))
    if len(paths) < 3:
        raise ValueError("Expected complete sequential corpus")
    rows = []
    total = legacy_zero = future = 0
    for path in (paths[0], paths[len(paths)//2], paths[-1]):
        shard = SequentialShard(path)
        for game in range(min(10, len(shard.game_offsets_i64)-1)):
            start, stop = map(int, shard.game_offsets_i64[game:game+2])
            board = chess.Board()
            history = [board.copy(stack=False)]
            stored = decode_classical_planes(shard.planes_u64[start:stop], shard.metadata_u8[start:stop],
                                             shard.input_format_u16[start:stop])
            for row in range(start, stop):
                encoded = encode_board(board, history)
                np.testing.assert_array_equal(encoded, stored[row-start])
                legacy = encode_board(board, [])
                if np.count_nonzero(legacy[13:104]) or np.count_nonzero(legacy[12]):
                    raise ValueError("Legacy zero-history expectation changed")
                legacy_zero += 1
                total += 1
                if row < stop-1:
                    move = decode_action(int(shard.played_actions_u16[row]),
                                         codec_id=ACTION_CODEC_LC0_CANONICAL_1858, board=board)
                    if move not in board.legal_moves:
                        raise ValueError("Stored action is illegal during replay")
                    board.push(move)
                    history.append(board.copy(stack=False))
            batch = shard.materialize(np.array([start]), horizon=min(8, stop-start-1))
            future += audit_full_history_batch(batch)["future_targets_with_history"]
            rows.append({"shard": str(path.relative_to(args.dataset_root)), "game": game,
                         "positions": stop-start, "stored_planes_sha256": hashlib.sha256(stored.tobytes()).hexdigest()})
    result = {"schema_version": "chess-dfm-full-history-input-parity-v1", "passed": True,
              "dataset_root": str(args.dataset_root), "positions": total, "games": len(rows),
              "legacy_zero_history_positions": legacy_zero, "verified_future_targets": future,
              "comparison": "Exact equality of all 112 encoded planes versus official LC0 training-record planes, replaying complete games from the initial position",
              "scope": "Input-plane parity, not native-LC0 neural-output parity", "records": rows}
    write_once(args.output, result)
    print(json.dumps({k: v for k, v in result.items() if k != "records"}))


if __name__ == "__main__":
    main()
