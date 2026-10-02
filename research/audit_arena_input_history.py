"""Verify history-aware arena encoding against immutable LC0 training planes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import chess
import numpy as np

from chess_dfm_jax.data.lc0_sequential import SequentialShard, decode_classical_planes
from chess_dfm_jax.encoding import encode_board
from chess_dfm_jax.policy import decode_lc0_canonical_1858


def audit(root: Path, *, games_per_shard: int = 10) -> dict:
    paths = sorted(root.glob("chunks/chunk-*/train/manifest.json"))
    if not paths:
        raise ValueError("No sequential training shards")
    selected = sorted({paths[0], paths[len(paths) // 2], paths[-1]})
    count = mismatch = current_only_mismatch = repeated_positions = 0
    records = []
    for manifest in selected:
        shard = SequentialShard(manifest.parent)
        for game in range(min(games_per_shard, len(shard.game_offsets_i64) - 1)):
            start, stop = map(int, shard.game_offsets_i64[game:game + 2])
            expected = decode_classical_planes(
                shard.planes_u64[start:stop], shard.metadata_u8[start:stop],
                shard.input_format_u16[start:stop],
            )
            board = chess.Board()
            history = []
            for row in range(stop - start):
                # Arena histories cross the sealed boundary as legal-EP FENs.
                history.append(chess.Board(board.fen(en_passant="legal")))
                actual = encode_board(history[-1], history)
                mismatch += int(not np.array_equal(actual, expected[row]))
                current_only_mismatch += int(not np.array_equal(encode_board(history[-1], []), expected[row]))
                repeated_positions += int(bool(np.any(expected[row, 12:104:13])))
                count += 1
                if row < stop - start - 1:
                    board.push(decode_lc0_canonical_1858(board, int(shard.played_actions_u16[start + row])))
            records.append({
                "shard": str(manifest.parent.relative_to(root)), "game_index": game,
                "positions": stop - start,
                "stored_planes_sha256": hashlib.sha256(expected.tobytes()).hexdigest(),
            })
    result = {
        "schema_version": "chess-dfm-arena-input-history-audit-v1",
        "dataset_root": str(root), "positions": count,
        "mismatched_positions_full_history": mismatch,
        "mismatched_positions_current_only": current_only_mismatch,
        "positions_with_repetition_planes": repeated_positions,
        "game_count": len(records), "games": records,
        "passed": mismatch == 0,
    }
    if mismatch:
        raise ValueError(json.dumps(result))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--games-per-shard", type=int, default=10)
    args = parser.parse_args()
    result = audit(args.dataset_root, games_per_shard=args.games_per_shard)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({key: value for key, value in result.items() if key != "games"}))


if __name__ == "__main__":
    main()
