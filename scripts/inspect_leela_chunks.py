#!/usr/bin/env python3
"""Inspect LC0 self-play chunks and preview one training batch."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chess_dfm_jax.data.leela import LeelaChunkDataLoader, discover_chunk_files


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk", action="append")
    parser.add_argument("--chunk-dir", default=None)
    parser.add_argument("--chunk-list", default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--decode-mode", choices=["raw", "board"], default="raw")
    parser.add_argument("--shuffle-buffer", type=int, default=0)
    parser.add_argument("--prefetch-batches", type=int, default=2)
    parser.add_argument("--include-moves", action="store_true")
    parser.add_argument("--shuffle-files", action="store_true")
    args = parser.parse_args()

    chunk_paths = discover_chunk_files(
        chunk_paths=args.chunk,
        chunk_dir=args.chunk_dir,
        chunk_list=args.chunk_list,
    )
    if not chunk_paths:
        raise SystemExit("No chunk files found. Pass --chunk, --chunk-dir, or --chunk-list.")

    print(f"Chunk files: {len(chunk_paths)}")
    print(f"First file: {chunk_paths[0]}")

    loader = LeelaChunkDataLoader(
        chunk_paths,
        batch_size=args.batch_size,
        decode_mode=args.decode_mode,
        include_board=True,
        include_moves=args.include_moves,
        shuffle_buffer=args.shuffle_buffer,
        prefetch_batches=args.prefetch_batches,
        shuffle_files=args.shuffle_files,
    )
    batch = next(iter(loader))

    print(f"planes shape: {batch['planes'].shape}")
    print(f"played_idx shape: {batch['played_idx'].shape}")
    print(f"best_idx shape: {batch['best_idx'].shape}")
    print(f"side_to_move: {batch['side_to_move'][: min(8, len(batch['side_to_move']))]}")
    print(f"input_format sample: {batch['input_format'][0]}")

    boards = batch.get("boards")
    if boards:
        board = boards[0]
        print("first board:")
        print(board)
        print(f"first fen: {board.fen()}")

    played_moves = batch.get("played_move")
    if played_moves:
        print(f"first played move: {played_moves[0]}")
    best_moves = batch.get("best_move")
    if best_moves:
        print(f"first best move: {best_moves[0]}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
