#!/usr/bin/env python3
"""Pre-process PGN games into fixed-length action chunks for JEPA/DFM training."""

import argparse
import glob
import os
import time
from pathlib import Path
from typing import Iterator

import jax
import jax.numpy as jnp
import numpy as np
import chess
import chess.pgn
import optax

from chess_dfm_jax.encoding import encode_board
from chess_dfm_jax.policy import move_to_policy_index
from chess_dfm_jax.nnx_bt4 import BT4Model, make_bt4_model, jit_encode_tokens
from chess_dfm_jax.weights import load_pb_gz, map_bt4_weights

def read_games(pgn_paths: list[str]) -> Iterator[chess.pgn.Game]:
    for path in pgn_paths:
        with open(path, "r", encoding="utf-8") as f:
            while True:
                game = chess.pgn.read_game(f)
                if game is None:
                    break
                yield game

def slice_game(game: chess.pgn.Game, horizon: int) -> Iterator[dict]:
    """Slice a game into chunks of length H."""
    board = game.board()
    moves = list(game.mainline_moves())
    
    # We need states at t and t+H, and actions a_t to a_{t+H-1}
    for t in range(len(moves) - horizon):
        # State at t
        current_board = board.copy(stack=False)
        for m in moves[:t]:
            current_board.push(m)
            
        planes_t = encode_board(current_board, []).astype(np.float32)
        
        # Actions t to t+H-1
        actions = []
        valid_actions = True
        for i in range(horizon):
            m = moves[t+i]
            try:
                idx = move_to_policy_index(m, "lc0_1858")
                actions.append(idx)
            except Exception:
                valid_actions = False
                break
        
        if not valid_actions:
            continue
            
        # State at t+H
        target_board = current_board.copy(stack=False)
        for i in range(horizon):
            target_board.push(moves[t+i])
            
        planes_target = encode_board(target_board, []).astype(np.float32)
        
        yield {
            "planes_t": planes_t,
            "actions": np.array(actions, dtype=np.int32),
            "planes_target": planes_target,
        }

def process_and_upload(args):
    print("Loading BT4 Model...")
    bundle = load_pb_gz(args.models_dir + "/BT4-1024x15x32h-swa-6147500-policytune-332.pb.gz")
    bt4_params = map_bt4_weights(bundle)
    
    # We run the encoder in fp16/bf16 to save space if needed
    model = make_bt4_model(bt4_params, dtype=jnp.bfloat16)
    
    pgn_files = glob.glob(os.path.join(args.pgn_dir, "**/*.pgn"), recursive=True)
    if not pgn_files:
        print(f"No PGN files found in {args.pgn_dir}")
        return
        
    print(f"Found {len(pgn_files)} PGN files.")
    
    batch_size = args.batch_size
    chunk_buffer = []
    chunk_index = 0
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Process
    for game in read_games(pgn_files):
        for sample in slice_game(game, args.horizon):
            chunk_buffer.append(sample)
            
            if len(chunk_buffer) >= batch_size:
                # Stack
                planes_t = np.stack([s["planes_t"] for s in chunk_buffer])
                planes_target = np.stack([s["planes_target"] for s in chunk_buffer])
                actions = np.stack([s["actions"] for s in chunk_buffer])
                
                # Encode via BT4
                # Note: We encode on GPU/TPU if available, CPU otherwise.
                z_t = jit_encode_tokens(model, jnp.asarray(planes_t))
                z_target = jit_encode_tokens(model, jnp.asarray(planes_target))
                
                # Save batch to disk as npz
                out_path = os.path.join(args.out_dir, f"chunk_{chunk_index:06d}.npz")
                np.savez_compressed(
                    out_path,
                    z_t=np.asarray(z_t),
                    actions=actions,
                    z_target=np.asarray(z_target)
                )
                print(f"Saved {out_path}")
                chunk_index += 1
                chunk_buffer = []
                
                if chunk_index >= args.max_chunks:
                    print(f"Reached max chunks {args.max_chunks}")
                    break
        if chunk_index >= args.max_chunks:
            break
            
    print("Done generating chunks.")
    if args.upload_gcs:
        print(f"Uploading to {args.upload_gcs}...")
        os.system(f"gcloud storage cp {args.out_dir}/*.npz {args.upload_gcs}")
        print("Upload complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pgn-dir", type=str, required=True)
    parser.add_argument("--models-dir", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-chunks", type=int, default=100)
    parser.add_argument("--upload-gcs", type=str, default="")
    args = parser.parse_args()
    process_and_upload(args)
