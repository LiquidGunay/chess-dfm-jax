#!/usr/bin/env python3
"""Stream PGNs from Lichess, encode them via BT4, and upload chunks directly to GCS."""

import argparse
import os
import io
import time
import sys
import requests
import zstandard as zstd
import jax
import jax.numpy as jnp
import numpy as np
import chess
import chess.pgn
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chess_dfm_jax.encoding import encode_board
from chess_dfm_jax.policy import move_to_policy_index
from chess_dfm_jax.nnx_bt4 import BT4Model, make_bt4_model, jit_encode_tokens
from chess_dfm_jax.weights import load_pb_gz, map_bt4_weights

def slice_game(game: chess.pgn.Game, horizon: int):
    """Slice a game into chunks of length H."""
    board = game.board()
    moves = list(game.mainline_moves())
    
    for t in range(len(moves) - horizon):
        current_board = board.copy(stack=False)
        for m in moves[:t]:
            current_board.push(m)
            
        planes_t = encode_board(current_board, []).astype(np.float32)
        
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
            
        target_board = current_board.copy(stack=False)
        for i in range(horizon):
            target_board.push(moves[t+i])
            
        planes_target = encode_board(target_board, []).astype(np.float32)
        
        yield {
            "planes_t": planes_t,
            "actions": np.array(actions, dtype=np.int32),
            "planes_target": planes_target,
        }

def process_stream(args):
    print("Loading BT4 Model...")
    bundle = load_pb_gz(os.path.join(args.models_dir, "BT4-1024x15x32h-swa-6147500-policytune-332.pb.gz"))
    bt4_params = map_bt4_weights(bundle)
    model = make_bt4_model(bt4_params, dtype=jnp.bfloat16)
    
    print(f"Streaming from {args.url}")
    response = requests.get(args.url, stream=True)
    response.raise_for_status()
    
    dctx = zstd.ZstdDecompressor()
    stream_reader = dctx.stream_reader(response.raw)
    text_stream = io.TextIOWrapper(stream_reader, encoding='utf-8')
    
    batch_size = args.batch_size
    chunk_buffer = []
    chunk_index = 0
    
    os.makedirs("/tmp/lichess_chunks", exist_ok=True)
    
    while True:
        try:
            game = chess.pgn.read_game(text_stream)
        except Exception as e:
            print(f"Error parsing game: {e}")
            continue
            
        if game is None:
            break
            
        for sample in slice_game(game, args.horizon):
            chunk_buffer.append(sample)
            
            if len(chunk_buffer) >= batch_size:
                planes_t = np.stack([s["planes_t"] for s in chunk_buffer])
                planes_target = np.stack([s["planes_target"] for s in chunk_buffer])
                actions = np.stack([s["actions"] for s in chunk_buffer])
                
                p, v, ml = model(jnp.asarray(planes_t))
                wdl_target = jax.nn.softmax(v, axis=-1) # [Batch, 3] usually Win, Draw, Loss
                # WDL format: W, D, L. So Q = W - L
                w = wdl_target[:, 0]
                l = wdl_target[:, 2]
                value_target = w - l
                
                out_path = f"/tmp/lichess_chunks/chunk_{chunk_index:06d}.npz"
                np.savez_compressed(
                    out_path,
                    planes_t=np.asarray(planes_t),
                    actions=actions,
                    planes_target=np.asarray(planes_target),
                    value_target=np.asarray(value_target),
                    wdl_target=np.asarray(wdl_target)
                )
                print(f"Saved chunk {chunk_index}, uploading...")
                os.system(f"gcloud storage cp {out_path} {args.upload_gcs}/chunk_{chunk_index:06d}.npz --quiet")
                os.remove(out_path)
                
                chunk_index += 1
                chunk_buffer = []
                
                if chunk_index >= args.max_chunks:
                    print(f"Reached max chunks {args.max_chunks}")
                    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", type=str, required=True)
    parser.add_argument("--models-dir", type=str, required=True)
    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-chunks", type=int, default=1000)
    parser.add_argument("--upload-gcs", type=str, required=True)
    args = parser.parse_args()
    process_stream(args)
