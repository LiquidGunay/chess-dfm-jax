#!/usr/bin/env python3
import argparse
import os
import sys
import requests
import zipfile
import io
import numpy as np
import chess
import chess.pgn
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chess_dfm_jax.encoding import encode_board
from chess_dfm_jax.policy import move_to_policy_index, legal_move_mask

def slice_game(game: chess.pgn.Game, horizon: int):
    result_str = game.headers.get("Result", "*")
    if result_str == "1-0":
        w_score, d_score, l_score = 1.0, 0.0, 0.0
    elif result_str == "0-1":
        w_score, d_score, l_score = 0.0, 0.0, 1.0
    elif result_str == "1/2-1/2":
        w_score, d_score, l_score = 0.0, 1.0, 0.0
    else:
        return
        
    board = game.board()
    moves = list(game.mainline_moves())
    
    for t in range(len(moves) - horizon):
        current_board = board.copy(stack=False)
        for m in moves[:t]:
            current_board.push(m)
            
        planes_t = encode_board(current_board, []).astype(np.float32)
        legal_mask = legal_move_mask(current_board, "lc0_1858").astype(np.float32)

        
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
        
        # WDL from perspective of player to move
        if current_board.turn == chess.WHITE:
            wdl = [w_score, d_score, l_score]
        else:
            wdl = [l_score, d_score, w_score]
            
        q_value = wdl[0] - wdl[2]
        
        yield {
            "planes_t": planes_t,
            "legal_mask": legal_mask,
            "actions": np.array(actions, dtype=np.int32),
            "planes_target": planes_target,
            "value_target": np.array(q_value, dtype=np.float32),
            "wdl_target": np.array(wdl, dtype=np.float32),
        }

def process_tcec(args):
    print(f"Downloading TCEC zip from {args.url}")
    zip_path = "/tmp/tcec_download.zip"
    with requests.get(args.url, stream=True) as r:
        r.raise_for_status()
        with open(zip_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
                
    batch_size = args.batch_size
    chunk_buffer = []
    chunk_index = 0
    
    os.makedirs("/tmp/tcec_chunks", exist_ok=True)
    
    with zipfile.ZipFile(zip_path) as z:
        for filename in z.namelist():
            if filename.endswith(".pgn"):
                print(f"Processing {filename}...")
                with z.open(filename) as f:
                    text_stream = io.TextIOWrapper(f, encoding='utf-8', errors='ignore')
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
                                legal_mask = np.stack([s["legal_mask"] for s in chunk_buffer])
                                planes_target = np.stack([s["planes_target"] for s in chunk_buffer])
                                actions = np.stack([s["actions"] for s in chunk_buffer])
                                value_target = np.stack([s["value_target"] for s in chunk_buffer])
                                wdl_target = np.stack([s["wdl_target"] for s in chunk_buffer])
                                
                                out_path = f"/tmp/tcec_chunks/chunk_{chunk_index:06d}.npz"
                                np.savez_compressed(
                                    out_path,
                                    planes_t=planes_t,
                                    legal_mask=legal_mask,
                                    actions=actions,
                                    planes_target=planes_target,
                                    value_target=value_target,
                                    wdl_target=wdl_target
                                )
                                print(f"Saved chunk {chunk_index}, uploading...")
                                subprocess.run(["gcloud", "storage", "cp", out_path, f"{args.upload_gcs}/chunk_{chunk_index:06d}.npz", "--quiet"], check=False)
                                os.remove(out_path)
                                
                                chunk_index += 1
                                chunk_buffer = []
                                
                                if chunk_index >= args.max_chunks:
                                    print(f"Reached max chunks {args.max_chunks}")
                                    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", type=str, default="https://github.com/TCEC-Chess/tcecgames/releases/download/S28-final/TCEC-all-in-one-compact.zip")
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--max-chunks", type=int, default=10)
    parser.add_argument("--upload-gcs", type=str, required=True)
    args = parser.parse_args()
    process_tcec(args)
