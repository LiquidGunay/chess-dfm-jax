#!/usr/bin/env python3
import sys
import chess
import numpy as np
import jax
import jax.numpy as jnp
from pathlib import Path
import argparse

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chess_dfm_jax.encoding import encode_board
from chess_dfm_jax.policy import legal_move_mask, policy_index_to_move
from chess_dfm_jax.analysis.profile_targets import load_mapped_bt4_params
from chess_dfm_jax.nnx_bt4 import make_bt4_model, jit_bt4_forward

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bt4-pb", type=str, required=True)
    args = parser.parse_args()

    # Load model
    params = load_mapped_bt4_params(pb=args.bt4_pb)
    model = make_bt4_model(params, dtype=jnp.float16)

    # UCI Loop
    board = chess.Board()
    
    while True:
        try:
            line = sys.stdin.readline()
        except EOFError:
            break
        if not line:
            break
        
        line = line.strip()
        if line == "uci":
            print("id name BT4_Baseline")
            print("id author lc0jax")
            print("uciok")
            sys.stdout.flush()
        elif line == "isready":
            # Warm up JIT
            dummy_planes = jnp.zeros((1, 112, 8, 8), dtype=jnp.float16)
            _ = jit_bt4_forward(model, dummy_planes)
            print("readyok")
            sys.stdout.flush()
        elif line.startswith("position"):
            parts = line.split()
            if "startpos" in parts:
                board.set_fen(chess.STARTING_FEN)
                if "moves" in parts:
                    moves_idx = parts.index("moves")
                    for m in parts[moves_idx + 1:]:
                        board.push(chess.Move.from_uci(m))
            elif "fen" in parts:
                fen_idx = parts.index("fen")
                fen = " ".join(parts[fen_idx + 1 : fen_idx + 7])
                board.set_fen(fen)
                if "moves" in parts:
                    moves_idx = parts.index("moves")
                    for m in parts[moves_idx + 1:]:
                        board.push(chess.Move.from_uci(m))
        elif line.startswith("go"):
            planes = encode_board(board, [], input_format="INPUT_CLASSICAL_112_PLANE")
            planes_jnp = jnp.asarray(planes, dtype=jnp.float16)[None, ...]
            policy, wdl, moves_left = jit_bt4_forward(model, planes_jnp)
            policy = np.asarray(policy[0])
            mask = legal_move_mask(board, "lc0_1858")
            masked_policy = np.where(mask, policy, -1e9)
            best_idx = int(np.argmax(masked_policy))
            best_move = policy_index_to_move(best_idx, "lc0_1858")
            print(f"bestmove {best_move.uci()}")
            sys.stdout.flush()
        elif line == "quit":
            break

if __name__ == "__main__":
    main()
