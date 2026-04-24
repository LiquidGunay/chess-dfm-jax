#!/usr/bin/env python3
import sys
import chess
import numpy as np
import jax
import jax.numpy as jnp
from pathlib import Path
import argparse
import json
import functools

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chess_dfm_jax.encoding import encode_board
from chess_dfm_jax.policy import legal_move_mask, policy_index_to_move
from chess_dfm_jax.analysis.profile_targets import load_mapped_bt4_params
from chess_dfm_jax.training.dfm import create_dfm_components, DFMConfig
from chess_dfm_jax.training.checkpoints import load_training_checkpoint

@functools.partial(jax.jit, static_argnames=['refinement_steps'])
def dfm_infer(model, planes, legal_mask, refinement_steps: int):
    batch_size = planes.shape[0]
    K = model.config.horizon
    MASK = model.config.action_vocab_size
    
    x = jnp.full((batch_size, K), MASK, dtype=jnp.int32)
    
    def step_fn(i, val):
        x_curr = val
        t = jnp.full((batch_size,), i / refinement_steps, dtype=jnp.float32)
        logits = model(planes, x_curr, t) # [B, K, V]
        
        # Masked Diffusion: constrain first action to legal moves
        # We only have the legal mask for k=0
        mask_expanded = legal_mask[:, None, :] # [B, 1, V]
        
        # Apply mask to k=0 logits
        logits_0 = logits[:, 0:1, :]
        logits_0 = jnp.where(mask_expanded, logits_0, -1e9)
        
        # Concatenate with rest of the horizon (unconstrained)
        if K > 1:
            logits = jnp.concatenate([logits_0, logits[:, 1:, :]], axis=1)
        else:
            logits = logits_0
            
        probs = jax.nn.softmax(logits, axis=-1)
        max_probs = jnp.max(probs, axis=-1) # [B, K]
        preds = jnp.argmax(logits, axis=-1) # [B, K]
        
        num_unmasked_target = (K * (i + 1)) // refinement_steps
        max_probs_all = jnp.where(x_curr == MASK, max_probs, 2.0)
        kth_idx = jnp.maximum(0, K - num_unmasked_target)
        sorted_probs = jnp.sort(max_probs_all, axis=-1)
        thresholds = jax.lax.dynamic_slice_in_dim(sorted_probs, kth_idx, 1, axis=-1)
        
        unmask_now = max_probs_all >= thresholds
        x_next = jnp.where(unmask_now, preds, x_curr)
        return x_next

    x_final = jax.lax.fori_loop(0, refinement_steps, step_fn, x)
    return x_final

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bt4-pb", type=str, required=True)
    parser.add_argument("--dfm-ckpt", type=str, required=True)
    parser.add_argument("--layers", type=int, default=4)
    args = parser.parse_args()

    params = load_mapped_bt4_params(pb=args.bt4_pb)
    config = DFMConfig(token_dim=512, num_layers=args.layers, use_qk_gain=True, use_muon=True)
    model, optimizer = create_dfm_components(params, config)

    # Load checkpoint
    ckpt_path = Path(args.dfm_ckpt)
    step = load_training_checkpoint(ckpt_path, model=model)
    if step is None:
        raise ValueError(f"Failed to load checkpoint from {ckpt_path}")

    board = chess.Board()
    refinement_steps = 8
    
    while True:
        try:
            line = sys.stdin.readline()
        except EOFError:
            break
        if not line:
            break
        
        line = line.strip()
        if line == "uci":
            print("id name DFM_Planner")
            print("id author lc0jax")
            print("option name RefinementSteps type spin default 8 min 1 max 8")
            print("uciok")
            sys.stdout.flush()
        elif line == "isready":
            dummy_planes = jnp.zeros((1, 112, 8, 8), dtype=jnp.float16)
            dummy_mask = jnp.ones((1, 1858), dtype=bool)
            _ = dfm_infer(model, dummy_planes, dummy_mask, refinement_steps=8)
            print("readyok")
            sys.stdout.flush()
        elif line.startswith("setoption name RefinementSteps value"):
            parts = line.split()
            refinement_steps = int(parts[-1])
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
            mask = legal_move_mask(board, "lc0_1858")
            mask_jnp = jnp.asarray(mask)[None, ...]
            
            actions = dfm_infer(model, planes_jnp, mask_jnp, refinement_steps=refinement_steps)
            actions = np.asarray(actions[0])
            
            best_idx = int(actions[0])
            
            # Final verification just in case, but dfm_infer should now handle it
            if not mask[best_idx]:
                with open("/tmp/illegal_moves.log", "a") as f:
                    f.write("1\n")
                p, _, _ = model.encoder(planes_jnp)
                p = np.asarray(p[0])
                masked_p = np.where(mask, p, -1e9)
                best_idx = int(np.argmax(masked_p))
                
            best_move = policy_index_to_move(best_idx, "lc0_1858")
            print(f"bestmove {best_move.uci()}")
            sys.stdout.flush()
        elif line == "quit":
            break

if __name__ == "__main__":
    main()
