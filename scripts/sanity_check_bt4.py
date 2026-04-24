
import sys
import jax.numpy as jnp
import numpy as np
import chess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chess_dfm_jax.encoding import encode_board
from chess_dfm_jax.policy import legal_move_mask, policy_index_to_move
from chess_dfm_jax.analysis.profile_targets import load_mapped_bt4_params
from chess_dfm_jax.nnx_bt4 import make_bt4_model, jit_bt4_forward

def sanity_check():
    pb_path = "models/BT4-1024x15x32h-swa-6147500-policytune-332.pb.gz"
    print(f"Loading {pb_path}...")
    params = load_mapped_bt4_params(pb=pb_path)
    model = make_bt4_model(params, dtype=jnp.float32)
    
    board = chess.Board() # Starting position
    planes = encode_board(board, [], input_format="INPUT_CLASSICAL_112_PLANE")
    planes_jnp = jnp.asarray(planes, dtype=jnp.float32)[None, ...]
    
    print("Running forward pass...")
    policy, _, _ = jit_bt4_forward(model, planes_jnp)
    policy = np.asarray(policy[0])
    
    mask = legal_move_mask(board, "lc0_1858")
    masked_policy = np.where(mask, policy, -1e9)
    
    # Get top 5 moves
    top_indices = np.argsort(masked_policy)[-5:][::-1]
    print("\nTop 5 moves for starting position:")
    for idx in top_indices:
        move = policy_index_to_move(idx, "lc0_1858")
        logit = masked_policy[idx]
        print(f"  {move.uci()}: {logit:.4f}")

if __name__ == "__main__":
    sanity_check()
