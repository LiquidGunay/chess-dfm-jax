import os
import sys
import jax
import jax.numpy as jnp
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chess_dfm_jax.analysis.profile_targets import load_mapped_bt4_params
from chess_dfm_jax.nnx_bt4 import make_bt4_model, jit_bt4_forward
from chess_dfm_jax.reference_bt4 import bt4_forward

def check_bt4_parity():
    print("Checking BT4 Parity...")
    
    models_dir = REPO_ROOT / "models"
    pb_path = None
    if models_dir.exists():
        for f in models_dir.iterdir():
            if f.name.endswith(".pb.gz") and "exported" not in f.name:
                pb_path = f
                break
    
    if pb_path is None:
        print("No pb file found.")
        return
        
    params = load_mapped_bt4_params(pb=str(pb_path))
    model = make_bt4_model(params, dtype=jnp.float32)
    
    rng = jax.random.PRNGKey(42)
    
    import chess
    from chess_dfm_jax.encoding import encode_board
    from chess_dfm_jax.policy import policy_index_to_move
    
    board = chess.Board()
    planes = encode_board(board, [], input_format="INPUT_CLASSICAL_112_PLANE")
    planes = jnp.asarray(planes, dtype=jnp.float32)[None, ...]
    
    # NNX version
    nnx_x = model.encode_tokens(planes)
    nnx_x = nnx_x.reshape((-1, 1024))
    nnx_p, nnx_v, nnx_ml = jit_bt4_forward(model, planes)
    
    # Reference version
    ref_p, ref_v, ref_ml, ref_acts = bt4_forward(params, planes, capture=True)
    ref_x = ref_acts["trunk"]
    
    print(f"Max Diff Input Embedding: {jnp.max(jnp.abs(model.embedding(planes, 1.0)[0].reshape((-1, 1024)) - ref_acts['attn_body']))}")
    
    x_nnx, _ = model.embedding(planes, 1.0)
    x_nnx = x_nnx.reshape((-1, 64, 1024))
    for i, layer in enumerate(model.layers):
        x_nnx = layer(x_nnx)
        print(f"Max Diff Layer {i}: {jnp.max(jnp.abs(x_nnx.reshape((-1, 1024)) - ref_acts[f'encoder_{i}']))}")
    
    x_diff = jnp.max(jnp.abs(nnx_x - ref_x))
    p_diff = jnp.max(jnp.abs(nnx_p - ref_p))
    v_diff = jnp.max(jnp.abs(nnx_v - ref_v))
    ml_diff = jnp.max(jnp.abs(nnx_ml - ref_ml))
    
    print(f"Max Diff Encoder: {x_diff}")
    print(f"Max Diff Policy: {p_diff}")
    print(f"Max Diff Value: {v_diff}")
    print(f"Max Diff Moves Left: {ml_diff}")
    
    if p_diff < 1e-4 and v_diff < 1e-4 and ml_diff < 1e-4:
        print("PARITY PASSED!")
    else:
        print("PARITY FAILED!")
        
    print("\n--- Top 5 Moves (NNX) ---")
    nnx_p_np = np.asarray(nnx_p[0])
    for idx in np.argsort(nnx_p_np)[-5:][::-1]:
        print(f"{policy_index_to_move(idx, 'lc0_1858').uci()}: {nnx_p_np[idx]:.4f}")
        
    print("\n--- Top 5 Moves (Reference) ---")
    ref_p_np = np.asarray(ref_p[0])
    for idx in np.argsort(ref_p_np)[-5:][::-1]:
        print(f"{policy_index_to_move(idx, 'lc0_1858').uci()}: {ref_p_np[idx]:.4f}")

if __name__ == "__main__":
    check_bt4_parity()
