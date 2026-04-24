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
from chess_dfm_jax.training.dfm import create_dfm_components, _dfm_loss_and_grad, DFMConfig
from chess_dfm_jax.training.jepa import build_synthetic_transition_batch
from flax import nnx

def check_gradients():
    print("Checking Gradients for DFM...")
    
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
    config = DFMConfig(token_dim=512, num_layers=2, use_muon=True)
    model, optimizer = create_dfm_components(params, config)
    
    batch = build_synthetic_transition_batch(batch_size=4, horizon=8)
    rng = jax.random.PRNGKey(42)
    
    (loss, aux), grads = _dfm_loss_and_grad(model, batch, rng)
    
    print(f"Loss: {loss}")
    print("Gradients:")
    
    flat_grads = nnx.to_pure_dict(grads)
    
    def print_grad_stats(d, prefix=""):
        for k, v in d.items():
            if isinstance(v, dict):
                print_grad_stats(v, prefix + str(k) + ".")
            else:
                if v is not None:
                    arr = np.asarray(v)
                    print(f"{prefix}{str(k)}: shape={arr.shape}, mean={np.mean(arr):.6e}, std={np.std(arr):.6e}, max={np.max(np.abs(arr)):.6e}")
                else:
                    print(f"{prefix}{str(k)}: None")

    print_grad_stats(flat_grads)
    
    old_params = nnx.to_pure_dict(model)
    optimizer.update(model, grads)
    new_params = nnx.to_pure_dict(model)
    
    diff = 0.0
    for k in old_params:
        if isinstance(old_params[k], dict):
            for k2 in old_params[k]:
                 diff += np.sum(np.abs(np.asarray(old_params[k][k2]) - np.asarray(new_params[k][k2])))
        else:
            diff += np.sum(np.abs(np.asarray(old_params[k]) - np.asarray(new_params[k])))
            
    print(f"Total parameter difference after update: {diff}")

if __name__ == "__main__":
    check_gradients()
