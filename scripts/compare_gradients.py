#!/usr/bin/env python3
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

def calculate_grad_norms(num_layers, params):
    config = DFMConfig(token_dim=512, num_layers=num_layers, use_muon=True)
    model, optimizer = create_dfm_components(params, config)
    
    batch = build_synthetic_transition_batch(batch_size=4, horizon=8)
    # Give random actions
    batch["action_indices"] = jax.random.randint(jax.random.PRNGKey(123), (4, 8), 0, 1858)
    rng = jax.random.PRNGKey(42)
    
    (loss, aux), grads = _dfm_loss_and_grad(model, batch, rng)
    
    flat_grads = nnx.to_pure_dict(grads)
    
    total_norm_sq = 0.0
    layer_norms = {}
    
    def accumulate_norms(d, prefix=""):
        nonlocal total_norm_sq
        for k, v in d.items():
            full_key = f"{prefix}{k}" if prefix else str(k)
            if isinstance(v, dict):
                accumulate_norms(v, full_key + ".")
            else:
                if v is not None:
                    arr = np.asarray(v)
                    sq = np.sum(np.square(arr))
                    total_norm_sq += sq
                    
                    # Accumulate by top-level block/layer
                    block_name = full_key.split('.')[0]
                    if full_key.startswith("blocks."):
                        block_name = ".".join(full_key.split('.')[:2])
                    layer_norms[block_name] = layer_norms.get(block_name, 0.0) + sq

    accumulate_norms(flat_grads)
    
    for k in layer_norms:
        layer_norms[k] = np.sqrt(layer_norms[k])
        
    return np.sqrt(total_norm_sq), layer_norms

def compare_gradients():
    pb_path = REPO_ROOT / "models" / "BT4-1024x15x32h-swa-6147500-policytune-332.pb.gz"
    if not pb_path.exists():
        print(f"Could not find {pb_path}")
        return
        
    params = load_mapped_bt4_params(pb=str(pb_path))
    
    print("--- L4 Model Gradients ---")
    l4_total, l4_layers = calculate_grad_norms(4, params)
    print(f"Total Grad Norm: {l4_total:.6f}")
    for k, v in sorted(l4_layers.items(), key=lambda x: int(x[0].split('.')[1]) if len(x[0].split('.')) > 1 else -1):
        if k.startswith("blocks"):
            print(f"  {k}: {v:.6f}")
            
    print("\n--- L12 Model Gradients ---")
    l12_total, l12_layers = calculate_grad_norms(12, params)
    print(f"Total Grad Norm: {l12_total:.6f}")
    for k, v in sorted(l12_layers.items(), key=lambda x: int(x[0].split('.')[1]) if len(x[0].split('.')) > 1 else -1):
        if k.startswith("blocks"):
            print(f"  {k}: {v:.6f}")

if __name__ == "__main__":
    compare_gradients()
