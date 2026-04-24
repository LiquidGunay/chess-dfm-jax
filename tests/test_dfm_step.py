import os
import sys
import jax
import jax.numpy as jnp
from pathlib import Path

# Ensure chess_dfm_jax is in path if run standalone
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chess_dfm_jax.analysis.profile_targets import load_mapped_bt4_params
from chess_dfm_jax.training.dfm import create_dfm_components, train_dfm_step, DFMConfig
from chess_dfm_jax.training.jepa import build_synthetic_transition_batch

def test_dfm_training_step():
    print("Testing DFM Training Step (Shapes and Gradients)...")
    
    models_dir = REPO_ROOT / "models"
    pb_path = None
    if models_dir.exists():
        for f in models_dir.iterdir():
            if f.name.endswith(".pb.gz") and "exported" not in f.name:
                pb_path = f
                break
    
    if pb_path is None:
        print(f"Warning: No .pb.gz found in {models_dir}. Cannot test full DFM without weights.")
        return
        
    print(f"Loading weights from {pb_path}...")
    params = load_mapped_bt4_params(pb=str(pb_path))
    
    print("Initializing DFM configuration (L=4, H=8)...")
    config = DFMConfig(token_dim=512, num_layers=4, use_muon=True, use_qk_gain=True)
    
    print("Creating DFM components (Model and Optimizer)...")
    model, optimizer = create_dfm_components(params, config)
    
    print("Building synthetic batch (Batch=4, Horizon=8)...")
    batch = build_synthetic_transition_batch(batch_size=4, horizon=8)
    rng = jax.random.PRNGKey(42)
    
    print("Running forward and backward pass...")
    loss, aux = train_dfm_step(model, optimizer, batch, rng)
    
    print("Validating outputs...")
    assert jnp.isfinite(loss), "Loss is not finite!"
    assert "accuracy" in aux, "Missing accuracy in aux metrics!"
    print(f"Success! Loss: {loss:.6f}, Masked Accuracy: {aux['accuracy']:.4f}")
    
    print("All DFM training step tests passed successfully.")

if __name__ == "__main__":
    test_dfm_training_step()
