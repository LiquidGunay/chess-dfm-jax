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
from chess_dfm_jax.training.jepa import (
    JEPAConfig,
    build_synthetic_transition_batch,
    create_jepa_components,
    train_step,
)

def test_jepa_training_step():
    print("Testing JEPA Training Step (Shapes and Gradients)...")
    
    # We expect models to be in REPO_ROOT/models
    models_dir = REPO_ROOT / "models"
    pb_path = None
    if models_dir.exists():
        for f in models_dir.iterdir():
            if f.name.endswith(".pb.gz") and "exported" not in f.name:
                pb_path = f
                break
    
    if pb_path is None:
        print(f"Warning: No .pb.gz found in {models_dir}. Please supply a valid model for full tests.")
        # We can't really test without weights, because load_mapped_bt4_params requires them.
        # But if this is run on the controller, the weights are there.
        return
        
    print(f"Loading weights from {pb_path}...")
    params = load_mapped_bt4_params(pb=str(pb_path))
    
    print("Initializing JEPA configuration (L=4, H=8)...")
    config = JEPAConfig(token_dim=512, num_layers=4, use_muon=True, use_qk_gain=True)
    
    print("Creating JEPA components (Model and Optimizer)...")
    model, optimizer = create_jepa_components(params, config)
    
    print("Building synthetic batch (Batch=4, Horizon=8)...")
    batch = build_synthetic_transition_batch(batch_size=4, horizon=8)

    print("Validating JEPA sequence output shapes...")
    pred_tokens, target_tokens, q_pred, wdl_pred = model(
        batch["current_planes"],
        batch["action_indices"],
        batch["future_planes"],
    )
    assert pred_tokens.shape == (4, 8, 64, config.token_dim), pred_tokens.shape
    assert target_tokens.shape == (4, 8, 64, config.token_dim), target_tokens.shape
    assert q_pred.shape == (4, 8), q_pred.shape
    assert wdl_pred.shape == (4, 8, 3), wdl_pred.shape

    print("Running forward and backward pass...")
    loss, aux = train_step(model, optimizer, batch)
    
    print("Validating outputs...")
    assert jnp.isfinite(loss), "Loss is not finite!"
    assert "mean_token_cosine" in aux, "Missing mean_token_cosine in aux metrics!"
    assert jnp.isfinite(aux["jepa_loss"]), "JEPA loss is not finite!"
    print(f"Success! Loss: {loss:.6f}, Mean Cosine: {aux['mean_token_cosine']:.4f}")
    
    print("All training step tests passed successfully.")

if __name__ == "__main__":
    test_jepa_training_step()
