import os
import sys
import jax
import jax.numpy as jnp
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chess_dfm_jax.analysis.profile_targets import load_mapped_bt4_params  # noqa: E402
from chess_dfm_jax.training.dfm import create_dfm_components, train_dfm_step, DFMConfig  # noqa: E402
from chess_dfm_jax.training.jepa import build_synthetic_transition_batch  # noqa: E402

def test_overfit():
    if os.environ.get("CHESS_DFM_RUN_SLOW") != "1":
        pytest.skip("Set CHESS_DFM_RUN_SLOW=1 to run the BT4-backed overfit diagnostic.")

    print("Testing DFM Overfitting on a single batch...")
    
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
    # Test with standard Transformer, no XSA, no Muon
    config = DFMConfig(token_dim=512, num_layers=2, use_muon=True, use_qk_gain=True, learning_rate=1e-4)
    model, optimizer = create_dfm_components(params, config)
    
    batch = build_synthetic_transition_batch(batch_size=4, horizon=8)
    batch["current_planes"] = jax.random.normal(jax.random.PRNGKey(42), (4, 112, 8, 8))
    # Give it random actions to prevent trivial all-zeros memorization
    batch["action_indices"] = jax.random.randint(jax.random.PRNGKey(123), (4, 8), 0, 1858)
    batch["deterministic_t"] = jnp.zeros(())
    rng = jax.random.PRNGKey(42)
    
    for i in range(1000):
        rng, step_rng = jax.random.split(rng)
        loss, aux = train_dfm_step(model, optimizer, batch, step_rng)
        if i % 100 == 0:
            print(f"Step {i} | Loss: {loss:.4f} | Acc: {aux['accuracy']:.4f} | Mask: {aux['mask_prob']:.4f}")

if __name__ == "__main__":
    test_overfit()
