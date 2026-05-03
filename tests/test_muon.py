import sys
import jax
import jax.numpy as jnp
from flax import nnx
import optax
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chess_dfm_jax.nnx_bt4 import (
    TrainableParam,
    _muon_dimension_numbers_for_params,
    muon_adamw,
)

class SimpleModel(nnx.Module):
    def __init__(self, rngs):
        # The public optimizer should handle both Muon-routed and AdamW-routed
        # leaves without requiring model-side special cases.
        self.w2d = TrainableParam(jax.random.normal(rngs.params(), (4, 4)))
        # AdamW applies to 1D parameters
        self.b1d = TrainableParam(jnp.zeros((4,)))
        
    def __call__(self, x):
        return x @ self.w2d[...] + self.b1d[...]

def test_muon():
    model = SimpleModel(nnx.Rngs(0))
    tx = muon_adamw(learning_rate=0.01, weight_decay=1e-4)
    optimizer = nnx.Optimizer(model, tx, wrt=TrainableParam)
    
    # Dummy loss
    def loss_fn(model, x, y):
        pred = model(x)
        return jnp.mean(jnp.square(pred - y))
    
    x = jax.random.normal(jax.random.PRNGKey(1), (2, 4))
    y = jax.random.normal(jax.random.PRNGKey(2), (2, 4))
    
    loss, grads = nnx.value_and_grad(loss_fn)(model, x, y)
    optimizer.update(model, grads)
    
    print("Muon optimizer successfully applied a gradient step!")
    print(f"Loss: {loss:.4f}")


def test_muon_routing_is_conservative():
    labels = _muon_dimension_numbers_for_params(
        {
            "attn": {"wq": jnp.zeros((256, 256))},
            "stacked": {"wo": jnp.zeros((4, 256, 256))},
            "ffn": {"expand": jnp.zeros((256, 1024)), "contract": jnp.zeros((1024, 256))},
            "embedding": jnp.zeros((1859, 256)),
            "norm": {"scale": jnp.zeros((256,))},
            "small": jnp.zeros((4, 4)),
        }
    )

    assert isinstance(labels["attn"]["wq"], optax.contrib.MuonDimensionNumbers)
    assert isinstance(labels["stacked"]["wo"], optax.contrib.MuonDimensionNumbers)
    assert labels["stacked"]["wo"].reduction_axis == 1
    assert labels["stacked"]["wo"].output_axis == 2
    assert labels["ffn"]["expand"] is None
    assert labels["ffn"]["contract"] is None
    assert labels["embedding"] is None
    assert labels["norm"]["scale"] is None
    assert labels["small"] is None

if __name__ == "__main__":
    test_muon()
