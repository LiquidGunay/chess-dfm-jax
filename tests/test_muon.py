import sys
import jax
import jax.numpy as jnp
from flax import nnx
import optax
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chess_dfm_jax.nnx_bt4 import muon_adamw, TrainableParam

class SimpleModel(nnx.Module):
    def __init__(self, rngs):
        # Muon applies to 2D parameters
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

if __name__ == "__main__":
    test_muon()
