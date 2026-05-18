import dataclasses
import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chess_dfm_jax.training.dfm import refine_actions_from_current  # noqa: E402


@dataclasses.dataclass
class _Config:
    horizon: int = 4
    action_vocab_size: int = 8


class _FakeDFM:
    def __init__(self):
        self.config = _Config()
        self.encode_count = 0
        self.planner_count = 0

    def encode_current(self, current_planes):
        self.encode_count += 1
        return jnp.zeros((current_planes.shape[0], 64, 4), dtype=jnp.float32)

    def planner_from_latents(self, z_dfm, action_tokens, t):
        self.planner_count += 1
        batch_size = z_dfm.shape[0]
        logits = jnp.full(
            (batch_size, self.config.horizon, self.config.action_vocab_size),
            -10.0,
            dtype=jnp.float32,
        )
        preferred = jnp.arange(self.config.horizon, dtype=jnp.int32) + 1
        return logits.at[:, jnp.arange(self.config.horizon), preferred].set(10.0)


def test_refine_actions_from_current_caches_board_encode():
    model = _FakeDFM()
    planes = jnp.zeros((2, 112, 8, 8), dtype=jnp.float32)
    legal_mask = jnp.zeros((2, model.config.action_vocab_size), dtype=bool).at[:, 1].set(True)

    actions = refine_actions_from_current(model, planes, legal_mask, refinement_steps=4)

    np.testing.assert_array_equal(np.asarray(actions), np.asarray([[1, 2, 3, 4], [1, 2, 3, 4]]))
    assert model.encode_count == 1
    assert model.planner_count == 4
