# ruff: noqa: E402
import sys
from pathlib import Path

import chess
import jax.numpy as jnp

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chess_dfm_jax.analysis.profile_targets import load_mapped_bt4_params  # noqa: E402
from chess_dfm_jax.encoding import encode_board  # noqa: E402
from chess_dfm_jax.nnx_bt4 import make_bt4_model  # noqa: E402


def test_bt4_manual_and_sdpa_attention_match_with_smolgen():
    models_dir = REPO_ROOT / "models"
    pb_path = None
    if models_dir.exists():
        for candidate in models_dir.iterdir():
            if candidate.name.endswith(".pb.gz") and "exported" not in candidate.name:
                pb_path = candidate
                break
    if pb_path is None:
        return

    params = load_mapped_bt4_params(pb=str(pb_path))
    manual = make_bt4_model(params, dtype=jnp.float32, attention_impl="manual")
    sdpa = make_bt4_model(params, dtype=jnp.float32, attention_impl="sdpa")
    planes = jnp.asarray(
        encode_board(chess.Board(), [], input_format="INPUT_CLASSICAL_112_PLANE"),
        dtype=jnp.float32,
    )[None]

    manual_tokens = manual.encode_tokens(planes)
    sdpa_tokens = sdpa.encode_tokens(planes)
    assert float(jnp.max(jnp.abs(manual_tokens - sdpa_tokens))) < 5e-4

    manual_policy, manual_value, manual_moves_left = manual(planes)
    sdpa_policy, sdpa_value, sdpa_moves_left = sdpa(planes)
    assert float(jnp.max(jnp.abs(manual_policy - sdpa_policy))) < 5e-4
    assert float(jnp.max(jnp.abs(manual_value - sdpa_value))) < 5e-4
    assert float(jnp.max(jnp.abs(manual_moves_left - sdpa_moves_left))) < 5e-4
