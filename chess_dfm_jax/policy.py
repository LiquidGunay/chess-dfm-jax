"""Policy index mapping and legal move masking."""

from __future__ import annotations

import numpy as np
import importlib.resources as resources

try:
    import chess
except ImportError:  # pragma: no cover
    chess = None


def _load_move_list() -> list[str]:
    with resources.files("chess_dfm_jax").joinpath("policy_moves.txt").open("r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def _load_attention_map() -> np.ndarray:
    with resources.files("chess_dfm_jax").joinpath("policy_attn_map.txt").open("r", encoding="utf-8") as f:
        data = [int(line.strip()) for line in f if line.strip()]
    return np.asarray(data, dtype=np.int32)


_MOVE_LIST = _load_move_list()
_MOVE_TO_INDEX = {move: idx for idx, move in enumerate(_MOVE_LIST)}
_ATTN_MAP = _load_attention_map()


def move_to_policy_index(move, policy_format: str) -> int:
    """Map a chess move to a policy index for the current output format."""
    if chess is None:
        raise ImportError("python-chess is required for move mapping.")
    if policy_format not in ("lc0", "lc0_1858", "auto"):
        raise ValueError(f"Unsupported policy_format: {policy_format}")

    if isinstance(move, chess.Move):
        uci = move.uci()
    else:
        uci = str(move)
    if uci not in _MOVE_TO_INDEX:
        raise KeyError(f"Move not in policy map: {uci}")
    return _MOVE_TO_INDEX[uci]


def policy_index_to_move(index: int, policy_format: str):
    """Map a policy index back to a chess move."""
    if chess is None:
        raise ImportError("python-chess is required for move mapping.")
    if policy_format not in ("lc0", "lc0_1858", "auto"):
        raise ValueError(f"Unsupported policy_format: {policy_format}")
    if index < 0 or index >= len(_MOVE_LIST):
        raise IndexError(f"Policy index out of range: {index}")
    return chess.Move.from_uci(_MOVE_LIST[index])


def legal_move_mask(board, policy_format: str) -> np.ndarray:
    """Return a boolean mask over the policy indices."""
    if chess is None:
        raise ImportError("python-chess is required for move masking.")
    if policy_format not in ("lc0", "lc0_1858", "auto"):
        raise ValueError(f"Unsupported policy_format: {policy_format}")

    mask = np.zeros(len(_MOVE_LIST), dtype=bool)
    for move in board.legal_moves:
        uci = move.uci()
        idx = _MOVE_TO_INDEX.get(uci)
        if idx is not None:
            mask[idx] = True
    return mask


def attention_policy_map() -> np.ndarray:
    """Return the attention head mapping table (length 1858)."""
    return _ATTN_MAP.copy()
