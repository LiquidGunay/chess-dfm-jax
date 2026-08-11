"""Policy index mapping and legal move masking."""

from __future__ import annotations

import importlib.resources as resources
import operator
from typing import Any

import numpy as np

try:
    import chess
except ImportError:  # pragma: no cover
    chess = None


ACTION_CODEC_LEGACY_ABSOLUTE_1858 = "legacy_absolute_1858"
ACTION_CODEC_LC0_CANONICAL_1858 = "lc0_canonical_1858"
LEGACY_ABSOLUTE_1858_CODEC_ID = ACTION_CODEC_LEGACY_ABSOLUTE_1858
LC0_CANONICAL_1858_CODEC_ID = ACTION_CODEC_LC0_CANONICAL_1858
LC0_CANONICAL_1858_INPUT_FORMAT = "INPUT_CLASSICAL_112_PLANE"
ACTION_VOCAB_SIZE = 1858


class ActionCodecError(ValueError):
    """Raised when an action cannot be represented unambiguously by a codec."""


def _load_move_list() -> list[str]:
    with resources.files("chess_dfm_jax").joinpath("policy_moves.txt").open(
        "r",
        encoding="utf-8",
    ) as f:
        return [line.strip() for line in f if line.strip()]


def _load_attention_map() -> np.ndarray:
    with resources.files("chess_dfm_jax").joinpath("policy_attn_map.txt").open(
        "r",
        encoding="utf-8",
    ) as f:
        data = [int(line.strip()) for line in f if line.strip()]
    return np.asarray(data, dtype=np.int32)


_MOVE_LIST = _load_move_list()
_MOVE_TO_INDEX = {move: idx for idx, move in enumerate(_MOVE_LIST)}
_ATTN_MAP = _load_attention_map()

if len(_MOVE_LIST) != ACTION_VOCAB_SIZE:  # pragma: no cover - packaged asset invariant
    raise RuntimeError(
        f"Expected {ACTION_VOCAB_SIZE} policy moves, found {len(_MOVE_LIST)}"
    )


def _vertical_mirror_policy_token(token: str) -> str:
    """Mirror only the square ranks in one shipped policy-list token."""

    if len(token) not in (4, 5):
        raise RuntimeError(f"Invalid shipped policy token: {token!r}")
    try:
        from_rank = str(9 - int(token[1]))
        to_rank = str(9 - int(token[3]))
    except ValueError as exc:  # pragma: no cover - packaged asset invariant
        raise RuntimeError(f"Invalid shipped policy token: {token!r}") from exc
    return token[0] + from_rank + token[2] + to_rank + token[4:]


_LEGACY_TO_CANONICAL_WHITE = np.arange(ACTION_VOCAB_SIZE, dtype=np.int32)
_LEGACY_TO_CANONICAL_BLACK = np.asarray(
    [
        _MOVE_TO_INDEX.get(_vertical_mirror_policy_token(token), -1)
        for token in _MOVE_LIST
    ],
    dtype=np.int32,
)


def legacy_to_lc0_canonical_1858_index_map(
    *,
    black_to_move: bool,
) -> np.ndarray:
    """Map each legacy logit slot to its canonical teacher-logit slot.

    The white map is identity. Black-to-move coordinates are mirrored
    vertically. A value of ``-1`` marks a legacy slot without a canonical
    inverse; these are the canonically oriented promotion suffix slots and
    cannot be legal legacy actions for black.
    """

    mapping = (
        _LEGACY_TO_CANONICAL_BLACK
        if black_to_move
        else _LEGACY_TO_CANONICAL_WHITE
    )
    return mapping.copy()


def _require_chess() -> None:
    if chess is None:
        raise ImportError("python-chess is required for move mapping.")


def _board_from_input(board: Any):
    _require_chess()
    if isinstance(board, chess.Board):
        return board
    try:
        return chess.Board(str(board))
    except (TypeError, ValueError) as exc:
        raise ActionCodecError(f"Invalid chess board: {board!r}") from exc


def _move_from_input(move: Any):
    _require_chess()
    if isinstance(move, chess.Move):
        return move
    try:
        return chess.Move.from_uci(str(move))
    except (TypeError, ValueError) as exc:
        raise ActionCodecError(f"Invalid UCI move: {move!r}") from exc


def _validate_lc0_canonical_input_format(input_format: str) -> None:
    if input_format != LC0_CANONICAL_1858_INPUT_FORMAT:
        raise ActionCodecError(
            f"{ACTION_CODEC_LC0_CANONICAL_1858} is defined only for "
            f"{LC0_CANONICAL_1858_INPUT_FORMAT}, got {input_format!r}"
        )


def _lc0_canonical_token(board, move) -> str:
    """Return the shipped 1,858-list token for one already-legal board move."""

    from_square = move.from_square
    to_square = move.to_square
    if board.turn == chess.BLACK:
        # INPUT_CLASSICAL_112_PLANE represents the side to move as white by
        # vertically mirroring black-to-move positions.
        from_square = chess.square_mirror(from_square)
        to_square = chess.square_mirror(to_square)

    if move.promotion is None:
        suffix = ""
    else:
        promotion_suffix = {
            chess.KNIGHT: "",
            chess.QUEEN: "q",
            chess.ROOK: "r",
            chess.BISHOP: "b",
        }
        suffix = promotion_suffix.get(move.promotion)
        if suffix is None:
            raise ActionCodecError(
                f"Unsupported promotion piece in move {move.uci()}: {move.promotion}"
            )
    return (
        chess.square_name(from_square)
        + chess.square_name(to_square)
        + suffix
    )


def _encode_lc0_canonical_legal_move(board, move) -> int:
    token = _lc0_canonical_token(board, move)
    index = _MOVE_TO_INDEX.get(token)
    if index is None:
        raise ActionCodecError(
            f"Legal move {move.uci()} has no {ACTION_CODEC_LC0_CANONICAL_1858} "
            f"token ({token})"
        )
    return index


def move_to_policy_index(move, policy_format: str) -> int:
    """Map absolute UCI coordinates using the legacy checkpoint/data codec.

    This boardless API intentionally retains ``legacy_absolute_1858``
    semantics. It does not mirror black-to-move coordinates and cannot encode
    knight promotions. New model inference should use
    :func:`encode_lc0_canonical_1858` with the current board.
    """

    _require_chess()
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
    """Decode an absolute move using the legacy checkpoint/data codec."""

    _require_chess()
    if policy_format not in ("lc0", "lc0_1858", "auto"):
        raise ValueError(f"Unsupported policy_format: {policy_format}")
    if index < 0 or index >= len(_MOVE_LIST):
        raise IndexError(f"Policy index out of range: {index}")
    return chess.Move.from_uci(_MOVE_LIST[index])


def legal_move_mask(board, policy_format: str) -> np.ndarray:
    """Return a legacy absolute-coordinate mask for checkpoint compatibility."""

    _require_chess()
    if policy_format not in ("lc0", "lc0_1858", "auto"):
        raise ValueError(f"Unsupported policy_format: {policy_format}")

    mask = np.zeros(len(_MOVE_LIST), dtype=bool)
    for move in board.legal_moves:
        uci = move.uci()
        idx = _MOVE_TO_INDEX.get(uci)
        if idx is not None:
            mask[idx] = True
    return mask


def encode_lc0_canonical_1858(
    board,
    move,
    *,
    input_format: str = LC0_CANONICAL_1858_INPUT_FORMAT,
) -> int:
    """Encode one legal move in the side-to-move frame used by classical BT4.

    Black-to-move squares are mirrored vertically. Knight promotion occupies
    the ordinary from/to slot; queen, rook, and bishop promotions use the
    shipped ``q``, ``r``, and ``b`` suffix slots respectively.
    """

    _validate_lc0_canonical_input_format(input_format)
    board_obj = _board_from_input(board)
    move_obj = _move_from_input(move)
    if not board_obj.is_legal(move_obj):
        raise ActionCodecError(
            f"Cannot encode illegal move {move_obj.uci()} in {board_obj.fen()}"
        )
    return _encode_lc0_canonical_legal_move(board_obj, move_obj)


def decode_lc0_canonical_1858(
    board,
    index: int,
    *,
    input_format: str = LC0_CANONICAL_1858_INPUT_FORMAT,
):
    """Decode an index only when it identifies exactly one board-legal move."""

    _validate_lc0_canonical_input_format(input_format)
    board_obj = _board_from_input(board)
    try:
        index = operator.index(index)
    except TypeError as exc:
        raise ActionCodecError(f"Invalid policy index: {index!r}") from exc
    if index < 0 or index >= ACTION_VOCAB_SIZE:
        raise ActionCodecError(f"Policy index out of range: {index}")

    matches = [
        move
        for move in board_obj.legal_moves
        if _encode_lc0_canonical_legal_move(board_obj, move) == index
    ]
    if len(matches) != 1:
        raise ActionCodecError(
            f"Policy index {index} has {len(matches)} legal decodes in {board_obj.fen()}"
        )
    return matches[0]


def legal_mask_lc0_canonical_1858(
    board,
    *,
    input_format: str = LC0_CANONICAL_1858_INPUT_FORMAT,
) -> np.ndarray:
    """Return the complete canonical mask, rejecting missing or colliding moves."""

    _validate_lc0_canonical_input_format(input_format)
    board_obj = _board_from_input(board)
    mask = np.zeros(ACTION_VOCAB_SIZE, dtype=bool)
    owners: dict[int, Any] = {}
    for move in board_obj.legal_moves:
        index = _encode_lc0_canonical_legal_move(board_obj, move)
        previous = owners.get(index)
        if previous is not None and previous != move:
            raise ActionCodecError(
                f"Legal moves {previous.uci()} and {move.uci()} collide at index {index}"
            )
        owners[index] = move
        mask[index] = True
    return mask


def encode_action(
    move,
    *,
    codec_id: str,
    board=None,
    input_format: str = LC0_CANONICAL_1858_INPUT_FORMAT,
) -> int:
    """Encode through an explicit, persisted action codec identifier."""

    if codec_id == ACTION_CODEC_LEGACY_ABSOLUTE_1858:
        return move_to_policy_index(move, "lc0_1858")
    if codec_id == ACTION_CODEC_LC0_CANONICAL_1858:
        if board is None:
            raise ActionCodecError(f"{codec_id} requires the current board")
        return encode_lc0_canonical_1858(
            board,
            move,
            input_format=input_format,
        )
    raise ActionCodecError(f"Unsupported action codec: {codec_id!r}")


def decode_action(
    index: int,
    *,
    codec_id: str,
    board=None,
    input_format: str = LC0_CANONICAL_1858_INPUT_FORMAT,
):
    """Decode through an explicit, persisted action codec identifier."""

    if codec_id == ACTION_CODEC_LEGACY_ABSOLUTE_1858:
        return policy_index_to_move(index, "lc0_1858")
    if codec_id == ACTION_CODEC_LC0_CANONICAL_1858:
        if board is None:
            raise ActionCodecError(f"{codec_id} requires the current board")
        return decode_lc0_canonical_1858(
            board,
            index,
            input_format=input_format,
        )
    raise ActionCodecError(f"Unsupported action codec: {codec_id!r}")


def legal_action_mask(
    board,
    *,
    codec_id: str,
    input_format: str = LC0_CANONICAL_1858_INPUT_FORMAT,
) -> np.ndarray:
    """Build a legal mask through an explicit, persisted codec identifier."""

    if codec_id == ACTION_CODEC_LEGACY_ABSOLUTE_1858:
        return legal_move_mask(board, "lc0_1858")
    if codec_id == ACTION_CODEC_LC0_CANONICAL_1858:
        return legal_mask_lc0_canonical_1858(
            board,
            input_format=input_format,
        )
    raise ActionCodecError(f"Unsupported action codec: {codec_id!r}")


def attention_policy_map() -> np.ndarray:
    """Return the attention head mapping table (length 1858)."""
    return _ATTN_MAP.copy()


__all__ = [
    "ACTION_CODEC_LC0_CANONICAL_1858",
    "ACTION_CODEC_LEGACY_ABSOLUTE_1858",
    "ACTION_VOCAB_SIZE",
    "ActionCodecError",
    "LC0_CANONICAL_1858_CODEC_ID",
    "LC0_CANONICAL_1858_INPUT_FORMAT",
    "LEGACY_ABSOLUTE_1858_CODEC_ID",
    "attention_policy_map",
    "decode_action",
    "decode_lc0_canonical_1858",
    "encode_action",
    "encode_lc0_canonical_1858",
    "legal_action_mask",
    "legal_mask_lc0_canonical_1858",
    "legal_move_mask",
    "legacy_to_lc0_canonical_1858_index_map",
    "move_to_policy_index",
    "policy_index_to_move",
]
