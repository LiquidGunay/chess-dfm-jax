"""LC0 input plane encoder."""

from __future__ import annotations

from typing import Iterable

import numpy as np

try:
    import chess
except ImportError:  # pragma: no cover
    chess = None

MOVE_HISTORY = 8
PLANES_PER_BOARD = 13
AUX_PLANE_BASE = MOVE_HISTORY * PLANES_PER_BOARD
TOTAL_PLANES = AUX_PLANE_BASE + 8
ALL_ONES_MASK = (1 << 64) - 1

INPUT_FORMAT_ALIASES = {
    "INPUT_CLASSICAL_112_PLANE": "classical_112",
    "INPUT_112_WITH_CASTLING_PLANE": "castling_plane",
    "INPUT_112_WITH_CANONICALIZATION": "canonical",
    "INPUT_112_WITH_CANONICALIZATION_HECTOPLIES": "canonical_hectoplies",
    "INPUT_112_WITH_CANONICALIZATION_HECTOPLIES_ARMAGEDDON": "canonical_hectoplies_armageddon",
    "INPUT_112_WITH_CANONICALIZATION_V2": "canonical_v2",
    "INPUT_112_WITH_CANONICALIZATION_V2_ARMAGEDDON": "canonical_v2_armageddon",
}


def _normalize_input_format(fmt: str) -> str:
    if fmt in INPUT_FORMAT_ALIASES:
        return INPUT_FORMAT_ALIASES[fmt]
    return fmt


def _is_canonical(fmt: str) -> bool:
    return fmt in {
        "canonical",
        "canonical_hectoplies",
        "canonical_hectoplies_armageddon",
        "canonical_v2",
        "canonical_v2_armageddon",
    }


def _is_canonical_armageddon(fmt: str) -> bool:
    return fmt in {"canonical_hectoplies_armageddon", "canonical_v2_armageddon"}


def _is_hectoplies(fmt: str) -> bool:
    return fmt in {"canonical_hectoplies", "canonical_hectoplies_armageddon"}


def _is_castling_plane(fmt: str) -> bool:
    return fmt != "classical_112"


def _is_v2(fmt: str) -> bool:
    return fmt in {"canonical_v2", "canonical_v2_armageddon"}


def _reverse_bits_in_bytes(v: int) -> int:
    v = ((v >> 1) & 0x5555555555555555) | ((v & 0x5555555555555555) << 1)
    v = ((v >> 2) & 0x3333333333333333) | ((v & 0x3333333333333333) << 2)
    v = ((v >> 4) & 0x0F0F0F0F0F0F0F0F) | ((v & 0x0F0F0F0F0F0F0F0F) << 4)
    return v & ALL_ONES_MASK


def _reverse_bytes_in_bytes(v: int) -> int:
    v = ((v & 0x00000000FFFFFFFF) << 32) | ((v & 0xFFFFFFFF00000000) >> 32)
    v = ((v & 0x0000FFFF0000FFFF) << 16) | ((v & 0xFFFF0000FFFF0000) >> 16)
    v = ((v & 0x00FF00FF00FF00FF) << 8) | ((v & 0xFF00FF00FF00FF00) >> 8)
    return v & ALL_ONES_MASK


def _transpose_bits_in_bytes(v: int) -> int:
    v = (
        (v & 0xAA00AA00AA00AA00) >> 9
        | (v & 0x0055005500550055) << 9
        | (v & 0x55AA55AA55AA55AA)
    )
    v = (
        (v & 0xCCCC0000CCCC0000) >> 18
        | (v & 0x0000333300003333) << 18
        | (v & 0x3333CCCC3333CCCC)
    )
    v = (
        (v & 0xF0F0F0F000000000) >> 36
        | (v & 0x000000000F0F0F0F) << 36
        | (v & 0x0F0F0F0FF0F0F0F0)
    )
    return v & ALL_ONES_MASK


def _transform_mask(mask: int, transform: int) -> int:
    if transform & 1:
        mask = _reverse_bits_in_bytes(mask)
    if transform & 2:
        mask = _reverse_bytes_in_bytes(mask)
    if transform & 4:
        mask = _transpose_bits_in_bytes(mask)
    return mask


def _choose_transform(our_king: int, ours: int, theirs: int, kings: int, queens: int, rooks: int, knights: int, bishops: int, pawns: int, castling_any: bool) -> int:
    # Port of ChooseTransform from LC0 encoder.cc.
    if castling_any:
        return 0

    transform = 0
    if our_king & 0x0F0F0F0F0F0F0F0F:
        transform |= 1
        our_king = _reverse_bits_in_bytes(our_king)

    if pawns != 0:
        return transform

    if our_king & 0xFFFFFFFF00000000:
        transform |= 2
        our_king = _reverse_bytes_in_bytes(our_king)

    if our_king & 0xE0C08000:
        transform |= 4
    elif our_king & 0x10204080:
        if _compare_transpose(ours | theirs, transform) == 1:
            transform |= 4
        elif _compare_transpose(ours | theirs, transform) == -1:
            return transform
        if _compare_transpose(ours, transform) == 1:
            transform |= 4
        elif _compare_transpose(ours, transform) == -1:
            return transform
        if _compare_transpose(kings, transform) == 1:
            transform |= 4
        elif _compare_transpose(kings, transform) == -1:
            return transform
        if _compare_transpose(queens, transform) == 1:
            transform |= 4
        elif _compare_transpose(queens, transform) == -1:
            return transform
        if _compare_transpose(rooks, transform) == 1:
            transform |= 4
        elif _compare_transpose(rooks, transform) == -1:
            return transform
        if _compare_transpose(knights, transform) == 1:
            transform |= 4
        elif _compare_transpose(knights, transform) == -1:
            return transform
        if _compare_transpose(bishops, transform) == 1:
            transform |= 4
        elif _compare_transpose(bishops, transform) == -1:
            return transform
    return transform


def _compare_transpose(mask: int, initial_transform: int) -> int:
    value = mask
    if initial_transform & 1:
        value = _reverse_bits_in_bytes(value)
    if initial_transform & 2:
        value = _reverse_bytes_in_bytes(value)
    alt = _transpose_bits_in_bytes(value)
    if value < alt:
        return -1
    if value > alt:
        return 1
    return 0


def _mask_to_plane(mask: int) -> np.ndarray:
    plane = np.zeros((8, 8), dtype=np.float32)
    for sq in range(64):
        if (mask >> sq) & 1:
            plane[sq // 8, sq % 8] = 1.0
    return plane


def _board_from_input(board):
    if chess is None:
        raise ImportError("python-chess is required for encoding.")
    if isinstance(board, chess.Board):
        return board
    return chess.Board(board)


def _oriented_bitboards(board: "chess.Board") -> dict:
    # Returns bitboards for the side-to-move perspective (mirrored for black).
    w = chess.WHITE
    b = chess.BLACK
    pawns_w = int(board.pieces(chess.PAWN, w))
    knights_w = int(board.pieces(chess.KNIGHT, w))
    bishops_w = int(board.pieces(chess.BISHOP, w))
    rooks_w = int(board.pieces(chess.ROOK, w))
    queens_w = int(board.pieces(chess.QUEEN, w))
    kings_w = int(board.pieces(chess.KING, w))

    pawns_b = int(board.pieces(chess.PAWN, b))
    knights_b = int(board.pieces(chess.KNIGHT, b))
    bishops_b = int(board.pieces(chess.BISHOP, b))
    rooks_b = int(board.pieces(chess.ROOK, b))
    queens_b = int(board.pieces(chess.QUEEN, b))
    kings_b = int(board.pieces(chess.KING, b))

    if board.turn == chess.WHITE:
        ours = {
            "pawns": pawns_w,
            "knights": knights_w,
            "bishops": bishops_w,
            "rooks": rooks_w,
            "queens": queens_w,
            "kings": kings_w,
        }
        theirs = {
            "pawns": pawns_b,
            "knights": knights_b,
            "bishops": bishops_b,
            "rooks": rooks_b,
            "queens": queens_b,
            "kings": kings_b,
        }
    else:
        ours = {
            "pawns": _reverse_bytes_in_bytes(pawns_b),
            "knights": _reverse_bytes_in_bytes(knights_b),
            "bishops": _reverse_bytes_in_bytes(bishops_b),
            "rooks": _reverse_bytes_in_bytes(rooks_b),
            "queens": _reverse_bytes_in_bytes(queens_b),
            "kings": _reverse_bytes_in_bytes(kings_b),
        }
        theirs = {
            "pawns": _reverse_bytes_in_bytes(pawns_w),
            "knights": _reverse_bytes_in_bytes(knights_w),
            "bishops": _reverse_bytes_in_bytes(bishops_w),
            "rooks": _reverse_bytes_in_bytes(rooks_w),
            "queens": _reverse_bytes_in_bytes(queens_w),
            "kings": _reverse_bytes_in_bytes(kings_w),
        }

    ep_mask = 0
    if board.ep_square is not None:
        ep_mask = 1 << board.ep_square
        if board.turn == chess.BLACK:
            ep_mask = _reverse_bytes_in_bytes(ep_mask)

    return {
        "ours": ours,
        "theirs": theirs,
        "ep_mask": ep_mask,
    }


def _mirror_state(state: dict) -> dict:
    # Mirror (swap colors + vertical flip) for LC0 history alignment.
    ours = state["ours"]
    theirs = state["theirs"]
    def flip(bb):
        return _reverse_bytes_in_bytes(bb)

    new_ours = {k: flip(theirs[k]) for k in theirs}
    new_theirs = {k: flip(ours[k]) for k in ours}
    ep_mask = flip(state["ep_mask"]) if state["ep_mask"] else 0
    return {"ours": new_ours, "theirs": new_theirs, "ep_mask": ep_mask}


def _castling_rights(board: "chess.Board") -> dict:
    w = chess.WHITE
    b = chess.BLACK
    return {
        "we_oo": board.has_kingside_castling_rights(board.turn),
        "we_ooo": board.has_queenside_castling_rights(board.turn),
        "they_oo": board.has_kingside_castling_rights(not board.turn),
        "they_ooo": board.has_queenside_castling_rights(not board.turn),
    }


def _swap_castling(castling: dict) -> dict:
    return {
        "we_oo": castling["they_oo"],
        "we_ooo": castling["they_ooo"],
        "they_oo": castling["we_oo"],
        "they_ooo": castling["we_ooo"],
    }


def _castling_mask(castling: dict, for_kingside: bool) -> int:
    mask = 0
    if for_kingside:
        if castling["we_oo"]:
            mask |= 1 << 7  # h1
        if castling["they_oo"]:
            mask |= 1 << 63  # h8
    else:
        if castling["we_ooo"]:
            mask |= 1 << 0  # a1
        if castling["they_ooo"]:
            mask |= 1 << 56  # a8
    return mask


def _repetition_counts(history_boards: list["chess.Board"]) -> list[int]:
    counts: dict[str, int] = {}
    reps: list[int] = []
    for board in history_boards:
        key = _repetition_key(board)
        reps.append(counts.get(key, 0))
        counts[key] = counts.get(key, 0) + 1
    return reps


def _repetition_key(board: "chess.Board") -> str:
    ep = "-" if board.ep_square is None else chess.square_name(board.ep_square)
    castling = board.castling_xfen()
    turn = "w" if board.turn == chess.WHITE else "b"
    return f"{board.board_fen()} {turn} {castling} {ep}"


def encode_board(board, history: Iterable, *, planes_layout: str = "nchw", input_format: str = "INPUT_CLASSICAL_112_PLANE") -> np.ndarray:
    """Return LC0 input planes as float32, shape [C, 8, 8]."""
    if chess is None:
        raise ImportError("python-chess is required for encoding.")
    fmt = _normalize_input_format(input_format)

    board_obj = _board_from_input(board)
    history_list = [board_obj] if not history else [_board_from_input(h) for h in history]
    if history_list[-1].fen() != board_obj.fen():
        history_list.append(board_obj)

    reps = _repetition_counts(history_list)
    stop_early = _is_canonical(fmt)
    skip_non_repeats = _is_v2(fmt)

    # Aux planes
    current_state = _oriented_bitboards(history_list[-1])
    castling = _castling_rights(history_list[-1])
    castling_any = any(castling.values())
    our_king_mask = current_state["ours"]["kings"]
    ours_mask = (
        current_state["ours"]["pawns"]
        | current_state["ours"]["knights"]
        | current_state["ours"]["bishops"]
        | current_state["ours"]["rooks"]
        | current_state["ours"]["queens"]
        | current_state["ours"]["kings"]
    )
    theirs_mask = (
        current_state["theirs"]["pawns"]
        | current_state["theirs"]["knights"]
        | current_state["theirs"]["bishops"]
        | current_state["theirs"]["rooks"]
        | current_state["theirs"]["queens"]
        | current_state["theirs"]["kings"]
    )
    pieces = {
        "kings": current_state["ours"]["kings"] | current_state["theirs"]["kings"],
        "queens": current_state["ours"]["queens"] | current_state["theirs"]["queens"],
        "rooks": current_state["ours"]["rooks"] | current_state["theirs"]["rooks"],
        "knights": current_state["ours"]["knights"] | current_state["theirs"]["knights"],
        "bishops": current_state["ours"]["bishops"] | current_state["theirs"]["bishops"],
        "pawns": current_state["ours"]["pawns"] | current_state["theirs"]["pawns"],
    }
    transform = 0
    if _is_canonical(fmt):
        transform = _choose_transform(
            our_king_mask,
            ours_mask,
            theirs_mask,
            pieces["kings"],
            pieces["queens"],
            pieces["rooks"],
            pieces["knights"],
            pieces["bishops"],
            pieces["pawns"],
            castling_any,
        )

    plane_masks: list[int] = [0 for _ in range(TOTAL_PLANES)]
    plane_values: list[float] = [1.0 for _ in range(TOTAL_PLANES)]

    if fmt == "classical_112":
        plane_masks[AUX_PLANE_BASE + 0] = ALL_ONES_MASK if castling["we_ooo"] else 0
        plane_masks[AUX_PLANE_BASE + 1] = ALL_ONES_MASK if castling["we_oo"] else 0
        plane_masks[AUX_PLANE_BASE + 2] = ALL_ONES_MASK if castling["they_ooo"] else 0
        plane_masks[AUX_PLANE_BASE + 3] = ALL_ONES_MASK if castling["they_oo"] else 0
    else:
        plane_masks[AUX_PLANE_BASE + 0] = _castling_mask(castling, for_kingside=False)
        plane_masks[AUX_PLANE_BASE + 1] = _castling_mask(castling, for_kingside=True)

    if _is_canonical(fmt):
        plane_masks[AUX_PLANE_BASE + 4] = current_state["ep_mask"]
    else:
        plane_masks[AUX_PLANE_BASE + 4] = ALL_ONES_MASK if board_obj.turn == chess.BLACK else 0

    rule50 = board_obj.halfmove_clock
    plane_masks[AUX_PLANE_BASE + 5] = ALL_ONES_MASK
    plane_values[AUX_PLANE_BASE + 5] = rule50 / 100.0 if _is_hectoplies(fmt) else float(rule50)

    if _is_canonical_armageddon(fmt) and board_obj.turn == chess.BLACK:
        plane_masks[AUX_PLANE_BASE + 6] = ALL_ONES_MASK
    else:
        plane_masks[AUX_PLANE_BASE + 6] = 0

    plane_masks[AUX_PLANE_BASE + 7] = ALL_ONES_MASK

    # History planes
    castling_ref = castling if stop_early else None
    history_idx = len(history_list) - 1
    flip = False
    i = 0
    while i < min(MOVE_HISTORY, len(history_list)):
        if history_idx < 0:
            break
        position = history_list[history_idx]
        state = _oriented_bitboards(position)
        castling_state = _castling_rights(position)
        if flip:
            state = _mirror_state(state)
            castling_state = _swap_castling(castling_state)

        if stop_early and castling_state != castling_ref:
            break
        if stop_early and history_idx != len(history_list) - 1 and state["ep_mask"]:
            break

        repetitions = reps[history_idx]
        if skip_non_repeats and repetitions == 0 and i > 0:
            if history_idx > 0:
                flip = not flip
            if position.halfmove_clock == 0:
                break
            history_idx -= 1
            continue

        base = i * PLANES_PER_BOARD
        plane_masks[base + 0] = state["ours"]["pawns"]
        plane_masks[base + 1] = state["ours"]["knights"]
        plane_masks[base + 2] = state["ours"]["bishops"]
        plane_masks[base + 3] = state["ours"]["rooks"]
        plane_masks[base + 4] = state["ours"]["queens"]
        plane_masks[base + 5] = state["ours"]["kings"]

        plane_masks[base + 6] = state["theirs"]["pawns"]
        plane_masks[base + 7] = state["theirs"]["knights"]
        plane_masks[base + 8] = state["theirs"]["bishops"]
        plane_masks[base + 9] = state["theirs"]["rooks"]
        plane_masks[base + 10] = state["theirs"]["queens"]
        plane_masks[base + 11] = state["theirs"]["kings"]

        if repetitions >= 1:
            plane_masks[base + 12] = ALL_ONES_MASK
        else:
            plane_masks[base + 12] = 0

        if history_idx > 0:
            flip = not flip
        if stop_early and position.halfmove_clock == 0:
            break

        history_idx -= 1
        i += 1

    # Apply canonicalization transform to applicable planes.
    if transform != 0:
        for idx in range(AUX_PLANE_BASE + 5):
            mask = plane_masks[idx]
            if mask in (0, ALL_ONES_MASK):
                continue
            plane_masks[idx] = _transform_mask(mask, transform)

    planes = np.zeros((TOTAL_PLANES, 8, 8), dtype=np.float32)
    for idx, mask in enumerate(plane_masks):
        if mask == 0 and plane_values[idx] == 1.0:
            continue
        if mask == ALL_ONES_MASK and plane_values[idx] != 1.0:
            planes[idx, :, :] = plane_values[idx]
        else:
            plane = _mask_to_plane(mask)
            if plane_values[idx] != 1.0:
                plane *= plane_values[idx]
            planes[idx] = plane

    if planes_layout == "nchw":
        return planes
    if planes_layout == "nhwc":
        return np.transpose(planes, (1, 2, 0))
    raise ValueError(f"Unsupported planes_layout: {planes_layout}")
