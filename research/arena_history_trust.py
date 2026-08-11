"""Sealed attestations for the arena's O(1) internal policy boundary.

Only ``research.play_arena`` mints these packets, after exact opening replay
and immediately after its authoritative full-stack board passes the normal
termination and ply-cap boundary checks.  Model adapters may verify a packet;
they must never manufacture one or treat an ordinary endpoint tuple as trusted.
"""

from __future__ import annotations

from typing import Any


HISTORY_VALIDATION_FULL_REPLAY = "full_replay_v1"
HISTORY_VALIDATION_TRUSTED_ARENA_ENDPOINT = "trusted_arena_endpoint_v1"
HISTORY_VALIDATION_SCHEMA = "chess-dfm-arena-history-validation-v1"
STANDARD_INITIAL_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
SUPPORTED_HISTORY_VALIDATION_MODES = frozenset(
    {
        HISTORY_VALIDATION_FULL_REPLAY,
        HISTORY_VALIDATION_TRUSTED_ARENA_ENDPOINT,
    }
)

_MINT_SEAL = object()


class TrustedArenaHistoryEndpoint:
    """Immutable process-local proof about one authoritative arena position."""

    __slots__ = (
        "schema_version",
        "pair_id",
        "opening_index",
        "game_in_pair",
        "standard_initial_fen",
        "current_fen",
        "position_count",
        "authoritative_move_stack_length",
        "root_ply",
        "additional_plies",
        "played_suffix_count",
        "ply_cap",
        "claim_draw_checked_nonterminal",
        "below_ply_cap",
        "_seal",
        "_frozen",
    )

    def __init__(
        self,
        *,
        pair_id: str,
        opening_index: int,
        game_in_pair: int,
        current_fen: str,
        position_count: int,
        authoritative_move_stack_length: int,
        root_ply: int,
        additional_plies: int,
        played_suffix_count: int,
        ply_cap: int,
        claim_draw_checked_nonterminal: bool,
        below_ply_cap: bool,
        _seal: object,
    ) -> None:
        if _seal is not _MINT_SEAL:
            raise TypeError("TrustedArenaHistoryEndpoint can only be minted by play_arena.")
        values = {
            "schema_version": HISTORY_VALIDATION_SCHEMA,
            "pair_id": pair_id,
            "opening_index": opening_index,
            "game_in_pair": game_in_pair,
            "standard_initial_fen": STANDARD_INITIAL_FEN,
            "current_fen": current_fen,
            "position_count": position_count,
            "authoritative_move_stack_length": authoritative_move_stack_length,
            "root_ply": root_ply,
            "additional_plies": additional_plies,
            "played_suffix_count": played_suffix_count,
            "ply_cap": ply_cap,
            "claim_draw_checked_nonterminal": claim_draw_checked_nonterminal,
            "below_ply_cap": below_ply_cap,
            "_seal": _seal,
            "_frozen": False,
        }
        for name, value in values.items():
            object.__setattr__(self, name, value)
        verify_trusted_arena_history_endpoint(self)
        object.__setattr__(self, "_frozen", True)

    def __setattr__(self, name: str, value: Any) -> None:
        if getattr(self, "_frozen", False):
            raise AttributeError("TrustedArenaHistoryEndpoint is immutable.")
        object.__setattr__(self, name, value)

    def __repr__(self) -> str:
        return (
            "TrustedArenaHistoryEndpoint("
            f"pair_id={self.pair_id!r}, game_in_pair={self.game_in_pair}, "
            f"current_fen={self.current_fen!r}, additional_plies="
            f"{self.additional_plies}, ply_cap={self.ply_cap})"
        )


def _require_nonnegative_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer.")
    return value


def verify_trusted_arena_history_endpoint(
    value: Any,
) -> TrustedArenaHistoryEndpoint:
    """Fail closed unless ``value`` is an intact internally minted packet."""

    if not isinstance(value, TrustedArenaHistoryEndpoint):
        raise TypeError("Expected a sealed TrustedArenaHistoryEndpoint.")
    if value._seal is not _MINT_SEAL:
        raise ValueError("Trusted arena history endpoint seal mismatch.")
    if value.schema_version != HISTORY_VALIDATION_SCHEMA:
        raise ValueError("Trusted arena history endpoint schema mismatch.")
    if not isinstance(value.pair_id, str) or not value.pair_id:
        raise ValueError("Trusted arena pair_id must be non-empty.")
    _require_nonnegative_int(
        value.opening_index,
        name="opening_index",
    )
    game_in_pair = _require_nonnegative_int(
        value.game_in_pair,
        name="game_in_pair",
    )
    if game_in_pair not in (0, 1):
        raise ValueError("game_in_pair must be 0 or 1.")
    position_count = _require_nonnegative_int(
        value.position_count,
        name="position_count",
    )
    move_stack_length = _require_nonnegative_int(
        value.authoritative_move_stack_length,
        name="authoritative_move_stack_length",
    )
    root_ply = _require_nonnegative_int(value.root_ply, name="root_ply")
    additional_plies = _require_nonnegative_int(
        value.additional_plies,
        name="additional_plies",
    )
    played_suffix_count = _require_nonnegative_int(
        value.played_suffix_count,
        name="played_suffix_count",
    )
    ply_cap = _require_nonnegative_int(value.ply_cap, name="ply_cap")
    if ply_cap < 1:
        raise ValueError("ply_cap must be positive.")
    if value.standard_initial_fen != STANDARD_INITIAL_FEN:
        raise ValueError("Trusted arena endpoint has the wrong standard start.")
    if not isinstance(value.current_fen, str) or not value.current_fen:
        raise ValueError("Trusted arena current_fen must be non-empty.")
    if position_count != move_stack_length + 1:
        raise ValueError("Trusted arena position count disagrees with move stack.")
    if move_stack_length != root_ply + additional_plies:
        raise ValueError("Trusted arena root/suffix ply arithmetic mismatch.")
    if played_suffix_count != additional_plies:
        raise ValueError("Trusted arena played suffix count mismatch.")
    if not isinstance(value.claim_draw_checked_nonterminal, bool) or not (
        value.claim_draw_checked_nonterminal
    ):
        raise ValueError("Trusted arena endpoint lacks nonterminal attestation.")
    if not isinstance(value.below_ply_cap, bool) or not value.below_ply_cap:
        raise ValueError("Trusted arena endpoint lacks ply-cap attestation.")
    if additional_plies >= ply_cap:
        raise ValueError("Trusted arena endpoint is not below its ply cap.")
    return value


def _mint_trusted_arena_history_endpoint(
    *,
    pair_id: str,
    opening_index: int,
    game_in_pair: int,
    current_fen: str,
    position_count: int,
    authoritative_move_stack_length: int,
    root_ply: int,
    additional_plies: int,
    played_suffix_count: int,
    ply_cap: int,
) -> TrustedArenaHistoryEndpoint:
    """Mint one packet; callers must have just checked terminal/cap boundaries."""

    return TrustedArenaHistoryEndpoint(
        pair_id=pair_id,
        opening_index=opening_index,
        game_in_pair=game_in_pair,
        current_fen=current_fen,
        position_count=position_count,
        authoritative_move_stack_length=authoritative_move_stack_length,
        root_ply=root_ply,
        additional_plies=additional_plies,
        played_suffix_count=played_suffix_count,
        ply_cap=ply_cap,
        claim_draw_checked_nonterminal=True,
        below_ply_cap=True,
        _seal=_MINT_SEAL,
    )


__all__ = [
    "HISTORY_VALIDATION_FULL_REPLAY",
    "HISTORY_VALIDATION_SCHEMA",
    "HISTORY_VALIDATION_TRUSTED_ARENA_ENDPOINT",
    "STANDARD_INITIAL_FEN",
    "SUPPORTED_HISTORY_VALIDATION_MODES",
    "TrustedArenaHistoryEndpoint",
    "verify_trusted_arena_history_endpoint",
]
