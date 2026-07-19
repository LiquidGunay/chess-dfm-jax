"""Fail-closed arena adapter for the original searchless BT4 policy head.

The localized DFM checkpoint and the original BT4 policy head do not share an
action codec.  DFM checkpoints retain the trajectory data's boardless
``legacy_absolute_1858`` labels.  The native BT4 head emits
``lc0_canonical_1858`` logits in side-to-move coordinates, so black moves must
be decoded with the current board.

This first strength anchor deliberately uses the same current-only classical
planes as the frozen trajectory and arena contract.  Supplying historical
planes is a separate distribution-changing experiment.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable
from typing import Any

import chess
import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from chess_dfm_jax.encoding import TOTAL_PLANES, encode_board
from chess_dfm_jax.nnx_bt4 import BT4Model
from chess_dfm_jax.policy import (
    ACTION_CODEC_LC0_CANONICAL_1858,
    ACTION_VOCAB_SIZE,
    LC0_CANONICAL_1858_INPUT_FORMAT,
    ActionCodecError,
    decode_action,
    encode_action,
    legal_action_mask,
)
from research.arena_history_trust import (
    HISTORY_VALIDATION_FULL_REPLAY,
    HISTORY_VALIDATION_SCHEMA,
    HISTORY_VALIDATION_TRUSTED_ARENA_ENDPOINT,
    TrustedArenaHistoryEndpoint,
)
from research.local_policy import (
    PLANE_HISTORY_MODE_CURRENT_ONLY_AS_PREPROCESSED,
    STATIC_INFERENCE_BATCHING_SCHEMA,
    STATIC_INFERENCE_PADDING_MODE,
    LocalPolicyBatchResult,
    LocalPolicyError,
    _materialize_batch,
    _pad_validated_inference_inputs,
    _require_integer,
    _require_standard_board,
    _validate_history,
    _validate_history_validation_mode,
    _validate_inference_batch_size,
    _validate_trusted_arena_history_endpoint,
    _readonly,
)


RAW_BT4_POLICY_COMPUTE_DTYPE = "bfloat16"


def _raw_bt4_policy_logits(
    model: BT4Model,
    planes: jax.Array,
) -> jax.Array:
    tokens = model.encode_tokens(planes)
    return model.policy_head(tokens)


@nnx.jit
def infer_raw_bt4_actions(
    model: BT4Model,
    planes: jax.Array,
    legal_mask: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Return greedy canonical actions and per-row finite-logit attestations."""

    logits = _raw_bt4_policy_logits(model, planes)
    finite = jnp.all(jnp.isfinite(logits), axis=-1)
    masked_logits = jnp.where(legal_mask, logits, -jnp.inf)
    actions = jnp.argmax(masked_logits, axis=-1).astype(jnp.int32)
    return actions, finite


@dataclasses.dataclass(frozen=True)
class LocalBT4Policy:
    """Checkpoint-free adapter around one original BT4 model."""

    model: BT4Model
    model_id: str
    inference_batch_size: int | None = None
    history_validation_mode: str = HISTORY_VALIDATION_FULL_REPLAY
    action_codec_id: str = dataclasses.field(
        default=ACTION_CODEC_LC0_CANONICAL_1858,
        init=False,
    )
    input_format: str = dataclasses.field(
        default=LC0_CANONICAL_1858_INPUT_FORMAT,
        init=False,
    )
    plane_history_mode: str = dataclasses.field(
        default=PLANE_HISTORY_MODE_CURRENT_ONLY_AS_PREPROCESSED,
        init=False,
    )
    inference_batching_schema: str = dataclasses.field(
        default=STATIC_INFERENCE_BATCHING_SCHEMA,
        init=False,
    )
    inference_padding_mode: str = dataclasses.field(
        default=STATIC_INFERENCE_PADDING_MODE,
        init=False,
    )
    trusted_arena_history_schema: str = dataclasses.field(
        default=HISTORY_VALIDATION_SCHEMA,
        init=False,
    )

    def __post_init__(self) -> None:
        if not isinstance(self.model_id, str) or not self.model_id.strip():
            raise LocalPolicyError("model_id must be a non-empty string")
        _validate_inference_batch_size(
            self.inference_batch_size,
            active_batch_size=1,
        )
        _validate_history_validation_mode(self.history_validation_mode)
        try:
            model_dtype = jnp.dtype(self.model.dtype)
        except (AttributeError, TypeError, ValueError) as exc:
            raise LocalPolicyError("raw BT4 model must expose a valid dtype") from exc
        if model_dtype != jnp.dtype(jnp.bfloat16):
            raise LocalPolicyError(
                "the frozen raw-BT4 arena adapter requires bfloat16 compute"
            )

    def select_actions(
        self,
        boards: Iterable[chess.Board],
        histories: Iterable[Iterable[chess.Board]],
    ) -> LocalPolicyBatchResult:
        """Select one legal canonical BT4 action after full history replay."""

        return _select_raw_bt4_actions_impl(
            self.model,
            boards,
            histories,
            inference_batch_size=self.inference_batch_size,
            history_validation_mode=HISTORY_VALIDATION_FULL_REPLAY,
        )

    def select_actions_from_trusted_arena(
        self,
        boards: Iterable[chess.Board],
        endpoints: Iterable[TrustedArenaHistoryEndpoint],
    ) -> LocalPolicyBatchResult:
        """Use sealed constant-size history attestations from the arena."""

        if self.history_validation_mode != HISTORY_VALIDATION_TRUSTED_ARENA_ENDPOINT:
            raise LocalPolicyError("trusted arena endpoint inference is disabled for this policy")
        return _select_raw_bt4_actions_impl(
            self.model,
            boards,
            endpoints,
            inference_batch_size=self.inference_batch_size,
            history_validation_mode=HISTORY_VALIDATION_TRUSTED_ARENA_ENDPOINT,
        )


def _select_raw_bt4_actions_impl(
    model: BT4Model,
    boards: Iterable[chess.Board],
    histories: Iterable[Any],
    *,
    inference_batch_size: int | None,
    history_validation_mode: str,
) -> LocalPolicyBatchResult:
    _validate_history_validation_mode(history_validation_mode)
    board_items = _materialize_batch(boards, name="boards")
    history_items = _materialize_batch(histories, name="histories")
    if len(history_items) != len(board_items):
        raise LocalPolicyError(
            "histories must have exactly one chronological board history per board"
        )

    checked_boards = tuple(
        _require_standard_board(item, name=f"boards[{row}]")
        for row, item in enumerate(board_items)
    )
    for row, (original_board, checked_board, history) in enumerate(
        zip(board_items, checked_boards, history_items, strict=True)
    ):
        if history_validation_mode == HISTORY_VALIDATION_FULL_REPLAY:
            _validate_history(checked_board, history, row=row)
        elif history_validation_mode == HISTORY_VALIDATION_TRUSTED_ARENA_ENDPOINT:
            _validate_trusted_arena_history_endpoint(
                original_board,
                checked_board,
                history,
                row=row,
            )
        else:  # pragma: no cover - guarded by the shared validator.
            raise RuntimeError("Validated history mode became unsupported.")

    planes_list: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    for row, board in enumerate(checked_boards):
        encoded = np.asarray(
            encode_board(
                board,
                [],
                planes_layout="nchw",
                input_format=LC0_CANONICAL_1858_INPUT_FORMAT,
            )
        )
        expected_plane_shape = (TOTAL_PLANES, 8, 8)
        if encoded.shape != expected_plane_shape:
            raise LocalPolicyError(
                f"encoded planes for row {row} must have shape "
                f"{expected_plane_shape}, got {encoded.shape}"
            )
        if encoded.dtype != np.dtype(np.float32):
            raise LocalPolicyError(
                f"encoded planes for row {row} must have float32 dtype, got {encoded.dtype}"
            )
        if not np.all(np.isfinite(encoded)):
            raise LocalPolicyError(f"encoded planes for row {row} contain nonfinite values")
        planes_list.append(encoded)

        try:
            mask_value = legal_action_mask(
                board,
                codec_id=ACTION_CODEC_LC0_CANONICAL_1858,
                input_format=LC0_CANONICAL_1858_INPUT_FORMAT,
            )
        except (ActionCodecError, IndexError, KeyError, TypeError, ValueError) as exc:
            raise LocalPolicyError(
                f"could not construct the canonical BT4 root mask for row {row}"
            ) from exc
        mask = np.asarray(mask_value)
        if mask.shape != (ACTION_VOCAB_SIZE,):
            raise LocalPolicyError(
                f"canonical BT4 root mask for row {row} must have shape "
                f"({ACTION_VOCAB_SIZE},), got {mask.shape}"
            )
        if mask.dtype != np.dtype(np.bool_):
            raise LocalPolicyError(
                f"canonical BT4 root mask for row {row} must have boolean dtype"
            )
        if int(np.count_nonzero(mask)) != board.legal_moves.count():
            raise LocalPolicyError(
                f"canonical BT4 root mask for row {row} does not cover every legal move"
            )
        if not np.any(mask):
            raise LocalPolicyError(f"boards[{row}] has no legal canonical BT4 action")
        masks.append(mask)

    current_planes = np.stack(planes_list, axis=0)
    root_legal_mask = np.stack(masks, axis=0)
    active_batch_size = len(checked_boards)
    physical_batch_size = _validate_inference_batch_size(
        inference_batch_size,
        active_batch_size=active_batch_size,
    )
    inference_planes, inference_root_legal_mask = _pad_validated_inference_inputs(
        current_planes,
        root_legal_mask,
        inference_batch_size=physical_batch_size,
    )

    try:
        action_values, finite_values = infer_raw_bt4_actions(
            model,
            inference_planes,
            inference_root_legal_mask,
        )
        action_values, finite_values = jax.block_until_ready(
            (action_values, finite_values)
        )
        physical_actions = np.asarray(jax.device_get(action_values))
        physical_finite = np.asarray(jax.device_get(finite_values))
    except TimeoutError:
        raise
    except LocalPolicyError:
        raise
    except Exception as exc:
        raise LocalPolicyError("raw BT4 inference failed closed") from exc

    if physical_actions.shape != (physical_batch_size,):
        raise LocalPolicyError(
            "raw BT4 actions must have physical shape "
            f"({physical_batch_size},), got {physical_actions.shape}"
        )
    _require_integer(physical_actions, name="raw_bt4_actions")
    if physical_finite.shape != (physical_batch_size,) or (
        physical_finite.dtype != np.dtype(np.bool_)
    ):
        raise LocalPolicyError(
            "raw BT4 finite-logit attestations have an invalid shape or dtype"
        )

    selected_indices = np.asarray(
        physical_actions[:active_batch_size],
        dtype=np.int32,
    )
    finite_rows = physical_finite[:active_batch_size]
    if not np.all(finite_rows):
        bad_rows = np.flatnonzero(~finite_rows).tolist()
        raise LocalPolicyError(f"raw BT4 policy logits are nonfinite for rows {bad_rows}")
    if np.any(
        (selected_indices < 0)
        | (selected_indices >= ACTION_VOCAB_SIZE)
    ):
        raise LocalPolicyError("raw BT4 actions contain an out-of-range index")

    moves: list[chess.Move] = []
    for row, (board, index) in enumerate(
        zip(checked_boards, selected_indices, strict=True)
    ):
        action_index = int(index)
        if not root_legal_mask[row, action_index]:
            raise LocalPolicyError(
                f"raw BT4 selected an illegal canonical action "
                f"{action_index} for row {row}"
            )
        try:
            move = decode_action(
                action_index,
                codec_id=ACTION_CODEC_LC0_CANONICAL_1858,
                board=board,
                input_format=LC0_CANONICAL_1858_INPUT_FORMAT,
            )
            round_trip_index = encode_action(
                move,
                codec_id=ACTION_CODEC_LC0_CANONICAL_1858,
                board=board,
                input_format=LC0_CANONICAL_1858_INPUT_FORMAT,
            )
        except (ActionCodecError, IndexError, KeyError, TypeError, ValueError) as exc:
            raise LocalPolicyError(
                f"raw BT4 action {action_index} for row {row} is undecodable"
            ) from exc
        if round_trip_index != action_index or not board.is_legal(move):
            raise LocalPolicyError(
                f"raw BT4 action {action_index} decodes illegally for row {row}"
            )
        moves.append(move)

    return LocalPolicyBatchResult(
        action_indices=_readonly(selected_indices, dtype=np.int32),
        moves=tuple(moves),
        diagnostics=None,
    )


__all__ = [
    "LocalBT4Policy",
    "RAW_BT4_POLICY_COMPUTE_DTYPE",
    "infer_raw_bt4_actions",
]
