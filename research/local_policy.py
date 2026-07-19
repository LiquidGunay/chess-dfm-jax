"""Fail-closed arena policy adapter for the localized DFM model.

The source trajectory pipeline encoded every training position with
``encode_board(current, [])``.  Arena histories are therefore required for
standard-game provenance and adjudication, but they must not be inserted into
the BT4 input planes for this checkpoint.  ``plane_history_mode`` makes that
otherwise surprising compatibility constraint explicit.

This module accepts an already-constructed localized model.  It deliberately
does not construct models, restore checkpoints, invoke a raw-BT4 policy head,
or remap between action codecs.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable
from typing import Any

import chess
import jax
import numpy as np

from chess_dfm_jax.encoding import TOTAL_PLANES, encode_board
from chess_dfm_jax.policy import (
    ACTION_CODEC_LEGACY_ABSOLUTE_1858,
    ACTION_VOCAB_SIZE,
    LC0_CANONICAL_1858_INPUT_FORMAT,
    ActionCodecError,
    legal_action_mask,
    move_to_policy_index,
    policy_index_to_move,
)
from research.inference import (
    LocalDFMInferenceModel,
    infer_dfm_actions_from_current,
    infer_dfm_from_current,
)


PLANE_HISTORY_MODE_CURRENT_ONLY_AS_PREPROCESSED = "current_only_as_preprocessed"
POLICY_INPUT_FORMAT = LC0_CANONICAL_1858_INPUT_FORMAT
STATIC_INFERENCE_BATCHING_SCHEMA = "chess-dfm-static-inference-batching-v1"
STATIC_INFERENCE_PADDING_MODE = "repeat_first_validated_encoded_row_v1"


class LocalPolicyError(ValueError):
    """Raised when a localized policy request or response fails validation."""


@dataclasses.dataclass(frozen=True)
class LocalPolicyDiagnostics:
    """Validated, host-resident diagnostics for one policy batch."""

    input_format: str
    plane_history_mode: str
    action_codec_id: str
    refinement_passes: int
    trace_top_k: int
    encoded_planes_shape: tuple[int, int, int, int]
    legal_move_counts: np.ndarray
    representable_legal_action_counts: np.ndarray
    times: np.ndarray
    actions_before: np.ndarray
    actions_after: np.ndarray
    root_raw_entropy: np.ndarray
    root_raw_legal_mass: np.ndarray
    root_legal_entropy: np.ndarray
    root_legal_topk_indices: np.ndarray
    root_legal_topk_probabilities: np.ndarray
    root_legal_topk_valid: np.ndarray


@dataclasses.dataclass(frozen=True)
class LocalPolicyBatchResult:
    """One selected root action per board plus checked inference diagnostics."""

    action_indices: np.ndarray
    moves: tuple[chess.Move, ...]
    diagnostics: LocalPolicyDiagnostics | None


@dataclasses.dataclass(frozen=True)
class LocalDFMPolicy:
    """Checkpoint-free arena adapter around one already-localized DFM model."""

    model: LocalDFMInferenceModel
    model_id: str
    refinement_passes: int = 8
    trace_top_k: int = 5
    collect_diagnostics: bool = True
    inference_batch_size: int | None = None
    action_codec_id: str = dataclasses.field(
        default=ACTION_CODEC_LEGACY_ABSOLUTE_1858,
        init=False,
    )
    input_format: str = dataclasses.field(default=POLICY_INPUT_FORMAT, init=False)
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

    def __post_init__(self) -> None:
        if not isinstance(self.model_id, str) or not self.model_id.strip():
            raise LocalPolicyError("model_id must be a non-empty string")
        if not isinstance(self.collect_diagnostics, bool):
            raise LocalPolicyError("collect_diagnostics must be boolean")
        _validate_inference_batch_size(
            self.inference_batch_size,
            active_batch_size=1,
        )
        _validate_policy_options(
            refinement_passes=self.refinement_passes,
            trace_top_k=self.trace_top_k,
        )

    def select_actions(
        self,
        boards: Iterable[chess.Board],
        histories: Iterable[Iterable[chess.Board]],
        *,
        refinement_passes: int | None = None,
        trace_top_k: int | None = None,
    ) -> LocalPolicyBatchResult:
        """Select one legal legacy-codec root action for every board."""

        passes = self.refinement_passes if refinement_passes is None else refinement_passes
        top_k = self.trace_top_k if trace_top_k is None else trace_top_k
        return select_local_dfm_actions(
            self.model,
            boards,
            histories,
            refinement_passes=passes,
            trace_top_k=top_k,
            collect_diagnostics=self.collect_diagnostics,
            inference_batch_size=self.inference_batch_size,
        )


def _validate_policy_options(*, refinement_passes: int, trace_top_k: int) -> None:
    if isinstance(refinement_passes, bool) or not isinstance(refinement_passes, int):
        raise LocalPolicyError("refinement_passes must be an integer")
    if refinement_passes < 1:
        raise LocalPolicyError(f"refinement_passes must be >= 1, got {refinement_passes}")
    if isinstance(trace_top_k, bool) or not isinstance(trace_top_k, int):
        raise LocalPolicyError("trace_top_k must be an integer")
    if not 1 <= trace_top_k <= ACTION_VOCAB_SIZE:
        raise LocalPolicyError(
            f"trace_top_k must be in [1, {ACTION_VOCAB_SIZE}], got {trace_top_k}"
        )


def _validate_inference_batch_size(
    inference_batch_size: int | None,
    *,
    active_batch_size: int,
) -> int:
    if inference_batch_size is None:
        return active_batch_size
    if isinstance(inference_batch_size, bool) or not isinstance(
        inference_batch_size,
        int,
    ):
        raise LocalPolicyError("inference_batch_size must be an integer or None")
    if inference_batch_size < 1:
        raise LocalPolicyError("inference_batch_size must be positive")
    if active_batch_size > inference_batch_size:
        raise LocalPolicyError(
            f"active batch size {active_batch_size} exceeds frozen inference "
            f"batch size {inference_batch_size}"
        )
    return inference_batch_size


def _materialize_batch(value: Any, *, name: str) -> tuple[Any, ...]:
    if isinstance(value, (str, bytes, chess.Board)):
        raise LocalPolicyError(f"{name} must be a batch, not {type(value).__name__}")
    try:
        materialized = tuple(value)
    except TypeError as exc:
        raise LocalPolicyError(f"{name} must be iterable") from exc
    if not materialized:
        raise LocalPolicyError(f"{name} must not be empty")
    return materialized


def _require_standard_board(board: Any, *, name: str) -> chess.Board:
    if not isinstance(board, chess.Board):
        raise LocalPolicyError(f"{name} must be a python-chess Board")
    if board.chess960:
        raise LocalPolicyError(f"{name} must use standard chess, not Chess960")
    if not board.is_valid():
        raise LocalPolicyError(f"{name} is not a valid standard-chess position")
    return board.copy(stack=False)


def _exact_fen(board: chess.Board) -> str:
    # Arena sidecars persist canonical legal-EP FENs.  A non-capturable
    # internal EP square is irrelevant to LC0's classical planes and may be
    # dropped by a sidecar round trip.
    return board.fen(en_passant="legal")


def _matching_legal_move(
    previous: chess.Board,
    target: chess.Board,
    *,
    row: int,
    history_index: int,
) -> chess.Move:
    target_fen = _exact_fen(target)
    matches: list[chess.Move] = []
    for move in previous.legal_moves:
        candidate = previous.copy(stack=False)
        candidate.push(move)
        if _exact_fen(candidate) == target_fen:
            matches.append(move)
    if len(matches) != 1:
        raise LocalPolicyError(
            f"histories[{row}][{history_index - 1}:{history_index + 1}] "
            f"must describe exactly one legal ply; found {len(matches)} transitions"
        )
    return matches[0]


def _validate_history(
    board: chess.Board,
    history: Any,
    *,
    row: int,
) -> tuple[chess.Board, ...]:
    history_items = _materialize_batch(history, name=f"histories[{row}]")
    checked = tuple(
        _require_standard_board(item, name=f"histories[{row}][{index}]")
        for index, item in enumerate(history_items)
    )

    standard_start = chess.Board()
    if _exact_fen(checked[0]) != _exact_fen(standard_start):
        raise LocalPolicyError(
            f"histories[{row}] must be the full standard game, oldest-to-newest, "
            "beginning at the standard initial position"
        )

    replay = standard_start
    for history_index, target in enumerate(checked[1:], start=1):
        move = _matching_legal_move(
            replay,
            target,
            row=row,
            history_index=history_index,
        )
        replay.push(move)

    if _exact_fen(checked[-1]) != _exact_fen(board):
        raise LocalPolicyError(
            f"histories[{row}] must end at boards[{row}] (oldest-to-newest convention)"
        )
    if _exact_fen(replay) != _exact_fen(board):
        raise LocalPolicyError(f"histories[{row}] replay does not reproduce boards[{row}]")
    if replay.is_game_over(claim_draw=True):
        raise LocalPolicyError(f"boards[{row}] is terminal when draw claims are honored")
    return checked


def _readonly(array: np.ndarray, *, dtype: np.dtype[Any] | None = None) -> np.ndarray:
    result = np.array(array, dtype=dtype, copy=True)
    result.flags.writeable = False
    return result


def _host_array(value: Any, *, name: str) -> np.ndarray:
    try:
        return np.asarray(jax.device_get(value))
    except (TypeError, ValueError) as exc:
        raise LocalPolicyError(f"inference diagnostic {name} is not array-like") from exc


def _require_shape(array: np.ndarray, expected: tuple[int, ...], *, name: str) -> None:
    if array.shape != expected:
        raise LocalPolicyError(
            f"inference diagnostic {name} must have shape {expected}, got {array.shape}"
        )


def _require_integer(array: np.ndarray, *, name: str) -> None:
    if np.issubdtype(array.dtype, np.bool_) or not np.issubdtype(
        array.dtype,
        np.integer,
    ):
        raise LocalPolicyError(f"inference diagnostic {name} must have integer dtype")


def _require_bool(array: np.ndarray, *, name: str) -> None:
    if array.dtype != np.dtype(np.bool_):
        raise LocalPolicyError(f"inference diagnostic {name} must have boolean dtype")


def _require_finite(array: np.ndarray, *, name: str) -> None:
    if not np.all(np.isfinite(array)):
        raise LocalPolicyError(f"inference diagnostic {name} contains nonfinite values")


def _pad_validated_inference_inputs(
    current_planes: np.ndarray,
    root_legal_mask: np.ndarray,
    *,
    inference_batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Repeat one checked row to a frozen physical inference batch shape."""

    active_batch_size = current_planes.shape[0]
    if root_legal_mask.shape[0] != active_batch_size:
        raise RuntimeError("Validated policy inputs disagree on active batch size.")
    if active_batch_size == inference_batch_size:
        return current_planes, root_legal_mask
    padding_rows = inference_batch_size - active_batch_size
    if padding_rows < 1:
        raise RuntimeError("Frozen inference batch size is smaller than its active batch.")
    padded_planes = np.concatenate(
        (
            current_planes,
            np.repeat(current_planes[:1], padding_rows, axis=0),
        ),
        axis=0,
    )
    padded_mask = np.concatenate(
        (
            root_legal_mask,
            np.repeat(root_legal_mask[:1], padding_rows, axis=0),
        ),
        axis=0,
    )
    return padded_planes, padded_mask


def _slice_inference_result_to_active_rows(
    inference_result: Any,
    *,
    active_batch_size: int,
    inference_batch_size: int,
    horizon: int,
    refinement_passes: int,
    trace_top_k: int,
) -> Any:
    """Check physical shapes, then discard every padded diagnostic row."""

    try:
        trace = inference_result.trace
        values = {
            "actions": inference_result.actions,
            "times": trace.times,
            "actions_before": trace.actions_before,
            "actions_after": trace.actions_after,
            "root_raw_entropy": trace.root_raw_entropy,
            "root_raw_legal_mass": trace.root_raw_legal_mass,
            "root_legal_entropy": trace.root_legal_entropy,
            "root_legal_topk_indices": trace.root_legal_topk_indices,
            "root_legal_topk_probabilities": trace.root_legal_topk_probabilities,
            "root_legal_topk_valid": trace.root_legal_topk_valid,
        }
    except AttributeError as exc:
        raise LocalPolicyError("localized inference returned an incomplete result") from exc
    expected_shapes = {
        "actions": (inference_batch_size, horizon),
        "times": (refinement_passes,),
        "actions_before": (refinement_passes, inference_batch_size, horizon),
        "actions_after": (refinement_passes, inference_batch_size, horizon),
        "root_raw_entropy": (refinement_passes, inference_batch_size),
        "root_raw_legal_mass": (refinement_passes, inference_batch_size),
        "root_legal_entropy": (refinement_passes, inference_batch_size),
        "root_legal_topk_indices": (
            refinement_passes,
            inference_batch_size,
            trace_top_k,
        ),
        "root_legal_topk_probabilities": (
            refinement_passes,
            inference_batch_size,
            trace_top_k,
        ),
        "root_legal_topk_valid": (
            refinement_passes,
            inference_batch_size,
            trace_top_k,
        ),
    }
    for name, expected_shape in expected_shapes.items():
        observed_shape = tuple(values[name].shape)
        if observed_shape != expected_shape:
            raise LocalPolicyError(
                f"inference diagnostic {name} must have shape {expected_shape} "
                f"for the physical batch, got {observed_shape}"
            )
    return inference_result._replace(
        actions=inference_result.actions[:active_batch_size],
        trace=trace._replace(
            actions_before=trace.actions_before[:, :active_batch_size],
            actions_after=trace.actions_after[:, :active_batch_size],
            root_raw_entropy=trace.root_raw_entropy[:, :active_batch_size],
            root_raw_legal_mass=trace.root_raw_legal_mass[:, :active_batch_size],
            root_legal_entropy=trace.root_legal_entropy[:, :active_batch_size],
            root_legal_topk_indices=trace.root_legal_topk_indices[:, :active_batch_size],
            root_legal_topk_probabilities=trace.root_legal_topk_probabilities[
                :, :active_batch_size
            ],
            root_legal_topk_valid=trace.root_legal_topk_valid[:, :active_batch_size],
        ),
    )


def _validate_trace(
    inference_result: Any,
    *,
    batch_size: int,
    horizon: int,
    refinement_passes: int,
    trace_top_k: int,
    root_legal_mask: np.ndarray,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    try:
        trace = inference_result.trace
        arrays = {
            "actions": _host_array(inference_result.actions, name="actions"),
            "times": _host_array(trace.times, name="times"),
            "actions_before": _host_array(trace.actions_before, name="actions_before"),
            "actions_after": _host_array(trace.actions_after, name="actions_after"),
            "root_raw_entropy": _host_array(
                trace.root_raw_entropy,
                name="root_raw_entropy",
            ),
            "root_raw_legal_mass": _host_array(
                trace.root_raw_legal_mass,
                name="root_raw_legal_mass",
            ),
            "root_legal_entropy": _host_array(
                trace.root_legal_entropy,
                name="root_legal_entropy",
            ),
            "root_legal_topk_indices": _host_array(
                trace.root_legal_topk_indices,
                name="root_legal_topk_indices",
            ),
            "root_legal_topk_probabilities": _host_array(
                trace.root_legal_topk_probabilities,
                name="root_legal_topk_probabilities",
            ),
            "root_legal_topk_valid": _host_array(
                trace.root_legal_topk_valid,
                name="root_legal_topk_valid",
            ),
        }
    except AttributeError as exc:
        raise LocalPolicyError("localized inference returned an incomplete result") from exc

    action_shape = (batch_size, horizon)
    action_trace_shape = (refinement_passes, batch_size, horizon)
    pass_batch_shape = (refinement_passes, batch_size)
    topk_shape = (refinement_passes, batch_size, trace_top_k)
    expected_shapes = {
        "actions": action_shape,
        "times": (refinement_passes,),
        "actions_before": action_trace_shape,
        "actions_after": action_trace_shape,
        "root_raw_entropy": pass_batch_shape,
        "root_raw_legal_mass": pass_batch_shape,
        "root_legal_entropy": pass_batch_shape,
        "root_legal_topk_indices": topk_shape,
        "root_legal_topk_probabilities": topk_shape,
        "root_legal_topk_valid": topk_shape,
    }
    for name, expected_shape in expected_shapes.items():
        _require_shape(arrays[name], expected_shape, name=name)

    for name in (
        "actions",
        "actions_before",
        "actions_after",
        "root_legal_topk_indices",
    ):
        _require_integer(arrays[name], name=name)
    _require_bool(arrays["root_legal_topk_valid"], name="root_legal_topk_valid")
    for name in (
        "times",
        "root_raw_entropy",
        "root_raw_legal_mass",
        "root_legal_entropy",
        "root_legal_topk_probabilities",
    ):
        _require_finite(arrays[name], name=name)

    expected_times = np.arange(refinement_passes, dtype=np.float32) / refinement_passes
    if not np.allclose(arrays["times"], expected_times, rtol=0.0, atol=1e-6):
        raise LocalPolicyError("inference diagnostic times does not match the pass schedule")

    actions = arrays["actions"]
    if np.any((actions < 0) | (actions >= ACTION_VOCAB_SIZE)):
        raise LocalPolicyError("inference actions contain a mask or out-of-range index")
    for name in ("actions_before", "actions_after"):
        values = arrays[name]
        if np.any((values < 0) | (values > ACTION_VOCAB_SIZE)):
            raise LocalPolicyError(f"inference diagnostic {name} has an invalid action index")
    if not np.all(arrays["actions_before"][0] == ACTION_VOCAB_SIZE):
        raise LocalPolicyError("inference trace must begin with every action masked")
    if refinement_passes > 1 and not np.array_equal(
        arrays["actions_before"][1:],
        arrays["actions_after"][:-1],
    ):
        raise LocalPolicyError("inference action trace is not pass-contiguous")
    if not np.array_equal(arrays["actions_after"][-1], actions):
        raise LocalPolicyError("inference final actions disagree with its trace")

    topk_indices = arrays["root_legal_topk_indices"]
    if np.any((topk_indices < 0) | (topk_indices >= ACTION_VOCAB_SIZE)):
        raise LocalPolicyError("inference legal top-k contains an out-of-range index")
    topk_probabilities = arrays["root_legal_topk_probabilities"]
    topk_valid = arrays["root_legal_topk_valid"]
    if np.any(~np.any(topk_valid, axis=-1)):
        raise LocalPolicyError("inference legal top-k has no valid root action")
    if np.any((topk_probabilities < 0.0) | (topk_probabilities > 1.0 + 1e-6)):
        raise LocalPolicyError("inference legal top-k probabilities are outside [0, 1]")
    if np.any(topk_probabilities[~topk_valid] != 0.0):
        raise LocalPolicyError("invalid legal top-k padding must have zero probability")
    if np.any(np.diff(topk_valid.astype(np.int8), axis=-1) > 0):
        raise LocalPolicyError("valid legal top-k entries must precede padding")
    if np.any(np.sum(topk_probabilities, axis=-1) > 1.0 + 1e-5):
        raise LocalPolicyError("inference legal top-k probabilities sum above one")

    valid_action_indices = topk_indices[topk_valid]
    valid_batch_indices = np.broadcast_to(
        np.arange(batch_size, dtype=np.int64)[None, :, None],
        topk_shape,
    )[topk_valid]
    if np.any(~root_legal_mask[valid_batch_indices, valid_action_indices]):
        raise LocalPolicyError("inference marked an unrepresentable root action as legal top-k")

    for name in ("root_raw_entropy", "root_legal_entropy"):
        if np.any(arrays[name] < -1e-5):
            raise LocalPolicyError(f"inference diagnostic {name} is negative")
    legal_mass = arrays["root_raw_legal_mass"]
    if np.any((legal_mass < -1e-6) | (legal_mass > 1.0 + 1e-6)):
        raise LocalPolicyError("inference root_raw_legal_mass is outside [0, 1]")

    return actions, arrays


def select_local_dfm_actions(
    model: LocalDFMInferenceModel,
    boards: Iterable[chess.Board],
    histories: Iterable[Iterable[chess.Board]],
    *,
    refinement_passes: int = 8,
    trace_top_k: int = 5,
    collect_diagnostics: bool = True,
    inference_batch_size: int | None = None,
) -> LocalPolicyBatchResult:
    """Run strict batched localized-DFM policy inference.

    ``histories[row]`` must be a full standard-game sequence ordered
    oldest-to-newest, starting at the standard initial position and ending at
    ``boards[row]``.  Histories are validated but, for source-checkpoint
    compatibility, the encoded planes intentionally use only the current board.
    """

    _validate_policy_options(
        refinement_passes=refinement_passes,
        trace_top_k=trace_top_k,
    )
    board_items = _materialize_batch(boards, name="boards")
    history_items = _materialize_batch(histories, name="histories")
    if len(history_items) != len(board_items):
        raise LocalPolicyError(
            "histories must have exactly one chronological board history per board"
        )

    checked_boards = tuple(
        _require_standard_board(item, name=f"boards[{row}]") for row, item in enumerate(board_items)
    )
    for row, (board, history) in enumerate(zip(checked_boards, history_items, strict=True)):
        _validate_history(board, history, row=row)

    planes_list: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    legal_move_counts = np.empty((len(checked_boards),), dtype=np.int32)
    representable_counts = np.empty((len(checked_boards),), dtype=np.int32)
    for row, board in enumerate(checked_boards):
        # Do not pass the validated history here.  The trajectory checkpoint was
        # trained on this exact current-only preprocessing call.
        encoded = np.asarray(
            encode_board(
                board,
                [],
                planes_layout="nchw",
                input_format=POLICY_INPUT_FORMAT,
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
                codec_id=ACTION_CODEC_LEGACY_ABSOLUTE_1858,
            )
        except (ActionCodecError, IndexError, KeyError, TypeError, ValueError) as exc:
            raise LocalPolicyError(
                f"could not construct the legacy root mask for row {row}"
            ) from exc
        mask = np.asarray(mask_value)
        if mask.shape != (ACTION_VOCAB_SIZE,):
            raise LocalPolicyError(
                f"legacy root mask for row {row} must have shape "
                f"({ACTION_VOCAB_SIZE},), got {mask.shape}"
            )
        if mask.dtype != np.dtype(np.bool_):
            raise LocalPolicyError(f"legacy root mask for row {row} must have boolean dtype")
        representable_count = int(np.count_nonzero(mask))
        if representable_count == 0:
            raise LocalPolicyError(
                f"boards[{row}] has no representable legal "
                f"{ACTION_CODEC_LEGACY_ABSOLUTE_1858} action"
            )
        masks.append(mask)
        legal_move_counts[row] = board.legal_moves.count()
        representable_counts[row] = representable_count

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
        horizon = int(model.config.horizon)
    except (AttributeError, TypeError, ValueError) as exc:
        raise LocalPolicyError("model.config must expose an integer horizon") from exc
    if horizon < 1:
        raise LocalPolicyError(f"model.config.horizon must be >= 1, got {horizon}")

    diagnostics: LocalPolicyDiagnostics | None
    try:
        if collect_diagnostics:
            inference_result = infer_dfm_from_current(
                model,
                inference_planes,
                inference_root_legal_mask,
                refinement_passes=refinement_passes,
                trace_top_k=trace_top_k,
                action_codec_id=ACTION_CODEC_LEGACY_ABSOLUTE_1858,
            )
            inference_result = jax.block_until_ready(inference_result)
            inference_result = _slice_inference_result_to_active_rows(
                inference_result,
                active_batch_size=active_batch_size,
                inference_batch_size=physical_batch_size,
                horizon=horizon,
                refinement_passes=refinement_passes,
                trace_top_k=trace_top_k,
            )
            actions, trace_arrays = _validate_trace(
                inference_result,
                batch_size=active_batch_size,
                horizon=horizon,
                refinement_passes=refinement_passes,
                trace_top_k=trace_top_k,
                root_legal_mask=root_legal_mask,
            )
        else:
            action_values = infer_dfm_actions_from_current(
                model,
                inference_planes,
                inference_root_legal_mask,
                refinement_passes=refinement_passes,
                action_codec_id=ACTION_CODEC_LEGACY_ABSOLUTE_1858,
            )
            physical_actions = np.asarray(jax.device_get(jax.block_until_ready(action_values)))
            expected_physical_shape = (physical_batch_size, horizon)
            if physical_actions.shape != expected_physical_shape:
                raise LocalPolicyError(
                    "lean inference actions must have physical shape "
                    f"{expected_physical_shape}, got {physical_actions.shape}"
                )
            _require_integer(physical_actions, name="actions")
            actions = physical_actions[:active_batch_size]
            if np.any((actions < 0) | (actions >= ACTION_VOCAB_SIZE)):
                raise LocalPolicyError(
                    "lean inference actions contain a mask or out-of-range index"
                )
            trace_arrays = None
    except TimeoutError:
        raise
    except LocalPolicyError:
        raise
    except Exception as exc:
        raise LocalPolicyError("localized DFM inference failed closed") from exc

    selected_indices = np.asarray(actions[:, 0], dtype=np.int32)
    moves: list[chess.Move] = []
    for row, (board, index) in enumerate(zip(checked_boards, selected_indices, strict=True)):
        action_index = int(index)
        if not root_legal_mask[row, action_index]:
            raise LocalPolicyError(
                f"localized DFM selected nonrepresentable or illegal root action "
                f"{action_index} for row {row}"
            )
        try:
            move = policy_index_to_move(action_index, "lc0_1858")
            round_trip_index = move_to_policy_index(move, "lc0_1858")
        except (IndexError, KeyError, TypeError, ValueError) as exc:
            raise LocalPolicyError(
                f"localized DFM action {action_index} for row {row} is undecodable"
            ) from exc
        if round_trip_index != action_index or not board.is_legal(move):
            raise LocalPolicyError(
                f"localized DFM action {action_index} decodes to an illegal move for row {row}"
            )
        moves.append(move)

    if trace_arrays is None:
        diagnostics = None
    else:
        diagnostics = LocalPolicyDiagnostics(
            input_format=POLICY_INPUT_FORMAT,
            plane_history_mode=PLANE_HISTORY_MODE_CURRENT_ONLY_AS_PREPROCESSED,
            action_codec_id=ACTION_CODEC_LEGACY_ABSOLUTE_1858,
            refinement_passes=refinement_passes,
            trace_top_k=trace_top_k,
            encoded_planes_shape=tuple(current_planes.shape),
            legal_move_counts=_readonly(legal_move_counts),
            representable_legal_action_counts=_readonly(representable_counts),
            times=_readonly(trace_arrays["times"]),
            actions_before=_readonly(trace_arrays["actions_before"]),
            actions_after=_readonly(trace_arrays["actions_after"]),
            root_raw_entropy=_readonly(trace_arrays["root_raw_entropy"]),
            root_raw_legal_mass=_readonly(trace_arrays["root_raw_legal_mass"]),
            root_legal_entropy=_readonly(trace_arrays["root_legal_entropy"]),
            root_legal_topk_indices=_readonly(trace_arrays["root_legal_topk_indices"]),
            root_legal_topk_probabilities=_readonly(trace_arrays["root_legal_topk_probabilities"]),
            root_legal_topk_valid=_readonly(trace_arrays["root_legal_topk_valid"]),
        )
    return LocalPolicyBatchResult(
        action_indices=_readonly(selected_indices, dtype=np.int32),
        moves=tuple(moves),
        diagnostics=diagnostics,
    )


__all__ = [
    "LocalDFMPolicy",
    "LocalPolicyBatchResult",
    "LocalPolicyDiagnostics",
    "LocalPolicyError",
    "PLANE_HISTORY_MODE_CURRENT_ONLY_AS_PREPROCESSED",
    "POLICY_INPUT_FORMAT",
    "STATIC_INFERENCE_BATCHING_SCHEMA",
    "STATIC_INFERENCE_PADDING_MODE",
    "select_local_dfm_actions",
]
