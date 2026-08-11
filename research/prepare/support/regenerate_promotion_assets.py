#!/usr/bin/env python3
"""Regenerate and audit the immutable promotion opening assets.

The trajectory shards are the provenance authority.  A FEN alone cannot
preserve repetition state, so this command reconstructs the complete standard
chess history for every selected opening from the consecutive trajectory rows,
replays the recorded moves, and rejects any opening whose exact root is
terminal.  Rejected roots are fed back into the deterministic hash-ranked pool
selection; they are never skipped or replaced at arena runtime.

Run from the repository root:

    .venv/bin/python \
      research/prepare/support/regenerate_promotion_assets.py
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import re
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import chess
import numpy as np

from research.arena import build_opening_pool, load_opening_pool, save_opening_pool
from research.prepare import REPO_ROOT, require_within_workspace, sha256_file


ASSET_CONTRACT_SCHEMA = "chess-dfm-promotion-opening-assets-v3"
ASSET_VERSION = "promotion-ply12-n2048-v3"
HISTORY_SCHEMA = "chess-dfm-arena-opening-histories-v1"
HISTORY_CONVENTION = "chronological_standard_initial_through_root_inclusive"
SELECTION_SEED = 2026071802
OPENING_PLY = 12
PAIR_COUNT = 2048
MAX_FILTER_PASSES = 16
DEFAULT_OUTPUT_DIR = REPO_ROOT / "research" / "assets" / "arena"
DEFAULT_POOL_NAME = f"{ASSET_VERSION}.json"
DEFAULT_HISTORIES_NAME = "promotion-ply12-n2048-histories-v3.json"
DEFAULT_CONTRACT_NAME = f"{ASSET_VERSION}-contract.json"
_CHUNK_NAME = re.compile(r"^(?P<prefix>.*?)(?P<number>[0-9]+)(?P<suffix>\.npz)$")


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def json_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    destination = require_within_workspace(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                payload,
                handle,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def _canonical_position_fen(board: chess.Board) -> str:
    return board.fen(en_passant="legal")


def _parse_standard_fen(raw_fen: Any) -> chess.Board:
    try:
        board = chess.Board(" ".join(str(raw_fen).split()))
    except ValueError as exc:
        raise ValueError("invalid_standard_history_fen") from exc
    if not board.is_valid():
        raise ValueError("invalid_standard_history_fen")
    return board


def heldout_fen_universe(
    shard_paths: Sequence[Path],
    *,
    opening_ply: int,
) -> set[str]:
    """Return every valid, nonterminal canonical root at one held-out ply."""

    result: set[str] = set()
    for path in sorted(shard_paths, key=lambda value: (value.name, str(value))):
        source = require_within_workspace(path)
        with np.load(source, allow_pickle=False) as shard:
            if str(np.asarray(shard["schema_version"]).item()) != "trajectory-v3":
                raise ValueError(f"Unexpected trajectory schema in {source}")
            if "fen_t" not in shard or "ply" not in shard:
                raise KeyError(f"Missing fen_t/ply provenance in {source}")
            fens = np.asarray(shard["fen_t"])
            plies = np.asarray(shard["ply"])
            if fens.ndim != 1 or plies.shape != fens.shape:
                raise ValueError(f"Invalid fen_t/ply shapes in {source}")
            for raw_fen in fens[plies == opening_ply].tolist():
                try:
                    board = _parse_standard_fen(raw_fen)
                except ValueError:
                    continue
                if not board.is_game_over(claim_draw=True):
                    result.add(_canonical_position_fen(board))
    return result


@dataclass(frozen=True)
class ShardRows:
    plies: np.ndarray
    fens: np.ndarray
    first_actions_uci: np.ndarray
    source_uri: str


@dataclass(frozen=True)
class HistoryRejection:
    fen: str
    selection_hash: str
    source_shard: str
    source_row: int
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "fen": self.fen,
            "reason": self.reason,
            "selection_hash": self.selection_hash,
            "source_row": self.source_row,
            "source_shard": self.source_shard,
        }


class TrajectoryHistoryReader:
    """Read the 13 consecutive provenance rows ending at a selected root."""

    def __init__(self, split_dir: Path):
        self.split_dir = require_within_workspace(split_dir)
        self.paths = tuple(sorted(self.split_dir.glob("*.npz")))
        if not self.paths:
            raise FileNotFoundError(f"No trajectory shards under {self.split_dir}")
        if len({path.name for path in self.paths}) != len(self.paths):
            raise ValueError("Trajectory shard names must be unique.")
        self.path_index = {path.name: index for index, path in enumerate(self.paths)}

    @lru_cache(maxsize=8)
    def _load(self, path_index: int) -> ShardRows:
        path = self.paths[path_index]
        with np.load(require_within_workspace(path), allow_pickle=False) as shard:
            if str(np.asarray(shard["schema_version"]).item()) != "trajectory-v3":
                raise ValueError(f"Unexpected trajectory schema in {path}")
            required = {"ply", "fen_t", "actions_uci", "source_uri"}
            missing = required - set(shard.files)
            if missing:
                raise KeyError(f"Missing history provenance {sorted(missing)} in {path}")
            plies = np.asarray(shard["ply"]).copy()
            fens = np.asarray(shard["fen_t"]).copy()
            actions = np.asarray(shard["actions_uci"])
            if (
                plies.ndim != 1
                or fens.shape != plies.shape
                or actions.ndim != 2
                or actions.shape[0] != plies.shape[0]
                or actions.shape[1] < 1
            ):
                raise ValueError(f"Invalid history provenance shapes in {path}")
            return ShardRows(
                plies=plies,
                fens=fens,
                first_actions_uci=actions[:, 0].copy(),
                source_uri=str(np.asarray(shard["source_uri"]).item()),
            )

    @staticmethod
    def _source_uri_is_predecessor(previous: str, current: str) -> bool:
        previous_path = previous.rsplit("/", 1)
        current_path = current.rsplit("/", 1)
        if len(previous_path) != 2 or len(current_path) != 2:
            return False
        if previous_path[0] != current_path[0]:
            return False
        previous_match = _CHUNK_NAME.fullmatch(previous_path[1])
        current_match = _CHUNK_NAME.fullmatch(current_path[1])
        if previous_match is None or current_match is None:
            return False
        return (
            previous_match.group("prefix") == current_match.group("prefix")
            and previous_match.group("suffix") == current_match.group("suffix")
            and int(previous_match.group("number")) + 1
            == int(current_match.group("number"))
        )

    def _history_rows(
        self,
        *,
        shard_name: str,
        source_row: int,
        position_count: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        try:
            path_index = self.path_index[shard_name]
        except KeyError as exc:
            raise ValueError(f"unknown_source_shard:{shard_name}") from exc
        if source_row < 0:
            raise ValueError("negative_source_row")

        chunks: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        needed = int(position_count)
        row = int(source_row)
        current_index = path_index
        while needed:
            if current_index < 0:
                raise ValueError("history_precedes_available_shards")
            current = self._load(current_index)
            if row >= len(current.plies):
                raise ValueError("source_row_out_of_bounds")
            take = min(needed, row + 1)
            start = row + 1 - take
            chunks.append(
                (
                    current.plies[start : row + 1],
                    current.fens[start : row + 1],
                    current.first_actions_uci[start : row + 1],
                )
            )
            needed -= take
            if not needed:
                break
            previous_index = current_index - 1
            if previous_index < 0:
                raise ValueError("history_precedes_available_shards")
            previous = self._load(previous_index)
            if not self._source_uri_is_predecessor(
                previous.source_uri,
                current.source_uri,
            ):
                raise ValueError("cross_shard_source_uri_is_not_contiguous")
            current_index = previous_index
            row = len(previous.plies) - 1

        chunks.reverse()
        return tuple(
            np.concatenate([chunk[position] for chunk in chunks])
            for position in range(3)
        )

    def reconstruct(self, opening: Mapping[str, Any]) -> tuple[str, ...]:
        ply = int(opening["ply"])
        if ply < 0:
            raise ValueError("negative_opening_ply")
        source_shard = str(opening["source_shard"])
        expected_prefix = f"{self.split_dir.name}/"
        if not source_shard.startswith(expected_prefix):
            raise ValueError("source_shard_split_mismatch")
        shard_name = source_shard[len(expected_prefix) :]
        plies, raw_fens, first_actions = self._history_rows(
            shard_name=shard_name,
            source_row=int(opening["source_row"]),
            position_count=ply + 1,
        )
        expected_plies = np.arange(ply + 1, dtype=plies.dtype)
        if not np.array_equal(plies, expected_plies):
            raise ValueError("history_ply_sequence_mismatch")

        histories: list[str] = []
        for raw_fen in raw_fens.tolist():
            histories.append(_canonical_position_fen(_parse_standard_fen(raw_fen)))
        board = chess.Board()
        if histories[0] != _canonical_position_fen(board):
            raise ValueError("history_does_not_begin_at_standard_initial")

        for history_index, target_fen in enumerate(histories[1:], start=1):
            raw_action = str(first_actions[history_index - 1])
            try:
                move = chess.Move.from_uci(raw_action)
            except ValueError as exc:
                raise ValueError(
                    f"invalid_recorded_action_at_ply_{history_index - 1}"
                ) from exc
            if move not in board.legal_moves:
                raise ValueError(
                    f"illegal_recorded_action_at_ply_{history_index - 1}"
                )
            board.push(move)
            if _canonical_position_fen(board) != target_fen:
                raise ValueError(
                    f"recorded_action_transition_mismatch_at_ply_{history_index}"
                )
            if (
                board.is_game_over(claim_draw=True)
                and history_index != len(histories) - 1
            ):
                termination = board.outcome(claim_draw=True)
                name = (
                    termination.termination.name.lower()
                    if termination is not None
                    else "unknown"
                )
                raise ValueError(
                    f"history_continues_after_terminal_{name}_at_ply_"
                    f"{history_index}"
                )

        if histories[-1] != str(opening["fen"]):
            raise ValueError("history_root_fen_mismatch")
        outcome = board.outcome(claim_draw=True)
        if outcome is not None:
            raise ValueError(
                f"history_root_terminal_{outcome.termination.name.lower()}"
            )
        if len(board.move_stack) != ply:
            raise ValueError("history_move_stack_length_mismatch")
        return tuple(histories)


def audit_opening_pool(
    pool: Mapping[str, Any],
    *,
    reader: TrajectoryHistoryReader,
) -> tuple[dict[str, tuple[str, ...]], list[HistoryRejection]]:
    openings = pool.get("openings")
    if not isinstance(openings, list):
        raise ValueError("Opening pool has no openings list.")
    histories: dict[str, tuple[str, ...]] = {}
    rejections: list[HistoryRejection] = []

    # Shard grouping makes the reader's bounded cache deterministic and avoids
    # repeatedly decompressing the same provenance arrays.
    ordered = sorted(
        openings,
        key=lambda opening: (
            str(opening["source_shard"]),
            int(opening["source_row"]),
            str(opening["fen"]),
        ),
    )
    for opening in ordered:
        fen = str(opening["fen"])
        try:
            histories[fen] = reader.reconstruct(opening)
        except (KeyError, TypeError, ValueError) as exc:
            rejections.append(
                HistoryRejection(
                    fen=fen,
                    selection_hash=str(opening["selection_hash"]),
                    source_shard=str(opening["source_shard"]),
                    source_row=int(opening["source_row"]),
                    reason=str(exc),
                )
            )
    rejections.sort(key=lambda item: (item.selection_hash, item.fen))
    return histories, rejections


def make_history_sidecar(
    pool: Mapping[str, Any],
    histories_by_fen: Mapping[str, Sequence[str]],
) -> dict[str, Any]:
    entries = []
    for index, opening in enumerate(pool["openings"]):
        fen = str(opening["fen"])
        try:
            history = histories_by_fen[fen]
        except KeyError as exc:
            raise ValueError(f"Missing exact history for opening {index}") from exc
        entries.append(
            {
                "opening_index": index,
                "fen": fen,
                "history_fens": list(history),
            }
        )
    sidecar: dict[str, Any] = {
        "schema_version": HISTORY_SCHEMA,
        "history_convention": HISTORY_CONVENTION,
        "pool_sha256": pool["pool_sha256"],
        "entries": entries,
    }
    sidecar["manifest_sha256"] = json_sha256(sidecar)
    return sidecar


def _repo_relative(path: Path) -> str:
    return str(require_within_workspace(path).relative_to(REPO_ROOT))


def _trajectory_archive_sha256() -> str:
    manifest = json.loads(
        (REPO_ROOT / "research" / "assets.json").read_text(encoding="utf-8")
    )
    archive = manifest["trajectory_v3"]["archive"]
    archive_path = require_within_workspace(
        REPO_ROOT / "data" / "source" / str(archive["name"])
    )
    expected_size = int(archive["size_bytes"])
    if archive_path.stat().st_size != expected_size:
        raise ValueError(
            f"Trajectory archive size mismatch for {archive_path}: "
            f"expected {expected_size}, found {archive_path.stat().st_size}."
        )
    expected_sha256 = str(archive["sha256"])
    observed_sha256 = sha256_file(archive_path)
    if observed_sha256 != expected_sha256:
        raise ValueError(
            f"Trajectory archive digest mismatch for {archive_path}: "
            f"expected {expected_sha256}, found {observed_sha256}."
        )
    return observed_sha256


def _require_split_directory(path: Path, *, expected_split: str) -> Path:
    split_dir = require_within_workspace(path)
    if split_dir.name != expected_split:
        raise ValueError(
            f"Expected the {expected_split!r} split directory, got {split_dir}."
        )
    return split_dir


def _validate_output_names(*names: str) -> tuple[str, ...]:
    validated: list[str] = []
    for name in names:
        if (
            not isinstance(name, str)
            or not name
            or Path(name).name != name
            or not name.endswith(".json")
        ):
            raise ValueError(
                "Promotion asset output names must be distinct .json basenames."
            )
        validated.append(name)
    if len(set(validated)) != len(validated):
        raise ValueError("Promotion asset output names must be distinct.")
    return tuple(validated)


def regenerate(
    *,
    val_dir: Path,
    test_dir: Path,
    output_dir: Path,
    seed: int = SELECTION_SEED,
    opening_ply: int = OPENING_PLY,
    pair_count: int = PAIR_COUNT,
    max_filter_passes: int = MAX_FILTER_PASSES,
    pool_name: str = DEFAULT_POOL_NAME,
    histories_name: str = DEFAULT_HISTORIES_NAME,
    contract_name: str = DEFAULT_CONTRACT_NAME,
) -> dict[str, Any]:
    val_dir = _require_split_directory(val_dir, expected_split="val")
    test_dir = _require_split_directory(test_dir, expected_split="test")
    output_dir = require_within_workspace(output_dir)
    pool_name, histories_name, contract_name = _validate_output_names(
        pool_name,
        histories_name,
        contract_name,
    )
    val_paths = tuple(sorted(val_dir.glob("*.npz")))
    test_paths = tuple(sorted(test_dir.glob("*.npz")))
    if not val_paths or not test_paths:
        raise FileNotFoundError("Promotion regeneration requires val and test shards.")
    if pair_count < 1:
        raise ValueError("pair_count must be positive.")

    val_fens = heldout_fen_universe(val_paths, opening_ply=opening_ply)
    excluded = set(val_fens)
    reader = TrajectoryHistoryReader(test_dir)
    all_rejections: dict[str, HistoryRejection] = {}
    pool: dict[str, Any] | None = None
    histories: dict[str, tuple[str, ...]] = {}

    for filter_pass in range(1, max_filter_passes + 1):
        pool = build_opening_pool(
            test_paths,
            seed=seed,
            count=pair_count,
            opening_ply=opening_ply,
            heldout_split="test",
            exclude_fens=excluded,
        )
        histories, rejections = audit_opening_pool(pool, reader=reader)
        if not rejections:
            break
        new_rejections = [item for item in rejections if item.fen not in excluded]
        if not new_rejections:
            raise RuntimeError("History filtering made no progress.")
        for rejection in new_rejections:
            all_rejections[rejection.fen] = rejection
            excluded.add(rejection.fen)
    else:
        raise RuntimeError(
            f"Promotion pool still has invalid histories after {max_filter_passes} passes."
        )
    assert pool is not None

    selected_fens = {str(opening["fen"]) for opening in pool["openings"]}
    overlap = selected_fens & val_fens
    if overlap:
        raise RuntimeError(
            f"Promotion pool overlaps the val FEN universe ({len(overlap)} roots)."
        )
    if len(histories) != pair_count or selected_fens != set(histories):
        raise RuntimeError("Final exact-history audit is incomplete.")

    output_dir.mkdir(parents=True, exist_ok=True)
    pool_path = require_within_workspace(output_dir / pool_name)
    histories_path = require_within_workspace(output_dir / histories_name)
    contract_path = require_within_workspace(output_dir / contract_name)
    save_opening_pool(pool, pool_path)
    loaded_pool = load_opening_pool(
        pool_path,
        expected_pool_sha256=str(pool["pool_sha256"]),
    )
    sidecar = make_history_sidecar(loaded_pool, histories)
    atomic_write_json(histories_path, sidecar)

    # This is the runtime's canonical digest/alignment/replay validator.  It
    # audits every sidecar entry, not just a sample.
    from research.play_arena import load_opening_history_sidecar

    loaded_histories = load_opening_history_sidecar(
        histories_path,
        opening_pool=loaded_pool,
        expected_manifest_sha256=str(sidecar["manifest_sha256"]),
    )
    if len(loaded_histories.histories_by_opening_index) != pair_count:
        raise RuntimeError("Runtime history validator returned the wrong count.")

    rejection_values = [
        rejection.as_dict()
        for rejection in sorted(
            all_rejections.values(),
            key=lambda item: (item.selection_hash, item.fen),
        )
    ]
    rejection_counts = dict(
        sorted(collections.Counter(item["reason"] for item in rejection_values).items())
    )
    contract: dict[str, Any] = {
        "schema_version": ASSET_CONTRACT_SCHEMA,
        "asset_version": (
            ASSET_VERSION
            if (
                seed,
                opening_ply,
                pair_count,
            )
            == (SELECTION_SEED, OPENING_PLY, PAIR_COUNT)
            else "custom-promotion-opening-assets-v3"
        ),
        "pair_count": pair_count,
        "opening_ply": opening_ply,
        "selection_seed": seed,
        "selection_algorithm": pool["selection"]["algorithm"],
        "source_dataset": {
            "trajectory_archive_sha256": _trajectory_archive_sha256(),
            "val_shard_count": len(val_paths),
            "test_shard_count": len(test_paths),
            "val_fen_universe_count": len(val_fens),
            "val_fen_universe_sha256": json_sha256(sorted(val_fens)),
            "test_source_manifest_sha256": pool["source_manifest_sha256"],
        },
        "disjointness": {
            "test_selected_fens": len(selected_fens),
            "val_test_selected_overlap_count": len(overlap),
            "assertion": "all_valid_val_ply12_fens_excluded_from_test_selection",
        },
        "history_filter": {
            "policy": (
                "reject_before_reselection; never_skip_or_replace_at_arena_runtime"
            ),
            "filter_passes": filter_pass,
            "rejected_opening_count": len(rejection_values),
            "rejection_counts": rejection_counts,
            "rejections": rejection_values,
            "final_exact_replay_count": len(
                loaded_histories.histories_by_opening_index
            ),
            "final_terminal_root_count": 0,
        },
        "pool": {
            "path": _repo_relative(pool_path),
            "file_sha256": sha256_file(pool_path),
            "pool_sha256": pool["pool_sha256"],
            "ordered_fens_sha256": pool["ordered_fens_sha256"],
            "selection_contract_sha256": pool["selection_contract_sha256"],
            "source_manifest_sha256": pool["source_manifest_sha256"],
        },
        "histories": {
            "path": _repo_relative(histories_path),
            "file_sha256": sha256_file(histories_path),
            "manifest_sha256": sidecar["manifest_sha256"],
            "pool_sha256": sidecar["pool_sha256"],
            "history_count": len(sidecar["entries"]),
            "history_convention": sidecar["history_convention"],
        },
    }
    contract["contract_sha256"] = json_sha256(contract)
    atomic_write_json(contract_path, contract)

    result = dict(contract)
    result["contract"] = {
        "path": _repo_relative(contract_path),
        "file_sha256": sha256_file(contract_path),
    }
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--val-dir",
        type=Path,
        default=REPO_ROOT / "data" / "trajectory_v3" / "val",
    )
    parser.add_argument(
        "--test-dir",
        type=Path,
        default=REPO_ROOT / "data" / "trajectory_v3" / "test",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=SELECTION_SEED)
    parser.add_argument("--opening-ply", type=int, default=OPENING_PLY)
    parser.add_argument("--pair-count", type=int, default=PAIR_COUNT)
    parser.add_argument("--max-filter-passes", type=int, default=MAX_FILTER_PASSES)
    parser.add_argument("--pool-name", default=DEFAULT_POOL_NAME)
    parser.add_argument("--histories-name", default=DEFAULT_HISTORIES_NAME)
    parser.add_argument("--contract-name", default=DEFAULT_CONTRACT_NAME)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = regenerate(
        val_dir=args.val_dir,
        test_dir=args.test_dir,
        output_dir=args.output_dir,
        seed=args.seed,
        opening_ply=args.opening_ply,
        pair_count=args.pair_count,
        max_filter_passes=args.max_filter_passes,
        pool_name=args.pool_name,
        histories_name=args.histories_name,
        contract_name=args.contract_name,
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
