"""Deterministic, engine-agnostic support for a persistent paired arena.

This module deliberately does not know how to load a model or play a move.  It
owns the stable pieces around those mutable systems: opening selection,
color-reversed game specifications, fail-closed outcome classification, paired
statistics, and sequential-test bookkeeping.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import chess
import numpy as np

from chess_dfm_jax.data.trajectory_v3 import TRAJECTORY_V3
from research.prepare import require_within_workspace


OPENING_POOL_SCHEMA = "chess-dfm-arena-opening-pool-v1"
OPENING_SELECTION_ALGORITHM = "sha256-seed-canonical-fen-v1"
HELDOUT_SPLITS = frozenset({"val", "test"})
LEGACY_INCOMPLETE_PROMOTION_COVERAGE_LIMITATION = (
    "legacy_lc0_1858_codec_has_incomplete_promotion_coverage"
)
LEGACY_ACTION_CODEC_CAPABILITY = {
    "action_codec_id": "legacy_absolute_1858",
    "complete_legal_move_coverage": False,
    "unrepresentable_move_classes": [
        "white_knight_promotion",
        "black_queen_promotion",
        "black_rook_promotion",
        "black_bishop_promotion",
        "black_knight_promotion",
    ],
    "unrepresentable_policy": (
        "mask_only_no_remap_or_fallback; zero_representable_or_undecodable_"
        "selected_action_is_a_loss"
    ),
}

NORMAL_TERMINATION = "normal"
PLY_CAP_TERMINATION = "ply_cap"
FAILURE_TERMINATIONS = frozenset({"illegal_move", "timeout", "exception"})
GAME_TERMINATIONS = frozenset(
    {NORMAL_TERMINATION, PLY_CAP_TERMINATION, *FAILURE_TERMINATIONS}
)
DECISIVE_RESULTS = frozenset({"1-0", "0-1"})
DRAW_RESULT = "1/2-1/2"
FINAL_RESULTS = frozenset({*DECISIVE_RESULTS, DRAW_RESULT})

PENTANOMIAL_PAIR_POINTS = (0.0, 0.5, 1.0, 1.5, 2.0)
PENTANOMIAL_LABELS = ("LL", "LD", "DD_or_WL_or_LW", "WD", "WW")
GSPRT_DECISIONS = frozenset(
    {"continue", "accept_h0", "accept_h1", "max_pairs"}
)


def arena_foundation_contract() -> dict[str, Any]:
    """Describe what this support layer does and explicitly does not provide."""
    return {
        "engine_adapter": "absent",
        "gpu_model_loading": "absent",
        "uci_adapter": "absent",
        "sequential_stopping_unit": "completed_color_reversed_pair",
        "promotion_stopping": {
            "status": "unsupported_pending_normalized_pentanomial_gsprt",
            "elo_model": "normalized",
            "elo0": 0.0,
            "elo1": 20.0,
            "alpha": 0.05,
            "beta": 0.05,
            "max_games": 4096,
            "max_pairs": 2048,
        },
        "legacy_action_codec": json.loads(
            json.dumps(LEGACY_ACTION_CODEC_CAPABILITY)
        ),
        "known_limitations": [
            {
                "id": LEGACY_INCOMPLETE_PROMOTION_COVERAGE_LIMITATION,
                "required_behavior": "fail_closed",
            }
        ],
    }


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonicalize_fen(fen: str) -> str:
    """Return python-chess's stable six-field representation or fail closed."""
    if not isinstance(fen, str) or not fen.strip():
        raise ValueError("FEN must be a non-empty string.")
    try:
        board = chess.Board(" ".join(fen.split()))
    except ValueError as exc:
        raise ValueError(f"Invalid arena FEN: {fen!r}") from exc
    if not board.is_valid():
        raise ValueError(f"Invalid standard-chess arena FEN: {fen!r}")
    if board.is_game_over(claim_draw=True):
        raise ValueError(f"Terminal position cannot seed an arena game: {fen!r}")
    return board.fen(en_passant="legal")


def opening_selection_hash(fen: str, *, seed: int) -> str:
    """Hash-rank one canonical FEN for deterministic seeded selection."""
    canonical_fen = canonicalize_fen(fen)
    seed = int(seed)
    if seed < 0 or seed >= 2**64:
        raise ValueError("Opening-pool seed must be an unsigned 64-bit integer.")
    material = (
        OPENING_SELECTION_ALGORITHM.encode("utf-8")
        + b"\0"
        + seed.to_bytes(8, "big", signed=False)
        + b"\0"
        + canonical_fen.encode("utf-8")
    )
    return hashlib.sha256(material).hexdigest()


def _opening_pool_payload_digest(pool: Mapping[str, Any]) -> str:
    payload = dict(pool)
    payload.pop("pool_sha256", None)
    return _json_sha256(payload)


def _validate_opening_pool(pool: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(pool)
    if payload.get("schema_version") != OPENING_POOL_SCHEMA:
        raise ValueError(
            "Unsupported opening-pool schema: "
            f"{payload.get('schema_version')!r}"
        )
    digest = payload.get("pool_sha256")
    if not isinstance(digest, str) or len(digest) != 64:
        raise ValueError("Opening pool is missing a valid pool_sha256.")
    expected_digest = _opening_pool_payload_digest(payload)
    if digest != expected_digest:
        raise ValueError(
            "Opening pool digest mismatch: "
            f"expected {expected_digest}, found {digest}"
        )

    split = payload.get("heldout_split")
    if split not in HELDOUT_SPLITS:
        raise ValueError(f"Opening pool is not held out: {split!r}")
    selection = payload.get("selection")
    if not isinstance(selection, Mapping):
        raise ValueError("Opening pool selection metadata is missing.")
    if selection.get("algorithm") != OPENING_SELECTION_ALGORITHM:
        raise ValueError(
            "Unsupported opening selection algorithm: "
            f"{selection.get('algorithm')!r}"
        )
    seed = int(selection["seed"])
    opening_ply = int(selection["opening_ply"])
    if opening_ply < 0:
        raise ValueError("Opening pool opening_ply must be non-negative.")
    excluded_fens = payload.get("excluded_fens")
    if not isinstance(excluded_fens, Mapping):
        raise ValueError("Opening pool exclusion metadata is missing.")
    if int(excluded_fens.get("count", -1)) < 0:
        raise ValueError("Opening pool excluded-FEN count is invalid.")
    excluded_digest = excluded_fens.get("sha256")
    if not isinstance(excluded_digest, str) or len(excluded_digest) != 64:
        raise ValueError("Opening pool excluded-FEN digest is invalid.")

    sources = payload.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("Opening pool must record at least one source shard.")
    expected_source_digest = _json_sha256(sources)
    if payload.get("source_manifest_sha256") != expected_source_digest:
        raise ValueError("Opening pool source-manifest digest mismatch.")

    openings = payload.get("openings")
    if not isinstance(openings, list) or not openings:
        raise ValueError("Opening pool must contain at least one opening.")
    if int(selection.get("selected_count", -1)) != len(openings):
        raise ValueError("Opening pool selected_count does not match openings.")
    if int(selection.get("requested_count", -1)) != len(openings):
        raise ValueError("Opening pool requested_count does not match openings.")

    seen_fens: set[str] = set()
    rank_keys: list[tuple[str, str]] = []
    for index, opening in enumerate(openings):
        if not isinstance(opening, Mapping):
            raise ValueError(f"Opening {index} is not an object.")
        canonical_fen = canonicalize_fen(str(opening.get("fen", "")))
        if canonical_fen != opening.get("fen"):
            raise ValueError(f"Opening {index} is not stored as canonical FEN.")
        if canonical_fen in seen_fens:
            raise ValueError(f"Duplicate FEN in opening pool: {canonical_fen}")
        seen_fens.add(canonical_fen)
        if int(opening.get("ply", -1)) != opening_ply:
            raise ValueError(f"Opening {index} does not match opening_ply.")
        if int(opening.get("source_row", -1)) < 0:
            raise ValueError(f"Opening {index} has invalid source_row.")
        if not str(opening.get("source_shard", "")).startswith(
            f"{split}/"
        ):
            raise ValueError(f"Opening {index} has invalid source_shard.")
        selection_hash = opening_selection_hash(canonical_fen, seed=seed)
        if opening.get("selection_hash") != selection_hash:
            raise ValueError(f"Opening {index} selection hash mismatch.")
        rank_keys.append((selection_hash, canonical_fen))
    if rank_keys != sorted(rank_keys):
        raise ValueError("Opening pool is not in deterministic hash-rank order.")
    ordered_fens_digest = hashlib.sha256(
        "".join(f"{opening['fen']}\n" for opening in openings).encode(
            "utf-8"
        )
    ).hexdigest()
    if payload.get("ordered_fens_sha256") != ordered_fens_digest:
        raise ValueError("Opening pool ordered-FEN digest mismatch.")
    selection_contract = {
        "heldout_split": split,
        "selection": dict(selection),
        "sources": sources,
        "source_manifest_sha256": expected_source_digest,
        "excluded_fens": dict(excluded_fens),
    }
    if payload.get("selection_contract_sha256") != _json_sha256(
        selection_contract
    ):
        raise ValueError("Opening pool selection-contract digest mismatch.")
    return payload


def build_opening_pool(
    shard_paths: Sequence[Path],
    *,
    seed: int,
    count: int,
    opening_ply: int = 12,
    heldout_split: str | None = None,
    exclude_fens: Iterable[str] = (),
) -> dict[str, Any]:
    """Select a deterministic, deduplicated early-game FEN pool.

    Selection is independent of caller-provided shard order.  Shards must come
    from one directory named ``val`` or ``test`` (or match the explicit
    ``heldout_split``), and every selected row is hash-ranked by seed and
    canonical FEN.  A single fixed ply is intentional: the legacy trajectory
    metadata does not provide a reliable unique game identifier, while one row
    at a fixed ply gives at most one candidate per game.
    """
    if count < 1:
        raise ValueError(f"count must be positive, got {count}")
    if opening_ply < 0:
        raise ValueError(f"opening_ply must be non-negative, got {opening_ply}")
    if not shard_paths:
        raise ValueError("At least one held-out trajectory shard is required.")

    paths = [require_within_workspace(path) for path in shard_paths]
    if len(set(paths)) != len(paths):
        raise ValueError("Duplicate trajectory shard path.")
    paths = sorted(paths, key=lambda path: (path.name, str(path)))
    if len({path.name for path in paths}) != len(paths):
        raise ValueError("Held-out shard basenames must be unique.")

    observed_splits = {path.parent.name for path in paths}
    if heldout_split is None:
        if len(observed_splits) != 1:
            raise ValueError(
                f"Held-out shards span multiple splits: {observed_splits}"
            )
        split = next(iter(observed_splits))
    else:
        split = str(heldout_split)
        if observed_splits != {split}:
            raise ValueError(
                f"Shard directories {observed_splits} do not match {split!r}."
            )
    if split not in HELDOUT_SPLITS:
        raise ValueError(
            f"Arena opening pool requires a held-out split, got {split!r}."
        )

    excluded = sorted({canonicalize_fen(fen) for fen in exclude_fens})
    excluded_set = set(excluded)
    excluded_digest = _json_sha256(excluded)
    candidates: dict[str, dict[str, Any]] = {}
    sources: list[dict[str, Any]] = []
    invalid_fen_count = 0
    terminal_fen_count = 0
    inconsistent_ply_count = 0
    duplicate_fen_count = 0
    excluded_fen_count = 0
    rows_scanned = 0
    fixed_ply_row_count = 0
    for path in paths:
        with np.load(path, allow_pickle=False) as shard:
            schema = str(np.asarray(shard["schema_version"]).item())
            if schema != TRAJECTORY_V3:
                raise ValueError(
                    f"Expected {TRAJECTORY_V3} in {path}, found {schema!r}."
                )
            if "fen_t" not in shard or "ply" not in shard:
                raise KeyError(
                    f"Arena opening source lacks fen_t/ply metadata: {path}"
                )
            fens = np.asarray(shard["fen_t"])
            plies = np.asarray(shard["ply"])
            if fens.ndim != 1 or plies.shape != fens.shape:
                raise ValueError(
                    f"fen_t and ply must be matching vectors in {path}: "
                    f"{fens.shape} versus {plies.shape}"
                )
            rows_scanned += int(fens.shape[0])
            source_uri = (
                str(np.asarray(shard["source_uri"]).item())
                if "source_uri" in shard
                else ""
            )
            sources.append(
                {
                    "name": path.name,
                    "rows": int(fens.shape[0]),
                    "sha256": _file_sha256(path),
                }
            )
            for row, (raw_fen, raw_ply) in enumerate(
                zip(fens.tolist(), plies.tolist(), strict=True)
            ):
                ply = int(raw_ply)
                if ply != opening_ply:
                    continue
                fixed_ply_row_count += 1
                try:
                    board = chess.Board(" ".join(str(raw_fen).split()))
                except ValueError:
                    invalid_fen_count += 1
                    continue
                if not board.is_valid():
                    invalid_fen_count += 1
                    continue
                if board.is_game_over(claim_draw=True):
                    terminal_fen_count += 1
                    continue
                fen_ply = 2 * (board.fullmove_number - 1) + (
                    0 if board.turn == chess.WHITE else 1
                )
                if fen_ply != opening_ply:
                    inconsistent_ply_count += 1
                    continue
                fen = board.fen(en_passant="legal")
                if fen in excluded_set:
                    excluded_fen_count += 1
                    continue
                candidate = {
                    "fen": fen,
                    "ply": ply,
                    "source_shard": f"{split}/{path.name}",
                    "source_row": row,
                    "source_uri": source_uri,
                }
                existing = candidates.get(fen)
                if existing is not None and int(existing["ply"]) != ply:
                    raise ValueError(
                        f"Canonical FEN has inconsistent ply metadata: {fen}"
                    )
                if existing is not None:
                    duplicate_fen_count += 1
                provenance_key = (source_uri, path.name, row)
                existing_key = (
                    (
                        str(existing["source_uri"]),
                        str(existing["source_shard"]).split("/", 1)[-1],
                        int(existing["source_row"]),
                    )
                    if existing is not None
                    else None
                )
                if existing_key is None or provenance_key < existing_key:
                    candidates[fen] = candidate

    if len(candidates) < count:
        raise ValueError(
            f"Only {len(candidates)} unique early-game FENs available; "
            f"requested {count}."
        )
    ranked: list[dict[str, Any]] = []
    for fen, candidate in candidates.items():
        ranked.append(
            candidate
            | {
                "selection_hash": opening_selection_hash(
                    fen,
                    seed=seed,
                )
            }
        )
    ranked.sort(key=lambda item: (item["selection_hash"], item["fen"]))
    openings = ranked[:count]
    source_digest = _json_sha256(sources)
    pool: dict[str, Any] = {
        "schema_version": OPENING_POOL_SCHEMA,
        "heldout_split": split,
        "selection": {
            "algorithm": OPENING_SELECTION_ALGORITHM,
            "seed": int(seed),
            "requested_count": int(count),
            "selected_count": len(openings),
            "deduplicated_candidate_count": len(candidates),
            "opening_ply": int(opening_ply),
            "fixed_ply_row_count": fixed_ply_row_count,
            "invalid_standard_fen_count": invalid_fen_count,
            "terminal_fen_count": terminal_fen_count,
            "inconsistent_fen_ply_count": inconsistent_ply_count,
            "duplicate_fen_count": duplicate_fen_count,
            "excluded_fen_rows": excluded_fen_count,
            "rows_scanned": rows_scanned,
        },
        "excluded_fens": {
            "count": len(excluded),
            "sha256": excluded_digest,
        },
        "sources": sources,
        "source_manifest_sha256": source_digest,
        "openings": openings,
    }
    pool["ordered_fens_sha256"] = hashlib.sha256(
        "".join(f"{opening['fen']}\n" for opening in openings).encode(
            "utf-8"
        )
    ).hexdigest()
    selection_contract = {
        "heldout_split": split,
        "selection": pool["selection"],
        "sources": sources,
        "source_manifest_sha256": source_digest,
        "excluded_fens": pool["excluded_fens"],
    }
    pool["selection_contract_sha256"] = _json_sha256(selection_contract)
    pool["pool_sha256"] = _opening_pool_payload_digest(pool)
    return _validate_opening_pool(pool)


def save_opening_pool(
    pool: Mapping[str, Any],
    path: Path,
) -> Path:
    """Atomically persist a validated opening pool inside the workspace."""
    validated = _validate_opening_pool(pool)
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
                validated,
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


def load_opening_pool(
    path: Path,
    *,
    expected_pool_sha256: str | None = None,
) -> dict[str, Any]:
    source = require_within_workspace(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"Opening pool root must be an object: {source}")
    validated = _validate_opening_pool(payload)
    if (
        expected_pool_sha256 is not None
        and validated["pool_sha256"] != expected_pool_sha256
    ):
        raise ValueError(
            "Opening pool does not match pinned digest: "
            f"expected {expected_pool_sha256}, "
            f"found {validated['pool_sha256']}"
        )
    return validated


@dataclasses.dataclass(frozen=True)
class ArenaGameSpec:
    pair_id: str
    opening_index: int
    game_in_pair: int
    fen: str
    white_model: str
    black_model: str

    def __post_init__(self) -> None:
        if not self.pair_id:
            raise ValueError("pair_id must be non-empty.")
        if self.opening_index < 0:
            raise ValueError("opening_index must be non-negative.")
        if self.game_in_pair not in (0, 1):
            raise ValueError("game_in_pair must be 0 or 1.")
        if not self.white_model or not self.black_model:
            raise ValueError("Both model identifiers must be non-empty.")
        if self.white_model == self.black_model:
            raise ValueError("Arena opponents must be distinct.")
        canonical_fen = canonicalize_fen(self.fen)
        if canonical_fen != self.fen:
            raise ValueError("ArenaGameSpec FEN must already be canonical.")


def make_color_reversed_pairs(
    fens: Sequence[str],
    *,
    model_a: str,
    model_b: str,
    start_index: int = 0,
) -> tuple[tuple[ArenaGameSpec, ArenaGameSpec], ...]:
    """Create two games per FEN with exact color reversal."""
    if not model_a or not model_b or model_a == model_b:
        raise ValueError("model_a and model_b must be distinct non-empty IDs.")
    if start_index < 0:
        raise ValueError("start_index must be non-negative.")
    pairs: list[tuple[ArenaGameSpec, ArenaGameSpec]] = []
    for offset, raw_fen in enumerate(fens):
        opening_index = start_index + offset
        fen = canonicalize_fen(raw_fen)
        fen_digest = hashlib.sha256(fen.encode("utf-8")).hexdigest()[:12]
        pair_id = f"pair-{opening_index:08d}-{fen_digest}"
        first = ArenaGameSpec(
            pair_id=pair_id,
            opening_index=opening_index,
            game_in_pair=0,
            fen=fen,
            white_model=model_a,
            black_model=model_b,
        )
        second = ArenaGameSpec(
            pair_id=pair_id,
            opening_index=opening_index,
            game_in_pair=1,
            fen=fen,
            white_model=model_b,
            black_model=model_a,
        )
        pairs.append((first, second))
    return tuple(pairs)


@dataclasses.dataclass(frozen=True)
class GameOutcome:
    pair_id: str
    opening_index: int
    game_in_pair: int
    fen: str
    white_model: str
    black_model: str
    result: str
    termination: str
    ply_count: int
    winner_model: str | None
    loser_model: str | None

    def __post_init__(self) -> None:
        if not self.pair_id or self.opening_index < 0:
            raise ValueError("GameOutcome has invalid pair identity.")
        if self.game_in_pair not in (0, 1):
            raise ValueError("GameOutcome game_in_pair must be 0 or 1.")
        if self.white_model == self.black_model:
            raise ValueError("GameOutcome opponents must be distinct.")
        if canonicalize_fen(self.fen) != self.fen:
            raise ValueError("GameOutcome FEN must already be canonical.")
        if self.result not in FINAL_RESULTS:
            raise ValueError(f"Invalid final game result: {self.result!r}")
        if self.termination not in GAME_TERMINATIONS or self.ply_count < 0:
            raise ValueError("GameOutcome termination metadata is invalid.")
        if self.result == DRAW_RESULT:
            if self.winner_model is not None or self.loser_model is not None:
                raise ValueError("Drawn GameOutcome cannot name winner/loser.")
        else:
            expected_winner = (
                self.white_model
                if self.result == "1-0"
                else self.black_model
            )
            expected_loser = (
                self.black_model
                if self.result == "1-0"
                else self.white_model
            )
            if (
                self.winner_model != expected_winner
                or self.loser_model != expected_loser
            ):
                raise ValueError("GameOutcome winner/loser does not match result.")

    @property
    def white_score(self) -> float:
        return {"1-0": 1.0, "0-1": 0.0, DRAW_RESULT: 0.5}[self.result]

    def score_for(self, model_id: str) -> float:
        if model_id == self.white_model:
            return self.white_score
        if model_id == self.black_model:
            return 1.0 - self.white_score
        raise ValueError(f"Model {model_id!r} did not play this game.")


def classify_game_outcome(
    spec: ArenaGameSpec,
    *,
    termination: str,
    ply_count: int,
    ply_cap: int,
    reported_result: str | None = None,
    fault_model: str | None = None,
) -> GameOutcome:
    """Classify a completed game, charging model faults as losses.

    Unknown or contradictory states raise instead of silently becoming draws.
    A ply-cap termination is always a draw, independent of color or side to
    move, which keeps the cap symmetric across a color-reversed pair.
    """
    if termination not in GAME_TERMINATIONS:
        raise ValueError(f"Unsupported game termination: {termination!r}")
    if ply_count < 0:
        raise ValueError("ply_count must be non-negative.")
    if ply_cap < 1:
        raise ValueError("ply_cap must be positive.")

    winner: str | None
    loser: str | None
    if termination in FAILURE_TERMINATIONS:
        if reported_result is not None:
            raise ValueError("Failure termination cannot also report a result.")
        if fault_model not in {spec.white_model, spec.black_model}:
            raise ValueError(
                "Failure termination must identify the model at fault."
            )
        loser = fault_model
        winner = (
            spec.black_model
            if fault_model == spec.white_model
            else spec.white_model
        )
        result = "0-1" if fault_model == spec.white_model else "1-0"
    elif termination == PLY_CAP_TERMINATION:
        if fault_model is not None or reported_result is not None:
            raise ValueError(
                "Ply-cap termination cannot include a fault or result."
            )
        if ply_count != ply_cap:
            raise ValueError(
                f"ply_cap termination must occur exactly at {ply_cap}, "
                f"got {ply_count}."
            )
        result = DRAW_RESULT
        winner = None
        loser = None
    else:
        if fault_model is not None:
            raise ValueError("Normal termination cannot include a fault.")
        if reported_result not in FINAL_RESULTS:
            raise ValueError(
                f"Normal termination requires a final result, got "
                f"{reported_result!r}."
            )
        if ply_count > ply_cap:
            raise ValueError(
                f"Normal termination exceeds ply cap: {ply_count} > {ply_cap}."
            )
        result = reported_result
        if result == "1-0":
            winner, loser = spec.white_model, spec.black_model
        elif result == "0-1":
            winner, loser = spec.black_model, spec.white_model
        else:
            winner = loser = None

    return GameOutcome(
        pair_id=spec.pair_id,
        opening_index=spec.opening_index,
        game_in_pair=spec.game_in_pair,
        fen=spec.fen,
        white_model=spec.white_model,
        black_model=spec.black_model,
        result=result,
        termination=termination,
        ply_count=int(ply_count),
        winner_model=winner,
        loser_model=loser,
    )


def pair_score_for_model(
    outcomes: Sequence[GameOutcome],
    *,
    model_id: str,
) -> float:
    """Validate one color-reversed pair and return model points in [0, 2]."""
    if len(outcomes) != 2:
        raise ValueError(f"A completed pair requires two games, got {len(outcomes)}.")
    first, second = sorted(outcomes, key=lambda outcome: outcome.game_in_pair)
    if (first.game_in_pair, second.game_in_pair) != (0, 1):
        raise ValueError("Paired outcomes must contain game_in_pair 0 and 1.")
    if first.pair_id != second.pair_id or first.fen != second.fen:
        raise ValueError("Paired outcomes must share pair_id and FEN.")
    if first.opening_index != second.opening_index:
        raise ValueError("Paired outcomes must share opening_index.")
    if not (
        first.white_model == second.black_model
        and first.black_model == second.white_model
    ):
        raise ValueError("Paired outcomes are not exact color reversals.")
    if model_id not in {first.white_model, first.black_model}:
        raise ValueError(f"Model {model_id!r} did not play this pair.")
    return first.score_for(model_id) + second.score_for(model_id)


def _pair_score_index(pair_score: float) -> int:
    doubled = float(pair_score) * 2.0
    index = int(round(doubled))
    if (
        index < 0
        or index >= len(PENTANOMIAL_PAIR_POINTS)
        or not math.isclose(doubled, float(index), abs_tol=1e-12)
    ):
        raise ValueError(
            "Pair score must be one of "
            f"{PENTANOMIAL_PAIR_POINTS}, got {pair_score}."
        )
    return index


@dataclasses.dataclass(frozen=True)
class PentanomialStats:
    counts: tuple[int, int, int, int, int]

    def __post_init__(self) -> None:
        if len(self.counts) != 5 or any(
            int(count) != count or count < 0 for count in self.counts
        ):
            raise ValueError(
                "Pentanomial counts must contain five non-negative integers."
            )

    @property
    def pair_count(self) -> int:
        return int(sum(self.counts))

    @property
    def game_count(self) -> int:
        return 2 * self.pair_count

    @property
    def points(self) -> float:
        return float(
            sum(
                count * points
                for count, points in zip(
                    self.counts,
                    PENTANOMIAL_PAIR_POINTS,
                    strict=True,
                )
            )
        )

    @property
    def score(self) -> float:
        if self.pair_count == 0:
            raise ValueError("Score is undefined without completed pairs.")
        return self.points / self.game_count

    def as_dict(self) -> dict[str, Any]:
        return {
            "labels": list(PENTANOMIAL_LABELS),
            "counts": list(self.counts),
            "pair_count": self.pair_count,
            "game_count": self.game_count,
            "points": self.points,
            "score": self.score,
        }


def pentanomial_stats(
    pair_scores: Iterable[float],
) -> PentanomialStats:
    counts = [0, 0, 0, 0, 0]
    for pair_score in pair_scores:
        counts[_pair_score_index(pair_score)] += 1
    return PentanomialStats(tuple(counts))


def expected_score_from_elo(elo: float) -> float:
    if not math.isfinite(elo):
        raise ValueError(f"Elo must be finite, got {elo}.")
    logit = float(elo) * math.log(10.0) / 400.0
    if logit >= 0.0:
        return 1.0 / (1.0 + math.exp(-logit))
    exp_logit = math.exp(logit)
    return exp_logit / (1.0 + exp_logit)


def elo_from_score(
    score: float,
    *,
    elo_cap: float = 4000.0,
) -> float:
    if not 0.0 <= score <= 1.0 or not math.isfinite(score):
        raise ValueError(f"Score must be finite and in [0, 1], got {score}.")
    if elo_cap <= 0.0 or not math.isfinite(elo_cap):
        raise ValueError(f"elo_cap must be positive and finite, got {elo_cap}.")
    probability_floor = expected_score_from_elo(-float(elo_cap))
    clipped = min(max(float(score), probability_floor), 1.0 - probability_floor)
    elo = 400.0 * math.log10(clipped / (1.0 - clipped))
    return min(max(elo, -float(elo_cap)), float(elo_cap))


@dataclasses.dataclass(frozen=True)
class ScoreEloInterval:
    score: float
    score_lower: float
    score_upper: float
    elo: float
    elo_lower: float
    elo_upper: float
    confidence: float
    pair_count: int
    method: str = "hoeffding_bounded_complete_pairs"
    elo_model: str = "logistic"
    promotion_eligible: bool = False


def pair_aware_score_elo_interval(
    stats: PentanomialStats,
    *,
    confidence: float = 0.95,
    elo_cap: float = 4000.0,
) -> ScoreEloInterval:
    """Conservative score interval using complete pairs as bounded samples.

    Each color-reversed pair contributes one normalized value in ``[0, 1]``.
    Hoeffding's inequality therefore remains valid without pretending the two
    games inside a pair are independent.  The Elo conversion is explicitly
    logistic and descriptive; normalized-Elo promotion uncertainty remains
    unsupported in this foundation.
    """
    if stats.pair_count < 1:
        raise ValueError("At least one completed pair is required.")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1).")
    alpha = 1.0 - confidence
    radius = math.sqrt(
        math.log(2.0 / alpha) / (2.0 * stats.pair_count)
    )
    score = stats.score
    score_lower = max(0.0, score - radius)
    score_upper = min(1.0, score + radius)
    return ScoreEloInterval(
        score=score,
        score_lower=float(score_lower),
        score_upper=float(score_upper),
        elo=elo_from_score(score, elo_cap=elo_cap),
        elo_lower=elo_from_score(float(score_lower), elo_cap=elo_cap),
        elo_upper=elo_from_score(float(score_upper), elo_cap=elo_cap),
        confidence=float(confidence),
        pair_count=stats.pair_count,
    )


@dataclasses.dataclass(frozen=True)
class GSPRTConfig:
    elo0: float
    elo1: float
    alpha: float = 0.05
    beta: float = 0.05
    max_pairs: int = 2048
    elo_model: str = "logistic"

    def __post_init__(self) -> None:
        if not (
            math.isfinite(self.elo0)
            and math.isfinite(self.elo1)
            and self.elo0 < self.elo1
        ):
            raise ValueError("GSPRT requires finite elo0 < elo1.")
        if not 0.0 < self.alpha < 1.0:
            raise ValueError("alpha must be in (0, 1).")
        if not 0.0 < self.beta < 1.0:
            raise ValueError("beta must be in (0, 1).")
        if self.alpha + self.beta >= 1.0:
            raise ValueError("alpha + beta must be less than 1.")
        if self.max_pairs < 1:
            raise ValueError("max_pairs must be positive.")
        if self.elo_model != "logistic":
            raise ValueError(
                "Only logistic pentanomial GSPRT bookkeeping is implemented; "
                "normalized-Elo promotion stopping is pending."
            )
        if not all(
            0.0 < expected_score_from_elo(elo) < 1.0
            for elo in (self.elo0, self.elo1)
        ):
            raise ValueError("GSPRT Elo hypotheses are numerically saturated.")

    @property
    def lower_bound(self) -> float:
        return math.log(self.beta / (1.0 - self.alpha))

    @property
    def upper_bound(self) -> float:
        return math.log((1.0 - self.beta) / self.alpha)


def _secular_root(
    centered_pdf: Sequence[tuple[float, float]],
) -> float:
    """Solve fishtest's constrained-multinomial secular equation."""
    support = [value for value, _probability in centered_pdf]
    lower_support = min(support)
    upper_support = max(support)
    if lower_support >= 0.0 or upper_support <= 0.0:
        raise ValueError("Constrained MLE support must straddle zero.")
    lower = -1.0 / upper_support
    upper = -1.0 / lower_support
    epsilon = 1e-12
    left = lower + epsilon * max(1.0, abs(lower))
    right = upper - epsilon * max(1.0, abs(upper))

    def equation(value: float) -> float:
        return sum(
            probability * support_value / (1.0 + value * support_value)
            for support_value, probability in centered_pdf
        )

    left_value = equation(left)
    right_value = equation(right)
    if left_value <= 0.0 or right_value >= 0.0:
        raise RuntimeError("Unable to bracket constrained-MLE secular root.")
    for _iteration in range(200):
        middle = (left + right) / 2.0
        middle_value = equation(middle)
        if abs(middle_value) <= 1e-14:
            return middle
        if middle_value > 0.0:
            left = middle
        else:
            right = middle
        if right - left <= 1e-14 * max(1.0, abs(middle)):
            return (left + right) / 2.0
    raise RuntimeError("Constrained-MLE secular root did not converge.")


def _mle_with_expected_score(
    empirical_pdf: Sequence[tuple[float, float]],
    expected_score: float,
) -> tuple[float, ...]:
    centered = [
        (value - expected_score, probability)
        for value, probability in empirical_pdf
    ]
    root = _secular_root(centered)
    probabilities = tuple(
        probability / (1.0 + root * (value - expected_score))
        for value, probability in empirical_pdf
    )
    if not math.isclose(sum(probabilities), 1.0, abs_tol=1e-9):
        raise RuntimeError("Constrained-MLE probabilities do not sum to one.")
    observed_expectation = sum(
        value * probability
        for (value, _empirical), probability in zip(
            empirical_pdf,
            probabilities,
            strict=True,
        )
    )
    if not math.isclose(
        observed_expectation,
        expected_score,
        abs_tol=1e-8,
    ):
        raise RuntimeError("Constrained-MLE expectation validation failed.")
    return probabilities


def pentanomial_gsprt_llr(
    counts: Sequence[int],
    *,
    elo0: float,
    elo1: float,
    elo_model: str = "logistic",
) -> float:
    """Generalized pentanomial LLR using constrained multinomial MLEs.

    This follows fishtest's ``LLR_logistic`` construction: empty categories get
    a small regularizing pseudocount, and the empirical five-outcome
    distribution is separately projected onto each hypothesis' expected score.
    """
    stats = PentanomialStats(tuple(counts))
    if stats.pair_count == 0:
        return 0.0
    if elo_model != "logistic":
        raise ValueError(
            "Normalized pentanomial GSPRT is not implemented in this foundation."
        )
    if not math.isfinite(elo0) or not math.isfinite(elo1) or elo0 >= elo1:
        raise ValueError("GSPRT LLR requires finite elo0 < elo1.")
    regularized = [
        float(count) if count > 0 else 1e-3
        for count in stats.counts
    ]
    sample_count = sum(regularized)
    empirical_pdf = tuple(
        (
            index / (len(regularized) - 1),
            count / sample_count,
        )
        for index, count in enumerate(regularized)
    )
    hypothesis0 = _mle_with_expected_score(
        empirical_pdf,
        expected_score_from_elo(elo0),
    )
    hypothesis1 = _mle_with_expected_score(
        empirical_pdf,
        expected_score_from_elo(elo1),
    )
    llr_per_pair = sum(
        empirical_probability * math.log(p1 / p0)
        for (
            _value,
            empirical_probability,
        ), p0, p1 in zip(
            empirical_pdf,
            hypothesis0,
            hypothesis1,
            strict=True,
        )
    )
    return sample_count * llr_per_pair


def _gsprt_decision(
    config: GSPRTConfig,
    *,
    llr: float,
    pair_count: int,
) -> str:
    if llr <= config.lower_bound:
        return "accept_h0"
    if llr >= config.upper_bound:
        return "accept_h1"
    if pair_count >= config.max_pairs:
        return "max_pairs"
    return "continue"


@dataclasses.dataclass(frozen=True)
class GSPRTState:
    config: GSPRTConfig
    counts: tuple[int, int, int, int, int] = (0, 0, 0, 0, 0)
    llr: float = 0.0
    decision: str = "continue"

    def __post_init__(self) -> None:
        stats = PentanomialStats(self.counts)
        if stats.pair_count > self.config.max_pairs:
            raise ValueError("GSPRT state exceeds max_pairs.")
        if not math.isfinite(self.llr):
            raise ValueError("GSPRT llr must be finite.")
        if self.decision not in GSPRT_DECISIONS:
            raise ValueError(f"Unknown GSPRT decision: {self.decision!r}")
        expected_llr = pentanomial_gsprt_llr(
            self.counts,
            elo0=self.config.elo0,
            elo1=self.config.elo1,
            elo_model=self.config.elo_model,
        )
        if not math.isclose(
            self.llr,
            expected_llr,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError(
                f"GSPRT llr/count mismatch: {self.llr} != {expected_llr}"
            )
        expected_decision = _gsprt_decision(
            self.config,
            llr=self.llr,
            pair_count=stats.pair_count,
        )
        if self.decision != expected_decision:
            raise ValueError(
                "GSPRT decision/state mismatch: "
                f"{self.decision!r} != {expected_decision!r}"
            )

    @property
    def stats(self) -> PentanomialStats:
        return PentanomialStats(self.counts)

    @property
    def pair_count(self) -> int:
        return self.stats.pair_count

    @property
    def terminal(self) -> bool:
        return self.decision != "continue"

    def update(self, pair_score: float) -> GSPRTState:
        """Consume exactly one completed pair and evaluate boundaries."""
        if self.terminal:
            raise RuntimeError(
                f"Cannot update terminal GSPRT state: {self.decision}"
            )
        index = _pair_score_index(pair_score)
        counts = list(self.counts)
        counts[index] += 1
        llr = pentanomial_gsprt_llr(
            counts,
            elo0=self.config.elo0,
            elo1=self.config.elo1,
            elo_model=self.config.elo_model,
        )
        decision = _gsprt_decision(
            self.config,
            llr=llr,
            pair_count=sum(counts),
        )
        return GSPRTState(
            config=self.config,
            counts=tuple(counts),
            llr=llr,
            decision=decision,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "chess-dfm-arena-gsprt-v1",
            "update_unit": "completed_color_reversed_pair",
            "llr_method": "pentanomial_constrained_mle_logistic",
            "promotion_eligible": False,
            "config": dataclasses.asdict(self.config),
            "counts": list(self.counts),
            "llr": self.llr,
            "decision": self.decision,
            "pair_count": self.pair_count,
            "score": self.stats.score if self.pair_count else None,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> GSPRTState:
        if payload.get("schema_version") != "chess-dfm-arena-gsprt-v1":
            raise ValueError("Unsupported GSPRT state schema.")
        if payload.get("update_unit") != "completed_color_reversed_pair":
            raise ValueError("GSPRT state is not pair-boundary bookkeeping.")
        if payload.get("llr_method") != "pentanomial_constrained_mle_logistic":
            raise ValueError("Unsupported GSPRT LLR method.")
        if payload.get("promotion_eligible") is not False:
            raise ValueError(
                "This logistic GSPRT state cannot claim promotion eligibility."
            )
        config_payload = payload.get("config")
        if not isinstance(config_payload, Mapping):
            raise ValueError("GSPRT state is missing config.")
        config = GSPRTConfig(**dict(config_payload))
        counts_raw = payload.get("counts")
        if not isinstance(counts_raw, list) or len(counts_raw) != 5:
            raise ValueError("GSPRT state counts must be a five-item list.")
        state = cls(
            config=config,
            counts=tuple(int(value) for value in counts_raw),
            llr=float(payload["llr"]),
            decision=str(payload["decision"]),
        )
        if int(payload.get("pair_count", -1)) != state.pair_count:
            raise ValueError("GSPRT persisted pair_count mismatch.")
        persisted_score = payload.get("score")
        expected_score = state.stats.score if state.pair_count else None
        if persisted_score != expected_score:
            raise ValueError("GSPRT persisted score mismatch.")
        return state


__all__ = [
    "ArenaGameSpec",
    "DRAW_RESULT",
    "FAILURE_TERMINATIONS",
    "GAME_TERMINATIONS",
    "GSPRTConfig",
    "GSPRTState",
    "GameOutcome",
    "HELDOUT_SPLITS",
    "LEGACY_ACTION_CODEC_CAPABILITY",
    "LEGACY_INCOMPLETE_PROMOTION_COVERAGE_LIMITATION",
    "NORMAL_TERMINATION",
    "OPENING_POOL_SCHEMA",
    "OPENING_SELECTION_ALGORITHM",
    "PENTANOMIAL_LABELS",
    "PENTANOMIAL_PAIR_POINTS",
    "PLY_CAP_TERMINATION",
    "PentanomialStats",
    "ScoreEloInterval",
    "arena_foundation_contract",
    "build_opening_pool",
    "canonicalize_fen",
    "classify_game_outcome",
    "elo_from_score",
    "expected_score_from_elo",
    "load_opening_pool",
    "make_color_reversed_pairs",
    "opening_selection_hash",
    "pair_aware_score_elo_interval",
    "pair_score_for_model",
    "pentanomial_gsprt_llr",
    "pentanomial_stats",
    "save_opening_pool",
]
