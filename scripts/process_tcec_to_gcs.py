#!/usr/bin/env python3
import argparse
import hashlib
import io
import json
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path
from urllib.parse import urlparse

import chess
import chess.pgn
import numpy as np
import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chess_dfm_jax.data.trajectory import DEFAULT_INPUT_FORMAT, TRAJECTORY_V2  # noqa: E402
from chess_dfm_jax.encoding import encode_board  # noqa: E402
from chess_dfm_jax.policy import legal_move_mask, move_to_policy_index  # noqa: E402


DEFAULT_TCEC_EVERYTHING_URL = (
    "https://github.com/TCEC-Chess/tcecgames/releases/download/S28-final/TCEC-everything-compact.zip"
)
EXCLUDED_EVENT_MARKERS = (
    "bonus",
    "test",
    "testing",
    "sanity",
    "kibitzer",
    "viewer submitted",
    "frc",
    "dfrc",
    "fischer random",
)
STANDARD_COMPETITION_PGN_MARKERS = ("compet-traditional",)
SPLITS = ("train", "val", "test")


def _wdl_for_turn(result_str: str, turn: bool) -> tuple[float, float, float] | None:
    if result_str == "1-0":
        white_wdl = (1.0, 0.0, 0.0)
    elif result_str == "0-1":
        white_wdl = (0.0, 0.0, 1.0)
    elif result_str == "1/2-1/2":
        white_wdl = (0.0, 1.0, 0.0)
    else:
        return None
    if turn == chess.WHITE:
        return white_wdl
    return (white_wdl[2], white_wdl[1], white_wdl[0])


def _game_id(game: chess.pgn.Game) -> str:
    headers = game.headers
    parts = [
        headers.get("Event", "?"),
        headers.get("Site", "?"),
        headers.get("Date", "?"),
        headers.get("Round", "?"),
        headers.get("White", "?"),
        headers.get("Black", "?"),
    ]
    return " | ".join(parts)


def _season(headers: chess.pgn.Headers) -> int | None:
    event = headers.get("Event", "")
    marker = "Season "
    if marker not in event:
        return None
    tail = event.split(marker, 1)[1]
    digits = []
    for char in tail:
        if char.isdigit():
            digits.append(char)
        elif digits:
            break
    if not digits:
        return None
    return int("".join(digits))


def _elo(headers: chess.pgn.Headers, key: str) -> int | None:
    try:
        return int(headers.get(key, ""))
    except ValueError:
        return None


def _game_split(game_id: str, *, val_fraction: float, test_fraction: float) -> str:
    bucket = int(hashlib.sha256(game_id.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
    if bucket < test_fraction:
        return "test"
    if bucket < test_fraction + val_fraction:
        return "val"
    return "train"


def should_keep_game(args: argparse.Namespace, game: chess.pgn.Game, filename: str) -> bool:
    headers = game.headers
    if args.standard_competition_only and not any(marker in filename.lower() for marker in STANDARD_COMPETITION_PGN_MARKERS):
        return False
    result = headers.get("Result", "*")
    if result == "*" or _wdl_for_turn(result, chess.WHITE) is None:
        return False
    variant = headers.get("Variant", "Standard")
    if args.require_standard_variant and variant not in ("", "Standard"):
        return False
    if args.require_startpos and (headers.get("SetUp") == "1" or headers.get("FEN")):
        return False
    event = headers.get("Event", "")
    event_lower = event.lower()
    if args.exclude_event_markers:
        markers = [marker.strip().lower() for marker in args.exclude_event_markers.split(",") if marker.strip()]
        if any(marker in event_lower for marker in markers):
            return False
    season = _season(headers)
    if args.min_season is not None and (season is None or season < args.min_season):
        return False
    if args.max_season is not None and (season is None or season > args.max_season):
        return False
    if args.min_elo > 0:
        white_elo = _elo(headers, "WhiteElo")
        black_elo = _elo(headers, "BlackElo")
        if white_elo is None or black_elo is None or min(white_elo, black_elo) < args.min_elo:
            return False
    return True


def _run_gcloud_cp(source: str, destination: str) -> None:
    subprocess.run(["gcloud", "storage", "cp", source, destination, "--quiet"], check=True)


def _write_json_gcs(payload: dict, destination: str) -> None:
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        tmp_name = handle.name
    try:
        _run_gcloud_cp(tmp_name, destination)
    finally:
        Path(tmp_name).unlink(missing_ok=True)


def _archive_name(url: str) -> str:
    path = urlparse(url).path
    return Path(path).name or "tcec_download.zip"


def _gcs_exists(uri: str) -> bool:
    result = subprocess.run(["gcloud", "storage", "ls", uri], capture_output=True, text=True, check=False)
    return result.returncode == 0


def ensure_archive(args: argparse.Namespace) -> Path:
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    archive_name = args.archive_name or _archive_name(args.url)
    zip_path = cache_dir / archive_name
    mirror_uri = f"{args.cache_gcs.rstrip('/')}/{archive_name}" if args.cache_gcs else ""

    if zip_path.exists() and zip_path.stat().st_size > 0:
        print(f"Using cached TCEC archive {zip_path}")
        return zip_path
    if mirror_uri and _gcs_exists(mirror_uri):
        print(f"Downloading TCEC archive from GCS cache {mirror_uri}")
        _run_gcloud_cp(mirror_uri, str(zip_path))
        return zip_path

    print(f"Downloading TCEC zip from {args.url}")
    with requests.get(args.url, stream=True) as response:
        response.raise_for_status()
        with open(zip_path, "wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
    if mirror_uri:
        print(f"Uploading TCEC archive to GCS cache {mirror_uri}")
        _run_gcloud_cp(str(zip_path), mirror_uri)
    return zip_path


def _flush_chunk(chunk_buffer: list[dict[str, np.ndarray]], out_path: str) -> None:
    if not chunk_buffer:
        return
    np.savez_compressed(
        out_path,
        schema_version=np.asarray(TRAJECTORY_V2),
        planes_t=np.stack([s["planes_t"] for s in chunk_buffer]),
        actions=np.stack([s["actions"] for s in chunk_buffer]),
        planes_future=np.stack([s["planes_future"] for s in chunk_buffer]),
        future_valid=np.stack([s["future_valid"] for s in chunk_buffer]),
        legal_masks=np.stack([s["legal_masks"] for s in chunk_buffer]),
        value_targets=np.stack([s["value_targets"] for s in chunk_buffer]),
        wdl_targets=np.stack([s["wdl_targets"] for s in chunk_buffer]),
        source=np.asarray([s["source"].item() for s in chunk_buffer]),
        game_id=np.asarray([s["game_id"].item() for s in chunk_buffer]),
        ply=np.asarray([s["ply"].item() for s in chunk_buffer], dtype=np.int32),
        result=np.asarray([s["result"].item() for s in chunk_buffer]),
        fen_t=np.asarray([s["fen_t"].item() for s in chunk_buffer]),
        input_format=np.asarray([s["input_format"].item() for s in chunk_buffer]),
        actions_uci=np.stack([s["actions_uci"] for s in chunk_buffer]),
    )


def slice_game(game: chess.pgn.Game, horizon: int, *, source: str = "tcec"):
    result_str = game.headers.get("Result", "*")
    if _wdl_for_turn(result_str, chess.WHITE) is None:
        return

    moves = list(game.mainline_moves())
    game_id = _game_id(game)
    board_at_t = game.board()

    for t in range(len(moves) - horizon + 1):
        current_board = board_at_t.copy(stack=False)

        rollout_board = current_board.copy(stack=False)
        actions = []
        actions_uci = []
        legal_masks = []
        planes_future = []
        value_targets = []
        wdl_targets = []
        for offset in range(horizon):
            legal_masks.append(legal_move_mask(rollout_board, "lc0_1858").astype(np.float32))
            move = moves[t + offset]
            try:
                action_idx = move_to_policy_index(move, "lc0_1858")
            except Exception:
                actions = []
                break
            if move not in rollout_board.legal_moves:
                actions = []
                break
            actions.append(action_idx)
            actions_uci.append(move.uci())
            rollout_board.push(move)
            planes_future.append(
                encode_board(rollout_board, [], input_format=DEFAULT_INPUT_FORMAT).astype(np.float32)
            )
            wdl = _wdl_for_turn(result_str, rollout_board.turn)
            if wdl is None:
                actions = []
                break
            wdl_targets.append(np.asarray(wdl, dtype=np.float32))
            value_targets.append(np.asarray(wdl[0] - wdl[2], dtype=np.float32))

        if not actions:
            board_at_t.push(moves[t])
            continue

        yield {
            "planes_t": encode_board(
                current_board, [], input_format=DEFAULT_INPUT_FORMAT
            ).astype(np.float32),
            "actions": np.asarray(actions, dtype=np.int32),
            "planes_future": np.stack(planes_future, axis=0).astype(np.float32),
            "future_valid": np.ones((horizon,), dtype=np.float32),
            "legal_masks": np.stack(legal_masks, axis=0).astype(np.float32),
            "value_targets": np.stack(value_targets, axis=0).astype(np.float32),
            "wdl_targets": np.stack(wdl_targets, axis=0).astype(np.float32),
            "source": np.asarray(source),
            "game_id": np.asarray(game_id),
            "ply": np.asarray(t, dtype=np.int32),
            "result": np.asarray(result_str),
            "fen_t": np.asarray(current_board.fen()),
            "input_format": np.asarray(DEFAULT_INPUT_FORMAT),
            "actions_uci": np.asarray(actions_uci),
        }
        board_at_t.push(moves[t])


class SplitWriters:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.buffers: dict[str, list[dict[str, np.ndarray]]] = {split: [] for split in SPLITS}
        self.chunk_indices: dict[str, int] = {split: args.start_chunk_index for split in SPLITS}
        self.sample_counts: dict[str, int] = {split: 0 for split in SPLITS}
        self.chunk_counts: dict[str, int] = {split: 0 for split in SPLITS}
        self.out_dir = Path(args.local_out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def add(self, split: str, sample: dict[str, np.ndarray]) -> bool:
        buffer = self.buffers[split]
        buffer.append(sample)
        if len(buffer) < self.args.batch_size:
            return False
        self.flush(split)
        return self.done

    @property
    def done(self) -> bool:
        return self.total_chunks >= self.args.max_chunks

    @property
    def total_chunks(self) -> int:
        return sum(self.chunk_counts.values())

    def split_prefix(self, split: str) -> str:
        if self.args.upload_gcs:
            return f"{self.args.upload_gcs.rstrip('/')}/{split}"
        return ""

    def flush(self, split: str) -> None:
        if self.done:
            return
        buffer = self.buffers[split]
        if not buffer:
            return
        chunk_index = self.chunk_indices[split]
        local_dir = self.out_dir / split
        local_dir.mkdir(parents=True, exist_ok=True)
        out_path = local_dir / f"chunk_{chunk_index:06d}.npz"
        _flush_chunk(buffer, str(out_path))
        if self.args.upload_gcs:
            _run_gcloud_cp(str(out_path), f"{self.split_prefix(split)}/chunk_{chunk_index:06d}.npz")
            if not self.args.keep_local:
                out_path.unlink(missing_ok=True)
        print(f"saved split={split} chunk={chunk_index} samples={len(buffer)}")
        sys.stdout.flush()
        self.sample_counts[split] += len(buffer)
        self.chunk_counts[split] += 1
        self.chunk_indices[split] += 1
        self.buffers[split] = []

    def flush_all(self) -> None:
        for split in SPLITS:
            if self.done and len(self.buffers[split]) < self.args.batch_size:
                continue
            self.flush(split)


def process_tcec(args):
    zip_path = ensure_archive(args)
    started = time.time()
    writers = SplitWriters(args)
    games_seen = 0
    games_kept = 0
    games_by_split = {split: 0 for split in SPLITS}
    samples_by_split = {split: 0 for split in SPLITS}

    with zipfile.ZipFile(zip_path) as archive:
        for filename in archive.namelist():
            if not filename.endswith(".pgn"):
                continue
            if args.include_pgn_marker and args.include_pgn_marker.lower() not in filename.lower():
                continue
            print(f"Processing {filename}...")
            with archive.open(filename) as handle:
                text_stream = io.TextIOWrapper(handle, encoding="utf-8", errors="ignore")
                while True:
                    try:
                        game = chess.pgn.read_game(text_stream)
                    except Exception as exc:
                        print(f"Error parsing game: {exc}")
                        continue

                    if game is None:
                        break

                    games_seen += 1
                    if not should_keep_game(args, game, filename):
                        continue
                    games_kept += 1
                    game_id = _game_id(game)
                    split = _game_split(game_id, val_fraction=args.val_fraction, test_fraction=args.test_fraction)
                    games_by_split[split] += 1

                    for sample in slice_game(game, args.horizon, source="tcec"):
                        samples_by_split[split] += 1
                        if writers.add(split, sample):
                            break
                    if writers.done:
                        print(f"Reached max chunks {args.max_chunks}")
                        break
                if writers.done:
                    break
            if writers.done:
                break

    writers.flush_all()
    manifest = {
        "schema_version": TRAJECTORY_V2,
        "source": "tcec",
        "url": args.url,
        "archive_cache_path": str(zip_path),
        "archive_cache_gcs": args.cache_gcs,
        "horizon": args.horizon,
        "samples_per_chunk": args.batch_size,
        "max_chunks": args.max_chunks,
        "chunks_written": writers.chunk_counts,
        "samples_written": writers.sample_counts,
        "samples_seen_before_chunking": samples_by_split,
        "games_seen": games_seen,
        "games_kept": games_kept,
        "games_by_split": games_by_split,
        "splits": {
            "train": 1.0 - args.val_fraction - args.test_fraction,
            "val": args.val_fraction,
            "test": args.test_fraction,
        },
        "filters": {
            "standard_competition_only": args.standard_competition_only,
            "include_pgn_marker": args.include_pgn_marker,
            "exclude_event_markers": args.exclude_event_markers,
            "min_season": args.min_season,
            "max_season": args.max_season,
            "min_elo": args.min_elo,
            "require_standard_variant": args.require_standard_variant,
            "require_startpos": args.require_startpos,
        },
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_s": time.time() - started,
    }
    local_manifest = Path(args.local_out_dir) / "manifest.json"
    local_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.upload_gcs:
        _write_json_gcs(manifest, f"{args.upload_gcs.rstrip('/')}/manifest.json")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--url",
        type=str,
        default=DEFAULT_TCEC_EVERYTHING_URL,
    )
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--max-chunks", type=int, default=10)
    parser.add_argument("--start-chunk-index", type=int, default=0)
    parser.add_argument("--upload-gcs", type=str, required=True)
    parser.add_argument("--cache-dir", type=str, default="/tmp/chess_dfm_data_cache/tcec")
    parser.add_argument("--cache-gcs", type=str, default="")
    parser.add_argument("--archive-name", type=str, default="")
    parser.add_argument("--local-out-dir", type=str, default="/tmp/tcec_chunks")
    parser.add_argument("--keep-local", action="store_true")
    parser.add_argument("--val-fraction", type=float, default=0.05)
    parser.add_argument("--test-fraction", type=float, default=0.05)
    parser.add_argument("--standard-competition-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-pgn-marker", type=str, default="compet-traditional")
    parser.add_argument("--exclude-event-markers", type=str, default=",".join(EXCLUDED_EVENT_MARKERS))
    parser.add_argument("--min-season", type=int, default=20)
    parser.add_argument("--max-season", type=int, default=None)
    parser.add_argument("--min-elo", type=int, default=0)
    parser.add_argument("--require-standard-variant", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--require-startpos", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()
    if args.val_fraction < 0 or args.test_fraction < 0 or args.val_fraction + args.test_fraction >= 1:
        raise SystemExit("--val-fraction and --test-fraction must be non-negative and sum to < 1.")
    process_tcec(args)
