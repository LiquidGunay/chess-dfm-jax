"""History-only reconstruction of the retained Hero trajectory windows.

No rows/labels are synthesized or dropped. Missing *prefix moves* can be
recovered only by a unique legal one-ply transition to a retained exact FEN.
Original source files and their ordering remain immutable.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import random
from concurrent.futures import ProcessPoolExecutor, as_completed

import chess
import numpy as np

from chess_dfm_jax.encoding import encode_board, _repetition_key

SCHEMA = "hero-trajectory-full-history-v1"
TRAIN_SHARDS = 27679
TRAIN_STEPS = 554
TRAIN_MANIFEST_SHA256 = "9389b98f7597acd9e4498d0fcc35903f0ff9922aa44c5b777c881cecc1f9a293"
META_KEYS = ("ply", "fen_t", "actions_uci", "result", "input_format", "source_uri")


class UnresolvedHistory(ValueError):
    """An omitted move, not a corrupt record; strict rebuilding still fails."""

    def __init__(self, message, affected_rows, missing_ply):
        super().__init__(message)
        self.affected_rows = list(affected_rows)
        self.missing_ply = int(missing_ply)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def training_order():
    result = list(range(TRAIN_SHARDS))
    random.Random(0).shuffle(result)
    return result[:TRAIN_STEPS]


def load_npz(path, *, metadata_only=False):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in (META_KEYS if metadata_only else data.files)}


def initial_board(fen):
    standard = chess.Board()
    if fen == standard.fen():
        return standard
    board = chess.Board(fen, chess960=True)
    if board.chess960_pos() is None:
        raise ValueError(f"Unverified setup root; need original game history: {fen}")
    if board.fen() != fen:
        raise ValueError("Noncanonical initial FEN")
    return board


class IncrementalHistoryEncoder:
    """Classical 112 planes, with prefix-only repetition and orientation.

    The reference encoder recomputes repetition over the entire prefix. This
    caches per-position piece planes and counts, leaving its current/auxiliary
    encoding unchanged. Parity checks below compare against the reference.
    """
    def __init__(self):
        self.pieces = []
        self.repetitions = []
        self.seen = Counter()

    def append(self, board):
        result = encode_board(board, []).astype(np.uint8)
        key = _repetition_key(board)
        repetition = int(self.seen[key] > 0)
        self.seen[key] += 1
        self.pieces.append(result[:12].copy())
        self.repetitions.append(repetition)
        for offset in range(min(8, len(self.pieces))):
            index = len(self.pieces) - 1 - offset
            pieces = self.pieces[index]
            if offset % 2:
                pieces = np.concatenate((pieces[6:], pieces[:6]), axis=0)[:, ::-1, :]
            result[offset*13:offset*13+12] = pieces
            result[offset*13+12] = self.repetitions[index]
        return result


def unique_missing_move(board, next_fen):
    matches = []
    for move in board.legal_moves:
        child = board.copy(stack=False)
        child.push(move)
        if child.fen() == next_fen:
            matches.append(move)
    if len(matches) != 1:
        raise ValueError(f"Missing move has {len(matches)} legal reconstructions; need source PGN")
    return matches[0]


def source_predecessor(previous, current):
    left, right = str(previous["source_uri"]), str(current["source_uri"])
    a, b = Path(left), Path(right)
    try:
        ai, bi = int(a.stem.split("_")[-1]), int(b.stem.split("_")[-1])
    except ValueError as exc:
        raise ValueError("Malformed source shard identity") from exc
    if a.parent != b.parent or ai+1 != bi or ai//10000 != bi//10000:
        raise ValueError("Cannot join history across source set/worker/shard discontinuity")


def reconstruct_shard(path, *, rows=None, max_prefix_shards=16):
    """Return selected original rows with only planes 12:104 changed."""
    path = Path(path)
    target = load_npz(path)
    n = int(target["batch_size"])
    selected = np.arange(n) if rows is None else np.asarray(rows, dtype=np.int64)
    if (selected.ndim != 1 or not len(selected) or np.any(selected < 0)
            or np.any(selected >= n) or np.any(np.diff(selected) <= 0)):
        raise ValueError("Selected rows must be increasing, unique and in range")
    if str(target["schema_version"]) != "trajectory-v3" or str(target["plane_codec"]) != "uint8":
        raise ValueError("Expected original uint8 trajectory-v3 shard")
    if not np.all(target["future_valid_u8"] == 1):
        raise ValueError("Expected full eight-ply legacy windows")
    horizon = target["actions_uci"].shape[1]
    payloads = [target]
    sources = [path]
    # Need predecessor shards only when the first requested game starts before
    # this shard. Do not assume game_id is unique, or join across worker ranges.
    if not np.any(target["ply"][:selected[0]+1] == 0):
        index = int(path.stem.split("_")[-1])
        for back in range(1, max_prefix_shards+1):
            if index-back < 0:
                raise ValueError("Missing prefix at corpus boundary")
            previous_path = path.with_name(f"chunk_{index-back:06d}.npz")
            previous = load_npz(previous_path, metadata_only=True)
            source_predecessor(previous, payloads[0])
            payloads.insert(0, previous)
            sources.insert(0, previous_path)
            if np.any(previous["ply"] == 0):
                break
        else:
            raise ValueError("Prefix exceeds bounded predecessor search")
    selected_set = set(map(int, selected))
    segments, segment = [], None
    for shard, data in enumerate(payloads):
        for row, raw_ply in enumerate(data["ply"]):
            ply = int(raw_ply)
            if ply == 0:
                segment = []
                segments.append(segment)
            if segment is not None:
                segment.append((shard, row, ply))
    result = {key: value[selected].copy() if value.ndim and value.shape[0] == n else value.copy()
              for key, value in target.items()}
    result["batch_size"] = np.asarray(len(selected), dtype=np.int32)
    destination = {int(row): i for i, row in enumerate(selected)}
    reconstructed, gaps, nonstandard, parity = set(), [], 0, 0
    for segment in segments:
        requested = [(row, ply) for shard, row, ply in segment
                     if shard == len(payloads)-1 and row in selected_set]
        if not requested:
            continue
        stop_ply = max(ply for _, ply in requested) + horizon
        moves, fens, results = {}, {}, set()
        previous_ply = -1
        for shard, row, ply in segment:
            if ply <= previous_ply:
                raise ValueError("Nonmonotonic source rows within game")
            previous_ply = ply
            data = payloads[shard]
            if str(data["input_format"][row]) != "INPUT_CLASSICAL_112_PLANE":
                raise ValueError("Unsupported input format")
            fens[ply] = str(data["fen_t"][row])
            results.add(str(data["result"][row]))
            for offset, value in enumerate(data["actions_uci"][row]):
                key, uci = ply + offset, str(value)
                if key in moves and moves[key] != uci:
                    raise ValueError("Conflicting overlapping action windows")
                moves[key] = uci
        if len(results) != 1 or 0 not in fens:
            raise ValueError("Missing root or inconsistent outcome in game")
        board = initial_board(fens[0])
        nonstandard += int(board.chess960)
        history, encoded, encoder = [], [], IncrementalHistoryEncoder()
        check_plies = {0, 1, 7, 8, stop_ply}
        for ply in range(stop_ply+1):
            if ply in fens and board.fen() != fens[ply]:
                raise ValueError(f"FEN mismatch in {path.name} at game ply {ply}")
            history.append(board.copy(stack=False))
            planes = encoder.append(board)
            if ply in check_plies:
                np.testing.assert_array_equal(planes, encode_board(board, history))
                parity += 1
            encoded.append(planes)
            if ply == stop_ply:
                break
            if ply in moves:
                move = board.parse_uci(moves[ply])
            else:
                affected = [row for row, start in requested if start+horizon > ply]
                if ply+1 not in fens:
                    raise UnresolvedHistory(
                        f"More than one missing prefix move at ply {ply}; need source PGN",
                        affected, ply)
                move = unique_missing_move(board, fens[ply+1])
                gaps.append({"game_root_fen": fens[0], "ply": ply, "uci": move.uci(),
                             "before_fen": board.fen(), "after_fen": fens[ply+1]})
            if move not in board.legal_moves:
                raise ValueError("Illegal source move")
            board.push(move)
        for row, ply in requested:
            out = destination[row]
            current, future = encoded[ply], np.stack(encoded[ply+1:ply+horizon+1])
            for expected, original in ((current, target["planes_t_u8"][row]),
                                       (future, target["planes_future_u8"][row])):
                stripped = expected.copy()
                stripped[..., 12:104, :, :] = 0
                np.testing.assert_array_equal(stripped, original, err_msg="Non-history input changed")
            result["planes_t_u8"][out] = current
            result["planes_future_u8"][out] = future
            reconstructed.add(row)
    if reconstructed != selected_set:
        raise ValueError("Not every requested row has a complete verified prefix")
    # All labels and metadata must survive exactly; only row selection and
    # current/future history bytes are permitted to differ.
    for key, value in result.items():
        if key not in {"batch_size", "planes_t_u8", "planes_future_u8"}:
            original = target[key]
            np.testing.assert_array_equal(value, original[selected] if original.ndim and original.shape[0] == n else original)
    record = {"source": path.name, "rows": selected.tolist(),
              "inputs": [{"name": p.name, "sha256": sha256(p), "source_uri": str(d["source_uri"])}
                         for p, d in zip(sources, payloads)],
              "recovered_moves": gaps, "chess960_games": nonstandard,
              "reference_parity_positions": parity, "examples": len(selected)}
    return result, record


def write_npz_once(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        np.savez_compressed(stream, **payload)


def write_json_once(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(record, stream, indent=2, sort_keys=True)
        stream.write("\n")


def _build_one(job):
    source, output, rows = job
    rebuilt, audit = reconstruct_shard(source, rows=rows)
    write_npz_once(output, rebuilt)
    record = {"file": str(Path(output).name), "sha256": sha256(output),
              "size_bytes": Path(output).stat().st_size, **audit}
    write_json_once(str(output)+".audit.json", record)
    return record


def build_dataset(source_root, output_root, indices_path, *, workers=8):
    source_root, output_root = Path(source_root), Path(output_root)
    if output_root.exists():
        raise FileExistsError("Use a new content-bound dataset destination")
    paths = sorted((source_root / "train").glob("*.npz"))
    inventory = "\n".join(f"{p.name}\t{p.stat().st_size}" for p in paths)
    if len(paths) != TRAIN_SHARDS or hashlib.sha256(inventory.encode()).hexdigest() != TRAIN_MANIFEST_SHA256:
        raise ValueError("Original Hero training inventory drift")
    validation_paths = sorted((source_root/"val").glob("*.npz"))
    val_inventory = "\n".join(f"{p.name}\t{p.stat().st_size}" for p in validation_paths)
    if (len(validation_paths) != 1524 or hashlib.sha256(val_inventory.encode()).hexdigest()
            != "c02ecfd114fa1dbf59e0d7189cc32b0aedd5a468c6e24145b131e0e58745b281"):
        raise ValueError("Original Hero validation inventory drift")
    with np.load(indices_path, allow_pickle=False) as data:
        validation = np.sort(data["fast_val_global_index"].astype(np.int64))
    if len(validation) != 8192 or len(np.unique(validation)) != 8192:
        raise ValueError("Original frozen validation pool drift")
    train_jobs = [(paths[index], output_root/"train"/f"batch_{step:06d}.npz", None)
                  for step,index in enumerate(training_order())]
    val_jobs = [(source_root/"val"/f"chunk_{index:06d}.npz",
                 output_root/"validation"/f"chunk_{index:06d}.npz",
                 (validation[validation//1024 == index] % 1024).tolist())
                for index in np.unique(validation//1024)]
    records = {"train": {}, "validation": {}}
    jobs = train_jobs + val_jobs
    output_root.mkdir(parents=True)
    failures = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_build_one, job): job for job in jobs}
        for done, future in enumerate(as_completed(futures), 1):
            job = futures[future]
            split = Path(job[1]).parent.name
            try:
                record = future.result()
                records[split][record["file"]] = record
            except Exception as exc:
                failures.append({"source": str(job[0]), "split": split,
                                 "error": f"{type(exc).__name__}: {exc}"})
                print(json.dumps({"rebuild_failure": failures[-1]}), flush=True)
            if done % 25 == 0:
                print(json.dumps({"rebuild_progress": done, "total": len(jobs),
                                  "failures": len(failures)}), flush=True)
    if failures:
        write_json_once(output_root / "failures.json", failures)
        raise ValueError(f"History rebuild has {len(failures)} unresolved shards; GPU launch forbidden")
    manifest = {
        "schema_version": SCHEMA, "passed": True,
        "history_mode": "lc0_full_history_as_preprocessed",
        "original_training_inventory_sha256": TRAIN_MANIFEST_SHA256,
        "original_training_shards_in_order": training_order(),
        "original_validation_indices_sha256": sha256(indices_path),
        "original_validation_global_indices": validation.tolist(),
        "train": [records["train"][Path(job[1]).name] for job in train_jobs],
        "validation": [records["validation"][Path(job[1]).name] for job in val_jobs],
        "train_examples": TRAIN_STEPS * 1024, "validation_examples": 8192,
        "unchanged": ["row identities/order", "actions", "legal indices", "value/WDL targets",
                      "future validity", "current piece planes", "auxiliary planes"],
        "changed_plane_slice": [12,104],
    }
    verify_dataset(output_root, manifest)
    write_json_once(output_root / "manifest.json", manifest)
    return manifest


def verify_dataset(root, manifest=None):
    root = Path(root)
    if manifest is None:
        manifest = json.loads((root/"manifest.json").read_text())
    if manifest.get("selection_contract") is not None:
        from research.hero_history_subset_data import verify_subset
        return verify_subset(root, manifest)
    if manifest.get("schema_version") != SCHEMA or not manifest.get("passed"):
        raise ValueError("Missing successful history rebuild contract")
    if (manifest["original_training_inventory_sha256"] != TRAIN_MANIFEST_SHA256
            or manifest["original_training_shards_in_order"] != training_order()):
        raise ValueError("History rebuild training identity drift")
    for split, expected in (("train", 567296), ("validation", 8192)):
        if sum(record["examples"] for record in manifest[split]) != expected:
            raise ValueError("History dataset example count drift")
        for record in manifest[split]:
            name = record["file"]
            if Path(name).name != name:
                raise ValueError("Unsafe dataset member")
            path = root/split/name
            if path.stat().st_size != record["size_bytes"] or sha256(path) != record["sha256"]:
                raise ValueError(f"History data content drift: {path}")
    return manifest
