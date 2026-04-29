#!/usr/bin/env python3
"""Download LC0 training PGN archives, convert trajectories, and upload to GCS."""

from __future__ import annotations

import argparse
import bz2
import hashlib
import io
import json
import random
import re
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path
from urllib.parse import urljoin

import chess
import chess.pgn
import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chess_dfm_jax.data.trajectory import TRAJECTORY_V2  # noqa: E402
from scripts.process_tcec_to_gcs import SplitWriters, _game_id, _game_split, slice_game  # noqa: E402


DEFAULT_INDEX_URL = "https://data.lczero.org/files/training_pgns/"


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


def _gcs_exists(uri: str) -> bool:
    result = subprocess.run(["gcloud", "storage", "ls", uri], capture_output=True, text=True, check=False)
    return result.returncode == 0


def discover_archive_urls(index_url: str, pattern: str, limit: int) -> list[str]:
    response = requests.get(index_url, timeout=120)
    response.raise_for_status()
    hrefs = re.findall(r'href=["\']([^"\']+)["\']', response.text)
    regex = re.compile(pattern)
    urls = []
    for href in hrefs:
        if not href.endswith((".tar.bz2", ".pgn.bz2", ".pgn")):
            continue
        filename = href.rsplit("/", 1)[-1]
        if not regex.search(filename):
            continue
        urls.append(urljoin(index_url, href))
    urls = sorted(set(urls))
    if limit > 0:
        urls = urls[:limit]
    return urls


def index_url_with_subdir(index_url: str, subdir: str) -> str:
    if not subdir:
        return index_url
    return urljoin(index_url.rstrip("/") + "/", subdir.strip("/") + "/")


def explicit_or_discovered_urls(args: argparse.Namespace) -> list[str]:
    urls = []
    if args.urls:
        urls.extend(url.strip() for url in args.urls.split(",") if url.strip())
    if args.url_file:
        urls.extend(line.strip() for line in Path(args.url_file).read_text(encoding="utf-8").splitlines() if line.strip())
    if not urls:
        urls = discover_archive_urls(index_url_with_subdir(args.index_url, args.subdir), args.filename_pattern, args.max_archives)
    if not urls:
        raise ValueError("No LC0 PGN archive URLs found. Pass --urls, --url-file, or adjust --filename-pattern.")
    return urls


def iter_pgn_texts_from_archive(url: str, cache_dir: Path):
    cache_dir.mkdir(parents=True, exist_ok=True)
    local_path = cache_dir / url.rsplit("/", 1)[-1]
    mirror_uri = ""
    cache_gcs = getattr(iter_pgn_texts_from_archive, "cache_gcs", "")
    if cache_gcs:
        mirror_uri = f"{cache_gcs.rstrip('/')}/{local_path.name}"
    if not local_path.exists():
        if mirror_uri and _gcs_exists(mirror_uri):
            print(f"downloading archive from GCS cache {mirror_uri}")
            _run_gcloud_cp(mirror_uri, str(local_path))
        else:
            print(f"downloading {url}")
            sys.stdout.flush()
            partial_path = local_path.with_suffix(local_path.suffix + ".part")
            for attempt in range(8):
                try:
                    with requests.get(url, stream=True, timeout=120) as response:
                        if response.status_code in {429, 500, 502, 503, 504}:
                            retry_after = response.headers.get("Retry-After")
                            if retry_after and retry_after.isdigit():
                                sleep_s = float(retry_after)
                            else:
                                sleep_s = min(300.0, 10.0 * (2**attempt)) + random.uniform(0.0, 10.0)
                            print(
                                f"download throttled status={response.status_code} attempt={attempt + 1}/8 "
                                f"sleep_s={sleep_s:.1f} url={url}"
                            )
                            sys.stdout.flush()
                            time.sleep(sleep_s)
                            continue
                        response.raise_for_status()
                        with partial_path.open("wb") as handle:
                            for chunk in response.iter_content(chunk_size=1024 * 1024):
                                if chunk:
                                    handle.write(chunk)
                        partial_path.replace(local_path)
                        break
                except requests.RequestException as exc:
                    if attempt == 7:
                        raise
                    sleep_s = min(300.0, 10.0 * (2**attempt)) + random.uniform(0.0, 10.0)
                    print(f"download error attempt={attempt + 1}/8 sleep_s={sleep_s:.1f} url={url}: {exc}")
                    sys.stdout.flush()
                    time.sleep(sleep_s)
            if not local_path.exists():
                raise RuntimeError(f"Failed to download archive after retries: {url}")
            if mirror_uri:
                print(f"uploading archive to GCS cache {mirror_uri}")
                _run_gcloud_cp(str(local_path), mirror_uri)

    if local_path.name.endswith(".tar.bz2"):
        with tarfile.open(local_path, "r:bz2") as archive:
            for member in archive.getmembers():
                if not member.isfile() or not member.name.endswith(".pgn"):
                    continue
                extracted = archive.extractfile(member)
                if extracted is None:
                    continue
                yield member.name, io.TextIOWrapper(extracted, encoding="utf-8", errors="ignore")
    elif local_path.name.endswith(".pgn.bz2"):
        yield local_path.name, io.TextIOWrapper(bz2.open(local_path, "rb"), encoding="utf-8", errors="ignore")
    elif local_path.name.endswith(".pgn"):
        yield local_path.name, local_path.open("r", encoding="utf-8", errors="ignore")
    else:
        raise ValueError(f"Unsupported PGN archive type: {local_path}")


def keep_lc0_game(game: chess.pgn.Game, args: argparse.Namespace) -> bool:
    result = game.headers.get("Result", "*")
    if result not in ("1-0", "0-1", "1/2-1/2"):
        return False
    variant = game.headers.get("Variant", "Standard")
    if args.require_standard_variant and variant not in ("", "Standard"):
        return False
    if args.require_startpos and (game.headers.get("SetUp") == "1" or game.headers.get("FEN")):
        return False
    return True


def process_lc0(args: argparse.Namespace) -> None:
    started = time.time()
    urls = explicit_or_discovered_urls(args)
    iter_pgn_texts_from_archive.cache_gcs = args.cache_gcs
    writers = SplitWriters(args)
    cache_dir = Path(args.cache_dir)
    games_seen = 0
    games_kept = 0
    games_by_split = {split: 0 for split in ("train", "val", "test")}
    samples_by_split = {split: 0 for split in ("train", "val", "test")}

    for url in urls:
        if writers.done:
            break
        for pgn_name, text_stream in iter_pgn_texts_from_archive(url, cache_dir):
            print(f"processing url={url} pgn={pgn_name}")
            sys.stdout.flush()
            with text_stream:
                while not writers.done:
                    try:
                        game = chess.pgn.read_game(text_stream)
                    except Exception as exc:
                        print(f"error parsing game from {pgn_name}: {exc}")
                        continue
                    if game is None:
                        break
                    games_seen += 1
                    if not keep_lc0_game(game, args):
                        continue
                    games_kept += 1
                    game_id = f"{url} | {_game_id(game)} | {games_seen}"
                    split = _game_split(game_id, val_fraction=args.val_fraction, test_fraction=args.test_fraction)
                    games_by_split[split] += 1
                    for sample in slice_game(game, args.horizon, source="lc0_pgn"):
                        samples_by_split[split] += 1
                        if writers.add(split, sample):
                            break

    writers.flush_all()
    manifest = {
        "schema_version": TRAJECTORY_V2,
        "source": "lc0_pgn",
        "index_url": index_url_with_subdir(args.index_url, args.subdir),
        "archive_cache_dir": args.cache_dir,
        "archive_cache_gcs": args.cache_gcs,
        "urls": urls,
        "url_count": len(urls),
        "filename_pattern": args.filename_pattern,
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
            "require_standard_variant": args.require_standard_variant,
            "require_startpos": args.require_startpos,
        },
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_s": time.time() - started,
        "url_digest": hashlib.sha256("\n".join(urls).encode("utf-8")).hexdigest(),
    }
    local_manifest = Path(args.local_out_dir) / "manifest.json"
    local_manifest.parent.mkdir(parents=True, exist_ok=True)
    local_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.upload_gcs:
        _write_json_gcs(manifest, f"{args.upload_gcs.rstrip('/')}/manifest.json")
    print(json.dumps(manifest, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-url", default=DEFAULT_INDEX_URL)
    parser.add_argument("--subdir", default="test80", help="LC0 training_pgns subdirectory, e.g. test80, run2, bt4rl.")
    parser.add_argument("--filename-pattern", default=r".*\.tar\.bz2$")
    parser.add_argument("--urls", default="", help="Comma-separated PGN archive URLs. Overrides discovery.")
    parser.add_argument("--url-file", default="", help="File containing one PGN archive URL per line.")
    parser.add_argument("--max-archives", type=int, default=4)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--max-chunks", type=int, default=1024)
    parser.add_argument("--start-chunk-index", type=int, default=0)
    parser.add_argument("--upload-gcs", required=True)
    parser.add_argument("--local-out-dir", default="/tmp/lc0_pgn_chunks")
    parser.add_argument("--cache-dir", default="/tmp/lc0_pgn_cache")
    parser.add_argument("--cache-gcs", default="")
    parser.add_argument("--keep-local", action="store_true")
    parser.add_argument("--val-fraction", type=float, default=0.05)
    parser.add_argument("--test-fraction", type=float, default=0.05)
    parser.add_argument("--require-standard-variant", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--require-startpos", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()
    if args.val_fraction < 0 or args.test_fraction < 0 or args.val_fraction + args.test_fraction >= 1:
        raise SystemExit("--val-fraction and --test-fraction must be non-negative and sum to < 1.")
    return args


if __name__ == "__main__":
    process_lc0(parse_args())
