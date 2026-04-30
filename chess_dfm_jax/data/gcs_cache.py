"""Local GCS shard cache with background prefetching."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import os
from pathlib import Path
import random
import subprocess
import threading
import time


def _run_gcloud(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, capture_output=True, text=True, check=False)


def split_gcs_prefixes(prefix: str) -> list[str]:
    """Return non-empty comma-separated GCS prefixes."""
    return [part.strip() for part in prefix.split(",") if part.strip()]


def list_gcs_npz(prefix: str) -> list[str]:
    uris: list[str] = []
    for part in split_gcs_prefixes(prefix):
        uri = part.rstrip("/") + "/*.npz"
        result = _run_gcloud(["gcloud", "storage", "ls", uri])
        if result.returncode != 0:
            continue
        uris.extend(line.strip() for line in result.stdout.splitlines() if line.strip().endswith(".npz"))
    return sorted(set(uris))


def local_name_for_uri(uri: str) -> str:
    # Include a short directory hash so train/chunk_000000 and val/chunk_000000 cannot collide.
    parent = uri.rsplit("/", 1)[0]
    filename = uri.rsplit("/", 1)[-1]
    suffix = hashlib.sha1(parent.encode("utf-8")).hexdigest()[:8]
    return f"{suffix}_{filename}"


@dataclass
class GCSShardCache:
    """Maintain a local cache of immutable GCS .npz shards."""

    gcs_prefix: str
    cache_dir: str | Path
    poll_interval_s: int = 60
    max_cached_shards: int = 0
    download_workers: int = 2
    shuffle_downloads: bool = True
    seed: int = 0
    _stop_event: threading.Event = field(default_factory=threading.Event, init=False)
    _thread: threading.Thread | None = field(default=None, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)
    _remote_seen: int = field(default=0, init=False)
    _downloaded: int = field(default=0, init=False)
    _failed: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self.cache_path = Path(self.cache_dir)
        self.cache_path.mkdir(parents=True, exist_ok=True)
        self.rng = random.Random(self.seed)

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._loop, name=f"gcs-cache-{self.cache_path.name}", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def local_paths(self) -> list[str]:
        return sorted(str(path) for path in self.cache_path.glob("*.npz"))

    def wait_for_minimum(self, min_shards: int, timeout_s: int = 1800) -> None:
        deadline = time.time() + timeout_s
        self.refresh_once()
        while len(self.local_paths()) < min_shards:
            if time.time() >= deadline:
                raise TimeoutError(
                    f"Timed out waiting for {min_shards} cached shards from {self.gcs_prefix}; "
                    f"have {len(self.local_paths())}."
                )
            time.sleep(min(5, self.poll_interval_s))
            self.refresh_once()

    def wait_for_all_visible(self, timeout_s: int = 7200) -> None:
        """Block until every currently visible remote shard is cached locally."""
        deadline = time.time() + timeout_s
        self.refresh_once()
        while True:
            stats = self.stats()
            if stats["remote_seen"] > 0 and stats["cached"] >= stats["remote_seen"]:
                return
            if time.time() >= deadline:
                raise TimeoutError(
                    f"Timed out waiting for all shards from {self.gcs_prefix}; "
                    f"have {stats['cached']} of {stats['remote_seen']} visible shards."
                )
            # Keep pulling immediately while there is known work. The background loop
            # is intentionally slow; startup prefill should not wait a full poll cycle.
            self.refresh_once()

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {
                "remote_seen": self._remote_seen,
                "cached": len(self.local_paths()),
                "downloaded": self._downloaded,
                "failed": self._failed,
            }

    def refresh_once(self) -> None:
        remote = list_gcs_npz(self.gcs_prefix)
        with self._lock:
            self._remote_seen = len(remote)
        if self.shuffle_downloads:
            self.rng.shuffle(remote)

        to_download = []
        for uri in remote:
            local = self.cache_path / local_name_for_uri(uri)
            if local.exists():
                continue
            to_download.append((uri, local))

        if self.max_cached_shards > 0:
            remaining = max(0, self.max_cached_shards - len(self.local_paths()))
            to_download = to_download[:remaining]
        if not to_download:
            return

        workers = max(1, self.download_workers)
        threads = []
        for uri, local in to_download[:workers]:
            thread = threading.Thread(target=self._download_one, args=(uri, local), daemon=True)
            thread.start()
            threads.append(thread)
        for thread in threads:
            thread.join()

    def _download_one(self, uri: str, local: Path) -> None:
        tmp = local.with_suffix(local.suffix + ".tmp")
        tmp.unlink(missing_ok=True)
        result = _run_gcloud(["gcloud", "storage", "cp", uri, str(tmp)])
        if result.returncode == 0 and tmp.exists():
            os.replace(tmp, local)
            with self._lock:
                self._downloaded += 1
            return
        tmp.unlink(missing_ok=True)
        with self._lock:
            self._failed += 1

    def _loop(self) -> None:
        while not self._stop_event.is_set():
            self.refresh_once()
            self._stop_event.wait(self.poll_interval_s)


__all__ = ["GCSShardCache", "list_gcs_npz", "local_name_for_uri", "split_gcs_prefixes"]
