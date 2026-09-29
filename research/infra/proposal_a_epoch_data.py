"""Content-bound full-corpus staging for the promoted Torch epoch on Modal.

The default command only verifies local inputs. --upload copies those inputs
to the existing Modal training Volume; neither mode starts a GPU.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any


DATASET_LABEL = "lc0-sequential-test80-20240401-0117-full-v1"
INVENTORY_SHA256 = "864afad45689a19a55962bef3cc39a59cbf7450614c19d415bf2aecdd2a573cb"
INVENTORY_FILES = 11_431
INVENTORY_BYTES = 8_635_794_032
PREREGISTER_RELATIVE = "research/analysis/proposal_a_one_epoch_preregister_20260810.json"
PREREGISTER_SHA256 = "ed9d5c0d05e995085686442f1530d18b67d59599a7b1ce870d9782c52ead2cf2"
REPO_ROOT = Path(__file__).resolve().parents[2]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_preregister(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    path = repo_root / PREREGISTER_RELATIVE
    if sha256_file(path) != PREREGISTER_SHA256:
        raise RuntimeError("Promoted Proposal A preregistration checksum drift")
    return json.loads(path.read_text(encoding="utf-8"))


def verify_dataset(root: Path) -> dict[str, Any]:
    root = root.resolve(strict=True)

    def row(path: Path) -> dict[str, Any] | None:
        if path.is_symlink():
            raise RuntimeError(f"Dataset may not contain symlinks: {path}")
        if path.is_file():
            return {
                "path": path.relative_to(root).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        return None

    # Preserve the frozen lexicographic inventory order while overlapping
    # independent reads. Serial per-file Volume I/O is costly for 11k files.
    with ThreadPoolExecutor(max_workers=16) as pool:
        rows = [item for item in pool.map(row, sorted(root.rglob("*"))) if item is not None]
    digest = hashlib.sha256(json.dumps(
        rows, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")).hexdigest()
    size = sum(row["size_bytes"] for row in rows)
    if (len(rows), size, digest) != (INVENTORY_FILES, INVENTORY_BYTES, INVENTORY_SHA256):
        raise RuntimeError("Full LC0 dataset inventory drift")
    return {
        "schema_version": "chess-dfm-modal-lc0-sequential-full-v1",
        "dataset_label": DATASET_LABEL,
        "inventory_sha256": digest,
        "inventory_file_count": len(rows),
        "inventory_size_bytes": size,
        "preregister_sha256": PREREGISTER_SHA256,
    }


def copy_verified_dataset(source: Path, destination: Path) -> dict[str, Any]:
    """Copy once in parallel, hash local bytes, and publish only exact inputs."""
    source = source.resolve(strict=True)
    if destination.exists():
        raise FileExistsError(f"Refusing to replace local dataset: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    required_free = INVENTORY_BYTES + 12 * 1024**3
    if shutil.disk_usage(destination.parent).free < required_free:
        raise RuntimeError("Container disk reserve too small for full corpus")
    with tempfile.TemporaryDirectory(prefix=".epoch-copy-", dir=destination.parent) as temporary:
        staged = Path(temporary)

        def copy(path: Path) -> bool:
            if path.is_symlink():
                raise RuntimeError(f"Dataset may not contain symlinks: {path}")
            if not path.is_file():
                return False
            target = staged / path.relative_to(source)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
            return True

        copied = 0
        with ThreadPoolExecutor(max_workers=16) as pool:
            for is_file in pool.map(copy, sorted(source.rglob("*"))):
                copied += int(is_file)
                if is_file and copied % 1000 == 0:
                    print(json.dumps({"data_preflight": "copying", "copied_files": copied}), flush=True)
        marker = verify_dataset(staged)
        os.replace(staged, destination)
    print(json.dumps({"data_preflight": "verified_local_copy", **marker}), flush=True)
    return marker


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=(
        REPO_ROOT / "data/lc0_sequential/test80_20240401_0117"
    ))
    parser.add_argument("--upload", action="store_true")
    args = parser.parse_args()
    load_preregister()
    marker = verify_dataset(args.data_root)
    print(json.dumps(marker, sort_keys=True), flush=True)
    if not args.upload:
        return
    from research.interpretability.remote_cost import modal_month_spend

    # Account for provider-waived storage exactly as the training launcher does.
    if modal_month_spend() >= 34:
        raise RuntimeError("Modal cost guard refused input staging at the $34 stop line")
    common = ["modal", "volume", "put", "--env", "chess-dfm-research",
              "chess-dfm-training-inputs"]
    subprocess.run([
        *common, str(args.data_root.resolve()), f"/{DATASET_LABEL}/lc0_sequential",
    ], check=True)
    # Publish the completion marker only after the entire directory upload.
    # No --force: an existing staged dataset cannot be overwritten silently.
    with tempfile.TemporaryDirectory(prefix="proposal-a-stage-") as temporary:
        marker_path = Path(temporary) / "stage.json"
        marker_path.write_text(json.dumps(marker, indent=2) + "\n", encoding="utf-8")
        subprocess.run([
            *common, str(marker_path), f"/{DATASET_LABEL}/.stage.json",
        ], check=True)


if __name__ == "__main__":
    main()
