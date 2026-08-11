"""Execute or exactly resume the promoted Proposal A epoch on a Runpod Pod."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
from typing import Any, Mapping, Sequence


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial-{os.getpid()}")
    temporary.write_bytes(_json_bytes(payload))
    os.replace(temporary, path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _load_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "proposal-a-runpod-launch-v1":
        raise ValueError("Unsupported Proposal A launch manifest")
    return payload


def _write_status(
    path: Path,
    *,
    state: str,
    stage: str,
    **extra: Any,
) -> None:
    _atomic_json(
        path,
        {
            "schema_version": "proposal-a-runpod-worker-status-v1",
            "created_utc": datetime.now(UTC).isoformat(),
            "pod_id": os.environ.get("RUNPOD_POD_ID"),
            "state": state,
            "stage": stage,
            **extra,
        },
    )


def _run_logged(
    command: Sequence[str],
    *,
    log_path: Path,
    cwd: Path,
    env: Mapping[str, str] | None = None,
) -> None:
    with log_path.open("a", encoding="utf-8") as log:
        process = subprocess.Popen(
            tuple(command),
            cwd=cwd,
            env=None if env is None else dict(env),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log.write(line)
            log.flush()
        returncode = process.wait()
    if returncode != 0:
        raise subprocess.CalledProcessError(returncode, tuple(command))


def _verify_file(path: Path, record: Mapping[str, Any], *, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")
    if path.stat().st_size != int(record["size_bytes"]):
        raise ValueError(f"{label} size drift")
    if _sha256_file(path) != record["sha256"]:
        raise ValueError(f"{label} checksum drift")


def _gpu_preflight(preregister: Mapping[str, Any]) -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Proposal A requires exactly one CUDA GPU")
    name = torch.cuda.get_device_name(0)
    if "L40S" not in name.upper():
        raise RuntimeError(f"Expected L40S, observed {name!r}")
    if not torch.__version__.startswith("2.5.1"):
        raise RuntimeError(f"Expected Torch 2.5.1, observed {torch.__version__}")
    total_memory = int(torch.cuda.get_device_properties(0).total_memory)
    if total_memory < 40 * 1024**3:
        raise RuntimeError(f"GPU memory is below the 40 GiB safety floor: {total_memory}")
    if preregister["execution_contract"]["expected_gpu"] != "NVIDIA L40S":
        raise RuntimeError("Preregistered GPU drift")
    return {
        "name": name,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "total_memory_bytes": total_memory,
    }


def _convert_or_resume_dataset(
    *,
    repository: Path,
    source_tar: Path,
    dataset_root: Path,
    log_path: Path,
) -> None:
    manifest_path = dataset_root / "dataset_manifest.json"
    if manifest_path.is_file():
        return
    command = [
        sys.executable,
        str(repository / "scripts" / "convert_lc0_tar_to_sequential.py"),
        "--input-tar",
        str(source_tar),
        "--output-dir",
        str(dataset_root),
        "--positions-per-chunk",
        "32768",
        "--validation-fraction",
        "0.01",
        "--test-fraction",
        "0.01",
        "--split-seed",
        "20260810",
    ]
    if dataset_root.exists():
        command.append("--resume")
    _run_logged(command, log_path=log_path, cwd=repository)


def _audit_dataset(
    *,
    repository: Path,
    dataset_root: Path,
    audit_path: Path,
    log_path: Path,
    preregister: Mapping[str, Any],
) -> dict[str, Any]:
    _run_logged(
        (
            sys.executable,
            str(repository / "scripts" / "audit_lc0_sequential.py"),
            "--dataset-root",
            str(dataset_root),
            "--output",
            str(audit_path),
        ),
        log_path=log_path,
        cwd=repository,
    )
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    expected = preregister["data_contract"]["audit"]
    if not audit.get("gate_pass"):
        raise RuntimeError("LC0 sequential audit gate failed")
    if audit.get("source_sha256") != preregister["data_contract"]["source_tar"]["sha256"]:
        raise RuntimeError("Converted dataset source checksum drift")
    if int(audit.get("storage_bytes", -1)) != int(expected["storage_bytes"]):
        raise RuntimeError("Converted dataset storage inventory drift")
    expected_splits = {
        "train": (73_908, 8_148_759, 254, 8_074_851),
        "validation": (745, 81_869, 243, 81_124),
        "test": (771, 84_626, 248, 83_855),
    }
    for split, expected_values in expected_splits.items():
        observed = audit["split_totals"][split]
        values = (
            int(observed["games"]),
            int(observed["positions"]),
            int(observed["shards"]),
            int(observed["trainable_starts"]),
        )
        if values != expected_values:
            raise RuntimeError(f"Converted {split} inventory drift: {values}")
    return audit


def _copy_dataset_to_container(
    *,
    source: Path,
    destination: Path,
    expected_bytes: int,
) -> None:
    free_bytes = shutil.disk_usage(destination.parent).free
    required_free = expected_bytes + 12 * 1024**3
    if free_bytes < required_free:
        raise RuntimeError(f"Container disk reserve too small: {free_bytes} < {required_free}")
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(source, destination)


def _verify_scheduler_contract(
    dataset_root: Path,
    preregister: Mapping[str, Any],
) -> dict[str, Any]:
    from chess_dfm_jax.data.lc0_sequential import SequentialBatches

    contracts = {
        "train": (
            preregister["data_contract"]["train_scheduler"],
            1024,
            0,
        ),
        "validation": (
            preregister["data_contract"]["validation_scheduler"],
            64,
            20_000,
        ),
    }
    result: dict[str, Any] = {}
    for split, (expected, batch_size, seed) in contracts.items():
        batches = SequentialBatches(
            dataset_root,
            split=split,
            batch_size=batch_size,
            horizon=8,
            seed=seed,
            shuffle_batches=True,
        )
        provenance = batches.provenance()
        for key in ("batch_size", "file_manifest_sha256", "seed", "shard_count"):
            if provenance[key] != expected[key]:
                raise RuntimeError(f"{split} scheduler {key} drift")
        expected_steps = (
            expected["steps_per_epoch"] if split == "train" else expected["available_full_batches"]
        )
        if provenance["steps_per_epoch"] != expected_steps:
            raise RuntimeError(f"{split} scheduler step-count drift")
        result[split] = provenance
    return result


def _checkpoint_candidate(path: Path) -> tuple[int, Path] | None:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        update = int(manifest["optimizer_update"])
        state = manifest["state"]
        state_path = path.parent / state["path"]
        if (
            manifest.get("format")
            not in {"chess-dfm-torch-training-v1", "chess-dfm-torch-training-v2"}
            or manifest.get("model_only") is not False
            or manifest.get("optimizer_resume_supported") is not True
            or state_path.stat().st_size != int(state["size_bytes"])
        ):
            return None
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return None
    return update, state_path


def find_latest_valid_recovery(run_root: Path) -> dict[str, Any] | None:
    candidates: list[tuple[int, Path, Path]] = []
    for manifest_path in run_root.glob("attempt-*/checkpoints/update*/manifest.json"):
        candidate = _checkpoint_candidate(manifest_path)
        if candidate is not None:
            update, state_path = candidate
            candidates.append((update, manifest_path, state_path))
    for update, manifest_path, state_path in sorted(candidates, reverse=True):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if _sha256_file(state_path) == manifest["state"]["sha256"]:
            return {
                "update": update,
                "checkpoint_dir": str(manifest_path.parent),
                "manifest_path": str(manifest_path),
                "state_sha256": manifest["state"]["sha256"],
                "state_size_bytes": int(manifest["state"]["size_bytes"]),
            }
    return None


def _next_attempt_path(run_root: Path) -> Path:
    indices = []
    for path in run_root.glob("attempt-*"):
        try:
            indices.append(int(path.name.removeprefix("attempt-")))
        except ValueError:
            continue
    return run_root / f"attempt-{(max(indices, default=-1) + 1):04d}"


def _remove_consumed_save_updates(
    command: Sequence[str],
    *,
    restored_update: int,
) -> list[str]:
    result = list(command)
    try:
        begin = result.index("--save-updates")
    except ValueError:
        return result
    end = begin + 1
    while end < len(result) and not result[end].startswith("--"):
        end += 1
    retained = [value for value in result[begin + 1 : end] if int(value) > restored_update]
    replacement = ["--save-updates", *retained] if retained else []
    return [*result[:begin], *replacement, *result[end:]]


def build_training_command(
    manifest: Mapping[str, Any],
    *,
    dataset_root: Path,
    raw_bt4_path: Path,
    attempt_output: Path,
    recovery: Mapping[str, Any] | None,
) -> list[str]:
    replacements = {
        "python": sys.executable,
        "{dataset_root}": str(dataset_root),
        "{raw_bt4_path}": str(raw_bt4_path),
        "{attempt_output_dir}": str(attempt_output),
    }
    command = [
        replacements.get(value, value) for value in manifest["preregister"]["training_command"]
    ]
    restored_update = 0 if recovery is None else int(recovery["update"])
    command = _remove_consumed_save_updates(
        command,
        restored_update=restored_update,
    )
    if recovery is not None:
        command.extend(("--resume-checkpoint", str(recovery["checkpoint_dir"])))
    return command


def _tail_jsonl(path: Path) -> dict[str, Any] | None:
    if not path.is_file() or path.stat().st_size == 0:
        return None
    with path.open("rb") as handle:
        handle.seek(max(path.stat().st_size - 256 * 1024, 0))
        lines = handle.read().splitlines()
    for line in reversed(lines):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def _training_status_loop(
    *,
    status_path: Path,
    attempt_output: Path,
    recovery: Mapping[str, Any] | None,
    stop: threading.Event,
) -> None:
    while not stop.is_set():
        last = _tail_jsonl(attempt_output / "metrics.jsonl")
        _write_status(
            status_path,
            state="running",
            stage="training",
            attempt=attempt_output.name,
            restored_update=0 if recovery is None else recovery["update"],
            latest_update=None if last is None else last.get("update"),
            latest_examples=None if last is None else last.get("examples"),
        )
        stop.wait(30)


def _verify_terminal_attempt(
    *,
    attempt_output: Path,
    run_root: Path,
) -> dict[str, Any]:
    report_path = attempt_output / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if int(report["last_metrics"]["update"]) != 7_774:
        raise RuntimeError("Terminal report did not reach update 7774")
    checkpoint_manifest_path = attempt_output / "checkpoint" / "manifest.json"
    checkpoint = json.loads(checkpoint_manifest_path.read_text(encoding="utf-8"))
    model_path = checkpoint_manifest_path.parent / checkpoint["state"]["path"]
    if (
        checkpoint.get("format") != "chess-dfm-torch-model-v1"
        or int(checkpoint["optimizer_update"]) != 7_774
        or model_path.stat().st_size != int(checkpoint["state"]["size_bytes"])
        or _sha256_file(model_path) != checkpoint["state"]["sha256"]
    ):
        raise RuntimeError("Terminal model checkpoint verification failed")
    pre_decay = None
    for manifest_path in run_root.glob("attempt-*/checkpoints/update00006219/manifest.json"):
        candidate = _checkpoint_candidate(manifest_path)
        if candidate is None or candidate[0] != 6_219:
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if _sha256_file(candidate[1]) == manifest["state"]["sha256"]:
            pre_decay = {
                "checkpoint_dir": str(manifest_path.parent),
                "state": manifest["state"],
            }
            break
    if pre_decay is None:
        raise RuntimeError("Checksum-valid update-6219 recovery checkpoint missing")
    compact_names = (
        "run_config.json",
        "optimizer_partition.json",
        "metrics.jsonl",
        "validation_metrics.jsonl",
        "loss_summary.json",
        "report.json",
    )
    compact = {}
    for name in compact_names:
        path = attempt_output / name
        if not path.is_file():
            raise RuntimeError(f"Terminal compact file missing: {name}")
        compact[name] = {
            "size_bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
    return {
        "attempt": attempt_output.name,
        "terminal_checkpoint": {
            "checkpoint_dir": str(checkpoint_manifest_path.parent),
            "state": checkpoint["state"],
        },
        "pre_decay_checkpoint": pre_decay,
        "compact": compact,
    }


def run(manifest_path: Path) -> int:
    manifest = _load_manifest(manifest_path)
    preregister = manifest["preregister"]
    repository = Path(__file__).resolve().parents[2]
    run_root = Path(manifest["paths"]["run_root"])
    run_root.mkdir(parents=True, exist_ok=True)
    status_path = run_root / "status.json"
    worker_log = run_root / "worker.log"
    _write_status(status_path, state="running", stage="preflight")
    try:
        expected_commit = manifest["git"]["commit"]
        observed_commit = subprocess.run(
            ("git", "rev-parse", "HEAD"),
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if observed_commit != expected_commit:
            raise RuntimeError("Worker source commit drift")
        source_tree_identity = hashlib.sha256(
            f"git-commit:{observed_commit}".encode("utf-8")
        ).hexdigest()
        os.environ["CHESS_DFM_SOURCE_TREE_SHA256"] = source_tree_identity
        gpu = _gpu_preflight(preregister)
        source_tar = Path(manifest["paths"]["source_tar"])
        raw_bt4 = Path(manifest["paths"]["raw_bt4"])
        _verify_file(
            source_tar,
            manifest["objects"]["source_tar"],
            label="LC0 source tar",
        )
        _verify_file(
            raw_bt4,
            manifest["objects"]["raw_bt4"],
            label="raw BT4",
        )
        _write_status(
            status_path,
            state="running",
            stage="convert_dataset",
            gpu=gpu,
        )
        network_dataset = Path(manifest["paths"]["dataset"])
        _convert_or_resume_dataset(
            repository=repository,
            source_tar=source_tar,
            dataset_root=network_dataset,
            log_path=worker_log,
        )
        _write_status(status_path, state="running", stage="audit_network_dataset")
        audit = _audit_dataset(
            repository=repository,
            dataset_root=network_dataset,
            audit_path=run_root / "dataset_audit.json",
            log_path=worker_log,
            preregister=preregister,
        )
        _write_status(status_path, state="running", stage="copy_dataset_to_container")
        local_dataset = Path("/opt/chess-dfm-data/lc0_sequential")
        local_dataset.parent.mkdir(parents=True, exist_ok=True)
        _copy_dataset_to_container(
            source=network_dataset,
            destination=local_dataset,
            expected_bytes=int(audit["storage_bytes"]),
        )
        scheduler = _verify_scheduler_contract(local_dataset, preregister)
        _atomic_json(run_root / "scheduler_provenance.json", scheduler)
        recovery = find_latest_valid_recovery(run_root)
        attempt_output = _next_attempt_path(run_root)
        command = build_training_command(
            manifest,
            dataset_root=local_dataset,
            raw_bt4_path=raw_bt4,
            attempt_output=attempt_output,
            recovery=recovery,
        )
        _atomic_json(
            run_root / f"{attempt_output.name}-launch.json",
            {
                "schema_version": "proposal-a-runpod-attempt-launch-v1",
                "created_utc": datetime.now(UTC).isoformat(),
                "attempt": attempt_output.name,
                "command": command,
                "recovery": recovery,
                "source_tree_identity": source_tree_identity,
                "gpu": gpu,
            },
        )
        status_stop = threading.Event()
        status_thread = threading.Thread(
            target=_training_status_loop,
            kwargs={
                "status_path": status_path,
                "attempt_output": attempt_output,
                "recovery": recovery,
                "stop": status_stop,
            },
            name="proposal-a-training-status",
            daemon=True,
        )
        status_thread.start()
        try:
            _run_logged(
                command,
                log_path=run_root / f"{attempt_output.name}.stdout.log",
                cwd=repository,
                env=os.environ,
            )
        finally:
            status_stop.set()
            status_thread.join(timeout=5)
        terminal = _verify_terminal_attempt(
            attempt_output=attempt_output,
            run_root=run_root,
        )
        _write_status(
            status_path,
            state="succeeded",
            stage="terminal_verified",
            gpu=gpu,
            terminal=terminal,
        )
        os.sync()
        return 0
    except BaseException as exc:
        recovery = find_latest_valid_recovery(run_root)
        _write_status(
            status_path,
            state="failed",
            stage="worker_exception",
            error_type=type(exc).__name__,
            latest_valid_recovery=recovery,
        )
        print(
            json.dumps(
                {
                    "proposal_a_epoch_failed": True,
                    "error_type": type(exc).__name__,
                    "latest_valid_recovery": recovery,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        os.sync()
        return 1


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    return run(args.manifest.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
