#!/usr/bin/env python3
"""Run a sequence of training experiments on one already-provisioned worker."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chess_dfm_jax.tracking import load_env_file  # noqa: E402


PHASE_ENTRY_MARKERS = {
    "dfm": "scripts/train_dfm.py",
    "jepa": "scripts/train_jepa.py",
    "joint": "scripts/train_joint_latent_sasa.py",
    "profile": "scripts/profile_jepa_tpu.py",
}

BLOCKED_COMMAND_MARKERS = (
    "/tmp/lc0jaxhuman",
    "verified_source_v27_dfm",
)


@dataclass
class QueueExperiment:
    experiment_id: str
    entry_command: str
    phase: str = "unclassified"
    env: dict[str, str] = field(default_factory=dict)
    cwd: str | None = None
    timeout_s: int = 0
    allow_failure: bool = False


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON on {path}:{line_number}: {exc}") from exc
    return rows


def load_queue(path: Path) -> list[QueueExperiment]:
    experiments: list[QueueExperiment] = []
    for row in read_jsonl(path):
        experiments.append(
            QueueExperiment(
                experiment_id=str(row["experiment_id"]),
                entry_command=str(row["entry_command"]),
                phase=str(row.get("phase", "unclassified")),
                env={str(k): str(v) for k, v in row.get("env", {}).items()},
                cwd=str(row["cwd"]) if row.get("cwd") else None,
                timeout_s=int(row.get("timeout_s", 0)),
                allow_failure=bool(row.get("allow_failure", False)),
            )
        )
    return experiments


def _extract_flag(tokens: list[str], name: str) -> str | None:
    flag = f"--{name}"
    prefix = f"{flag}="
    for idx, token in enumerate(tokens):
        if token == flag and idx + 1 < len(tokens):
            return tokens[idx + 1]
        if token.startswith(prefix):
            return token[len(prefix) :]
    return None


def command_checkpoint_uri(command: str) -> str | None:
    try:
        tokens = shlex.split(command)
    except ValueError:
        return None
    return _extract_flag(tokens, "checkpoint-uri")


def validate_queue(experiments: list[QueueExperiment]) -> None:
    if not experiments:
        raise ValueError("Experiment queue is empty.")
    seen_ids: set[str] = set()
    checkpoint_owners: dict[str, str] = {}
    for exp in experiments:
        if not exp.experiment_id:
            raise ValueError("Every queue entry must set experiment_id.")
        if exp.experiment_id in seen_ids:
            raise ValueError(f"Duplicate experiment_id in queue: {exp.experiment_id}")
        seen_ids.add(exp.experiment_id)
        if not exp.entry_command:
            raise ValueError(f"{exp.experiment_id} has an empty entry_command.")
        for marker in BLOCKED_COMMAND_MARKERS:
            if marker in exp.entry_command:
                raise ValueError(f"{exp.experiment_id} uses blocked command marker {marker!r}.")
        expected = PHASE_ENTRY_MARKERS.get(exp.phase)
        if expected and expected not in exp.entry_command:
            raise ValueError(
                f"{exp.experiment_id} phase={exp.phase!r} must run {expected!r}."
            )
        checkpoint_uri = command_checkpoint_uri(exp.entry_command)
        if checkpoint_uri:
            owner = checkpoint_owners.get(checkpoint_uri)
            if owner is not None:
                raise ValueError(
                    f"{exp.experiment_id} and {owner} both write checkpoint {checkpoint_uri!r}."
                )
            checkpoint_owners[checkpoint_uri] = exp.experiment_id


def download_queue(queue_uri: str, destination: Path) -> Path:
    if not queue_uri.startswith("gs://"):
        return Path(queue_uri)
    destination.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["gcloud", "storage", "cp", queue_uri, str(destination)],
        check=True,
    )
    return destination


def upload_status(path: Path, status_uri: str) -> None:
    if not status_uri:
        return
    destination = status_uri.rstrip("/") + "/" + path.name
    subprocess.run(
        ["gcloud", "storage", "cp", str(path), destination],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def sync_existing_status(status_uri: str, status_dir: Path) -> None:
    if not status_uri:
        return
    status_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["gcloud", "storage", "cp", "--recursive", status_uri.rstrip("/") + "/*.json", str(status_dir)],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def write_json(path: Path, payload: dict[str, Any], status_uri: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)
    upload_status(path, status_uri)


class QueueRunner:
    def __init__(
        self,
        *,
        experiments: list[QueueExperiment],
        workdir: Path,
        status_dir: Path,
        status_uri: str = "",
        stop_on_failure: bool = True,
        dry_run: bool = False,
        skip_completed: bool = True,
    ):
        self.experiments = experiments
        self.workdir = workdir
        self.status_dir = status_dir
        self.status_uri = status_uri
        self.stop_on_failure = stop_on_failure
        self.dry_run = dry_run
        self.skip_completed = skip_completed
        self.stop_requested = False
        self.current_process: subprocess.Popen[str] | None = None

    def request_stop(self, signum: int) -> None:
        self.stop_requested = True
        if self.current_process is not None and self.current_process.poll() is None:
            self.current_process.terminate()
        print(f"queue_signal_received={signal.Signals(signum).name}", flush=True)

    def _queue_status(self, state: str, **extra: Any) -> None:
        payload = {
            "state": state,
            "timestamp": utc_now(),
            "total": len(self.experiments),
            **extra,
        }
        write_json(self.status_dir / "queue_status.json", payload, self.status_uri)

    def _experiment_status(self, exp: QueueExperiment, state: str, **extra: Any) -> None:
        payload = {
            "experiment_id": exp.experiment_id,
            "phase": exp.phase,
            "state": state,
            "timestamp": utc_now(),
            "entry_command": exp.entry_command,
            "checkpoint_uri": command_checkpoint_uri(exp.entry_command),
            **extra,
        }
        write_json(self.status_dir / f"{exp.experiment_id}.json", payload, self.status_uri)

    def run(self) -> int:
        self.workdir.mkdir(parents=True, exist_ok=True)
        completed = 0
        failed = 0
        skipped = 0
        self._queue_status("running", completed=0, failed=0, skipped=0)
        for index, exp in enumerate(self.experiments):
            if self.skip_completed and self._is_completed(exp):
                completed += 1
                print(f"[queue] skipping completed {exp.experiment_id}", flush=True)
                continue
            if self.stop_requested:
                skipped += 1
                self._experiment_status(exp, "skipped", reason="stop_requested")
                continue
            code = self._run_one(index, exp)
            if code == 0 or exp.allow_failure:
                completed += 1
                continue
            failed += 1
            if self.stop_on_failure:
                skipped += len(self.experiments) - index - 1
                break
        state = "completed" if failed == 0 and not self.stop_requested else "failed"
        if self.stop_requested:
            state = "stopped"
        self._queue_status(state, completed=completed, failed=failed, skipped=skipped)
        return 0 if failed == 0 and not self.stop_requested else 1

    def _is_completed(self, exp: QueueExperiment) -> bool:
        path = self.status_dir / f"{exp.experiment_id}.json"
        if not path.exists():
            return False
        try:
            state = json.loads(path.read_text(encoding="utf-8")).get("state")
        except Exception:
            return False
        return state in {"completed", "allowed_failure"}

    def _run_one(self, index: int, exp: QueueExperiment) -> int:
        cwd = Path(exp.cwd) if exp.cwd else self.workdir
        env = os.environ.copy()
        env.update(exp.env)
        env["CHESS_DFM_QUEUE_INDEX"] = str(index)
        env["CHESS_DFM_QUEUE_EXPERIMENT_ID"] = exp.experiment_id
        env["CHESS_DFM_QUEUE_PHASE"] = exp.phase
        self._experiment_status(exp, "dry_run" if self.dry_run else "running", index=index)
        if self.dry_run:
            print(f"[dry-run] {exp.experiment_id}: {exp.entry_command}", flush=True)
            self._experiment_status(exp, "dry_run", index=index, exit_code=0, dry_run=True)
            return 0

        started_at = utc_now()
        print(f"[queue] starting {exp.experiment_id}: {exp.entry_command}", flush=True)
        self.current_process = subprocess.Popen(
            exp.entry_command,
            cwd=str(cwd),
            env=env,
            shell=True,
            text=True,
        )
        try:
            exit_code = self.current_process.wait(timeout=exp.timeout_s or None)
        except subprocess.TimeoutExpired:
            self.current_process.terminate()
            try:
                exit_code = self.current_process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                self.current_process.kill()
                exit_code = self.current_process.wait()
            self._experiment_status(
                exp,
                "failed",
                index=index,
                started_at=started_at,
                finished_at=utc_now(),
                exit_code=exit_code,
                reason="timeout",
            )
            return exit_code or 124
        finally:
            self.current_process = None

        state = "completed" if exit_code == 0 else "failed"
        if exp.allow_failure and exit_code != 0:
            state = "allowed_failure"
        self._experiment_status(
            exp,
            state,
            index=index,
            started_at=started_at,
            finished_at=utc_now(),
            exit_code=exit_code,
        )
        return int(exit_code)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--queue", type=str, help="Local experiment queue JSONL.")
    source.add_argument("--queue-uri", type=str, help="Local path or gs:// experiment queue JSONL.")
    parser.add_argument("--workdir", type=str, default=".", help="Working directory for experiment commands.")
    parser.add_argument("--status-dir", type=str, default="artifacts/experiment_queue")
    parser.add_argument("--status-uri", type=str, default="", help="Optional gs:// prefix for status JSON uploads.")
    parser.add_argument("--env-file", type=str, default=None, help="Optional .env file to load before running.")
    parser.add_argument("--keep-going", action="store_true", help="Continue after a failed experiment.")
    parser.add_argument("--rerun-completed", action="store_true", help="Do not skip completed status files.")
    parser.add_argument("--dry-run", action="store_true", help="Validate and print commands without running them.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    load_env_file(args.env_file)
    queue_path = (
        Path(args.queue)
        if args.queue
        else download_queue(args.queue_uri, Path(args.status_dir) / "queue.jsonl")
    )
    experiments = load_queue(queue_path)
    validate_queue(experiments)
    if args.status_uri and not args.rerun_completed:
        sync_existing_status(args.status_uri, Path(args.status_dir))
    runner = QueueRunner(
        experiments=experiments,
        workdir=Path(args.workdir),
        status_dir=Path(args.status_dir),
        status_uri=args.status_uri,
        stop_on_failure=not args.keep_going,
        dry_run=args.dry_run,
        skip_completed=not args.rerun_completed,
    )
    signal.signal(signal.SIGTERM, lambda signum, _frame: runner.request_stop(signum))
    signal.signal(signal.SIGINT, lambda signum, _frame: runner.request_stop(signum))
    return runner.run()


if __name__ == "__main__":
    raise SystemExit(main())
