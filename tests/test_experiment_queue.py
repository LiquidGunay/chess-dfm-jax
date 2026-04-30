import json
from pathlib import Path
import shlex
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_experiment_queue import (  # noqa: E402
    QueueExperiment,
    QueueRunner,
    command_checkpoint_uri,
    load_queue,
    validate_queue,
)


def _python_command(source: str) -> str:
    return f"{shlex.quote(sys.executable)} -c {shlex.quote(source)}"


def test_command_checkpoint_uri_supports_split_and_equals_forms():
    assert (
        command_checkpoint_uri("python scripts/train_dfm.py --checkpoint-uri gs://bucket/a")
        == "gs://bucket/a"
    )
    assert (
        command_checkpoint_uri("python scripts/train_dfm.py --checkpoint-uri=gs://bucket/b")
        == "gs://bucket/b"
    )


def test_validate_queue_rejects_duplicate_checkpoint_uri():
    experiments = [
        QueueExperiment(
            experiment_id="a",
            phase="dfm",
            entry_command="python scripts/train_dfm.py --checkpoint-uri gs://bucket/shared",
        ),
        QueueExperiment(
            experiment_id="b",
            phase="dfm",
            entry_command="python scripts/train_dfm.py --checkpoint-uri gs://bucket/shared",
        ),
    ]
    with pytest.raises(ValueError, match="both write checkpoint"):
        validate_queue(experiments)


def test_load_queue_and_dry_run(tmp_path):
    queue_path = tmp_path / "queue.jsonl"
    queue_path.write_text(
        json.dumps(
            {
                "experiment_id": "smoke",
                "phase": "smoke",
                "entry_command": "echo smoke",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    experiments = load_queue(queue_path)
    validate_queue(experiments)
    status_dir = tmp_path / "status"
    runner = QueueRunner(
        experiments=experiments,
        workdir=tmp_path,
        status_dir=status_dir,
        dry_run=True,
    )
    assert runner.run() == 0
    status = json.loads((status_dir / "smoke.json").read_text(encoding="utf-8"))
    assert status["state"] == "completed"
    assert status["dry_run"] is True


def test_queue_runner_runs_multiple_experiments_on_same_worker(tmp_path):
    experiments = [
        QueueExperiment(
            experiment_id="first",
            phase="smoke",
            entry_command=_python_command("from pathlib import Path; Path('first.txt').write_text('1')"),
        ),
        QueueExperiment(
            experiment_id="second",
            phase="smoke",
            entry_command=_python_command("from pathlib import Path; Path('second.txt').write_text('2')"),
        ),
    ]
    validate_queue(experiments)
    status_dir = tmp_path / "status"
    runner = QueueRunner(
        experiments=experiments,
        workdir=tmp_path,
        status_dir=status_dir,
    )
    assert runner.run() == 0
    assert (tmp_path / "first.txt").read_text(encoding="utf-8") == "1"
    assert (tmp_path / "second.txt").read_text(encoding="utf-8") == "2"
    queue_status = json.loads((status_dir / "queue_status.json").read_text(encoding="utf-8"))
    assert queue_status["state"] == "completed"
    assert queue_status["completed"] == 2
