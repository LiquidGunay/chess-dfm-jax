from __future__ import annotations

import dataclasses
import fcntl
import subprocess
import sys
from pathlib import Path

import pytest

from research.resource_guard import (
    RESOURCE_GUARD_EXIT,
    GIB,
    CheckpointPlan,
    GuardConfig,
    GuardViolation,
    checkpoint_plan,
    config_from_environ,
    parse_mem_available_bytes,
    selected_cpu_affinity,
    validate_preflight,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_mem_available_parser_requires_linux_kib_field() -> None:
    assert parse_mem_available_bytes("MemTotal: 100 kB\nMemAvailable: 42 kB\n") == 42 * 1024
    with pytest.raises(GuardViolation, match="no valid MemAvailable"):
        parse_mem_available_bytes("MemFree: 42 kB\n")
    with pytest.raises(GuardViolation, match="no valid MemAvailable"):
        parse_mem_available_bytes("MemAvailable: 42 MB\n")


def test_checkpoint_plan_counts_sparse_and_final_writes() -> None:
    command = [
        ".venv/bin/python",
        "research/train.py",
        "--save-every",
        "0",
        "--save-updates",
        "800",
        "--save-final",
        "--max-checkpoints=2",
    ]
    assert checkpoint_plan(command) == CheckpointPlan(
        planned_writes=2,
        save_every=0,
        max_retained=2,
    )


def test_checkpoint_plan_ignores_evaluators_and_respects_no_save_final() -> None:
    assert checkpoint_plan(["python", "research/arena.py"]) == CheckpointPlan(
        planned_writes=0,
        save_every=0,
        max_retained=0,
    )
    assert (
        checkpoint_plan(
            [
                "python",
                "research/train.py",
                "--save-final",
                "--no-save-final",
            ]
        ).planned_writes
        == 0
    )


def test_preflight_rejects_periodic_or_three_checkpoint_schedules() -> None:
    config = GuardConfig()
    with pytest.raises(GuardViolation, match="Periodic --save-every"):
        validate_preflight(
            config=config,
            plan=CheckpointPlan(planned_writes=0, save_every=100, max_retained=2),
            available_bytes=10 * GIB,
            disk_free_bytes=50 * GIB,
        )
    with pytest.raises(GuardViolation, match="requests 3 writes"):
        validate_preflight(
            config=config,
            plan=CheckpointPlan(planned_writes=3, save_every=0, max_retained=2),
            available_bytes=10 * GIB,
            disk_free_bytes=50 * GIB,
        )


def test_preflight_requires_bounded_retention_and_disk_reserve() -> None:
    config = GuardConfig()
    with pytest.raises(GuardViolation, match="--max-checkpoints"):
        validate_preflight(
            config=config,
            plan=CheckpointPlan(planned_writes=1, save_every=0, max_retained=0),
            available_bytes=10 * GIB,
            disk_free_bytes=50 * GIB,
        )
    with pytest.raises(GuardViolation, match="projected disk reserve"):
        validate_preflight(
            config=config,
            plan=CheckpointPlan(planned_writes=2, save_every=0, max_retained=2),
            available_bytes=10 * GIB,
            disk_free_bytes=33 * GIB,
        )


def test_preflight_requires_launch_memory_headroom() -> None:
    with pytest.raises(GuardViolation, match="Insufficient MemAvailable"):
        validate_preflight(
            config=GuardConfig(),
            plan=CheckpointPlan(planned_writes=0, save_every=0, max_retained=2),
            available_bytes=7 * GIB,
            disk_free_bytes=50 * GIB,
        )


def test_two_checkpoint_preflight_passes_with_reserve() -> None:
    validate_preflight(
        config=GuardConfig(),
        plan=CheckpointPlan(planned_writes=2, save_every=0, max_retained=2),
        available_bytes=10 * GIB,
        disk_free_bytes=50 * GIB,
    )


def test_cpu_affinity_is_sorted_deduplicated_and_capped() -> None:
    assert selected_cpu_affinity([7, 3, 7, 5], 2) == (3, 5)
    assert selected_cpu_affinity([4], 2) == (4,)
    with pytest.raises(GuardViolation, match="no available CPU"):
        selected_cpu_affinity([], 2)


def test_environment_contract_fails_closed() -> None:
    assert config_from_environ({}).poll_seconds == 0.05
    with pytest.raises(GuardViolation, match="must be an integer"):
        config_from_environ({"CHESS_DFM_GUARD_CPU_COUNT": "many"})
    with pytest.raises(GuardViolation, match="Runtime MemAvailable floor"):
        config_from_environ(
            {
                "CHESS_DFM_GUARD_MIN_START_AVAILABLE_BYTES": str(8 * GIB),
                "CHESS_DFM_GUARD_MIN_RUNTIME_AVAILABLE_BYTES": str(8 * GIB),
            }
        )
    with pytest.raises(GuardViolation, match=r"must be in \[1, 2\]"):
        config_from_environ({"CHESS_DFM_GUARD_CPU_COUNT": "3"})


def test_custom_checkpoint_limit_is_enforced() -> None:
    config = dataclasses.replace(GuardConfig(), max_checkpoint_writes=1)
    with pytest.raises(GuardViolation, match="requests 2 writes"):
        validate_preflight(
            config=config,
            plan=CheckpointPlan(planned_writes=2, save_every=0, max_retained=1),
            available_bytes=10 * GIB,
            disk_free_bytes=50 * GIB,
        )


def test_runtime_guard_terminates_child_over_rss_limit() -> None:
    script = """
import sys
from pathlib import Path
from research.resource_guard import GuardConfig, guarded_run

config = GuardConfig(
    cpu_count=1,
    min_start_available_bytes=1,
    min_runtime_available_bytes=1,
    max_process_group_rss_bytes=32 * 1024**2,
    min_disk_reserve_bytes=1,
    checkpoint_bytes=1,
    max_checkpoint_writes=0,
    poll_seconds=0.05,
    termination_grace_seconds=0.1,
)
command = [
    sys.executable,
    "-c",
    "import time; payload = bytearray(64 * 1024**2); time.sleep(5)",
]
raise SystemExit(guarded_run(command, config, Path(sys.argv[1])))
"""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            str(REPO_ROOT),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )
    assert result.returncode == RESOURCE_GUARD_EXIT
    assert "RESOURCE_GUARD_ABORT process-group RSS exceeded guard" in result.stderr


def test_gpu_launcher_refuses_a_second_host_visible_lock() -> None:
    lock_dir = REPO_ROOT / ".local" / "locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    with (lock_dir / "gpu-workload.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = subprocess.run(
            ["bash", "research/run_gpu.sh", "/bin/true"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    assert result.returncode == 73
    assert "RESOURCE_GUARD_BUSY" in result.stderr
