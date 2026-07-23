#!/usr/bin/env python3
"""Fail-closed host resource guard for local GPU research commands.

The A10G host has much less RAM than GPU memory and no swap.  JAX compilation
can therefore destabilize the whole machine before accelerator OOM handling is
relevant.  This wrapper constrains CPU affinity, gives the child a high OOM
victim preference, monitors process-group RSS and system MemAvailable, and
rejects checkpoint schedules that would consume too much local disk.
"""

from __future__ import annotations

import argparse
import ctypes
import dataclasses
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path

GIB = 1024**3
RESOURCE_GUARD_EXIT = 75
CONFIG_EXIT = 64
PR_SET_PDEATHSIG = 1


class GuardViolation(RuntimeError):
    """A fail-closed resource or checkpoint-policy violation."""


@dataclasses.dataclass(frozen=True)
class GuardConfig:
    cpu_count: int = 2
    min_start_available_bytes: int = 8 * GIB
    min_runtime_available_bytes: int = 3 * GIB
    max_process_group_rss_bytes: int = 7 * GIB
    min_disk_reserve_bytes: int = 30 * GIB
    checkpoint_bytes: int = 2 * GIB
    max_checkpoint_writes: int = 2
    poll_seconds: float = 0.05
    termination_grace_seconds: float = 5.0
    child_oom_score_adj: int = 500
    child_nice: int = 5


@dataclasses.dataclass(frozen=True)
class CheckpointPlan:
    planned_writes: int
    save_every: int
    max_retained: int


def _env_int(
    environ: Mapping[str, str],
    name: str,
    default: int,
    *,
    minimum: int = 0,
    maximum: int | None = None,
) -> int:
    raw = environ.get(name)
    try:
        value = default if raw is None else int(raw)
    except ValueError as exc:
        raise GuardViolation(f"{name} must be an integer, got {raw!r}") from exc
    if value < minimum or (maximum is not None and value > maximum):
        bounds = f">={minimum}" if maximum is None else f"in [{minimum}, {maximum}]"
        raise GuardViolation(f"{name} must be {bounds}, got {value}")
    return value


def _env_float(
    environ: Mapping[str, str],
    name: str,
    default: float,
    *,
    minimum: float,
    maximum: float | None = None,
) -> float:
    raw = environ.get(name)
    try:
        value = default if raw is None else float(raw)
    except ValueError as exc:
        raise GuardViolation(f"{name} must be numeric, got {raw!r}") from exc
    if not value >= minimum or (maximum is not None and value > maximum):
        bounds = f">={minimum}" if maximum is None else f"in [{minimum}, {maximum}]"
        raise GuardViolation(f"{name} must be {bounds}, got {value}")
    return value


def config_from_environ(environ: Mapping[str, str] | None = None) -> GuardConfig:
    """Build and validate the guard contract from explicit environment knobs."""

    env = os.environ if environ is None else environ
    config = GuardConfig(
        cpu_count=_env_int(
            env,
            "CHESS_DFM_GUARD_CPU_COUNT",
            2,
            minimum=1,
            maximum=2,
        ),
        min_start_available_bytes=_env_int(
            env,
            "CHESS_DFM_GUARD_MIN_START_AVAILABLE_BYTES",
            8 * GIB,
            minimum=8 * GIB,
        ),
        min_runtime_available_bytes=_env_int(
            env,
            "CHESS_DFM_GUARD_MIN_RUNTIME_AVAILABLE_BYTES",
            3 * GIB,
            minimum=3 * GIB,
        ),
        max_process_group_rss_bytes=_env_int(
            env,
            "CHESS_DFM_GUARD_MAX_PROCESS_RSS_BYTES",
            7 * GIB,
            minimum=1,
            maximum=7 * GIB,
        ),
        min_disk_reserve_bytes=_env_int(
            env,
            "CHESS_DFM_GUARD_MIN_DISK_RESERVE_BYTES",
            30 * GIB,
            minimum=30 * GIB,
        ),
        checkpoint_bytes=_env_int(
            env,
            "CHESS_DFM_GUARD_CHECKPOINT_BYTES",
            2 * GIB,
            minimum=2 * GIB,
        ),
        max_checkpoint_writes=_env_int(
            env,
            "CHESS_DFM_GUARD_MAX_CHECKPOINT_WRITES",
            2,
            minimum=0,
            maximum=2,
        ),
        poll_seconds=_env_float(
            env,
            "CHESS_DFM_GUARD_POLL_SECONDS",
            0.05,
            minimum=0.05,
            maximum=0.5,
        ),
        termination_grace_seconds=_env_float(
            env,
            "CHESS_DFM_GUARD_TERMINATION_GRACE_SECONDS",
            5.0,
            minimum=0.0,
            maximum=5.0,
        ),
        child_oom_score_adj=_env_int(
            env,
            "CHESS_DFM_GUARD_CHILD_OOM_SCORE_ADJ",
            500,
            minimum=500,
            maximum=1000,
        ),
        child_nice=_env_int(
            env,
            "CHESS_DFM_GUARD_CHILD_NICE",
            5,
            minimum=5,
            maximum=19,
        ),
    )
    if config.min_runtime_available_bytes >= config.min_start_available_bytes:
        raise GuardViolation(
            "Runtime MemAvailable floor must be below the launch floor: "
            f"{config.min_runtime_available_bytes} >= "
            f"{config.min_start_available_bytes}"
        )
    return config


def parse_mem_available_bytes(text: str) -> int:
    """Read Linux MemAvailable from procfs text."""

    for line in text.splitlines():
        name, separator, raw = line.partition(":")
        if name == "MemAvailable" and separator:
            fields = raw.split()
            if len(fields) != 2 or fields[1] != "kB":
                break
            try:
                return int(fields[0]) * 1024
            except ValueError:
                break
    raise GuardViolation("/proc/meminfo has no valid MemAvailable entry")


def mem_available_bytes(path: Path = Path("/proc/meminfo")) -> int:
    return parse_mem_available_bytes(path.read_text(encoding="utf-8"))


def _option_value(command: Sequence[str], name: str, default: int) -> int:
    value = default
    for index, argument in enumerate(command):
        if argument == name:
            if index + 1 >= len(command):
                raise GuardViolation(f"{name} requires a value")
            raw = command[index + 1]
        elif argument.startswith(f"{name}="):
            raw = argument.split("=", 1)[1]
        else:
            continue
        try:
            value = int(raw)
        except ValueError as exc:
            raise GuardViolation(f"{name} must be an integer, got {raw!r}") from exc
    return value


def _boolean_option(
    command: Sequence[str],
    positive: str,
    negative: str,
    *,
    default: bool,
) -> bool:
    value = default
    for argument in command:
        if argument == positive:
            value = True
        elif argument == negative:
            value = False
    return value


def _save_updates(command: Sequence[str]) -> tuple[int, ...]:
    values: list[int] = []
    index = 0
    while index < len(command):
        argument = command[index]
        if argument.startswith("--save-updates="):
            raw_values = [argument.split("=", 1)[1]]
            index += 1
        elif argument == "--save-updates":
            raw_values = []
            index += 1
            while index < len(command) and not command[index].startswith("-"):
                raw_values.append(command[index])
                index += 1
        else:
            index += 1
            continue
        for raw in raw_values:
            try:
                values.append(int(raw))
            except ValueError as exc:
                raise GuardViolation(
                    f"--save-updates values must be integers, got {raw!r}"
                ) from exc
    return tuple(values)


def _is_research_train(command: Sequence[str]) -> bool:
    trainer_names = {"train.py", "train_torch.py"}
    return any(Path(argument).name in trainer_names for argument in command)


def checkpoint_plan(command: Sequence[str]) -> CheckpointPlan:
    """Conservatively count state writes requested from research/train.py."""

    if not _is_research_train(command):
        return CheckpointPlan(planned_writes=0, save_every=0, max_retained=0)
    save_every = _option_value(command, "--save-every", 0)
    max_retained = _option_value(command, "--max-checkpoints", 2)
    save_final = _boolean_option(
        command,
        "--save-final",
        "--no-save-final",
        default=False,
    )
    planned_writes = len(_save_updates(command)) + int(save_final)
    return CheckpointPlan(
        planned_writes=planned_writes,
        save_every=save_every,
        max_retained=max_retained,
    )


def validate_preflight(
    *,
    config: GuardConfig,
    plan: CheckpointPlan,
    available_bytes: int,
    disk_free_bytes: int,
) -> None:
    """Validate launch memory, disk reserve, and bounded checkpoint cadence."""

    if available_bytes < config.min_start_available_bytes:
        raise GuardViolation(
            "Insufficient MemAvailable to launch safely: "
            f"{available_bytes} < {config.min_start_available_bytes} bytes"
        )
    if plan.save_every != 0:
        raise GuardViolation(
            "Periodic --save-every is disabled by the disk guard; use at most "
            "one sparse --save-updates checkpoint plus --save-final"
        )
    if plan.planned_writes > config.max_checkpoint_writes:
        raise GuardViolation(
            "Checkpoint schedule requests "
            f"{plan.planned_writes} writes; guarded maximum is "
            f"{config.max_checkpoint_writes}"
        )
    if plan.planned_writes and (
        plan.max_retained <= 0 or plan.max_retained > config.max_checkpoint_writes
    ):
        raise GuardViolation(
            "--max-checkpoints must be between 1 and the guarded maximum "
            f"{config.max_checkpoint_writes}, got {plan.max_retained}"
        )
    projected_free = disk_free_bytes - plan.planned_writes * config.checkpoint_bytes
    if projected_free < config.min_disk_reserve_bytes:
        raise GuardViolation(
            "Insufficient projected disk reserve: "
            f"{projected_free} < {config.min_disk_reserve_bytes} bytes after "
            f"{plan.planned_writes} checkpoint write(s)"
        )


def selected_cpu_affinity(available: Sequence[int], cpu_count: int) -> tuple[int, ...]:
    cpus = tuple(sorted(set(available)))
    if not cpus:
        raise GuardViolation("The launcher has no available CPU in its affinity mask")
    return cpus[: min(cpu_count, len(cpus))]


def _status_rss_bytes(pid: int) -> int:
    try:
        text = Path(f"/proc/{pid}/status").read_text(encoding="utf-8")
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return 0
    for line in text.splitlines():
        if line.startswith("VmRSS:"):
            fields = line.split()
            if len(fields) == 3 and fields[2] == "kB":
                return int(fields[1]) * 1024
    return 0


def process_group_rss_bytes(process_group: int) -> int:
    """Sum resident bytes for every process in the guarded process group."""

    total = 0
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        try:
            if os.getpgid(pid) != process_group:
                continue
        except (PermissionError, ProcessLookupError):
            continue
        total += _status_rss_bytes(pid)
    return total


def _child_setup(config: GuardConfig, expected_parent_pid: int) -> None:
    """Apply last-resort safety settings immediately before child exec."""

    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(PR_SET_PDEATHSIG, signal.SIGTERM) != 0:
        errno = ctypes.get_errno()
        raise OSError(errno, "prctl(PR_SET_PDEATHSIG) failed")
    if os.getppid() != expected_parent_pid:
        os.kill(os.getpid(), signal.SIGTERM)
    os.nice(config.child_nice)
    Path("/proc/self/oom_score_adj").write_text(
        str(config.child_oom_score_adj),
        encoding="ascii",
    )


def _terminate_process_group(
    child: subprocess.Popen[bytes],
    *,
    grace_seconds: float,
) -> None:
    if child.poll() is not None:
        return
    try:
        os.killpg(child.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        child.wait(timeout=grace_seconds)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(child.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    child.wait()


def guarded_run(command: Sequence[str], config: GuardConfig, workspace: Path) -> int:
    """Launch one command and stop it before it can exhaust host resources."""

    if not command:
        raise GuardViolation("No guarded command was provided")
    workspace = workspace.resolve()
    try:
        workspace.relative_to(Path("/mountpoint/.exp"))
    except ValueError as exc:
        raise GuardViolation(
            f"Guard workspace must be below /mountpoint/.exp: {workspace}"
        ) from exc

    available = mem_available_bytes()
    disk_free = shutil.disk_usage(workspace).free
    plan = checkpoint_plan(command)
    validate_preflight(
        config=config,
        plan=plan,
        available_bytes=available,
        disk_free_bytes=disk_free,
    )

    affinity = selected_cpu_affinity(tuple(os.sched_getaffinity(0)), config.cpu_count)
    os.sched_setaffinity(0, affinity)
    start_record = {
        "checkpoint_writes": plan.planned_writes,
        "cpu_affinity": affinity,
        "disk_free_bytes": disk_free,
        "max_process_group_rss_bytes": config.max_process_group_rss_bytes,
        "mem_available_bytes": available,
        "min_disk_reserve_bytes": config.min_disk_reserve_bytes,
        "min_runtime_available_bytes": config.min_runtime_available_bytes,
    }
    print(
        "RESOURCE_GUARD_START " + json.dumps(start_record, sort_keys=True),
        file=sys.stderr,
        flush=True,
    )

    parent_pid = os.getpid()
    child = subprocess.Popen(
        list(command),
        start_new_session=True,
        preexec_fn=lambda: _child_setup(config, parent_pid),
    )
    forwarded_signal: int | None = None

    def forward_signal(signum: int, _frame: object) -> None:
        nonlocal forwarded_signal
        forwarded_signal = signum
        try:
            os.killpg(child.pid, signum)
        except ProcessLookupError:
            pass

    previous_handlers = {
        signum: signal.signal(signum, forward_signal)
        for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
    }
    started = time.monotonic()
    minimum_available = available
    peak_rss = 0
    violation: str | None = None
    try:
        while child.poll() is None:
            available = mem_available_bytes()
            rss = process_group_rss_bytes(child.pid)
            minimum_available = min(minimum_available, available)
            peak_rss = max(peak_rss, rss)
            if rss > config.max_process_group_rss_bytes:
                violation = (
                    "process-group RSS exceeded guard: "
                    f"{rss} > {config.max_process_group_rss_bytes} bytes"
                )
                break
            if available < config.min_runtime_available_bytes:
                violation = (
                    "system MemAvailable crossed guard: "
                    f"{available} < {config.min_runtime_available_bytes} bytes"
                )
                break
            time.sleep(config.poll_seconds)
        if violation is not None:
            print(f"RESOURCE_GUARD_ABORT {violation}", file=sys.stderr, flush=True)
            _terminate_process_group(
                child,
                grace_seconds=config.termination_grace_seconds,
            )
            return RESOURCE_GUARD_EXIT
        return_code = child.wait()
        if forwarded_signal is not None and return_code == 0:
            return_code = 128 + forwarded_signal
        return return_code if return_code >= 0 else 128 - return_code
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
        end_record = {
            "elapsed_seconds": time.monotonic() - started,
            "minimum_mem_available_bytes": minimum_available,
            "peak_process_group_rss_bytes": peak_rss,
        }
        print(
            "RESOURCE_GUARD_END " + json.dumps(end_record, sort_keys=True),
            file=sys.stderr,
            flush=True,
        )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help="Command to execute; prefix with -- to separate wrapper options.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    try:
        config = config_from_environ()
        workspace = Path(os.environ.get("CHESS_DFM_REPO_ROOT", "/mountpoint/.exp"))
        return guarded_run(command, config, workspace)
    except GuardViolation as exc:
        print(f"RESOURCE_GUARD_REFUSAL {exc}", file=sys.stderr)
        return CONFIG_EXIT


if __name__ == "__main__":
    raise SystemExit(main())
