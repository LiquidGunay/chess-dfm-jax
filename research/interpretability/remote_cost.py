"""Fail-closed cost estimates for Modal interpretation jobs."""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import asdict, dataclass
from decimal import Decimal
from typing import Any


PRICE_SNAPSHOT_DATE = "2026-08-09"
PRICE_SOURCE = "https://modal.com/pricing"
GPU_DOLLARS_PER_SECOND = {
    "T4": Decimal("0.000164"),
    "L4": Decimal("0.000222"),
    "A10": Decimal("0.000306"),
    "L40S": Decimal("0.000542"),
    "A100-40GB": Decimal("0.000583"),
    "A100-80GB": Decimal("0.000694"),
    "H100": Decimal("0.001097"),
    "H200": Decimal("0.001261"),
    "B200": Decimal("0.001736"),
    "B300": Decimal("0.001972"),
}
CPU_DOLLARS_PER_CORE_SECOND = Decimal("0.0000131")
MEMORY_DOLLARS_PER_GIB_SECOND = Decimal("0.00000222")
WORKSPACE_MONTHLY_BUDGET = Decimal("42.50")
STOP_LAUNCH_FRACTION = Decimal("0.80")


@dataclass(frozen=True)
class ModalJobSpec:
    name: str
    timeout_seconds: int
    startup_timeout_seconds: int
    retries: int
    cpu_cores: Decimal
    memory_gib: Decimal
    gpu: str | None = None

    def validate(self) -> None:
        if not self.name:
            raise ValueError("Job name must be non-empty")
        if self.timeout_seconds <= 0 or self.startup_timeout_seconds < 0:
            raise ValueError("Job timeouts must be positive/non-negative")
        if self.retries < 0 or self.cpu_cores < 0 or self.memory_gib < 0:
            raise ValueError("Retries and resources must be non-negative")
        if self.gpu is not None and self.gpu not in GPU_DOLLARS_PER_SECOND:
            raise ValueError(f"No frozen price for GPU {self.gpu!r}")


def estimate_modal_job(spec: ModalJobSpec) -> dict[str, Any]:
    """Upper bound for declared attempts, before provider crash rescheduling."""

    spec.validate()
    attempt_count = spec.retries + 1
    seconds = Decimal(attempt_count * (spec.startup_timeout_seconds + spec.timeout_seconds))
    gpu_rate = GPU_DOLLARS_PER_SECOND.get(spec.gpu, Decimal("0"))
    gpu_cost = seconds * gpu_rate
    cpu_cost = seconds * spec.cpu_cores * CPU_DOLLARS_PER_CORE_SECOND
    memory_cost = seconds * spec.memory_gib * MEMORY_DOLLARS_PER_GIB_SECOND
    total = gpu_cost + cpu_cost + memory_cost
    return {
        "schema_version": "modal-job-cost-estimate-v2",
        "job": {
            **asdict(spec),
            "cpu_cores": str(spec.cpu_cores),
            "memory_gib": str(spec.memory_gib),
        },
        "pricing": {
            "snapshot_date": PRICE_SNAPSHOT_DATE,
            "source": PRICE_SOURCE,
            "gpu_dollars_per_second": str(gpu_rate),
            "cpu_dollars_per_core_second": str(CPU_DOLLARS_PER_CORE_SECOND),
            "memory_dollars_per_gib_second": str(MEMORY_DOLLARS_PER_GIB_SECOND),
        },
        "declared_attempt_billable_seconds": int(seconds),
        "attempt_count": attempt_count,
        "gpu_dollars": str(gpu_cost),
        "cpu_dollars": str(cpu_cost),
        "memory_dollars": str(memory_cost),
        "declared_attempt_upper_bound_dollars": str(total),
        "assumption": (
            "upper bound for configured attempts charges startup_timeout + execution "
            "timeout for every allowed attempt"
        ),
        "excluded_provider_behavior": (
            "Modal may reschedule crashed containers independently of configured "
            "function retries; image builds, scheduling, collection lag, and those "
            "platform reschedules are covered by the 20% workspace reserve and the "
            "provider-native hard budget, not this per-job estimate"
        ),
    }


def enforce_modal_budget(
    estimate: dict[str, Any],
    *,
    recorded_month_spend: Decimal,
    per_job_cap: Decimal,
    workspace_budget: Decimal = WORKSPACE_MONTHLY_BUDGET,
    stop_fraction: Decimal = STOP_LAUNCH_FRACTION,
) -> dict[str, Any]:
    if any(value < 0 for value in (recorded_month_spend, per_job_cap, workspace_budget)):
        raise ValueError("Budget values must be non-negative")
    if not Decimal("0") < stop_fraction <= Decimal("1"):
        raise ValueError("stop_fraction must be in (0, 1]")
    estimated = Decimal(str(estimate["declared_attempt_upper_bound_dollars"]))
    stop_threshold = workspace_budget * stop_fraction
    projected = recorded_month_spend + estimated
    reasons: list[str] = []
    if estimated > per_job_cap:
        reasons.append(f"declared-attempt estimate ${estimated} exceeds per-job cap ${per_job_cap}")
    if recorded_month_spend >= stop_threshold:
        reasons.append(
            f"recorded spend ${recorded_month_spend} already reached stop threshold "
            f"${stop_threshold}"
        )
    if projected > stop_threshold:
        reasons.append(f"projected spend ${projected} exceeds stop threshold ${stop_threshold}")
    decision = {
        "schema_version": "modal-budget-decision-v2",
        "allowed": not reasons,
        "recorded_month_spend": str(recorded_month_spend),
        "declared_attempt_upper_bound": str(estimated),
        "projected_month_spend": str(projected),
        "per_job_cap": str(per_job_cap),
        "workspace_budget": str(workspace_budget),
        "stop_fraction": str(stop_fraction),
        "stop_threshold": str(stop_threshold),
        "reasons": reasons,
        "estimate": estimate,
    }
    if reasons:
        raise RuntimeError("Modal cost guard refused launch: " + "; ".join(reasons))
    return decision


def modal_month_spend() -> Decimal:
    """Read this month's metered cost from the authenticated Modal CLI."""

    result = subprocess.run(
        ["modal", "billing", "summary", "--for", "this month", "--json"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "Cannot obtain Modal month spend; refusing an unmetered launch: "
            + result.stderr.strip()
        )
    try:
        payload = json.loads(result.stdout)
        return Decimal(str(payload["metered_cost"]))
    except (json.JSONDecodeError, KeyError, ValueError) as exc:
        raise RuntimeError("Modal billing summary had an unexpected schema") from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--timeout-seconds", type=int, required=True)
    parser.add_argument("--startup-timeout-seconds", type=int, default=0)
    parser.add_argument("--retries", type=int, default=0)
    parser.add_argument("--cpu-cores", type=Decimal, default=Decimal("0.125"))
    parser.add_argument("--memory-gib", type=Decimal, default=Decimal("0.125"))
    parser.add_argument("--gpu", choices=sorted(GPU_DOLLARS_PER_SECOND))
    parser.add_argument("--recorded-spend", type=Decimal)
    parser.add_argument("--per-job-cap", type=Decimal, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    estimate = estimate_modal_job(
        ModalJobSpec(
            name=args.name,
            timeout_seconds=args.timeout_seconds,
            startup_timeout_seconds=args.startup_timeout_seconds,
            retries=args.retries,
            cpu_cores=args.cpu_cores,
            memory_gib=args.memory_gib,
            gpu=args.gpu,
        )
    )
    spend = args.recorded_spend if args.recorded_spend is not None else modal_month_spend()
    decision = enforce_modal_budget(
        estimate,
        recorded_month_spend=spend,
        per_job_cap=args.per_job_cap,
    )
    print(json.dumps(decision, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
