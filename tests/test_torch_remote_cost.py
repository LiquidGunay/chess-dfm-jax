from __future__ import annotations

from decimal import Decimal

import pytest

from research.interpretability.remote_cost import (
    ModalJobSpec,
    enforce_modal_budget,
    estimate_modal_job,
)


def test_cost_estimate_includes_startup_timeout_and_every_retry() -> None:
    estimate = estimate_modal_job(
        ModalJobSpec(
            name="test",
            timeout_seconds=60,
            startup_timeout_seconds=120,
            retries=1,
            cpu_cores=Decimal("0.25"),
            memory_gib=Decimal("0.5"),
        )
    )
    assert estimate["attempt_count"] == 2
    assert estimate["declared_attempt_billable_seconds"] == 360
    assert Decimal(estimate["declared_attempt_upper_bound_dollars"]) > 0
    assert "reschedule" in estimate["excluded_provider_behavior"]


def test_t4_smoke_estimate_stays_under_ten_cent_cap() -> None:
    estimate = estimate_modal_job(
        ModalJobSpec(
            name="gpu-smoke",
            timeout_seconds=60,
            startup_timeout_seconds=300,
            retries=0,
            cpu_cores=Decimal("1"),
            memory_gib=Decimal("2"),
            gpu="T4",
        )
    )
    assert Decimal(estimate["declared_attempt_upper_bound_dollars"]) < Decimal("0.10")
    decision = enforce_modal_budget(
        estimate,
        recorded_month_spend=Decimal("0"),
        per_job_cap=Decimal("0.10"),
    )
    assert decision["allowed"] is True
    assert Decimal(decision["declared_attempt_upper_bound"]) < Decimal("0.10")
    assert Decimal(decision["stop_threshold"]) == Decimal("34.000")


def test_budget_guard_fails_on_job_cap_or_month_stop() -> None:
    estimate = estimate_modal_job(
        ModalJobSpec(
            name="large",
            timeout_seconds=3600,
            startup_timeout_seconds=600,
            retries=2,
            cpu_cores=Decimal("4"),
            memory_gib=Decimal("32"),
            gpu="H100",
        )
    )
    with pytest.raises(RuntimeError, match="per-job cap"):
        enforce_modal_budget(
            estimate,
            recorded_month_spend=Decimal("0"),
            per_job_cap=Decimal("1"),
        )
    small = estimate_modal_job(
        ModalJobSpec(
            name="small",
            timeout_seconds=1,
            startup_timeout_seconds=0,
            retries=0,
            cpu_cores=Decimal("0.125"),
            memory_gib=Decimal("0.125"),
        )
    )
    with pytest.raises(RuntimeError, match="already reached"):
        enforce_modal_budget(
            small,
            recorded_month_spend=Decimal("34"),
            per_job_cap=Decimal("1"),
        )


def test_unknown_gpu_fails_closed() -> None:
    with pytest.raises(ValueError, match="No frozen price"):
        estimate_modal_job(
            ModalJobSpec(
                name="unknown",
                timeout_seconds=10,
                startup_timeout_seconds=0,
                retries=0,
                cpu_cores=Decimal("1"),
                memory_gib=Decimal("1"),
                gpu="mystery",
            )
        )
