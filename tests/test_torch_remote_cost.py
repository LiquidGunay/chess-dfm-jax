from __future__ import annotations

from decimal import Decimal

import pytest

from research.interpretability.remote_cost import (
    ModalJobSpec,
    budget_usage_from_summary,
    enforce_modal_budget,
    estimate_modal_job,
)


def test_budget_excludes_free_storage_but_not_compute_credits() -> None:
    assert budget_usage_from_summary({
        "metered_cost": "22.71684919", "billed_cost": "0",
        "adjustments": {"free_storage": "-19.61684919", "credits": "-3.1"},
        "metered_cost_breakdown": {"volumes": "19.61684919"},
    }) == Decimal("3.10000000")
    assert budget_usage_from_summary({
        "metered_cost": "120", "billed_cost": "0",
        "adjustments": {"free_storage": "-90", "credits": "-30"},
        "metered_cost_breakdown": {"volumes": "100"},
    }) == Decimal("30")  # Includes $10 of non-waived storage.
    assert budget_usage_from_summary({"metered_cost": "22.7"}) == Decimal("22.7")


@pytest.mark.parametrize("metered,free,volumes", [
    ("NaN", "0", "0"), ("1", "-2", "2"), ("10", "-5", "4"),
    ("10", "1", "1"), ("Infinity", "0", "0"), ("-1", "0", "0"),
])
def test_inconsistent_billing_refuses_launch(metered: str, free: str, volumes: str) -> None:
    with pytest.raises(RuntimeError, match="inconsistent"):
        budget_usage_from_summary({
            "metered_cost": metered,
            "adjustments": {"free_storage": free},
            "metered_cost_breakdown": {"volumes": volumes},
        })


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
