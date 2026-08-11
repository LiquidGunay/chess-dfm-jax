from __future__ import annotations

import pytest

from research.interpretability.hero_encoder_lr_audit import (
    optimizer_movement,
    schedule_budget,
    validation_trajectory,
)


def test_schedule_budget_preserves_constant_main_to_encoder_ratio() -> None:
    result = schedule_budget(
        {
            "learning_rate": 3e-4,
            "bt4_learning_rate": 1e-4,
            "lr_warmup_examples": 20,
            "lr_total_examples": 100,
            "lr_min_ratio": 0.01,
        },
        examples_per_update=10,
    )

    assert result["updates"] == 10
    assert result["main_to_encoder_peak_ratio"] == pytest.approx(3.0)
    assert result["integrated_main_learning_rate"] == pytest.approx(
        3.0 * result["integrated_encoder_learning_rate"]
    )
    assert result["checkpoints"]["1.0"]["ratio"] == pytest.approx(0.01)


def test_optimizer_movement_joins_encoder_names_and_partitions_energy() -> None:
    partition = {
        "schema_version": "synthetic",
        "leaves": [
            {
                "path": "encoder.a",
                "learning_rate_kind": "bt4",
                "optimizer": "muon",
                "dtype": "torch.bfloat16",
            },
            {
                "path": "encoder.b",
                "learning_rate_kind": "bt4",
                "optimizer": "nesterov_adamw",
                "dtype": "torch.bfloat16",
            },
            {
                "path": "head.c",
                "learning_rate_kind": "main",
                "optimizer": "nesterov_adamw",
                "dtype": "torch.float32",
            },
        ],
    }
    parameter_diff = {
        "group_metrics": {"all": {"delta_l2": 5.0}},
        "leaves": [
            {
                "name": "a",
                "parameter_count": 100,
                "unchanged_count": 99,
                "delta_l2": 3.0,
            },
            {
                "name": "b",
                "parameter_count": 50,
                "unchanged_count": 40,
                "delta_l2": 4.0,
            },
        ],
    }

    result = optimizer_movement(partition, parameter_diff)

    assert result["matched_encoder_leaf_count"] == 2
    assert result["by_optimizer"]["muon"]["unchanged_fraction"] == 0.99
    assert result["by_optimizer"]["muon"]["global_delta_energy_fraction"] == 0.36
    assert result["by_optimizer"]["nesterov_adamw"][
        "global_delta_energy_fraction"
    ] == 0.64


def test_validation_trajectory_requires_exact_milestones_and_reports_delta() -> None:
    rows = []
    for percentage in range(10, 101, 10):
        rows.append(
            {
                "target_percentage": percentage,
                "update": percentage,
                "examples": percentage * 100,
                "metrics": {
                    "loss": 10.0 - percentage / 10,
                    "root_legal_conditional_ce": 3.0 - percentage / 100,
                    "root_legal_top1_accuracy": percentage / 200,
                    "first_legal_mass": percentage / 100,
                    "dfm_ce_loss": 8.0 - percentage / 20,
                    "jepa_positive_loss": 1.0 - percentage / 200,
                    "wdl_loss": 2.0 - percentage / 200,
                },
            }
        )

    result = validation_trajectory(rows)

    assert len(result["milestones"]) == 10
    assert result["ten_to_hundred_percent_delta"]["loss"] == -9.0
    assert "not a Raw step-zero baseline" in result["caveat"]
