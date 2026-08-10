from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from research.compare_training_runs import (
    CONTROL_SCHEMA_VERSION,
    FIXED_POLICY_DISTILLATION_SCHEMA_VERSION,
    CURVE_METRICS,
    OBJECTIVE_ABLATION_SCHEMA_VERSION,
    POLICY_DISTILLATION_SCHEMA_VERSION,
    SCHEMA_VERSION,
    _compare_movement_reports,
    _metrics_records,
    compare_candidate,
    compare_fixed_policy_distillation,
    compare_legality_ablation,
    compare_policy_distillation,
    freeze_control,
)


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _run_config(
    *,
    data_seed: int = 0,
    recipe: str = "hero",
    legality_coeff: float = 2.0,
    policy_distill_coeff: float = 0.0,
    policy_distill_teacher_mode: str = "online",
    policy_distill_teacher_state_sha256: str = "",
) -> dict[str, object]:
    return {
        "source": {
            "combined_sha256": "a" * 64,
            "raw_asset": {"sha256": "b" * 64, "size_bytes": 1234},
        },
        "config": {
            "action_codec": "lc0_canonical_1858",
            "horizon": 8,
            "token_dim": 256,
            "z_dim": 1024,
            "dfm_ce_coeff": 1.0,
            "legality_coeff": legality_coeff,
            "policy_distill_coeff": policy_distill_coeff,
            "policy_distill_teacher_mode": policy_distill_teacher_mode,
            "policy_distill_teacher_state_sha256": (
                policy_distill_teacher_state_sha256
            ),
            "jepa_positive_coeff": 1.0,
            "root_legal_ce_coeff": 0.087,
            "target_sigreg_coeff": 2.0,
            "pred_sigreg_coeff": 2.0,
            "wdl_coeff": 0.25,
            # Intentionally excluded from matched scientific identity.
            "learning_rate": 0.0005 if recipe == "hero" else 0.000625,
        },
        "policy_distill_teacher": (
            {
                "mode": "checkpoint",
                "checkpoint_state_sha256": (
                    policy_distill_teacher_state_sha256
                ),
                "trainable_parameter_count": 0,
            }
            if policy_distill_teacher_mode == "checkpoint"
            else {
                "mode": "online",
                "checkpoint_state_sha256": None,
            }
        ),
        "args": {"batch_size": 1024, "data_start": 0, "seed": data_seed},
        "data": {
            "batch_schedule": "global_permutation",
            "batch_size": 1024,
            "batches_per_shard": 1,
            "file_manifest_sha256": "c" * 64,
            "samples_per_shard": 1024,
            "seed": data_seed,
            "seed_effective": True,
            "shard_count": 1_000,
            "shuffle_files": True,
            "steps_per_epoch": 27_679,
        },
        "live_validation": {
            "enabled": True,
            "fast_pool": {
                "pool_name": "fast",
                "indices_sha256": "d" * 64,
                "data": {
                    "file_manifest_sha256": "e" * 64,
                    "indices_sha256": "d" * 64,
                    "pool_name": "fast",
                    "position_count": 8_192,
                    "samples_per_shard": 1024,
                    "shard_count": 1_524,
                },
                "pool_definition": {
                    "array": "fast_val_global_index",
                    "count": 8_192,
                    "seed": 20_260_725_01,
                    "split": "val",
                },
            },
        },
    }


def _metric_record(update: int, *, improvement: float) -> dict[str, object]:
    noise = 0.03 * math.sin(update / 7.0)
    values: dict[str, float] = {
        "loss": 10.0 - 0.004 * update + noise - improvement,
        "dfm_ce_loss": 6.5 - 0.002 * update + 0.5 * noise - 0.5 * improvement,
        "weighted_legality_loss": 1.5 - 0.0005 * update - 0.1 * improvement,
        "jepa_positive_loss": 0.3 - 0.0001 * update - 0.05 * improvement,
        "jepa_sigreg_loss": 0.2 - 0.00005 * update - 0.05 * improvement,
        "jepa_pred_sigreg_loss": 0.2 - 0.00005 * update - 0.05 * improvement,
        "weighted_root_legal_conditional_ce": 0.3 - 0.0002 * update - 0.05 * improvement,
        "wdl_weighted_loss": 0.3 - 0.00005 * update - 0.05 * improvement,
        "accuracy": 0.03 + 0.0001 * update + 0.02 * improvement,
        "first_legal_mass": 0.2 + 0.0002 * update + 0.02 * improvement,
        "root_legal_conditional_ce": 3.0 - 0.003 * update - 0.2 * improvement,
        "wdl_loss": 1.1 - 0.0002 * update - 0.1 * improvement,
        "wdl_accuracy": 0.4 + 0.0001 * update + 0.02 * improvement,
        "gradient_global_norm": 30.0,
        "gradient_clip_scale": 1.0 / 30.0,
        "learning_rate": 0.0005,
        "bt4_learning_rate": 1.0e-5,
    }
    assert set(values) == set(CURVE_METRICS)
    return {
        "schema_version": "torch-eager-train-metrics-v1",
        "update": update,
        "examples": update * 1024,
        "optimizer_skipped_nonfinite": False,
        **values,
    }


def _write_run(
    root: Path,
    *,
    improvement: float,
    data_seed: int = 0,
    recipe: str = "hero",
    legality_coeff: float = 2.0,
    policy_distill_coeff: float = 0.0,
    policy_distill_teacher_mode: str = "online",
    policy_distill_teacher_state_sha256: str = "",
) -> None:
    root.mkdir()
    records = [_metric_record(update, improvement=improvement) for update in range(1, 555)]
    (root / "metrics.jsonl").write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    _write_json(
        root / "run_config.json",
        _run_config(
            data_seed=data_seed,
            recipe=recipe,
            legality_coeff=legality_coeff,
            policy_distill_coeff=policy_distill_coeff,
            policy_distill_teacher_mode=policy_distill_teacher_mode,
            policy_distill_teacher_state_sha256=(
                policy_distill_teacher_state_sha256
            ),
        ),
    )
    validation_improvement = improvement
    _write_json(
        root / "report.json",
        {
            "updates": 554,
            "recovery_checkpoints": [{"optimizer_update": update} for update in (100, 200, 300, 400, 500, 554)],
            "live_validation": {
                "records": [
                    {
                        "update": 554,
                        "examples": 554 * 1024,
                        "evaluation_examples": 8192,
                        "metrics": {
                            "dfm_ce_loss": 5.0 - 0.5 * validation_improvement,
                            "root_legal_conditional_ce": 1.5 - 0.2 * validation_improvement,
                            "jepa_positive_loss": 0.2 - 0.05 * validation_improvement,
                            "wdl_loss": 1.0 - 0.1 * validation_improvement,
                            "accuracy": 0.1 + 0.02 * validation_improvement,
                            "first_legal_mass": 0.3 + 0.02 * validation_improvement,
                            "wdl_accuracy": 0.5 + 0.02 * validation_improvement,
                            "root_legal_top1_accuracy": 0.4
                            + 0.02 * validation_improvement,
                            "base_policy_root_legal_conditional_ce": 1.4,
                            "base_policy_dfm_root_legal_kl": 0.02,
                            "weighted_base_policy_dfm_root_legal_kl": 0.02,
                            "base_policy_root_legal_top1_accuracy": 0.45,
                            "base_policy_dfm_root_legal_top1_agreement": 0.9,
                            "policy_teacher_active": 1.0,
                            "policy_teacher_root_legal_conditional_ce": 1.35,
                            "policy_teacher_dfm_root_legal_kl": 0.015,
                            "weighted_policy_teacher_dfm_root_legal_kl": 0.015,
                            "policy_teacher_root_legal_top1_accuracy": 0.46,
                            "policy_teacher_dfm_root_legal_top1_agreement": 0.92,
                            "wdl_brier_score": 0.7 - 0.02 * validation_improvement,
                            "wdl_ece_15": 0.2 - 0.01 * validation_improvement,
                            "wdl_expected_value_mse": 0.5
                            - 0.02 * validation_improvement,
                        },
                    }
                ]
            },
        },
    )


def test_freeze_then_compare_uses_matched_curve_and_frozen_gates(tmp_path: Path) -> None:
    control_dir = tmp_path / "control"
    candidate_dir = tmp_path / "candidate"
    _write_run(control_dir, improvement=0.0)
    _write_run(candidate_dir, improvement=0.25, recipe="hero_v2")

    contract = freeze_control(control_dir)
    assert contract["schema_version"] == CONTROL_SCHEMA_VERSION
    assert [
        threshold["anchor_update"]
        for threshold in contract["analysis_contract"]["loss_thresholds"]
    ] == [300, 400, 500, 554]
    contract_path = tmp_path / "control_contract.json"
    _write_json(contract_path, contract)

    result = compare_candidate(contract_path, control_dir, candidate_dir)
    assert result["schema_version"] == SCHEMA_VERSION
    assert result["allocation_decision"]["curve_status"] == "faster"
    assert result["allocation_decision"]["eligible_for_more_compute"] is True
    assert result["frozen_validation"]["passed"] is True
    diagnostics = result["frozen_validation"]["descriptive_diagnostics"]
    assert diagnostics["eligibility_effect"] is False
    assert diagnostics["declared_after_control_freeze"] is True
    assert diagnostics["comparisons"]["root_legal_top1_accuracy"]["delta"] > 0.0
    assert result["paired_curve_differences"]["loss"]["ci"][1] < 0.0
    assert all(
        crossing["candidate_minus_control_examples"] <= 0
        for crossing in result["loss_threshold_crossings"]
    )


def test_legality_ablation_compares_canonical_loss_and_ignores_legal_mass_gate(
    tmp_path: Path,
) -> None:
    control_dir = tmp_path / "control"
    candidate_dir = tmp_path / "candidate"
    _write_run(control_dir, improvement=0.0)
    _write_run(
        candidate_dir,
        improvement=0.25,
        recipe="hero_v2",
        legality_coeff=0.0,
    )
    contract = freeze_control(control_dir)
    contract_path = tmp_path / "control_contract.json"
    _write_json(contract_path, contract)

    result = compare_legality_ablation(
        contract_path,
        control_dir,
        candidate_dir,
        root_top1_floor=0.4,
    )

    assert result["schema_version"] == OBJECTIVE_ABLATION_SCHEMA_VERSION
    assert result["objective_difference"]["total_training_loss_comparable"] is False
    assert result["paired_curve_differences"]["canonical_nonlegality_loss"]["ci"][1] < 0
    assert result["frozen_validation"]["checks"]["first_legal_mass"][
        "promotion_effect"
    ] is False
    assert result["decision"]["eligible_for_movement_and_arena"] is True
    assert result["decision"]["phase2_authorized"] is False


def test_policy_distillation_compares_only_the_canonical_predictive_objective(
    tmp_path: Path,
) -> None:
    control_dir = tmp_path / "control"
    candidate_dir = tmp_path / "candidate"
    _write_run(
        control_dir,
        improvement=0.0,
        recipe="hero_v2",
        legality_coeff=0.0,
        policy_distill_coeff=0.0,
    )
    _write_run(
        candidate_dir,
        improvement=0.25,
        recipe="hero_v2",
        legality_coeff=0.0,
        policy_distill_coeff=1.0,
    )
    contract_path = tmp_path / "control_contract.json"
    _write_json(contract_path, freeze_control(control_dir))

    result = compare_policy_distillation(
        contract_path,
        control_dir,
        candidate_dir,
        root_top1_floor=0.4,
    )

    assert result["schema_version"] == POLICY_DISTILLATION_SCHEMA_VERSION
    assert result["objective_difference"]["only_allowed_config_key"] == (
        "policy_distill_coeff"
    )
    assert result["objective_difference"]["shared_legality_coeff"] == 0.0
    assert result["objective_difference"]["total_training_loss_comparable"] is False
    assert result["paired_curve_differences"]["canonical_predictive_loss"]["ci"][1] < 0
    assert result["decision"]["canonical_predictive_loss_improved"] is True
    assert result["distillation_diagnostics"] == {
        "base_policy_root_legal_conditional_ce": 1.4,
        "base_policy_dfm_root_legal_kl": 0.02,
        "weighted_base_policy_dfm_root_legal_kl": 0.02,
        "base_policy_root_legal_top1_accuracy": 0.45,
        "base_policy_dfm_root_legal_top1_agreement": 0.9,
    }
    assert result["decision"]["eligible_for_movement_and_arena"] is True

    with pytest.raises(ValueError, match="candidate policy-distillation"):
        compare_policy_distillation(
            contract_path,
            control_dir,
            candidate_dir,
            expected_candidate_policy_distill_coeff=2.0,
            root_top1_floor=0.4,
        )


def test_fixed_policy_teacher_comparison_binds_checkpoint_and_uses_frozen_gates(
    tmp_path: Path,
) -> None:
    teacher_state_sha256 = "f" * 64
    control_dir = tmp_path / "control"
    candidate_dir = tmp_path / "candidate"
    _write_run(
        control_dir,
        improvement=0.0,
        recipe="hero_v2",
        legality_coeff=0.0,
        policy_distill_coeff=0.0,
    )
    _write_run(
        candidate_dir,
        improvement=0.25,
        recipe="hero_v2",
        legality_coeff=0.0,
        policy_distill_coeff=1.0,
        policy_distill_teacher_mode="checkpoint",
        policy_distill_teacher_state_sha256=teacher_state_sha256,
    )
    contract_path = tmp_path / "control_contract.json"
    _write_json(contract_path, freeze_control(control_dir))

    result = compare_fixed_policy_distillation(
        contract_path,
        control_dir,
        candidate_dir,
        expected_candidate_teacher_state_sha256=teacher_state_sha256,
        root_top1_floor=0.4,
    )

    assert result["schema_version"] == FIXED_POLICY_DISTILLATION_SCHEMA_VERSION
    assert result["objective_difference"]["allowed_changed_config_keys"] == [
        "policy_distill_coeff",
        "policy_distill_teacher_mode",
        "policy_distill_teacher_state_sha256",
    ]
    assert result["teacher_contrast"]["control"]["mode"] == "online"
    assert result["teacher_contrast"]["candidate"]["mode"] == "checkpoint"
    assert (
        result["teacher_contrast"]["candidate"]["record"][
            "trainable_parameter_count"
        ]
        == 0
    )
    assert result["fixed_teacher_diagnostics"] == {
        "policy_teacher_active": 1.0,
        "policy_teacher_root_legal_conditional_ce": 1.35,
        "policy_teacher_dfm_root_legal_kl": 0.015,
        "weighted_policy_teacher_dfm_root_legal_kl": 0.015,
        "policy_teacher_root_legal_top1_accuracy": 0.46,
        "policy_teacher_dfm_root_legal_top1_agreement": 0.92,
    }
    assert result["decision"]["canonical_predictive_curve_status"] == "faster"
    assert result["decision"]["eligible_for_movement_and_arena"] is True
    assert result["decision"]["phase2_authorized"] is False

    with pytest.raises(ValueError, match="checkpoint state drift"):
        compare_fixed_policy_distillation(
            contract_path,
            control_dir,
            candidate_dir,
            expected_candidate_teacher_state_sha256="e" * 64,
            root_top1_floor=0.4,
        )


def test_compare_rejects_a_different_ordered_data_seed(tmp_path: Path) -> None:
    control_dir = tmp_path / "control"
    candidate_dir = tmp_path / "candidate"
    _write_run(control_dir, improvement=0.0)
    _write_run(candidate_dir, improvement=0.25, data_seed=1, recipe="hero_v2")
    contract = freeze_control(control_dir)
    contract_path = tmp_path / "control_contract.json"
    _write_json(contract_path, contract)

    with pytest.raises(ValueError, match="identity does not match"):
        compare_candidate(contract_path, control_dir, candidate_dir)


def test_metrics_reader_rejects_a_gap_in_the_matched_prefix(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    records = [_metric_record(1, improvement=0.0), _metric_record(3, improvement=0.0)]
    (run_dir / "metrics.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="contiguous"):
        _metrics_records(run_dir)


def test_metrics_reader_rejects_nonfinite_health_values(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    record = _metric_record(1, improvement=0.0)
    record["gradient_global_norm"] = float("inf")
    (run_dir / "metrics.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="must be finite"):
        _metrics_records(run_dir)



def _write_movement_report(
    path: Path,
    *,
    checkpoint_sha256: str,
    multiplier: float,
) -> None:
    groups = {}
    for index, group in enumerate(("all", "trunk", "embedding", "layers", "policy_head"), 1):
        groups[group] = {
            "relative_delta_l2": multiplier * index * 0.0001,
            "unchanged_fraction": 1.0 - multiplier * index * 0.001,
            "cosine": 0.999,
        }
    _write_json(
        path,
        {
            "schema_version": "bt4-raw-hero-parameter-diff-v1",
            "models": {
                "raw_asset_sha256": "a" * 64,
                "encoder_layout_sha256": "b" * 64,
                "hero_state_sha256": checkpoint_sha256,
            },
            "group_metrics": groups,
        },
    )


def test_movement_screen_requires_more_visible_change_without_runaway(tmp_path: Path) -> None:
    control_path = tmp_path / "control_movement.json"
    candidate_path = tmp_path / "candidate_movement.json"
    control_checkpoint = "c" * 64
    candidate_checkpoint = "d" * 64
    _write_movement_report(
        control_path,
        checkpoint_sha256=control_checkpoint,
        multiplier=1.0,
    )
    _write_movement_report(
        candidate_path,
        checkpoint_sha256=candidate_checkpoint,
        multiplier=2.0,
    )

    comparison = _compare_movement_reports(
        control_path,
        candidate_path,
        control_checkpoint_sha256=control_checkpoint,
        candidate_checkpoint_sha256=candidate_checkpoint,
    )

    assert comparison["screen"]["healthier_visible_encoder_movement"] is True
    assert comparison["screen"]["guardrail_pass"] is True
    assert comparison["groups"]["trunk"]["relative_delta_l2_ratio"] == pytest.approx(2.0)
    assert comparison["groups"]["trunk"]["changed_fraction_ratio"] == pytest.approx(2.0)


def test_movement_screen_rejects_checkpoint_mismatch_and_runaway(tmp_path: Path) -> None:
    control_path = tmp_path / "control_movement.json"
    candidate_path = tmp_path / "candidate_movement.json"
    _write_movement_report(
        control_path,
        checkpoint_sha256="c" * 64,
        multiplier=1.0,
    )
    _write_movement_report(
        candidate_path,
        checkpoint_sha256="d" * 64,
        multiplier=30.0,
    )

    comparison = _compare_movement_reports(
        control_path,
        candidate_path,
        control_checkpoint_sha256="c" * 64,
        candidate_checkpoint_sha256="d" * 64,
    )
    assert comparison["screen"]["healthier_visible_encoder_movement"] is True
    assert comparison["screen"]["guardrail_pass"] is False

    with pytest.raises(ValueError, match="does not match"):
        _compare_movement_reports(
            control_path,
            candidate_path,
            control_checkpoint_sha256="c" * 64,
            candidate_checkpoint_sha256="e" * 64,
        )
