from __future__ import annotations

from pathlib import Path

from research.analyze_full_loop_ablation import (
    build_analysis,
    evaluate_preregistered_gates,
    paired_block_bootstrap,
)


def _arena(score: float, lower: float, upper: float) -> dict[str, object]:
    return {
        "score": score,
        "score_interval_95": [lower, upper],
        "scientific_role": "preregistered primary",
        "zero_faults": True,
        "complete_move_codec_coverage": True,
    }


def test_paired_block_bootstrap_is_deterministic() -> None:
    deltas = [-0.2 + index * 0.001 for index in range(96)]
    first = paired_block_bootstrap(deltas, replicates=200, seed=7)
    second = paired_block_bootstrap(deltas, replicates=200, seed=7)

    assert first == second
    assert first["mean_candidate_minus_control"] < 0.0
    assert first["ci"][1] < 0.0


def test_exact_gate_rule_rejects_material_one_pass_damage() -> None:
    arenas = {
        "candidate_8_vs_candidate_1": _arena(0.74, 0.62, 0.86),
        "candidate_8_vs_control_1": _arena(0.51, 0.39, 0.63),
        "candidate_1_vs_control_1": _arena(0.26, 0.14, 0.38),
        "candidate_8_vs_control_8": _arena(0.81, 0.69, 0.93),
    }

    result = evaluate_preregistered_gates(
        arenas,
        latent_health_passed=True,
        systems_passed=True,
        movement_passed=True,
    )

    assert result["mechanism_success"] is False
    assert result["failed_gates"] == ["candidate_one_pass_not_materially_damaged"]


def test_repository_artifacts_reproduce_sealed_decision() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    result = build_analysis(repo_root)

    assert result["matched_training_contract"]["matched"] is True
    assert result["latent_health"]["passed"] is True
    assert result["systems_diagnostic"]["passed"] is True
    assert result["parameter_movement"]["passed"] is True
    assert result["preregistered_decision"]["mechanism_success"] is False
    assert result["arenas"]["candidate_8_vs_candidate_1"]["score"] == 0.74609375
