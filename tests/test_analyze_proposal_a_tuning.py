from __future__ import annotations

from research.analyze_proposal_a_tuning import _decay_analysis, _lr_analysis


def _row(
    arm: str,
    *,
    lr: float,
    loss: float,
    top1: float,
    wdl: float,
    decay: float = 0.0,
    mode: str = "decoupled",
) -> dict:
    return {
        "arm": arm,
        "learning_rate": lr,
        "weight_decay": decay,
        "weight_decay_mode": mode,
        "validation": {
            "loss": loss,
            "root_legal_top1_accuracy": top1,
            "wdl_current_loss": wdl,
        },
    }


def test_lr_analysis_applies_policy_and_value_guardrails() -> None:
    report = _lr_analysis(
        [
            _row("low", lr=5e-5, loss=9.0, top1=0.50, wdl=0.90),
            _row("winner", lr=1e-4, loss=8.7, top1=0.49, wdl=0.80),
            _row("policy-collapse", lr=2e-4, loss=8.4, top1=0.40, wdl=0.75),
            _row("higher-collapse", lr=4e-4, loss=9.4, top1=0.10, wdl=1.20),
        ]
    )
    assert report["winner_arm"] == "winner"
    assert report["substantial_vs_lowest_rate"] is True
    by_arm = {row["arm"]: row for row in report["runs"]}
    assert by_arm["policy-collapse"]["promotion_eligible"] is False


def test_decay_analysis_defaults_to_zero_without_material_gain() -> None:
    report = _decay_analysis(
        [
            _row("zero", lr=1e-4, loss=8.0, top1=0.50, wdl=0.80),
            _row(
                "ordinary",
                lr=1e-4,
                loss=7.98,
                top1=0.50,
                wdl=0.79,
                decay=0.02,
            ),
            _row(
                "cautious",
                lr=1e-4,
                loss=7.95,
                top1=0.48,
                wdl=0.75,
                decay=0.02,
                mode="cautious",
            ),
        ]
    )
    assert report["winner_arm"] == "zero"
