import pytest
from pathlib import Path

from research.analysis.import_wandb_logs import display_identity, merged_events, pool_identity, scalar_metrics


def validation(update=2, count=8192):
    return {"update": update, "examples": update * 1024, "evaluation_examples": count,
            "pool": {"data": {"file_manifest_sha256": "abc"},
                     "pool_definition": {"count": count, "seed": 20000}},
            "metrics": {"loss": 2.0}}


def test_metrics_no_strings_or_nonfinite():
    assert scalar_metrics({"loss": 1.0, "token": "secret", "nested": {"key": "secret"},
                           "bad": float("nan"), "update": 5, "skipped": False}, "train") == {
                               "train/loss": 1.0, "train/skipped": False}


def test_merge_preserves_examples_and_original_pool_namespace():
    train = [{"update": i, "examples": i * 1024, "loss": 3.0} for i in (1, 2)]
    events, pools = merged_events(train, [validation()])
    prefix = next(iter(pools))
    assert len(events) == 2
    assert events[-1]["train/loss"] == 3.0
    assert events[-1][prefix + "/loss"] == 2.0
    assert prefix.startswith("val_original_8192_")
    assert "val/loss" not in events[-1]


def test_pool_identity_separates_equal_sized_but_different_data():
    a, b = validation(), validation()
    b["pool"]["data"]["file_manifest_sha256"] = "other"
    assert pool_identity(a)[1] != pool_identity(b)[1]


def test_duplicates_fail_closed():
    with pytest.raises(ValueError, match="unique increasing"):
        merged_events([{"update": 1, "examples": 1024}] * 2, [])
    with pytest.raises(ValueError, match="Duplicate validation"):
        merged_events([], [validation(), validation()])


def test_conflicting_examples_fail_closed():
    with pytest.raises(ValueError, match="examples disagree"):
        merged_events([{"update": 2, "examples": 42}], [validation()])


def test_architecture_exposures_and_diagnostics_have_separate_groups():
    groups = [display_identity("hero_v2_phase1", Path(path), {}, {}, updates)[1]
              for path, updates in [("v2", 554), ("u1024/v2", 1024), ("diagnostics/v2", 10)]]
    assert groups == ["hero-554-current-only", "hero-1024-current-only", "diagnostics"]
