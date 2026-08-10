from __future__ import annotations

import dataclasses
import importlib
import sys

import pytest


def test_native_torch_arena_import_does_not_load_jax() -> None:
    sys.modules.pop("research.evaluate_arena", None)

    module = importlib.import_module("research.evaluate_arena")

    assert module.RELATIVE_ELO_SCOPE == "checkpoint_pool_relative_only"
    assert "jax" not in sys.modules
    assert "flax" not in sys.modules


def test_torch_hero_config_accepts_pre_teacher_provenance_fields() -> None:
    module = importlib.import_module("research.evaluate_arena")
    from research.train_torch import HERO_CONFIG

    recorded = dataclasses.asdict(
        dataclasses.replace(
            HERO_CONFIG,
            remat_bt4_blocks=True,
            remat_projector_blocks=True,
            remat_dfm_blocks=False,
            use_bt4_sdpa=True,
            use_head_sdpa=True,
        )
    )
    legacy_missing = {
        "policy_distill_coeff",
        "policy_distill_teacher_mode",
        "policy_distill_teacher_state_sha256",
    }
    for name in legacy_missing:
        recorded.pop(name)

    validated = module._validated_torch_hero_model_config(recorded)

    assert validated == recorded
    assert legacy_missing.isdisjoint(validated)


def test_arena_git_commit_accepts_verified_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = importlib.import_module("research.evaluate_arena")
    expected = "a" * 40
    monkeypatch.setenv("CHESS_DFM_GIT_COMMIT", expected)

    def unexpected_git(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("verified commit override should avoid invoking Git")

    monkeypatch.setattr(module.subprocess, "run", unexpected_git)

    assert module._git_commit() == expected
