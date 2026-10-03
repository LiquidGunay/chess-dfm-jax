from __future__ import annotations

import importlib.util
import json
import sys
import types
from decimal import Decimal
from pathlib import Path

import pytest

if "modal" not in sys.modules and importlib.util.find_spec("modal") is None:
    fake_modal = types.ModuleType("modal")

    class _Image:
        @classmethod
        def debian_slim(cls, **_kwargs: object) -> "_Image":
            return cls()

        @classmethod
        def from_registry(cls, *_args: object, **_kwargs: object) -> "_Image":
            return cls()

        def __getattr__(self, _name: str):
            return lambda *_args, **_kwargs: self

    class _Volume:
        @classmethod
        def from_name(cls, *_args: object, **_kwargs: object) -> "_Volume":
            return cls()

        def with_mount_options(self, **_kwargs: object) -> "_Volume":
            return self

    class _App:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def function(self, **_kwargs: object):
            return lambda function: function

        def local_entrypoint(self):
            return lambda function: function

    fake_modal.Secret = type(
        "_Secret", (), {"from_name": classmethod(lambda cls, *args, **kwargs: cls())}
    )
    fake_modal.App = _App
    fake_modal.Image = _Image
    fake_modal.Volume = _Volume
    sys.modules["modal"] = fake_modal

from research.infra.modal_arena_app import (
    ALLOWED_PAIR_COUNTS,
    ALLOWED_POLICY_MODES,
    ARENA_SOURCE_TREE_PATHS,
    _arena_command,
    _arena_identity,
    _arena_job_spec,
    _arena_source_tree_sha256,
    _completed_arena_summary,
)
from research.interpretability.remote_cost import estimate_modal_job


def _label(suffix: str) -> str:
    return f"hero-training-phase1-v2_fp32_1_30-l40s-{suffix}"


def test_arena_identity_binds_both_checkpoints_and_options() -> None:
    common = {
        "candidate_label": _label("a" * 20),
        "candidate_state_sha256": "1" * 64,
        "opponent_label": _label("b" * 20),
        "opponent_state_sha256": "2" * 64,
        "candidate_policy_mode": "dfm",
        "opponent_policy_mode": "dfm",
        "pair_count": 128,
        "candidate_refinement_passes": 8,
        "opponent_refinement_passes": 1,
        "opening_start_index": 256,
        "arena_source_tree_sha256": "3" * 64,
    }
    first = _arena_identity(**common)
    assert first != _arena_identity(**common, raw_bt4_plane_history_mode="current_only_as_preprocessed")

    assert first == _arena_identity(**common)
    assert first.startswith("hero-arena-128p-o256-cpass8-opass1-")
    assert first != _arena_identity(
        **{**common, "candidate_state_sha256": "4" * 64}
    )
    assert first != _arena_identity(
        **{**common, "candidate_refinement_passes": 1}
    )
    assert first != _arena_identity(
        **{**common, "opponent_refinement_passes": 8}
    )
    assert first != _arena_identity(
        **{**common, "opening_start_index": 384}
    )
    assert first != _arena_identity(
        **{**common, "candidate_policy_mode": "policy_only"}
    )
    assert set(ALLOWED_POLICY_MODES) == {"dfm", "policy_only"}


def test_arena_command_is_native_torch_bounded_and_resumable(tmp_path: Path) -> None:
    command = _arena_command(
        candidate_root=tmp_path / "candidate",
        opponent_root=tmp_path / "opponent",
        models_dir=tmp_path / "models",
        output_dir=tmp_path / "output",
        candidate_policy_mode="policy_only",
        opponent_policy_mode="dfm",
        pair_count=128,
        candidate_refinement_passes=8,
        opponent_refinement_passes=1,
        opening_start_index=256,
        resume=True,
    )

    assert command[:3] == [sys.executable, "-m", "research.evaluate_arena"]
    assert "--candidate-torch-hero" in command
    assert "--opponent-torch-hero" in command
    assert command[command.index("--candidate-torch-hero-policy-mode") + 1] == (
        "policy_only"
    )
    assert command[command.index("--opponent-torch-hero-policy-mode") + 1] == "dfm"
    assert command[command.index("--pair-count") + 1] == "128"
    assert command[command.index("--opening-start-index") + 1] == "256"
    assert command[command.index("--block-pairs") + 1] == "16"
    assert command[command.index("--additional-ply-cap") + 1] == "256"
    assert command[command.index("--candidate-refinement-passes") + 1] == "8"
    assert command[command.index("--opponent-refinement-passes") + 1] == "1"
    assert command[-1] == "--resume"


def test_arena_cost_spec_fits_cap_and_uses_l4() -> None:
    for pair_count in ALLOWED_PAIR_COUNTS:
        spec, cap = _arena_job_spec(pair_count, 8, 1)
        estimate = estimate_modal_job(spec)
        assert spec.gpu == "L4"
        assert spec.retries == 0
        assert Decimal(estimate["declared_attempt_upper_bound_dollars"]) <= cap


def test_arena_source_identity_is_content_bound(tmp_path: Path) -> None:
    for relative in ARENA_SOURCE_TREE_PATHS:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative, encoding="utf-8")
    package = tmp_path / "chess_dfm_jax"
    package.mkdir(exist_ok=True)
    (package / "encoding.py").write_text("VALUE = 1\n", encoding="utf-8")
    (package / "policy_moves.txt").write_text("move\n", encoding="utf-8")
    (package / "policy_attn_map.txt").write_text("map\n", encoding="utf-8")

    first = _arena_source_tree_sha256(tmp_path)
    assert first == _arena_source_tree_sha256(tmp_path)
    (tmp_path / "research/evaluate_arena.py").write_text(
        "arena revision 2\n",
        encoding="utf-8",
    )
    assert first != _arena_source_tree_sha256(tmp_path)


def test_arena_image_supports_inductor_and_failures_surface_bounded_tail() -> None:
    source = Path("research/infra/modal_arena_app.py").read_text(
        encoding="utf-8"
    )

    assert '.apt_install("gcc", "g++")' in source
    assert "[-4_000:]" in source
    assert '"stdout_log_sha256"' in source
    assert "terminal output:\\n{stdout_tail}" in source
    assert "call = arena_l4.spawn(" in source
    assert "arena_l4.remote(" not in source
    assert '"function_call_id": call.object_id' in source


@pytest.mark.parametrize("raw_opponent", [False, True])
def test_completed_arena_summary_verifies_checkpoint_binding(
    tmp_path: Path, raw_opponent: bool,
) -> None:
    output_dir = tmp_path / "arena"
    output_dir.mkdir()
    state = {
        "schema_version": "chess-dfm-relative-arena-run-v3",
        "status": "complete",
        "contract": {
            "models": {
                "candidate": {
                    "arena_policy_mode": "policy_only",
                    "state": {"sha256": "1" * 64},
                },
                "opponent": {
                    "arena_policy_mode": "dfm",
                    "state": {"sha256": "2" * 64},
                },
            },
            "run": {
                "candidate_refinement_passes": 8,
                "opponent_refinement_passes": 1,
                "opening_start_index": 256,
            },
        },
        "aggregate": {
            "pair_count": 128,
            "pentanomial": {"counts": [1, 2, 3, 4, 5], "score": 0.55},
            "pair_aware_logistic_interval": {"elo": 12.0},
            "cap_draw_rate": 0.0,
            "fault_counts": {},
        },
    }
    if raw_opponent:
        state["contract"]["models"]["opponent"] = {
            "kind": "raw_bt4", "arena_policy_mode": "default",
            "bt4_checkpoint": {"sha256": "2" * 64},
            "plane_history_mode": "lc0_full_history_as_preprocessed",
        }
    opponent_mode = "policy_only" if raw_opponent else "dfm"
    (output_dir / "state.json").write_text(
        json.dumps(state),
        encoding="utf-8",
    )

    summary = _completed_arena_summary(
        output_dir,
        arena_label="hero-arena-test",
        candidate_state_sha256="1" * 64,
        opponent_state_sha256="2" * 64,
        candidate_policy_mode="policy_only",
        opponent_policy_mode=opponent_mode,
        pair_count=128,
        candidate_refinement_passes=8,
        opponent_refinement_passes=1,
        opening_start_index=256,
    )

    assert summary["pair_score"] == 0.55
    assert summary["pair_count"] == 128
    assert summary["opening_start_index"] == 256
    assert summary["candidate_refinement_passes"] == 8
    assert summary["opponent_refinement_passes"] == 1
    assert summary["fault_counts"] == {}
    assert summary["candidate_policy_mode"] == "policy_only"
    assert summary["opponent_policy_mode"] == opponent_mode


def test_raw_bt4_arena_command_has_no_hero_opponent_flags(tmp_path: Path) -> None:
    kwargs = dict(
        candidate_root=tmp_path / "candidate", opponent_root=None,
        models_dir=tmp_path / "models", output_dir=tmp_path / "arena",
        candidate_policy_mode="dfm", opponent_policy_mode="policy_only",
        pair_count=16, candidate_refinement_passes=8,
        opponent_refinement_passes=1, opening_start_index=384, resume=False,
    )
    command = _arena_command(**kwargs)
    assert "--opponent-raw-bt4" in command
    assert "--opponent-torch-hero" not in command
    assert "--opponent-torch-hero-policy-mode" not in command
    with pytest.raises(ValueError, match="Raw BT4 requires"):
        _arena_command(**{**kwargs, "opponent_refinement_passes": 8})
    with pytest.raises(ValueError, match="Raw BT4 requires"):
        _arena_command(**{**kwargs, "opponent_policy_mode": "dfm"})


def test_raw_bt4_arena_identity_is_content_bound() -> None:
    from research.infra.modal_arena_app import RAW_BT4_SHA256

    kwargs = dict(
        candidate_label=_label("a" * 20), candidate_state_sha256="1" * 64,
        opponent_label="raw-bt4", opponent_state_sha256=RAW_BT4_SHA256,
        candidate_policy_mode="dfm", opponent_policy_mode="policy_only",
        pair_count=128, candidate_refinement_passes=1,
        opponent_refinement_passes=1, opening_start_index=384,
        arena_source_tree_sha256="3" * 64,
    )
    original = _arena_identity(**kwargs)
    assert original != _arena_identity(**{**kwargs, "opponent_state_sha256": "4" * 64})
    assert original != _arena_identity(**{**kwargs, "opening_start_index": 512})
