from __future__ import annotations

import dataclasses
import importlib.util
import io
import json
import sys
import tarfile
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

from research.infra.modal_train_app import (
    COMPILE_SOURCE_TREE_PATHS,
    HERO1_EXACT_PHASE1_RESULT_LABEL,
    HERO1_EXACT_PHASE1_STATE_SHA256,
    MOVEMENT_SOURCE_TREE_PATHS,
    SOURCE_TREE_PATHS,
    TRAINING_ARMS,
    TRAINING_PROFILES,
    _archive_partial_result,
    _extract_streaming_tar,
    _compile_source_tree_sha256,
    _latest_recovery_checkpoint,
    _movement_source_tree_sha256,
    _parameter_pair_movement,
    _policy_distill_teacher_checkpoint_dir,
    _recovery_checkpoint_candidate,
    _require_result_label,
    _run_identity,
    _safe_tar_relative,
    _source_tree_sha256,
    _retarget_hero_eval_manifest,
    _spec,
    _training_command,
)
from research.interpretability.remote_cost import estimate_modal_job


def test_phase1_matrix_contains_exact_hero1_and_separated_v2_controls() -> None:
    assert {
        "hero1_exact",
        "v2_bf16_1_30",
        "v2_fp32_1_30",
        "v2_fp32_1_12",
        "v2_fp32_1_10",
        "v2_fp32_1_3",
        "v2_encoder_fp32_1_30",
        "v2_encoder_fp32_1_12",
        "v2_trunk_fp32_1_12",
        "v2_main_fp32_1_30",
        "v2_fp32_1_12_legality0",
        "v2_fp32_1_12_legality0_distill1",
        "v2_fp32_1_12_legality0_hero1distill1",
        "v2_fp32_1_12_legality0_hero1distill1_closedloop",
    } <= set(TRAINING_ARMS)
    assert TRAINING_ARMS["hero1_exact"].recipe == "hero"
    assert TRAINING_ARMS["v2_bf16_1_30"].optimizer_precision == "parameter"
    assert TRAINING_ARMS["v2_fp32_1_30"].optimizer_precision == "fp32_master"
    assert (
        TRAINING_ARMS["v2_encoder_fp32_1_12"].optimizer_precision
        == "encoder_fp32_master"
    )
    assert (
        TRAINING_ARMS["v2_trunk_fp32_1_12"].optimizer_precision
        == "encoder_trunk_fp32_master"
    )
    assert (
        TRAINING_ARMS["v2_main_fp32_1_30"].optimizer_precision
        == "main_fp32_master"
    )
    assert TRAINING_ARMS["v2_fp32_1_12_legality0"].legality_coeff == 0.0
    assert TRAINING_ARMS["v2_fp32_1_12_legality0_distill1"].legality_coeff == 0.0
    assert (
        TRAINING_ARMS["v2_fp32_1_12_legality0_distill1"].policy_distill_coeff
        == 1.0
    )
    fixed_teacher = TRAINING_ARMS[
        "v2_fp32_1_12_legality0_hero1distill1"
    ]
    assert fixed_teacher.legality_coeff == 0.0
    assert fixed_teacher.policy_distill_coeff == 1.0
    assert (
        fixed_teacher.policy_distill_teacher_result_label
        == HERO1_EXACT_PHASE1_RESULT_LABEL
    )
    assert (
        fixed_teacher.policy_distill_teacher_state_sha256
        == HERO1_EXACT_PHASE1_STATE_SHA256
    )
    full_loop = TRAINING_ARMS[
        "v2_fp32_1_12_legality0_hero1distill1_closedloop"
    ]
    assert full_loop == dataclasses.replace(
        fixed_teacher,
        dfm_closed_loop_mode="predicted_jepa_tokens",
    )
    assert TRAINING_ARMS["v2_fp32_1_12"].encoder_lr_ratio == pytest.approx(
        1.0 / 12.0
    )
    assert TRAINING_ARMS["v2_fp32_1_10"].encoder_lr_ratio == pytest.approx(0.1)
    assert TRAINING_ARMS["v2_fp32_1_3"].encoder_lr_ratio == pytest.approx(1.0 / 3.0)
    assert "research/stitch_training_segments.py" in SOURCE_TREE_PATHS
    assert "research/stitch_training_segments.py" not in COMPILE_SOURCE_TREE_PATHS

    phase1 = TRAINING_PROFILES["phase1"]
    assert phase1.steps == 554
    assert phase1.validation_updates == (554,)
    assert phase1.save_recovery is True
    assert phase1.save_model is True
    assert phase1.recovery_updates == (100, 200, 300, 400, 500, 554)

    u1024 = TRAINING_PROFILES["u1024"]
    assert u1024.steps == 1_024
    assert u1024.validation_updates == (554, 1_024)
    assert u1024.save_recovery is True
    assert u1024.save_model is True
    assert u1024.resume_from_profile is None
    assert u1024.recovery_updates == (200, 400, 554, 700, 850, 1_024)

    diagnostic = TRAINING_PROFILES["diagnostic"]
    assert diagnostic.steps == 10
    assert diagnostic.log_every == 1
    assert diagnostic.save_recovery is False
    assert diagnostic.save_model is False

    phase2 = TRAINING_PROFILES["phase2"]
    assert phase2.steps == 2_768
    assert phase2.validation_updates == (1_384, 2_768)
    assert phase2.resume_from_profile == "phase1"
    assert phase2.save_recovery is True
    assert phase2.save_model is True
    assert phase2.recovery_updates[-1] == phase2.steps

def test_training_command_preserves_hero1_and_explicitly_serializes_v2(tmp_path: Path) -> None:
    common = {
        "profile_name": "phase1",
        "output_dir": tmp_path / "output",
        "raw_bt4_path": tmp_path / "raw.pb.gz",
        "data_root": tmp_path / "trajectory_v3",
        "eval_manifest": tmp_path / "eval.json",
    }
    hero1 = _training_command(arm_name="hero1_exact", **common)
    assert hero1[:4] == [sys.executable, "-m", "research.train_torch", "train"]
    assert hero1[hero1.index("--recipe") + 1] == "hero"
    assert "--main-lr-multiplier" not in hero1
    assert hero1[hero1.index("--steps") + 1] == "554"
    save_start = hero1.index("--save-updates") + 1
    save_end = hero1.index("--save-final")
    assert hero1[save_start:save_end] == ["100", "200", "300", "400", "500", "554"]
    assert "--save-final" in hero1
    assert hero1[hero1.index("--max-checkpoints") + 1] == "7"

    hero2 = _training_command(arm_name="v2_fp32_1_10", **common)
    assert hero2[hero2.index("--recipe") + 1] == "hero_v2"
    assert hero2[hero2.index("--main-lr-multiplier") + 1] == "1.0"
    assert float(hero2[hero2.index("--encoder-lr-ratio") + 1]) == pytest.approx(0.1)
    assert hero2[hero2.index("--optimizer-precision") + 1] == "fp32_master"
    assert hero2[hero2.index("--lr-schedule-kind") + 1] == (
        "warmup_stable_linear_decay"
    )

    hybrid = _training_command(arm_name="v2_encoder_fp32_1_12", **common)
    assert hybrid[hybrid.index("--optimizer-precision") + 1] == (
        "encoder_fp32_master"
    )
    assert float(hybrid[hybrid.index("--encoder-lr-ratio") + 1]) == pytest.approx(1 / 12)

    trunk_only = _training_command(arm_name="v2_trunk_fp32_1_12", **common)
    assert trunk_only[trunk_only.index("--optimizer-precision") + 1] == (
        "encoder_trunk_fp32_master"
    )
    assert float(trunk_only[trunk_only.index("--encoder-lr-ratio") + 1]) == pytest.approx(1 / 12)

    main_only = _training_command(arm_name="v2_main_fp32_1_30", **common)
    assert main_only[main_only.index("--optimizer-precision") + 1] == (
        "main_fp32_master"
    )
    assert float(main_only[main_only.index("--encoder-lr-ratio") + 1]) == (
        pytest.approx(1 / 30)
    )

    no_legality = _training_command(
        arm_name="v2_fp32_1_12_legality0",
        **common,
    )
    assert no_legality[no_legality.index("--optimizer-precision") + 1] == (
        "fp32_master"
    )
    assert float(no_legality[no_legality.index("--encoder-lr-ratio") + 1]) == (
        pytest.approx(1 / 12)
    )
    assert no_legality[no_legality.index("--legality-coeff") + 1] == "0.0"

    distilled = _training_command(
        arm_name="v2_fp32_1_12_legality0_distill1",
        **common,
    )
    assert distilled[distilled.index("--legality-coeff") + 1] == "0.0"
    assert distilled[distilled.index("--policy-distill-coeff") + 1] == "1.0"
    assert distilled[distilled.index("--policy-distill-teacher") + 1] == "online"

    teacher_checkpoint = tmp_path / "teacher" / "checkpoint"
    fixed_distilled = _training_command(
        arm_name="v2_fp32_1_12_legality0_hero1distill1",
        policy_distill_teacher_checkpoint_dir=teacher_checkpoint,
        **common,
    )
    assert fixed_distilled[
        fixed_distilled.index("--policy-distill-teacher") + 1
    ] == "checkpoint"
    assert fixed_distilled[
        fixed_distilled.index("--policy-distill-teacher-checkpoint-dir") + 1
    ] == str(teacher_checkpoint)
    assert fixed_distilled[
        fixed_distilled.index("--policy-distill-teacher-state-sha256") + 1
    ] == HERO1_EXACT_PHASE1_STATE_SHA256
    assert "--dfm-closed-loop-mode" not in fixed_distilled

    u1024_common = {**common, "profile_name": "u1024"}
    full_loop = _training_command(
        arm_name="v2_fp32_1_12_legality0_hero1distill1_closedloop",
        policy_distill_teacher_checkpoint_dir=teacher_checkpoint,
        **u1024_common,
    )
    assert full_loop[full_loop.index("--steps") + 1] == "1024"
    assert full_loop[full_loop.index("--dfm-closed-loop-mode") + 1] == (
        "predicted_jepa_tokens"
    )
    full_loop_save_start = full_loop.index("--save-updates") + 1
    full_loop_save_end = full_loop.index("--save-final")
    assert full_loop[full_loop_save_start:full_loop_save_end] == [
        str(update) for update in TRAINING_PROFILES["u1024"].recovery_updates
    ]
    with pytest.raises(RuntimeError, match="no checkpoint path"):
        _training_command(
            arm_name="v2_fp32_1_12_legality0_hero1distill1",
            **common,
        )
    with pytest.raises(RuntimeError, match="Online policy teacher"):
        _training_command(
            arm_name="v2_fp32_1_12_legality0_distill1",
            policy_distill_teacher_checkpoint_dir=teacher_checkpoint,
            **common,
        )


    resume_checkpoint = (
        tmp_path / "runs" / "phase1" / "checkpoints" / "update00000554"
    )
    continuation = _training_command(
        profile_name="phase2",
        arm_name="v2_fp32_1_10",
        output_dir=tmp_path / "phase2-output",
        raw_bt4_path=tmp_path / "raw.pb.gz",
        data_root=tmp_path / "trajectory_v3",
        eval_manifest=tmp_path / "eval.json",
        resume_checkpoint=resume_checkpoint,
        resume_update=554,
    )
    assert continuation[continuation.index("--resume-checkpoint") + 1] == str(
        resume_checkpoint
    )
    assert continuation[continuation.index("--steps") + 1] == "2768"
    continuation_save_start = continuation.index("--save-updates") + 1
    continuation_save_end = continuation.index("--save-final")
    assert continuation[continuation_save_start:continuation_save_end] == [
        str(update) for update in TRAINING_PROFILES["phase2"].recovery_updates
    ]


@pytest.mark.parametrize(
    "stage",
    (
        "stage-data",
        "movement",
        "smoke-l40s",
        "smoke-a100",
        "phase1-l40s",
        "phase1-a100",
        "u1024-l40s",
        "u1024-a100",
        "phase2-l40s",
        "phase2-a100",
    ),
)
def test_training_modal_specs_fit_their_fail_closed_caps(stage: str) -> None:
    spec, cap = _spec(stage)
    estimate = estimate_modal_job(spec)
    assert Decimal(estimate["declared_attempt_upper_bound_dollars"]) <= cap
    assert spec.retries == 0
    if stage in {"stage-data", "movement"}:
        assert spec.gpu is None
    else:
        assert spec.gpu in {"L40S", "A100-40GB"}


def test_movement_audit_accepts_only_canonical_training_result_labels() -> None:
    label = "hero-training-phase1-v2_fp32_1_10-l40s-0123456789abcdef0123"
    assert _require_result_label(label) == label
    for unsafe in ("", "../escape", "Hero-training-upper", "other-result"):
        with pytest.raises(ValueError, match="Unsafe"):
            _require_result_label(unsafe)


def test_parameter_pair_movement_is_exact_and_shape_safe() -> None:
    torch = pytest.importorskip("torch")
    initial = torch.tensor([1.0, 0.0], dtype=torch.float32)
    trained = torch.tensor([1.0, 2.0], dtype=torch.float32)

    metrics = _parameter_pair_movement(initial, trained)

    assert metrics["parameter_count"] == 2
    assert metrics["initial_l2"] == pytest.approx(1.0)
    assert metrics["trained_l2"] == pytest.approx(5.0**0.5)
    assert metrics["delta_l2"] == pytest.approx(2.0)
    assert metrics["relative_delta_l2"] == pytest.approx(2.0)
    assert metrics["cosine"] == pytest.approx(1.0 / (5.0**0.5))
    assert metrics["unchanged_count"] == 1
    assert metrics["changed_fraction"] == pytest.approx(0.5)

    with pytest.raises(ValueError, match="ABI mismatch"):
        _parameter_pair_movement(initial, trained.unsqueeze(0))


def test_movement_identity_binds_the_audit_wrapper() -> None:
    assert "research/infra/modal_train_app.py" in MOVEMENT_SOURCE_TREE_PATHS


def test_fixed_policy_teacher_accepts_only_the_sealed_phase1_result(
    tmp_path: Path,
) -> None:
    result_dir = tmp_path / "runs" / HERO1_EXACT_PHASE1_RESULT_LABEL
    checkpoint_dir = result_dir / "checkpoint"
    checkpoint_dir.mkdir(parents=True)
    state = b"sealed-test-state"
    (checkpoint_dir / "state.safetensors").write_bytes(state)
    (result_dir / "modal_run.json").write_text(
        json.dumps(
            {
                "result_label": HERO1_EXACT_PHASE1_RESULT_LABEL,
                "profile": "phase1",
                "arm": "hero1_exact",
                "gpu": "L40S",
            }
        ),
        encoding="utf-8",
    )
    (result_dir / "report.json").write_text(
        json.dumps({"updates": TRAINING_PROFILES["phase1"].steps}),
        encoding="utf-8",
    )
    (result_dir / "run_config.json").write_text(
        json.dumps({"framework": "torch", "config": {}}),
        encoding="utf-8",
    )
    manifest = {
        "format": "chess-dfm-torch-model-v1",
        "model_only": True,
        "optimizer_update": TRAINING_PROFILES["phase1"].steps,
        "state": {
            "path": "state.safetensors",
            "size_bytes": len(state),
            "sha256": HERO1_EXACT_PHASE1_STATE_SHA256,
        },
    }
    manifest_path = checkpoint_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    assert _policy_distill_teacher_checkpoint_dir(
        arm_name="v2_fp32_1_12_legality0_hero1distill1",
        result_mount=tmp_path,
    ) == checkpoint_dir
    assert _policy_distill_teacher_checkpoint_dir(
        arm_name="v2_fp32_1_12_legality0_hero1distill1_closedloop",
        result_mount=tmp_path,
    ) == checkpoint_dir

    manifest["state"]["sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(RuntimeError, match="contract drift"):
        _policy_distill_teacher_checkpoint_dir(
            arm_name="v2_fp32_1_12_legality0_hero1distill1",
            result_mount=tmp_path,
        )


def _write_test_recovery_checkpoint(root: Path, update: int) -> Path:
    checkpoint_dir = root / "checkpoints" / f"update{update:08d}"
    checkpoint_dir.mkdir(parents=True)
    state = b"checkpoint-state"
    (checkpoint_dir / "state.safetensors").write_bytes(state)
    manifest = {
        "format": "chess-dfm-torch-training-v2",
        "model_only": False,
        "optimizer_resume_supported": True,
        "optimizer_update": update,
        "state": {
            "path": "state.safetensors",
            "size_bytes": len(state),
        },
    }
    (checkpoint_dir / "manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    return checkpoint_dir


def test_partial_result_archives_and_selects_latest_nonterminal_recovery(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "runs" / "immutable-label"
    update100 = _write_test_recovery_checkpoint(output_dir, 100)
    _write_test_recovery_checkpoint(output_dir, 554)

    attempt_root, archive_dir = _archive_partial_result(output_dir)

    assert not output_dir.exists()
    assert archive_dir == attempt_root / "attempt000"
    assert _recovery_checkpoint_candidate(
        archive_dir / update100.relative_to(output_dir)
    ) == (100, archive_dir / "checkpoints" / "update00000100")
    assert _latest_recovery_checkpoint((archive_dir,)) == (
        554,
        archive_dir / "checkpoints" / "update00000554",
    )
    assert _latest_recovery_checkpoint(
        (archive_dir,),
        before_update=554,
    ) == (100, archive_dir / "checkpoints" / "update00000100")


def test_safe_tar_paths_reject_absolute_and_parent_traversal() -> None:
    assert _safe_tar_relative("./train/chunk_1.npz") == Path("train/chunk_1.npz")
    assert _safe_tar_relative("./") is None
    with pytest.raises(ValueError, match="Absolute"):
        _safe_tar_relative("/escape")
    with pytest.raises(ValueError, match="Traversing"):
        _safe_tar_relative("train/../../escape")


def test_eval_manifest_paths_target_final_atomic_dataset_root() -> None:
    final_root = Path(
        "/workspace/training-inputs/trajectory-v3-lc0-test80-h8-sets1-3-20260430"
    )
    template = {
        "dataset": {
            "manifest": "stale",
            "train": {"path": "stale"},
            "val": {"path": "stale"},
            "test": {"path": "stale"},
        },
        "position_indices": {"path": "stale"},
        "paired_arena": {
            "opening_pool": {"path": "stale"},
            "opening_histories": {"path": "stale"},
            "opening_contract": {"path": "stale"},
        },
    }
    observed = _retarget_hero_eval_manifest(template, dataset_root=final_root)
    serialized = str(observed)
    assert ".partial" not in serialized
    assert observed["dataset"]["manifest"] == str(
        final_root / "trajectory_v3" / "manifest.json"
    )
    assert observed["position_indices"]["path"] == str(
        final_root / "contracts" / "hero_epoch_v1" / "position_indices.npz"
    )
    assert observed["paired_arena"]["opening_pool"]["path"] == str(
        final_root
        / "contracts"
        / "hero_epoch_v1"
        / "arena"
        / "promotion-ply12-n2048-v3.json"
    )


def _tar_bytes(*, symlink: bool = False) -> bytes:
    payload = io.BytesIO()
    with tarfile.open(fileobj=payload, mode="w") as archive:
        first = tarfile.TarInfo("./manifest.json")
        data = b'{"schema":"test"}\n'
        first.size = len(data)
        archive.addfile(first, io.BytesIO(data))
        second = tarfile.TarInfo("./train/chunk_000000.npz")
        if symlink:
            second.type = tarfile.SYMTYPE
            second.linkname = "../../escape"
            archive.addfile(second)
        else:
            chunk = b"0123456789"
            second.size = len(chunk)
            archive.addfile(second, io.BytesIO(chunk))
    return payload.getvalue()


def test_streaming_extractor_hashes_full_tar_and_writes_only_regular_files(
    tmp_path: Path,
) -> None:
    payload = _tar_bytes()
    result = _extract_streaming_tar(io.BytesIO(payload), tmp_path)
    assert result["archive_size_bytes"] == len(payload)
    assert result["file_count"] == 2
    assert result["file_bytes"] == len(b'{"schema":"test"}\n') + 10
    assert (tmp_path / "manifest.json").read_bytes() == b'{"schema":"test"}\n'
    assert (tmp_path / "train" / "chunk_000000.npz").read_bytes() == b"0123456789"

    with pytest.raises(ValueError, match="Non-regular"):
        _extract_streaming_tar(io.BytesIO(_tar_bytes(symlink=True)), tmp_path / "bad")


def test_source_tree_and_run_identity_are_content_and_gpu_specific(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for relative in COMPILE_SOURCE_TREE_PATHS:
        assert relative in SOURCE_TREE_PATHS

    for relative in sorted(set(SOURCE_TREE_PATHS) | set(MOVEMENT_SOURCE_TREE_PATHS)):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative, encoding="utf-8")
    package = tmp_path / "chess_dfm_jax"
    package.mkdir(exist_ok=True)
    (package / "encoding.py").write_text("VALUE = 1\n", encoding="utf-8")
    (package / "policy_moves.txt").write_text("move\n", encoding="utf-8")
    (package / "policy_attn_map.txt").write_text("map\n", encoding="utf-8")

    first = _source_tree_sha256(tmp_path)
    compile_first = _compile_source_tree_sha256(tmp_path)
    movement_first = _movement_source_tree_sha256(tmp_path)
    assert first == _source_tree_sha256(tmp_path)
    assert movement_first == _movement_source_tree_sha256(tmp_path)
    l40s = _run_identity(
        profile_name="smoke",
        arm_name="v2_fp32_1_10",
        gpu="L40S",
        source_tree_sha256=first,
    )
    a100 = _run_identity(
        profile_name="smoke",
        arm_name="v2_fp32_1_10",
        gpu="A100",
        source_tree_sha256=first,
    )
    assert l40s != a100
    fixed_arm_name = "v2_fp32_1_12_legality0_hero1distill1"
    fixed_identity = _run_identity(
        profile_name="smoke",
        arm_name=fixed_arm_name,
        gpu="L40S",
        source_tree_sha256=first,
    )
    monkeypatch.setitem(
        TRAINING_ARMS,
        fixed_arm_name,
        dataclasses.replace(
            TRAINING_ARMS[fixed_arm_name],
            policy_distill_teacher_state_sha256="b" * 64,
        ),
    )
    assert fixed_identity != _run_identity(
        profile_name="smoke",
        arm_name=fixed_arm_name,
        gpu="L40S",
        source_tree_sha256=first,
    )
    runner = tmp_path / "research/infra/modal_train_app.py"
    runner.write_text("runner revision 2\n", encoding="utf-8")
    assert first != _source_tree_sha256(tmp_path)
    assert compile_first == _compile_source_tree_sha256(tmp_path)
    assert movement_first != _movement_source_tree_sha256(tmp_path)

    parameter_diff = tmp_path / "research/interpretability/parameter_diff.py"
    parameter_diff.write_text("movement revision 2\n", encoding="utf-8")
    assert movement_first != _movement_source_tree_sha256(tmp_path)
    assert compile_first == _compile_source_tree_sha256(tmp_path)

    (package / "encoding.py").write_text("VALUE = 2\n", encoding="utf-8")
    assert first != _source_tree_sha256(tmp_path)
    assert compile_first != _compile_source_tree_sha256(tmp_path)
    assert movement_first != _movement_source_tree_sha256(tmp_path)


def test_training_runner_stitches_verified_resume_metrics_before_commit() -> None:
    source = Path("research/infra/modal_train_app.py").read_text(
        encoding="utf-8"
    )

    stitch_call = source.index("stitch_training_segments(")
    cache_commit = source.index(
        "training_cache_volume.commit()",
        stitch_call,
    )
    assert stitch_call < cache_commit
    assert 'terminal_segment="terminal"' in source[stitch_call:cache_commit]


def test_long_running_launchers_submit_asynchronous_function_calls() -> None:
    source = Path("research/infra/modal_train_app.py").read_text(
        encoding="utf-8"
    )

    stage_dispatch = source.index('if mode == "stage-data":')
    movement_dispatch = source.index('if mode == "movement":')
    training_dispatch = source.index("function = functions[(stage_profile, gpu)]")
    assert "call = stage_trajectory_dataset.spawn()" in source[
        stage_dispatch:movement_dispatch
    ]
    assert "call = audit_training_movement.spawn(" in source[
        movement_dispatch:training_dispatch
    ]
    assert '"stage_submission"' in source[stage_dispatch:movement_dispatch]
    assert '"movement_submission"' in source[
        movement_dispatch:training_dispatch
    ]
    assert "call = function.spawn(" in source[training_dispatch:]
    assert '"training_submission"' in source[training_dispatch:]
    assert '"mode": "asynchronous_spawn"' in source[training_dispatch:]
