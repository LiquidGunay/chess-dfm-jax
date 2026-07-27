from __future__ import annotations

import copy
import hashlib
import json
import tempfile
from pathlib import Path

import chess
import numpy as np
import pytest

import research.local_policy as local_policy
from chess_dfm_jax.policy import (
    ACTION_CODEC_LEGACY_ABSOLUTE_1858,
    ACTION_VOCAB_SIZE,
    legal_action_mask,
)
from research.arena import build_opening_pool, load_opening_pool, make_color_reversed_pairs
from research.arena_history_trust import (
    HISTORY_VALIDATION_FULL_REPLAY,
    HISTORY_VALIDATION_SCHEMA,
    HISTORY_VALIDATION_TRUSTED_ARENA_ENDPOINT,
)
from research.evaluate_arena import (
    ARENA_RUN_SCHEMA,
    FROZEN_TIERS,
    PolicyTracker,
    TrackingPolicy,
    _resolved_run_options,
    _static_inference_batch_size,
    load_run_state,
    parse_args,
    run_blocks,
    torch_hero_checkpoint_descriptor,
    torch_research_checkpoint_descriptor,
)
from research.local_policy import (
    STATIC_INFERENCE_BATCHING_SCHEMA,
    STATIC_INFERENCE_PADDING_MODE,
    LocalDFMPolicy,
)
from research.play_arena import (
    FAULT_TIMEOUT,
    OPENING_HISTORY_CONVENTION,
    OPENING_HISTORY_SCHEMA,
    histories_for_pairs,
    load_opening_history_sidecar,
    play_arena_pairs,
)
from research.prepare import REPO_ROOT


@pytest.fixture
def workspace_tmp() -> Path:
    root = REPO_ROOT / ".local" / "tmp" / "arena-cli-tests"
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=root) as directory:
        yield Path(directory)


def _canonical_json_bytes(value) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _history(*moves: str) -> list[str]:
    board = chess.Board()
    result = [board.fen(en_passant="legal")]
    for move in moves:
        board.push_uci(move)
        result.append(board.fen(en_passant="legal"))
    return result


def test_arena_cli_accepts_raw_bt4_update_zero_self_match(workspace_tmp: Path):
    args = parse_args(
        [
            "--candidate-raw-bt4",
            "--opponent-raw-bt4",
            "--output-dir",
            str(workspace_tmp / "raw-self-match"),
        ]
    )

    assert args.candidate_raw_bt4 is True
    assert args.candidate is None
    assert args.candidate_torch is None
    assert args.opponent_raw_bt4 is True


def _fixture_assets(root: Path):
    histories = [
        _history("e2e4", "e7e5"),
        _history("d2d4", "d7d5"),
    ]
    val_dir = root / "val"
    val_dir.mkdir()
    shard = val_dir / "chunk_000000.npz"
    np.savez(
        shard,
        schema_version=np.asarray("trajectory-v3"),
        fen_t=np.asarray([values[-1] for values in histories]),
        ply=np.asarray([2, 2], dtype=np.int32),
        source_uri=np.asarray("fixture://arena"),
    )
    pool = build_opening_pool(
        [shard],
        seed=7,
        count=2,
        opening_ply=2,
    )
    by_fen = {values[-1]: values for values in histories}
    sidecar = {
        "schema_version": OPENING_HISTORY_SCHEMA,
        "history_convention": OPENING_HISTORY_CONVENTION,
        "pool_sha256": pool["pool_sha256"],
        "entries": [
            {
                "opening_index": index,
                "fen": opening["fen"],
                "history_fens": by_fen[opening["fen"]],
            }
            for index, opening in enumerate(pool["openings"])
        ],
    }
    sidecar["manifest_sha256"] = hashlib.sha256(_canonical_json_bytes(sidecar)).hexdigest()
    sidecar_path = root / "histories.json"
    sidecar_path.write_text(
        json.dumps(sidecar, sort_keys=True),
        encoding="utf-8",
    )
    loaded = load_opening_history_sidecar(
        sidecar_path,
        opening_pool=pool,
        expected_manifest_sha256=sidecar["manifest_sha256"],
    )
    return pool, sidecar_path, sidecar, loaded


def test_torch_checkpoint_descriptor_is_strict_and_plot_reproducible(
    workspace_tmp: Path,
):
    import dataclasses

    from research.train_torch import CONFIG

    models_dir = workspace_tmp / "models"
    models_dir.mkdir()
    model_path = models_dir / "BT4_exported.pb.gz"
    model_path.write_bytes(b"fixture-bt4")
    run_root = workspace_tmp / "torch-run"
    checkpoint_dir = run_root / "checkpoint"
    checkpoint_dir.mkdir(parents=True)
    state_path = checkpoint_dir / "model.safetensors"
    state_path.write_bytes(b"fixture-state")
    state_sha256 = hashlib.sha256(state_path.read_bytes()).hexdigest()
    manifest = {
        "format": "chess-dfm-torch-model-v1",
        "model_only": True,
        "optimizer_resume_supported": False,
        "optimizer_update": 7,
        "source_mapping_sha256": "a" * 64,
        "state": {
            "path": "model.safetensors",
            "size_bytes": state_path.stat().st_size,
            "sha256": state_sha256,
            "leaf_count": 455,
        },
    }
    (checkpoint_dir / "manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    run_config = {
        "framework": "torch",
        "execution": "eager",
        "torch_compile": False,
        "git_commit": "b" * 40,
        "config": dataclasses.asdict(CONFIG),
        "source": {"step": 265000},
    }
    (run_root / "run_config.json").write_text(
        json.dumps(run_config),
        encoding="utf-8",
    )

    descriptor = torch_research_checkpoint_descriptor(
        run_root,
        models_dir=models_dir,
    )

    assert descriptor.checkpoint_dir == checkpoint_dir
    assert descriptor.manifest == manifest
    assert descriptor.run_config == run_config
    assert descriptor.descriptor["kind"] == "torch_research"
    assert descriptor.descriptor["research_update"] == 7
    assert descriptor.descriptor["state"]["sha256"] == state_sha256
    assert descriptor.descriptor["torch_run_config"]["git_commit"] == "b" * 40


def test_torch_hero_checkpoint_descriptor_pins_canonical_recipe(
    workspace_tmp: Path,
):
    import dataclasses

    from research.train_torch import HERO_CONFIG

    models_dir = workspace_tmp / "models"
    models_dir.mkdir()
    model_path = models_dir / "BT4_exported.pb.gz"
    model_path.write_bytes(b"fixture-bt4")
    run_root = workspace_tmp / "torch-hero-run"
    checkpoint_dir = run_root / "checkpoint"
    checkpoint_dir.mkdir(parents=True)
    state_path = checkpoint_dir / "model.safetensors"
    state_path.write_bytes(b"fixture-hero-state")
    state_sha256 = hashlib.sha256(state_path.read_bytes()).hexdigest()
    manifest = {
        "format": "chess-dfm-torch-model-v1",
        "model_only": True,
        "optimizer_resume_supported": False,
        "optimizer_update": 27_679,
        "source_mapping_sha256": "a" * 64,
        "state": {
            "path": "model.safetensors",
            "size_bytes": state_path.stat().st_size,
            "sha256": state_sha256,
            "leaf_count": 462,
        },
    }
    (checkpoint_dir / "manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    run_config = {
        "framework": "torch",
        "execution": "regional-compile",
        "torch_compile": True,
        "recipe": "hero",
        "git_commit": "c" * 40,
        "compile_regions": [
            "state_projector_blocks",
            "dfm_blocks",
            "jepa_transition",
        ],
        "config": dataclasses.asdict(
            dataclasses.replace(
                HERO_CONFIG,
                sigreg_example_count=256,
                wdl_coeff=0.0,
                jepa_feedback_mode="final_pass_adjoint",
                remat_bt4_blocks=True,
                remat_projector_blocks=True,
                remat_dfm_blocks=False,
                use_bt4_sdpa=True,
                use_head_sdpa=True,
            )
        ),
    }
    (run_root / "run_config.json").write_text(
        json.dumps(run_config),
        encoding="utf-8",
    )

    descriptor = torch_hero_checkpoint_descriptor(
        run_root,
        models_dir=models_dir,
    )

    assert descriptor.checkpoint_dir == checkpoint_dir
    assert descriptor.descriptor["kind"] == "torch_hero"
    assert (
        descriptor.descriptor["action_codec_id"]
        == "lc0_canonical_1858"
    )
    assert descriptor.descriptor["research_update"] == 27_679
    assert descriptor.descriptor["state"]["sha256"] == state_sha256
    assert descriptor.descriptor["checkpoint_storage_kind"] == "model_only"
    assert descriptor.descriptor["model_config"]["sigreg_example_count"] == 256
    assert descriptor.descriptor["model_config"]["wdl_coeff"] == 0.0
    assert (
        descriptor.descriptor["model_config"]["jepa_feedback_mode"]
        == "final_pass_adjoint"
    )


def test_torch_hero_checkpoint_descriptor_accepts_recovery_state(
    workspace_tmp: Path,
):
    import dataclasses

    from research.train_torch import HERO_CONFIG

    models_dir = workspace_tmp / "models"
    models_dir.mkdir()
    model_path = models_dir / "BT4_exported.pb.gz"
    model_path.write_bytes(b"fixture-bt4")
    run_root = workspace_tmp / "torch-hero-run"
    checkpoint_dir = run_root / "checkpoints" / "update00013840"
    checkpoint_dir.mkdir(parents=True)
    state_path = checkpoint_dir / "state.safetensors"
    state_path.write_bytes(b"fixture-hero-recovery-state")
    state_sha256 = hashlib.sha256(state_path.read_bytes()).hexdigest()
    expected_config = dataclasses.asdict(
        dataclasses.replace(
            HERO_CONFIG,
            remat_bt4_blocks=True,
            remat_projector_blocks=True,
            remat_dfm_blocks=False,
            use_bt4_sdpa=True,
            use_head_sdpa=True,
        )
    )
    compile_regions = [
        "state_projector_blocks",
        "dfm_blocks",
        "jepa_transition",
    ]
    git_commit = "c" * 40
    resume_contract = {
        "schema_version": "torch-training-resume-contract-v1",
        "framework": "torch",
        "recipe": "hero",
        "git_commit": git_commit,
        "config": expected_config,
        "runtime": {"compiled_regions": compile_regions},
    }
    manifest = {
        "format": "chess-dfm-torch-training-v1",
        "model_only": False,
        "optimizer_resume_supported": True,
        "optimizer_update": 13_840,
        "source_mapping_sha256": "a" * 64,
        "resume_contract": resume_contract,
        "resume_contract_sha256": hashlib.sha256(
            _canonical_json_bytes(resume_contract)
        ).hexdigest(),
        "state": {
            "path": "state.safetensors",
            "size_bytes": state_path.stat().st_size,
            "sha256": state_sha256,
            "model_leaf_count": 462,
        },
    }
    (checkpoint_dir / "manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    run_config = {
        "framework": "torch",
        "execution": "regional-compile",
        "torch_compile": True,
        "recipe": "hero",
        "git_commit": git_commit,
        "compile_regions": compile_regions,
        "config": expected_config,
    }
    (run_root / "run_config.json").write_text(
        json.dumps(run_config),
        encoding="utf-8",
    )

    descriptor = torch_hero_checkpoint_descriptor(
        checkpoint_dir,
        models_dir=models_dir,
    )

    assert descriptor.checkpoint_dir == checkpoint_dir
    assert descriptor.descriptor["research_update"] == 13_840
    assert descriptor.descriptor["state"]["sha256"] == state_sha256
    assert descriptor.descriptor["state"]["leaf_count"] == 462
    assert (
        descriptor.descriptor["checkpoint_storage_kind"]
        == "training_recovery"
    )


def test_hero_development_tier_resolves_large_cap256_run(
    workspace_tmp: Path,
):
    args = parse_args(
        [
            "--candidate-torch-hero",
            str(workspace_tmp / "hero"),
            "--opponent-raw-bt4",
            "--tier",
            "hero_development",
            "--pair-count",
            "1024",
            "--block-pairs",
            "16",
            "--output-dir",
            str(workspace_tmp / "arena"),
        ]
    )

    assert args.candidate_torch_hero == workspace_tmp / "hero"
    assert _resolved_run_options(
        args,
        FROZEN_TIERS["hero_development"],
    ) == (1024, 16, 256)


class _Selection:
    def __init__(self, action_indices):
        self.action_indices = np.asarray(action_indices, dtype=np.int32)


class _LeanModel:
    class Config:
        horizon = 4
        action_vocab_size = ACTION_VOCAB_SIZE

    config = Config()


class _FirstLegalPolicy:
    action_codec_id = ACTION_CODEC_LEGACY_ABSOLUTE_1858
    inference_batching_schema = STATIC_INFERENCE_BATCHING_SCHEMA
    inference_padding_mode = STATIC_INFERENCE_PADDING_MODE
    inference_batch_size = 1
    history_validation_mode = HISTORY_VALIDATION_TRUSTED_ARENA_ENDPOINT
    trusted_arena_history_schema = HISTORY_VALIDATION_SCHEMA

    def __init__(self, model_id: str):
        self.model_id = model_id

    def select_actions(self, boards, histories):
        del histories
        return _Selection(
            [
                int(
                    np.flatnonzero(
                        legal_action_mask(
                            board,
                            codec_id=self.action_codec_id,
                        )
                    )[0]
                )
                for board in boards
            ]
        )

    def select_actions_from_trusted_arena(self, boards, endpoints):
        del endpoints
        return self.select_actions(boards, ())


class _StaticPhysicalRecordingPolicy(_FirstLegalPolicy):
    inference_batch_size = 4

    def __init__(self, model_id: str):
        super().__init__(model_id)
        self.physical_batch_sizes: list[int] = []

    def select_actions(self, boards, histories):
        self.physical_batch_sizes.append(self.inference_batch_size)
        return super().select_actions(boards, histories)


def _contract(*, promotion: bool = False):
    return {
        "relative_elo_scope": "checkpoint_pool_relative_only",
        "absolute_elo_claim": False,
        "tier": {
            "name": "promotion" if promotion else "correctness",
            "promotion_eligible": promotion,
        },
        "models": {
            "candidate": {"model_id": "candidate"},
            "opponent": {"model_id": "opponent"},
        },
        "run": {
            "pair_count": 1 if promotion else 2,
            "block_pairs": 1,
            "additional_ply_cap": 1,
            "policy_timeout_seconds": 30.0,
            "policy_batch_size_cap": 2,
        },
        "inference_batching": {
            "schema_version": STATIC_INFERENCE_BATCHING_SCHEMA,
            "physical_batch_size": 1,
            "padding_mode": STATIC_INFERENCE_PADDING_MODE,
            "padding_source": "first_validated_active_encoded_row",
            "active_rows": "ordered_prefix",
            "output_handling": "slice_to_active_rows_before_semantic_validation",
            "host_validation_scope": "active_rows_only",
            "metrics_basis": "real_rows_only",
            "timing_basis": "physical_padded_call_wall_time",
            "runtime_rng": "none_deterministic_greedy_inference",
        },
        "history_validation": {
            "schema_version": HISTORY_VALIDATION_SCHEMA,
            "public_policy_default": HISTORY_VALIDATION_FULL_REPLAY,
            "arena_hot_path": HISTORY_VALIDATION_TRUSTED_ARENA_ENDPOINT,
            "opening_validation": "full_exact_replay_sidecar_and_selected_pairs",
            "authoritative_state": "full_stack_with_checked_legal_pushes",
            "policy_board_copy": "stack_false",
            "policy_payload": "sealed_constant_size_state_attestation",
            "repetition_authority": "runner_claim_draw_boundary_check",
            "complexity": "constant_per_policy_row",
        },
    }


@pytest.mark.parametrize(
    ("block_pairs", "policy_batch_size_cap", "expected"),
    [(16, 64, 16), (128, 32, 32), (1, 64, 1)],
)
def test_static_physical_shape_uses_smaller_block_or_chunk_bound(
    block_pairs: int,
    policy_batch_size_cap: int,
    expected: int,
):
    assert (
        _static_inference_batch_size(
            block_pairs=block_pairs,
            policy_batch_size_cap=policy_batch_size_cap,
        )
        == expected
    )


def test_history_sidecar_is_digest_checked_aligned_and_replayed(
    workspace_tmp: Path,
):
    pool, sidecar_path, sidecar, loaded = _fixture_assets(workspace_tmp)
    assert loaded.pool_sha256 == pool["pool_sha256"]
    assert loaded.manifest_sha256 == sidecar["manifest_sha256"]
    assert len(loaded.histories_by_opening_index) == 2

    from research.arena import make_color_reversed_pairs

    second = make_color_reversed_pairs(
        [pool["openings"][1]["fen"]],
        model_a="candidate",
        model_b="opponent",
        start_index=1,
    )
    bound = histories_for_pairs(second, loaded)
    assert tuple(bound.values())[0].fens[-1] == pool["openings"][1]["fen"]

    tampered = json.loads(sidecar_path.read_text(encoding="utf-8"))
    tampered["entries"][0]["history_fens"][1] = tampered["entries"][1]["history_fens"][1]
    sidecar_path.write_text(
        json.dumps(tampered, sort_keys=True),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="manifest digest mismatch"):
        load_opening_history_sidecar(
            sidecar_path,
            opening_pool=pool,
            expected_manifest_sha256=sidecar["manifest_sha256"],
        )


def test_resumable_blocks_persist_relative_stats_and_verify_immutable_files(
    workspace_tmp: Path,
):
    pool, _path, _sidecar, loaded = _fixture_assets(workspace_tmp)
    output_dir = workspace_tmp / "run"
    contract = _contract()
    candidate = _FirstLegalPolicy("candidate")
    opponent = _FirstLegalPolicy("opponent")

    state = run_blocks(
        output_dir=output_dir,
        contract=contract,
        opening_pool=pool,
        loaded_histories=loaded,
        candidate_policy=candidate,
        opponent_policy=opponent,
        resume=False,
        session_setup={"kind": "fixture"},
    )

    assert state["schema_version"] == ARENA_RUN_SCHEMA
    assert state["status"] == "complete"
    assert state["absolute_elo_claim"] is False
    assert state["candidate_promoted"] is False
    assert state["aggregate"]["pair_count"] == 2
    assert state["aggregate"]["game_count"] == 4
    assert state["aggregate"]["cap_draw_rate"] == 1.0
    assert len(state["blocks"]) == 2
    assert state["normalized_promotion_state"] is None
    assert (
        load_run_state(
            output_dir,
            expected_contract=contract,
        )
        == state
    )

    resumed = run_blocks(
        output_dir=output_dir,
        contract=contract,
        opening_pool=pool,
        loaded_histories=loaded,
        candidate_policy=candidate,
        opponent_policy=opponent,
        resume=True,
        session_setup={"kind": "unused"},
    )
    assert resumed == state

    changed_batching = copy.deepcopy(contract)
    changed_batching["inference_batching"]["physical_batch_size"] = 2
    with pytest.raises(ValueError, match="resume contract mismatch"):
        load_run_state(
            output_dir,
            expected_contract=changed_batching,
        )

    changed_version = copy.deepcopy(contract)
    changed_version["inference_batching"]["schema_version"] = (
        "chess-dfm-static-inference-batching-v0"
    )
    with pytest.raises(ValueError, match="resume contract mismatch"):
        load_run_state(
            output_dir,
            expected_contract=changed_version,
        )

    changed_history_mode = copy.deepcopy(contract)
    changed_history_mode["history_validation"]["arena_hot_path"] = HISTORY_VALIDATION_FULL_REPLAY
    with pytest.raises(ValueError, match="resume contract mismatch"):
        load_run_state(
            output_dir,
            expected_contract=changed_history_mode,
        )

    changed_history_version = copy.deepcopy(contract)
    changed_history_version["history_validation"]["schema_version"] = (
        "chess-dfm-arena-history-validation-v0"
    )
    with pytest.raises(ValueError, match="resume contract mismatch"):
        load_run_state(
            output_dir,
            expected_contract=changed_history_version,
        )

    block_path = Path(state["blocks"][0]["path"])
    block_path.write_text(
        block_path.read_text(encoding="utf-8") + " ",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="checksum mismatch"):
        load_run_state(
            output_dir,
            expected_contract=contract,
        )


def test_tracking_counts_only_active_rows_while_physical_shape_stays_frozen():
    delegate = _StaticPhysicalRecordingPolicy("candidate")
    tracker = PolicyTracker()
    policy = TrackingPolicy(delegate=delegate, tracker=tracker)
    board = chess.Board()

    policy.select_actions([board, board, board], [(), (), ()])
    policy.select_actions([board], [()])

    assert delegate.physical_batch_sizes == [4, 4]
    assert tracker.positions == 4
    assert tracker.legal_moves == 80
    assert tracker.representable_legal_actions == 80
    assert tracker.incomplete_coverage_positions == 0
    assert tracker.calls == 2
    assert tracker.call_seconds >= 0.0


def test_static_padded_timeout_faults_only_the_real_active_game(
    workspace_tmp: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    pool, _path, _sidecar, loaded = _fixture_assets(workspace_tmp)
    pairs = make_color_reversed_pairs(
        [pool["openings"][0]["fen"]],
        model_a="candidate",
        model_b="opponent",
    )
    histories = histories_for_pairs(pairs, loaded)

    def time_out(*args, **kwargs):
        raise TimeoutError("synthetic padded device timeout")

    monkeypatch.setattr(local_policy, "infer_dfm_actions_from_current", time_out)
    tracker = PolicyTracker()
    candidate = TrackingPolicy(
        delegate=LocalDFMPolicy(
            model=_LeanModel(),
            model_id="candidate",
            collect_diagnostics=False,
            inference_batch_size=4,
        ),
        tracker=tracker,
    )
    result = play_arena_pairs(
        pairs,
        opening_histories=histories,
        policies={
            "candidate": candidate,
            "opponent": _FirstLegalPolicy("opponent"),
        },
        additional_ply_cap=1,
        policy_batch_size_cap=4,
    )

    timeout_records = [record for record in result.records if record.fault_kind == FAULT_TIMEOUT]
    assert len(result.records) == 2
    assert len(timeout_records) == 1
    assert timeout_records[0].fault_model == "candidate"
    candidate_stats = {stats.model_id: stats for stats in result.model_stats}["candidate"]
    assert candidate_stats.policy_calls == 1
    assert candidate_stats.positions_evaluated == 1
    assert tracker.calls == 1
    assert tracker.positions == 1


def test_promotion_state_advances_at_one_complete_pair_boundary(
    workspace_tmp: Path,
):
    pool, _path, _sidecar, loaded = _fixture_assets(workspace_tmp)
    contract = _contract(promotion=True)
    state = run_blocks(
        output_dir=workspace_tmp / "promotion-run",
        contract=contract,
        opening_pool=pool,
        loaded_histories=loaded,
        candidate_policy=_FirstLegalPolicy("candidate"),
        opponent_policy=_FirstLegalPolicy("opponent"),
        resume=False,
        session_setup={"kind": "fixture"},
    )

    promotion = state["normalized_promotion_state"]
    assert promotion["pair_count"] == 1
    assert promotion["decision"] == "continue"
    assert promotion["update_unit"] == "completed_color_reversed_pair"
    assert state["candidate_promoted"] is False


def test_pinned_promotion_tier_resolves_and_loads_verified_v3_assets():
    args = type(
        "Args",
        (),
        {
            "pair_count": None,
            "block_pairs": None,
            "additional_ply_cap": 256,
        },
    )()
    tier = FROZEN_TIERS["promotion"]
    assert _resolved_run_options(args, tier) == (2048, 1, 256)

    pool = load_opening_pool(
        tier.pool_path,
        expected_pool_sha256=tier.pool_sha256,
    )
    loaded = load_opening_history_sidecar(
        tier.histories_path,
        opening_pool=pool,
        expected_manifest_sha256=tier.history_manifest_sha256,
    )
    assert len(pool["openings"]) == 2048
    assert len(loaded.histories_by_opening_index) == 2048
