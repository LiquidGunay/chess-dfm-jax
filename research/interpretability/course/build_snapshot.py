"""Build the compact, content-bound Raw/Hero teaching snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


SCHEMA_VERSION = "chess-interpretability-course-snapshot-v1"
SOURCE_DATA_COMMIT = "8cf451f9ea20cd2bea5bc5f56eb68a2e958a0b29"
DEFAULT_OUTPUT = Path(
    "curriculum/interpretability/artifacts/raw_hero_snapshot_v1.json"
)

SOURCE_SPECS: dict[str, tuple[str, str]] = {
    "synthesis": (
        "research/analysis/raw_hero_interpretability_report_v1/synthesis.json",
        "compact validated Raw/Hero report tables",
    ),
    "model_diff": (
        "research/analysis/raw_hero_pilot_v1/metrics.json",
        "paired model-diff metrics",
    ),
    "model_diff_examples": (
        "research/analysis/raw_hero_pilot_v1/examples.jsonl",
        "paired development-position summaries",
    ),
    "literature": (
        "research/analysis/raw_hero_literature_pilot_v2/metrics.json",
        "attribution, attention, logit-lens, and patching pilot",
    ),
    "attribution_examples": (
        "research/analysis/raw_hero_literature_pilot_v2/attribution_examples.jsonl",
        "position-level attribution maps",
    ),
    "probes": (
        "research/analysis/modal_probe_all_layers_dev_v1/metrics.json",
        "group-split concept/lookahead probes and steering controls",
    ),
    "parameter_diff": (
        "research/analysis/raw_hero_parameter_diff_20260807.json",
        "Raw/Hero parameter deltas",
    ),
    "spectra": (
        "research/analysis/raw_hero_matrix_spectra_20260807.json",
        "selected parameter-delta spectra",
    ),
    "hero_lr": (
        "research/analysis/hero_encoder_lr_audit_20260808.json",
        "encoder update-budget audit",
    ),
    "transcoder": (
        "research/analysis/published_tc_l14_transfer_dev64_v1/metrics.json",
        "published MLP-transcoder compatibility pilot",
    ),
    "lorsa_transfer": (
        "research/analysis/published_lorsa_l14_transfer_dev512_v3/metrics.json",
        "frozen LoRSA Raw-source and Hero-transfer metrics",
    ),
    "lorsa_bootstrap": (
        "research/analysis/published_lorsa_l14_transfer_dev512_paired_bootstrap_v3/metrics.json",
        "paired LoRSA transfer uncertainty",
    ),
    "token_metrics": (
        "research/analysis/published_lorsa_l14_token_semantics_dev512_v3/metrics.json",
        "exact token-semantics summary",
    ),
    "token_selection": (
        "research/analysis/published_lorsa_l14_token_semantics_dev512_v3/selection.json",
        "frozen exact token-semantics pairs",
    ),
    "lorsa_causal": (
        "research/analysis/published_lorsa_l14_causal_dev256_v2/metrics.json",
        "matched-control sparse-feature interventions",
    ),
    "hero_handoff": (
        "docs/hero_proposal_a_continuation.md",
        "paused Proposal A estimand and predicted-action gate",
    ),
    "hero_config": (
        "research/analysis/hero_v2_phase1/hero1_exact/run_config.json",
        "tracked exact-Hero1 architecture manifest copy",
    ),
    "refinement_sweep": (
        "research/analysis/hero_refinement_pass_sweep_20260728.json",
        "paired one-versus-eight DFM refinement confirmation",
    ),
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _load_json(root: Path, source_id: str) -> Any:
    path = root / SOURCE_SPECS[source_id][0]
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(root: Path, source_id: str, *, limit: int) -> list[dict[str, Any]]:
    path = root / SOURCE_SPECS[source_id][0]
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
            if len(rows) == limit:
                break
    return rows


def _source_records(root: Path) -> list[dict[str, Any]]:
    records = []
    for source_id, (relative, role) in SOURCE_SPECS.items():
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        schema = None
        if path.suffix == ".json":
            value = json.loads(path.read_text(encoding="utf-8"))
            schema = value.get("schema_version") if isinstance(value, Mapping) else None
        records.append(
            {
                "id": source_id,
                "path": relative,
                "role": role,
                "sha256": _sha256_file(path),
                "size_bytes": path.stat().st_size,
                "schema_version": schema,
                "redistribution": "derived_metrics_or_minimal_fen_only",
                "license_note": (
                    "Course does not redistribute source checkpoints, LoRSA/probe "
                    "weights, bulk positions, or corpora; upstream license remains controlling."
                ),
            }
        )
    return records


def _trim_behavior_example(row: Mapping[str, Any]) -> dict[str, Any]:
    behavior = row["behavior"]
    return {
        "position_id": row["position_id"],
        "game_id": row["game_id"],
        "fen": row["fen"],
        "ply": row["ply"],
        "legal_count": behavior["legal_count"],
        "target_action": behavior["target_action"],
        "arms": behavior["arms"],
        "pairs": behavior["pairs"],
        "interaction": behavior["decomposition"],
    }


def _trim_attribution_example(row: Mapping[str, Any]) -> dict[str, Any]:
    arms: dict[str, Any] = {}
    for arm in ("RR", "HH"):
        source = row["arms"][arm]
        arms[arm] = {
            "maps": {
                name: {
                    "normalized_mass": payload["normalized_mass"],
                    "top_8": payload["top_8"],
                    "top_8_mass": payload["top_8_mass"],
                }
                for name, payload in source["maps"].items()
            },
            "diagnostics": source["diagnostics"],
            "sarfa": source["sarfa"],
        }
    return {
        "position_id": row["position_id"],
        "puzzle_id": row["puzzle_id"],
        "fen": row["root_fen"],
        "target_origin_token": row["target_origin_token"],
        "target_destination_token": row["target_destination_token"],
        "exact_plane_reconstruction": row["exact_plane_reconstruction"],
        "raw_hero_map_agreement": row["raw_hero_map_agreement"],
        "arms": arms,
    }


def _probe_steering(probes: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for layer in probes["layers"]:
        for model in ("raw", "hero"):
            effect = probes["causal_lookahead"][str(layer)][model][
                "probe_direction_policy_derivative"
            ]
            rows.append(
                {
                    "layer": layer,
                    "model": model,
                    "mean": effect["mean"],
                    "lower": effect["lower"],
                    "upper": effect["upper"],
                    "resolved": effect["lower"] > 0.0 or effect["upper"] < 0.0,
                }
            )
    return rows


def _semantic_pairs(selection: Mapping[str, Any]) -> list[dict[str, Any]]:
    wanted = {
        (10843, "piece_code", "our_knight"),
        (5429, "piece_code", "our_queen"),
        (10784, "attacked_undefended_ours", "present"),
        (6661, "pinned_ours", "present"),
        (4516, "attack_count_theirs", "two"),
        (5851, "pinned_theirs", "absent"),
    }
    rows = []
    for row in selection["pairs"]:
        key = (row["feature_index"], row["concept"], row["class_name"])
        if key in wanted:
            rows.append(row)
    if len(rows) != len(wanted):
        raise ValueError(f"Missing frozen semantic pairs: found {len(rows)}")
    return rows


def _causal_rows(metrics: Mapping[str, Any]) -> list[dict[str, Any]]:
    pairs = (
        "piece_code:2:feature_10843",
        "piece_code:5:feature_5429",
        "attacked_undefended_ours:1:feature_10784",
        "pinned_ours:1:feature_6661",
        "pinned_theirs:0:feature_5851",
        "attack_count_theirs:2:feature_4516",
    )
    rows = []
    for pair in pairs:
        models: dict[str, Any] = {}
        for arm in ("RR", "HH"):
            value = metrics["summary"]["arms"][arm][pair]
            effect = value["active_position_ablation_specificity"]["paired_difference"]
            models[arm] = {
                "active_positions": value["natural_feature_active_position_count"],
                "feature_index": value["target"]["feature_index"],
                "effect": effect,
                "gate_passed": value["exploratory_causal_feature_policy_gate"]["passed"],
                "holm_p": value["active_position_ablation_specificity"][
                    "sign_flip_pvalue_holm"
                ],
            }
        rows.append({"pair": pair, "models": models})
    return rows


def _transcoder_table(metrics: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Extract one coherent 64-position activation/replacement comparison."""

    rows = []
    for key, label in (("raw", "Raw BT4"), ("hero", "Hero")):
        reconstruction = metrics["reconstruction"][key]
        replacement = metrics["policy_replacement"][key]
        rows.append(
            {
                "model": label,
                "normalized_mse": reconstruction["normalized_mse"],
                "cosine": reconstruction["cosine"],
                "policy_js": replacement["js_divergence"]["mean"],
                "top1_agreement": replacement["top1_agreement"]["mean"],
            }
        )
    return rows


def _circuit_evidence_matrix(
    datasets: Mapping[str, Any], transcoder: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Version every synthesis row and replace the stale transcoder pilot."""

    rows = [dict(row) for row in datasets["evidence_matrix"]]
    for row in rows:
        row["source_run"] = "raw_hero_interpretability_report_v1"
        if row["method"] != "Published L14 transcoder transfer":
            continue
        raw_nmse = float(transcoder["reconstruction"]["raw"]["normalized_mse"])
        hero_nmse = float(transcoder["reconstruction"]["hero"]["normalized_mse"])
        row.update(
            {
                "finding": (
                    "Source compatibility passed; Hero normalized MSE was "
                    f"{hero_nmse / raw_nmse:.3f}× Raw."
                ),
                "scope": f"{int(transcoder['position_count'])} development positions",
                "source_run": "published_tc_l14_transfer_dev64_v1",
            }
        )
    return rows


def build_payload(root: Path) -> dict[str, Any]:
    synthesis = _load_json(root, "synthesis")
    model_diff = _load_json(root, "model_diff")
    literature = _load_json(root, "literature")
    probes = _load_json(root, "probes")
    parameters = _load_json(root, "parameter_diff")
    spectra = _load_json(root, "spectra")
    hero_lr = _load_json(root, "hero_lr")
    hero_config = _load_json(root, "hero_config")["config"]
    refinement = _load_json(root, "refinement_sweep")
    transcoder = _load_json(root, "transcoder")
    lorsa_transfer = _load_json(root, "lorsa_transfer")
    lorsa_bootstrap = _load_json(root, "lorsa_bootstrap")
    token_metrics = _load_json(root, "token_metrics")
    token_selection = _load_json(root, "token_selection")
    lorsa_causal = _load_json(root, "lorsa_causal")
    datasets = synthesis["datasets"]
    selected_spectrum = next(
        row for row in spectra["matrices"] if row["name"] == "embedding.proj.w"
    )
    parameter_groups = {
        key: parameters["group_metrics"][key]
        for key in (
            "all", "trunk", "smolgen", "embedding", "attention",
            "layer_norm", "mlp", "policy_head",
        )
    }
    if parameters["models"] != spectra["models"]:
        raise ValueError("Parameter-diff and spectrum model identities disagree")
    return {
        "architecture": {
            "input_planes": 112,
            "square_tokens": 64,
            "width": 1024,
            "layers": 15,
            "heads": 32,
            "head_width": 32,
            "mlp_width": 1536,
            "action_vocab": 1858,
            "jepa_state_width": hero_config["z_dim"],
            "dfm_state_width": hero_config["token_dim"],
            "dfm_state_source": hero_config["dfm_state_source"],
            "dfm_condition_on_current_jepa_state": hero_config[
                "dfm_condition_on_current_jepa_state"
            ],
            "dfm_jepa_fusion_mode": hero_config["dfm_jepa_fusion_mode"],
            "dfm_closed_loop_mode": hero_config["dfm_closed_loop_mode"],
            "model_identities": {
                "raw_asset_sha256": parameters["models"]["raw_asset_sha256"],
                "hero_state_sha256": parameters["models"]["hero_state_sha256"],
                "raw_mapping_sha256": parameters["models"]["raw_mapping_sha256"],
                "encoder_layout_sha256": parameters["models"]["encoder_layout_sha256"],
            },
            "hooks": {
                "hook_attn_in": "layer input before Q/K/V and SmolGen",
                "hook_attn_out": "projected attention branch before residual scale",
                "resid_mid_after_ln": "post-attention normalized residual",
                "hook_mlp_out": "MLP branch before residual scale",
                "resid_post_after_ln": "post-MLP normalized residual / next layer input",
            },
        },
        "design": {
            "claim_matrix_axes": ["operation", "target", "scope", "endpoint"],
            "tuning_validation": {"positions": 1152, "games": 15},
            "full_validation": {"positions": 8192, "games": 167},
            "heldout_input_overlap": {"validation": 0.0438, "test": 0.0414},
        },
        "behavior": {
            "arms": datasets["model_diff"]["arms"],
            "swaps": datasets["model_diff"]["swaps"],
            "examples": [
                _trim_behavior_example(row)
                for row in _load_jsonl(root, "model_diff_examples", limit=12)
            ],
            "scientific_status": model_diff["scientific_status"],
        },
        "geometry": {
            "activation_depth": datasets["model_diff"]["activation_depth"],
            "paired_delta": datasets["literature"]["paired_delta"],
            "parameter_groups": parameter_groups,
            "largest_delta_contributors": parameters["ranked_views"][
                "largest_delta_energy_contributors"
            ][:12],
            "embedding_projection_spectrum": selected_spectrum,
            "encoder_lr_verdict": hero_lr["verdict"],
        },
        "probes": {
            "concept_metrics": datasets["concept_metrics"],
            "concept_deltas": datasets["concept_deltas"],
            "lookahead_metrics": datasets["lookahead_metrics"],
            "lookahead_deltas": datasets["lookahead_deltas"],
            "steering": _probe_steering(probes),
            "correctness_gates": probes["correctness_gates"],
            "split_counts": probes["split_counts"],
        },
        "attribution": {
            "summary": datasets["literature"]["attribution"],
            "attention_pattern_js": datasets["literature"]["attention_pattern_js"],
            "examples": [
                _trim_attribution_example(row)
                for row in _load_jsonl(root, "attribution_examples", limit=4)
            ],
            "correctness_gates": literature["correctness_gates"],
        },
        "intervention": {
            "patching": datasets["literature"]["patching"],
            "identity_control": literature["identity_patch_control"],
        },
        "lookahead": {
            "metrics": datasets["lookahead_metrics"],
            "deltas": datasets["lookahead_deltas"],
            "teacher_action_jepa": {
                "future_target_latents_consumed": False,
                "recorded_actions_consumed": True,
                "clean_bidirectional_dfm_hidden_consumed": True,
                "inference_matched": False,
            },
            "open_loop_p1_p8": {
                "dfm_invoked": True,
                "jepa_invoked": False,
                "planning_claim_permitted": False,
            },
            "refinement_sweep": {
                "checkpoint_scope": refinement["decision"]["scope"],
                "decision": refinement["decision"]["status"],
                "pair_count": refinement["confirmation"]["combined_first_256"][
                    "pair_count"
                ],
                "pass_1_score": refinement["confirmation"]["pass_1"]["score"],
                "pass_8_score": refinement["confirmation"]["pass_8"]["score"],
                "pass_1_minus_pass_8": refinement["confirmation"][
                    "combined_first_256"
                ]["candidate_minus_control_score"],
                "paired_t_95_lower": refinement["confirmation"][
                    "combined_first_256"
                ]["paired_t_95_lower"],
                "paired_t_95_upper": refinement["confirmation"][
                    "combined_first_256"
                ]["paired_t_95_upper"],
                "pass_1_over_pass_8_latency": refinement["confirmation"][
                    "latency_ratio_pass_1_over_pass_8"
                ],
                "jepa_used_at_inference": refinement["confirmation"]["pass_1"][
                    "run_contract"
                ]["jepa_used_at_inference"],
                "seed": refinement["confirmation"]["pass_1"]["run_contract"][
                    "seed"
                ],
            },
        },
        "sparse": {
            "transcoder_position_count": transcoder["position_count"],
            "transcoder_source_compatibility_gate": transcoder[
                "source_compatibility_gate"
            ],
            "transcoder_table": _transcoder_table(transcoder),
            "transcoder_support": transcoder["support_transfer"],
        },
        "lorsa": {
            "transfer": lorsa_transfer,
            "paired_bootstrap": lorsa_bootstrap,
            "token_summary": {
                "raw_heldout": token_metrics["raw_heldout"],
                "hero_heldout": token_metrics["hero_heldout"],
                "model_diff": token_metrics["raw_hero_model_diff"],
                "claim_gates": token_metrics["claim_gates"],
            },
            "semantic_pairs": _semantic_pairs(token_selection),
            "causal": {
                "claim_gates": lorsa_causal["claim_gates"],
                "negative_results_retained": lorsa_causal["negative_results_retained"],
                "rows": _causal_rows(lorsa_causal),
            },
        },
        "circuits": {
            "evidence_matrix": _circuit_evidence_matrix(datasets, transcoder),
            "required_tests": [
                "faithfulness", "completeness", "necessity", "sufficiency",
                "redundancy", "cross-position stability", "interchange consistency",
            ],
        },
    }


def build_snapshot(root: Path) -> dict[str, Any]:
    snapshot = {
        "schema_version": SCHEMA_VERSION,
        "snapshot_id": "raw-hero-course-v1",
        "source_repository_commit": SOURCE_DATA_COMMIT,
        "extraction_version": 1,
        "evidence_scope": (
            "Frozen development artifacts only; reproduces named course figures, "
            "does not rerun models or create confirmatory evidence."
        ),
        "sources": _source_records(root),
        "payload": build_payload(root),
    }
    snapshot["integrity"] = {
        "algorithm": "sha256-canonical-json-without-integrity",
        "sha256": hashlib.sha256(_canonical_bytes(snapshot)).hexdigest(),
    }
    return snapshot


def write_snapshot(root: Path, output: Path) -> dict[str, Any]:
    snapshot = build_snapshot(root)
    destination = output if output.is_absolute() else root / output
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(snapshot, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return snapshot


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    snapshot = write_snapshot(args.root.resolve(), args.output)
    print(snapshot["integrity"]["sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
