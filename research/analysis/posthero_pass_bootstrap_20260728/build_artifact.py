#!/usr/bin/env python3
"""Build the portable technical report artifact for the post-Hero audit."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
AUDIT_PATH = (
    ROOT / "research/analysis/posthero_pass_bootstrap_audit_20260728.json"
)
OUTPUT_PATH = Path(__file__).resolve().parent / "artifact.json"
DERIVED_DIR = Path(__file__).resolve().parent / "derived"
GENERATED_AT = "2026-07-28T09:20:46Z"
TITLE = "Inference-Pass Scaling and Horizon Initialization Audit"
SERIES_LABELS = {
    "control_1024": "Matched 1,024-update control",
    "candidate_1024": "1,024-update JEPA broadcast",
    "hero_one_epoch": "One-epoch Hero",
}
STAGE_LABELS = {
    "initialization": "Initialization",
    "control_1024": "Matched 1,024-update control",
    "candidate_1024": "1,024-update JEPA broadcast",
    "hero_one_epoch": "One-epoch Hero",
}


def load_audit() -> dict[str, Any]:
    value = json.loads(AUDIT_PATH.read_text(encoding="utf-8"))
    if value.get("schema_version") != "posthero-pass-bootstrap-audit-v1":
        raise ValueError("Unexpected audit schema")
    return value


def metric_card(
    *,
    card_id: str,
    description: str,
    metrics: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "id": card_id,
        "dataset": "headline",
        "description": description,
        "metrics": metrics,
        "sourceId": "audit_record",
        "source": widget_source(
            dataset="headline",
            description=(
                "Reads the compact headline metrics derived from the "
                "immutable audit record."
            ),
        ),
    }


def widget_source(
    *,
    dataset: str,
    description: str,
) -> dict[str, Any]:
    relative_path = (
        "research/analysis/posthero_pass_bootstrap_20260728/"
        f"derived/{dataset}.json"
    )
    return {
        "id": f"{dataset}_source",
        "label": f"Derived {dataset.replace('_', ' ')} dataset",
        "path": relative_path,
        "query": {
            "id": f"read-{dataset}-20260728",
            "description": description,
            "engine": "DuckDB",
            "executed_at": GENERATED_AT,
            "language": "sql",
            "sql": (
                "SELECT *\n"
                f"FROM read_json_auto('{relative_path}');"
            ),
            "tables_used": [relative_path],
            "metric_definitions": [
                (
                    "Arena score is candidate points divided by two games per "
                    "color-reversed opening pair."
                ),
                (
                    "MSE / identity is recurrent JEPA MSE divided by the "
                    "current-state identity baseline MSE."
                ),
            ],
        },
    }


def source(
    *,
    source_id: str,
    label: str,
    path: str,
    description: str,
    tables_used: list[str],
    sql: str,
) -> dict[str, Any]:
    return {
        "id": source_id,
        "label": label,
        "path": path,
        "query": {
            "id": source_id,
            "description": description,
            "engine": "Repository source and immutable JSON audit",
            "executed_at": GENERATED_AT,
            "language": "sql",
            "sql": sql,
            "metric_definitions": [
                (
                    "Arena score is candidate points divided by two games per "
                    "color-reversed opening pair; it is descriptive on the "
                    "reused development pool."
                ),
                (
                    "MSE / identity divides recurrent JEPA prediction MSE by "
                    "the MSE obtained by retaining the current-state latent."
                ),
            ],
            "tables_used": tables_used,
        },
    }


def main() -> int:
    audit = load_audit()
    pass_scaling = [
        {
            "series": SERIES_LABELS[series],
            "passes": int(row["passes"]),
            "score": float(row["score"]),
            "mean_call_ms": float(
                row["mean_physical_call_ms"]
                if "mean_physical_call_ms" in row
                else row["full_run_mean_physical_call_ms"]
            ),
        }
        for series, rows in audit["pass_curves"].items()
        for row in rows
    ]
    pass_comparison = []
    for passes in (1, 2, 4, 8, 16):
        control = next(
            row
            for row in audit["pass_curves"]["control_1024"]
            if int(row["passes"]) == passes
        )
        candidate = next(
            row
            for row in audit["pass_curves"]["candidate_1024"]
            if int(row["passes"]) == passes
        )
        hero = next(
            row
            for row in audit["pass_curves"]["hero_one_epoch"]
            if int(row["passes"]) == passes
        )
        paired = audit["candidate_vs_control_paired_by_pass"][
            f"pass_{passes}"
        ]
        pass_comparison.append(
            {
                "passes": passes,
                "control_score": float(control["score"]),
                "candidate_score": float(candidate["score"]),
                "candidate_minus_control": float(
                    paired["candidate_minus_control_score"]
                ),
                "paired_lower": float(paired["paired_t_95_lower"]),
                "paired_upper": float(paired["paired_t_95_upper"]),
                "hero_score": float(hero["score"]),
                "control_call_ms": float(
                    control["mean_physical_call_ms"]
                ),
                "candidate_call_ms": float(
                    candidate["mean_physical_call_ms"]
                ),
            }
        )
    horizon_ratio = [
        {
            "stage": STAGE_LABELS[stage],
            "horizon": int(row["horizon"]),
            "mse_over_identity": float(row["mse_over_identity"]),
            "jepa_mse": float(row["jepa_mse"]),
            "identity_mse": float(row["identity_mse"]),
            "dfm_ce": float(row["dfm_ce"]),
        }
        for stage, rows in audit["horizon_audit"].items()
        for row in rows
    ]
    one_epoch_horizons = [
        {
            "horizon": int(row["horizon"]),
            "jepa_mse": float(row["jepa_mse"]),
            "identity_mse": float(row["identity_mse"]),
            "mse_over_identity": float(row["mse_over_identity"]),
            "dfm_ce": float(row["dfm_ce"]),
            "pred_target_cosine": float(row["pred_target_cosine"]),
            "pred_effective_rank": float(row["pred_effective_rank"]),
        }
        for row in audit["horizon_audit"]["hero_one_epoch"]
    ]
    hero_delta = audit["within_series_vs_one_pass"]["hero_one_epoch"][
        "pass_16"
    ]["candidate_minus_control_score"]
    headline = [
        {
            "direct_candidate_score": float(
                audit["direct_candidate_screen"]["score"]
            ),
            "direct_interval_lower": float(
                audit["direct_candidate_screen"][
                    "descriptive_paired_t_95"
                ][0]
            ),
            "direct_interval_upper": float(
                audit["direct_candidate_screen"][
                    "descriptive_paired_t_95"
                ][1]
            ),
            "replay_comparisons": int(
                audit["deterministic_candidate_replay"][
                    "scientific_scalar_comparisons"
                ]
            ),
            "replay_mismatches": int(
                audit["deterministic_candidate_replay"]["mismatch_count"]
            ),
            "hero_one_pass_score": float(
                audit["pass_curves"]["hero_one_epoch"][0]["score"]
            ),
            "hero_sixteen_pass_score": float(
                audit["pass_curves"]["hero_one_epoch"][-1]["score"]
            ),
            "hero_sixteen_minus_one": float(hero_delta),
        }
    ]
    derived_datasets = {
        "headline": headline,
        "pass_scaling": pass_scaling,
        "pass_comparison": pass_comparison,
        "horizon_ratio": horizon_ratio,
        "one_epoch_horizons": one_epoch_horizons,
    }
    DERIVED_DIR.mkdir(parents=True, exist_ok=True)
    for dataset, rows in derived_datasets.items():
        (DERIVED_DIR / f"{dataset}.json").write_text(
            json.dumps(rows, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    manifest = {
        "title": TITLE,
        "description": (
            "Technical audit of checkpoint identity, refinement-pass scaling, "
            "horizon initialization, and target-safe bootstrap options."
        ),
        "surface": "report",
        "version": 1,
        "generatedAt": GENERATED_AT,
        "blocks": [
            {
                "id": "title",
                "type": "markdown",
                "body": f"# {TITLE}",
            },
            {
                "id": "technical_summary",
                "type": "markdown",
                "sourceId": "audit_record",
                "body": (
                    "## Technical summary\n\n"
                    "The `0.509766` centered score compared the JEPA-broadcast "
                    "candidate with the **matched 1,024-update control**, not "
                    "the one-epoch Hero. It is a near tie, so the candidate "
                    "should be called **not promoted / inconclusive**, not "
                    "chess-rejected. An exact replay reproduced its checkpoint "
                    "and every retained scientific scalar. Both short models "
                    "still weaken overall with more passes, but the one-epoch "
                    "Hero improves monotonically from `0.664062` at one pass "
                    "to `0.935547` at sixteen. Initialization is asymmetric: "
                    "H1 inherits the current-board BT4 policy while H2–H8 "
                    "begin uniform and JEPA begins at the identity."
                ),
            },
            {
                "id": "headline_metrics",
                "type": "metric-strip",
                "cardIds": [
                    "direct_card",
                    "replay_card",
                    "hero_transition_card",
                ],
            },
            {
                "id": "pass_finding",
                "type": "markdown",
                "sourceId": "audit_record",
                "body": (
                    "## Refinement behavior changes during training\n\n"
                    "At 1,024 updates, sixteen passes remain about `0.05` "
                    "score worse than one for both the control and broadcast "
                    "candidate. By one epoch the sign has reversed: every "
                    "measured increase in passes helps, and sixteen beats one "
                    "by `0.271484`. Useful iterative denoising therefore "
                    "emerges during training rather than being guaranteed by "
                    "the DFM architecture at initialization."
                ),
            },
            {
                "id": "pass_chart_block",
                "type": "chart",
                "chartId": "pass_scaling_chart",
            },
            {
                "id": "candidate_finding",
                "type": "markdown",
                "sourceId": "arena_source",
                "body": (
                    "## The broadcast candidate has mixed evidence\n\n"
                    "Against raw BT4 it exceeds the control at all five pass "
                    "counts, with descriptive paired intervals excluding zero "
                    "at 1, 4, and 8 passes. Yet their direct centered Arena is "
                    "a tie and the candidate worsens held-out DFM CE, legal "
                    "mass, accuracy, and latent rank. Raw-BT4 play is near a "
                    "ceiling and these openings were reused, so this may "
                    "include style or non-transitivity. A disjoint centered "
                    "confirmation is required before extending or closing "
                    "the architecture."
                ),
            },
            {
                "id": "pass_table_block",
                "type": "table",
                "tableId": "pass_comparison_table",
            },
            {
                "id": "horizon_finding",
                "type": "markdown",
                "sourceId": "evaluation_source",
                "body": (
                    "## Long-horizon JEPA error needs a normalized baseline\n\n"
                    "At initialization, recurrent JEPA exactly matches the "
                    "identity baseline at every horizon. After one epoch H8 "
                    "raw MSE is about twice H1, but the H8 target has moved "
                    "about `5.3×` farther from the identity. H8 error is only "
                    "`2.7%` of identity error versus `7.4%` at H1. The more "
                    "persistent long-horizon weakness is action CE: `5.61` "
                    "at H8 versus `1.96` at H1."
                ),
            },
            {
                "id": "horizon_chart_block",
                "type": "chart",
                "chartId": "horizon_ratio_chart",
            },
            {
                "id": "horizon_table_block",
                "type": "table",
                "tableId": "one_epoch_horizon_table",
            },
            {
                "id": "initialization_finding",
                "type": "markdown",
                "sourceId": "code_source",
                "body": (
                    "## The initialization asymmetry is real\n\n"
                    "The learned DFM output projection starts at zero. H1 is "
                    "still exactly useful because current-board BT4 policy "
                    "logits are added as a permanent residual; H2–H8 have no "
                    "future-board residual and start uniform. Separately, "
                    "JEPA conditioning starts at zero and its residual "
                    "down-projection is tiny, making the recurrent transition "
                    "approximately identity."
                ),
            },
            {
                "id": "bootstrap_recommendation",
                "type": "markdown",
                "sourceId": "code_source",
                "body": (
                    "## Recommended first bootstrap\n\n"
                    "Use **training-only future-policy distillation**. The "
                    "already encoded sampled future state supplies stopped-"
                    "gradient BT4 legal-policy targets for the aligned H2–H8 "
                    "DFM slot, requiring no extra encoder call and no teacher "
                    "at inference. Profile the additional policy-head forward, "
                    "calibrate its initial weighted contribution, and decay "
                    "the auxiliary to zero so LC0-MCTS action labels remain "
                    "the terminal objective. Test teacher-forced adjacent JEPA "
                    "as a separate later candidate; do not combine both first."
                ),
            },
            {
                "id": "scope",
                "type": "markdown",
                "sourceId": "arena_source",
                "body": (
                    "## Scope and methodology\n\n"
                    "All pass curves use deterministic greedy policies, strict "
                    "root legality masking, the same first 128 color-reversed "
                    "`hero_development` pairs, seed 0, policy batch cap 16, "
                    "and 256 additional plies. Every run has zero faults, cap "
                    "draws, or abnormal terminations. The candidate replay "
                    "matches the sealed checkpoint hash and 53,248 scientific "
                    "scalar comparisons exactly."
                ),
            },
            {
                "id": "limitations",
                "type": "markdown",
                "sourceId": "audit_record",
                "body": (
                    "## Limitations\n\n"
                    "The opening pool is reused development data and raw-BT4 "
                    "scores are ceiling-prone; descriptive paired intervals "
                    "are not promotion tests or absolute Elo. Pass count also "
                    "changes the deterministic reveal/revision schedule, not "
                    "only FLOPs. The horizon audit diagnoses association and "
                    "initialization geometry; it does not prove which auxiliary "
                    "will improve chess strength."
                ),
            },
            {
                "id": "next_steps",
                "type": "markdown",
                "sourceId": "audit_record",
                "body": (
                    "## Next steps\n\n"
                    "1. Keep the matched control as incumbent and retain the "
                    "replayed broadcast tensor.\n"
                    "2. Run a disjoint centered confirmation before a longer "
                    "broadcast extension.\n"
                    "3. Profile future-policy distillation with the frozen "
                    "systems contract.\n"
                    "4. If opened, test it alone with one-pass Elo primary and "
                    "the full pass curve as a refinement diagnostic."
                ),
            },
            {
                "id": "further_questions",
                "type": "markdown",
                "body": (
                    "## Further questions\n\n"
                    "- At what update does the pass-gradient sign flip?\n"
                    "- Does future-policy distillation move that transition "
                    "earlier without sacrificing final LC0 imitation?\n"
                    "- Is the broadcast candidate’s raw-BT4 gain reproduced "
                    "in disjoint direct play?\n"
                    "- Which horizon-weighted CE best predicts multi-pass Elo?"
                ),
            },
        ],
        "cards": [
            metric_card(
                card_id="direct_card",
                description=(
                    "Centered candidate score and descriptive paired interval "
                    "against the matched short control."
                ),
                metrics=[
                    {
                        "field": "direct_candidate_score",
                        "format": "number",
                        "label": "Direct score vs control",
                    },
                    {
                        "field": "direct_interval_lower",
                        "format": "number",
                        "label": "Interval lower",
                    },
                    {
                        "field": "direct_interval_upper",
                        "format": "number",
                        "label": "Interval upper",
                    },
                ],
            ),
            metric_card(
                card_id="replay_card",
                description=(
                    "Bit-exact scientific scalar comparisons in the restored "
                    "1,024-update candidate replay."
                ),
                metrics=[
                    {
                        "field": "replay_comparisons",
                        "format": "compact",
                        "label": "Replay comparisons",
                    },
                    {
                        "field": "replay_mismatches",
                        "format": "number",
                        "label": "Mismatches",
                    },
                ],
            ),
            metric_card(
                card_id="hero_transition_card",
                description=(
                    "One-epoch Hero score versus raw BT4 at one and sixteen "
                    "refinement passes."
                ),
                metrics=[
                    {
                        "field": "hero_one_pass_score",
                        "format": "number",
                        "label": "Hero @ 1 pass",
                    },
                    {
                        "field": "hero_sixteen_pass_score",
                        "format": "number",
                        "label": "Hero @ 16 passes",
                    },
                    {
                        "field": "hero_sixteen_minus_one",
                        "format": "number",
                        "label": "16 minus 1",
                        "signed": True,
                    },
                ],
            ),
        ],
        "charts": [
            {
                "id": "pass_scaling_chart",
                "dataset": "pass_scaling",
                "type": "line",
                "title": "Score versus raw BT4 by refinement passes",
                "subtitle": (
                    "Only the one-epoch model improves monotonically with "
                    "additional refinement."
                ),
                "sourceId": "arena_source",
                "source": widget_source(
                    dataset="pass_scaling",
                    description=(
                        "Reads the tidy model-by-pass Arena score and latency "
                        "dataset."
                    ),
                ),
                "encodings": {
                    "x": {
                        "field": "passes",
                        "type": "quantitative",
                        "label": "Refinement passes",
                    },
                    "y": {
                        "field": "score",
                        "type": "quantitative",
                        "label": "Score versus raw BT4",
                        "format": "number",
                    },
                    "color": {
                        "field": "series",
                        "type": "nominal",
                        "label": "Model",
                    },
                    "tooltip": [
                        {
                            "field": "mean_call_ms",
                            "type": "quantitative",
                            "label": "Mean policy call (ms)",
                        }
                    ],
                },
            },
            {
                "id": "horizon_ratio_chart",
                "dataset": "horizon_ratio",
                "type": "line",
                "title": (
                    "JEPA MSE relative to the identity baseline by horizon"
                ),
                "subtitle": (
                    "After one epoch, relative error falls with horizon even "
                    "though absolute H8 MSE remains larger than H1."
                ),
                "sourceId": "evaluation_source",
                "source": widget_source(
                    dataset="horizon_ratio",
                    description=(
                        "Reads horizon-level JEPA error, identity baseline, "
                        "and DFM cross-entropy by training stage."
                    ),
                ),
                "encodings": {
                    "x": {
                        "field": "horizon",
                        "type": "quantitative",
                        "label": "Prediction horizon",
                    },
                    "y": {
                        "field": "mse_over_identity",
                        "type": "quantitative",
                        "label": "JEPA MSE / identity MSE",
                        "format": "number",
                    },
                    "color": {
                        "field": "stage",
                        "type": "nominal",
                        "label": "Training stage",
                    },
                    "tooltip": [
                        {
                            "field": "jepa_mse",
                            "type": "quantitative",
                            "label": "JEPA MSE",
                        },
                        {
                            "field": "identity_mse",
                            "type": "quantitative",
                            "label": "Identity MSE",
                        },
                        {
                            "field": "dfm_ce",
                            "type": "quantitative",
                            "label": "DFM CE",
                        },
                    ],
                },
            },
        ],
        "tables": [
            {
                "id": "pass_comparison_table",
                "dataset": "pass_comparison",
                "title": "Matched pass-count comparison",
                "subtitle": (
                    "Candidate-minus-control intervals are descriptive on "
                    "the same 128 opening pairs."
                ),
                "sourceId": "arena_source",
                "source": widget_source(
                    dataset="pass_comparison",
                    description=(
                        "Reads matched pass-count scores, candidate-control "
                        "paired deltas, intervals, and latency."
                    ),
                ),
                "defaultSort": {"field": "passes", "direction": "asc"},
                "columns": [
                    {
                        "field": "passes",
                        "label": "Passes",
                        "type": "number",
                    },
                    {
                        "field": "control_score",
                        "label": "Control score",
                        "format": "number",
                    },
                    {
                        "field": "candidate_score",
                        "label": "Broadcast score",
                        "format": "number",
                    },
                    {
                        "field": "candidate_minus_control",
                        "label": "Broadcast − control",
                        "format": "number",
                    },
                    {
                        "field": "paired_lower",
                        "label": "Paired lower",
                        "format": "number",
                    },
                    {
                        "field": "paired_upper",
                        "label": "Paired upper",
                        "format": "number",
                    },
                    {
                        "field": "hero_score",
                        "label": "One-epoch score",
                        "format": "number",
                    },
                    {
                        "field": "candidate_call_ms",
                        "label": "Broadcast call (ms)",
                        "format": "number",
                    },
                ],
            },
            {
                "id": "one_epoch_horizon_table",
                "dataset": "one_epoch_horizons",
                "title": "One-epoch horizon diagnostics",
                "subtitle": (
                    "Absolute JEPA MSE, identity-normalized error, and action "
                    "cross-entropy use the same frozen evaluation."
                ),
                "sourceId": "evaluation_source",
                "source": widget_source(
                    dataset="one_epoch_horizons",
                    description=(
                        "Reads the one-epoch horizon-level JEPA, identity, "
                        "action-CE, cosine, and rank diagnostics."
                    ),
                ),
                "defaultSort": {"field": "horizon", "direction": "asc"},
                "columns": [
                    {
                        "field": "horizon",
                        "label": "Horizon",
                        "type": "number",
                    },
                    {
                        "field": "jepa_mse",
                        "label": "JEPA MSE",
                        "format": "number",
                    },
                    {
                        "field": "identity_mse",
                        "label": "Identity MSE",
                        "format": "number",
                    },
                    {
                        "field": "mse_over_identity",
                        "label": "MSE / identity",
                        "format": "number",
                    },
                    {
                        "field": "dfm_ce",
                        "label": "DFM CE",
                        "format": "number",
                    },
                    {
                        "field": "pred_target_cosine",
                        "label": "Pred/target cosine",
                        "format": "number",
                    },
                    {
                        "field": "pred_effective_rank",
                        "label": "Prediction eff. rank",
                        "format": "number",
                    },
                ],
            },
        ],
        "sources": [
            source(
                source_id="audit_record",
                label="Immutable post-Hero audit record",
                path=(
                    "research/analysis/"
                    "posthero_pass_bootstrap_audit_20260728.json"
                ),
                description=(
                    "Checks model identities and frozen contracts, reconstructs "
                    "pass curves, computes paired opening deltas, verifies the "
                    "deterministic replay, and extracts horizon diagnostics."
                ),
                tables_used=[
                    "research/analysis/"
                    "posthero_pass_bootstrap_audit_20260728.json"
                ],
                sql=(
                    "SELECT *\n"
                    "FROM read_json_auto('research/analysis/"
                    "posthero_pass_bootstrap_audit_20260728.json');"
                ),
            ),
            source(
                source_id="arena_source",
                label="Frozen Arena states",
                path="artifacts/arena",
                description=(
                    "Reads immutable state.json records for matched "
                    "1/2/4/8/16-pass raw-BT4 Arenas."
                ),
                tables_used=[
                    "artifacts/arena/*128pairs*/state.json",
                    (
                        "artifacts/arena/"
                        "hero-epoch-v1-terminal-vs-raw-bt4-1024pairs/"
                        "state.json"
                    ),
                ],
                sql=(
                    "SELECT *\n"
                    "FROM read_json_auto('research/analysis/"
                    "posthero_pass_bootstrap_20260728/derived/"
                    "pass_comparison.json');"
                ),
            ),
            source(
                source_id="evaluation_source",
                label="Frozen horizon evaluations",
                path="research/runs",
                description=(
                    "Reads initialization, matched 1,024-update, and "
                    "one-epoch frozen evaluation reports."
                ),
                tables_used=[
                    "research/runs/torch_hero_init_primary_eval_v1/report.json",
                    (
                        "research/runs/"
                        "torch_autoresearch_sigreg64_u1024_v1_fast_eval_v1/"
                        "report.json"
                    ),
                    (
                        "research/runs/"
                        "torch_autoresearch_jepa_condition_u1024_v1_fast_eval_v1/"
                        "report.json"
                    ),
                    (
                        "research/runs/"
                        "torch_hero_epoch_v1_terminal_blind_eval_v1/"
                        "report.json"
                    ),
                ],
                sql=(
                    "SELECT *\n"
                    "FROM read_json_auto('research/analysis/"
                    "posthero_pass_bootstrap_20260728/derived/"
                    "horizon_ratio.json');"
                ),
            ),
            source(
                source_id="code_source",
                label="Torch model and trajectory alignment audit",
                path="research/train_torch.py",
                description=(
                    "Audits zero initialization, H1 BT4 residual policy, "
                    "recurrent JEPA transition, DFM reveal schedule, and "
                    "future-plane/action alignment."
                ),
                tables_used=[
                    "research/train_torch.py",
                    "chess_dfm_jax/data/trajectory.py",
                ],
                sql=(
                    "SELECT 'research/train_torch.py' AS path\n"
                    "UNION ALL\n"
                    "SELECT 'chess_dfm_jax/data/trajectory.py' AS path;"
                ),
            ),
        ],
    }
    artifact = {
        "surface": "report",
        "manifest": manifest,
        "snapshot": {
            "version": 1,
            "status": "ready",
            "generatedAt": GENERATED_AT,
            "datasets": {
                **derived_datasets,
            }
        },
        "package_info": {
            "analysisVersion": 1,
            "origin": (
                "chess-dfm-jax/research/analysis/"
                "posthero_pass_bootstrap_20260728"
            ),
        },
    }
    OUTPUT_PATH.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(OUTPUT_PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
