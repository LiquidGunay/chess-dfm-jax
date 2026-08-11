# /// script
# dependencies = ["marimo", "numpy", "python-chess"]
# requires-python = ">=3.11"
# ///

import marimo

__generated_with = "0.23.3"
app = marimo.App(width="full")


@app.cell
def imports():
    import sys
    from pathlib import Path

    import chess
    import marimo as mo
    import numpy as np

    repo_root = Path(__file__).resolve().parents[3]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from research.interpretability.course import get_module, load_course_bundle
    from research.interpretability.course.evidence import (
        ClaimCard,
        Endpoint,
        EvidenceOperation,
        ResultStatus,
        TargetLevel,
    )
    from research.interpretability.course.experiments import toy_lookahead_rollout
    from research.interpretability.course.ui import (
        callout,
        chessboard,
        claim_card_html,
        committed_form,
        line_chart,
        metric_cards,
        mode_badge,
        module_header,
        resource_card,
        table,
    )

    return (
        ClaimCard,
        Endpoint,
        EvidenceOperation,
        ResultStatus,
        TargetLevel,
        callout,
        chess,
        chessboard,
        claim_card_html,
        committed_form,
        get_module,
        line_chart,
        load_course_bundle,
        metric_cards,
        mo,
        mode_badge,
        module_header,
        np,
        resource_card,
        table,
        toy_lookahead_rollout,
    )


@app.cell
def title(get_module, mo, module_header):
    module = get_module(6)
    mo.Html(module_header(module))
    return (module,)


@app.cell
def opening(mo, module, table):
    objective_rows = [
        {"objective": objective, "mastery": mastery}
        for objective, mastery in zip(
            module.objectives,
            (
                "Classify an unseen result as probe decodability, DFM refinement, conditional state prediction, or explicit search.",
                "Write a rollout that replays generated actions, stops at first illegality, and scores exact state targets.",
                "Construct at least three cheaper non-planning explanations for a partial result.",
            ),
            strict=True,
        )
    ]
    mo.vstack([
        mo.md(r"""
        ## “Planning” is a hypothesis, not a metric label

        Chess lets us ask an unusually sharp question: does a model generate
        candidate actions, predict the resulting states or values, and use that
        information to select behavior? Several weaker observations are
        compatible with that story:

        - a future move is linearly decodable from one activation;
        - more DFM denoising passes improve an action trajectory;
        - a JEPA predicts future latents when given the recorded action line; or
        - a model has memorized motifs that correlate with future play.

        The lesson is successful only if those remain distinct in your claim.
        """),
        mo.Html(table(objective_rows, (("objective", "Objective"), ("mastery", "Unseen formative evidence")))),
    ])
    return


@app.cell
def prerequisite(committed_form, mo):
    response = mo.ui.text_area(
        label="Prerequisite diagnostic — intervention scope",
        placeholder="Why does a future-action probe plus any p1/p8 pass-count difference not yet establish state-based planning? Name the two missing links.",
    )
    prerequisite_form = committed_form(
        mo, {"lookahead_boundary": response}, submit_label="Commit prerequisite response", min_words=6
    )
    prerequisite_form
    return (prerequisite_form,)


@app.cell
def prerequisite_feedback(mo, prerequisite_form):
    mo.stop(prerequisite_form.value is None, mo.md("Answer the prerequisite diagnostic to reveal the bridge."))
    mo.callout("Decodability does not show causal use, and DFM-only refinement does not show JEPA state feedback. The missing experiment rolls out generated legal actions, predicts exact states/values, and tests whether that feedback changes action selection under matched compute.", kind="info")
    return


@app.cell
def mode_controls(mo):
    evidence_mode = mo.ui.radio(
        options={
            "Snapshot — retained Raw/Hero lookahead records": "snapshot",
            "Toy — constructed rollout mismatch": "toy",
            "Source — verify retained lookahead records": "source",
        },
        value="Snapshot — retained Raw/Hero lookahead records",
        label="Evidence mode",
    )
    verify_source = mo.ui.run_button(label="Verify retained sources", kind="warn")
    mo.hstack([evidence_mode, verify_source], justify="start", gap=1)
    return evidence_mode, verify_source


@app.cell
def load_mode(evidence_mode, load_course_bundle, mo, verify_source):
    selected_mode = evidence_mode.value
    mo.stop(selected_mode == "source" and not verify_source.value, mo.callout("Source mode requires explicit checksum verification.", kind="warn"))
    bundle = load_course_bundle(selected_mode)
    return bundle, selected_mode


@app.cell
def mode_view(bundle, mo, mode_badge, selected_mode):
    mo.Html(mode_badge(selected_mode, bundle.evidence_scope))
    return


@app.cell
def concept_ladder(mo, table):
    distinction_rows = [
        {"term": "future decodability", "operational test": "held-out decoder predicts a future action/state label", "still compatible with": "correlation, copied input, motif memory"},
        {"term": "DFM refinement", "operational test": "more discrete denoising calls improve the final action vector", "still compatible with": "iterative error correction without state simulation"},
        {"term": "conditional JEPA prediction", "operational test": "future latent error beats baselines given an action line", "still compatible with": "teacher-action conditional dynamics only"},
        {"term": "inference-matched rollout", "operational test": "generated actions are replayed; predicted states/values are scored at valid prefixes", "still compatible with": "world model exists but policy ignores it"},
        {"term": "planning-like use", "operational test": "state/value rollout changes action selection under causal controls and beats non-rollout alternatives", "still compatible with": "bounded amortized computation, not necessarily tree search"},
        {"term": "explicit search", "operational test": "multiple candidates/branches are expanded and compared under a defined budget", "still compatible with": "search can be shallow or weak"},
    ]
    mo.Html(table(distinction_rows, (("term", "Term"), ("operational test", "Operational test"), ("still compatible with", "Strong alternative still left"))))
    return


@app.cell
def language_boundary(callout, mo):
    mo.Html(callout(
        "limit",
        "Chess gives us unusually strong ground truth",
        (
            "A language model has no deterministic sentence simulator, exact "
            "legal-token oracle, or unique future hidden-state target analogous "
            "to replaying a chess move into an exact board. An LM analogue must "
            "operationalize state, validity, and value with proxies, preserve "
            "multiple plausible continuations, and test stronger alternatives. "
            "The chess experiment transfers as a design pattern—not as proof "
            "that an LM performs the same internal computation."
        ),
    ))
    return


@app.cell
def dfm_explanation(mo, table):
    train_test_rows = [
        {"system": "DFM training", "state supplied": "Z_DFM [B,64,256] projected from trunk tokens", "action supplied": "randomly corrupted/masked 8-slot vector", "network calls": "typically one sampled corruption", "objective": "recover clean action tokens"},
        {"system": "DFM inference p1/p8", "state supplied": "the same Z_DFM; no pooled JEPA state", "action supplied": "initial noisy/masked slots, then its own revised slots", "network calls": "1 or 8 refinement calls", "objective": "produce final action vector"},
        {"system": "current JEPA train/eval", "state supplied": "pooled z_JEPA [B,1024], then its own predicted latent recurrently", "action supplied": "recorded future actions + clean bidirectional DFM hidden states", "network calls": "one horizon rollout", "objective": "predict exact future pooled encoder latents"},
        {"system": "required inference-matched JEPA", "state supplied": "z_JEPA then predicted pooled latent", "action supplied": "DFM-generated legal prefix", "network calls": "one rollout per generated line/candidate", "objective": "predict replayed states/values and test policy use"},
    ]
    mo.vstack([
        mo.md("## How discrete refinement differs from autoregressive generation"),
        mo.Html(table(train_test_rows, (("system", "Procedure"), ("state supplied", "State stream"), ("action supplied", "Action stream"), ("network calls", "Calls"), ("objective", "What is optimized/scored")))),
        mo.md(r"""
        At denoising step \(k\), the DFM sees a partially specified vector
        \(x^{(k)}=(a_0,\ldots,a_7)\) with masks/noise and predicts distributions
        for all slots. A schedule keeps or resamples slots to form
        \(x^{(k+1)}\). This resembles masked diffusion: multiple calls refine
        the **same whole vector**, unlike an autoregressive model that appends a
        new suffix token each call.

        In Proposal A, JEPA loss sends a **training gradient** into the shared
        encoder that produces both state streams. But the DFM directly reads
        (Z_{\text{DFM}}\), while JEPA reads the separately pooled
        (z_{\text{JEPA}}\). During current DFM inference, neither the pooled
        JEPA state nor its predictions are read by the denoiser. A pass-count
        effect in either direction is therefore a DFM refinement result, not
        JEPA feedback or state-based planning.
        """),
    ])
    return


@app.cell
def classification_exercise(committed_form, mo):
    evidence_scenario = mo.ui.dropdown(
        options={
            "A third-ply probe scores 70%": "probe",
            "Hypothetical: p8 beats p1 while JEPA is disabled": "dfm",
            "JEPA latent MSE is low under recorded actions": "teacher_jepa",
            "Generated legal lines produce accurate values that alter chosen moves": "planning_use",
        },
        value=None,
        label="Evidence scenario",
    )
    evidence_class = mo.ui.dropdown(
        options=["future decodability", "DFM refinement", "conditional state prediction", "planning-like use"],
        value=None,
        label="Classification",
    )
    classification_form = committed_form(
        mo,
        {"scenario": evidence_scenario, "classification": evidence_class},
        submit_label="Commit classification",
    )
    classification_form
    return (classification_form,)


@app.cell
def classification_feedback(callout, classification_form, mo):
    answer_key = {
        "probe": "future decodability",
        "dfm": "DFM refinement",
        "teacher_jepa": "conditional state prediction",
        "planning_use": "planning-like use",
    }
    mo.stop(classification_form.value is None, mo.md("Commit before revealing."))
    scenario = classification_form.value["scenario"]
    class_ok = classification_form.value["classification"] == answer_key[scenario]
    mo.Html(callout(
        "positive" if class_ok else "limit",
        "Evidence named precisely" if class_ok else "Keep the operation in the name",
        f"The strongest direct label is **{answer_key[scenario]}**. Promotion requires the next operation, not stronger prose about the same metric.",
    ))
    return


@app.cell
def rollout_controls(mo):
    action_condition = mo.ui.radio(
        options={
            "Recorded / teacher actions": "teacher",
            "Predicted actions": "predicted",
            "Identity state baseline": "identity",
        },
        value="Predicted actions",
        label="State-prediction condition",
    )
    error_rate = mo.ui.number(0.0, 0.5, step=0.05, value=0.2, label="predicted-action error rate")
    horizon = mo.ui.number(1, 8, step=1, value=8, label="maximum scored horizon")
    legality_rule = mo.ui.radio(
        options={
            "Mask first illegal ply and every later horizon": "prefix",
            "Drop only the illegal token; keep later horizons": "token_only",
        },
        value="Mask first illegal ply and every later horizon",
        label="Legality rule",
    )
    mo.vstack([action_condition, mo.hstack([error_rate, horizon]), legality_rule])
    return action_condition, error_rate, horizon, legality_rule


@app.cell
def rollout_experiment(
    action_condition,
    error_rate,
    horizon,
    legality_rule,
    np,
    toy_lookahead_rollout,
):
    rollout = toy_lookahead_rollout(predicted_error_rate=error_rate.value)
    predictions = {
        "teacher": rollout["teacher_prediction"],
        "predicted": rollout["predicted_prediction"],
        "identity": rollout["identity_prediction"],
    }[action_condition.value]
    squared_error = np.mean((predictions - rollout["targets"]) ** 2, axis=-1)
    if legality_rule.value == "prefix":
        valid = rollout["valid_prefix"]
    else:
        valid = rollout["legal"]
    valid = valid[:, : horizon.value]
    horizon_mse = []
    horizon_counts = []
    for step in range(horizon.value):
        step_valid = valid[:, step]
        horizon_counts.append(int(np.count_nonzero(step_valid)))
        horizon_mse.append(float(np.mean(squared_error[step_valid, step])) if np.any(step_valid) else float("nan"))
    legal_line_rate = float(np.mean(rollout["first_illegal"] < 0))
    return horizon_counts, horizon_mse, legal_line_rate


@app.cell
def rollout_view(
    action_condition,
    horizon_counts,
    horizon_mse,
    legal_line_rate,
    legality_rule,
    line_chart,
    metric_cards,
    mo,
):
    rollout_series = [{"name": f"{action_condition.value} condition", "points": [(index + 1, value) for index, value in enumerate(horizon_mse)], "color": "#3366a3"}]
    mo.vstack([
        mo.md("## Interaction: exposure error and the denominator by horizon"),
        mo.Html(line_chart(rollout_series, x_label="rollout horizon", y_label="constructed latent MSE", alt_text="State prediction error by rollout horizon under the selected action condition and legality rule.")),
        mo.Html(metric_cards([
            (f"{horizon_mse[0]:.3f}", "MSE at horizon 1"),
            (f"{horizon_mse[-1]:.3f}", "MSE at final selected horizon"),
            (horizon_counts[-1], "valid examples at final horizon"),
            (f"{legal_line_rate:.1%}", "fully legal predicted lines"),
            (legality_rule.value, "scoring rule"),
        ])),
        mo.md("Teacher-action error estimates conditional dynamics under an easier action distribution. Predicted-action error includes exposure and legality failures. They answer different estimands and should appear side by side, not be pooled."),
    ])
    return


@app.cell
def board_line_control(mo):
    line_choice = mo.ui.dropdown(
        options={
            "Recorded legal Ruy Lopez line": "teacher",
            "Different but legal Queen's Gambit line": "alternative",
            "Line with an illegal third action": "illegal",
        },
        value="Line with an illegal third action",
        label="Replay an action proposal from the initial board",
    )
    line_choice
    return (line_choice,)


@app.cell
def board_replay(chess, line_choice):
    action_lines = {
        "teacher": ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6", "b5a4", "g8f6"],
        "alternative": ["d2d4", "d7d5", "c2c4", "e7e6", "b1c3", "g8f6", "c1g5", "f8e7"],
        "illegal": ["e2e4", "e7e5", "e2e5", "b8c6", "f1b5", "a7a6", "b5a4", "g8f6"],
    }
    proposal = action_lines[line_choice.value]
    replay_board = chess.Board()
    replay_rows = []
    prefix_valid = True
    final_valid_board = replay_board.copy()
    for ply, uci in enumerate(proposal, start=1):
        move = chess.Move.from_uci(uci)
        action_legal = prefix_valid and move in replay_board.legal_moves
        if action_legal:
            san = replay_board.san(move)
            replay_board.push(move)
            final_valid_board = replay_board.copy()
        else:
            san = "—"
            prefix_valid = False
        replay_rows.append({
            "ply": ply,
            "UCI": uci,
            "SAN": san,
            "token legal now": action_legal,
            "horizon scoreable": prefix_valid,
            "exact target": replay_board.fen() if prefix_valid else "masked after first illegal action",
        })
    return final_valid_board, replay_rows


@app.cell
def board_replay_view(chessboard, final_valid_board, mo, replay_rows, table):
    mo.vstack([
        mo.md("## Exact simulator replay: why later plies cannot survive an illegal prefix"),
        mo.Html(table(replay_rows, (("ply", "Ply"), ("UCI", "UCI"), ("SAN", "SAN"), ("token legal now", "Legal"), ("horizon scoreable", "Scoreable"), ("exact target", "Resulting-state target FEN")))),
        mo.Html(chessboard(final_valid_board.fen())),
        mo.md("Once a proposed action is illegal, the model has no defined chess state from which its later actions could be interpreted. Dropping only that token creates targets for a trajectory that never occurred."),
    ])
    return


@app.cell
def real_lookahead_panel(
    ClaimCard,
    Endpoint,
    EvidenceOperation,
    ResultStatus,
    TargetLevel,
    bundle,
    callout,
    claim_card_html,
    line_chart,
    mo,
    table,
):
    if bundle.is_empirical:
        lookahead = bundle.section("lookahead")
        target = "Third-ply source | predicted destination"
        metric_rows = [row for row in lookahead["metrics"] if row["target"] == target]
        lookahead_series = []
        for model, color in (("Raw BT4", "#3366a3"), ("Hero", "#a95f19")):
            lookahead_series.append({"name": model, "color": color, "points": [(row["layer"], row["accuracy"]) for row in metric_rows if row["model"] == model]})
        delta_rows = [row for row in lookahead["deltas"] if row["target"] == target]
        hero_higher = sum(row["direction"] == "Hero higher" for row in delta_rows)
        raw_higher = sum(row["direction"] == "Raw higher" for row in delta_rows)
        unresolved = sum(row["direction"] == "unresolved" for row in delta_rows)
        lookahead_card = ClaimCard(
            operation=EvidenceOperation.PREDICTION,
            target=TargetLevel.REPRESENTATION,
            scope="512 retained public puzzle continuations for Raw BT4 and Hero",
            endpoint=Endpoint.PROBE,
            estimand="third-ply source conditional on predicted destination accuracy by layer",
            unit="puzzle continuation",
            grouping="paired puzzle examples in a development evaluation pool",
            intervention=None,
            controls=("published decoder architecture", "simple/permutation controls", "paired Raw/Hero layer deltas"),
            assumptions=("puzzle continuation is a valid operational future-action label", "probe selection did not use the unopened test split"),
            uncertainty="paired retained intervals per layer",
            multiplicity="15-layer exploratory scan; no familywise confirmatory claim",
            allowed=(f"For this target, the retained layer scan contains {hero_higher} Hero-higher, {raw_higher} Raw-higher, and {unresolved} unresolved paired intervals; it does not show a systematic Hero advantage."),
            excluded="Future-action decodability does not establish inference-time JEPA use, explicit search, or an internally replayed board.",
            falsifier="A preregistered grouped benchmark where Hero exceeds Raw on the frozen multi-target criterion and survives causal-use controls.",
            status=ResultStatus.UNRESOLVED,
        )
        mismatch = lookahead["teacher_action_jepa"]
        open_loop = lookahead["open_loop_p1_p8"]
        refinement = lookahead["refinement_sweep"]
        refinement_card = ClaimCard(
            operation=EvidenceOperation.INTERVENTION,
            target=TargetLevel.BEHAVIOR,
            scope=refinement["checkpoint_scope"],
            endpoint=Endpoint.ARENA,
            estimand="paired deterministic-arena score(pass 1) minus score(pass 8) against the same Raw BT4 reference",
            unit="opening pair",
            grouping="256 frozen paired openings; two color-swapped games per pair",
            intervention="change only the DFM refinement-pass budget from one to eight",
            controls=("same checkpoint", "same Raw BT4 reference", "same ordered openings", "same seed 0", "JEPA disabled in both arms"),
            assumptions=("paired opening score is the intended checkpoint-specific endpoint", "the descriptive paired-t interval is not the preregistered decision rule"),
            uncertainty=f"descriptive paired-t 95% interval [{refinement['paired_t_95_lower']:+.5f}, {refinement['paired_t_95_upper']:+.5f}]",
            multiplicity="development sweep selected one versus eight; this 256-pair confirmation is checkpoint-specific",
            allowed=(f"For this retained checkpoint, pass 1 exceeded pass 8 by {refinement['pass_1_minus_pass_8']:+.5f} paired score (pass-1 {refinement['pass_1_score']:.5f}; pass-8 {refinement['pass_8_score']:.5f}) and used {refinement['pass_1_over_pass_8_latency']:.3f}× the latency."),
            excluded="This does not show that one pass is universally optimal, that refinement cannot help another checkpoint, or that either arm uses JEPA or plans.",
            falsifier="A separately preregistered checkpoint-specific paired sweep where a larger pass budget clears its frozen strength-and-cost gate.",
            status=ResultStatus.POSITIVE,
        )
        mismatch_rows = [
            {"record": "JEPA recurrent latent", "value": not mismatch["future_target_latents_consumed"], "meaning": "does not consume future target latents"},
            {"record": "recorded actions consumed", "value": mismatch["recorded_actions_consumed"], "meaning": "teacher-action conditional"},
            {"record": "clean DFM hidden consumed", "value": mismatch["clean_bidirectional_dfm_hidden_consumed"], "meaning": "additional clean action context"},
            {"record": "inference matched", "value": mismatch["inference_matched"], "meaning": "predicted-action rollout still required"},
            {"record": "p1/p8 invokes JEPA", "value": open_loop["jepa_invoked"], "meaning": "p1/p8 currently measures DFM refinement"},
        ]
        real_lookahead_view = mo.vstack([
            mo.md("## Frozen Raw/Hero evidence and the central negative case"),
            mo.Html(line_chart(lookahead_series, x_label="BT4 layer", y_label="third-ply probe accuracy", y_domain=(0.2, 0.8), alt_text="Raw and Hero third-ply source-given-destination probe curves rise through middle layers and differ inconsistently.")),
            mo.Html(claim_card_html(lookahead_card)),
            mo.md("## Genuine negative result: more refinement was worse here"),
            mo.Html(claim_card_html(refinement_card)),
            mo.Html(table(mismatch_rows, (("record", "Recorded fact"), ("value", "Value"), ("meaning", "Scientific meaning")))),
            mo.Html(callout("danger", "No inference-time JEPA planning result yet", "JEPA rollout is recurrent in predicted state but conditioned on recorded actions and clean DFM hidden states. The p1/p8 arena path invokes DFM and does not invoke JEPA. Both are useful results; neither is the missing predicted-action, JEPA-influenced action-selection experiment.")),
        ])
    else:
        real_lookahead_view = mo.Html(callout("neutral", "Toy boundary", "Constructed exposure error cannot support a Raw/Hero lookahead conclusion."))
    real_lookahead_view
    return


@app.cell
def promotion_protocol(mo, table):
    protocol_rows = [
        {"step": 1, "freeze": "DFM sampling schedule, p1/p8 budget, candidate count, seeds"},
        {"step": 2, "freeze": "exact board replay; first-illegal prefix mask; horizon denominators"},
        {"step": 3, "freeze": "state target E(sₜ), value target/source, identity and teacher-action baselines"},
        {"step": 4, "freeze": "intervention that disables/swaps JEPA feedback while matching DFM calls and compute"},
        {"step": 5, "freeze": "policy/value/arena endpoint, game clusters, openings, multiple-comparison rule"},
        {"step": 6, "freeze": "non-planning alternatives: motif retrieval, policy smoothing, extra compute, decoder leakage"},
    ]
    mo.vstack([
        mo.md("## Promotion experiment: State → Action → State → Value → Action"),
        mo.Html(table(protocol_rows, (("step", "Step"), ("freeze", "Preregister before running")))),
        mo.md("A value head makes predicted latents decision-relevant only after its labels and use are defined. Candidate sources include LC0 node values or outcome-calibrated targets; merely attaching a head does not create value supervision."),
    ])
    return


@app.cell
def alternatives(committed_form, mo):
    alternatives_answer = mo.ui.multiselect(
        options=[
            "memorized tactical motifs",
            "extra DFM compute improves consistency",
            "future label leaks through the dataset construction",
            "a world model exists but the policy ignores it",
            "explicit branch expansion and comparison",
        ],
        value=[],
        label="For the hypothetical p8>p1 + teacher-action JEPA-accuracy pattern, select alternatives that remain consistent",
    )
    alternatives_form = committed_form(
        mo, {"alternatives": alternatives_answer}, submit_label="Check alternatives"
    )
    alternatives_form
    return (alternatives_form,)


@app.cell
def alternatives_feedback(alternatives_form, callout, mo):
    mo.stop(alternatives_form.value is None, mo.md("Choose before revealing."))
    submitted_alternatives = alternatives_form.value["alternatives"]
    required_alternatives = {"memorized tactical motifs", "extra DFM compute improves consistency", "a world model exists but the policy ignores it"}
    alternatives_ok = required_alternatives.issubset(set(submitted_alternatives)) and "explicit branch expansion and comparison" not in submitted_alternatives
    mo.Html(callout(
        "positive" if alternatives_ok else "limit",
        "Alternatives survive" if alternatives_ok else "Do not smuggle the conclusion into the alternatives",
        "Explicit branch expansion is the stronger phenomenon to establish, not a cheaper explanation of the current partial evidence. Motif memory, generic iterative consistency, leakage, and unused world-model accuracy all remain live until targeted controls remove them.",
    ))
    return


@app.cell
def assessments(committed_form, mo, module):
    separation_transfer = mo.ui.text_area(label="Transfer 1 — classify evidence", placeholder="An LM predicts sentence-final embeddings from gold intermediate tokens, while eight refinement calls improve output. Classify both results.")
    rollout_transfer = mo.ui.text_area(label="Transfer 2 — preregister rollout", placeholder="Specify generated actions, exact state/value targets, illegal-prefix handling, baselines, grouping, and endpoint.")
    alternative_transfer = mo.ui.text_area(label="Transfer 3 — alternatives", placeholder="Give three non-planning mechanisms consistent with future-token decodability and one causal discriminator.")
    assessment_form = committed_form(
        mo,
        {"separation": separation_transfer, "rollout": rollout_transfer, "alternatives": alternative_transfer},
        submit_label="Commit responses and reveal lookahead rubric",
        min_words=5,
    )
    mo.vstack([mo.md(f"## Unseen transfer assessment\n\n{module.transfer_assessment}"), assessment_form])
    return (assessment_form,)


@app.cell
def assessment_feedback(assessment_form, mo):
    mo.stop(assessment_form.value is None, mo.md("Submit all three objectives before revealing."))
    response_lengths = [len(assessment_form.value[key].split()) for key in ("separation", "rollout", "alternatives")]
    mo.md(f"""
    **Rubric (response lengths {response_lengths}):** identify conditional latent
    prediction and iterative refinement separately; replay the model's own
    generated actions; mask the first illegal action and suffix; score exact
    horizon states/value with identity and teacher-action baselines; test whether
    feedback changes selected behavior under matched compute; group by
    game/trajectory; and name motif memory, smoothing/error correction, leakage,
    or unused world modeling as alternatives.
    """)
    return


@app.cell
def close(bundle, callout, mo, resource_card):
    mo.vstack([
        mo.Html(callout("limit", "More refinement is not automatically better", "In the retained 256-pair confirmation, one DFM pass was both stronger and faster than eight for the named checkpoint. The correct result is checkpoint-specific p1>p8—not a p8 gain and not evidence against every future refinement recipe.")),
        mo.Html(callout("limit", "What this does not show", "The source-backed interface mismatch, conditional future decodability, and checkpoint-specific p1/p8 result do not establish predicted-state feedback, value-guided choice, candidate comparison, or explicit search. Low teacher-action JEPA error remains a hypothetical pattern unless its exact metric artifact is bound.")),
        mo.md("""
        ## Retrieval before the advanced track

        1. Which future quantities are teacher-conditioned in the retained JEPA assay?
        2. Why are all horizons after an illegal action masked?
        3. What matched-compute intervention would test JEPA use by the policy?

        **Sources:** Jenner et al. (NeurIPS 2024) combines future-move
        decodability with scoped causal evidence in Leela. The pass-count result
        is extracted from the tracked local record
        `research/analysis/hero_refinement_pass_sweep_20260728.json`, whose
        content identity is bound by the snapshot. See the
        [course source register](../SOURCES.md).
        """),
        mo.Html(resource_card(
            mode=bundle.mode.value,
            runtime="< 3 seconds for 96×8 constructed rollouts and exact board replay",
            device="CPU / NumPy + python-chess; no checkpoint or network",
            determinism="toy rollout seed 29; exact replay deterministic; NumPy float64",
            sources=f"content identity {bundle.integrity_sha256[:16]}…",
            limitations="Constructed exposure-error lab and retained probe/mismatch metadata; the required predicted-action JEPA experiment has not yet run.",
        )),
    ])
    return


if __name__ == "__main__":
    app.run()
