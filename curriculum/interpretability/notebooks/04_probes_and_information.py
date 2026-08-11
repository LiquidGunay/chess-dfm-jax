# /// script
# dependencies = ["marimo", "numpy"]
# requires-python = ">=3.11"
# ///

import marimo

__generated_with = "0.23.3"
app = marimo.App(width="full")


@app.cell
def imports():
    import sys
    from pathlib import Path

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
    from research.interpretability.course.experiments import toy_probe_experiment
    from research.interpretability.course.stats import (
        binary_accuracy,
        fit_ridge_binary,
        ridge_binary_scores,
    )
    from research.interpretability.course.ui import (
        callout,
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
        binary_accuracy,
        callout,
        claim_card_html,
        committed_form,
        fit_ridge_binary,
        get_module,
        line_chart,
        load_course_bundle,
        metric_cards,
        mo,
        mode_badge,
        module_header,
        np,
        resource_card,
        ridge_binary_scores,
        table,
        toy_probe_experiment,
    )


@app.cell
def title(get_module, mo, module_header):
    module = get_module(4)
    mo.Html(module_header(module))
    return (module,)


@app.cell
def opening(mo, module, table):
    objective_rows = [
        {"objective": objective, "assessment": assessment}
        for objective, assessment in zip(
            module.objectives,
            (
                "Derive and fit a ridge-linear classifier on grouped data.",
                "Make leakage appear, then remove it without looking at final labels.",
                "Beat simple-input, shuffled-label, and random-feature controls under a frozen capacity.",
            ),
            strict=True,
        )
    ]
    mo.vstack([
        mo.md(r"""
        ## Decodable is not necessarily used

        A probe asks whether a restricted decoder can recover a label from a
        representation. For binary ridge regression, augment each activation
        row \(x_i\) with a bias to obtain \(\tilde X=[X,\mathbf 1]\),
        encode the binary target as \(t=2y-1\), and solve

        \[
        \hat w=(\tilde X^\top\tilde X+\lambda D)^{-1}\tilde X^\top t,
        \qquad D=\operatorname{diag}(1,\ldots,1,0).
        \]

        The final zero leaves the intercept unregularized.
        High held-out accuracy shows **usable information for this decoder under
        this split**. It may reflect the input, a game fingerprint, excess probe
        capacity, or a representation the model itself never reads in that way.
        """),
        mo.Html(table(objective_rows, (("objective", "Objective"), ("assessment", "Direct evidence")))),
    ])
    return


@app.cell
def prerequisite(committed_form, mo):
    response = mo.ui.text_area(
        label="Prerequisite diagnostic — representation gauges",
        placeholder="Two encoders differ by an orthogonal feature rotation. Why can coordinate-wise cosine fall while their usable linear information is unchanged?",
    )
    prerequisite_form = committed_form(
        mo, {"gauge": response}, submit_label="Commit prerequisite response", min_words=6
    )
    prerequisite_form
    return (prerequisite_form,)


@app.cell
def prerequisite_feedback(mo, prerequisite_form):
    mo.stop(prerequisite_form.value is None, mo.md("Answer the prerequisite diagnostic to reveal the bridge."))
    mo.callout("An invertible rotation changes individual coordinates while preserving paired inner-product geometry and linearly accessible information. Probe claims must therefore name the decoder and controls, not a privileged neuron basis.", kind="info")
    return


@app.cell
def mode_controls(mo):
    evidence_mode = mo.ui.radio(
        options={
            "Snapshot — retained Raw/Hero probe curves": "snapshot",
            "Toy — constructed leakage mechanism": "toy",
            "Source — verify retained probe records": "source",
        },
        value="Snapshot — retained Raw/Hero probe curves",
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
def probe_vocabulary(mo, table):
    vocabulary_rows = [
        {"term": "feature extractor", "meaning": "frozen mapping from board/example to activation row"},
        {"term": "probe capacity", "meaning": "function class, width, regularization, and training budget of decoder"},
        {"term": "selectivity", "meaning": "target performance relative to a matched control task; not a causal metric"},
        {"term": "nested selection", "meaning": "choose layer/λ/capacity on development data, report once on held-out evaluation"},
        {"term": "MDL intuition", "meaning": "information is more accessible when labels can be encoded with a shorter online code"},
    ]
    mo.Html(table(vocabulary_rows, (("term", "Term"), ("meaning", "Operational meaning"))))
    return


@app.cell
def worked_ridge(fit_ridge_binary, np, ridge_binary_scores):
    worked_features = np.asarray([[-1.0], [1.0]])
    worked_labels = np.asarray([0, 1])
    worked_weights = fit_ridge_binary(worked_features, worked_labels, ridge=0.0)
    worked_probe_scores = ridge_binary_scores(worked_features, worked_weights)
    return worked_features, worked_labels, worked_probe_scores, worked_weights


@app.cell
def worked_ridge_view(
    mo,
    table,
    worked_features,
    worked_labels,
    worked_probe_scores,
    worked_weights,
):
    hand_rows = [
        {
            "x": float(worked_features[index, 0]),
            "y": int(worked_labels[index]),
            "t=2y−1": int(2 * worked_labels[index] - 1),
            "score": f"{worked_probe_scores[index]:+.1f}",
        }
        for index in range(worked_labels.size)
    ]
    mo.vstack([
        mo.md(r"""
        ### Hand-worked ridge fit

        With \(X=[[-1],[1]]\), \(y=[0,1]\), and \(\lambda=0\),
        \(\tilde X=[[-1,1],[1,1]]\) and \(t=[-1,1]\).
        Solving the two normal equations gives slope \(1\), intercept \(0\).
        The scores are therefore \([-1,1]\), which the zero threshold
        classifies exactly.

        ```python
        design = np.column_stack([X, np.ones(len(X))])
        D = np.diag([1., 0.])
        target = 2 * y - 1
        w = np.linalg.solve(
            design.T @ design + ridge * D,
            design.T @ target,
        )
        ```

        This closed form is the whole learner used below—there is no hidden
        optimizer. For nonzero \(\lambda\), the slope shrinks while the
        intercept remains unpenalized.
        """),
        mo.md(
            f"Computed weights `[slope, intercept]` = "
            f"`{worked_weights.round(6).tolist()}`."
        ),
        mo.Html(table(hand_rows, (("x", "x"), ("y", "y"), ("t=2y−1", "Target t"), ("score", "Probe score")))),
    ])
    return


@app.cell
def leakage_prediction(committed_form, mo):
    leakage_answer = mo.ui.radio(
        options=[
            "Randomly split rows from every game",
            "Assign whole games to fit, selection, or evaluation before feature selection",
            "Use all games to choose dimensions, then split for the final fit",
        ],
        value=None,
        label="Predict: which design tests generalization to unseen games?",
    )
    leakage_form = committed_form(
        mo, {"split_design": leakage_answer}, submit_label="Lock split prediction"
    )
    leakage_form
    return (leakage_form,)


@app.cell
def leakage_feedback(callout, leakage_form, mo):
    mo.stop(leakage_form.value is None, mo.md("Commit before the answer is revealed."))
    split_ok = leakage_form.value["split_design"] == "Assign whole games to fit, selection, or evaluation before feature selection"
    mo.Html(callout(
        "positive" if split_ok else "limit",
        "Correct independent unit" if split_ok else "Selection can leak too",
        "Rows from one game share opening, players, trajectory, and fingerprints. Group assignment must happen before layer, capacity, or feature selection—not only before the final regression fit.",
    ))
    return


@app.cell
def probe_controls(mo):
    split_strategy = mo.ui.radio(
        options={"Leaky random-row split": "row", "Whole-game split": "group"},
        value="Whole-game split",
        label="Fit/selection/evaluation split",
    )
    feature_source = mo.ui.dropdown(
        options={
            "Full representation (board + game fingerprints + noise)": "full",
            "Simple board-input baseline": "board",
            "Fixed random projection of board input": "random_projection",
        },
        value="Full representation (board + game fingerprints + noise)",
        label="Feature source",
    )
    capacity = mo.ui.number(
        1,
        32,
        step=1,
        value=16,
        label="Largest width admitted to selection grid",
    )
    target_mode = mo.ui.radio(
        options={"Real constructed label": "real", "Shuffled-label control": "shuffled"},
        value="Real constructed label",
        label="Target",
    )
    mo.vstack([
        split_strategy,
        feature_source,
        capacity,
        target_mode,
        mo.md("The selector tries widths `1, 2, 4, 8, 16, 32` up to the cap and λ in `{0.01, 0.1, 1, 10}` **only on the selection split**."),
    ])
    return capacity, feature_source, split_strategy, target_mode


@app.cell
def probe_fit(
    binary_accuracy,
    capacity,
    feature_source,
    fit_ridge_binary,
    np,
    ridge_binary_scores,
    split_strategy,
    target_mode,
    toy_probe_experiment,
):
    probe_data = toy_probe_experiment()
    if feature_source.value == "full":
        candidate_features = probe_data.features
    elif feature_source.value == "board":
        candidate_features = probe_data.board_baseline
    else:
        random_matrix = np.random.default_rng(404).normal(size=(probe_data.board_baseline.shape[1], 32))
        candidate_features = probe_data.board_baseline @ random_matrix
    fit_mask = probe_data.row_fit if split_strategy.value == "row" else probe_data.group_fit
    selection_mask = (
        probe_data.row_selection
        if split_strategy.value == "row"
        else probe_data.group_selection
    )
    evaluation_mask = probe_data.row_eval if split_strategy.value == "row" else probe_data.group_eval
    if target_mode.value == "shuffled":
        labels = probe_data.labels[np.random.default_rng(405).permutation(probe_data.labels.size)]
    else:
        labels = probe_data.labels
    width_grid = sorted({
        min(width, capacity.value, candidate_features.shape[1])
        for width in (1, 2, 4, 8, 16, 32)
        if width <= capacity.value
    } | {min(capacity.value, candidate_features.shape[1])})
    ridge_grid = (0.01, 0.1, 1.0, 10.0)

    def nested_fit(target_labels):
        candidate_rows = []
        for trial_width in width_grid:
            trial_features = candidate_features[:, :trial_width]
            for trial_ridge in ridge_grid:
                trial_weights = fit_ridge_binary(
                    trial_features[fit_mask],
                    target_labels[fit_mask],
                    ridge=trial_ridge,
                )
                trial_selection = binary_accuracy(
                    ridge_binary_scores(trial_features[selection_mask], trial_weights),
                    target_labels[selection_mask],
                )
                candidate_rows.append(
                    (trial_selection, -trial_width, trial_ridge, trial_width)
                )
        best_selection, _, selected_ridge, selected_width = max(candidate_rows)
        refit_mask = fit_mask | selection_mask
        final_features = candidate_features[:, :selected_width]
        final_weights = fit_ridge_binary(
            final_features[refit_mask],
            target_labels[refit_mask],
            ridge=selected_ridge,
        )
        development_accuracy = binary_accuracy(
            ridge_binary_scores(final_features[refit_mask], final_weights),
            target_labels[refit_mask],
        )
        final_accuracy = binary_accuracy(
            ridge_binary_scores(final_features[evaluation_mask], final_weights),
            target_labels[evaluation_mask],
        )
        return (
            development_accuracy,
            final_accuracy,
            best_selection,
            selected_ridge,
            selected_width,
        )

    (
        development_accuracy,
        evaluation_accuracy,
        selection_accuracy,
        selected_ridge,
        used_width,
    ) = nested_fit(labels)

    shuffled_labels = probe_data.labels[np.random.default_rng(406).permutation(probe_data.labels.size)]
    _, shuffled_accuracy, _, _, _ = nested_fit(shuffled_labels)
    majority_accuracy = max(float(np.mean(labels[evaluation_mask])), 1.0 - float(np.mean(labels[evaluation_mask])))
    fit_games = set(probe_data.games[fit_mask].tolist())
    selection_games_set = set(probe_data.games[selection_mask].tolist())
    evaluation_games_set = set(probe_data.games[evaluation_mask].tolist())
    train_games = len(fit_games)
    selection_games = len(selection_games_set)
    evaluation_games = len(evaluation_games_set)
    overlapping_games = len(
        (fit_games & selection_games_set)
        | (fit_games & evaluation_games_set)
        | (selection_games_set & evaluation_games_set)
    )
    return (
        development_accuracy,
        evaluation_accuracy,
        evaluation_games,
        majority_accuracy,
        overlapping_games,
        selected_ridge,
        selection_accuracy,
        selection_games,
        shuffled_accuracy,
        train_games,
        used_width,
    )


@app.cell
def probe_results(
    development_accuracy,
    evaluation_accuracy,
    evaluation_games,
    feature_source,
    majority_accuracy,
    metric_cards,
    mo,
    overlapping_games,
    selected_ridge,
    selection_accuracy,
    selection_games,
    shuffled_accuracy,
    split_strategy,
    target_mode,
    train_games,
    used_width,
):
    mo.vstack([
        mo.md("## Interaction: leakage × capacity × control task"),
        mo.Html(metric_cards([
            (f"{development_accuracy:.1%}", "refit development accuracy"),
            (f"{selection_accuracy:.1%}", "winning selection accuracy"),
            (f"{evaluation_accuracy:.1%}", "final evaluation accuracy"),
            (f"{majority_accuracy:.1%}", "majority baseline"),
            (f"{shuffled_accuracy:.1%}", "matched shuffled-label score"),
            (used_width, "probe input dimensions"),
            (selected_ridge, "selected λ"),
            (overlapping_games, "games crossing split roles"),
        ])),
        mo.md(f"Selected **{split_strategy.value}** split, **{feature_source.value}** features, and **{target_mode.value}** labels. Fit/selection/evaluation contain {train_games}/{selection_games}/{evaluation_games} distinct games. Width and λ are chosen on selection, then refit on fit+selection and scored once on evaluation. In this interactive teaching lab you can deliberately reuse evaluation by changing controls; that behavior is a demonstration, not a valid confirmatory workflow."),
    ])
    return


@app.cell
def control_matrix(mo, table):
    control_rows = [
        {"control": "simple board/input", "detects": "label is directly available without learned representation", "failure meaning": "probe does not isolate learned abstraction"},
        {"control": "shuffled labels", "detects": "memorization, leakage, or selection overfit", "failure meaning": "assay can fit arbitrary targets"},
        {"control": "untrained / independent model", "detects": "architecture and random features alone", "failure meaning": "trained-model specificity is weak"},
        {"control": "capacity and λ sweep", "detects": "performance bought by decoder complexity", "failure meaning": "accessibility claim depends on probe"},
        {"control": "grouped nested split", "detects": "shared entities and hyperparameter reuse", "failure meaning": "reported generalization target was wrong"},
    ]
    mo.vstack([
        mo.md("## Controls answer different counterfactuals"),
        mo.Html(table(control_rows, (("control", "Control"), ("detects", "Counterfactual"), ("failure meaning", "If the control also succeeds")))),
    ])
    return


@app.cell
def real_probe_panel(
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
        probes = bundle.section("probes")
        legal_metrics = [row for row in probes["concept_metrics"] if row["concept"] == "Legal destination"]
        probe_series = []
        for model, color in (("Raw BT4", "#3366a3"), ("Hero", "#a95f19")):
            probe_series.append({"name": model, "color": color, "points": [(row["layer"], row["score"]) for row in legal_metrics if row["model"] == model]})
        split_rows = [{"role": key, "positions": value} for key, value in probes["split_counts"].items()]
        steering_resolved = sum(bool(row["resolved"]) for row in probes["steering"])
        gates = probes["correctness_gates"]
        probe_card = ClaimCard(
            operation=EvidenceOperation.PREDICTION,
            target=TargetLevel.REPRESENTATION,
            scope="retained development splits for Raw BT4 and Hero concept probes",
            endpoint=Endpoint.PROBE,
            estimand="held-out legal-destination AUROC by layer under a fixed linear probe protocol",
            unit="square-token label within a position",
            grouping="game-disjoint nested fit, selection, and evaluation pools",
            intervention=None,
            controls=("label permutation", "train-only frequency/mean constant baseline", "group-disjoint selection"),
            assumptions=("operational labels match the intended chess concept", "probe class and split were frozen before evaluation"),
            uncertainty="retained paired intervals for model deltas; curve remains exploratory",
            multiplicity="15-layer scan; layer selection belongs to development data",
            allowed="Linear probes predict legal-destination labels well from both retained Raw and Hero representations under the recorded grouped protocol.",
            excluded="Probe performance does not show that either policy reads the representation through the fitted direction.",
            falsifier="A group-disjoint replication where the primary score does not beat the recorded permutation/constant baselines, or a stronger future raw-input baseline matches it.",
            status=ResultStatus.POSITIVE,
        )
        real_probe_view = mo.vstack([
            mo.md("## Frozen Raw/Hero probe evidence"),
            mo.Html(line_chart(probe_series, x_label="BT4 layer", y_label="legal-destination AUROC", y_domain=(0.5, 1.0), alt_text="Raw and Hero legal-destination linear probe AUROC is high across all fifteen layers.")),
            mo.Html(table(split_rows, (("role", "Split role"), ("positions", "Positions")))),
            mo.Html(claim_card_html(probe_card)),
            mo.Html(callout("positive", "Correctness gates", f"Group-disjoint nested splits passed: {gates['group_disjoint_nested_splits']}; the frozen test split was opened: {gates['test_split_evaluated']}.")),
            mo.Html(callout("limit", "Controls taught versus controls retained", "The frozen Raw/Hero run records label permutation and a train-only frequency/mean constant baseline. The simple raw-board-input and untrained-feature controls used in the teaching lab were not run in this retained experiment, so they are proposed replication requirements—not completed evidence.")),
            mo.Html(callout("limit", "Causal-use check is weak", f"Only {steering_resolved} of {len(probes['steering'])} retained probe-direction policy intervals excludes zero. A separate matched random-direction diagnostic also produced four exclusions, so the single directional result is not clean evidence of use.")),
        ])
    else:
        real_probe_view = mo.Html(callout("neutral", "Toy boundary", "Toy mode teaches a known fingerprint leak. It cannot support a claim about Raw or Hero decodability."))
    real_probe_view
    return


@app.cell
def causal_language_exercise(committed_form, mo):
    conclusion = mo.ui.radio(
        options=[
            "The model uses legal-destination features to choose moves",
            "A fixed linear decoder predicts legal-destination labels from the sampled representations",
            "The representation contains the complete legal-move algorithm",
        ],
        value=None,
        label="Strongest probe-only conclusion",
    )
    conclusion_form = committed_form(
        mo, {"probe_conclusion": conclusion}, submit_label="Commit conclusion"
    )
    conclusion_form
    return (conclusion_form,)


@app.cell
def conclusion_feedback(callout, conclusion_form, mo):
    mo.stop(conclusion_form.value is None, mo.md("Commit before revealing."))
    conclusion_ok = conclusion_form.value["probe_conclusion"].startswith("A fixed linear decoder predicts")
    mo.Html(callout(
        "positive" if conclusion_ok else "danger",
        "Prediction claim" if conclusion_ok else "Probe-to-use leap",
        "To test use, intervene on a representation while preserving matched norms and plausible states, then measure a model endpoint against random and semantic-placebo controls.",
    ))
    return


@app.cell
def mdl_intuition(mo):
    mo.md(r"""
    ## MDL intuition without magic

    Accuracy asks whether a probe eventually learns the label. Online minimum
    description length asks how many bits it costs to transmit labels as the
    probe sees progressively more data. A representation that supports a short
    code makes the pattern accessible sooner. MDL still depends on the decoder,
    training schedule, block sizes, and grouping; it does not convert a
    prediction result into a causal one.
    """)
    return


@app.cell
def assessments(committed_form, mo, module):
    fit_transfer = mo.ui.text_area(label="Transfer 1 — derive and fit", placeholder="For X=[[-1],[1]], y=[0,1], λ=0, solve the slope/intercept; then write the fit/selection/evaluation order for a layerwise probe.")
    leakage_transfer = mo.ui.text_area(label="Transfer 2 — leakage", placeholder="Eight paraphrases share one source document. Diagnose a random-row split and repair it.")
    control_transfer = mo.ui.text_area(label="Transfer 3 — controls", placeholder="A 3-layer MLP probe gets 99%. Design simple-input, shuffle, untrained, and capacity controls; state the allowed conclusion.")
    assessment_form = committed_form(
        mo,
        {"fit": fit_transfer, "leakage": leakage_transfer, "controls": control_transfer},
        submit_label="Commit responses and reveal probe rubric",
        min_words=5,
    )
    mo.vstack([mo.md(f"## Unseen transfer assessment\n\n{module.transfer_assessment}"), assessment_form])
    return (assessment_form,)


@app.cell
def assessment_feedback(assessment_form, mo):
    mo.stop(assessment_form.value is None, mo.md("Submit every objective before revealing."))
    response_lengths = [len(assessment_form.value[key].split()) for key in ("fit", "leakage", "controls")]
    mo.md(f"""
    **Rubric (response lengths {response_lengths}):** the worked weights are
    slope 1 and intercept 0. Split documents first; fit probes on training
    documents; choose layer/λ/capacity on selection
    documents; evaluate once. Compare a bag-of-input baseline, permuted labels,
    an untrained/independent model, and matched-capacity probes. The conclusion
    remains predictive and names the decoder, sample, and metric.
    """)
    return


@app.cell
def close(bundle, callout, mo, resource_card):
    mo.vstack([
        mo.Html(callout("positive", "Positive result", "Grouped probes can establish that a simple decoder generalizes to unseen games when it clears the right controls.")),
        mo.Html(callout("limit", "What this does not show", "Decodability is not behavioral use, necessity, sufficiency, or a complete algorithm. A null steering pilot may also be underpowered rather than evidence of absence.")),
        mo.md("""
        ## Retrieval before module 05

        1. Why does fitting a probe after a row split leak game identity?
        2. What is the distinction between selectivity and causal use?
        3. Where may layer and capacity be selected?

        **Sources:** Hewitt & Liang (EMNLP-IJCNLP 2019), Voita & Titov (EMNLP
        2020), and Karvonen (COLM 2024). See the
        [course source register](../SOURCES.md) for status and links.
        """),
        mo.Html(resource_card(
            mode=bundle.mode.value,
            runtime="< 2 seconds for 192 rows and closed-form ridge fits",
            device="CPU / NumPy; no checkpoint or network",
            determinism="toy seed 17; control seeds 404/405/406; NumPy float64",
            sources=f"content identity {bundle.integrity_sha256[:16]}…",
            limitations="Constructed leakage mechanism and frozen probe summaries; no new probe fit on Raw/Hero activations.",
        )),
    ])
    return


if __name__ == "__main__":
    app.run()
