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

    import numpy as np
    import marimo as mo

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
        classify_language,
    )
    from research.interpretability.course.experiments import (
        toy_paired_outcomes,
        toy_probe_experiment,
    )
    from research.interpretability.course.stats import (
        binary_accuracy,
        fit_ridge_binary,
        paired_cluster_bootstrap,
        ridge_binary_scores,
    )
    from research.interpretability.course.ui import (
        callout,
        claim_card_html,
        committed_form,
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
        classify_language,
        committed_form,
        fit_ridge_binary,
        get_module,
        load_course_bundle,
        metric_cards,
        mo,
        mode_badge,
        module_header,
        np,
        paired_cluster_bootstrap,
        resource_card,
        ridge_binary_scores,
        table,
        toy_paired_outcomes,
        toy_probe_experiment,
    )


@app.cell
def title(get_module, mo, module_header):
    module = get_module(0)
    mo.Html(module_header(module))
    return (module,)


@app.cell
def opening(mo):
    mo.md(r"""
    ## Why begin here?

    Interpretability is not a property a plot either has or lacks. It is a
    relationship between an operation, a target, a population, and an
    endpoint. Before opening a model, we need enough experimental language
    to tell these sentences apart:

    1. *An activation is larger on knight squares.*
    2. *A linear function of the residual predicts knight squares.*
    3. *Replacing an activation changes a selected move distribution.*
    4. *A path implements a knight-related subcomputation.*

    They require progressively different experiments, but they are not a
    single ladder. Behavior can be observed or intervened on; activation
    evidence can be predictive or causal. We will track four axes instead.
    """)
    return


@app.cell
def mode_controls(mo):
    evidence_mode = mo.ui.radio(
        options={
            "Snapshot — frozen real figures": "snapshot",
            "Toy — constructed illustration": "toy",
            "Source — verify and re-extract": "source",
        },
        value="Snapshot — frozen real figures",
        label="Evidence mode",
    )
    verify_source = mo.ui.run_button(
        label="Verify retained sources",
        kind="warn",
        tooltip="Hashes 18 retained files and recomputes the compact extraction.",
    )
    mo.hstack([evidence_mode, verify_source], justify="start", gap=1.2)
    return evidence_mode, verify_source


@app.cell
def load_mode(evidence_mode, load_course_bundle, mo, verify_source):
    selected_mode = evidence_mode.value
    mo.stop(
        selected_mode == "source" and not verify_source.value,
        mo.callout(
            "Source mode is explicit and potentially slower. Press **Verify retained sources**; "
            "no snapshot result is silently substituted.",
            kind="warn",
        ),
    )
    bundle = load_course_bundle(selected_mode)
    return bundle, selected_mode


@app.cell
def mode_view(bundle, mo, mode_badge, selected_mode):
    mo.Html(mode_badge(selected_mode, bundle.evidence_scope))
    return


@app.cell
def claim_matrix(mo, table):
    claim_axes = [
        {
            "axis": "Evidence operation",
            "values": "observation · prediction · intervention · interchange",
            "question": "What did we do to obtain evidence?",
        },
        {
            "axis": "Target level",
            "values": "behavior · activation · representation · component · circuit",
            "question": "What object does the claim concern?",
        },
        {
            "axis": "Scope",
            "values": "example · sampled population · task family · model family",
            "question": "Where could this conclusion generalize?",
        },
        {
            "axis": "Endpoint",
            "values": "representation metric · probe · logit · legal move · policy · value · arena",
            "question": "What outcome actually moved?",
        },
    ]
    mo.vstack(
        [
            mo.md("## The four-axis claim matrix"),
            mo.Html(table(claim_axes, (("axis", "Axis"), ("values", "Values"), ("question", "Question")))),
            mo.md(
                r"""
                The **estimand** is the exact quantity we intend to estimate.
                For a paired policy comparison it might be

                \[
                \theta = \mathbb{E}_{g\sim\mathcal G}
                \left[\frac{1}{|g|}\sum_{i\in g}
                JS\!\left(p_{\text{Hero},i},p_{\text{Raw},i}\right)\right].
                \]

                This equation declares that games, not positions, define the
                sampling population. Changing the outer expectation changes
                the scientific question—not just the error bar.
                """
            ),
        ]
    )
    return


@app.cell
def prerequisite(committed_form, mo):
    diagnostic_statement = mo.ui.dropdown(
        options={
            "A probe predicts pins with 97% held-out accuracy.": "probe",
            "A layer-14 patch changes policy JS on eight positions.": "patch",
            "Hero wins more games in a paired arena.": "arena",
        },
        value=None,
        label="Classify this result",
    )
    diagnostic_operation = mo.ui.dropdown(
        ["observation", "prediction", "intervention", "interchange / causal abstraction"],
        value=None,
        label="Evidence operation",
    )
    diagnostic_target = mo.ui.dropdown(
        ["behavior", "activation", "representation", "component", "path / circuit"],
        value=None,
        label="Target level",
    )
    diagnostic_form = committed_form(
        mo,
        {
            "statement": diagnostic_statement,
            "operation": diagnostic_operation,
            "target": diagnostic_target,
        },
        submit_label="Commit classification",
    )
    diagnostic_form
    return (diagnostic_form,)


@app.cell
def diagnostic_result(callout, diagnostic_form, mo):
    answer_key = {
        "probe": ("prediction", "representation"),
        "patch": ("intervention", "component"),
        "arena": ("observation", "behavior"),
    }
    mo.stop(diagnostic_form.value is None, mo.md("Commit a complete classification before feedback is revealed."))
    selected_answer = (diagnostic_form.value["operation"], diagnostic_form.value["target"])
    expected_answer = answer_key[diagnostic_form.value["statement"]]
    correct_answer = selected_answer == expected_answer
    mo.Html(
        callout(
            "positive" if correct_answer else "limit",
            "Classification matches" if correct_answer else "Revise one axis",
            (
                f"Expected **{expected_answer[0]} × {expected_answer[1]}**. "
                "The metric endpoint still needs to be named before this becomes a complete claim."
            ),
        )
    )
    return


@app.cell
def grouping_controls(mo):
    grouping_unit = mo.ui.radio(
        options={
            "Position population — equal weight per position": "position",
            "Game population — equal weight per game": "game",
        },
        value="Game population — equal weight per game",
        label="Sampling target and bootstrap unit",
    )
    bootstrap_draws = mo.ui.number(
        200, 5000, step=200, value=2000, label="Bootstrap resamples"
    )
    mo.hstack([grouping_unit, bootstrap_draws], justify="start", gap=1.2)
    return bootstrap_draws, grouping_unit


@app.cell
def grouping_experiment(
    bootstrap_draws,
    grouping_unit,
    np,
    paired_cluster_bootstrap,
    toy_paired_outcomes,
):
    paired_first, paired_second, paired_games = toy_paired_outcomes()
    bootstrap_groups = (
        np.arange(paired_first.size) if grouping_unit.value == "position" else paired_games
    )
    paired_interval = paired_cluster_bootstrap(
        paired_first,
        paired_second,
        bootstrap_groups,
        weighting="observation" if grouping_unit.value == "position" else "equal_group",
        resamples=bootstrap_draws.value,
        seed=20260811,
    )
    return paired_games, paired_interval


@app.cell
def grouping_view(
    grouping_unit,
    metric_cards,
    mo,
    paired_games,
    paired_interval,
):
    mo.vstack(
        [
            mo.md("## Interaction: the unit changes the question"),
            mo.Html(
                metric_cards(
                    [
                        (
                            f"{paired_interval.estimate:+.5f}",
                            "position-weighted mean"
                            if paired_interval.weighting == "observation"
                            else "equal-game mean",
                        ),
                        (
                            f"[{paired_interval.lower:+.3f}, {paired_interval.upper:+.3f}]",
                            "percentile interval",
                        ),
                        (paired_interval.group_count, "resampled units"),
                        (len(set(paired_games.tolist())), "actual games"),
                    ]
                )
            ),
            mo.md(
                f"You selected the **{grouping_unit.value}** target. Position mode estimates "
                "the mean over rows and resamples rows. Game mode first computes one mean per "
                "game, then estimates and resamples the equal-game mean. Because this toy has "
                "unequal game lengths, the point estimates differ as well as the intervals. "
                "A narrower interval is not automatically a better interval."
            ),
        ]
    )
    return


@app.cell
def failure_lab_controls(committed_form, mo):
    leakage_split = mo.ui.radio(
        options={
            "Split rows; the same games can occur on both sides": "row",
            "Split whole games before fitting": "game",
        },
        value=None,
        label="Entity split",
    )
    leakage_prediction = mo.ui.radio(
        options=[
            "Game fingerprints will inflate held-out-row accuracy",
            "A row split removes every game-level shortcut",
            "Grouping changes only the plot, not the estimand",
        ],
        value=None,
        label="Predict the hidden confounder",
    )
    comparison_structure = mo.ui.dropdown(
        options={
            "Both models score the identical positions": "paired",
            "Each model scores a different independent cohort": "unpaired",
        },
        value=None,
        label="Comparison structure",
    )
    scan_count = mo.ui.number(1, 40, step=1, value=20, label="Exploratory hypotheses scanned")
    failure_form = committed_form(
        mo,
        {
            "split": leakage_split,
            "prediction": leakage_prediction,
            "comparison": comparison_structure,
            "scan_count": scan_count,
        },
        submit_label="Commit design choices and reveal failure lab",
    )
    failure_form
    return (failure_form,)


@app.cell
def failure_lab_experiment(
    binary_accuracy,
    failure_form,
    fit_ridge_binary,
    mo,
    ridge_binary_scores,
    toy_probe_experiment,
):
    mo.stop(failure_form.value is None, mo.md("Commit all design choices before computing."))
    leakage_data = toy_probe_experiment()
    if failure_form.value["split"] == "row":
        leakage_fit = leakage_data.row_fit
        leakage_eval = leakage_data.row_eval
    else:
        leakage_fit = leakage_data.group_fit
        leakage_eval = leakage_data.group_eval
    full_weights = fit_ridge_binary(
        leakage_data.features[leakage_fit], leakage_data.labels[leakage_fit], ridge=1.0
    )
    board_weights = fit_ridge_binary(
        leakage_data.board_baseline[leakage_fit],
        leakage_data.labels[leakage_fit],
        ridge=1.0,
    )
    full_accuracy = binary_accuracy(
        ridge_binary_scores(leakage_data.features[leakage_eval], full_weights),
        leakage_data.labels[leakage_eval],
    )
    board_accuracy = binary_accuracy(
        ridge_binary_scores(leakage_data.board_baseline[leakage_eval], board_weights),
        leakage_data.labels[leakage_eval],
    )
    family_size = int(failure_form.value["scan_count"])
    independent_family_error = 1.0 - (1.0 - 0.05) ** family_size
    bonferroni_threshold = 0.05 / family_size
    return (
        board_accuracy,
        bonferroni_threshold,
        family_size,
        full_accuracy,
        independent_family_error,
    )


@app.cell
def failure_lab_view(
    board_accuracy,
    bonferroni_threshold,
    callout,
    failure_form,
    family_size,
    full_accuracy,
    independent_family_error,
    metric_cards,
    mo,
    table,
):
    pairing_row = (
        {
            "design": "paired",
            "estimand": "mean within-position model difference",
            "uncertainty": "resample the independent game/root while preserving model pairs",
        }
        if failure_form.value["comparison"] == "paired"
        else {
            "design": "unpaired",
            "estimand": "difference between two cohort means",
            "uncertainty": "resample each independent cohort; no invented row pairing",
        }
    )
    prediction_ok = failure_form.value["prediction"].startswith("Game fingerprints")
    mo.vstack([
        mo.md("## Failure lab: leakage, pairing, and multiplicity change the claim"),
        mo.Html(metric_cards([
            (f"{full_accuracy:.1%}", "probe with hidden game fingerprints"),
            (f"{board_accuracy:.1%}", "simple board-only baseline"),
            (f"{independent_family_error:.1%}", f"P(any false positive), {family_size} independent null scans"),
            (f"{bonferroni_threshold:.4f}", "Bonferroni per-test threshold at family α=.05"),
        ])),
        mo.Html(table([pairing_row], (("design", "Comparison"), ("estimand", "Estimand"), ("uncertainty", "Valid uncertainty")))),
        mo.Html(callout(
            "positive" if prediction_ok else "limit",
            "Hidden confounder identified" if prediction_ok else "The shortcut was game identity",
            (
                "The constructed representation appends one-hot game fingerprints. A row split "
                "lets the decoder see the same games during fit and evaluation; a whole-game split "
                "removes that route. Repeated openings create the analogous risk even without exact "
                "duplicate rows. The family-error calculation assumes independent null tests and is "
                "therefore a teaching bound, not a substitute for the actual dependence structure."
            ),
        )),
    ])
    return


@app.cell
def preregistration_exercise(committed_form, mo):
    promotion_rule = mo.ui.text_area(
        label="Freeze a numerical promotion criterion",
        placeholder="Name direction, endpoint, population, interval/margin, controls, and multiplicity rule before outcomes.",
    )
    falsification_rule = mo.ui.text_area(
        label="Freeze a falsifier",
        placeholder="Name a result that would reject or materially narrow this exact claim while diagnostics remain valid.",
    )
    preregistration_form = committed_form(
        mo,
        {"promotion": promotion_rule, "falsifier": falsification_rule},
        submit_label="Commit criterion and falsifier",
        min_words=10,
    )
    preregistration_form
    return (preregistration_form,)


@app.cell
def preregistration_feedback(mo, preregistration_form):
    mo.stop(preregistration_form.value is None, mo.md("Commit both rules before revealing the audit."))
    preregistration_lengths = {
        key: len(preregistration_form.value[key].split())
        for key in ("promotion", "falsifier")
    }
    mo.md(f"""
    **Preregistration audit (response lengths {preregistration_lengths}):** a usable
    criterion fixes sign, endpoint, sampling population, grouping, uncertainty
    or equivalence margin, required controls, and scan correction. A falsifier
    names a possible result that changes the claim; “the experiment crashes” or
    “a different metric looks better” does not count.
    """)
    return


@app.cell
def real_design(
    ClaimCard,
    Endpoint,
    EvidenceOperation,
    ResultStatus,
    TargetLevel,
    bundle,
    claim_card_html,
    mo,
):
    if bundle.is_empirical:
        design = bundle.section("design")
        tuning = design["tuning_validation"]
        full = design["full_validation"]
        design_claim = ClaimCard(
            operation=EvidenceOperation.OBSERVATION,
            target=TargetLevel.BEHAVIOR,
            scope="retained Proposal A development and validation records",
            endpoint=Endpoint.POLICY,
            estimand="position and distinct-game counts in each recorded validation pool",
            unit="position with game identifier",
            grouping="game",
            intervention=None,
            controls=("whole-game split audit", "cross-split exact-game duplicate audit"),
            assumptions=("recorded game IDs identify dependence clusters",),
            uncertainty="descriptive counts; no population interval",
            multiplicity="not applicable",
            allowed=(
                f"The tuning record contains {tuning['positions']} positions from {tuning['games']} "
                f"games; the full record contains {full['positions']} positions from {full['games']} games."
            ),
            excluded="The position counts are not independent sample sizes or evidence of unseen positions.",
            falsifier="A content audit showing game-ID drift or cross-split duplicate games.",
            status=ResultStatus.DESCRIPTIVE,
        )
        design_view = mo.vstack(
            [
                mo.md("## The real validation-grain warning"),
                mo.Html(claim_card_html(design_claim)),
            ]
        )
    else:
        design_view = mo.Html(
            '<div class="ci-callout ci-neutral"><strong>Toy boundary.</strong> '
            "Constructed grouping behavior cannot support a Raw/Hero conclusion.</div>"
        )
    design_view
    return


@app.cell
def status_controls(mo):
    status_scenario = mo.ui.dropdown(
        options={
            "95% interval crosses zero": "unresolved",
            "Random-label probe scores at chance": "negative_control",
            "Effect lies inside a frozen ±0.01 equivalence margin": "equivalent",
            "Primary effect is opposite the preregistered direction": "falsified",
        },
        value="95% interval crosses zero",
        label="Result to report",
    )
    status_wording = mo.ui.text(
        value="The models are the same.",
        label="Draft one sentence",
        placeholder="Draft one sentence",
        full_width=True,
    )
    mo.vstack([status_scenario, status_wording])
    return status_scenario, status_wording


@app.cell
def status_feedback(
    classify_language,
    mo,
    status_scenario,
    status_wording,
    table,
):
    status_key = {
        "unresolved": "Failure to reject is not equivalence.",
        "negative_control": "A passing negative control validates one aspect of the assay.",
        "equivalent": "Equivalence needs a frozen margin and appropriate interval test.",
        "falsified": "A preregistered directional failure may falsify the scoped hypothesis.",
    }
    language_flags = classify_language(status_wording.value)
    mo.vstack(
        [
            mo.md("## Reporting negative evidence precisely"),
            mo.Html(
                table(
                    [
                        {"check": "Scenario reading", "result": status_key[status_scenario.value]},
                        {"check": "Causal wording present", "result": language_flags["contains_causal_language"]},
                        {"check": "Uncertainty stated", "result": language_flags["states_uncertainty"]},
                    ],
                    (("check", "Check"), ("result", "Result")),
                )
            ),
        ]
    )
    return


@app.cell
def transfer_assessment(committed_form, mo, module):
    transfer_response = mo.ui.text_area(
        label="Unseen transfer: audit a language-model feature claim",
        placeholder=(
            "A linear probe predicts truthfulness at 92%, therefore the model uses a truthfulness "
            "circuit. Name the operation, target, grouping risk, missing control, and falsifier."
        ),
        full_width=True,
    )
    status_response = mo.ui.text_area(
        label="Unseen transfer: classify two negative results",
        placeholder=(
            "A preregistered directional intervention interval crosses zero; a random-label "
            "control scores at chance; no equivalence margin was frozen. Classify the "
            "intervention and control separately, then say what is not licensed."
        ),
        full_width=True,
    )
    assessment_form = committed_form(
        mo,
        {"claim_audit": transfer_response, "status_audit": status_response},
        submit_label="Commit response and show rubric",
        min_words=16,
    )
    mo.vstack([mo.md(f"**Assessment:** {module.transfer_assessment}"), assessment_form])
    return (assessment_form,)


@app.cell
def assessment_feedback(assessment_form, mo):
    mo.stop(assessment_form.value is None, mo.md("Submit a response before revealing the rubric."))
    response_lengths = {
        key: len(assessment_form.value[key].split())
        for key in ("claim_audit", "status_audit")
    }
    mo.md(
        f"""
        **Rubric (response lengths {response_lengths}):** full credit names
        prediction × representation, the example/population scope, an entity- or
        document-level grouping risk, a simple-input or shuffled-label/capacity
        control, and an intervention-based falsifier. It explicitly rejects the
        word *therefore* and does not promise that an intervention would reveal
        a complete circuit. For the second case it reports the intervention as
        **unresolved**, the chance result as a **passing negative control**, and
        does not call either result **equivalent** because no meaningful margin
        was frozen. A directional result is **falsified** only when its frozen
        rejection rule—not merely a zero-crossing interval—says so.
        """
    )
    return


@app.cell
def close(bundle, mo, resource_card):
    mo.vstack(
        [
            mo.md(
                """
                ## Retrieval before module 01

                1. Why can a causal activation claim have a behavioral endpoint?
                2. What changes when positions rather than games are resampled?
                3. What extra evidence is required to move from a component
                   intervention to a circuit claim?

                **Sources:** Doshi-Velez & Kim (position paper/preprint) and
                Lipton (workshop/preprint lineage). See the
                [course source register](../SOURCES.md).
                """
            ),
            mo.Html(
                resource_card(
                    mode=bundle.mode.value,
                    runtime="< 2 seconds in toy/snapshot; source hashing is disk-bound",
                    device="CPU; no model or network",
                    determinism="toy seed 7; bootstrap seed 20260811; NumPy float64",
                    sources=f"snapshot {bundle.integrity_sha256[:16]}…",
                    limitations="Development artifacts teach design; they do not create new evidence.",
                )
            ),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
