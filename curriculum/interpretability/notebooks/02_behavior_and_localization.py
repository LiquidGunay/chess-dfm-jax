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
    from research.interpretability.course.experiments import toy_policy_experiment
    from research.interpretability.course.stats import (
        js_divergence,
        paired_cluster_bootstrap,
        stable_softmax,
        total_variation,
    )
    from research.interpretability.course.ui import (
        callout,
        chessboard,
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
        callout,
        chessboard,
        claim_card_html,
        committed_form,
        get_module,
        js_divergence,
        load_course_bundle,
        metric_cards,
        mo,
        mode_badge,
        module_header,
        np,
        paired_cluster_bootstrap,
        resource_card,
        stable_softmax,
        table,
        total_variation,
        toy_policy_experiment,
    )


@app.cell
def title(get_module, mo, module_header):
    module = get_module(2)
    mo.Html(module_header(module))
    return (module,)


@app.cell
def opening(mo, module, table):
    objective_rows = [
        {"objective": objective, "check": check}
        for objective, check in zip(
            module.objectives,
            (
                "Compute JS, TV, top-1/top-3 agreement, target NLL, Brier score, and binned ECE on paired policies.",
                "Explain why a whole-game interval answers a different question from a row interval.",
                "Name at least two mechanisms consistent with the same hybrid lattice.",
            ),
            strict=True,
        )
    ]
    mo.vstack([
        mo.md(r"""
        ## Measure the delta before naming a mechanism

        Model A and model B can have the same aggregate accuracy while making
        different moves, or the same top move while redistributing probability
        mass. We therefore begin with paired policies on identical positions.

        `RR`, `RH`, `HR`, and `HH` form a 2×2 lattice. The first letter selects
        the encoder (Raw or Hero); the second selects the native output head.
        A swap is a controlled module-level localization experiment. It cannot
        identify a unique neuron, feature, or algorithm inside that module.
        """),
        mo.Html(table(objective_rows, (("objective", "Observable objective"), ("check", "Mastery check")))),
    ])
    return


@app.cell
def prerequisite(committed_form, mo):
    response = mo.ui.text_area(
        label="Prerequisite diagnostic — policy comparison",
        placeholder="Two positions come from the same game. Why are they not two independent experimental units, and what should be resampled?",
    )
    prerequisite_form = committed_form(
        mo, {"grouping": response}, submit_label="Commit prerequisite response", min_words=6
    )
    prerequisite_form
    return (prerequisite_form,)


@app.cell
def prerequisite_feedback(mo, prerequisite_form):
    mo.stop(prerequisite_form.value is None, mo.md("Answer the prerequisite diagnostic to reveal the bridge."))
    mo.callout("Positions in one game share a trajectory, opening, and players. Keep paired model outcomes at the position level, but estimate uncertainty by resampling the game/root groups that define the sampling population.", kind="info")
    return


@app.cell
def mode_controls(mo):
    evidence_mode = mo.ui.radio(
        options={
            "Snapshot — retained Raw/Hero hybrid pilot": "snapshot",
            "Toy — constructed encoder-dominant lattice": "toy",
            "Source — verify retained hybrid records": "source",
        },
        value="Snapshot — retained Raw/Hero hybrid pilot",
        label="Evidence mode",
    )
    verify_source = mo.ui.run_button(label="Verify retained sources", kind="warn")
    mo.hstack([evidence_mode, verify_source], justify="start", gap=1)
    return evidence_mode, verify_source


@app.cell
def load_mode(evidence_mode, load_course_bundle, mo, verify_source):
    selected_mode = evidence_mode.value
    mo.stop(selected_mode == "source" and not verify_source.value, mo.callout("Source mode requires an explicit verification run.", kind="warn"))
    bundle = load_course_bundle(selected_mode)
    return bundle, selected_mode


@app.cell
def mode_view(bundle, mo, mode_badge, selected_mode):
    mo.Html(mode_badge(selected_mode, bundle.evidence_scope))
    return


@app.cell
def metric_primer(mo, table):
    primer_rows = [
        {"metric": "top-1 agreement", "sees": "whether argmax actions match", "misses": "all subleading mass"},
        {"metric": "top-k overlap", "sees": "mean |Top-k(p)∩Top-k(q)|/k", "misses": "ordering and probability inside the selected sets"},
        {"metric": "target NLL", "sees": "−log probability of a recorded target", "misses": "quality of alternatives"},
        {"metric": "Brier / calibration error", "sees": "probability accuracy against a target and confidence-frequency alignment", "misses": "whether the target is uniquely optimal or strategically valid"},
        {"metric": "total variation", "sees": "maximum event-probability gap; ½‖p−q‖₁", "misses": "which action matters strategically"},
        {"metric": "Jensen–Shannon", "sees": "symmetric bounded distributional separation", "misses": "causal source of the difference"},
        {"metric": "arena result", "sees": "closed-loop game outcome under a protocol", "misses": "mechanism and often sample efficiency"},
    ]
    mo.vstack([
        mo.md(r"""
        ## Behavioral lenses answer non-equivalent questions

        \[
        TV(p,q)=\tfrac12\sum_a|p_a-q_a|,\qquad
        JS(p,q)=\tfrac12 KL(p\|m)+\tfrac12 KL(q\|m),\quad m=\tfrac12(p+q).
        \]
        For one-hot target (y), multiclass Brier score is
        (sum_a(p_a-y_a)^2). Expected calibration error (ECE) bins the
        model's top-action confidence and averages
        (|\text{accuracy}-\text{confidence}|), weighted by bin frequency.
        Neither needs a ground-truth move. NLL and accuracy do, and inherit the
        assumptions of that target source; Brier and ECE inherit them too.
        """),
        mo.Html(table(primer_rows, (("metric", "Metric"), ("sees", "What it measures"), ("misses", "What it does not establish")))),
    ])
    return


@app.cell
def interaction_controls(mo):
    pair_choice = mo.ui.dropdown(
        options={
            "Raw vs Hero end-to-end (RR ↔ HH)": ("RR", "HH"),
            "Encoder swap under Raw head (RR ↔ HR)": ("RR", "HR"),
            "Head swap under Raw encoder (RR ↔ RH)": ("RR", "RH"),
            "Head swap under Hero encoder (HR ↔ HH)": ("HR", "HH"),
        },
        value="Raw vs Hero end-to-end (RR ↔ HH)",
        label="Behavioral contrast",
    )
    temperature = mo.ui.number(0.5, 2.0, step=0.1, value=1.0, label="Recomputed policy temperature")
    grouping = mo.ui.radio(
        options={"Independent positions": "position", "Whole games": "game"},
        value="Whole games",
        label="Uncertainty unit",
    )
    mo.vstack([pair_choice, mo.hstack([temperature, grouping], justify="start")])
    return grouping, pair_choice, temperature


@app.cell
def constructed_analysis(
    grouping,
    js_divergence,
    np,
    pair_choice,
    paired_cluster_bootstrap,
    stable_softmax,
    temperature,
    total_variation,
    toy_policy_experiment,
):
    toy_policy = toy_policy_experiment()
    first_arm, second_arm = pair_choice.value

    # Explicit loop keeps illegal slots at zero and makes the normalization visible.
    def temperature_policy(probabilities):
        output = np.zeros_like(probabilities)
        for row_index, (row, legal_row) in enumerate(zip(probabilities, toy_policy.legal, strict=True)):
            output[row_index, legal_row] = stable_softmax(np.log(np.clip(row[legal_row], 1e-300, None)) / temperature.value)
        return output

    first_policy = temperature_policy(toy_policy.probabilities[first_arm])
    second_policy = temperature_policy(toy_policy.probabilities[second_arm])
    per_position_js = js_divergence(first_policy, second_policy)
    per_position_tv = total_variation(first_policy, second_policy)
    top1_agreement = float(np.mean(np.argmax(first_policy, axis=1) == np.argmax(second_policy, axis=1)))
    top_k = 3
    first_top_k = np.argpartition(first_policy, -top_k, axis=1)[:, -top_k:]
    second_top_k = np.argpartition(second_policy, -top_k, axis=1)[:, -top_k:]
    top_k_overlap = float(
        np.mean(
            [
                len(set(left.tolist()) & set(right.tolist())) / top_k
                for left, right in zip(first_top_k, second_top_k, strict=True)
            ]
        )
    )
    first_nll = -np.log(np.clip(first_policy[np.arange(first_policy.shape[0]), toy_policy.targets], 1e-12, None))
    second_nll = -np.log(np.clip(second_policy[np.arange(second_policy.shape[0]), toy_policy.targets], 1e-12, None))
    one_hot_targets = np.eye(first_policy.shape[1])[toy_policy.targets]
    first_brier = float(np.mean(np.sum((first_policy - one_hot_targets) ** 2, axis=1)))
    second_brier = float(np.mean(np.sum((second_policy - one_hot_targets) ** 2, axis=1)))

    def expected_calibration_error(probabilities):
        confidence = np.max(probabilities, axis=1)
        correct = np.argmax(probabilities, axis=1) == toy_policy.targets
        edges = np.linspace(0.0, 1.0, 6)
        error = 0.0
        for lower, upper in zip(edges[:-1], edges[1:], strict=True):
            in_bin = (confidence >= lower) & (
                confidence <= upper if upper == 1.0 else confidence < upper
            )
            if np.any(in_bin):
                error += float(np.mean(in_bin)) * abs(
                    float(np.mean(correct[in_bin])) - float(np.mean(confidence[in_bin]))
                )
        return error

    first_ece = expected_calibration_error(first_policy)
    second_ece = expected_calibration_error(second_policy)
    interval_groups = np.arange(per_position_js.size) if grouping.value == "position" else toy_policy.groups
    js_interval = paired_cluster_bootstrap(
        per_position_js,
        np.zeros_like(per_position_js),
        interval_groups,
        weighting="observation",
        seed=2202,
    )
    return (
        first_arm,
        first_brier,
        first_ece,
        first_nll,
        js_interval,
        per_position_tv,
        second_arm,
        second_brier,
        second_ece,
        second_nll,
        top1_agreement,
        top_k_overlap,
        toy_policy,
    )


@app.cell
def constructed_view(
    first_arm,
    first_brier,
    first_ece,
    first_nll,
    grouping,
    js_interval,
    metric_cards,
    mo,
    np,
    per_position_tv,
    second_arm,
    second_brier,
    second_ece,
    second_nll,
    top1_agreement,
    top_k_overlap,
):
    mo.vstack([
        mo.md("## Interaction: change both the policy and the sampling question"),
        mo.Html(metric_cards([
            (f"{js_interval.estimate:.4f}", f"mean JS, {first_arm} ↔ {second_arm}"),
            (f"[{js_interval.lower:.4f}, {js_interval.upper:.4f}]", f"95% {grouping.value}-bootstrap interval"),
            (f"{np.mean(per_position_tv):.4f}", "mean total variation"),
            (f"{top1_agreement:.1%}", "top-1 agreement"),
            (f"{top_k_overlap:.1%}", "mean top-3 set overlap"),
            (f"{np.mean(second_nll-first_nll):+.3f}", "second − first target NLL"),
            (f"{second_brier-first_brier:+.3f}", "second − first multiclass Brier"),
            (f"{first_ece:.3f} / {second_ece:.3f}", "first / second 5-bin ECE"),
        ])),
        mo.md("These are **constructed** policies with a known encoder-dominant mechanism. Top-3 overlap exposes subleading-set agreement that top-1 misses. Brier and five-bin ECE use the synthetic recorded target; changing the target source changes their meaning. They validate the estimator and build intuition; they are never Raw/Hero evidence."),
    ])
    return


@app.cell
def lattice_calculation(js_divergence, np, toy_policy):
    toy_pair_js = {}
    for left, right in (("RR", "RH"), ("RR", "HR"), ("RH", "HH"), ("HR", "HH"), ("RR", "HH")):
        toy_pair_js[f"{left}→{right}"] = float(np.mean(js_divergence(toy_policy.probabilities[left], toy_policy.probabilities[right])))
    probability_interaction = (
        toy_policy.probabilities["HH"]
        - toy_policy.probabilities["HR"]
        - toy_policy.probabilities["RH"]
        + toy_policy.probabilities["RR"]
    )
    toy_interaction_rms = float(np.sqrt(np.mean(probability_interaction**2)))
    return toy_interaction_rms, toy_pair_js


@app.cell
def lattice_view(mo, table, toy_interaction_rms, toy_pair_js):
    lattice_rows = [
        {"contrast": contrast, "mean JS": f"{value:.5f}", "localized change": "encoder" if contrast in {"RR→HR", "RH→HH"} else "head / end-to-end"}
        for contrast, value in toy_pair_js.items()
    ]
    mo.vstack([
        mo.md(r"""
        ## Read the 2×2 lattice

        Main encoder effect: compare `RR→HR` and `RH→HH`.
        Main head effect: compare `RR→RH` and `HR→HH`.
        The vector interaction is `HH − HR − RH + RR`; zero would mean the two
        swaps combine additively **in probability coordinates**. A different
        coordinate choice defines a different interaction.
        """),
        mo.Html(table(lattice_rows, (("contrast", "Contrast"), ("mean JS", "Mean JS"), ("localized change", "Reading")))),
        mo.md(f"Constructed probability-interaction RMS: **{toy_interaction_rms:.6f}**."),
    ])
    return


@app.cell
def real_behavior_panel(
    ClaimCard,
    Endpoint,
    EvidenceOperation,
    ResultStatus,
    TargetLevel,
    bundle,
    callout,
    claim_card_html,
    mo,
    table,
):
    if bundle.is_empirical:
        behavior = bundle.section("behavior")
        arm_rows = [
            {"arm": item["arm"], "top-1": f"{item['top1_accuracy']:.1%}", "target NLL": f"{item['target_nll']:.3f}"}
            for item in behavior["arms"]
        ]
        swap_rows = [
            {"comparison": item["comparison"], "mean JS": f"{item['js_divergence']:.6f}", "top-1 agreement": f"{item['top1_agreement']:.1%}"}
            for item in behavior["swaps"]
        ]
        encoder_js = behavior["swaps"][0]["js_divergence"]
        head_js = behavior["swaps"][1]["js_divergence"]
        behavior_card = ClaimCard(
            operation=EvidenceOperation.OBSERVATION,
            target=TargetLevel.BEHAVIOR,
            scope="128 retained development positions in the Raw/Hero hybrid pilot",
            endpoint=Endpoint.POLICY,
            estimand="mean legal-policy JS under each encoder/head swap",
            unit="paired position",
            grouping="one unresolved placeholder game cluster",
            intervention=None,
            controls=("RR/RH/HR/HH factorial lattice", "identical input ordering", "native-head parity checks"),
            assumptions=("swapped components are ABI-compatible", "the development prefix is not population-representative"),
            uncertainty="descriptive pilot values; population interval prohibited by the placeholder grouping",
            multiplicity="four prespecified hybrid arms",
            allowed=(f"In this development pilot, the Raw-to-Hero encoder swap has mean JS {encoder_js:.3f}, while the Raw-to-Hero native-head swap has mean JS {head_js:.6f}."),
            excluded="The lattice does not identify a unique internal mechanism or estimate a game-population effect.",
            falsifier="A grouped held-out replication where encoder-swap change is not larger than both head-swap controls under the frozen criterion.",
            status=ResultStatus.DESCRIPTIVE,
        )
        real_view = mo.vstack([
            mo.md("## Frozen Raw/Hero result"),
            mo.Html(table(arm_rows, (("arm", "Arm"), ("top-1", "Target top-1"), ("target NLL", "Target NLL")))),
            mo.Html(table(swap_rows, (("comparison", "Swap"), ("mean JS", "Mean JS"), ("top-1 agreement", "Top-1 agreement")))),
            mo.Html(claim_card_html(behavior_card)),
            mo.Html(callout("limit", "The strongest limitation is the grouping record", "All positions carry one placeholder game identifier. The pilot localizes engineering effort, but a bootstrap would only resample one cluster and cannot support population uncertainty.")),
        ])
    else:
        real_view = mo.Html(callout("neutral", "Toy boundary", "No Raw/Hero table is shown in toy mode. Switch explicitly to snapshot or source to inspect retained empirical metrics."))
    real_view
    return


@app.cell
def empirical_example_control(bundle, mo):
    if bundle.is_empirical:
        empirical_examples = bundle.section("behavior")["examples"]
        example_options = {f"Example {index + 1}: {item['legal_count']} legal actions": index for index, item in enumerate(empirical_examples)}
        empirical_example = mo.ui.dropdown(options=example_options, value=next(iter(example_options)), label="Inspect one paired position")
        example_widget = empirical_example
    else:
        empirical_examples = []
        empirical_example = None
        example_widget = mo.md("Empirical position inspector is unavailable in toy mode.")
    example_widget
    return empirical_example, empirical_examples


@app.cell
def empirical_example_view(
    chessboard,
    empirical_example,
    empirical_examples,
    metric_cards,
    mo,
):
    if empirical_example is not None:
        record = empirical_examples[empirical_example.value]
        pair = record["pairs"]["RR__HH"]
        example_view = mo.hstack([
            mo.Html(chessboard(record["fen"])),
            mo.Html(metric_cards([
                (record["legal_count"], "legal actions"),
                (f"{pair['js_divergence']:.4f}", "RR ↔ HH JS"),
                (f"{pair['total_variation']:.4f}", "RR ↔ HH TV"),
                ("yes" if pair["top1_agreement"] else "no", "top-1 agreement"),
            ])),
        ], widths=[1, 2])
    else:
        example_view = mo.md("")
    example_view
    return


@app.cell
def prediction_exercise(committed_form, mo):
    lattice_prediction = mo.ui.radio(
        options=[
            "The encoder contains the unique mechanism",
            "The behavioral delta localizes mostly upstream of the native head",
            "The output head is mathematically irrelevant on every input",
        ],
        value=None,
        label="Strongest claim licensed by encoder-large/head-small swaps",
    )
    prediction_form = committed_form(
        mo, {"lattice_claim": lattice_prediction}, submit_label="Commit lattice interpretation"
    )
    prediction_form
    return (prediction_form,)


@app.cell
def prediction_feedback(callout, mo, prediction_form):
    mo.stop(prediction_form.value is None, mo.md("Submit before revealing."))
    prediction_ok = prediction_form.value["lattice_claim"] == "The behavioral delta localizes mostly upstream of the native head"
    mo.Html(callout(
        "positive" if prediction_ok else "limit",
        "Calibrated localization" if prediction_ok else "The claim outruns the swap",
        "A module swap bounds where the functional difference is expressed under this interface. Redundant mechanisms, distributed updates, and compensating interactions remain possible.",
    ))
    return


@app.cell
def assessments(committed_form, mo, module):
    metric_transfer = mo.ui.text_area(label="Transfer 1 — metric", placeholder="Two language models have equal top-1 accuracy but different calibrated distributions. Choose an estimand and justify it.")
    grouping_transfer = mo.ui.text_area(label="Transfer 2 — grouping", placeholder="Prompts are ten paraphrases per document. What is the bootstrap unit and why?")
    lattice_transfer = mo.ui.text_area(label="Transfer 3 — hybrid lattice", placeholder="A swapped embedding changes behavior, while a swapped unembedding does not. State the strongest legal claim and two alternatives.")
    assessment_form = committed_form(
        mo,
        {"metric": metric_transfer, "grouping": grouping_transfer, "lattice": lattice_transfer},
        submit_label="Commit responses and reveal transfer rubric",
        min_words=5,
    )
    mo.vstack([mo.md(f"## Unseen transfer assessment\n\n{module.transfer_assessment}"), assessment_form])
    return (assessment_form,)


@app.cell
def assessment_feedback(assessment_form, mo):
    mo.stop(assessment_form.value is None, mo.md("Submit all three transfer items first."))
    response_counts = [len(assessment_form.value[key].split()) for key in ("metric", "grouping", "lattice")]
    mo.md(f"""
    **Rubric (response lengths {response_counts}):** full credit (1) chooses a
    distributional, calibration, or proper-scoring endpoint rather than accuracy
    alone; (2) resamples documents, not paraphrase rows; and (3) says
    *localizes the observed delta upstream under this interface*, then names
    redundancy/compensation and distribution shift as alternatives.
    """)
    return


@app.cell
def close(bundle, callout, mo, resource_card):
    mo.vstack([
        mo.Html(callout("positive", "What we learned", "Behavioral hybrid localization makes one practical result clear: subsequent representation analyses should focus on the encoder while keeping head controls.")),
        mo.Html(callout("limit", "What this does not show", "Large encoder-swap JS is not a circuit, a semantic feature, or proof that every position changed for the same reason.")),
        mo.md("""
        ## Retrieval before module 03

        1. Name one case where top-1 agreement hides substantial distributional change.
        2. Why can a position bootstrap be spuriously narrow?
        3. What does `HH − HR − RH + RR` measure, and in which coordinates?

        **Source:** Searchless Chess (NeurIPS 2024) supplies the specimen lineage;
        retained hybrid metrics are local development records. See the
        [course source register](../SOURCES.md).
        """),
        mo.Html(resource_card(
            mode=bundle.mode.value,
            runtime="< 3 seconds; 2,000 CPU bootstrap resamples",
            device="CPU / NumPy; no model or network",
            determinism="toy seed 11; bootstrap seed 2202; NumPy float64",
            sources=f"content identity {bundle.integrity_sha256[:16]}…",
            limitations="Toy estimator plus a 128-position empirical pilot with one placeholder game cluster.",
        )),
    ])
    return


if __name__ == "__main__":
    app.run()
