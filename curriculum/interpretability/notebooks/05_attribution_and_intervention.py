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
    from research.interpretability.course.experiments import (
        toy_attribution,
        toy_intervention,
    )
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
        square_heatmap,
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
        line_chart,
        load_course_bundle,
        metric_cards,
        mo,
        mode_badge,
        module_header,
        np,
        resource_card,
        square_heatmap,
        table,
        toy_attribution,
        toy_intervention,
    )


@app.cell
def title(get_module, mo, module_header):
    module = get_module(5)
    mo.Html(module_header(module))
    return (module,)


@app.cell
def opening(mo, module, table):
    objective_rows = [
        {"objective": objective, "observable evidence": evidence}
        for objective, evidence in zip(
            module.objectives,
            (
                "Predict which sensitivity diagnostic fails as a tanh output saturates.",
                "Write clean, corrupt, patch, unit, dose, and policy/logit endpoint exactly.",
                "Construct a symmetric dose study with positive, random, norm, and semantic-placebo controls.",
            ),
            strict=True,
        )
    ]
    mo.vstack([
        mo.md(r"""
        ## A map is a hypothesis generator; an intervention is a new experiment

        A derivative answers “what changes infinitesimally around this input?”
        An occlusion answers “what changes under this removal?” A patch asks
        “what happens when a named activation in one run is replaced by another?”
        These operations have different counterfactuals.

        We will first break attribution methods on a known saturating function.
        Then we will design a bounded activation intervention with controls that
        separate direction semantics from norm, generic sensitivity, and
        off-manifold displacement.
        """),
        mo.Html(table(objective_rows, (("objective", "Objective"), ("observable evidence", "Mastery evidence")))),
    ])
    return


@app.cell
def prerequisite(committed_form, mo):
    response = mo.ui.text_area(
        label="Prerequisite diagnostic — decodability versus use",
        placeholder="A linear probe predicts pins at 99%. What additional operation is required before saying the policy uses that direction?",
    )
    prerequisite_form = committed_form(
        mo, {"causal_bridge": response}, submit_label="Commit prerequisite response", min_words=6
    )
    prerequisite_form
    return (prerequisite_form,)


@app.cell
def prerequisite_feedback(mo, prerequisite_form):
    mo.stop(prerequisite_form.value is None, mo.md("Answer the prerequisite diagnostic to reveal the bridge."))
    mo.callout("Intervene on a named activation boundary and measure a model endpoint against no-op, matched-random, positive, dose/sign, and semantic-placebo controls. High probe accuracy alone licenses prediction, not use.", kind="info")
    return


@app.cell
def mode_controls(mo):
    evidence_mode = mo.ui.radio(
        options={
            "Snapshot — retained Raw/Hero attribution and patching": "snapshot",
            "Toy — known saturating mechanism": "toy",
            "Source — verify retained attribution records": "source",
        },
        value="Snapshot — retained Raw/Hero attribution and patching",
        label="Evidence mode",
    )
    verify_source = mo.ui.run_button(label="Verify retained sources", kind="warn")
    mo.hstack([evidence_mode, verify_source], justify="start", gap=1)
    return evidence_mode, verify_source


@app.cell
def load_mode(evidence_mode, load_course_bundle, mo, verify_source):
    selected_mode = evidence_mode.value
    mo.stop(selected_mode == "source" and not verify_source.value, mo.callout("Source mode requires an explicit checksum run.", kind="warn"))
    bundle = load_course_bundle(selected_mode)
    return bundle, selected_mode


@app.cell
def mode_view(bundle, mo, mode_badge, selected_mode):
    mo.Html(mode_badge(selected_mode, bundle.evidence_scope))
    return


@app.cell
def method_table(mo, table):
    method_rows = [
        {"method": "gradient", "operation": "local derivative ∂f/∂x", "required check": "saturation + parameter randomization"},
        {"method": "input × gradient", "operation": "local derivative weighted by input coordinate", "required check": "baseline/units and sign"},
        {"method": "integrated gradients", "operation": "path integral from a named baseline", "required check": "completeness residual + path plausibility"},
        {"method": "SmoothGrad", "operation": "average derivatives around noisy inputs", "required check": "noise scale and rule-valid perturbations"},
        {"method": "SARFA-style removal", "operation": "remove a chess piece and compare target-vs-alternative policy", "required check": "resulting board and legal support"},
        {"method": "attention map", "operation": "QK mixing weights", "required check": "values, output, and a causal intervention"},
    ]
    mo.Html(table(method_rows, (("method", "Method"), ("operation", "Exact operation"), ("required check", "Minimum diagnostic"))))
    return


@app.cell
def hand_worked_attribution(mo, np, table):
    # Scalar teaching case: f(x) = tanh(2x), baseline 0, input 1.
    baseline = 0.0
    input_value = 1.0
    output = float(np.tanh(2.0 * input_value))
    local_gradient = float(2.0 * (1.0 - np.tanh(2.0 * input_value) ** 2))
    input_times_gradient = input_value * local_gradient
    alpha_midpoints = (np.arange(4096, dtype=float) + 0.5) / 4096.0
    path_points = baseline + alpha_midpoints * (input_value - baseline)
    path_gradients = 2.0 * (1.0 - np.tanh(2.0 * path_points) ** 2)
    integrated_gradient = float((input_value - baseline) * np.mean(path_gradients))
    exact_difference = output - float(np.tanh(2.0 * baseline))
    rows = [
        {"quantity": "local gradient f'(1)", "calculation": "2(1−tanh²(2))", "value": f"{local_gradient:.6f}"},
        {"quantity": "input × gradient", "calculation": "1 × f'(1)", "value": f"{input_times_gradient:.6f}"},
        {"quantity": "integrated gradients", "calculation": "(1−0) ∫₀¹ f'(α)dα", "value": f"{integrated_gradient:.6f}"},
        {"quantity": "completeness target", "calculation": "f(1)−f(0)=tanh(2)", "value": f"{exact_difference:.6f}"},
    ]
    mo.vstack([
        mo.md(r"""
        ## Hand-worked attribution: local slope versus path effect

        Let (f(x)=\tanh(2x)), choose baseline (x'=0), and explain the
        input (x=1). The local derivative and input-times-gradient both equal
        about 0.141: saturation makes the endpoint locally flat. In one
        dimension, integrated gradients instead accumulates the derivative
        along the named straight path:

        \[
        IG(x)=(x-x')\int_0^1 f'\!\left(x'+\alpha(x-x')\right)d\alpha
             = f(x)-f(x').
        \]

        The equality is a completeness check, not a causal claim. It also says
        nothing about whether zero is a rule-valid or on-manifold baseline.
        """),
        mo.Html(table(rows, (("quantity", "Quantity"), ("calculation", "Calculation"), ("value", "Value")))),
        mo.md(f"Midpoint-rule completeness error with 4,096 steps: **{abs(integrated_gradient - exact_difference):.2e}**."),
        mo.md(r"""
        The complete minimal computation is visible here rather than hidden in
        a helper:

        ```python
        x0, x, steps = 0.0, 1.0, 4096
        alpha = (np.arange(steps) + 0.5) / steps
        path = x0 + alpha * (x - x0)
        grad = 2 * (1 - np.tanh(2 * path) ** 2)
        integrated_gradient = (x - x0) * grad.mean()
        ```

        SmoothGrad answers another question: average **local** gradients over
        a declared noise distribution around the input. SARFA is perturbational:
        remove a piece, recompute a legal policy, and separate change in the
        target action from redistribution among alternatives. Neither inherits
        integrated gradients' path-completeness identity.
        """),
    ])
    return


@app.cell
def attribution_controls(committed_form, mo):
    saturation = mo.ui.number(0.25, 6.0, step=0.25, value=1.0, label="tanh saturation scale")
    attribution_method = mo.ui.dropdown(
        options={
            "Plain gradient": "gradient",
            "Input × gradient": "input_gradient",
            "Integrated gradients": "integrated_gradients",
            "Single-coordinate occlusion": "occlusion",
        },
        value="Integrated gradients",
        label="Attribution estimator",
    )
    saturation_prediction = mo.ui.radio(
        options=[
            "The local gradient can vanish even when features matter globally",
            "Every method becomes exact at high saturation",
            "Occlusion and gradients always rank squares identically",
        ],
        value=None,
        label="Predict before computing",
    )
    attribution_form = committed_form(
        mo,
        {"saturation": saturation, "method": attribution_method, "prediction": saturation_prediction},
        submit_label="Commit prediction and compute",
    )
    attribution_form
    return (attribution_form,)


@app.cell
def attribution_experiment(attribution_form, mo, toy_attribution):
    mo.stop(attribution_form.value is None)
    attribution_result = toy_attribution(saturation=attribution_form.value["saturation"])
    selected_map = attribution_result[attribution_form.value["method"]]
    return attribution_result, selected_map


@app.cell
def attribution_view(
    attribution_form,
    attribution_result,
    callout,
    metric_cards,
    mo,
    np,
    selected_map,
    square_heatmap,
):
    gradient_norm = float(np.linalg.norm(attribution_result["gradient"]))
    occlusion_norm = float(np.linalg.norm(attribution_result["occlusion"]))
    prediction_ok = attribution_form.value["prediction"].startswith("The local gradient")
    mo.vstack([
        mo.md("## Interaction: saturation changes the local question"),
        mo.Html(metric_cards([
            (f"{attribution_result['output']:+.4f}", "tanh output"),
            (f"{gradient_norm:.4f}", "plain-gradient L2"),
            (f"{occlusion_norm:.4f}", "occlusion-effect L2"),
            (f"{attribution_result['completeness_error']:.4f}", "IG absolute completeness error"),
        ])),
        mo.Html(square_heatmap(selected_map, alt_text=f"Constructed {attribution_form.value['method']} attribution by chess square.", signed=True)),
        mo.Html(callout(
            "positive" if prediction_ok else "limit",
            "Saturation prediction retained" if prediction_ok else "Local is not global",
            "As tanh approaches ±1, its derivative shrinks. The chosen input can still depend on large weighted coordinates even while a local gradient is nearly zero. Integrated gradients changes the path question; occlusion changes the input itself.",
        )),
    ])
    return


@app.cell
def parameter_randomization_check(
    attribution_form,
    attribution_result,
    callout,
    metric_cards,
    mo,
    np,
):
    learned_features = np.asarray(attribution_result["features"], dtype=float)
    learned_weights = np.asarray(attribution_result["weights"], dtype=float)
    randomized_weights = np.random.default_rng(519).normal(size=learned_weights.size)
    randomized_weights *= np.linalg.norm(learned_weights) / np.linalg.norm(randomized_weights)
    randomized_score = float(
        np.tanh(attribution_form.value["saturation"] * (learned_features @ randomized_weights))
    )
    randomized_gradient = (
        attribution_form.value["saturation"]
        * (1.0 - randomized_score**2)
        * randomized_weights
    )
    randomized_map = learned_features * randomized_gradient
    learned_map = np.asarray(attribution_result["input_gradient"], dtype=float)
    map_denominator = float(np.linalg.norm(learned_map) * np.linalg.norm(randomized_map))
    randomized_map_cosine = float(learned_map @ randomized_map / max(map_denominator, 1e-15))
    learned_top = set(np.argsort(np.abs(learned_map))[-8:].tolist())
    randomized_top = set(np.argsort(np.abs(randomized_map))[-8:].tolist())
    top_overlap = len(learned_top & randomized_top) / 8.0
    mo.vstack([
        mo.md("## Executable sanity check: destroy the learned parameters"),
        mo.Html(metric_cards([
            (f"{randomized_map_cosine:+.3f}", "learned/randomized input×gradient cosine"),
            (f"{top_overlap:.1%}", "top-8 coordinate overlap"),
            (f"{attribution_result['output']:+.3f} → {randomized_score:+.3f}", "output before → after randomization"),
            (519, "fixed randomization seed"),
        ])),
        mo.Html(callout(
            "positive" if abs(randomized_map_cosine) < 0.5 else "danger",
            "Map responds to parameter destruction" if abs(randomized_map_cosine) < 0.5 else "Sanity check failed on this draw",
            (
                "The replacement weights are norm-matched but otherwise random. A learned-parameter "
                "attribution should not remain essentially unchanged after this intervention. This "
                "single constructed draw demonstrates the check; a real claim must freeze a similarity "
                "statistic, repeat across randomizations, and also consider label randomization. Passing "
                "is necessary, not evidence that the original map is causal."
            ),
        )),
    ])
    return


@app.cell
def attribution_diagnostics(mo, table):
    diagnostic_rows = [
        {"diagnostic": "completeness", "pass condition": "sum attributions ≈ f(x)−f(baseline)", "does not guarantee": "correct baseline, causal semantics, or stable ranking"},
        {"diagnostic": "parameter randomization", "pass condition": "map changes as learned weights are destroyed", "does not guarantee": "faithful downstream mechanism"},
        {"diagnostic": "input invariance", "pass condition": "function-preserving input shift does not arbitrarily rewrite map", "does not guarantee": "plausible counterfactual"},
        {"diagnostic": "method agreement", "pass condition": "prespecified rank/mass statistic replicates", "does not guarantee": "both methods are right"},
    ]
    mo.vstack([
        mo.md("## A failed diagnostic invalidates the method result in scope"),
        mo.Html(table(diagnostic_rows, (("diagnostic", "Diagnostic"), ("pass condition", "Pass condition"), ("does not guarantee", "Still not shown")))),
    ])
    return


@app.cell
def intervention_controls(mo):
    dose_span = mo.ui.number(0.5, 4.0, step=0.5, value=2.0, label="symmetric maximum dose")
    dose_fraction = mo.ui.number(-1.0, 1.0, step=0.125, value=0.5, label="inspected fraction of maximum dose")
    mo.hstack([dose_span, dose_fraction])
    return dose_fraction, dose_span


@app.cell
def intervention_experiment(dose_fraction, dose_span, np, toy_intervention):
    intervention_doses = np.linspace(-dose_span.value, dose_span.value, 17)
    intervention_result = toy_intervention(intervention_doses)
    requested_dose = dose_fraction.value * dose_span.value
    nearest_dose_index = int(np.argmin(np.abs(intervention_doses - requested_dose)))
    inspected_dose = float(intervention_doses[nearest_dose_index])
    return (
        inspected_dose,
        intervention_doses,
        intervention_result,
        nearest_dose_index,
    )


@app.cell
def intervention_view(
    inspected_dose,
    intervention_doses,
    intervention_result,
    line_chart,
    metric_cards,
    mo,
    nearest_dose_index,
):
    intervention_series = [
        {"name": "candidate", "color": "#3366a3", "points": list(zip(intervention_doses, intervention_result["candidate"], strict=True))},
        {"name": "known positive", "color": "#7450a8", "points": list(zip(intervention_doses, intervention_result["positive_control"], strict=True))},
        {"name": "semantic placebo", "color": "#a95f19", "points": list(zip(intervention_doses, intervention_result["semantic_placebo"], strict=True))},
        {"name": "matched random", "color": "#14735a", "points": list(zip(intervention_doses, intervention_result["random_matched"], strict=True))},
    ]
    mo.vstack([
        mo.md("## Interaction: direction is not dose"),
        mo.Html(line_chart(intervention_series, x_label="intervention dose", y_label="mean constructed logit change", alt_text="Candidate, known-positive, semantic-placebo, and matched-random intervention dose curves with distinct line patterns.")),
        mo.Html(metric_cards([
            (f"{intervention_result['candidate'][nearest_dose_index]:+.3f}", "candidate effect at nearest dose"),
            (f"{intervention_result['positive_control'][nearest_dose_index]:+.3f}", "known-positive effect"),
            (f"{intervention_result['semantic_placebo'][nearest_dose_index]:+.3f}", "semantic-placebo effect"),
            (f"{intervention_result['random_matched'][nearest_dose_index]:+.3f}", "matched-random effect"),
            (f"{intervention_result['density_distance'][nearest_dose_index]:.2f}", "latent displacement norm"),
            (f"{inspected_dose:+.2f}", "actual inspected dose"),
            ("constructed target logit", "declared endpoint"),
        ])),
        mo.md("The construction declares an orthonormal concept basis before outcomes: the candidate is a pin-signal direction with known endpoint loading, the semantic placebo is an own-queen direction with zero ground-truth loading, the matched random direction is a third frozen basis vector, and the distinct positive control is the exact endpoint gradient. Thus ‘semantic’ is part of the generator, not a name attached to Gaussian noise after seeing the curve. This lab computes a target-logit endpoint only. A legal-policy endpoint would require legality masking; an arena endpoint would require a separate closed-loop protocol."),
    ])
    return


@app.cell
def patch_design(mo, table):
    patch_rows = [
        {"field": "clean run", "example": "Hero activation on the same board"},
        {"field": "corrupt / baseline run", "example": "Raw activation on that exact board"},
        {"field": "patch", "example": "replace named layer/token/component activation, preserving all else"},
        {"field": "estimand", "example": "projection of patched policy delta onto full Raw→Hero policy delta"},
        {"field": "identity control", "example": "patch a run with its own activation; JS should be numerical zero"},
        {"field": "scope limit", "example": "whole-residual patch localizes a layer but does not isolate a path"},
    ]
    mo.vstack([
        mo.md("## Activation patching as a controlled counterfactual"),
        mo.Html(table(patch_rows, (("field", "Design field"), ("example", "Raw/Hero example")))),
    ])
    return


@app.cell
def real_attribution_panel(bundle, callout, chessboard, mo, square_heatmap):
    if bundle.is_empirical:
        attribution = bundle.section("attribution")
        example = attribution["examples"][0]
        real_map = example["arms"]["RR"]["maps"]["integrated_gradients"]["normalized_mass"]
        summary = attribution["summary"]
        ig = summary["integrated_gradients_relative_completeness_error"]
        sarfa = summary["sarfa_cosine"]
        attribution_view = mo.vstack([
            mo.md("## Frozen attribution diagnostic"),
            mo.hstack([
                mo.Html(chessboard(example["fen"])),
                mo.Html(square_heatmap(real_map, alt_text="Retained Raw integrated-gradients normalized attribution mass for one chess position.", signed=False)),
            ], widths=[1, 2]),
            mo.Html(callout("positive", "One reproducible agreement", f"Across two retained examples, Raw/Hero SARFA map cosine averages {sarfa['mean']:.3f}. This is descriptive method/model agreement.")),
            mo.Html(callout("danger", "Integrated gradients failed its own check", f"Across four arm-examples, mean relative completeness error is {ig['mean']:.2f} (retained interval {ig['lower']:.2f}–{ig['upper']:.2f}). The displayed map is a failure case, not evidence for square importance.")),
        ])
    else:
        attribution_view = mo.Html(callout("neutral", "Toy boundary", "No retained Raw/Hero map is shown in toy mode."))
    attribution_view
    return


@app.cell
def real_patch_panel(
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
):
    if bundle.is_empirical:
        intervention = bundle.section("intervention")
        patching = intervention["patching"]
        patch_series = []
        for direction, color in (("Hero → Raw", "#3366a3"), ("Raw → Hero", "#a95f19")):
            patch_series.append({"name": direction, "color": color, "points": [(row["layer"], row["distribution_delta_projection"]) for row in patching if row["direction"] == direction]})
        identity_js = intervention["identity_control"]["baseline_to_patched_js"]["maximum"]
        final_projection = min(row["distribution_delta_projection"] for row in patching if row["layer"] == 14)
        patch_card = ClaimCard(
            operation=EvidenceOperation.INTERVENTION,
            target=TargetLevel.COMPONENT,
            scope="eight retained development positions under bidirectional whole-residual Raw/Hero patches",
            endpoint=Endpoint.POLICY,
            estimand="projection of the patched legal-policy delta onto the full cross-model distribution delta",
            unit="paired position and patch direction",
            grouping="position; tiny selected development sample",
            intervention="replace the named layer whole residual with its paired-model activation on the identical input",
            controls=("same-run identity patch", "both Raw→Hero and Hero→Raw directions", "layers 0, 7, and 14"),
            assumptions=("the fixed-head policy projection is the intended endpoint", "whole-residual replacement is ABI-valid"),
            uncertainty="descriptive eight-position pilot; no population interval",
            multiplicity="three prespecified layers × two directions",
            allowed=(f"At layer 14 in this exact patch experiment, whole-residual replacement mediates at least {final_projection:.6f} of the fixed-head distribution-delta projection."),
            excluded="The patch does not establish a sparse feature, minimal path, chess concept, or population-wide mechanism.",
            falsifier="A larger grouped replication where the layer-14 projection misses a frozen near-one margin while identity patches still pass.",
            status=ResultStatus.POSITIVE,
        )
        patch_view = mo.vstack([
            mo.md("## Frozen bidirectional residual-patching result"),
            mo.Html(line_chart(patch_series, x_label="BT4 layer", y_label="distribution-delta projection", y_domain=(0.0, 1.05), alt_text="Bidirectional Raw and Hero whole-residual patch projections rise from early to final layer and reach approximately one.")),
            mo.Html(claim_card_html(patch_card)),
            mo.Html(callout("positive", "Identity control", f"Maximum identity-patch JS is {identity_js:.2e}; the intervention implementation is numerically self-consistent on this sample.")),
        ])
    else:
        patch_view = mo.md("")
    patch_view
    return


@app.cell
def intervention_checklist(mo, table):
    checklist_rows = [
        {"axis": "strength", "required": "negative, zero, and positive dose; curve not one hand-picked point"},
        {"axis": "magnitude", "required": "norm-matched random and semantic-placebo directions"},
        {"axis": "manifold", "required": "activation-density / nearest-neighbor displacement diagnostic"},
        {"axis": "specificity", "required": "positive control, off-target concepts, and alternate outputs"},
        {"axis": "validity", "required": "legal-action remasking and exact board replay"},
        {"axis": "inference", "required": "necessity versus sufficiency; grouped uncertainty; multiplicity"},
    ]
    mo.Html(table(checklist_rows, (("axis", "Intervention axis"), ("required", "Required record"))))
    return


@app.cell
def method_coverage_boundary(callout, mo):
    mo.Html(callout(
        "limit",
        "Decomposition methods are a deliberate next layer",
        (
            "This module derives attribution and whole-activation patching. "
            "Logit lens/direct-logit attribution, per-head output decomposition, "
            "path patching, and causal scrubbing are distinguished but not presented "
            "as completed Raw/Hero experiments here. Their prerequisites, failure "
            "modes, and expansion route are explicit in ../METHOD_COVERAGE.md."
        ),
    ))
    return


@app.cell
def assessments(committed_form, mo, module):
    attribution_transfer = mo.ui.text_area(label="Transfer 1 — attribution", placeholder="A language-model attention head focuses on a name. Design two diagnostics and state the legal claim before intervention.")
    patch_transfer = mo.ui.text_area(label="Transfer 2 — patch estimand", placeholder="Define clean, corrupt, patch, unit, and logit endpoint for a sentiment intervention.")
    placebo_transfer = mo.ui.text_area(label="Transfer 3 — steering controls", placeholder="Design sign/dose, norm-matched random, positive, and plausible semantic-placebo controls for a pin direction.")
    assessment_form = committed_form(
        mo,
        {"attribution": attribution_transfer, "patch": patch_transfer, "placebo": placebo_transfer},
        submit_label="Commit responses and reveal intervention rubric",
        min_words=5,
    )
    mo.vstack([mo.md(f"## Unseen transfer assessment\n\n{module.transfer_assessment}"), assessment_form])
    return (assessment_form,)


@app.cell
def assessment_feedback(assessment_form, mo):
    mo.stop(assessment_form.value is None, mo.md("Submit all three objectives before revealing."))
    answer_lengths = [len(assessment_form.value[key].split()) for key in ("attribution", "patch", "placebo")]
    mo.md(f"""
    **Rubric (response lengths {answer_lengths}):** attribution remains
    descriptive after completeness/randomization checks; patching names a single
    activation boundary and endpoint; steering sweeps both signs and multiple
    doses, matches norms, uses a plausible wrong concept, checks density and
    off-target logits, preserves legality, and groups uncertainty by game.
    """)
    return


@app.cell
def close(bundle, callout, mo, resource_card):
    mo.vstack([
        mo.Html(callout("positive", "Positive result", "The identity control and bidirectional late-layer patches validate a scoped component-level localization result.")),
        mo.Html(callout("limit", "What this does not show", "Whole-residual mediation is not a minimal circuit. The retained IG maps fail completeness, and high probe accuracy plus a steering direction can still overstate behavioral causation.")),
        mo.md("""
        ## Retrieval before module 06

        1. Why can a gradient vanish on an important feature?
        2. What must an identity patch do?
        3. Why is a semantic placebo different from a random direction?

        **Sources:** Sundararajan et al. (ICML 2017), Smilkov et al. (2017
        preprint), Adebayo et al. (NeurIPS 2018), Puri et al. (ICLR 2020),
        Jain & Wallace (NAACL 2019), Wiegreffe & Pinter (EMNLP 2019), Meng et
        al. (NeurIPS 2022), and the chess-pin intervention study (ICML 2026
        Mechanistic Interpretability Workshop virtual poster). See
        [course source register](../SOURCES.md).
        """),
        mo.Html(resource_card(
            mode=bundle.mode.value,
            runtime="< 3 seconds for constructed maps and 17-point dose curves",
            device="CPU / NumPy; no checkpoint or network",
            determinism="toy seeds 19/23; parameter-randomization seed 519; NumPy float64",
            sources=f"content identity {bundle.integrity_sha256[:16]}…",
            limitations="Known toy mechanism plus tiny retained attribution/patching pilots; no live activation intervention.",
        )),
    ])
    return


if __name__ == "__main__":
    app.run()
