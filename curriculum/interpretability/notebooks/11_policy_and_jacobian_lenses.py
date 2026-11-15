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
    from research.interpretability.course.activations import load_optional_activation_gallery
    from research.interpretability.course.evidence import (
        ClaimCard,
        Endpoint,
        EvidenceOperation,
        ResultStatus,
        TargetLevel,
    )
    from research.interpretability.course.experiments import toy_lens_experiment
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
        load_optional_activation_gallery,
        metric_cards,
        mo,
        mode_badge,
        module_header,
        np,
        repo_root,
        resource_card,
        square_heatmap,
        table,
        toy_lens_experiment,
    )


@app.cell
def title(get_module, mo, module_header):
    module = get_module(11)
    mo.Html(module_header(
        module,
        subtitle="A visual essay about reading intermediate states without confusing a readout with the computation.",
    ))
    return (module,)


@app.cell
def opening(mo, module, table):
    objective_rows = [
        {"question": question, "observable success": success}
        for question, success in zip(
            module.objectives,
            (
                "Write each transport map and say which quantities are learned or differentiated.",
                "Construct a setting where identity decoding fails but a Jacobian transport succeeds.",
                "Turn a language-model lens into a legality-aware chess policy experiment.",
            ),
            strict=True,
        )
    ]
    mo.vstack([
        mo.md(r"""
        ## The tempting picture

        A model is about to play a move. We pause after every transformer
        block, attach the final decoder, and watch a probability climb. It is
        natural to narrate the curve as a thought becoming clearer.

        That narration skips a problem: an intermediate vector may live in a
        different basis, scale, or distribution from the vector the decoder
        was trained to receive. A lens is a measurement instrument. Its
        transport rule is part of the claim.

        We will anchor the question in a real Raw/Hero activation trace, build
        three instruments from first principles, and then break them on a
        controlled nonlinear system where the correct transport is known.
        """),
        mo.Html(table(
            objective_rows,
            (("question", "Learning objective"), ("observable success", "What mastery looks like")),
        )),
    ])
    return


@app.cell
def prerequisite(committed_form, mo):
    prerequisite_answer = mo.ui.text_area(
        label="Prerequisite diagnostic — decoder versus mechanism",
        placeholder="A linear decoder predicts pins from layer 5. What does that establish, and what would still be needed to establish policy relevance?",
    )
    prerequisite_form = committed_form(
        mo,
        {"decoder_claim": prerequisite_answer},
        submit_label="Commit the boundary",
        min_words=8,
    )
    prerequisite_form
    return (prerequisite_form,)


@app.cell
def prerequisite_feedback(mo, prerequisite_form):
    mo.stop(
        prerequisite_form.value is None,
        mo.md("Commit a boundary statement to reveal the short answer."),
    )
    mo.callout(
        "The probe establishes scoped decodability under its training and split protocol. "
        "Policy relevance needs a separate intervention or interchange experiment with "
        "matched controls. A more accurate lens is still a better measurement, not that intervention.",
        kind="info",
    )
    return


@app.cell
def mode_controls(mo):
    evidence_mode = mo.ui.radio(
        options={
            "Snapshot — retained Raw/Hero policy-lens pilot": "snapshot",
            "Toy — constructed transport system": "toy",
            "Source — verify retained policy-lens records": "source",
        },
        value="Snapshot — retained Raw/Hero policy-lens pilot",
        label="Evidence mode",
    )
    verify_source = mo.ui.run_button(label="Verify retained sources", kind="warn")
    mo.hstack([evidence_mode, verify_source], justify="start", gap=1)
    return evidence_mode, verify_source


@app.cell
def load_mode(evidence_mode, load_course_bundle, mo, verify_source):
    selected_mode = evidence_mode.value
    mo.stop(
        selected_mode == "source" and not verify_source.value,
        mo.callout("Press the verification button before entering source mode.", kind="warn"),
    )
    bundle = load_course_bundle(selected_mode)
    return bundle, selected_mode


@app.cell
def mode_view(bundle, mo, mode_badge, selected_mode):
    mo.Html(mode_badge(selected_mode, bundle.evidence_scope))
    return


@app.cell
def real_lens_controls(
    bundle,
    load_optional_activation_gallery,
    mo,
    repo_root,
    selected_mode,
):
    if bundle.is_empirical:
        lens_gallery = load_optional_activation_gallery(
            repo_root,
            verify_sources=selected_mode == "source",
        )
        _lens_positions = (
            lens_gallery.position_options()
            if lens_gallery is not None
            else {"Local activation pack is not materialized": 0}
        )
    else:
        lens_gallery = None
        _lens_positions = {"Constructed mode has no checkpoint lens": 0}
    lens_position = mo.ui.dropdown(
        options=_lens_positions,
        value=next(iter(_lens_positions)),
        label="Real chess position",
    )
    lens_layer = mo.ui.number(
        0,
        14,
        step=1,
        value=14,
        label="Policy-lens depth",
    )
    lens_gradient_layer = mo.ui.dropdown(
        options={"Layer 0": 0, "Layer 7": 7, "Layer 14": 14},
        value="Layer 14",
        label="Stored VJP depth",
    )
    lens_gradient_model = mo.ui.radio(
        options={"Raw → Raw": "raw", "Hero → Hero": "hero"},
        value="Hero → Hero",
        label="VJP arm",
        inline=True,
    )
    mo.vstack([
        mo.md("## Start with a real layer-by-layer chess readout"),
        mo.md(
            "The table and curves below decode stored residual activations from "
            "the actual Raw and Hero checkpoints through both native nonlinear "
            "policy heads. The controls change the tensor slice, not a simulation seed."
        ),
        mo.hstack(
            [lens_position, lens_layer, lens_gradient_layer, lens_gradient_model],
            justify="start",
            gap=1,
        ),
    ])
    return (
        lens_gallery,
        lens_gradient_layer,
        lens_gradient_model,
        lens_layer,
        lens_position,
    )


@app.cell
def real_lens_microscope(
    bundle,
    callout,
    chessboard,
    lens_gallery,
    lens_gradient_layer,
    lens_gradient_model,
    lens_layer,
    lens_position,
    line_chart,
    metric_cards,
    mo,
    resource_card,
    square_heatmap,
    table,
):
    if bundle.is_empirical and lens_gallery is not None:
        _position_index = int(lens_position.value)
        _selected_layer = int(lens_layer.value)
        real_lens_rows = lens_gallery.policy_lens_view(
            position=_position_index,
            layer=_selected_layer,
            top_k=5,
        )
        _record = lens_gallery.position(_position_index)
        _arm_colors = {
            "RR": "#3366a3",
            "HR": "#7450a8",
            "RH": "#a95f19",
            "HH": "#14735a",
        }
        _all_depth_rows = {
            layer: lens_gallery.policy_lens_view(
                position=_position_index,
                layer=layer,
                top_k=1,
            )
            for layer in lens_gallery.layer_indices
        }
        _target_series = []
        _legal_mass_series = []
        for _arm in ("RR", "HR", "RH", "HH"):
            _target_series.append({
                "name": _arm,
                "color": _arm_colors[_arm],
                "points": [
                    (layer, next(row for row in rows if row["arm"] == _arm)["target_probability"])
                    for layer, rows in _all_depth_rows.items()
                ],
            })
            _legal_mass_series.append({
                "name": _arm,
                "color": _arm_colors[_arm],
                "points": [
                    (layer, next(row for row in rows if row["arm"] == _arm)["legal_mass_before_masking"])
                    for layer, rows in _all_depth_rows.items()
                ],
            })
        _table_rows = [
            {
                "arm": row["arm"],
                "top moves": row["top_moves"],
                "P(target | legal)": f"{row['target_probability']:.2%}",
                "pre-mask legal mass": f"{row['legal_mass_before_masking']:.2%}",
                "legal entropy": f"{row['entropy']:.3f}",
            }
            for row in real_lens_rows
        ]
        _gradient = lens_gallery.target_gradient_view(
            position=_position_index,
            layer=int(lens_gradient_layer.value),
            model=str(lens_gradient_model.value),
        )
        real_lens_microscope_view = mo.vstack([
            mo.Html(metric_cards([
                (_record["puzzle_id"], "public puzzle specimen"),
                (_record["target_move"], "recorded target move"),
                (_record["legal_count"], "legal moves"),
                (_selected_layer, "decoded residual depth"),
                (f"{_gradient['gradient_rms']:.3e}", "selected VJP residual RMS"),
            ])),
            mo.hstack([
                mo.Html(chessboard(_record["fen"])),
                mo.vstack([
                    mo.md(
                        f"**Solution line:** `{_record['solution_uci']}`  \n"
                        f"**Themes:** {_record['themes']}  \n"
                        "RR/HR/RH/HH mean Raw-or-Hero encoder followed by "
                        "Raw-or-Hero native policy head."
                    ),
                    mo.Html(table(
                        _table_rows,
                        (
                            ("arm", "Encoder → head"),
                            ("top moves", "Top legal moves at this depth"),
                            ("P(target | legal)", "Legal-renormalized P(target)"),
                            ("pre-mask legal mass", "Pre-mask legal mass"),
                            ("legal entropy", "Legal entropy"),
                        ),
                    )),
                ]),
            ], justify="start", gap=1),
            mo.ui.tabs({
                "Target emergence": mo.Html(line_chart(
                    _target_series,
                    x_label="residual depth",
                    y_label="P(recorded target | legal)",
                    y_domain=(0.0, 1.0),
                    alt_text=(
                        "Real position-level legal-renormalized target probability "
                        "through all fifteen layers for the four Raw/Hero hybrid arms."
                    ),
                )),
                "Legality before masking": mo.Html(line_chart(
                    _legal_mass_series,
                    x_label="residual depth",
                    y_label="pre-mask legal-action mass",
                    y_domain=(0.0, 1.0),
                    alt_text=(
                        "Real position-level probability mass assigned to legal actions "
                        "before masking through all layers for all hybrid arms."
                    ),
                )),
                "Per-square VJP norm": mo.Html(square_heatmap(
                    _gradient["square_gradient_norm"],
                    alt_text=(
                        f"{str(lens_gradient_model.value)} target-log-odds VJP norm at "
                        f"layer {int(lens_gradient_layer.value)} by absolute board square."
                    ),
                    signed=False,
                )),
            }),
            mo.Html(callout(
                "limit",
                "What is real here—and what is not",
                "The residuals, policy-head outputs, legal masks, moves, and endpoint "
                "gradient are checkpoint computations. The depth readout is still an "
                "identity/native-head lens, and the stored map is a per-square norm "
                "reduction of one full endpoint-specific VJP. Neither is the fitted "
                "corpus-average transport called J-Lens.",
            )),
            mo.Html(resource_card(
                mode="real activation and policy-lens replay · frozen development gallery",
                runtime="instant from 30.8 MB stored tensors and lens logits",
                device="extracted locally on GTX 1660 Ti",
                determinism="float32 extraction; float16 stored residuals/logits; fixed legal support",
                sources=f"{lens_gallery.run_id}; data SHA-256 {lens_gallery.data_sha256}",
                limitations=(
                    "Eight development positions; float16 replay; no fitted J-Lens, "
                    "population interval, or causal-use conclusion."
                ),
            )),
        ])
    elif bundle.is_empirical:
        real_lens_rows = []
        real_lens_microscope_view = mo.Html(callout(
            "limit",
            "The local real-activation pack is absent",
            "This public repository intentionally omits the license-gated 30.8 MB "
            "tensor pack. Rebuild it locally from authorized sources to enable the "
            "interactive policy-lens and VJP replay.",
        ))
    else:
        real_lens_rows = []
        real_lens_microscope_view = mo.Html(callout(
            "neutral",
            "Toy mode has no checkpoint policy trace",
            "Switch to Snapshot or source-verified mode to inspect real residuals and "
            "move distributions. The transport laboratory below is still valuable "
            "because its nonlinear map and Jacobian are known exactly.",
        ))
    real_lens_microscope_view
    return


@app.cell
def three_lenses(mo, table):
    lens_rows = [
        {
            "instrument": "identity / logit lens",
            "transport": "T_l(h)=h",
            "estimated from": "nothing",
            "good at": "direct, comparable diagnostic",
            "main failure": "basis and scale drift",
        },
        {
            "instrument": "tuned lens",
            "transport": "T_l(h)=A_l h+b_l",
            "estimated from": "fit intermediate to final distributions",
            "good at": "learned affine alignment",
            "main failure": "fit leakage and distribution shift",
        },
        {
            "instrument": "Jacobian lens (J-Lens)",
            "transport": "T_l(h)=J_l h",
            "estimated from": "average downstream Jacobian",
            "good at": "first-order downstream transport",
            "main failure": "averaging and finite-step curvature",
        },
    ]
    mo.vstack([
        mo.md(r"""
        ## One decoder, three transports

        Let \(h_l\) be an intermediate residual state, \(h_L=F_l(h_l)\)
        the downstream state, \(N\) the final normalization, and \(W_U\) a
        language model's unembedding. The ordinary logit lens computes

        \[
        p_l^{\mathrm{id}}=\operatorname{softmax}(W_U N(h_l)).
        \]

        A tuned lens fits an affine translator \(A_lh_l+b_l\). The Jacobian
        lens instead differentiates the actual downstream map. In the
        sequence-model form, its corpus-level transport is

        \[
        J_l=\mathbb E_{x,t,t'\ge t}
          \left[\frac{\partial h_{L,t'}}{\partial h_{l,t}}\right],
        \qquad
        p_l^{J}=\operatorname{softmax}(W_U N(J_lh_l)).
        \]

        The expectation is important. This is not the local Jacobian of one
        prompt, and \(J_lh_l\) is not guaranteed to equal \(F_l(h_l)\).
        The method asks whether one fixed, corpus-averaged first-order
        transport exposes a more stable workspace-like representation.

        J-Lens is the transport and readout. The paper's **J-space** is a
        separate sparse construction over J-Lens vectors; the two names are
        not synonyms. Canonical J-Lens vectors are tied to single vocabulary
        tokens, so multi-token concepts and structured relations remain a
        limitation.
        """),
        mo.Html(table(
            lens_rows,
            (
                ("instrument", "Lens"),
                ("transport", "Transport"),
                ("estimated from", "Information used"),
                ("good at", "Diagnostic value"),
                ("main failure", "Characteristic failure"),
            ),
        )),
    ])
    return


@app.cell
def toy_controls(mo):
    basis_angle = mo.ui.number(
        0.0, 90.0, step=5.0, value=55.0, label="Intermediate-basis rotation (degrees)"
    )
    nonlinearity = mo.ui.number(
        0.0, 1.5, step=0.05, value=0.35, label="Downstream nonlinearity"
    )
    corpus_mix = mo.ui.number(
        0.0, 1.0, step=0.05, value=0.5, label="Fraction of fit corpus in context 1"
    )
    test_context = mo.ui.radio(
        options={"Test context 0": 0, "Test context 1": 1},
        value="Test context 0",
        label="Evaluation context",
        inline=True,
    )
    perturbation_size = mo.ui.number(
        0.0, 2.0, step=0.05, value=0.2, label="Finite perturbation size"
    )
    winner_prediction = mo.ui.radio(
        options=[
            "identity / logit lens",
            "average Jacobian lens",
            "local Jacobian lens",
            "OLS affine state translator (not Tuned Lens)",
        ],
        value=None,
        label="Before changing the controls, which approximation will have the smallest KL?",
    )
    mo.vstack([
        mo.md(r"""
        ## Laboratory: rotate the basis, bend the computation

        To isolate transport, this construction sets the final normalization
        \(N\) to the identity. Its OLS affine baseline is fitted by ordinary
        least squares from intermediate to final states; the canonical Tuned
        Lens instead fits output distributions by a distillation-KL objective.
        The OLS row is therefore a state-regression analogue, not a Tuned Lens.
        The local Jacobian is an oracle diagnostic for one state, not the
        published J-Lens. Its \(J(h)h\) readout is also not an affine Taylor
        approximation around \(h\), which would require an intercept. The
        finite-perturbation panel separately uses the valid local relation
        \(F(h+\delta)-F(h)\approx J(h)\delta\). The corpus-average Jacobian is
        the J-Lens analogue.
        """),
        mo.hstack([basis_angle, nonlinearity, corpus_mix], justify="start"),
        mo.hstack([test_context, perturbation_size], justify="start"),
        winner_prediction,
    ])
    return (
        basis_angle,
        corpus_mix,
        nonlinearity,
        perturbation_size,
        test_context,
        winner_prediction,
    )


@app.cell
def run_toy_lens(
    basis_angle,
    corpus_mix,
    nonlinearity,
    perturbation_size,
    test_context,
    toy_lens_experiment,
):
    lens_toy = toy_lens_experiment(
        basis_angle_degrees=float(basis_angle.value),
        nonlinearity=float(nonlinearity.value),
        corpus_mix=float(corpus_mix.value),
        test_context=int(test_context.value),
        perturbation_size=float(perturbation_size.value),
    )
    return (lens_toy,)


@app.cell
def lens_toy_table(lens_toy, mo, np, table, winner_prediction):
    display_names = {
        "identity_logit_lens": "identity / logit lens",
        "average_jacobian_lens": "average Jacobian lens",
        "local_jacobian_lens": "local Jacobian lens",
        "ols_affine_state_translator": "OLS affine state translator",
    }
    lens_rows_toy = []
    for lens_key, lens_name in display_names.items():
        probability = lens_toy["probabilities"][lens_key]
        lens_rows_toy.append({
            "lens": lens_name,
            "KL(actual || lens)": f"{lens_toy['kl_to_actual'][lens_key]:.6f}",
            "top output": int(np.argmax(probability)),
            "top probability": f"{float(np.max(probability)):.3f}",
        })
    best_lens = min(
        display_names,
        key=lambda name: lens_toy["kl_to_actual"][name],
    )
    prediction_note = (
        "Make a prediction above, then perturb the system."
        if winner_prediction.value is None
        else (
            f"You predicted {winner_prediction.value}; under the current controls, "
            f"{display_names[best_lens]} has the smallest KL."
        )
    )
    mo.vstack([
        mo.Html(table(
            lens_rows_toy,
            (
                ("lens", "Lens"),
                ("KL(actual || lens)", "KL to true downstream readout"),
                ("top output", "Top output"),
                ("top probability", "Top probability"),
            ),
        )),
        mo.md(prediction_note),
        mo.callout(
            "Set nonlinearity to zero. Average and local Jacobians then recover the "
            "known rotation, while identity decoding still sees the wrong basis. "
            "Raise nonlinearity and skew the corpus mix: the average transport can "
            "lose to a context-local one. This is a controlled failure, not a leaderboard.",
            kind="neutral",
        ),
    ])
    return


@app.cell
def finite_step_view(lens_toy, metric_cards, mo, np, table):
    actual_delta = np.asarray(lens_toy["actual_logit_delta"])
    average_delta = np.asarray(lens_toy["average_linear_delta"])
    local_delta = np.asarray(lens_toy["local_linear_delta"])
    average_error = float(np.sqrt(np.mean((actual_delta - average_delta) ** 2)))
    local_error = float(np.sqrt(np.mean((actual_delta - local_delta) ** 2)))
    delta_rows = [
        {
            "output": index,
            "actual finite delta": f"{actual:.4f}",
            "average-J prediction": f"{average:.4f}",
            "local-J prediction": f"{local:.4f}",
        }
        for index, (actual, average, local) in enumerate(
            zip(actual_delta, average_delta, local_delta, strict=True)
        )
    ]
    mo.vstack([
        mo.md(r"""
        ### A Jacobian predicts infinitesimal change—not an arbitrary jump

        For perturbation \(\delta\), first-order transport predicts
        \(F(h+\delta)-F(h)\approx J(h)\delta\). The table compares that
        prediction after the fixed readout with the actual finite change.
        Increase the perturbation size and watch curvature accumulate.
        """),
        mo.Html(metric_cards([
            (f"{average_error:.4f}", "average-J finite-step RMSE"),
            (f"{local_error:.4f}", "local-J finite-step RMSE"),
            (
                f"{lens_toy['jacobian_finite_difference_max_error']:.2e}",
                "analytic-vs-finite-difference Jacobian error",
            ),
        ])),
        mo.Html(table(
            delta_rows,
            (
                ("output", "Output"),
                ("actual finite delta", "Actual logit delta"),
                ("average-J prediction", "Average-J prediction"),
                ("local-J prediction", "Local-J prediction"),
            ),
        )),
    ])
    return


@app.cell
def retained_lens_data(bundle):
    lens_section = bundle.section("lenses")
    has_retained_lens = "arms" in lens_section
    retained_series = []
    retained_legality_series = []
    retained_rows = []
    if has_retained_lens:
        colors = {"RR": "#3366a3", "HR": "#7450a8", "RH": "#a95f19", "HH": "#14735a"}
        for arm in ("RR", "HR", "RH", "HH"):
            arm_data = lens_section["arms"][arm]
            probabilities = arm_data["metrics"]["target_probability"]
            legal_mass = arm_data["metrics"]["legal_mass_before_masking"]
            layers = arm_data["layer_indices"]
            threshold = arm_data["first_target_probability_ge_0.5"]
            parity = arm_data["final_lens_parity"]
            retained_series.append({
                "name": arm,
                "color": colors[arm],
                "points": list(zip(layers, probabilities, strict=True)),
            })
            retained_legality_series.append({
                "name": arm,
                "color": colors[arm],
                "points": list(zip(layers, legal_mass, strict=True)),
            })
            retained_rows.append({
                "arm": arm,
                "median first legal P(target) >= 0.5": (
                    f"depth {threshold['median']:.0f} "
                    f"({threshold['finite_count']}/{threshold['count']} positions)"
                ),
                "final parity": (
                    f"{'pass' if parity['passed'] else 'fail'}; "
                    f"max |delta|={parity['maximum_absolute_error']:.1e}"
                ),
                "final mean legal P(target)": f"{probabilities[-1]:.3f}",
                "final pre-mask legal mass": f"{legal_mass[-1]:.1%}",
            })
    return (
        has_retained_lens,
        lens_section,
        retained_legality_series,
        retained_rows,
        retained_series,
    )


@app.cell
def retained_lens_view(
    callout,
    has_retained_lens,
    lens_section,
    line_chart,
    mo,
    retained_legality_series,
    retained_rows,
    retained_series,
    table,
):
    if has_retained_lens:
        retained_panel = mo.vstack([
            mo.Html(line_chart(
                retained_series,
                x_label="hooked depth",
                y_label="mean legal-renormalized P(recorded target)",
                y_domain=(0.0, 1.0),
                alt_text=(
                    "Mean legal-renormalized recorded-target probability across "
                    "eight exploratory positions for each Raw/Hero hybrid arm."
                ),
            )),
            mo.Html(line_chart(
                retained_legality_series,
                x_label="hooked depth",
                y_label="mean pre-mask legal-action mass",
                y_domain=(0.0, 0.12),
                alt_text=(
                    "Mean probability mass assigned to legal actions before masking "
                    "across eight exploratory positions for each hybrid arm."
                ),
            )),
            mo.Html(table(
                retained_rows,
                (
                    ("arm", "Hybrid arm"),
                    (
                        "median first legal P(target) >= 0.5",
                        "Median first legal-renormalized threshold depth",
                    ),
                    ("final parity", "Final readout parity"),
                    (
                        "final mean legal P(target)",
                        "Final mean legal-renormalized P(target)",
                    ),
                    (
                        "final pre-mask legal mass",
                        "Final mean pre-mask legal mass",
                    ),
                ),
            )),
            mo.Html(callout(
                "limit",
                "What the curve does not say",
                (
                    f"This is {lens_section['position_count']} development positions from "
                    f"{lens_section['run_id']}. The readout is the native nonlinear chess "
                    "policy head, not an LM unembedding. The target curve is conditional "
                    "on legal masking and renormalization; the second curve exposes the "
                    "much smaller pre-mask legal mass. No Jacobian transport was fitted, "
                    "and no population uncertainty or intervention licenses a mechanism claim."
                ),
            )),
        ])
    else:
        retained_panel = mo.Html(callout(
            "limit",
            "Toy mode has no Raw/Hero trace",
            "Switch to Snapshot for the frozen eight-position policy-lens pilot.",
        ))
    mo.vstack([
        mo.md("## The retained chess trace: a policy lens, not yet J-Lens"),
        retained_panel,
    ])
    return


@app.cell
def retained_claim(
    ClaimCard,
    Endpoint,
    EvidenceOperation,
    ResultStatus,
    TargetLevel,
    claim_card_html,
    has_retained_lens,
    mo,
):
    policy_lens_claim = ClaimCard(
        operation=EvidenceOperation.PREDICTION,
        target=TargetLevel.REPRESENTATION,
        scope="eight retained development positions; RR, HR, RH, and HH hybrid arms",
        endpoint=Endpoint.POLICY,
        estimand="mean legal-renormalized native-head recorded-target probability at each post-block normalized residual",
        unit="position by hooked depth and hybrid arm",
        grouping="descriptive position set; no population interval",
        intervention=None,
        controls=("final-layer readout parity", "pre-mask legal-action mass", "four encoder/head hybrid arms"),
        assumptions=("hook tensors match the final policy-head ABI", "target action identity is frozen"),
        uncertainty="none; exploratory development trace only",
        multiplicity="all four arms and depths displayed",
        allowed="The retained sample has the displayed mean legal-renormalized native-head target-probability curves.",
        excluded="The trace does not identify absence of information, a unique computation, planning, or a fitted Jacobian lens.",
        falsifier="Source verification or final-layer parity fails.",
        status=ResultStatus.DESCRIPTIVE,
    ).validate()
    if has_retained_lens:
        retained_claim_view = mo.Html(claim_card_html(policy_lens_claim))
    else:
        retained_claim_view = mo.md("The empirical claim card is hidden in Toy mode.")
    retained_claim_view
    return


@app.cell
def chess_translation(bundle, mo, table):
    architecture = bundle.section("architecture")
    chess_rows = [
        {
            "LM object": "token state h_l,t",
            "BT4 object": "64 square-token matrix H_l in R^(64 x d)",
            "design consequence": "a square average can erase origin-to-destination routing",
        },
        {
            "LM object": "linear unembedding W_U",
            "BT4 object": "dense + Mish + Q/K attention policy head + promotion mapping",
            "design consequence": "calling the native-head readout a literal logit lens would hide nonlinearity",
        },
        {
            "LM object": "vocabulary token",
            "BT4 object": "one of 1,858 encoded actions, then a position-specific legal subset",
            "design consequence": "report pre-mask and legal-renormalized endpoints separately",
        },
        {
            "LM object": "causally later text positions t' >= t",
            "BT4 object": "no causal square-token order; every square-to-square block may contribute",
            "design consequence": "define a square-block aggregation instead of borrowing temporal future",
        },
    ]
    mo.vstack([
        mo.md(r"""
        ## Translating J-Lens into chess without breaking the object

        BT4 does not end in a token-wise linear unembedding. Its policy head
        transforms all 64 square tokens and constructs move logits from
        origin–destination interactions. Its encoder is bidirectional: a move
        destination is not temporally later than its origin, and all
        square-to-square Jacobian blocks may matter. Therefore two chess
        adaptations are worth distinguishing:

        1. **State transport.** Estimate a structured Jacobian from \(H_l\) to
           \(H_L\), preserve square axes, then apply the unchanged native
           policy head.
        2. **Action-direction lens.** For action \(a\), estimate
           \(v_{l,a}=\mathbb E[\nabla_{H_l}\,\mathrm{logit}_a]\) over positions
           where \(a\) is legal, then test its held-out predictive and
           interventional behavior.

        The first stays closest to J-Lens. The second is cheaper and
        action-specific, but it changes the object from transported states to
        gradient directions.
        """),
        mo.Html(table(
            chess_rows,
            (
                ("LM object", "Language-model object"),
                ("BT4 object", "Chess counterpart"),
                ("design consequence", "What must change"),
            ),
        )),
        mo.md(
            f"Selected specimen ABI: {architecture['square_tokens']} square tokens, "
            f"width {architecture['width']}, {architecture['layers']} trunk blocks, "
            f"and {architecture['action_vocab']} encoded actions."
        ),
    ])
    return (architecture,)


@app.cell
def chess_protocol(architecture, metric_cards, mo, table):
    width = int(architecture["width"])
    layer_count = int(architecture["layers"])
    token_count = int(architecture["square_tokens"])
    one_dense_mib = width * width * 4 / (1024 ** 2)
    all_dense_mib = one_dense_mib * layer_count
    full_state_gib = (token_count * width) ** 2 * 4 / (1024 ** 3)
    protocol_rows = [
        {"stage": "1 · freeze", "operation": "freeze checkpoint, hook boundary, legal-action contrasts, corpus groups, and split", "failure checked": "selection leakage"},
        {"stage": "2 · parity", "operation": "apply the native head to H_L and require exact final-logit parity", "failure checked": "ABI or normalization mismatch"},
        {"stage": "3 · estimate", "operation": "fit only on fit groups; compare square-averaged d x d, token-structured, and preregistered low-rank estimates", "failure checked": "estimator instability or spatial-routing loss"},
        {"stage": "4 · predict", "operation": "evaluate held-out KL, legal target rank, calibration, and pre-mask illegal mass", "failure checked": "decoder fit without generalization"},
        {"stage": "5 · intervene", "operation": "patch or steer only after the predictive lens and controls are frozen", "failure checked": "readout mistaken for policy relevance"},
        {"stage": "6 · replicate", "operation": "new game/root groups and an independent estimator seed", "failure checked": "development-set story fitting"},
    ]
    mo.vstack([
        mo.md("### A compute-aware Raw/Hero protocol"),
        mo.Html(metric_cards([
            (
                f"{one_dense_mib:.1f} MiB",
                "one square-averaged dense d x d FP32 transport",
            ),
            (
                f"{all_dense_mib:.1f} MiB",
                "one square-averaged transport per trunk depth",
            ),
            (
                f"{full_state_gib:.1f} GiB",
                "one fully position-aware state Jacobian",
            ),
            ("many VJPs/JVPs", "dominant cost: derivative estimation, not storage"),
        ])),
        mo.Html(table(
            protocol_rows,
            (
                ("stage", "Stage"),
                ("operation", "Frozen operation"),
                ("failure checked", "Failure mode"),
            ),
        )),
        mo.callout(
            "A random low-rank sketch can make estimation practical on the local GPU, "
            "but it is a different estimator. Compare it with a dense or higher-rank "
            "reference on a small fit slice before scaling.",
            kind="warn",
        ),
    ])
    return


@app.cell
def retrieval_control(mo):
    retrieval_choice = mo.ui.radio(
        options={
            "The move is already represented, therefore the model will play it": "overclaim",
            "The final policy head assigns that move high probability from this hooked state on the sampled positions": "bounded",
            "The downstream blocks are unnecessary for this move": "overclaim",
        },
        value=None,
        label="Retrieval practice — which claim is licensed by an accurate native-head policy lens alone?",
    )
    retrieval_choice
    return (retrieval_choice,)


@app.cell
def retrieval_feedback(mo, retrieval_choice):
    if retrieval_choice.value is None:
        retrieval_feedback = mo.md("Commit an answer before reading the transfer prompt.")
    elif retrieval_choice.value == "bounded":
        retrieval_feedback = mo.callout(
            "Correct: the claim names the fixed readout, sample, state, and endpoint. "
            "It does not convert predictability into necessity.",
            kind="success",
        )
    else:
        retrieval_feedback = mo.callout(
            "That conclusion crosses from readout to policy relevance. Patch, ablate, "
            "or interchange the relevant state with matched controls before crossing.",
            kind="warn",
        )
    retrieval_feedback
    return


@app.cell
def transfer_and_sources(mo, module, resource_card, selected_mode):
    mo.vstack([
        mo.md(f"""
        ## Transfer assessment

        {module.transfer_assessment}

        Write a claim card for the early-layer readout, then add one
        intervention that distinguishes “readable by this instrument” from
        “required by the native policy.” Include legality, grouping, a
        matched-random control, a semantic placebo, and a falsifier.

        ### Source and status notes

        **Source register.** The identity/logit and tuned-lens comparison is
        grounded in the Tuned Lens paper. The average-Jacobian construction is
        grounded in the 2026 Jacobian-lens lab report and its official code.
        No primary source currently establishes a chess J-Lens result; the BT4
        protocol above is a proposed adaptation. See SOURCES.md for publication
        status and links.

        The retained Raw/Hero policy-lens panel is a source-bound exploratory
        pilot. The nonlinear constructed laboratory is a teaching mechanism.
        Neither is a fresh checkpoint run.
        """),
        mo.Html(resource_card(
            mode=selected_mode,
            runtime="instant toy recomputation; retained panel is a frozen eight-position extraction",
            device="CPU; NumPy only",
            determinism="seed 47; NumPy float64; analytic Jacobian checked by finite differences",
            sources="Tuned Lens; Jacobian Lens; retained raw_hero_literature_pilot_v2; course source register",
            limitations="no fitted Raw/Hero Jacobian transport, no population interval, and no causal-use conclusion",
        )),
        mo.Html('<div class="ci-callout ci-limit"><strong>Retrieval boundary</strong> A lens is a calibrated view into an intermediate state. It is not, by itself, an intervention on the model.</div>'),
    ])
    return


if __name__ == "__main__":
    app.run()
