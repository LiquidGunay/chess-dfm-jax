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
    animation_asset_root = repo_root / "curriculum/interpretability/assets/animations"

    from research.interpretability.course import get_module, load_course_bundle
    from research.interpretability.course.activations import load_optional_activation_gallery
    from research.interpretability.course.evidence import (
        ClaimCard,
        Endpoint,
        EvidenceOperation,
        ResultStatus,
        TargetLevel,
    )
    from research.interpretability.course.experiments import (
        toy_jl_projection,
        toy_representations,
        transform_representation,
    )
    from research.interpretability.course.stats import (
        linear_cka,
        mean_row_cosine,
        procrustes_similarity,
        symmetric_relative_l2,
    )
    from research.interpretability.course.ui import (
        animation_card,
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
        animation_asset_root,
        animation_card,
        callout,
        chessboard,
        claim_card_html,
        committed_form,
        get_module,
        line_chart,
        linear_cka,
        load_course_bundle,
        load_optional_activation_gallery,
        mean_row_cosine,
        metric_cards,
        mo,
        mode_badge,
        module_header,
        np,
        procrustes_similarity,
        repo_root,
        resource_card,
        square_heatmap,
        symmetric_relative_l2,
        table,
        toy_jl_projection,
        toy_representations,
        transform_representation,
    )


@app.cell
def title(get_module, mo, module_header):
    module = get_module(3)
    mo.Html(module_header(module))
    return (module,)


@app.cell
def opening(mo, module, table):
    objective_rows = [
        {"objective": objective, "performance": performance}
        for objective, performance in zip(
            module.objectives,
            (
                "Explain the numerator, denominator, centering, and invariance of four metrics.",
                "Commit metric directions before transforming a held-out matrix.",
                "Give one counterexample where parameters move little but representations move greatly.",
            ),
            strict=True,
        )
    ]
    mo.vstack([
        mo.md(r"""
        ## Coordinates are not concepts

        Let \(X,Y\in\mathbb R^{n\times d}\) contain paired observations as
        rows. A neuron-wise comparison assumes the two feature axes already
        mean the same thing. But an orthogonal change of basis can preserve all
        pairwise geometry while destroying coordinate-by-coordinate cosine.

        A similarity metric is therefore a bundle of invariances. Before
        interpreting a low score, ask what transformations the metric treats as
        equivalent and whether those match the scientific question.
        """),
        mo.Html(table(objective_rows, (("objective", "Objective"), ("performance", "Observable success")))),
    ])
    return


@app.cell
def prerequisite(committed_form, mo):
    response = mo.ui.text_area(
        label="Prerequisite diagnostic — levels of evidence",
        placeholder="A module swap changes policy. Does that identify a semantic feature or circuit? State the strongest conclusion and one remaining alternative.",
    )
    prerequisite_form = committed_form(
        mo, {"localization": response}, submit_label="Commit prerequisite response", min_words=6
    )
    prerequisite_form
    return (prerequisite_form,)


@app.cell
def prerequisite_feedback(mo, prerequisite_form):
    mo.stop(prerequisite_form.value is None, mo.md("Answer the prerequisite diagnostic to reveal the bridge."))
    mo.callout("A swap localizes where the observed functional delta is expressed under that interface. Distributed, redundant, or compensating mechanisms remain; geometry is the next descriptive question, not a circuit verdict.", kind="info")
    return


@app.cell
def mode_controls(mo):
    evidence_mode = mo.ui.radio(
        options={
            "Snapshot — retained Raw/Hero geometry": "snapshot",
            "Toy — constructed paired representations": "toy",
            "Source — verify retained geometry records": "source",
        },
        value="Snapshot — retained Raw/Hero geometry",
        label="Evidence mode",
    )
    verify_source = mo.ui.run_button(label="Verify retained sources", kind="warn")
    mo.hstack([evidence_mode, verify_source], justify="start", gap=1)
    return evidence_mode, verify_source


@app.cell
def load_mode(evidence_mode, load_course_bundle, mo, verify_source):
    selected_mode = evidence_mode.value
    mo.stop(selected_mode == "source" and not verify_source.value, mo.callout("Press the verification button to enter source mode.", kind="warn"))
    bundle = load_course_bundle(selected_mode)
    return bundle, selected_mode


@app.cell
def mode_view(bundle, mo, mode_badge, selected_mode):
    mo.Html(mode_badge(selected_mode, bundle.evidence_scope))
    return


@app.cell
def real_activation_controls(
    bundle,
    load_optional_activation_gallery,
    mo,
    repo_root,
    selected_mode,
):
    if bundle.is_empirical:
        activation_gallery = load_optional_activation_gallery(
            repo_root,
            verify_sources=selected_mode == "source",
        )
        _position_options = (
            activation_gallery.position_options()
            if activation_gallery is not None
            else {"Local activation pack is not materialized": 0}
        )
    else:
        activation_gallery = None
        _position_options = {"Constructed mode has no Raw/Hero tensor": 0}
    activation_position = mo.ui.dropdown(
        options=_position_options,
        value=next(iter(_position_options)),
        label="Real chess position",
    )
    activation_layer = mo.ui.number(
        0,
        14,
        step=1,
        value=14,
        label="Residual layer",
    )
    activation_channel_rank = mo.ui.number(
        0,
        7,
        step=1,
        value=0,
        label="Channel rank by mean |Hero − Raw| at this layer",
    )
    mo.vstack([
        mo.md("## Put an actual activation under the microscope"),
        mo.md(
            "The default specimen below is not simulated: select one retained "
            "public chess position, one of all 15 layers, and a real residual "
            "coordinate from the Raw and Hero checkpoints."
        ),
        mo.hstack(
            [activation_position, activation_layer, activation_channel_rank],
            justify="start",
            gap=1,
        ),
    ])
    return (
        activation_channel_rank,
        activation_gallery,
        activation_layer,
        activation_position,
    )


@app.cell
def real_activation_microscope(
    activation_channel_rank,
    activation_gallery,
    activation_layer,
    activation_position,
    bundle,
    callout,
    chessboard,
    line_chart,
    metric_cards,
    mo,
    np,
    resource_card,
    square_heatmap,
    table,
):
    if bundle.is_empirical and activation_gallery is not None:
        gallery_activation = activation_gallery.activation_view(
            position=int(activation_position.value),
            layer=int(activation_layer.value),
            channel_rank=int(activation_channel_rank.value),
        )
        _record = gallery_activation["position"]
        _delta = np.asarray(gallery_activation["delta_channel"])
        _raw = np.asarray(gallery_activation["raw_channel"])
        _hero = np.asarray(gallery_activation["hero_channel"])
        _largest = np.argsort(-np.abs(_delta), kind="stable")[:8]
        _square_rows = [
            {
                "square": f"{chr(ord('a') + int(_index) % 8)}{int(_index) // 8 + 1}",
                "Raw": f"{_raw[_index]:+.4f}",
                "Hero": f"{_hero[_index]:+.4f}",
                "Hero − Raw": f"{_delta[_index]:+.4f}",
            }
            for _index in _largest
        ]
        _layer_points = list(enumerate(gallery_activation["per_layer_delta_rms"]))
        _cosine_points = list(
            enumerate(gallery_activation["per_layer_mean_token_cosine"])
        )
        _selected_layer = int(activation_layer.value)
        _selected_delta = gallery_activation["per_layer_delta_rms"][_selected_layer]
        _selected_cosine = gallery_activation["per_layer_mean_token_cosine"][
            _selected_layer
        ]
        activation_microscope_view = mo.vstack([
            mo.Html(metric_cards([
                (_record["puzzle_id"], "public puzzle specimen"),
                (_record["target_move"], "recorded target move"),
                (_selected_layer, "selected layer"),
                (gallery_activation["channel"], "actual residual coordinate"),
                (f"{_selected_delta:.4f}", "position-level activation Δ RMS"),
                (f"{_selected_cosine:.4f}", "mean paired-token cosine"),
            ])),
            mo.hstack([
                mo.Html(chessboard(_record["fen"])),
                mo.vstack([
                    mo.md(
                        f"**Themes:** {_record['themes']}  \n"
                        f"**Solution:** `{_record['solution_uci']}`  \n"
                        "Activations use BT4's side-to-move token frame internally; "
                        "these maps have been returned to absolute board squares."
                    ),
                    mo.Html(table(
                        _square_rows,
                        (
                            ("square", "Largest-change square"),
                            ("Raw", "Raw activation"),
                            ("Hero", "Hero activation"),
                            ("Hero − Raw", "Difference"),
                        ),
                    )),
                ]),
            ], justify="start", gap=1),
            mo.ui.tabs({
                "Raw channel values": mo.Html(square_heatmap(
                    gallery_activation["raw_channel"],
                    alt_text=(
                        f"Actual Raw residual coordinate {gallery_activation['channel']} "
                        f"at layer {_selected_layer}, mapped to absolute squares."
                    ),
                    signed=True,
                )),
                "Hero channel values": mo.Html(square_heatmap(
                    gallery_activation["hero_channel"],
                    alt_text=(
                        f"Actual Hero residual coordinate {gallery_activation['channel']} "
                        f"at layer {_selected_layer}, mapped to absolute squares."
                    ),
                    signed=True,
                )),
                "Hero − Raw channel": mo.Html(square_heatmap(
                    gallery_activation["delta_channel"],
                    alt_text=(
                        f"Hero minus Raw residual coordinate {gallery_activation['channel']} "
                        f"at layer {_selected_layer}."
                    ),
                    signed=True,
                )),
                "Whole-vector Δ RMS": mo.Html(square_heatmap(
                    gallery_activation["delta_square_rms"],
                    alt_text=(
                        f"RMS difference across all 1024 residual coordinates per square "
                        f"at layer {_selected_layer}."
                    ),
                    signed=False,
                )),
            }),
            mo.Html(line_chart(
                [{
                    "name": "position-level activation Δ RMS",
                    "points": _layer_points,
                    "color": "#a95f19",
                }],
                x_label="BT4 layer",
                y_label="RMS(Hero − Raw)",
                alt_text=(
                    "For the selected real chess position, residual activation "
                    "difference across all squares and channels at every layer."
                ),
            )),
            mo.Html(line_chart(
                [{
                    "name": "mean paired-token cosine",
                    "points": _cosine_points,
                    "color": "#3366a3",
                }],
                x_label="BT4 layer",
                y_label="cosine",
                y_domain=(-1.0, 1.0),
                alt_text=(
                    "For the selected real chess position, mean Raw versus Hero "
                    "token cosine at every layer."
                ),
            )),
            mo.Html(callout(
                "limit",
                "A coordinate is visible, not automatically meaningful",
                "Channel selection is based on mean absolute Raw/Hero difference. "
                "That makes the coordinate interesting for inspection but does not "
                "make it a chess concept, a monosemantic feature, or a policy mediator.",
            )),
            mo.Html(resource_card(
                mode="real activation tensor · frozen development gallery",
                runtime="instant from the 30.8 MB local activation pack",
                device="gallery was extracted on the local GTX 1660 Ti",
                determinism="float32 inference; float16 stored residuals; fixed 8-position selection",
                sources=(
                    f"{activation_gallery.run_id}; data SHA-256 "
                    f"{activation_gallery.data_sha256}"
                ),
                limitations=(
                    "Eight deliberately selected development positions; channel ranking "
                    "is descriptive and no population interval is implied."
                ),
            )),
        ])
    elif bundle.is_empirical:
        gallery_activation = None
        activation_microscope_view = mo.Html(callout(
            "limit",
            "The local real-activation pack is absent",
            "This public repository intentionally omits the license-gated 30.8 MB "
            "tensor pack. Rebuild it locally from authorized sources to enable the "
            "interactive Raw/Hero activation microscope.",
        ))
    else:
        gallery_activation = None
        activation_microscope_view = mo.Html(callout(
            "neutral",
            "Toy mode keeps its boundary",
            "Switch to Snapshot or source-verified mode to inspect checkpoint activations. "
            "The constructed rotation lab below remains useful because its mechanism is known.",
        ))
    activation_microscope_view
    return


@app.cell
def derivations(mo, table):
    metric_rows = [
        {"metric": "mean row cosine", "definition": "meanᵢ xᵢ·yᵢ/(‖xᵢ‖‖yᵢ‖)", "invariant to": "positive rowwise scale", "sensitive to": "feature rotation / basis"},
        {"metric": "centered linear CKA", "definition": "‖XᵀY‖²_F/(‖XᵀX‖_F‖YᵀY‖_F)", "invariant to": "orthogonal basis + isotropic scale", "sensitive to": "anisotropic scale, lost dimensions"},
        {"metric": "Procrustes similarity", "definition": "‖XᵀY‖_* /(‖X‖_F‖Y‖_F)", "invariant to": "optimal orthogonal alignment", "sensitive to": "non-isometric distortion"},
        {"metric": "symmetric relative L2", "definition": "‖X−Y‖_F / ((‖X‖_F+‖Y‖_F)/2)", "invariant to": "nothing beyond paired ordering", "sensitive to": "scale and basis"},
    ]
    mo.vstack([
        mo.md("## Four metrics, four questions"),
        mo.Html(table(metric_rows, (("metric", "Metric"), ("definition", "Definition (center X,Y for CKA/Procrustes)"), ("invariant to", "Treats as same"), ("sensitive to", "Detects")))),
        mo.md("SVCCA adds a variance-preserving subspace selection before canonical correlation. That can suppress noisy dimensions, but its variance cutoff becomes part of the estimand."),
    ])
    return


@app.cell
def worked_rotation(
    linear_cka,
    mean_row_cosine,
    np,
    procrustes_similarity,
    symmetric_relative_l2,
):
    worked_x = np.asarray([[-1.0, 0.0], [1.0, 0.0]])
    quarter_turn = np.asarray([[0.0, -1.0], [1.0, 0.0]])
    worked_y = worked_x @ quarter_turn
    worked_scores = {
        "mean row cosine": mean_row_cosine(worked_x, worked_y),
        "centered linear CKA": linear_cka(worked_x, worked_y),
        "Procrustes similarity": procrustes_similarity(worked_x, worked_y),
        "symmetric relative L2": symmetric_relative_l2(worked_x, worked_y),
    }
    return quarter_turn, worked_scores, worked_x, worked_y


@app.cell
def worked_rotation_view(
    mo,
    quarter_turn,
    table,
    worked_scores,
    worked_x,
    worked_y,
):
    worked_rows = [
        {"metric": name, "value": f"{value:.6f}"}
        for name, value in worked_scores.items()
    ]
    mo.vstack([
        mo.md(r"""
        ### Hand-worked two-point rotation

        Take
        \(X=[[-1,0],[1,0]]\) and rotate its feature axis by 90°:
        \(Q=[[0,-1],[1,0]]\), so \(Y=XQ=[[0,1],[0,-1]]\).

        Every paired row is perpendicular, hence mean row cosine is 0. But
        \(XX^\top=YY^\top\), so centered linear CKA is 1; Procrustes rotates
        \(Y\) back exactly and is also 1. The unaligned distance is
        \(\lVert X-Y\rVert_F / \lVert X\rVert_F=\sqrt 2\).

        The minimal computation below is deliberately visible. In
        `marimo edit`, change \(Q\) and predict the table before rerunning.

        ```python
        X = np.array([[-1., 0.], [1., 0.]])
        Q = np.array([[0., -1.], [1., 0.]])
        Y = X @ Q
        scores = {
            "row cosine": mean_row_cosine(X, Y),
            "linear CKA": linear_cka(X, Y),
            "Procrustes": procrustes_similarity(X, Y),
            "relative L2": symmetric_relative_l2(X, Y),
        }
        ```
        """),
        mo.md(
            f"Computed matrices: `X={worked_x.tolist()}`, "
            f"`Q={quarter_turn.tolist()}`, `Y={worked_y.tolist()}`."
        ),
        mo.Html(table(worked_rows, (("metric", "Metric"), ("value", "Computed value")))),
    ])
    return


@app.cell
def prediction_controls(committed_form, mo):
    transform_kind = mo.ui.dropdown(
        options={
            "Identity": "identity",
            "Orthogonal rotation": "rotate",
            "Feature permutation": "permute",
            "Anisotropic feature scaling": "scale",
            "Additive noise": "noise",
            "Duplicate half the dimensions": "duplicate",
        },
        value="Orthogonal rotation",
        label="Transformation",
    )
    strength = mo.ui.number(0.0, 1.0, step=0.1, value=1.0, label="Transformation strength")
    pairing = mo.ui.radio(
        options={"Correct paired rows": "paired", "Shuffle Y rows (null)": "shuffled"},
        value="Correct paired rows",
        label="Observation pairing",
    )
    predicted_stable = mo.ui.multiselect(
        options=["row cosine", "linear CKA", "Procrustes", "relative L2"],
        value=[],
        label="Predict which metrics remain most similarity-like",
    )
    prediction_form = committed_form(
        mo,
        {
            "transformation": transform_kind,
            "strength": strength,
            "pairing": pairing,
            "stable_metrics": predicted_stable,
        },
        submit_label="Commit prediction and compute",
        allow_empty=("stable_metrics",),
    )
    prediction_form
    return (prediction_form,)


@app.cell
def metric_experiment(
    linear_cka,
    mean_row_cosine,
    mo,
    np,
    prediction_form,
    procrustes_similarity,
    symmetric_relative_l2,
    toy_representations,
    transform_representation,
):
    mo.stop(prediction_form.value is None)
    base_representation, _ = toy_representations()
    transformed_representation = transform_representation(
        base_representation,
        prediction_form.value["transformation"],
        strength=prediction_form.value["strength"],
    )
    if prediction_form.value["pairing"] == "shuffled":
        transformed_representation = transformed_representation[np.random.default_rng(303).permutation(transformed_representation.shape[0])]
    geometry_scores = {
        "row cosine": mean_row_cosine(base_representation, transformed_representation),
        "linear CKA": linear_cka(base_representation, transformed_representation),
        "Procrustes": procrustes_similarity(base_representation, transformed_representation),
        "relative L2": symmetric_relative_l2(base_representation, transformed_representation),
    }
    return (geometry_scores,)


@app.cell
def metric_feedback(
    callout,
    geometry_scores,
    metric_cards,
    mo,
    prediction_form,
):
    score_cards = [
        (f"{geometry_scores[name]:.3f}", name)
        for name in ("row cosine", "linear CKA", "Procrustes", "relative L2")
    ]
    expected = {
        name
        for name in ("row cosine", "linear CKA", "Procrustes")
        if geometry_scores[name] >= 0.95
    }
    if geometry_scores["relative L2"] <= 0.10:
        expected.add("relative L2")
    prediction_match = expected == set(prediction_form.value["stable_metrics"])
    mo.vstack([
        mo.Html(metric_cards(score_cards)),
        mo.Html(callout(
            "positive" if prediction_match else "limit",
            "Prediction calibrated" if prediction_match else "Revisit the invariance table",
            f"Under the frozen teaching thresholds, similarity metrics count as stable at ≥0.95 and relative L2 at ≤0.10. Expected: {', '.join(sorted(expected)) or 'none'}. A full orthogonal basis change preserves CKA and Procrustes; shuffled rows destroy correspondence. Relative L2 is a distance, so lower—not higher—is more similar.",
        )),
    ])
    return


@app.cell
def gauge_lesson(mo):
    mo.md(r"""
    ## Why a rotation is not a cosmetic trick

    Suppose the next layer begins with weights \(W\). Replacing a hidden state
    \(h\) by \(hQ\) and the next weights by \(Q^\top W\), for orthogonal
    \(Q\), leaves the composed linear function unchanged. Individual feature
    coordinates move, while function and inner-product geometry can remain.

    This is a **gauge freedom**. Comparing neuron 417 across independently
    trained models assumes a gauge alignment that training never promised.
    Model diffing must either use an invariant statistic, fit an alignment only
    on discovery data, or explicitly argue that parameter lineage fixes a basis.
    """)
    return


@app.cell
def jl_visual_essay(animation_asset_root, animation_card, mo):
    mo.vstack([
        mo.md(r"""
        ## Visual essay: what random projection actually promises

        A basis rotation preserves every pairwise distance exactly. The
        Johnson–Lindenstrauss lemma makes a weaker but surprisingly useful
        statement: for a finite set of points, a suitable random linear map
        can reduce dimension while approximately preserving all pairwise
        distances. The required target dimension scales logarithmically with
        the number of points and polynomially with the inverse distortion.

        Watch the distance matrix, not the coordinates. The points are free to
        look completely different after projection; the guarantee concerns
        finite-set distances.
        """),
        mo.Html(animation_card(
            animation_asset_root / "johnson_lindenstrauss.mp4",
            animation_asset_root / "johnson_lindenstrauss.png",
            animation_asset_root / "johnson_lindenstrauss.vtt",
            title="Johnson–Lindenstrauss as a distance-preservation story",
            description="A 64-dimensional finite point cloud is randomly projected to 12 dimensions while its pairwise distance matrix is compared.",
            transcript=(
                "The animation begins with seven points represented only through their "
                "pairwise distance matrix in 64 dimensions. Both matrices use one "
                "shared color scale. A Gaussian random matrix projects the points into "
                "12 dimensions. The second matrix remains visually "
                "similar, and the maximum relative distance distortion for this sampled "
                "map is displayed. The closing equation states the asymptotic dimension "
                "bound. A final warning separates this finite-set geometric theorem from "
                "a theory of which features a trained neural network learns."
            ),
        )),
    ])
    return


@app.cell
def jl_controls(mo):
    jl_point_count = mo.ui.number(
        8, 64, step=4, value=24, label="Finite-set point count"
    )
    jl_target_dim = mo.ui.number(
        2, 48, step=2, value=12, label="Projected dimension (ambient = 96)"
    )
    jl_tolerance = mo.ui.number(
        0.05, 0.8, step=0.05, value=0.35, label="Chosen relative-distortion tolerance"
    )
    jl_seed = mo.ui.number(0, 1000, step=1, value=43, label="Projection seed")
    mo.hstack(
        [jl_point_count, jl_target_dim, jl_tolerance, jl_seed],
        justify="start",
    )
    return jl_point_count, jl_seed, jl_target_dim, jl_tolerance


@app.cell
def run_jl_lab(jl_point_count, jl_seed, jl_target_dim, toy_jl_projection):
    jl_result = toy_jl_projection(
        point_count=int(jl_point_count.value),
        ambient_dim=96,
        target_dim=int(jl_target_dim.value),
        seed=int(jl_seed.value),
    )
    return (jl_result,)


@app.cell
def jl_lab_view(jl_result, jl_tolerance, line_chart, metric_cards, mo, np):
    jl_order = np.argsort(jl_result["original_distances"])
    jl_original = np.asarray(jl_result["original_distances"])[jl_order]
    jl_projected = np.asarray(jl_result["projected_distances"])[jl_order]
    jl_within = float(
        np.mean(np.asarray(jl_result["relative_distortion"]) <= jl_tolerance.value)
    )
    jl_series = [
        {
            "name": "original 96D distance",
            "color": "#3366a3",
            "points": list(enumerate(jl_original)),
        },
        {
            "name": "projected distance",
            "color": "#a95f19",
            "points": list(enumerate(jl_projected)),
        },
    ]
    mo.vstack([
        mo.Html(metric_cards([
            (
                f"{jl_result['maximum_relative_distortion']:.1%}",
                "maximum pairwise relative distortion",
            ),
            (
                f"{jl_result['median_relative_distortion']:.1%}",
                "median pairwise relative distortion",
            ),
            (
                f"{jl_within:.1%}",
                f"pairs within chosen {jl_tolerance.value:.0%} tolerance",
            ),
        ])),
        mo.Html(line_chart(
            jl_series,
            x_label="pair index, sorted by original distance",
            y_label="Euclidean distance",
            alt_text=(
                "Original and projected pairwise distances for every distinct "
                "pair in one sampled Gaussian random projection."
            ),
        )),
        mo.callout(
            "This cell samples one map. It is not a numerical proof of the "
            "Johnson–Lindenstrauss lemma, and a good median does not imply that "
            "every pair meets the chosen tolerance. Change both seed and target "
            "dimension; the maximum is the relevant finite-set diagnostic.",
            kind="warn",
        ),
    ])
    return


@app.cell
def real_geometry_panel(
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
        geometry = bundle.section("geometry")
        depth = geometry["activation_depth"]
        cka_series = [{"name": "corresponding-layer CKA", "points": [(row["layer"], row["corresponding_layer_cka"]) for row in depth], "color": "#3366a3"}]
        l2_series = [{"name": "symmetric relative L2", "points": [(row["layer"], row["symmetric_relative_l2"]) for row in depth], "color": "#a95f19"}]
        contributor_rows = [
            {"parameter": row["name"], "delta energy": f"{row['delta_energy_fraction']:.1%}", "relative ΔL2": f"{row['relative_delta_l2']:.4f}"}
            for row in geometry["largest_delta_contributors"][:6]
        ]
        first_layer = depth[0]
        final_layer = depth[-1]
        geometry_card = ClaimCard(
            operation=EvidenceOperation.OBSERVATION,
            target=TargetLevel.REPRESENTATION,
            scope="bounded retained Raw/Hero activation sketch on paired development positions",
            endpoint=Endpoint.REPRESENTATION,
            estimand="centered linear CKA and symmetric relative L2 between corresponding layer activations",
            unit="paired position-token activation row",
            grouping="paired examples; bounded sketch without a population interval",
            intervention=None,
            controls=("corresponding versus best-layer matching", "parameter-delta spectrum", "identical input ordering"),
            assumptions=("activation extraction and centering are identical across models", "sample pairing is intact"),
            uncertainty="descriptive layer curve; no inferential interval",
            multiplicity="15-layer exploratory scan",
            allowed=(f"In the retained sketch, corresponding-layer CKA is {first_layer['corresponding_layer_cka']:.3f} at layer 0 and {final_layer['corresponding_layer_cka']:.3f} at layer 14."),
            excluded="A low CKA score does not identify a semantic change or explain the policy delta.",
            falsifier="A checksum-matched re-extraction that does not reproduce the layer curve within the frozen numerical tolerance.",
            status=ResultStatus.DESCRIPTIVE,
        )
        empirical_geometry_view = mo.vstack([
            mo.md("## Frozen Raw/Hero dense diff"),
            mo.Html(line_chart(cka_series, x_label="BT4 layer", y_label="linear CKA", y_domain=(0.0, 1.0), alt_text="Corresponding Raw and Hero layer CKA declines from layer zero toward layer fourteen.")),
            mo.Html(line_chart(l2_series, x_label="BT4 layer", y_label="symmetric relative L2", y_domain=(0.0, 1.25), alt_text="Symmetric relative L2 rises overall across Raw and Hero layers.")),
            mo.Html(table(contributor_rows, (("parameter", "Largest Δ contributor"), ("delta energy", "Share of squared Δ"), ("relative ΔL2", "Relative ΔL2")))),
            mo.Html(claim_card_html(geometry_card)),
            mo.Html(callout("limit", "Small parameter cosine can coexist with large activation change", "All-parameter cosine is about 0.999999, yet late-layer CKA is low. High-dimensional parameter summaries can hide structured updates and nonlinear amplification.")),
        ])
    else:
        empirical_geometry_view = mo.Html(callout("neutral", "Toy boundary", "The transformation lab has known ground truth. It does not reproduce or claim a Raw/Hero layer curve."))
    empirical_geometry_view
    return


@app.cell
def learning_rate_audit(bundle, callout, mo):
    if bundle.is_empirical:
        verdict = bundle.section("geometry")["encoder_lr_verdict"]
        audit_view = mo.Html(callout(
            "limit",
            "A useful negative conclusion about Hero training",
            (
                f"The encoder update budget was severely constrained: {verdict['encoder_update_budget_was_severely_constrained']}. "
                f"But lower LR alone is proven causal: {verdict['lower_lr_alone_proven_causal']}; precision and LR are confounded: {verdict['precision_and_lr_are_confounded']}. "
                "A controlled crossed rerun is the promotion experiment."
            ),
        ))
    else:
        audit_view = mo.md("")
    audit_view
    return


@app.cell
def parameter_vs_function(mo, table):
    distinction_rows = [
        {"level": "parameter", "question": "Which tensors moved, by how much, and in what spectrum?", "control": "same checkpoint; independent seed; random low-rank update"},
        {"level": "representation", "question": "How did paired activation geometry change?", "control": "rotation, shuffled pairs, aligned versus invariant metrics"},
        {"level": "function", "question": "How did policy/value behavior change?", "control": "identical input and hybrid/patch interventions"},
    ]
    mo.Html(table(distinction_rows, (("level", "Level"), ("question", "Estimand"), ("control", "Useful null / control"))))
    return


@app.cell
def assessments(committed_form, mo, module):
    metric_transfer = mo.ui.text_area(label="Transfer 1 — metric choice", placeholder="Two encoders differ by an unknown rotation; downstream behavior is unchanged. Which metric and null do you choose?")
    transformation_transfer = mo.ui.text_area(label="Transfer 2 — prediction", placeholder="Predict cosine, CKA, and relative-L2 behavior under uniform scaling and under shuffled pairs.")
    level_transfer = mo.ui.text_area(label="Transfer 3 — inference", placeholder="Weights have cosine 0.999999 but policies differ. Give two mathematically compatible explanations and a next experiment.")
    assessment_form = committed_form(
        mo,
        {"metric": metric_transfer, "transformation": transformation_transfer, "inference": level_transfer},
        submit_label="Commit responses and reveal geometry rubric",
        min_words=5,
    )
    mo.vstack([mo.md(f"## Unseen transfer assessment\n\n{module.transfer_assessment}"), assessment_form])
    return (assessment_form,)


@app.cell
def assessment_feedback(assessment_form, mo):
    mo.stop(assessment_form.value is None, mo.md("Submit all objectives before revealing."))
    word_counts = [len(assessment_form.value[key].split()) for key in ("metric", "transformation", "inference")]
    mo.md(f"""
    **Rubric (response lengths {word_counts}):** a strong answer chooses CKA or
    held-out Procrustes for unknown rotations, preserves paired rows, predicts
    uniform-scale invariance for CKA/cosine but not relative L2, predicts all
    paired metrics lose their matched signal and move toward shuffled baselines,
    and separates structured high-dimensional updates/nonlinear amplification
    from semantic stories.
    """)
    return


@app.cell
def close(bundle, callout, mo, resource_card):
    mo.vstack([
        mo.Html(callout("positive", "Positive result", "Invariant metrics can correctly recognize a pure basis rotation that coordinate metrics label as change.")),
        mo.Html(callout("limit", "What this does not show", "Representational similarity is neither functional equivalence nor semantic identity. Alignment fitted and scored on the same sample can overfit.")),
        mo.md("""
        ## Retrieval before module 04

        1. Which metric is invariant to an orthogonal feature rotation?
        2. Why must alignment be selected away from the final evaluation set?
        3. Give a case where parameter, representation, and behavior metrics disagree.

        **Sources:** Kornblith et al. (ICML 2019) introduces CKA for
        representation comparison; Johnson and Lindenstrauss (1984) provides
        the finite-set random-projection theorem. Local Raw/Hero records are
        exploratory. See the
        [course source register](../SOURCES.md).
        """),
        mo.Html(resource_card(
            mode=bundle.mode.value,
            runtime="< 3 seconds on 160×16 constructed matrices",
            device="CPU / NumPy; no checkpoint or network",
            determinism="toy seed 13; transform seed 101; shuffle seed 303; NumPy float64",
            sources=f"content identity {bundle.integrity_sha256[:16]}…",
            limitations="Constructed invariance lab plus bounded retained activation/parameter sketches.",
        )),
    ])
    return


if __name__ == "__main__":
    app.run()
