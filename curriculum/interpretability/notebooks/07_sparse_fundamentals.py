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
    from research.interpretability.course.experiments import toy_sparse_frontier
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
        callout,
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
        toy_sparse_frontier,
    )


@app.cell
def title(get_module, mo, module_header):
    module = get_module(7)
    mo.Html(module_header(module))
    return (module,)


@app.cell
def opening(mo, module, table):
    objective_rows = [
        {"objective": objective, "mastery evidence": evidence}
        for objective, evidence in zip(
            module.objectives,
            (
                "Classify an unseen sparse module only from its input, target, and replacement boundary.",
                "Compute NMSE, active fraction, dead fraction, cosine, and downstream replacement change.",
                "Choose a Pareto point without looking at feature names or cherry-picked examples.",
            ),
            strict=True,
        )
    ]
    mo.vstack([
        mo.md(r"""
        ## Sparsity is an engineering constraint, not a semantic certificate

        A dense vector can superpose more candidate factors than it has axes.
        A sparse dictionary writes an observed activation approximately as

        \[
        x \approx \hat x = \sum_{j=1}^{m} f_j d_j,
        \qquad f_j\ge 0,\quad \|f\|_0\ll m.
        \]

        Overcompleteness (\(m>d\)) creates room for many directions; sparsity
        makes only a few active per token. Neither ensures that one feature is
        one human concept, that the dictionary is unique, or that replacing the
        native computation preserves behavior.
        """),
        mo.Html(table(objective_rows, (("objective", "Objective"), ("mastery evidence", "Formative evidence")))),
    ])
    return


@app.cell
def prerequisite(committed_form, mo):
    response = mo.ui.text_area(
        label="Prerequisite diagnostic — replacement boundary",
        placeholder="A learned module consumes one hook and reconstructs another. Why are equal tensor widths insufficient to decide what it replaces?",
    )
    prerequisite_form = committed_form(
        mo, {"boundary": response}, submit_label="Commit prerequisite response", min_words=6
    )
    prerequisite_form
    return (prerequisite_form,)


@app.cell
def prerequisite_feedback(mo, prerequisite_form):
    mo.stop(prerequisite_form.value is None, mo.md("Answer the prerequisite diagnostic to reveal the bridge."))
    mo.callout("The input hook, target hook, normalization point, residual algebra, token semantics, and layer define the replacement estimand. Width compatibility alone says only that substitution is syntactically possible.", kind="info")
    return


@app.cell
def mode_controls(mo):
    evidence_mode = mo.ui.radio(
        options={
            "Snapshot — retained Raw/Hero transcoder metrics": "snapshot",
            "Toy — known sparse mixture": "toy",
            "Source — verify retained sparse records": "source",
        },
        value="Snapshot — retained Raw/Hero transcoder metrics",
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
def architecture_table(mo, table):
    architecture_rows = [
        {"model": "residual SAE", "input x": "one residual hook", "target y": "same activation x", "replacement question": "can x̂ substitute for x?", "feature meaning": "decoder direction in residual space"},
        {"model": "transcoder", "input x": "component input", "target y": "component output", "replacement question": "can ŷ substitute for native component output?", "feature meaning": "sparse input-conditioned output contribution"},
        {"model": "crosscoder", "input x": "multiple models/layers", "target y": "joint reconstruction(s)", "replacement question": "depends on decoder and shared/private design", "feature meaning": "shared or model-specific aligned dictionary element"},
        {"model": "LoRSA", "input x": "L14 hook_attn_in [B,64,1024]", "target y": "L14 hook_attn_out [B,64,1024]", "replacement question": "can learned sparse attention branch replace native attention branch?", "feature meaning": "sparse OV feature inside a separately learned attention computation"},
    ]
    mo.vstack([
        mo.md("## Name the boundary before the acronym"),
        mo.Html(table(architecture_rows, (("model", "Sparse architecture"), ("input x", "Consumes"), ("target y", "Fits"), ("replacement question", "Fidelity estimand"), ("feature meaning", "What one feature is")))),
    ])
    return


@app.cell
def architecture_prediction(committed_form, mo):
    unseen_boundary = mo.ui.dropdown(
        options={
            "input and target are the same residual hook": "sae",
            "input is MLP input; target is MLP output": "transcoder",
            "jointly reconstructs matched activations from two models": "crosscoder",
            "uses learned attention to map attention input to branch output": "lorsa",
        },
        value=None,
        label="Unseen sparse module",
    )
    architecture_answer = mo.ui.dropdown(
        options=["residual SAE", "transcoder", "crosscoder", "LoRSA"],
        value=None,
        label="Classification",
    )
    architecture_form = committed_form(
        mo,
        {"boundary": unseen_boundary, "classification": architecture_answer},
        submit_label="Commit architecture classification",
    )
    architecture_form
    return (architecture_form,)


@app.cell
def architecture_feedback(architecture_form, callout, mo):
    answer_key = {"sae": "residual SAE", "transcoder": "transcoder", "crosscoder": "crosscoder", "lorsa": "LoRSA"}
    mo.stop(architecture_form.value is None, mo.md("Commit before revealing."))
    boundary = architecture_form.value["boundary"]
    architecture_ok = architecture_form.value["classification"] == answer_key[boundary]
    mo.Html(callout(
        "positive" if architecture_ok else "limit",
        "Boundary classified" if architecture_ok else "Ignore equal widths; inspect input and target",
        f"The correct class is **{answer_key[boundary]}**. Two tensors can both be width 1024 while representing different residual or branch locations.",
    ))
    return


@app.cell
def hand_worked_sparse_code(mo, np, table):
    target = np.asarray([2.0, 1.0])
    dictionary = np.eye(2)
    preactivations = np.asarray([2.0, 0.5])
    top_k = 1
    selected = np.argpartition(preactivations, -top_k)[-top_k:]
    code = np.zeros_like(preactivations)
    code[selected] = np.maximum(preactivations[selected], 0.0)
    hand_reconstruction = code @ dictionary
    mse = float(np.mean((hand_reconstruction - target) ** 2))
    variance = float(np.mean((target - np.mean(target)) ** 2))
    normalized_mse = mse / variance
    cosine = float(hand_reconstruction @ target / (np.linalg.norm(hand_reconstruction) * np.linalg.norm(target)))
    rows = [
        {"object": "target y", "value": target.tolist(), "meaning": "native component output to replace"},
        {"object": "encoder preactivations", "value": preactivations.tolist(), "meaning": "scores before top-k and ReLU"},
        {"object": "top-1 code f", "value": code.tolist(), "meaning": "one nonzero of two coordinates"},
        {"object": "decoded ŷ=fD", "value": hand_reconstruction.tolist(), "meaning": "sparse replacement output"},
    ]
    mo.vstack([
        mo.md(r"""
        ## Hand-worked top-k sparse code

        Use the identity dictionary (D=I_2), encoder preactivations
        ([2,0.5]), and (k=1). Top-k retains index 0; ReLU leaves code
        (f=[2,0]); decoding gives (hat y=fD=[2,0]). For target
        (y=[2,1]), the active fraction is (1/2), MSE is (1/2), and
        target variance is (1/4), so NMSE is 2. The code is sparse but not
        automatically faithful.
        """),
        mo.Html(table(rows, (("object", "Object"), ("value", "Value"), ("meaning", "Role")))),
        mo.md(f"Computed NMSE: **{normalized_mse:.3f}** · cosine: **{cosine:.3f}** · active fraction: **{np.count_nonzero(code) / code.size:.1%}**."),
        mo.md(r"""
        ```python
        selected = np.argpartition(preactivations, -k)[-k:]
        code = np.zeros_like(preactivations)
        code[selected] = np.maximum(preactivations[selected], 0)
        reconstruction = code @ dictionary
        nmse = ((reconstruction - target) ** 2).mean() / target.var()
        ```

        An SAE sets its target equal to its input activation. A transcoder can
        use exactly this encode/top-k/decode skeleton while fitting a **different
        downstream hook**; that different target is what changes the replacement
        question.
        """),
    ])
    return


@app.cell
def frontier_prediction(committed_form, mo):
    frontier_prediction = mo.ui.radio(
        options=[
            "Increasing top-k usually improves reconstruction while reducing sparsity",
            "Increasing top-k always makes every feature more monosemantic",
            "The lowest reconstruction error automatically gives the best causal replacement",
        ],
        value=None,
        label="Predict the trade-off",
    )
    frontier_form = committed_form(
        mo, {"tradeoff": frontier_prediction}, submit_label="Lock frontier prediction"
    )
    frontier_form
    return (frontier_form,)


@app.cell
def frontier_controls(mo):
    selected_k = mo.ui.number(1, 12, step=1, value=3, label="top-k active features per sample")
    promotion_rule = mo.ui.dropdown(
        options=[
            "smallest k with normalized MSE ≤ 0.20",
            "minimum normalized MSE regardless of sparsity",
            "k with the lowest dead-feature fraction",
        ],
        value="smallest k with normalized MSE ≤ 0.20",
        label="Frozen selection rule",
    )
    mo.hstack([selected_k, promotion_rule], justify="start")
    return promotion_rule, selected_k


@app.cell
def frontier_experiment(np, selected_k, toy_sparse_frontier):
    k_grid = np.arange(1, 13)
    sparse_frontier = toy_sparse_frontier(k_grid)
    selected_index = int(np.flatnonzero(k_grid == selected_k.value)[0])
    return k_grid, selected_index, sparse_frontier


@app.cell
def frontier_view(
    callout,
    frontier_form,
    k_grid,
    line_chart,
    metric_cards,
    mo,
    np,
    promotion_rule,
    selected_index,
    sparse_frontier,
):
    mo.stop(frontier_form.value is None, mo.md("Commit the qualitative prediction before viewing the curve."))
    frontier_series = [
        {"name": "normalized MSE", "color": "#3366a3", "points": list(zip(k_grid, sparse_frontier["normalized_mse"], strict=True))},
        {"name": "active fraction", "color": "#a95f19", "points": list(zip(k_grid, sparse_frontier["active_fraction"], strict=True))},
        {"name": "dead fraction", "color": "#14735a", "points": list(zip(k_grid, sparse_frontier["dead_fraction"], strict=True))},
    ]
    prediction_ok = frontier_form.value["tradeoff"].startswith("Increasing top-k usually")
    if promotion_rule.value.startswith("smallest k"):
        eligible = np.flatnonzero(sparse_frontier["normalized_mse"] <= 0.20)
        rule_index = int(eligible[0]) if eligible.size else None
    elif promotion_rule.value.startswith("minimum normalized"):
        rule_index = int(np.argmin(sparse_frontier["normalized_mse"]))
    else:
        minimum_dead = np.min(sparse_frontier["dead_fraction"])
        rule_index = int(np.flatnonzero(sparse_frontier["dead_fraction"] == minimum_dead)[0])
    rule_k = None if rule_index is None else int(k_grid[rule_index])
    mo.vstack([
        mo.md("## Interaction: the reconstruction–sparsity frontier"),
        mo.Html(line_chart(frontier_series, x_label="top-k active features", y_label="fraction / normalized error", y_domain=(0.0, 1.05), alt_text="As top-k increases, normalized reconstruction MSE falls while active fraction rises in the constructed sparse mixture.")),
        mo.Html(metric_cards([
            (int(k_grid[selected_index]), "selected k"),
            (f"{sparse_frontier['normalized_mse'][selected_index]:.3f}", "normalized MSE"),
            (f"{sparse_frontier['active_fraction'][selected_index]:.1%}", "active fraction"),
            (f"{sparse_frontier['dead_fraction'][selected_index]:.1%}", "dead features"),
            (f"{float(sparse_frontier['minimum_encoded_coefficient']):.1f}", "minimum encoded coefficient"),
            (rule_k, "k selected by frozen rule"),
        ])),
        mo.md(f"Frozen rule: **{promotion_rule.value}**. The constructed generator has three nonnegative true sources per example; its approximate encoder applies ReLU before top-k, so the displayed minimum coefficient is zero rather than a hidden signed code. Dictionary recovery is still affected by noise, coherence, and the approximate encoder."),
        mo.Html(callout("positive" if prediction_ok else "limit", "Trade-off predicted" if prediction_ok else "Sparsity is not free", "Increasing k expands the allowed code. Reconstruction can improve while the explanation becomes less sparse. Semantic inspection must not choose the operating point after seeing appealing features.")),
    ])
    return


@app.cell
def sensitivity_control(mo):
    error_location = mo.ui.radio(
        options={
            "Place reconstruction error in a downstream-sensitive coordinate": "sensitive",
            "Place equal-norm error in an insensitive coordinate": "insensitive",
        },
        value="Place reconstruction error in a downstream-sensitive coordinate",
        label="Same activation NMSE, different downstream consequence",
    )
    error_location
    return (error_location,)


@app.cell
def sensitivity_experiment(error_location, metric_cards, mo, np):
    base = np.asarray([1.0, -0.5, 0.25, 0.0])
    output_weights = np.asarray([8.0, 0.5, 0.1, 0.0])
    error = np.zeros(4)
    error[0 if error_location.value == "sensitive" else 3] = 0.25
    reconstruction = base + error
    activation_mse = float(np.mean((reconstruction - base) ** 2))
    output_change = float((reconstruction - base) @ output_weights)
    mo.vstack([
        mo.md("## Reconstruction fidelity is not replacement fidelity"),
        mo.Html(metric_cards([
            (f"{activation_mse:.5f}", "activation MSE (fixed)"),
            (f"{np.linalg.norm(error):.2f}", "error L2 (fixed)"),
            (f"{output_change:+.2f}", "downstream logit change"),
        ])),
        mo.md("The downstream Jacobian weights reconstruction errors unequally. Replacement policy JS, top-1 agreement, and task loss therefore belong beside NMSE and cosine."),
    ])
    return


@app.cell
def metrics_table(mo, table):
    sparse_metric_rows = [
        {"metric": "normalized MSE", "numerator": "mean ‖ŷ−y‖²", "denominator": "target variance", "failure mode": "small high-sensitivity errors"},
        {"metric": "cosine", "numerator": "ŷ·y", "denominator": "‖ŷ‖‖y‖", "failure mode": "ignores scale"},
        {"metric": "L0 / active fraction", "numerator": "nonzero codes", "denominator": "tokens × features", "failure mode": "threshold/top-k definition"},
        {"metric": "dead fraction", "numerator": "features never active", "denominator": "dictionary width", "failure mode": "finite sample and threshold"},
        {"metric": "replacement JS", "numerator": "policy distribution change", "denominator": "implicit via JS", "failure mode": "can hide task-specific logit effects"},
    ]
    mo.Html(table(sparse_metric_rows, (("metric", "Metric"), ("numerator", "Numerator"), ("denominator", "Normalization"), ("failure mode", "Blind spot"))))
    return


@app.cell
def real_sparse_panel(
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
        sparse = bundle.section("sparse")
        position_count = sparse["transcoder_position_count"]
        retained_rows = [
            {"model": row["model"], "NMSE": f"{row['normalized_mse']:.3f}", "cosine": f"{row['cosine']:.3f}", "policy JS": f"{row['policy_js']:.6f}", "top-1 agreement": f"{row['top1_agreement']:.1%}"}
            for row in sparse["transcoder_table"]
        ]
        support = sparse["transcoder_support"]
        max_js = max(row["policy_js"] for row in sparse["transcoder_table"])
        sparse_card = ClaimCard(
            operation=EvidenceOperation.INTERVENTION,
            target=TargetLevel.COMPONENT,
            scope=f"retained {position_count}-position development compatibility pilot for a published layer-14 transcoder",
            endpoint=Endpoint.POLICY,
            estimand="legal-policy JS after replacing the native component output with transcoder reconstruction",
            unit="paired position",
            grouping="selected development positions; no population interval",
            intervention="replace the layer-14 target component output with the frozen transcoder reconstruction",
            controls=("native output baseline", "Raw source model", "Hero transfer model", "activation NMSE and cosine"),
            assumptions=("artifact conversion and hook ABI passed", "the selected sample is a compatibility check only"),
            uncertainty=f"descriptive {position_count}-position pilot; no population interval",
            multiplicity="two model replacements under one frozen artifact",
            allowed=(f"In the retained compatibility pilot, transcoder replacement policy JS is at most {max_js:.6f} for Raw and Hero."),
            excluded="Low replacement JS does not establish monosemantic features or model-invariant feature meaning.",
            falsifier="A grouped held-out compatibility sample where NMSE, cosine, or policy-replacement gates fail.",
            status=ResultStatus.POSITIVE,
        )
        empirical_sparse_view = mo.vstack([
            mo.md("## Frozen Raw/Hero transcoder compatibility"),
            mo.Html(table(retained_rows, (("model", "Model"), ("NMSE", "NMSE"), ("cosine", "Cosine"), ("policy JS", "Replacement JS"), ("top-1 agreement", "Top-1 agreement")))),
            mo.Html(claim_card_html(sparse_card)),
            mo.Html(callout("limit", "Dead features are a first-class failure", f"Dead-feature fractions are {support['first_dead_feature_fraction']:.1%} and {support['second_dead_feature_fraction']:.1%}. Support Jaccard is {support['mean_support_jaccard']:.3f}; the surviving feature sample is highly selected.")),
        ])
    else:
        empirical_sparse_view = mo.Html(callout("neutral", "Toy boundary", "The known sparse mixture supports only claims about the construction."))
    empirical_sparse_view
    return


@app.cell
def identifiability(mo, table):
    identifiability_rows = [
        {"failure": "feature splitting", "symptom": "one human concept activates several correlated dictionary elements", "test": "stability across seeds/widths and joint intervention"},
        {"failure": "feature absorption", "symptom": "one broad feature takes variance from several narrower factors", "test": "conditional examples and alternative dictionaries"},
        {"failure": "dead features", "symptom": "large unused dictionary region", "test": "activity on held-out, diverse tokens"},
        {"failure": "decoder geometry without causal use", "symptom": "clean examples but replacement/ablation endpoint is null", "test": "opportunity-gated intervention with controls"},
        {"failure": "source asymmetry", "symptom": "dictionary learned on A is treated as if native to B", "test": "B-trained/bi-model controls and transfer diagnostics"},
    ]
    mo.Html(table(identifiability_rows, (("failure", "Failure mode"), ("symptom", "Symptom"), ("test", "Required diagnostic"))))
    return


@app.cell
def assessments(committed_form, mo, module):
    architecture_transfer = mo.ui.text_area(label="Transfer 1 — ABI", placeholder="A sparse module maps MLP input to MLP output. Classify it and state the replacement estimand.")
    metric_transfer = mo.ui.text_area(label="Transfer 2 — metrics", placeholder="NMSE improves but policy JS worsens. Explain how and name two diagnostics.")
    frontier_transfer = mo.ui.text_area(label="Transfer 3 — selection", placeholder="Preregister a Pareto rule for width/top-k without selecting on semantic examples.")
    assessment_form = committed_form(
        mo,
        {"architecture": architecture_transfer, "metrics": metric_transfer, "selection": frontier_transfer},
        submit_label="Commit responses and reveal sparse-method rubric",
        min_words=5,
    )
    mo.vstack([mo.md(f"## Unseen transfer assessment\n\n{module.transfer_assessment}"), assessment_form])
    return (assessment_form,)


@app.cell
def assessment_feedback(assessment_form, mo):
    mo.stop(assessment_form.value is None, mo.md("Submit all objectives before revealing."))
    response_lengths = [len(assessment_form.value[key].split()) for key in ("architecture", "metrics", "selection")]
    mo.md(f"""
    **Rubric (response lengths {response_lengths}):** classify by different
    input/output boundaries as a transcoder; explain that downstream Jacobians
    amplify structured error; report activation and replacement fidelity plus
    sparsity/dead features; freeze a smallest-k-under-error or Pareto-knee rule
    on development data before inspecting named semantics.
    """)
    return


@app.cell
def close(bundle, callout, mo, resource_card):
    mo.vstack([
        mo.Html(callout("positive", "Positive result", "A sparse replacement can achieve bounded activation and policy fidelity, creating a usable intervention substrate.")),
        mo.Html(callout("limit", "What this does not show", "Sparse, overcomplete, and faithful does not imply unique, monosemantic, native, or causally important features.")),
        mo.md("""
        ## Retrieval before module 08

        1. What exact target distinguishes an SAE from a transcoder?
        2. Why can identical NMSE produce different policy change?
        3. When must the sparsity operating point be frozen?

        **Sources:** Anthropic's monosemanticity report (lab report 2023) and
        Dunefsky et al. on transcoders (NeurIPS 2024). See the
        [course source register](../SOURCES.md).
        """),
        mo.Html(resource_card(
            mode=bundle.mode.value,
            runtime="< 3 seconds for a 256×24 constructed sparse frontier",
            device="CPU / NumPy; no weights or network",
            determinism="toy sparse seed 31; hand calculation deterministic; NumPy float64",
            sources=f"content identity {bundle.integrity_sha256[:16]}…",
            limitations="Known sparse mixture and frozen compatibility metrics; external sparse weights are not redistributed.",
        )),
    ])
    return


if __name__ == "__main__":
    app.run()
