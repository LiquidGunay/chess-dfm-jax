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
    from research.interpretability.course.stats import support_jaccard
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
        callout,
        claim_card_html,
        committed_form,
        get_module,
        load_course_bundle,
        metric_cards,
        mo,
        mode_badge,
        module_header,
        np,
        resource_card,
        support_jaccard,
        table,
    )


@app.cell
def title(get_module, mo, module_header):
    module = get_module(8)
    mo.Html(module_header(module))
    return (module,)


@app.cell
def opening(mo, module, table):
    objective_rows = [
        {"objective": objective, "mastery": mastery}
        for objective, mastery in zip(
            module.objectives,
            (
                "Draw the exact input/QK/sparse-OV/output boundary and recover [B,64,1024].",
                "Classify each number as fidelity, support, semantics, or intervention evidence.",
                "Refuse a semantic claim when opportunity, held-out direction, control, or test role fails.",
            ),
            strict=True,
        )
    ]
    mo.vstack([
        mo.md(r"""
        ## LoRSA is a sparse attention-branch replacement

        The published artifact was learned on Raw BT4 at layer 14. It consumes
        `hook_attn_in`, computes its own learned attention/SmolGen path, uses a
        sparse bank of OV features, and reconstructs `hook_attn_out`. Applying
        that frozen module to Hero is a **source-asymmetric transfer**.

        That asymmetry is scientifically useful: it asks whether one Raw-learned
        coordinate system remains a faithful and stable lens on Hero. It does
        not make the lens native to Hero, and it does not turn one sparse
        decoder direction into a residual-stream feature.
        """),
        mo.Html(table(objective_rows, (("objective", "Objective"), ("mastery", "Formative evidence")))),
    ])
    return


@app.cell
def prerequisite(committed_form, mo):
    response = mo.ui.text_area(
        label="Prerequisite diagnostic — sparse architecture",
        placeholder="Contrast a residual SAE with an attention-branch replacement in one sentence about inputs, targets, and downstream fidelity.",
    )
    prerequisite_form = committed_form(
        mo, {"sparse_boundary": response}, submit_label="Commit prerequisite response", min_words=6
    )
    prerequisite_form
    return (prerequisite_form,)


@app.cell
def prerequisite_feedback(mo, prerequisite_form):
    mo.stop(prerequisite_form.value is None, mo.md("Answer the prerequisite diagnostic to reveal the bridge."))
    mo.callout("A residual SAE reconstructs the same activation space. LoRSA consumes attention input, computes learned routing and sparse OV contributions, and reconstructs the projected attention branch, so replacement-policy fidelity is part of its validity.", kind="info")
    return


@app.cell
def mode_controls(mo):
    evidence_mode = mo.ui.radio(
        options={
            "Snapshot — retained LoRSA transfer/semantics/causal metrics": "snapshot",
            "Toy — constructed source-asymmetry screen": "toy",
            "Source — verify retained LoRSA records": "source",
        },
        value="Snapshot — retained LoRSA transfer/semantics/causal metrics",
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
def attention_primer(mo, table):
    _routing = [0.75, 0.25]
    _source_values = [2.0, -1.0]
    _routed = sum(
        weight * value
        for weight, value in zip(_routing, _source_values, strict=True)
    )
    _trace_rows = [
        {"source square": "s₁", "routing weight": "0.75", "value score": "+2.00", "contribution": "+1.50"},
        {"source square": "s₂", "routing weight": "0.25", "value score": "−1.00", "contribution": "−0.25"},
        {"source square": "sum", "routing weight": "1.00", "value score": "—", "contribution": f"{_routed:+.2f}"},
    ]
    mo.vstack([
        mo.md(r"""
        ## Prerequisite: attention routes values

        For sequence activations \(X\in\mathbb R^{B\times S\times d}\),
        one ordinary attention head computes

        \[
        Q_h=XW_Q^h,\quad K_h=XW_K^h,\quad V_h=XW_V^h,
        \]
        \[
        A_h=\operatorname{softmax}_{\text{source}}\left(
          Q_hK_h^\top/\sqrt{d_h}+B_h^{\text{SmolGen}}(X)
        \right),\qquad
        O=\operatorname{concat}_h(A_hV_h)W_O.
        \]

        Each row of \(A_h\) says how one target square mixes source-square
        values. **SmolGen** is a content-dependent additive
        \(S\times S\) bias to the attention logits for each head. It changes
        routing before softmax; it is not a value vector, a chess engine, or
        search.

        That first equation is the ordinary-attention reference. The retained
        LoRSA artifact uses its learned attention divisor \(s_{attn}\), not
        an assumed \(\sqrt{32}\):

        \[
        A_h^{L}=\operatorname{softmax}_{\text{source}}\left(
          Q_h^{L}(K_h^{L})^\top/s_{attn}
          +c_{smol}B_h^{\text{SmolGen}}(X)
        \right).
        \]

        Both scales are part of the frozen artifact and normalization
        contract.

        Published LoRSA keeps this routing idea but supplies 128 learned Q/K
        heads of width 32, without causal or rotary masking, and a bank of
        \(d_{SAE}=16{,}384\) dynamic sparse OV feature heads. A top-30 rule
        chooses sparse feature contributions at each token; an output map
        returns their sum to width 1024. Thus feature \(f\) is a routed,
        sparse attention/value contribution—not coordinate \(f\) of the
        residual stream.
        """),
        mo.md("### Two-source hand trace (schematic, not checkpoint arithmetic)"),
        mo.Html(table(_trace_rows, (("source square", "Source"), ("routing weight", "Routing A"), ("value score", "Value"), ("contribution", "A × value")))),
        mo.md(f"The routed score is `0.75·2 + 0.25·(−1) = {_routed:.2f}`. If top-k retains feature (f), its decoded branch contribution is `{_routed:.2f} d_f`; otherwise it is zero. This isolates routing, selection, and decoding without pretending that two scalars reproduce the full artifact."),
    ])
    return


@app.cell
def architecture(mo, table):
    architecture_rows = [
        {"stage": "input hook", "tensor": "blocks.14.hook_attn_in", "shape": "[B,64,1024]", "role": "normalized layer input before native Q/K/V and SmolGen"},
        {"stage": "learned Q/K + SmolGen", "tensor": "LoRSA attention patterns", "shape": "[B,128,64,64]", "role": "128 separately learned routing heads of width 32; no causal/rotary mask"},
        {"stage": "sparse OV code", "tensor": "nonnegative feature activations", "shape": "[B,64,16384] sparse", "role": "top-30 among 16,384 dynamic OV feature heads per token"},
        {"stage": "decoder/output map", "tensor": "reconstructed branch", "shape": "[B,64,1024]", "role": "substitute for native projected attention branch"},
        {"stage": "output hook", "tensor": "blocks.14.hook_attn_out", "shape": "[B,64,1024]", "role": "before residual scaling/addition"},
    ]
    mo.vstack([
        mo.md("## Exact hook ABI"),
        mo.Html(table(architecture_rows, (("stage", "Stage"), ("tensor", "Named object"), ("shape", "Shape"), ("role", "Meaning")))),
        mo.md("Native-versus-LoRSA head-mean attention JS compares aggregate distributions with different learned head systems. It is not a one-to-one native-head match."),
    ])
    return


@app.cell
def normalization_contract(callout, mo):
    mo.Html(callout(
        "danger",
        "Normalization is part of the ABI",
        (
            "Training rescales each hook family to mean L2 norm sqrt(1024)=32. "
            "For inference, the published fold multiplies LoRSA WQ, WK, and WV "
            "by c_in and divides WO and output bias by c_out. A conversion must "
            "apply that fold exactly or reproduce the training normalization—"
            "never both. Equal tensor shapes with the wrong fold are not the "
            "same replacement experiment."
        ),
    ))
    return


@app.cell
def abi_exercise(committed_form, mo):
    abi_choice = mo.ui.radio(
        options=[
            "Replace the full post-block residual with one decoder direction",
            "Replace hook_attn_out with the reconstructed LoRSA branch, then keep the native residual path",
            "Add the reconstructed branch after the next layer norm",
        ],
        value=None,
        label="Which substitution preserves the trained LoRSA estimand?",
    )
    abi_form = committed_form(
        mo, {"replacement": abi_choice}, submit_label="Commit ABI answer"
    )
    abi_form
    return (abi_form,)


@app.cell
def abi_feedback(abi_form, callout, mo):
    mo.stop(abi_form.value is None, mo.md("Commit before revealing."))
    abi_ok = abi_form.value["replacement"].startswith("Replace hook_attn_out")
    mo.Html(callout(
        "positive" if abi_ok else "danger",
        "Replacement boundary preserved" if abi_ok else "Equal shape is not equal intervention",
        "LoRSA approximates the attention branch output. Replacing a whole residual or moving across normalization changes the target, residual algebra, and downstream distribution.",
    ))
    return


@app.cell
def toy_controls(mo):
    source_model = mo.ui.radio(
        options={"Select feature on Constructed A": "A", "Select feature on Constructed B": "B"},
        value="Select feature on Constructed A",
        label="Discovery source",
    )
    activation_threshold = mo.ui.number(0.0, 2.5, step=0.1, value=1.0, label="active-support threshold")
    correlation_shift = mo.ui.number(0.0, 1.5, step=0.1, value=0.5, label="A↔B feature drift")
    mo.vstack([source_model, mo.hstack([activation_threshold, correlation_shift])])
    return activation_threshold, correlation_shift, source_model


@app.cell
def toy_diff(
    activation_threshold,
    correlation_shift,
    np,
    source_model,
    support_jaccard,
):
    rng = np.random.default_rng(808)
    examples, features = 512, 64
    latent = rng.normal(size=(examples, 6))
    concept = (latent[:, 0] + 0.3 * rng.normal(size=examples) > 0.4).astype(float)
    feature_a = latent @ rng.normal(size=(6, features)) + 0.4 * rng.normal(size=(examples, features))
    feature_b = feature_a + correlation_shift.value * rng.normal(size=(examples, features))
    source_values = feature_a if source_model.value == "A" else feature_b
    target_values = feature_b if source_model.value == "A" else feature_a
    discovery = slice(0, 256)
    evaluation = slice(256, 512)
    discovery_scores = np.asarray([
        np.corrcoef(source_values[discovery, index], concept[discovery])[0, 1]
        for index in range(features)
    ])
    selected_feature = int(np.nanargmax(np.abs(discovery_scores)))
    discovery_score = float(discovery_scores[selected_feature])
    source_score = float(
        np.corrcoef(
            source_values[evaluation, selected_feature],
            concept[evaluation],
        )[0, 1]
    )
    target_score = float(
        np.corrcoef(
            target_values[evaluation, selected_feature],
            concept[evaluation],
        )[0, 1]
    )
    support_a = feature_a[evaluation, selected_feature] > activation_threshold.value
    support_b = feature_b[evaluation, selected_feature] > activation_threshold.value
    support_overlap = support_jaccard(support_a, support_b)
    evaluation_concept = concept[evaluation] > 0
    opportunity_count = int(np.count_nonzero(evaluation_concept))
    joint_a = int(np.count_nonzero(support_a & evaluation_concept))
    joint_b = int(np.count_nonzero(support_b & evaluation_concept))
    return (
        discovery_score,
        joint_a,
        joint_b,
        opportunity_count,
        selected_feature,
        source_score,
        support_overlap,
        target_score,
    )


@app.cell
def toy_diff_view(
    discovery_score,
    joint_a,
    joint_b,
    metric_cards,
    mo,
    opportunity_count,
    selected_feature,
    source_model,
    source_score,
    support_overlap,
    target_score,
):
    mo.vstack([
        mo.md("## Interaction: source selection and threshold are part of the result"),
        mo.Html(metric_cards([
            (selected_feature, f"feature selected on {source_model.value}"),
            (f"{discovery_score:+.3f}", "discovery source correlation"),
            (f"{source_score:+.3f}", "held-out source correlation"),
            (f"{target_score:+.3f}", "held-out other-model correlation"),
            (f"{support_overlap:.3f}", "held-out A/B support Jaccard"),
            (opportunity_count, "held-out concept opportunities"),
            (f"{joint_a}/{joint_b}", "held-out active ∩ concept in A/B"),
        ])),
        mo.md("The first 256 examples select one feature on the declared source; the second 256 measure source replication, transfer, support, and opportunity. The discovery score is winner-selected, while both held-out scores are audits. The target score remains source-asymmetric rather than symmetric evidence. Changing the activity threshold changes support and therefore the estimand."),
    ])
    return


@app.cell
def evidence_pipeline(mo, table):
    pipeline_rows = [
        {"gate": "1 · artifact / ABI", "question": "Does converted LoRSA exactly match layer, hooks, dimensions, normalization, and source commit?", "failure action": "stop; no feature claims"},
        {"gate": "2 · source fidelity", "question": "Does Raw reconstruction and policy replacement pass frozen thresholds?", "failure action": "stop transfer"},
        {"gate": "3 · Hero transfer", "question": "Does the same frozen replacement remain faithful on paired Hero positions?", "failure action": "do not interpret cross-model features"},
        {"gate": "4 · opportunity", "question": "Were enough active/concept joint events possible before selecting a lift?", "failure action": "mark unsupported, not zero"},
        {"gate": "5 · held-out semantics", "question": "Does source direction replicate and transfer on a group-disjoint half?", "failure action": "retain negative result"},
        {"gate": "6 · causal surrogate", "question": "Does target ablation exceed scalar/token/norm-matched controls across dose?", "failure action": "no scoped causal-use claim"},
        {"gate": "7 · confirmation", "question": "Was a frozen test opened once with multiplicity and independent clusters?", "failure action": "remain exploratory"},
    ]
    mo.Html(table(pipeline_rows, (("gate", "Gate"), ("question", "Question"), ("failure action", "Fail-closed action"))))
    return


@app.cell
def transfer_panel(
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
        lorsa = bundle.section("lorsa")
        transfer = lorsa["transfer"]
        source = transfer["source_stage"]
        hero_transfer = transfer["transfer_stage"]["hero"]
        support = transfer["transfer_stage"]["raw_hero"]["support_transfer"]
        bootstrap = lorsa["paired_bootstrap"]["policy_js_mean_per_position"]
        transfer_rows = [
            {"model": "Raw source", "NMSE": f"{source['reconstruction']['normalized_mse']:.5f}", "cosine": f"{source['reconstruction']['cosine']:.4f}", "policy JS": f"{source['policy_replacement']['js_divergence']['mean']:.6f}", "top-1": f"{source['policy_replacement']['top1_agreement']['mean']:.1%}"},
            {"model": "Hero transfer", "NMSE": f"{hero_transfer['reconstruction']['normalized_mse']:.5f}", "cosine": f"{hero_transfer['reconstruction']['cosine']:.4f}", "policy JS": f"{hero_transfer['policy_replacement']['js_divergence']['mean']:.6f}", "top-1": f"{hero_transfer['policy_replacement']['top1_agreement']['mean']:.1%}"},
        ]
        transfer_card = ClaimCard(
            operation=EvidenceOperation.INTERVENTION,
            target=TargetLevel.COMPONENT,
            scope="512 retained development positions under frozen Raw-trained LoRSA replacement in Raw and Hero",
            endpoint=Endpoint.POLICY,
            estimand="mean legal-policy JS between native and LoRSA-replaced attention branch",
            unit="paired position",
            grouping="512 unique retained group IDs",
            intervention="replace native layer-14 hook_attn_out with frozen LoRSA reconstruction from hook_attn_in",
            controls=("Raw source compatibility", "Hero transfer", "native recompute parity", "paired top-1 contingency"),
            assumptions=("converted published artifact satisfies its contract", "one retained group ID is an independent root for this development summary"),
            uncertainty=f"20,000 paired-root bootstrap; Hero-minus-Raw policy-JS interval [{bootstrap['hero_minus_raw_mean_percentile_ci95']['lower']:.6f}, {bootstrap['hero_minus_raw_mean_percentile_ci95']['upper']:.6f}]",
            multiplicity="exploratory transfer family; no confirmatory correction",
            allowed=(f"The frozen LoRSA branch has low replacement policy JS in both models; Hero degradation is {bootstrap['hero_minus_raw_mean']:+.6f} larger on average in this development sample."),
            excluded="Transfer fidelity does not establish semantic equivalence, native Hero features, or confirmatory generalization.",
            falsifier="A checksum-matched grouped replication where source or transfer compatibility gates fail.",
            status=ResultStatus.POSITIVE,
        )
        transfer_view = mo.vstack([
            mo.md("## Frozen source fidelity and Hero transfer"),
            mo.Html(table(transfer_rows, (("model", "Application"), ("NMSE", "NMSE"), ("cosine", "Cosine"), ("policy JS", "Replacement JS"), ("top-1", "Native top-1 preserved")))),
            mo.Html(claim_card_html(transfer_card)),
            mo.Html(callout("limit", "Faithful outputs, nonidentical supports", f"Mean Raw/Hero support Jaccard is {support['mean_support_jaccard']:.3f}; resolved activation correlation is {support['mean_resolved_activation_correlation']:.3f}. A shared feature index does not imply equal token support or magnitude.")),
        ])
    else:
        transfer_view = mo.Html(callout("neutral", "Toy boundary", "No Raw/Hero transfer table is shown in toy mode."))
    transfer_view
    return


@app.cell
def semantic_control(bundle, mo):
    if bundle.is_empirical:
        semantic_pairs = bundle.section("lorsa")["semantic_pairs"]
        semantic_options = {f"feature {row['feature_index']} · {row['concept']}={row['class_name']}": index for index, row in enumerate(semantic_pairs)}
        semantic_choice = mo.ui.dropdown(options=semantic_options, value=next(iter(semantic_options)), label="Inspect a frozen semantic association")
        semantic_widget = semantic_choice
    else:
        semantic_pairs = []
        semantic_choice = None
        semantic_widget = mo.md("Semantic inspector is empirical-only.")
    semantic_widget
    return semantic_choice, semantic_pairs


@app.cell
def semantic_view(callout, metric_cards, mo, semantic_choice, semantic_pairs):
    if semantic_choice is not None:
        semantic = semantic_pairs[semantic_choice.value]
        raw_semantics = semantic["raw_heldout"]
        hero_semantics = semantic["hero_heldout"]
        semantic_result_view = mo.vstack([
            mo.md("## Opportunity-gated exact-token association"),
            mo.Html(metric_cards([
                (semantic["feature_index"], "feature"),
                (semantic["fit_direction"], "Raw-fit selected direction"),
                (f"{raw_semantics['evaluation_log2_lift']:+.3f}", "Raw held-out log2 lift"),
                (f"{hero_semantics['evaluation_log2_lift']:+.3f}", "Hero held-out log2 lift"),
                (raw_semantics["direction_replicated"], "Raw direction replicated"),
                (hero_semantics["direction_replicated"], "Hero direction transferred"),
            ])),
            mo.Html(callout(
                "positive" if raw_semantics["direction_replicated"] and hero_semantics["direction_replicated"] else "limit",
                "Association transfers" if raw_semantics["direction_replicated"] and hero_semantics["direction_replicated"] else "Retained semantic failure",
                "The pair was selected on a Raw fit half and audited on a group-disjoint Raw/Hero half. Direction agreement is descriptive activation association, not semantic equivalence or policy use.",
            )),
        ])
    else:
        semantic_result_view = mo.md("")
    semantic_result_view
    return


@app.cell
def causal_control(bundle, mo):
    if bundle.is_empirical:
        causal_rows = bundle.section("lorsa")["causal"]["rows"]
        causal_options = {row["pair"]: index for index, row in enumerate(causal_rows)}
        causal_choice = mo.ui.dropdown(options=causal_options, value=next(iter(causal_options)), label="Inspect a frozen causal-surrogate target")
        causal_widget = causal_choice
    else:
        causal_choice = None
        causal_rows = []
        causal_widget = mo.md("Causal-surrogate inspector is empirical-only.")
    causal_widget
    return causal_choice, causal_rows


@app.cell
def causal_view(
    ClaimCard,
    Endpoint,
    EvidenceOperation,
    ResultStatus,
    TargetLevel,
    callout,
    causal_choice,
    causal_rows,
    claim_card_html,
    mo,
    table,
):
    if causal_choice is not None:
        causal = causal_rows[causal_choice.value]
        model_rows = []
        for arm, label in (("RR", "Raw"), ("HH", "Hero")):
            record = causal["models"][arm]
            effect = record["effect"]
            model_rows.append({"model": label, "active n": record["active_positions"], "target−control TV": f"{effect['mean_difference']:+.6f}", "95% interval": f"[{effect['lower']:+.6f}, {effect['upper']:+.6f}]", "Holm p": f"{record['holm_p']:.4g}", "gate": record["gate_passed"]})
        both_pass = all(record["gate_passed"] for record in causal["models"].values())
        pair_name = causal["pair"]
        if both_pass:
            status = ResultStatus.POSITIVE
            allowed = f"Within the frozen LoRSA replacement, {pair_name} has larger natural-ablation policy TV than scalar/token/norm-matched random decoder controls in both Raw and Hero development samples."
        elif "5851" in pair_name:
            status = ResultStatus.NEGATIVE_CONTROL
            allowed = f"The intended null {pair_name} does not pass the frozen LoRSA causal-specificity gate in either model."
        elif min(record["active_positions"] for record in causal["models"].values()) < 10:
            status = ResultStatus.UNRESOLVED
            allowed = f"The {pair_name} intervention is unresolved because the retained active-position opportunity is too small for the frozen gate."
        else:
            status = ResultStatus.FALSIFIED
            allowed = f"The {pair_name} target fails the frozen LoRSA direction-specificity gate in both retained model arms."
        causal_card = ClaimCard(
            operation=EvidenceOperation.INTERVENTION,
            target=TargetLevel.COMPONENT,
            scope="256 held-out development roots inside the frozen layer-14 LoRSA replacement path",
            endpoint=Endpoint.POLICY,
            estimand="target-direction natural-ablation legal-policy TV minus mean of three matched random decoder-direction TVs on target-active positions",
            unit="active held-out position",
            grouping="unique retained development root",
            intervention="remove the frozen target feature contribution at its natural active tokens; retain matched scalar, token support, and decoder norm",
            controls=("three matched random decoder directions", "batched no-op", "Raw and Hero arms", "ablation/insertion dose checks"),
            assumptions=("LoRSA decoder replacement is the causal substrate", "semantic selection was frozen before intervention outcomes"),
            uncertainty="20,000 paired bootstrap/sign-flip samples with Holm correction across 12 arm-target tests",
            multiplicity="Holm family of six targets × two model arms",
            allowed=allowed,
            excluded="The result does not establish a native dense-attention feature, input-concept causality, monosemanticity, or a confirmatory test claim.",
            falsifier="An independently grouped, preregistered replication that reverses the frozen gate result with a semantic-placebo direction included.",
            status=status,
        )
        causal_result_view = mo.vstack([
            mo.md("## Strict matched-direction intervention"),
            mo.Html(table(model_rows, (("model", "Model"), ("active n", "Active positions"), ("target−control TV", "Target−control TV"), ("95% interval", "95% interval"), ("Holm p", "Holm p"), ("gate", "Gate passed")))),
            mo.Html(claim_card_html(causal_card)),
            mo.Html(callout("limit", "Missing semantic placebo", "The definitive retained run has three scalar/token/norm-matched random decoder directions, but not a plausible wrong-concept direction. Confirmation should add a semantic placebo while preserving the same matching contract.")),
        ])
    else:
        causal_result_view = mo.md("")
    causal_result_view
    return


@app.cell
def placebo_design(committed_form, mo):
    placebo_answer = mo.ui.text_area(
        label="Design a semantic placebo for the selected feature",
        placeholder="Choose a plausible but wrong concept/direction; match activation opportunity, token support, decoder norm, dose, and selection timing.",
    )
    placebo_form = committed_form(
        mo,
        {"placebo_design": placebo_answer},
        submit_label="Commit design and reveal placebo checklist",
        min_words=10,
    )
    placebo_form
    return (placebo_form,)


@app.cell
def placebo_feedback(mo, placebo_form):
    mo.stop(placebo_form.value is None, mo.md("Submit the placebo draft before revealing."))
    word_count = len(placebo_form.value["placebo_design"].split())
    mo.md(f"""
    **Checklist (your draft: {word_count} words):** the placebo is semantically
    plausible but contradicts the named hypothesis; it is selected without
    intervention outcomes; receives the exact target scalar, token opportunity,
    decoder norm, sign/dose grid, and model arms; and is scored on the same
    policy/value endpoint with the same grouping and multiplicity rule.
    """)
    return


@app.cell
def license_gate(callout, mo):
    mo.Html(callout(
        "danger",
        "Artifact/license gate",
        "The course bundles derived metrics only. Published LoRSA weights, Raw/Hero checkpoints, and corpora are not redistributed because the retained record does not establish redistribution permission. Source reruns require a user-supplied local artifact with its exact identity.",
    ))
    return


@app.cell
def assessments(committed_form, mo, module):
    abi_transfer = mo.ui.text_area(label="Transfer 1 — hook ABI", placeholder="An unseen sparse attention module consumes resid_mid and predicts resid_post. Is it a LoRSA replacement? Name the exact missing information.")
    evidence_transfer = mo.ui.text_area(label="Transfer 2 — evidence types", placeholder="Classify NMSE, support Jaccard, held-out lift, and ablation TV-minus-controls by operation and target.")
    asymmetry_transfer = mo.ui.text_area(label="Transfer 3 — source asymmetry", placeholder="A dictionary trained on LM A transfers to LM B. State permitted and excluded cross-model claims plus an opportunity gate.")
    assessment_form = committed_form(
        mo,
        {"abi": abi_transfer, "evidence": evidence_transfer, "asymmetry": asymmetry_transfer},
        submit_label="Commit responses and reveal LoRSA rubric",
        min_words=5,
    )
    mo.vstack([mo.md(f"## Unseen transfer assessment\n\n{module.transfer_assessment}"), assessment_form])
    return (assessment_form,)


@app.cell
def assessment_feedback(assessment_form, mo):
    mo.stop(assessment_form.value is None, mo.md("Submit every objective before revealing."))
    answer_lengths = [len(assessment_form.value[key].split()) for key in ("abi", "evidence", "asymmetry")]
    mo.md(f"""
    **Rubric (response lengths {answer_lengths}):** recover input/output hooks,
    normalization, residual algebra, layer, token ordering, and learned routing;
    label NMSE/fidelity as replacement diagnostics, Jaccard/lift as observations,
    and controlled ablation as a scoped component intervention; preserve source
    direction, fit/audit split, joint opportunity, test status, and the exclusion
    of native-feature or semantic-equivalence claims.
    """)
    return


@app.cell
def close(bundle, callout, mo, resource_card):
    mo.vstack([
        mo.Html(callout("positive", "Positive result", "Raw-trained LoRSA is a numerically viable branch replacement on both models, and three frozen decoder directions pass strict matched-random causal-surrogate gates in both development arms.")),
        mo.Html(callout("limit", "What this does not show", "Support overlap is incomplete; two semantic directions disagree, one intended null stays null, one target is underpowered, semantic placebos are absent, and no native dense or confirmatory claim is licensed.")),
        mo.md("""
        ## Retrieval before module 09

        1. What exact native component does LoRSA replace?
        2. Why is a Raw-learned feature index source-asymmetric on Hero?
        3. Which controls make the strict causal-surrogate estimand specific?

        **Sources:** Crosscoders (Transformer Circuits 2024, lab report) for
        cross-model sparse comparison and *Tracing the Thought of a
        Grandmaster-level Chess-Playing Transformer* (2026 preprint) for the
        published chess LoRSA artifact.
        The retained Raw/Hero applications remain local exploratory records.
        See the [course source register](../SOURCES.md) and
        [method-coverage matrix](../METHOD_COVERAGE.md).
        """),
        mo.Html(resource_card(
            mode=bundle.mode.value,
            runtime="< 3 seconds; interactive views use compact derived metrics",
            device="CPU / NumPy; no sparse weights, checkpoint, GPU, or network",
            determinism="visible synthetic seed 808; snapshot extraction deterministic; NumPy float64",
            sources=f"content identity {bundle.integrity_sha256[:16]}…",
            limitations="Development-only derived records; weights are user-supplied for any source rerun and are not licensed for course redistribution.",
        )),
    ])
    return


if __name__ == "__main__":
    app.run()
