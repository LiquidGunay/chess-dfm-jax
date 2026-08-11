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
    from research.interpretability.course.experiments import toy_circuit_truth_table
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
        table,
        toy_circuit_truth_table,
    )


@app.cell
def title(get_module, mo, module_header):
    module = get_module(9)
    mo.Html(module_header(module))
    return (module,)


@app.cell
def opening(mo, module, table):
    objective_rows = [
        {"objective": objective, "mastery": mastery}
        for objective, mastery in zip(
            module.objectives,
            (
                "Compute all seven criteria on an unseen redundant Boolean mechanism.",
                "Specify low-level and abstract variables plus an interchange map and intervention distribution.",
                "Reject a larger or post-selected subgraph when a smaller matched alternative explains the endpoint.",
            ),
            strict=True,
        )
    ]
    mo.vstack([
        mo.md(r"""
        ## Localization is not yet an algorithm

        A patch can identify a layer where replacing a residual moves policy.
        A probe can name information available there. A sparse feature can make
        one direction manipulable. A **circuit claim** adds structure: a set of
        components and edges implements an abstract computation over a stated
        input/intervention distribution.

        That claim is stronger because it predicts counterfactual behavior. It
        should survive removal of irrelevant parts, preserve the model where it
        claims faithfulness, and commute with interventions on corresponding
        low-level and abstract variables.
        """),
        mo.Html(table(objective_rows, (("objective", "Objective"), ("mastery", "Formative evidence")))),
    ])
    return


@app.cell
def prerequisite(committed_form, mo):
    response = mo.ui.text_area(
        label="Prerequisite diagnostic — component versus circuit",
        placeholder="A direction has a controlled policy effect. What additional structure and tests are required before calling it a circuit?",
    )
    prerequisite_form = committed_form(
        mo, {"circuit_boundary": response}, submit_label="Commit prerequisite response", min_words=6
    )
    prerequisite_form
    return (prerequisite_form,)


@app.cell
def prerequisite_feedback(mo, prerequisite_form):
    mo.stop(prerequisite_form.value is None, mo.md("Answer the prerequisite diagnostic to reveal the bridge."))
    mo.callout("A circuit claim names components, edges, and an abstract computation, then tests faithfulness, completeness, necessity, sufficiency, redundancy, stability, and interchange consistency against smaller and random alternatives.", kind="info")
    return


@app.cell
def mode_controls(mo):
    evidence_mode = mo.ui.radio(
        options={
            "Snapshot — retained Raw/Hero evidence audit": "snapshot",
            "Toy — known redundant circuit": "toy",
            "Source — verify retained evidence audit": "source",
        },
        value="Snapshot — retained Raw/Hero evidence audit",
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
def criteria(mo, table):
    criteria_rows = [
        {"criterion": "faithfulness", "question": "Does the candidate reproduce the model endpoint on the declared observational and intervention distribution?", "common trap": "score only easy clean inputs"},
        {"criterion": "completeness", "question": "Do excluded components have negligible residual effect under matched interventions?", "common trap": "call a high-fidelity subset complete"},
        {"criterion": "necessity", "question": "Does removing the component/path degrade the scoped endpoint?", "common trap": "redundancy masks individual ablations"},
        {"criterion": "sufficiency", "question": "Does retaining/patching the candidate recover the endpoint from an appropriate baseline?", "common trap": "off-manifold sufficiency baseline"},
        {"criterion": "redundancy", "question": "Can alternate paths substitute singly or jointly?", "common trap": "one-at-a-time ablations"},
        {"criterion": "stability", "question": "Do components and effects replicate across positions, seeds, openings, models, and dose?", "common trap": "post-select a showcase"},
        {"criterion": "interchange", "question": "Do corresponding low-level and abstract interventions commute?", "common trap": "patch a variable without its redundant implementation"},
    ]
    mo.Html(table(criteria_rows, (("criterion", "Criterion"), ("question", "Operational question"), ("common trap", "Common trap"))))
    return


@app.cell
def toy_mechanism(mo):
    mo.md(r"""
    ## Ground-truth specimen

    Inputs are `attack` \(A\), `defender` \(D\), and `king_exposed` \(K\).

    \[
    T_1=A\land\neg D,\qquad T_2=A\land\neg D,\qquad
    Y=(T_1\lor T_2)\land K.
    \]

    `T₁` and `T₂` are **distinct causal nodes** that independently compute the
    same threat function from shared parents; neither is downstream of the
    other. We may therefore intervene on either realization while holding the
    other fixed. This tiny system is
    deliberately simple enough that every observational and interventional
    claim can be checked exhaustively over all eight inputs.
    """)
    return


@app.cell
def circuit_controls(mo):
    candidate_components = mo.ui.multiselect(
        options=["threat path T₁", "redundant path T₂", "king-exposure gate K"],
        value=["threat path T₁", "king-exposure gate K"],
        label="Candidate circuit",
    )
    baseline_choice = mo.ui.radio(
        options={"Zero ablated components": "zero", "Set ablated gates to one": "one"},
        value="Zero ablated components",
        label="Ablation baseline",
    )
    mo.vstack([candidate_components, baseline_choice])
    return baseline_choice, candidate_components


@app.cell
def candidate_evaluation(
    baseline_choice,
    candidate_components,
    np,
    toy_circuit_truth_table,
):
    truth_rows = toy_circuit_truth_table()
    selected_nodes = set(candidate_components.value)
    evaluated_rows = []
    candidate_outputs = []
    full_outputs = []
    for truth_row in truth_rows:
        missing_value = 0 if baseline_choice.value == "zero" else 1
        candidate_threat = truth_row["threat"] if "threat path T₁" in selected_nodes else missing_value
        candidate_redundant = truth_row["redundant_copy"] if "redundant path T₂" in selected_nodes else missing_value
        candidate_gate = truth_row["king_exposed"] if "king-exposure gate K" in selected_nodes else missing_value
        candidate_output = (candidate_threat | candidate_redundant) & candidate_gate
        candidate_outputs.append(candidate_output)
        full_outputs.append(truth_row["output"])
        evaluated_rows.append({
            "A": truth_row["attack"], "D": truth_row["defender"], "K": truth_row["king_exposed"],
            "T1": truth_row["threat"], "T2": truth_row["redundant_copy"],
            "full Y": truth_row["output"], "candidate Ŷ": candidate_output,
            "match": candidate_output == truth_row["output"],
        })
    candidate_outputs_array = np.asarray(candidate_outputs)
    full_outputs_array = np.asarray(full_outputs)
    faithfulness = float(np.mean(candidate_outputs_array == full_outputs_array))
    false_negative = int(np.count_nonzero((candidate_outputs_array == 0) & (full_outputs_array == 1)))
    false_positive = int(np.count_nonzero((candidate_outputs_array == 1) & (full_outputs_array == 0)))
    return (
        candidate_outputs_array,
        evaluated_rows,
        faithfulness,
        false_negative,
        false_positive,
    )


@app.cell
def candidate_view(
    evaluated_rows,
    faithfulness,
    false_negative,
    false_positive,
    metric_cards,
    mo,
    table,
):
    mo.vstack([
        mo.md("## Interaction: assemble and falsify a candidate"),
        mo.Html(metric_cards([
            (f"{faithfulness:.1%}", "truth-table faithfulness"),
            (false_negative, "false negatives"),
            (false_positive, "false positives"),
            ("pass" if faithfulness == 1.0 else "fail", "exact exhaustive gate"),
        ])),
        mo.Html(table(evaluated_rows, (("A", "A"), ("D", "D"), ("K", "K"), ("T1", "T₁"), ("T2", "T₂"), ("full Y", "Y"), ("candidate Ŷ", "Ŷ"), ("match", "Match")))),
        mo.md("With the zero baseline, either threat copy plus the king gate is sufficient on this distribution. Neither copy is individually necessary in the full implementation. That is redundancy, not evidence that threat computation is irrelevant."),
    ])
    return


@app.cell
def necessity_analysis(mo, table, toy_circuit_truth_table):
    necessity_truth_table = toy_circuit_truth_table()
    necessity_rows = []
    for component in ("threat path T₁", "redundant path T₂", "king-exposure gate K"):
        changed = 0
        for row in necessity_truth_table:
            threat = 0 if component == "threat path T₁" else row["threat"]
            redundant = 0 if component == "redundant path T₂" else row["redundant_copy"]
            gate = 0 if component == "king-exposure gate K" else row["king_exposed"]
            changed += int(((threat | redundant) & gate) != row["output"])
        necessity_rows.append({"component": component, "inputs changed by ablation": changed, "individually necessary?": changed > 0})
    mo.vstack([
        mo.md("## Necessity, sufficiency, and redundancy are separate"),
        mo.Html(table(necessity_rows, (("component", "Component"), ("inputs changed by ablation", "Changed / 8"), ("individually necessary?", "Individually necessary")))),
    ])
    return


@app.cell
def seven_criterion_audit(
    candidate_components,
    candidate_outputs_array,
    mo,
    np,
    table,
    toy_circuit_truth_table,
):
    truth = toy_circuit_truth_table()
    audit_nodes = set(candidate_components.value)
    full = np.asarray([row["output"] for row in truth])

    def output_with_zeroed(zeroed):
        values = []
        for row in truth:
            t1 = 0 if "threat path T₁" in zeroed else row["threat"]
            t2 = 0 if "redundant path T₂" in zeroed else row["redundant_copy"]
            gate = 0 if "king-exposure gate K" in zeroed else row["king_exposed"]
            values.append((t1 | t2) & gate)
        return np.asarray(values)

    all_components = {"threat path T₁", "redundant path T₂", "king-exposure gate K"}
    excluded = all_components - audit_nodes
    set_ablated = output_with_zeroed(audit_nodes)
    excluded_ablated = output_with_zeroed(excluded)
    k_zero = np.asarray([row["king_exposed"] == 0 for row in truth])
    k_one = ~k_zero
    stability = min(
        float(np.mean(candidate_outputs_array[mask] == full[mask]))
        for mask in (k_zero, k_one)
    )
    joint_threat_ablation = output_with_zeroed({"threat path T₁", "redundant path T₂"})
    criterion_rows = [
        {"criterion": "faithfulness", "computed quantity": "Pr[candidate Ŷ = full Y]", "value": f"{np.mean(candidate_outputs_array == full):.1%}"},
        {"criterion": "sufficiency", "computed quantity": "retain candidate from zero baseline", "value": f"{np.mean(candidate_outputs_array == full):.1%}"},
        {"criterion": "set necessity", "computed quantity": "Pr[Y changes when all selected nodes are zeroed]", "value": f"{np.mean(set_ablated != full):.1%}"},
        {"criterion": "completeness", "computed quantity": "Pr[Y unchanged when all excluded nodes are zeroed]", "value": f"{np.mean(excluded_ablated == full):.1%}"},
        {"criterion": "redundancy", "computed quantity": "Pr[Y changes only when T₁ and T₂ are jointly zeroed]", "value": f"{np.mean(joint_threat_ablation != full):.1%}"},
        {"criterion": "stability", "computed quantity": "minimum faithfulness over K=0 and K=1 strata", "value": f"{stability:.1%}"},
        {"criterion": "interchange", "computed quantity": "tested below on source/base counterfactuals", "value": "pending selected interchange"},
    ]
    mo.vstack([
        mo.md("## One specimen, seven distinct questions"),
        mo.Html(table(criterion_rows, (("criterion", "Criterion"), ("computed quantity", "Operational computation"), ("value", "Current candidate")))),
        mo.md("Completeness here is explicitly redundancy-relative: an excluded copy can be dispensable because an included copy substitutes. Stability across two toy strata is only a worked definition, not evidence of real cross-position stability."),
    ])
    return


@app.cell
def interchange_controls(mo):
    base_attack = mo.ui.switch(value=True, label="base attack A")
    base_defender = mo.ui.switch(value=False, label="base defender D")
    base_exposed = mo.ui.switch(value=True, label="base king exposed K")
    source_attack = mo.ui.switch(value=False, label="source attack A")
    source_defender = mo.ui.switch(value=False, label="source defender D")
    patch_scope = mo.ui.radio(
        options={"Patch only T₁": "one_copy", "Patch both T₁ and T₂": "both_copies"},
        value="Patch only T₁",
        label="Low-level realization of abstract threat interchange",
    )
    mo.vstack([
        mo.md("**Base example**"), mo.hstack([base_attack, base_defender, base_exposed]),
        mo.md("**Source example**"), mo.hstack([source_attack, source_defender]),
        patch_scope,
    ])
    return (
        base_attack,
        base_defender,
        base_exposed,
        patch_scope,
        source_attack,
        source_defender,
    )


@app.cell
def interchange_experiment(
    base_attack,
    base_defender,
    base_exposed,
    patch_scope,
    source_attack,
    source_defender,
):
    base_threat = int(base_attack.value and not base_defender.value)
    source_threat = int(source_attack.value and not source_defender.value)
    abstract_counterfactual = source_threat & int(base_exposed.value)
    patched_threat = source_threat
    patched_redundant = source_threat if patch_scope.value == "both_copies" else base_threat
    low_level_counterfactual = (patched_threat | patched_redundant) & int(base_exposed.value)
    interchange_consistent = low_level_counterfactual == abstract_counterfactual
    return (
        abstract_counterfactual,
        base_threat,
        interchange_consistent,
        low_level_counterfactual,
        patched_redundant,
        patched_threat,
        source_threat,
    )


@app.cell
def interchange_view(
    abstract_counterfactual,
    base_threat,
    callout,
    interchange_consistent,
    low_level_counterfactual,
    metric_cards,
    mo,
    patched_redundant,
    patched_threat,
    source_threat,
):
    mo.vstack([
        mo.md("## Interchange: does the diagram commute?"),
        mo.Html(metric_cards([
            (base_threat, "base abstract threat"),
            (source_threat, "source abstract threat"),
            (f"T₁={patched_threat}, T₂={patched_redundant}", "patched low-level copies"),
            (abstract_counterfactual, "abstract counterfactual Y"),
            (low_level_counterfactual, "low-level patched Y"),
            (interchange_consistent, "interchange consistent"),
        ])),
        mo.Html(callout(
            "positive" if interchange_consistent else "limit",
            "Intervention commutes" if interchange_consistent else "Redundant realization breaks the one-node patch",
            "An abstract variable can be implemented redundantly. A valid low-level interchange must patch the complete realization named by the correspondence, or explicitly model the leftover copy.",
        )),
    ])
    return


@app.cell
def abstraction_definition(mo):
    mo.md(r"""
    ## Causal abstraction in one equation

    Let \(M_L\) be the neural/low-level model, \(M_H\) an abstract algorithm,
    \(\tau\) map low-level states to abstract variables, and \(\omega\) map an
    abstract intervention to its low-level realization. Interchange consistency
    asks whether, on a declared input and intervention distribution,

    \[
    \tau\!\left(M_L\mid \omega(do(V_H=v))\right)
    \approx
    M_H\mid do(V_H=v).
    \]

    The approximation needs a metric and margin. A single successful patch is
    one row of this test, not a causal abstraction result by itself.
    """)
    return


@app.cell
def real_evidence_audit(
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
        circuits = bundle.section("circuits")
        audit_rows = circuits["evidence_matrix"]
        audit_card = ClaimCard(
            operation=EvidenceOperation.OBSERVATION,
            target=TargetLevel.COMPONENT,
            scope="retained Raw/Hero development evidence families summarized in the frozen course snapshot",
            endpoint=Endpoint.POLICY,
            estimand="which component-level and predictive criteria have recorded evidence versus missing circuit criteria",
            unit="retained experiment family",
            grouping="method family and its recorded sample",
            intervention=None,
            controls=("identity patches", "hybrid swaps", "probe controls", "matched LoRSA random directions"),
            assumptions=("each retained record is interpreted only within its documented scope",),
            uncertainty="heterogeneous exploratory records; no pooled effect",
            multiplicity="not pooled; each family retains its own scan correction",
            allowed="The retained evidence localizes Raw/Hero policy differences to encoder and late residual/branch components and records predictive and scoped sparse-intervention results.",
            excluded="No retained experiment establishes a complete, minimal, stable Raw/Hero circuit or an interchange-consistent abstract algorithm.",
            falsifier="A complete preregistered interchange record passing faithfulness, completeness, necessity, sufficiency, redundancy, and stability gates.",
            status=ResultStatus.UNRESOLVED,
        )
        evidence_view = mo.vstack([
            mo.md("## Where the current Raw/Hero evidence actually stops"),
            mo.Html(table(audit_rows, (("method", "Method"), ("source_run", "Run identity"), ("evidence_rank", "Rank"), ("finding", "Finding"), ("scope", "Scope"), ("main_limit", "Main limit")))),
            mo.Html(claim_card_html(audit_card)),
            mo.Html(callout("limit", "The missing bridge", "Whole-residual patching is a strong locator; LoRSA directions are scoped component interventions. We still need edge/path hypotheses, redundancy-aware necessity/sufficiency, cross-position stability, and interchange tests before naming a circuit.")),
        ])
    else:
        evidence_view = mo.Html(callout("neutral", "Toy boundary", "The exhaustive Boolean mechanism supports a circuit claim about the construction only."))
    evidence_view
    return


@app.cell
def automated_discovery(mo, table):
    discovery_rows = [
        {"risk": "metric circularity", "example": "select edges and score with the same corrupted-logit objective", "repair": "held-out interventions and alternate endpoints"},
        {"risk": "threshold instability", "example": "tiny cutoff change doubles circuit size", "repair": "size–faithfulness frontier and seed stability"},
        {"risk": "missing redundancy", "example": "greedy ablation drops either substitute path", "repair": "joint ablations and interaction terms"},
        {"risk": "baseline artifact", "example": "zero/mean patch creates off-manifold activation", "repair": "resampled, counterfactual, and density-matched baselines"},
        {"risk": "overlarge explanation", "example": "nearly the full model is faithful", "repair": "compare smaller and random matched subgraphs"},
    ]
    mo.vstack([
        mo.md("## Automated circuit discovery proposes; independent tests dispose"),
        mo.Html(table(discovery_rows, (("risk", "Risk"), ("example", "Failure"), ("repair", "Required repair")))),
    ])
    return


@app.cell
def claim_prediction(committed_form, mo):
    claim_choice = mo.ui.radio(
        options=[
            "A layer-14 whole-residual patch recovers policy, therefore layer 14 is the circuit",
            "The patch localizes a sufficient whole-residual state on eight positions; a circuit remains unresolved",
            "A high probe plus attention map establishes the algorithm",
        ],
        value=None,
        label="Strongest current Raw/Hero conclusion",
    )
    claim_form = committed_form(
        mo, {"strongest_claim": claim_choice}, submit_label="Commit circuit audit"
    )
    claim_form
    return (claim_form,)


@app.cell
def claim_feedback(callout, claim_form, mo):
    mo.stop(claim_form.value is None, mo.md("Commit before revealing."))
    claim_ok = claim_form.value["strongest_claim"].startswith("The patch localizes")
    mo.Html(callout(
        "positive" if claim_ok else "danger",
        "Evidence boundary preserved" if claim_ok else "Locator-to-circuit leap",
        "A whole residual is a high-bandwidth carrier. Recovery shows that enough information/state is present at that boundary under the patch; it does not decompose the computation that produced or reads it.",
    ))
    return


@app.cell
def assessments(committed_form, mo, module):
    criteria_transfer = mo.ui.text_area(label="Transfer 1 — criteria", placeholder="A two-path LM circuit is 98% faithful and survives either single ablation on one template. Audit faithfulness, completeness, necessity, sufficiency, redundancy, cross-template stability, and the next test.")
    interchange_transfer = mo.ui.text_area(label="Transfer 2 — abstraction", placeholder="Map a high-level negation variable to low-level heads and define a commuting interchange experiment.")
    alternative_transfer = mo.ui.text_area(label="Transfer 3 — alternatives", placeholder="An automated 40-head circuit is 98% faithful. Specify smaller, random, and independently seeded comparisons.")
    assessment_form = committed_form(
        mo,
        {"criteria": criteria_transfer, "interchange": interchange_transfer, "alternatives": alternative_transfer},
        submit_label="Commit responses and reveal circuit rubric",
        min_words=5,
    )
    mo.vstack([mo.md(f"## Unseen transfer assessment\n\n{module.transfer_assessment}"), assessment_form])
    return (assessment_form,)


@app.cell
def assessment_feedback(assessment_form, mo):
    mo.stop(assessment_form.value is None, mo.md("Submit every objective before revealing."))
    answer_lengths = [len(assessment_form.value[key].split()) for key in ("criteria", "interchange", "alternatives")]
    mo.md(f"""
    **Rubric (response lengths {answer_lengths}):** distinguish **faithfulness**
    (retained behavior) from **completeness** (excluded components add no needed
    computation); test **necessity** with single and joint ablations,
    **sufficiency** with isolated activation, **redundancy** with interaction
    terms, and **stability** across templates/seeds. Define low/high variables, correspondence,
    intervention map, distribution, endpoint, and equivalence margin; compare
    size–faithfulness frontiers, random matched subgraphs, independent seeds,
    alternate baselines/endpoints, and held-out interchange behavior.
    """)
    return


@app.cell
def close(bundle, callout, mo, resource_card):
    mo.vstack([
        mo.Html(callout("positive", "Positive result", "On a known mechanism, exhaustive truth tables and interchange interventions distinguish a minimal sufficient implementation from redundant copies.")),
        mo.Html(callout("limit", "What this does not show", "The retained Raw/Hero stack has strong localization and scoped component evidence, but no complete interchange-consistent circuit result yet.")),
        mo.md("""
        ## Retrieval before the capstone

        1. Why can two individually unnecessary paths be jointly necessary?
        2. What four objects define an interchange test?
        3. Why is 98% faithfulness not completeness?

        **Sources:** Geiger et al. (JMLR 2025) for causal
        abstraction, Conmy et al. (NeurIPS 2023) for ACDC, and Mueller et al.
        (ICML 2025) for circuit evaluation. See the
        [course source register](../SOURCES.md) and the explicit
        deferred path-patching/causal-scrubbing boundary in
        [method-coverage matrix](../METHOD_COVERAGE.md).
        """),
        mo.Html(resource_card(
            mode=bundle.mode.value,
            runtime="< 2 seconds; exhaustive eight-row circuit",
            device="CPU only; no model or network",
            determinism="exhaustive truth table; no RNG; integer and Boolean arithmetic",
            sources=f"content identity {bundle.integrity_sha256[:16]}…",
            limitations="Known Boolean mechanism plus a heterogeneous retained evidence audit; no Raw/Hero interchange experiment.",
        )),
    ])
    return


if __name__ == "__main__":
    app.run()
