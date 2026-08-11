# /// script
# dependencies = ["marimo", "numpy"]
# requires-python = ">=3.11"
# ///

import marimo

__generated_with = "0.23.3"
app = marimo.App(width="full")


@app.cell
def imports():
    import json
    import sys
    from pathlib import Path

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
        capstone_control_families,
        capstone_record_gates,
        classify_language,
    )
    from research.interpretability.course.ui import (
        accessible_text,
        accessible_text_area,
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
        accessible_text,
        accessible_text_area,
        callout,
        capstone_control_families,
        capstone_record_gates,
        claim_card_html,
        classify_language,
        committed_form,
        get_module,
        json,
        load_course_bundle,
        metric_cards,
        mo,
        mode_badge,
        module_header,
        resource_card,
        table,
    )


@app.cell
def title(get_module, mo, module_header):
    module = get_module(10)
    mo.Html(module_header(module))
    return (module,)


@app.cell
def opening(mo, module, table):
    objective_rows = [
        {"objective": objective, "deliverable": deliverable}
        for objective, deliverable in zip(
            module.objectives,
            (
                "A timestampable hypothesis, claim card, and immutable promotion/rejection gate.",
                "A manifest and analysis plan another researcher can execute without oral context.",
                "A conclusion whose score does not depend on whether the central hypothesis survives.",
            ),
            strict=True,
        )
    ]
    mo.vstack([
        mo.md(r"""
        ## The product is a reviewable scientific record

        The capstone is not “find an interpretable feature.” It is: define a
        mechanism precisely enough that evidence can promote, constrain, or
        falsify it without changing the rules after seeing results.

        A rigorous falsification can receive full credit. A vivid positive
        example with leaked selection, weak controls, or an unfrozen claim
        cannot. This notebook turns your design into a portable JSON record and
        runs a structural self-audit; human scientific review is still required.
        """),
        mo.Html(table(objective_rows, (("objective", "Objective"), ("deliverable", "Required deliverable")))),
    ])
    return


@app.cell
def prerequisite(committed_form, mo):
    response = mo.ui.text_area(
        label="Prerequisite diagnostic — frozen scientific claim",
        placeholder="Name the four claim axes, the strongest live alternative, and a result that would narrow or falsify your planned mechanism.",
    )
    prerequisite_form = committed_form(
        mo, {"claim_contract": response}, submit_label="Commit prerequisite response", min_words=10
    )
    prerequisite_form
    return (prerequisite_form,)


@app.cell
def prerequisite_feedback(mo, prerequisite_form):
    mo.stop(prerequisite_form.value is None, mo.md("Answer the prerequisite diagnostic to reveal the bridge."))
    mo.callout("A reviewable record binds operation, target, scope, and endpoint to an estimand, grouping unit, controls, uncertainty, artifact identities, strongest alternative, and numerical promotion/rejection/equivalence rules.", kind="info")
    return


@app.cell
def mode_controls(mo):
    evidence_mode = mo.ui.radio(
        options={
            "Snapshot — bind capstone to frozen course evidence": "snapshot",
            "Toy — practice-only preregistration": "toy",
            "Source — verify all retained course sources first": "source",
        },
        value="Snapshot — bind capstone to frozen course evidence",
        label="Evidence mode",
    )
    verify_source = mo.ui.run_button(label="Verify retained sources", kind="warn")
    mo.hstack([evidence_mode, verify_source], justify="start", gap=1)
    return evidence_mode, verify_source


@app.cell
def load_mode(evidence_mode, load_course_bundle, mo, verify_source):
    selected_mode = evidence_mode.value
    mo.stop(selected_mode == "source" and not verify_source.value, mo.callout("Source mode requires explicit checksum verification before preregistration.", kind="warn"))
    bundle = load_course_bundle(selected_mode)
    return bundle, selected_mode


@app.cell
def mode_view(bundle, mo, mode_badge, selected_mode):
    mo.Html(mode_badge(selected_mode, bundle.evidence_scope))
    return


@app.cell
def workflow(mo, table):
    workflow_rows = [
        {"phase": "1 · discovery", "may do": "explore layers/features/examples; develop operational labels", "must not do": "call final evidence confirmatory"},
        {"phase": "2 · freeze", "may do": "sign claim, estimator, code identity, controls, margins, seeds", "must not do": "inspect held-out outcomes"},
        {"phase": "3 · selection", "may do": "choose hyperparameters under the frozen rule", "must not do": "retune on evaluation"},
        {"phase": "4 · evaluation", "may do": "run once; report every frozen endpoint and failure", "must not do": "replace primary endpoint post hoc"},
        {"phase": "5 · replication", "may do": "independent groups/seeds/artifacts and a stronger alternative", "must not do": "treat same-sample rerun as independent"},
    ]
    mo.Html(table(workflow_rows, (("phase", "Phase"), ("may do", "Permitted"), ("must not do", "Prohibited"))))
    return


@app.cell
def worked_example(mo):
    mo.accordion({
        "Worked example — inspect, do not submit as your capstone": mo.md(r"""
        **Question.** On group-disjoint roots where Raw-trained LoRSA feature
        10843 is naturally active, does ablating its frozen decoder
        contribution change legal-policy TV more than matched random and
        own-queen placebo directions in both Raw and Hero?

        - **Operation × target × endpoint:** intervention × component × legal-policy distribution.
        - **Unit/grouping:** naturally active position; resample unique game/root groups.
        - **Controls:** no-op, three scalar/token/norm-matched random directions,
          source-selected own-queen semantic placebo, insertion positive control,
          symmetric dose, legality remasking, and density diagnostics.
        - **Gate:** both model arms must have a Holm-adjusted target-minus-control
          interval above zero, exceed the frozen placebo, and pass fidelity and
          opportunity checks; otherwise falsified or unresolved under the
          prespecified opportunity/power rules.
        - **Excluded:** no native dense-feature, input-concept-causality,
          monosemanticity, internal-search, or confirmatory-generalization claim.
        - **Replication:** new game clusters plus independently learned sparse
          coordinates, with feature mapping frozen before evaluation.

        This example illustrates field specificity. It is not copied into the
        blank learner record and its retained development result is not a new
        confirmatory claim.
        """)
    })
    return


@app.cell
def hypothesis_controls(accessible_text, accessible_text_area, mo):
    project_title = accessible_text(mo, value="", label="Project title", full_width=True)
    hypothesis_family = mo.ui.dropdown(
        options={
            "Raw/Hero behavioral or representation difference": "model_diff",
            "Probe decodability and causal-use follow-up": "probe_use",
            "Inference-matched DFM–JEPA lookahead": "lookahead",
            "LoRSA sparse feature intervention": "lorsa",
            "Circuit / causal abstraction": "circuit",
        },
        value=None,
        label="Hypothesis family",
    )
    hypothesis = accessible_text_area(
        mo,
        value="",
        label="Frozen directional hypothesis",
        full_width=True,
    )
    strongest_alternative = accessible_text_area(
        mo,
        value="",
        label="Strongest alternative explanation",
        full_width=True,
    )
    mo.vstack([project_title, hypothesis_family, hypothesis, strongest_alternative])
    return hypothesis, hypothesis_family, project_title, strongest_alternative


@app.cell
def matrix_controls(mo):
    operation_choice = mo.ui.dropdown(
        options={
            "observation": "observation",
            "prediction": "prediction",
            "intervention": "intervention",
            "interchange / causal abstraction": "interchange",
        },
        value=None,
        label="Evidence operation",
    )
    target_choice = mo.ui.dropdown(
        options={
            "behavior": "behavior",
            "activation": "activation",
            "representation": "representation",
            "component": "component",
            "path / circuit": "circuit",
        },
        value=None,
        label="Target level",
    )
    endpoint_choice = mo.ui.dropdown(
        options={
            "representation metric": "representation",
            "probe objective": "probe",
            "logit": "logit",
            "legal action": "legal_action",
            "policy distribution": "policy",
            "value": "value",
            "arena result": "arena",
        },
        value=None,
        label="Endpoint",
    )
    mo.hstack([operation_choice, target_choice, endpoint_choice], justify="start")
    return endpoint_choice, operation_choice, target_choice


@app.cell
def design_fields(accessible_text, accessible_text_area, mo):
    scope = accessible_text_area(mo, value="", label="Scope and sampled population")
    estimand = accessible_text_area(mo, value="", label="Endpoint and exact estimand")
    unit = accessible_text(mo, value="", label="Unit", full_width=True)
    grouping = accessible_text(mo, value="", label="Grouping / independent unit", full_width=True)
    intervention = accessible_text_area(mo, value="", label="Intervention (write 'none' for non-interventional work)")
    assumptions = accessible_text_area(mo, value="", label="Identification assumptions")
    uncertainty = accessible_text_area(mo, value="", label="Uncertainty procedure")
    multiplicity = accessible_text(mo, value="", label="Multiplicity rule", full_width=True)
    mo.vstack([scope, estimand, mo.hstack([unit, grouping]), intervention, assumptions, uncertainty, multiplicity])
    return (
        assumptions,
        estimand,
        grouping,
        intervention,
        multiplicity,
        scope,
        uncertainty,
        unit,
    )


@app.cell
def control_fields(accessible_text_area, mo):
    controls = mo.ui.multiselect(
        options=[
            "identity / no-op",
            "simple-input or representation baseline",
            "label shuffle / shuffled pairing",
            "untrained or independent-seed model",
            "matched random direction/subgraph",
            "semantic placebo",
            "positive control",
            "symmetric sign and dose curve",
            "density / off-manifold diagnostic",
            "legality and exact replay",
            "smaller and larger circuit alternatives",
        ],
        value=[],
        label="Frozen controls (select only those you specify operationally)",
    )
    control_protocol = accessible_text_area(
        mo,
        value="",
        label="Exact control matching and dose protocol",
    )
    mo.vstack([
        mo.md("""
        Structural control credit is **operation-aware**:

        - observation — identity, simple-input/representation baseline,
          shuffled pairing, independent-seed model;
        - prediction — simple-input baseline, label shuffle, untrained model,
          known positive;
        - intervention — no-op, norm-matched random, semantic placebo, known
          positive; and
        - interchange — no-op, matched random, known positive, and smaller or
          larger circuit alternatives.

        Extra diagnostics can still be essential, but selecting an irrelevant
        causal control does not raise an observational project's score.
        """),
        controls,
        control_protocol,
    ])
    return control_protocol, controls


@app.cell
def gate_fields(accessible_text_area, mo):
    promotion = accessible_text_area(
        mo,
        value="",
        label="Promotion criterion",
    )
    rejection = accessible_text_area(
        mo,
        value="",
        label="Rejection / falsification criterion",
    )
    equivalence = accessible_text_area(
        mo, value="", label="Equivalence margin and test (or 'not claimed')"
    )
    falsifier = accessible_text_area(
        mo,
        value="", label="Strongest falsifier / promotion experiment"
    )
    mo.vstack([promotion, rejection, equivalence, falsifier])
    return equivalence, falsifier, promotion, rejection


@app.cell
def conclusion_fields(accessible_text_area, mo):
    result_status_choice = mo.ui.dropdown(
        options={
            "positive under the stated criterion": "positive",
            "falsified under the stated criterion": "falsified",
            "unresolved / failure to reject": "unresolved",
            "negative control": "negative_control",
            "evidence of equivalence under margin": "equivalent",
            "descriptive only": "descriptive",
        },
        value="unresolved / failure to reject",
        label="Result status (fill after running; default is unresolved)",
    )
    allowed = accessible_text_area(
        mo,
        value="",
        label="Allowed conclusion",
    )
    excluded = accessible_text_area(
        mo,
        value="",
        label="Explicitly excluded conclusion",
    )
    mo.vstack([result_status_choice, allowed, excluded])
    return allowed, excluded, result_status_choice


@app.cell
def provenance_fields(accessible_text, accessible_text_area, bundle, mo):
    code_identity = accessible_text(mo, value="", label="Code identity — commit SHA + clean/dirty, or code SHA-256", full_width=True)
    artifact_identity = accessible_text(mo, value="", label="Primary artifact SHA-256", full_width=True)
    environment_identity = accessible_text(mo, value="", label="Environment/device/dtype identity", full_width=True)
    data_roles = accessible_text_area(mo, value="", label="Discovery / selection / evaluation / test / replication roles")
    license_record = accessible_text_area(mo, value="", label="License and redistribution record")
    seeds = accessible_text(mo, value="", label="Seeds and deterministic settings", full_width=True)
    freeze_timestamp = accessible_text(mo, value="", label="Preregistration freeze time — ISO-8601 UTC, e.g. 2026-08-11T18:30Z", full_width=True)
    mo.vstack([
        mo.md(f"Verified course evidence identity (reference only): `{bundle.integrity_sha256}`. Enter the identity of the artifact your capstone actually tests."),
        code_identity, artifact_identity, environment_identity, data_roles, license_record, seeds, freeze_timestamp,
    ])
    return (
        artifact_identity,
        code_identity,
        data_roles,
        environment_identity,
        freeze_timestamp,
        license_record,
        seeds,
    )


@app.cell
def replication_fields(accessible_text_area, mo):
    replication = accessible_text_area(
        mo,
        value="",
        label="Independent replication plan",
    )
    reviewer_objection = mo.ui.dropdown(
        options=[
            "Your semantic placebo was selected after seeing outcomes.",
            "Positions, not games, define your interval.",
            "The intervention leaves the activation manifold.",
            "Your null result is merely underpowered.",
            "The sparse feature is not a native dense component.",
        ],
        value=None,
        label="Cold-review objection",
    )
    reviewer_response = accessible_text_area(mo, label="Response: change the claim or design; do not merely reassure the reviewer", placeholder="Name evidence, limitation, and a concrete repair.")
    mo.vstack([replication, reviewer_objection, reviewer_response])
    return replication, reviewer_objection, reviewer_response


@app.cell
def build_record(
    ClaimCard,
    Endpoint,
    EvidenceOperation,
    ResultStatus,
    TargetLevel,
    allowed,
    artifact_identity,
    assumptions,
    bundle,
    code_identity,
    control_protocol,
    controls,
    data_roles,
    endpoint_choice,
    environment_identity,
    equivalence,
    estimand,
    excluded,
    falsifier,
    freeze_timestamp,
    grouping,
    hypothesis,
    hypothesis_family,
    intervention,
    license_record,
    mo,
    multiplicity,
    operation_choice,
    project_title,
    promotion,
    rejection,
    replication,
    result_status_choice,
    reviewer_objection,
    reviewer_response,
    scope,
    seeds,
    strongest_alternative,
    target_choice,
    uncertainty,
    unit,
):
    mo.stop(
        any(
            choice.value is None
            for choice in (hypothesis_family, operation_choice, target_choice, endpoint_choice)
        ),
        mo.callout(
            "Choose a hypothesis family and every claim-matrix axis before building the record.",
            kind="warn",
        ),
    )
    operation_map = {
        "observation": EvidenceOperation.OBSERVATION,
        "prediction": EvidenceOperation.PREDICTION,
        "intervention": EvidenceOperation.INTERVENTION,
        "interchange": EvidenceOperation.INTERCHANGE,
    }
    target_map = {
        "behavior": TargetLevel.BEHAVIOR,
        "activation": TargetLevel.ACTIVATION,
        "representation": TargetLevel.REPRESENTATION,
        "component": TargetLevel.COMPONENT,
        "circuit": TargetLevel.CIRCUIT,
    }
    endpoint_map = {
        "representation": Endpoint.REPRESENTATION,
        "probe": Endpoint.PROBE,
        "logit": Endpoint.LOGIT,
        "legal_action": Endpoint.LEGAL_ACTION,
        "policy": Endpoint.POLICY,
        "value": Endpoint.VALUE,
        "arena": Endpoint.ARENA,
    }
    status_map = {
        "positive": ResultStatus.POSITIVE,
        "falsified": ResultStatus.FALSIFIED,
        "unresolved": ResultStatus.UNRESOLVED,
        "negative_control": ResultStatus.NEGATIVE_CONTROL,
        "equivalent": ResultStatus.EQUIVALENT,
        "descriptive": ResultStatus.DESCRIPTIVE,
    }
    declared_intervention = intervention.value.strip()
    if operation_choice.value in {"observation", "prediction"}:
        declared_intervention = None
    capstone_card = ClaimCard(
        operation=operation_map[operation_choice.value],
        target=target_map[target_choice.value],
        scope=scope.value,
        endpoint=endpoint_map[endpoint_choice.value],
        estimand=estimand.value,
        unit=unit.value,
        grouping=grouping.value,
        intervention=declared_intervention,
        controls=tuple(controls.value),
        assumptions=tuple(part.strip() for part in assumptions.value.split(";") if part.strip()),
        uncertainty=uncertainty.value + (f"; equivalence: {equivalence.value}" if result_status_choice.value == "equivalent" else ""),
        multiplicity=multiplicity.value,
        allowed=allowed.value,
        excluded=excluded.value,
        falsifier=falsifier.value,
        status=status_map[result_status_choice.value],
    )
    card_problems = capstone_card.problems()
    capstone_record = {
        "schema_version": "chess-interpretability-capstone-v1",
        "project_title": project_title.value,
        "hypothesis_family": hypothesis_family.value,
        "hypothesis": hypothesis.value,
        "strongest_alternative": strongest_alternative.value,
        "claim_card": capstone_card.as_dict(),
        "control_protocol": control_protocol.value,
        "gates": {"promotion": promotion.value, "rejection": rejection.value, "equivalence": equivalence.value},
        "provenance": {
            "code_identity": code_identity.value,
            "artifact_identity": artifact_identity.value,
            "course_snapshot_id": bundle.snapshot_id,
            "course_mode": bundle.mode.value,
            "environment": environment_identity.value,
            "data_roles": data_roles.value,
            "freeze_timestamp_utc": freeze_timestamp.value,
            "seeds": seeds.value,
            "license": license_record.value,
        },
        "replication": replication.value,
        "cold_review": {"objection": reviewer_objection.value, "response": reviewer_response.value},
        "structural_validation": {"claim_card_problems": list(card_problems)},
    }
    return capstone_card, capstone_record, card_problems


@app.cell
def card_view(callout, capstone_card, card_problems, claim_card_html, mo):
    if card_problems:
        card_output = mo.Html(callout("danger", "Claim card is not valid yet", " · ".join(card_problems)))
    else:
        card_output = mo.vstack([mo.md("## Generated claim card"), mo.Html(claim_card_html(capstone_card))])
    card_output
    return


@app.cell
def structural_audit(
    artifact_identity,
    audit_run,
    capstone_control_families,
    capstone_record_gates,
    card_problems,
    classify_language,
    code_identity,
    control_protocol,
    controls,
    data_roles,
    environment_identity,
    equivalence,
    estimand,
    falsifier,
    freeze_timestamp,
    grouping,
    hypothesis,
    license_record,
    mo,
    multiplicity,
    operation_choice,
    project_title,
    promotion,
    rejection,
    replication,
    result_status_choice,
    reviewer_objection,
    reviewer_response,
    scope,
    seeds,
    strongest_alternative,
    target_choice,
    uncertainty,
    unit,
):
    mo.stop(not audit_run.value, "Press **Run structural self-audit** after completing the learner record.")
    language = classify_language(hypothesis.value + " " + strongest_alternative.value)
    claim_points = 15 if not card_problems else max(0, 15 - 3 * len(card_problems))
    design_fields = [scope.value, estimand.value, unit.value, grouping.value, uncertainty.value, multiplicity.value]
    design_points = min(20, 3 * sum(len(value.split()) >= 3 for value in design_fields) + 2)
    applicable_control_types = capstone_control_families(operation_choice.value)
    covered_control_types = set(controls.value) & applicable_control_types
    control_points = min(
        20,
        3 * len(covered_control_types)
        + min(8, len(control_protocol.value.split()) // 5),
    )
    gate_points = min(15, 5 * sum(len(value.value.split()) >= 8 for value in (promotion, rejection, equivalence)))
    provenance_values = [artifact_identity.value, code_identity.value, environment_identity.value, data_roles.value, license_record.value, seeds.value]
    record_gates = capstone_record_gates(
        project_title=project_title.value,
        hypothesis=hypothesis.value,
        strongest_alternative=strongest_alternative.value,
        artifact_identity=artifact_identity.value,
        code_identity=code_identity.value,
        environment_identity=environment_identity.value,
        data_roles=data_roles.value,
        license_record=license_record.value,
        seeds=seeds.value,
        freeze_timestamp=freeze_timestamp.value,
        promotion=promotion.value,
        rejection=rejection.value,
        equivalence=equivalence.value,
        falsifier=falsifier.value,
        control_protocol=control_protocol.value,
        replication=replication.value,
        reviewer_objection=reviewer_objection.value or "",
        reviewer_response=reviewer_response.value,
    )
    provenance_points = min(15, 2 * sum(len(value.split()) >= 3 for value in provenance_values) + 3 * int(record_gates["artifact identity is a hexadecimal SHA-256"]))
    synthesis_values = [strongest_alternative.value, falsifier.value, replication.value, reviewer_response.value]
    synthesis_points = min(15, 3 * sum(len(value.split()) >= 8 for value in synthesis_values) + 3)
    audit_score = claim_points + design_points + control_points + gate_points + provenance_points + synthesis_points
    hard_gates = {
        "claim card validates": not card_problems,
        "circuit target uses interchange": target_choice.value != "circuit" or operation_choice.value == "interchange",
        "causal operation has all applicable control families": (
            operation_choice.value not in {"intervention", "interchange"}
            or applicable_control_types.issubset(set(controls.value))
        ),
        **record_gates,
        "equivalence wording has a margin": result_status_choice.value != "equivalent" or "margin" in equivalence.value.lower(),
        "planning language has a named operational estimand": (
            not language["contains_planning_language"]
            or len(estimand.value.split()) >= 6
        ),
    }
    return (
        audit_score,
        claim_points,
        control_points,
        design_points,
        gate_points,
        hard_gates,
        provenance_points,
        synthesis_points,
    )


@app.cell
def audit_control(mo):
    audit_run = mo.ui.run_button(label="Run structural self-audit", kind="success")
    audit_run
    return (audit_run,)


@app.cell
def audit_view(
    audit_score,
    callout,
    claim_points,
    control_points,
    design_points,
    gate_points,
    hard_gates,
    metric_cards,
    mo,
    provenance_points,
    synthesis_points,
    table,
):
    gate_rows = [{"gate": name, "passed": passed} for name, passed in hard_gates.items()]
    all_hard_gates = all(hard_gates.values())
    mo.vstack([
        mo.md("## Structural self-audit — not an automated scientific grade"),
        mo.Html(metric_cards([
            (f"{audit_score}/100", "structural completeness"),
            (f"{claim_points}/15", "claim calibration"),
            (f"{design_points}/20", "design"),
            (f"{control_points}/20", "controls"),
            (f"{gate_points}/15", "frozen gates"),
            (f"{provenance_points}/15", "provenance"),
            (f"{synthesis_points}/15", "alternatives / replication"),
        ])),
        mo.Html(table(gate_rows, (("gate", "Hard gate"), ("passed", "Passed")))),
        mo.Html(callout(
            "positive" if audit_score >= 90 and all_hard_gates else "limit",
            "Ready for independent review" if audit_score >= 90 and all_hard_gates else "Structural work remains",
            "The score checks filled fields and minimum control families. It cannot judge label validity, statistical power, off-manifold behavior, artifact truth, or whether a semantic placebo is genuinely plausible.",
        )),
    ])
    return


@app.cell
def export_record(
    audit_run,
    audit_score,
    capstone_record,
    hard_gates,
    json,
    mo,
):
    mo.stop(
        not audit_run.value or audit_score < 90 or not all(hard_gates.values()),
        mo.md("A JSON export is enabled only after a deliberate audit run, a score of at least 90, and every hard structural gate passes."),
    )
    serialized_record = json.dumps(capstone_record, indent=2, sort_keys=True, ensure_ascii=False)
    mo.vstack([
        mo.md("## Export the preregistration record"),
        mo.download(serialized_record, filename="interpretability_capstone_record.json", mimetype="application/json", label="Download JSON record"),
        mo.accordion({"Preview JSON": mo.md(f"```json\n{serialized_record}\n```")}),
    ])
    return


@app.cell
def result_reporting(mo, table):
    status_rows = [
        {"status": "positive", "write": "criterion passed in the scoped sample", "never substitute": "proved the universal story"},
        {"status": "falsified", "write": "frozen directional/specificity gate failed with diagnostics intact", "never substitute": "the entire research program is false"},
        {"status": "unresolved", "write": "interval/power/opportunity leaves the gate undecided", "never substitute": "no effect or models are equal"},
        {"status": "negative control", "write": "assay behaved as expected on the named control", "never substitute": "all confounds are eliminated"},
        {"status": "equivalent", "write": "effect lies inside a prespecified meaningful margin under the named test", "never substitute": "identical"},
    ]
    mo.vstack([
        mo.md("## Falsification is a successful endpoint"),
        mo.Html(table(status_rows, (("status", "Status"), ("write", "Permitted report"), ("never substitute", "Overclaim")))),
    ])
    return


@app.cell
def review_loop(mo, table):
    review_rows = [
        {"review": "scientific/code", "must reproduce": "estimand, split, estimator, numerical gates, artifact identities", "blocker example": "toy result enters Raw/Hero claim"},
        {"review": "cold learner / pedagogy", "must reproduce": "claim boundary and strongest alternative without oral context", "blocker example": "undefined chess or hook term changes interpretation"},
        {"review": "runtime/accessibility", "must reproduce": "offline CPU snapshot, keyboard path, alt text, static narrative", "blocker example": "slider loads checkpoint or color is sole encoding"},
        {"review": "independent replication", "must reproduce": "direction/status under new groups and source choices", "blocker example": "same sample or source-selected feature reused as independent"},
    ]
    mo.vstack([
        mo.md("## Creator–critic loop for the capstone"),
        mo.Html(table(review_rows, (("review", "Reviewer"), ("must reproduce", "Review object"), ("blocker example", "Example blocker")))),
        mo.md("Every finding receives one of three recorded responses: **changed**, **deferred with scope removed**, or **rejected with evidence**. A hard scientific blocker cannot be deferred while retaining the affected claim."),
    ])
    return


@app.cell
def transfer_assessment(committed_form, mo, module):
    audit_transfer = mo.ui.text_area(
        label="Final unseen transfer",
        placeholder=(
            "A language-model paper selects a truthfulness feature on all data, steers it once, and reports no change. "
            "Rewrite the claim, data roles, controls, uncertainty, status, falsifier, and replication plan."
        ),
        full_width=True,
    )
    transfer_form = committed_form(
        mo,
        {"final_audit": audit_transfer},
        submit_label="Commit response and reveal capstone formative rubric",
        min_words=30,
    )
    mo.vstack([mo.md(f"## Final transfer assessment\n\n{module.transfer_assessment}"), transfer_form])
    return (transfer_form,)


@app.cell
def transfer_feedback(mo, transfer_form):
    mo.stop(transfer_form.value is None, mo.md("Submit a complete response before revealing."))
    response_words = len(transfer_form.value["final_audit"].split())
    mo.md(f"""
    **Mastery rubric (your response: {response_words} words):** split entities
    before feature selection; restrict the current result to exploratory
    observation/intervention on one example; call the null *unresolved* unless a
    powered equivalence margin was frozen; add simple, shuffle, untrained,
    matched-random, semantic-placebo, positive, sign/dose, density, and off-target
    controls as relevant; group uncertainty by entity; bind code/model/data;
    name a quantitative falsifier and an independent feature-selection
    replication. Full credit is independent of the hypothesis outcome.
    """)
    return


@app.cell
def close(bundle, callout, mo, resource_card):
    mo.vstack([
        mo.Html(callout("positive", "Course endpoint", "You now have an executable structure for turning a mechanistic story into a claim, estimator, intervention, controls, artifact record, and falsifier.")),
        mo.Html(callout("limit", "What the template cannot do", "No form can guarantee construct validity, adequate power, a good semantic placebo, or honest execution. Independent criticism and replication remain part of the method.")),
        mo.md("""
        ## Final retrieval

        1. Locate your central claim on all four axes.
        2. Name the weakest passed gate and strongest live alternative.
        3. State exactly what result would make you abandon or narrow the claim.

        **Sources:** the capstone integrates the primary literature registered
        in the [course source register](../SOURCES.md); Geiger et al. and
        Mueller et al. are the most direct
        sources for abstraction and circuit-evaluation criteria.
        """),
        mo.Html(resource_card(
            mode=bundle.mode.value,
            runtime="< 2 seconds; form assembly and structural checks only",
            device="CPU/browser; no model, GPU, or network",
            determinism="form-only; no RNG or model dtype; deterministic UTF-8 JSON serialization",
            sources=f"course evidence identity {bundle.integrity_sha256}",
            limitations="Self-audit is structural, not a substitute for scientific, ethical, statistical, or licensing review.",
        )),
    ])
    return


if __name__ == "__main__":
    app.run()
