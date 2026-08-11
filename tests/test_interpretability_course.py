from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest

from research.interpretability.course.build_snapshot import build_snapshot
from research.interpretability.course.catalog import MODULES, get_module, validate_catalog
from research.interpretability.course.data import (
    SNAPSHOT_RELATIVE_PATH,
    load_course_bundle,
)
from research.interpretability.course.evidence import (
    ClaimCard,
    Endpoint,
    EvidenceOperation,
    ResultStatus,
    TargetLevel,
    capstone_control_families,
    capstone_record_gates,
)
from research.interpretability.course.experiments import (
    toy_circuit_truth_table,
    toy_intervention,
    toy_lookahead_rollout,
    toy_probe_experiment,
    toy_representations,
    toy_sparse_frontier,
    transform_representation,
)
from research.interpretability.course.stats import (
    binary_accuracy,
    first_illegal_mask,
    fit_ridge_binary,
    js_divergence,
    linear_cka,
    masked_softmax,
    mean_row_cosine,
    paired_cluster_bootstrap,
    procrustes_similarity,
    ridge_binary_scores,
    support_jaccard,
    total_variation,
)
from research.interpretability.course.ui import (
    accessible_text,
    accessible_text_area,
    chessboard,
    committed_form,
    line_chart,
    resource_card,
    square_heatmap,
    table,
)
from research.interpretability.course.validate import (
    embedded_notebook_source,
    notebook_paths,
    validate_committed_site,
    validate_notebook_source,
)


ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / SNAPSHOT_RELATIVE_PATH


def _valid_card(**overrides: object) -> ClaimCard:
    values: dict[str, object] = {
        "operation": EvidenceOperation.OBSERVATION,
        "target": TargetLevel.REPRESENTATION,
        "scope": "paired development sample",
        "endpoint": Endpoint.REPRESENTATION,
        "estimand": "centered linear CKA",
        "unit": "paired row",
        "grouping": "game",
        "intervention": None,
        "controls": ("shuffled pairing",),
        "assumptions": ("rows are paired",),
        "uncertainty": "game bootstrap interval",
        "multiplicity": "one frozen metric",
        "allowed": "The sampled representations have the reported CKA.",
        "excluded": "The metric does not identify semantics.",
        "falsifier": "A source-verified recomputation outside tolerance.",
        "status": ResultStatus.DESCRIPTIVE,
    }
    values.update(overrides)
    return ClaimCard(**values)  # type: ignore[arg-type]


def _complete_capstone_gate_args(**overrides: str) -> dict[str, str]:
    values = {
        "project_title": "Causal pin feature audit",
        "hypothesis": "Removing the frozen pin direction will reduce legal target policy mass more than every matched control direction.",
        "strongest_alternative": "The direction merely tracks piece density and generic activation magnitude.",
        "artifact_identity": "a" * 64,
        "code_identity": "commit 0123456789abcdef0123456789abcdef01234567 clean",
        "environment_identity": "PyTorch 2.5 CUDA A100 GPU with BF16 dtype",
        "data_roles": "Selection uses development; test is separate and opened once after freezing.",
        "license_record": "Checkpoint license restricts redistribution; only derived metrics are published.",
        "seeds": "Seed 17 with deterministic kernels enabled.",
        "freeze_timestamp": "2026-08-11T18:30Z",
        "promotion": "Promote only if the paired interval exceeds every frozen control.",
        "rejection": "Reject if the paired interval includes zero under adequate power.",
        "equivalence": "not claimed",
        "falsifier": "A held-out matched intervention reverses the frozen directional effect.",
        "control_protocol": "Match token count norm dose sign and legal-policy opportunity before evaluation.",
        "replication": "Repeat on new game clusters with independently trained sparse coordinates.",
        "reviewer_objection": "Your semantic placebo was selected after seeing outcomes.",
        "reviewer_response": "Freeze the placebo from labels alone and rerun every held-out arm.",
    }
    values.update(overrides)
    return values


def test_catalog_is_a_valid_ordered_prerequisite_dag() -> None:
    validate_catalog()
    assert len(MODULES) == 11
    assert [module.index for module in MODULES] == list(range(11))
    assert [module.track for module in MODULES[:7]] == ["core"] * 7
    assert [module.track for module in MODULES[7:]] == ["advanced"] * 4
    assert get_module("lookahead_case_study").index == 6
    assert get_module(10).slug == "capstone"


def test_claim_card_accepts_representation_endpoint() -> None:
    card = _valid_card().validate()
    assert card.endpoint is Endpoint.REPRESENTATION
    assert card.problems() == ()


def test_capstone_gates_reject_substrings_and_near_empty_records() -> None:
    bad = capstone_record_gates(
        project_title="",
        hypothesis="",
        strongest_alternative="",
        artifact_identity="a" * 64,
        code_identity="abcdefg",
        environment_identity="",
        data_roles="contest selection",
        license_record="",
        seeds="",
        freeze_timestamp="",
        promotion="",
        rejection="",
        equivalence="",
        falsifier="",
        control_protocol="",
        replication="",
        reviewer_objection="",
        reviewer_response="",
    )
    assert not all(bad.values())
    assert not bad["data roles explicitly keep test separate"]
    assert not bad["code identity binds commit plus clean/dirty state or code SHA-256"]
    contradictory = capstone_record_gates(
        project_title="Causal pin feature audit",
        hypothesis="Removing the frozen pin direction will reduce legal target policy mass more than every matched control direction.",
        strongest_alternative="The direction merely tracks piece density and generic activation magnitude.",
        artifact_identity="a" * 64,
        code_identity="commit 0123456789abcdef0123456789abcdef01234567 clean",
        environment_identity="PyTorch 2.5 CUDA A100 GPU with BF16 dtype",
        data_roles="Test is separate from development but we use the test set to tune every run.",
        license_record="Checkpoint license restricts redistribution; only derived metrics are published.",
        seeds="Seed 17 with deterministic kernels enabled.",
        freeze_timestamp="2026-08-11T18:30Z",
        promotion="Promote only if the paired interval exceeds every frozen control.",
        rejection="Reject if the paired interval includes zero under adequate power.",
        equivalence="not claimed",
        falsifier="A held-out matched intervention reverses the frozen directional effect.",
        control_protocol="Match token count norm dose sign and legal-policy opportunity before evaluation.",
        replication="Repeat on new game clusters with independently trained sparse coordinates.",
        reviewer_objection="Your semantic placebo was selected after seeing outcomes.",
        reviewer_response="Freeze the placebo from labels alone and rerun every held-out arm.",
    )
    assert not contradictory["data roles explicitly keep test separate"]


def test_capstone_gates_accept_a_traceable_complete_record() -> None:
    good = capstone_record_gates(**_complete_capstone_gate_args())
    assert all(good.values()), good


@pytest.mark.parametrize(
    ("field", "gate"),
    [
        ("promotion", "promotion criterion is frozen and substantive"),
        ("rejection", "rejection criterion is frozen and substantive"),
        ("equivalence", "equivalence decision is explicit"),
        ("falsifier", "falsifier is substantive"),
        ("control_protocol", "control protocol is operational"),
        ("replication", "independent replication plan is substantive"),
        ("reviewer_objection", "cold-review objection is selected"),
        ("reviewer_response", "cold-review response records a repair"),
    ],
)
def test_capstone_gates_fail_closed_for_every_deliverable(
    field: str, gate: str
) -> None:
    values = _complete_capstone_gate_args(**{field: ""})
    assert not capstone_record_gates(**values)[gate]


def test_capstone_gate_rejects_impossible_calendar_timestamp() -> None:
    values = _complete_capstone_gate_args(freeze_timestamp="2026-02-30T18:30Z")
    gates = capstone_record_gates(**values)
    assert not gates["freeze timestamp is a real ISO-8601 UTC instant"]


def test_capstone_control_families_are_operation_aware() -> None:
    observation = capstone_control_families("observation")
    prediction = capstone_control_families("prediction")
    intervention = capstone_control_families("intervention")
    interchange = capstone_control_families("interchange")
    assert "label shuffle / shuffled pairing" in observation
    assert "untrained or independent-seed model" in prediction
    assert "semantic placebo" in intervention
    assert "smaller and larger circuit alternatives" in interchange
    assert "semantic placebo" not in observation
    with pytest.raises(ValueError, match="Unknown capstone evidence operation"):
        capstone_control_families("invalid")


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"allowed": "This representation causes policy change."}, "causal language"),
        (
            {
                "operation": EvidenceOperation.INTERVENTION,
                "intervention": "ablate direction",
                "controls": (),
            },
            "require controls",
        ),
        (
            {
                "target": TargetLevel.CIRCUIT,
                "allowed": "The sampled path has the reported score.",
            },
            "require interchange",
        ),
        (
            {
                "status": ResultStatus.EQUIVALENT,
                "uncertainty": "ordinary interval",
            },
            "prespecified margin",
        ),
    ],
)
def test_claim_card_rejects_overclaim(overrides: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _valid_card(**overrides).validate()


def test_probability_metrics_respect_legality_and_identity() -> None:
    logits = np.asarray([[2.0, 20.0, 0.0], [0.0, 1.0, 2.0]])
    legal = np.asarray([[True, False, True], [True, True, False]])
    probabilities = masked_softmax(logits, legal)
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert np.all(probabilities[~legal] == 0.0)
    assert np.allclose(js_divergence(probabilities, probabilities), 0.0)
    assert np.allclose(total_variation(probabilities, probabilities), 0.0)


def test_cluster_bootstrap_preserves_group_count() -> None:
    first = np.asarray([1.0, 1.2, 2.0, 2.2, 3.0])
    second = np.asarray([0.8, 1.0, 2.1, 2.0, 2.5])
    groups = np.asarray(["a", "a", "b", "b", "c"])
    observation = paired_cluster_bootstrap(
        first,
        second,
        groups,
        weighting="observation",
        resamples=500,
        seed=7,
    )
    equal_group = paired_cluster_bootstrap(
        first,
        second,
        groups,
        weighting="equal_group",
        resamples=500,
        seed=7,
    )
    assert observation.group_count == equal_group.group_count == 3
    assert observation.lower <= observation.estimate <= observation.upper
    assert equal_group.lower <= equal_group.estimate <= equal_group.upper
    assert observation.estimate == pytest.approx(np.mean(first - second))
    expected_group_mean = np.mean(
        [np.mean((first - second)[groups == group]) for group in ("a", "b", "c")]
    )
    assert equal_group.estimate == pytest.approx(expected_group_mean)
    assert observation.estimate != pytest.approx(equal_group.estimate)


def test_cluster_bootstrap_refuses_singleton_population_interval() -> None:
    with pytest.raises(ValueError, match="at least two independent groups"):
        paired_cluster_bootstrap(
            np.asarray([1.0, 2.0]),
            np.asarray([0.0, 0.0]),
            np.asarray(["only", "only"]),
            weighting="equal_group",
        )


def test_geometry_metrics_expose_rotation_gauge() -> None:
    first, _ = toy_representations()
    rotated = transform_representation(first, "rotate", strength=1.0)
    assert linear_cka(first, rotated) == pytest.approx(1.0, abs=1e-10)
    assert procrustes_similarity(first, rotated) == pytest.approx(1.0, abs=1e-10)
    assert mean_row_cosine(first, rotated) < 0.5


def test_probe_fixture_contains_a_group_leak() -> None:
    experiment = toy_probe_experiment()

    def accuracy(fit: np.ndarray, evaluation: np.ndarray) -> float:
        weights = fit_ridge_binary(
            experiment.features[fit],
            experiment.labels[fit],
            ridge=1.0,
        )
        return binary_accuracy(
            ridge_binary_scores(experiment.features[evaluation], weights),
            experiment.labels[evaluation],
        )

    row_accuracy = accuracy(experiment.row_fit, experiment.row_eval)
    group_accuracy = accuracy(experiment.group_fit, experiment.group_eval)
    assert row_accuracy > group_accuracy + 0.15
    assert not set(experiment.games[experiment.group_fit]) & set(
        experiment.games[experiment.group_eval]
    )
    assert not set(experiment.games[experiment.group_fit]) & set(
        experiment.games[experiment.group_selection]
    )
    assert not set(experiment.games[experiment.group_selection]) & set(
        experiment.games[experiment.group_eval]
    )
    for prefix in ("row", "group"):
        masks = [
            getattr(experiment, f"{prefix}_fit"),
            getattr(experiment, f"{prefix}_selection"),
            getattr(experiment, f"{prefix}_eval"),
        ]
        assert np.all(np.sum(np.column_stack(masks), axis=1) == 1)


def test_illegal_rollout_masks_the_entire_suffix() -> None:
    legal = np.asarray([[True, False, True, True], [True, True, True, True]])
    valid, first_illegal = first_illegal_mask(legal)
    assert valid.tolist() == [[True, False, False, False], [True, True, True, True]]
    assert first_illegal.tolist() == [1, -1]
    rollout = toy_lookahead_rollout(predicted_error_rate=0.35)
    teacher_mse = np.mean((rollout["teacher_prediction"] - rollout["targets"]) ** 2)
    predicted_mse = np.mean((rollout["predicted_prediction"] - rollout["targets"]) ** 2)
    assert predicted_mse > teacher_mse


def test_sparse_frontier_and_support_overlap_are_bounded() -> None:
    frontier = toy_sparse_frontier(np.arange(1, 9))
    assert np.all(np.diff(frontier["active_fraction"]) > 0.0)
    assert frontier["normalized_mse"][-1] < frontier["normalized_mse"][0]
    assert float(frontier["minimum_ground_truth_code"]) >= 0.0
    assert float(frontier["minimum_encoded_coefficient"]) >= 0.0
    assert support_jaccard([True, False, True], [True, True, False]) == pytest.approx(1 / 3)


def test_toy_intervention_has_distinct_ground_truth_controls() -> None:
    result = toy_intervention(np.asarray([0.0, 1.0]))
    assert result["positive_control"][1] == pytest.approx(1.0)
    assert 0.0 < result["candidate"][1] < result["positive_control"][1]
    assert result["semantic_placebo"][1] == pytest.approx(0.0, abs=1e-12)
    assert result["random_matched"][1] == pytest.approx(0.0, abs=1e-12)


def test_redundant_circuit_truth_table_is_exhaustive() -> None:
    rows = toy_circuit_truth_table()
    assert len(rows) == 8
    assert all(row["threat"] == row["redundant_copy"] for row in rows)
    assert sum(row["output"] for row in rows) == 1


def test_ui_charts_have_accessible_redundant_encodings() -> None:
    chart = line_chart(
        [
            {"name": "first", "points": [(0, 0.0), (1, 1.0)]},
            {"name": "second", "points": [(0, 1.0), (1, 0.0)]},
        ],
        x_label="layer",
        y_label="score",
        alt_text="Two crossing score curves.",
    )
    heatmap = square_heatmap(range(64), alt_text="Square values.", signed=False)
    assert 'role="img"' in chart and "stroke-dasharray" in chart
    assert "Two crossing score curves." in chart
    assert "x axis: layer" in chart and ">layer</text>" in chart
    assert 'aria-label="Square values."' in heatmap
    assert "every cell also prints" in heatmap


def test_tables_and_chessboards_expose_keyboard_and_semantic_cues() -> None:
    rendered_table = table(
        [{"name": "Raw", "score": 0.5}],
        (("name", "Model"), ("score", "Score")),
    )
    rendered_board = chessboard(
        "4k3/8/8/8/8/8/4R3/4K3 w - - 0 1",
        highlighted=("e2",),
    )
    assert 'role="region"' in rendered_table
    assert 'tabindex="0"' in rendered_table
    assert rendered_table.count('scope="col"') == 2
    assert "e2: white rook, highlighted" in rendered_board
    assert "ci-highlight-marker" in rendered_board
    assert "★" in rendered_board
    provenance = resource_card(
        mode="toy",
        runtime="instant",
        device="CPU",
        determinism="seed 7; NumPy float64",
        sources="constructed",
        limitations="teaching case",
    )
    assert "Seed / dtype" in provenance and "seed 7; NumPy float64" in provenance


def test_form_validation_fails_closed_and_text_controls_have_ax_fallbacks() -> None:
    class FakeForm:
        def form(self, **kwargs):
            self.options = kwargs
            return self

    class FakeUI:
        def dictionary(self, fields):
            self.fields = fields
            return FakeForm()

        def text(self, **kwargs):
            return kwargs

        def text_area(self, **kwargs):
            return kwargs

    class FakeMo:
        ui = FakeUI()

    form = committed_form(
        FakeMo(), {"answer": object()}, submit_label="Submit", min_words=2
    )
    validate = form.options["validate"]
    assert validate({}) == "Form field mismatch: answer."
    assert validate({"answer": "two words"}) is None
    assert validate({"answer": "two words", "extra": "bad"}) == "Form field mismatch: extra."

    text_control = accessible_text(FakeMo(), label="Project title")
    area_control = accessible_text_area(FakeMo(), label="Frozen hypothesis")
    assert text_control["placeholder"] == text_control["label"] == "Project title"
    assert area_control["placeholder"] == area_control["label"] == "Frozen hypothesis"


def test_committed_static_blog_editions_have_scope_and_provenance() -> None:
    site = ROOT / "curriculum/interpretability/site"
    for module, notebook in zip(MODULES, notebook_paths(ROOT), strict=True):
        export = site / f"{notebook.stem}.html"
        assert export.is_file()
        html = export.read_text(encoding="utf-8")
        assert module.title in html
        assert "SNAPSHOT" in html
        assert "ci-limit" in html
        assert "Resource / provenance" in html
        assert embedded_notebook_source(html) == notebook.read_text(encoding="utf-8")
        assert export.stat().st_size >= 50_000
    assert validate_committed_site(ROOT, notebook_paths(ROOT)) > 500_000


def test_snapshot_is_content_bound_compact_and_deterministic() -> None:
    committed = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    assert build_snapshot(ROOT) == committed
    bundle = load_course_bundle("snapshot", root=ROOT)
    assert bundle.integrity_sha256 == committed["integrity"]["sha256"]
    assert SNAPSHOT.stat().st_size < 1_000_000
    assert len(bundle.sources) == 18
    sparse = bundle.section("sparse")
    assert sparse["transcoder_position_count"] == 64
    assert sparse["transcoder_source_compatibility_gate"]["passed"] is True
    assert {row["model"] for row in sparse["transcoder_table"]} == {"Raw BT4", "Hero"}
    transcoder_row = next(
        row
        for row in bundle.section("circuits")["evidence_matrix"]
        if row["method"] == "Published L14 transcoder transfer"
    )
    assert transcoder_row["scope"] == "64 development positions"
    assert transcoder_row["source_run"] == "published_tc_l14_transfer_dev64_v1"
    raw_row, hero_row = sparse["transcoder_table"]
    if raw_row["model"] != "Raw BT4":
        raw_row, hero_row = hero_row, raw_row
    ratio = hero_row["normalized_mse"] / raw_row["normalized_mse"]
    assert f"{ratio:.3f}× Raw" in transcoder_row["finding"]
    identities = bundle.section("architecture")["model_identities"]
    assert identities["raw_asset_sha256"] == "61e43e98d2c4cb747498c6bc13a5bbb3b07e985d3f5d87f0351a4db855fa0651"
    assert identities["hero_state_sha256"] == "665ac91ad37e14a6c8810068700dba021e39e9c0fe145c8bf620882bb5a71692"
    refinement = bundle.section("lookahead")["refinement_sweep"]
    assert refinement["pair_count"] == 256
    assert refinement["pass_1_minus_pass_8"] == pytest.approx(0.0478515625)
    assert refinement["pass_1_score"] > refinement["pass_8_score"]
    assert refinement["jepa_used_at_inference"] is False
    assert all(
        Path(record["path"]).suffix in {".json", ".jsonl", ".md"}
        for record in bundle.sources
    )


def test_toy_mode_cannot_masquerade_as_raw_hero() -> None:
    toy = load_course_bundle("toy", root=ROOT)
    assert not toy.is_empirical
    assert toy.model_labels == ("Constructed A", "Constructed B")
    assert "Raw" not in json.dumps(toy.payload)
    architecture = toy.section("architecture")
    assert architecture["square_tokens"] == 64
    assert architecture["dfm_state_width"] == 4
    assert architecture["jepa_state_width"] == 8
    assert architecture["dfm_condition_on_current_jepa_state"] is False
    with pytest.raises(ValueError):
        load_course_bundle("auto", root=ROOT)


def test_snapshot_corruption_fails_closed(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\nname='fixture'\n", encoding="utf-8")
    (tmp_path / "research").mkdir()
    target = tmp_path / SNAPSHOT_RELATIVE_PATH
    target.parent.mkdir(parents=True)
    shutil.copyfile(SNAPSHOT, target)
    value = json.loads(target.read_text(encoding="utf-8"))
    value["payload"]["architecture"]["width"] += 1
    target.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="integrity mismatch"):
        load_course_bundle("snapshot", root=tmp_path)


def test_source_mode_reproduces_frozen_payload() -> None:
    snapshot = load_course_bundle("snapshot", root=ROOT)
    source = load_course_bundle("source", root=ROOT)
    assert source.payload == snapshot.payload
    assert source.integrity_sha256 == snapshot.integrity_sha256


def test_every_notebook_satisfies_static_course_contract() -> None:
    paths = notebook_paths(ROOT)
    for module, path in zip(MODULES, paths, strict=True):
        validate_notebook_source(path, expected_index=module.index)


def test_every_notebook_runs_in_all_evidence_modes() -> None:
    """Startup smoke: mode loading only; reactive branches are tested below."""

    bundles = {
        mode: load_course_bundle(mode, root=ROOT)
        for mode in ("toy", "snapshot", "source")
    }
    for index, path in enumerate(notebook_paths(ROOT)):
        spec = importlib.util.spec_from_file_location(f"course_mode_smoke_{index}", path)
        assert spec is not None and spec.loader is not None
        notebook_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(notebook_module)
        for mode, bundle in bundles.items():
            overrides: dict[str, object] = {
                "bundle": bundle,
                "selected_mode": mode,
            }
            if index == 1:
                # M01's loader also materializes the selected architecture mapping.
                overrides["architecture"] = bundle.section("architecture")
            outputs, definitions = notebook_module.app.run(defs=overrides)
            assert outputs
            assert definitions["selected_mode"] == mode
            assert definitions["bundle"].mode.value == mode


def test_live_reactive_branches_cover_button_number_form_stop_and_export() -> None:
    class Value:
        def __init__(self, value):
            self.value = value

    def load_notebook(index: int, name: str):
        path = notebook_paths(ROOT)[index]
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    # Run-button branch: unlike the startup smoke, this lets Module 00 enter
    # source mode through the same explicit verification control as a learner.
    module_00 = load_notebook(0, "course_branch_m00")
    _, source_defs = module_00.app.run(
        defs={
            "evidence_mode": Value("source"),
            "verify_source": Value(True),
        }
    )
    assert source_defs["selected_mode"] == "source"
    assert source_defs["bundle"].mode.value == "source"

    # Number-control + committed-form + mo.stop branch: execute a nondefault
    # saturation and dose through the attribution notebook.
    module_05 = load_notebook(5, "course_branch_m05")
    snapshot = load_course_bundle("snapshot", root=ROOT)
    _, intervention_defs = module_05.app.run(
        defs={
            "bundle": snapshot,
            "selected_mode": "snapshot",
            "saturation": Value(5.0),
            "attribution_method": Value("integrated_gradients"),
            "saturation_prediction": Value(
                "The local gradient can vanish even when features matter globally"
            ),
            "attribution_form": Value(
                {
                    "saturation": 5.0,
                    "method": "integrated_gradients",
                    "prediction": (
                        "The local gradient can vanish even when features "
                        "matter globally"
                    ),
                }
            ),
            "dose_span": Value(3.0),
            "dose_fraction": Value(-0.5),
        }
    )
    assert intervention_defs["selected_map"].shape == (64,)
    assert intervention_defs["inspected_dose"] == pytest.approx(-1.5)

    # Full capstone branch: a substantive record must pass every hard gate and
    # cross the score threshold before the stopped JSON-export cell executes.
    valid = {
        "project_title": "Causal pin feature audit",
        "hypothesis_family": "lorsa",
        "hypothesis": _complete_capstone_gate_args()["hypothesis"],
        "strongest_alternative": _complete_capstone_gate_args()[
            "strongest_alternative"
        ],
        "operation_choice": "intervention",
        "target_choice": "component",
        "endpoint_choice": "policy",
        "scope": "held-out paired positions from separate games in both frozen models",
        "estimand": "paired legal-policy total variation change relative to every matched control",
        "unit": "one paired board position from the frozen evaluation corpus",
        "grouping": "independent game cluster with no repeated opening overlap",
        "intervention": "ablate one frozen decoder direction at natural active tokens",
        "assumptions": "exact board pairing; opportunity labels frozen; intervention dose remains in density support",
        "uncertainty": "game-cluster bootstrap interval with Holm correction across the frozen family",
        "multiplicity": "Holm correction across every frozen feature and endpoint",
        "controls": [
            "identity / no-op",
            "matched random direction/subgraph",
            "semantic placebo",
            "positive control",
        ],
        "control_protocol": "Match token count activation norm dose sign and legal-policy opportunity across all evaluation arms before inspecting outcomes.",
        "promotion": "Promote only if the paired held-out interval exceeds every frozen control by the prespecified meaningful effect threshold.",
        "rejection": "Reject the directional claim if its paired held-out interval includes zero under the prespecified adequate-power rule.",
        "equivalence": "Do not claim equivalence unless the entire interval lies inside the prespecified practical margin.",
        "falsifier": "A held-out matched intervention reverses the frozen directional effect under all diagnostics.",
        "result_status_choice": "unresolved",
        "allowed": "The scoped held-out intervention has the reported legal-policy effect in both frozen model arms.",
        "excluded": "This result does not identify a native monosemantic feature or demonstrate internal search or planning.",
        "code_identity": _complete_capstone_gate_args()["code_identity"],
        "artifact_identity": "a" * 64,
        "environment_identity": _complete_capstone_gate_args()[
            "environment_identity"
        ],
        "data_roles": "Discovery and selection use development; test is separate and opened once after freezing every rule.",
        "license_record": _complete_capstone_gate_args()["license_record"],
        "seeds": _complete_capstone_gate_args()["seeds"],
        "freeze_timestamp": "2026-08-11T18:30Z",
        "replication": "Repeat on new game clusters with independently trained sparse coordinates and a frozen mapping.",
        "reviewer_objection": _complete_capstone_gate_args()[
            "reviewer_objection"
        ],
        "reviewer_response": "Freeze the placebo from labels alone and rerun every held-out arm before restoring the claim.",
        "audit_run": True,
    }
    module_10 = load_notebook(10, "course_branch_m10")
    base_overrides = {
        "bundle": snapshot,
        "selected_mode": "snapshot",
    }
    _, capstone_defs = module_10.app.run(
        defs=base_overrides | {key: Value(value) for key, value in valid.items()}
    )
    assert capstone_defs["audit_score"] >= 90
    assert all(capstone_defs["hard_gates"].values())
    assert json.loads(capstone_defs["serialized_record"])["replication"]

    # Observation and prediction records earn control credit from their own
    # valid control families; neither must select irrelevant causal controls.
    observational = dict(valid)
    observational.update(
        {
            "hypothesis_family": "model_diff",
            "hypothesis": "Hero will have higher centered representation similarity than Raw on the frozen paired evaluation sample.",
            "operation_choice": "observation",
            "target_choice": "representation",
            "endpoint_choice": "representation",
            "intervention": "none",
            "controls": [
                "identity / no-op",
                "simple-input or representation baseline",
                "label shuffle / shuffled pairing",
                "untrained or independent-seed model",
            ],
            "allowed": "The paired held-out representations have the reported similarity under the frozen metric.",
            "excluded": "This observation does not establish causal use or a unique internal algorithm.",
        }
    )
    _, observation_defs = module_10.app.run(
        defs=base_overrides
        | {key: Value(value) for key, value in observational.items()}
    )
    assert observation_defs["audit_score"] >= 90
    assert all(observation_defs["hard_gates"].values())
    assert "serialized_record" in observation_defs

    predictive = dict(valid)
    predictive.update(
        {
            "hypothesis_family": "probe_use",
            "hypothesis": "A grouped linear probe will achieve higher held-out accuracy than every frozen baseline and shuffled-label control.",
            "operation_choice": "prediction",
            "target_choice": "representation",
            "endpoint_choice": "probe",
            "intervention": "none",
            "controls": [
                "simple-input or representation baseline",
                "label shuffle / shuffled pairing",
                "untrained or independent-seed model",
                "positive control",
            ],
            "allowed": "The frozen grouped probe has the reported held-out predictive performance.",
            "excluded": "Decodability alone does not establish that the policy uses the probed information.",
        }
    )
    _, prediction_defs = module_10.app.run(
        defs=base_overrides | {key: Value(value) for key, value in predictive.items()}
    )
    assert prediction_defs["audit_score"] >= 90
    assert all(prediction_defs["hard_gates"].values())
    assert "serialized_record" in prediction_defs

    invalid = dict(valid)
    invalid["replication"] = ""
    _, blocked_defs = module_10.app.run(
        defs=base_overrides | {key: Value(value) for key, value in invalid.items()}
    )
    assert not blocked_defs["hard_gates"][
        "independent replication plan is substantive"
    ]
    assert "serialized_record" not in blocked_defs


def test_all_notebooks_pass_strict_marimo_check() -> None:
    command = [
        str(ROOT / ".venv/bin/marimo"),
        "check",
        "--strict",
        str(ROOT / "curriculum/interpretability/notebooks"),
    ]
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
    assert completed.returncode == 0, completed.stdout + completed.stderr
