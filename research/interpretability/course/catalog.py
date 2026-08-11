"""The frozen module catalog and prerequisite graph."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModuleSpec:
    index: int
    slug: str
    title: str
    track: str
    prerequisites: tuple[str, ...]
    mission: str
    objectives: tuple[str, ...]
    transfer_assessment: str
    source_ids: tuple[str, ...]

    @property
    def notebook_name(self) -> str:
        return f"{self.index:02d}_{self.slug}.py"


MODULES = (
    ModuleSpec(
        0,
        "evidence_and_experiments",
        "Evidence and experimental design",
        "core",
        (),
        "Turn a mechanistic story into a scoped estimand, controls, and falsifier.",
        (
            "Locate a claim on the operation × target × scope × endpoint matrix.",
            "Choose the independent unit and a grouping-aware estimator.",
            "Separate unresolved, falsified, control, and equivalence results.",
        ),
        "Audit an unseen language-model feature claim and repair its design.",
        ("doshi_velez_2017", "lipton_2016"),
    ),
    ModuleSpec(
        1,
        "chess_and_model_anatomy",
        "Chess and model anatomy",
        "core",
        ("evidence_and_experiments",),
        "Trace a legal chess position from LC0 planes through BT4, DFM, and JEPA.",
        (
            "Use the minimal chess notation and legality vocabulary.",
            "Trace every tensor shape in the BT4 forward pass.",
            "Identify valid hook, branch-replacement, and intervention boundaries.",
        ),
        "Given an unseen hook and tensor shape, identify what can be replaced.",
        ("searchless_chess_2024", "mcgrath_2022"),
    ),
    ModuleSpec(
        2,
        "behavior_and_localization",
        "Behavior and paired model localization",
        "core",
        ("chess_and_model_anatomy",),
        "Measure paired behavioral change before interpreting mechanism.",
        (
            "Compute distributional and decision-level paired metrics.",
            "Use game-cluster rather than position-level uncertainty.",
            "Interpret RR/RH/HR/HH hybrids as localization, not a unique circuit.",
        ),
        "Choose the strongest legal claim for an unseen hybrid lattice.",
        ("searchless_chess_2024",),
    ),
    ModuleSpec(
        3,
        "representation_geometry",
        "Representation geometry and dense model diffing",
        "core",
        ("behavior_and_localization",),
        "Learn metric invariances before assigning semantic meaning.",
        (
            "Derive cosine, CKA, Procrustes, and relative-L2 behavior.",
            "Predict metric changes under rotations, scales, permutations, and noise.",
            "Separate parameter, representation, and functional change.",
        ),
        "Select and defend a metric for an unseen transformation scenario.",
        ("kornblith_2019",),
    ),
    ModuleSpec(
        4,
        "probes_and_information",
        "Probes and usable information",
        "core",
        ("representation_geometry",),
        "Test decodability without confusing it with use.",
        (
            "Fit a grouped linear probe from first principles.",
            "Diagnose leakage, capacity, and selection effects.",
            "Use simple-input, label-shuffle, and untrained-feature controls.",
        ),
        "Repair an unseen high-accuracy probe study with the right controls.",
        ("hewitt_liang_2019", "voita_titov_2020", "karvonen_2024"),
    ),
    ModuleSpec(
        5,
        "attribution_and_intervention",
        "Attribution and intervention",
        "core",
        ("probes_and_information",),
        "Progress from sensitivity maps to bounded causal interventions.",
        (
            "Compare attribution methods and their sanity checks.",
            "Define clean/corrupt/patch estimands and endpoints.",
            "Design dose, norm, random, positive, and semantic-placebo controls.",
        ),
        "Design a placebo and dose protocol for an unseen steering direction.",
        ("jain_wallace_2019", "wiegreffe_pinter_2019", "meng_2022", "pins"),
    ),
    ModuleSpec(
        6,
        "lookahead_case_study",
        "Lookahead as an evidential case study",
        "core",
        ("attribution_and_intervention",),
        "Decide what evidence would justify a planning-like claim.",
        (
            "Separate decodability, DFM refinement, conditional JEPA prediction, and search.",
            "Design a predicted-action rollout with exact state targets.",
            "Construct non-planning alternatives consistent with partial evidence.",
        ),
        "Preregister an inference-matched rollout and its illegal-prefix rules.",
        ("jenner_2024",),
    ),
    ModuleSpec(
        7,
        "sparse_fundamentals",
        "Sparse representation fundamentals",
        "advanced",
        ("attribution_and_intervention",),
        "Understand sparse objectives before interpreting sparse features.",
        (
            "Distinguish SAE, transcoder, crosscoder, and branch replacement.",
            "Measure reconstruction, sparsity, dead features, and replacement fidelity.",
            "Navigate a reconstruction–sparsity frontier without semantic overclaiming.",
        ),
        "Classify an unseen sparse architecture and state its exact estimand.",
        ("monosemanticity_2023", "transcoders_2024"),
    ),
    ModuleSpec(
        8,
        "lorsa_and_sparse_diff",
        "LoRSA and sparse model diffing",
        "advanced",
        ("sparse_fundamentals",),
        "Compare Raw and Hero through a source-asymmetric sparse branch replacement.",
        (
            "Trace the layer-14 LoRSA hook ABI and sparse OV features.",
            "Separate fidelity, support transfer, semantics, and causal effects.",
            "Apply opportunity gates and held-out replication; design semantic placebos.",
        ),
        "Audit an unseen cross-model feature claim for source asymmetry.",
        ("crosscoders_2024", "grandmaster_trace_2026"),
    ),
    ModuleSpec(
        9,
        "circuits_and_abstraction",
        "Circuits and causal abstraction",
        "advanced",
        (
            "representation_geometry",
            "attribution_and_intervention",
            "lookahead_case_study",
            "lorsa_and_sparse_diff",
        ),
        "Turn component evidence into a minimal falsifiable algorithmic claim.",
        (
            "Test faithfulness, completeness, necessity, sufficiency, and redundancy.",
            "Formulate an interchange intervention and abstract causal model.",
            "Compare candidate circuits with smaller and matched-random alternatives.",
        ),
        "Reject or repair an unseen circuit claim that passes localization only.",
        ("geiger_2023", "acdc_2023", "mib_2025"),
    ),
    ModuleSpec(
        10,
        "capstone",
        "Capstone: defend or falsify a mechanism",
        "advanced",
        ("circuits_and_abstraction",),
        "Produce a reproducible scientific record rather than a persuasive story.",
        (
            "Preregister a scoped mechanism and frozen evidence gate.",
            "Bind artifacts, controls, uncertainty, alternatives, and falsifiers.",
            "Defend a conclusion or report a rigorous falsification.",
        ),
        "Submit an executable claim card, manifest, analysis, and replication plan.",
        ("geiger_2023", "mib_2025"),
    ),
)


def get_module(slug_or_index: str | int) -> ModuleSpec:
    for module in MODULES:
        if module.slug == slug_or_index or module.index == slug_or_index:
            return module
    raise KeyError(f"Unknown course module: {slug_or_index!r}")


def validate_catalog() -> None:
    slugs = {module.slug for module in MODULES}
    if len(slugs) != len(MODULES):
        raise ValueError("Duplicate module slug")
    if [module.index for module in MODULES] != list(range(len(MODULES))):
        raise ValueError("Module indices must be contiguous")
    seen: set[str] = set()
    for module in MODULES:
        missing = set(module.prerequisites) - seen
        if missing:
            raise ValueError(f"{module.slug} has forward/missing prerequisites: {missing}")
        if not module.objectives or not module.transfer_assessment:
            raise ValueError(f"{module.slug} has no assessable objective")
        seen.add(module.slug)


validate_catalog()
