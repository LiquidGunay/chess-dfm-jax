"""Claim calibration primitives shared by every course notebook."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import re
from datetime import datetime


class EvidenceOperation(StrEnum):
    OBSERVATION = "observation"
    PREDICTION = "prediction"
    INTERVENTION = "intervention"
    INTERCHANGE = "interchange / causal abstraction"


class TargetLevel(StrEnum):
    BEHAVIOR = "behavior"
    ACTIVATION = "activation"
    REPRESENTATION = "representation"
    COMPONENT = "component"
    CIRCUIT = "path / circuit"


class Endpoint(StrEnum):
    REPRESENTATION = "representation metric"
    PROBE = "probe objective"
    LOGIT = "logit"
    LEGAL_ACTION = "legal action"
    POLICY = "policy distribution"
    VALUE = "value"
    ARENA = "arena result"


class ResultStatus(StrEnum):
    POSITIVE = "positive under the stated criterion"
    FALSIFIED = "falsified under the stated criterion"
    UNRESOLVED = "unresolved / failure to reject"
    NEGATIVE_CONTROL = "negative control"
    EQUIVALENT = "evidence of equivalence under a prespecified margin"
    DESCRIPTIVE = "descriptive only"


_CAUSAL_WORDS = re.compile(
    r"\b(cause[sd]?|causal(?:ly)?|mediate[sd]?|implements?|uses?|drives?|"
    r"affects?|changes?|determines?|responsible\s+for|necessary|sufficient|circuit)\b",
    flags=re.IGNORECASE,
)
_PLANNING_WORDS = re.compile(
    r"\b(plan(?:s|ning|ned)?|search(?:es|ed|ing)?|reason(?:s|ed|ing)?)\b",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class ClaimCard:
    operation: EvidenceOperation
    target: TargetLevel
    scope: str
    endpoint: Endpoint
    estimand: str
    unit: str
    grouping: str
    intervention: str | None
    controls: tuple[str, ...]
    assumptions: tuple[str, ...]
    uncertainty: str
    multiplicity: str
    allowed: str
    excluded: str
    falsifier: str
    status: ResultStatus = ResultStatus.DESCRIPTIVE

    def problems(self) -> tuple[str, ...]:
        problems: list[str] = []
        required = {
            "scope": self.scope,
            "estimand": self.estimand,
            "unit": self.unit,
            "grouping": self.grouping,
            "uncertainty": self.uncertainty,
            "multiplicity": self.multiplicity,
            "allowed": self.allowed,
            "excluded": self.excluded,
            "falsifier": self.falsifier,
        }
        problems.extend(f"Missing {name}" for name, value in required.items() if not value.strip())
        if self.operation in {EvidenceOperation.INTERVENTION, EvidenceOperation.INTERCHANGE}:
            if not self.intervention:
                problems.append("Interventional claims require a named intervention")
            if not self.controls:
                problems.append("Interventional claims require controls")
        elif self.intervention:
            problems.append("Observational/predictive claims cannot declare an intervention")
        if self.operation in {EvidenceOperation.OBSERVATION, EvidenceOperation.PREDICTION}:
            if _CAUSAL_WORDS.search(self.allowed):
                problems.append("Observational/predictive allowed conclusion uses causal language")
        if _PLANNING_WORDS.search(self.allowed) and "operational" not in self.allowed.lower():
            problems.append("Planning/search language must name its operational definition")
        if self.status is ResultStatus.EQUIVALENT and "margin" not in self.uncertainty.lower():
            problems.append("Equivalence claims require a prespecified margin")
        if self.target is TargetLevel.CIRCUIT and self.operation is not EvidenceOperation.INTERCHANGE:
            problems.append("Circuit claims require interchange/causal-abstraction evidence")
        return tuple(problems)

    def validate(self) -> "ClaimCard":
        problems = self.problems()
        if problems:
            raise ValueError("Invalid claim card: " + "; ".join(problems))
        return self

    def as_dict(self) -> dict[str, object]:
        return {
            "operation": self.operation.value,
            "target": self.target.value,
            "scope": self.scope,
            "endpoint": self.endpoint.value,
            "estimand": self.estimand,
            "unit": self.unit,
            "grouping": self.grouping,
            "intervention": self.intervention,
            "controls": list(self.controls),
            "assumptions": list(self.assumptions),
            "uncertainty": self.uncertainty,
            "multiplicity": self.multiplicity,
            "allowed": self.allowed,
            "excluded": self.excluded,
            "falsifier": self.falsifier,
            "status": self.status.value,
        }


def classify_language(text: str) -> dict[str, bool]:
    """Expose wording hazards for interactive claim-repair exercises."""

    return {
        "contains_causal_language": bool(_CAUSAL_WORDS.search(text)),
        "contains_planning_language": bool(_PLANNING_WORDS.search(text)),
        "states_scope": any(token in text.lower() for token in ("sample", "position", "model")),
        "states_uncertainty": any(
            token in text.lower() for token in ("interval", "uncertain", "margin", "bootstrap")
        ),
    }


_PLACEHOLDER_WORDS = re.compile(r"\b(todo|tbd|placeholder|fill\s+me|example)\b", re.I)
_ISO_UTC = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}(?::\d{2})?Z")
_DIRECTIONAL_WORDS = re.compile(
    r"\b(increase[sd]?|decrease[sd]?|reduce[sd]?|raise[sd]?|lower[sed]*|"
    r"higher|lower|greater|smaller|more|less|improve[sd]?|worsen[sed]*|"
    r"differ(?:s|ent)?|equivalent|noninferior)\b",
    re.I,
)


def _substantive(text: str, *, min_words: int) -> bool:
    return len(text.split()) >= min_words and not _PLACEHOLDER_WORDS.search(text)


def _valid_iso_utc(text: str) -> bool:
    stripped = text.strip()
    if not _ISO_UTC.fullmatch(stripped):
        return False
    try:
        datetime.fromisoformat(stripped.replace("Z", "+00:00"))
    except ValueError:
        return False
    return True


def capstone_record_gates(
    *,
    project_title: str,
    hypothesis: str,
    strongest_alternative: str,
    artifact_identity: str,
    code_identity: str,
    environment_identity: str,
    data_roles: str,
    license_record: str,
    seeds: str,
    freeze_timestamp: str,
    promotion: str,
    rejection: str,
    equivalence: str,
    falsifier: str,
    control_protocol: str,
    replication: str,
    reviewer_objection: str,
    reviewer_response: str,
) -> dict[str, bool]:
    """Validate anti-placeholder gates used by the capstone self-audit.

    These are structural checks, not scientific-quality judgments. Keeping
    them outside the notebook makes adversarial empty-form cases testable.
    """

    artifact = artifact_identity.strip()
    code = code_identity.strip().lower()
    environment = environment_identity.strip().lower()
    roles = data_roles.strip().lower()
    license_text = license_record.strip().lower()
    seed_text = seeds.strip().lower()
    equivalence_text = equivalence.strip().lower()
    role_tokens = set(re.findall(r"[a-z]+", roles))
    forbidden_test_use = bool(
        re.search(r"\b(?:use|uses|used|using)\s+(?:the\s+)?test\b", roles)
        or re.search(
            r"\b(?:tune|tuned|tuning|select|selected|selecting|iterate|iterated|iterating|peek|peeked|peeking)\s+(?:on|with|using)\s+(?:the\s+)?test\b",
            roles,
        )
        or re.search(
            r"\btest(?:\s+set)?\s+(?:to|for)\s+(?:tune|select|iterate|peek)\w*\b",
            roles,
        )
        or re.search(r"\btest\b[^.;]{0,48}\brepeated(?:ly)?\b", roles)
    )
    test_is_separate = (
        "test" in role_tokens
        and bool(role_tokens & {"discovery", "selection", "development", "evaluation"})
        and any(
            phrase in roles
            for phrase in (
                "held out",
                "held-out",
                "separate",
                "unopened",
                "open once",
                "opened once",
            )
        )
        and not forbidden_test_use
    )
    commit_with_tree_state = bool(re.search(r"\b[0-9a-f]{12,40}\b", code)) and bool(
        re.search(r"\b(clean|dirty)\b", code)
    )
    code_content_hash = bool(re.search(r"\b[0-9a-f]{64}\b", code)) and "sha" in code
    environment_has_device = bool(
        re.search(r"\b(cpu|gpu|cuda|mps|tpu|a100|a10g|l40s|t4|1660ti)\b", environment)
    )
    environment_has_dtype = bool(
        re.search(r"\b(dtype|fp(?:16|32|64)|bf16|bfloat16|float(?:16|32|64))\b", environment)
    )
    return {
        "project title is substantive": _substantive(project_title, min_words=3),
        "directional hypothesis is frozen and substantive": (
            _substantive(hypothesis, min_words=12)
            and bool(_DIRECTIONAL_WORDS.search(hypothesis))
        ),
        "strongest alternative is substantive": _substantive(
            strongest_alternative, min_words=8
        ),
        "artifact identity is a hexadecimal SHA-256": bool(
            re.fullmatch(r"[0-9a-fA-F]{64}", artifact)
        ),
        "code identity binds commit plus clean/dirty state or code SHA-256": (
            commit_with_tree_state or code_content_hash
        ),
        "data roles explicitly keep test separate": test_is_separate,
        "environment records device and dtype": (
            _substantive(environment_identity, min_words=4)
            and environment_has_device
            and environment_has_dtype
        ),
        "license and redistribution record is substantive": (
            _substantive(license_record, min_words=6)
            and bool(re.search(r"\b(licen[cs]e|permission|redistribut|restricted)\w*\b", license_text))
        ),
        "seeds and determinism are recorded": (
            _substantive(seeds, min_words=3)
            and bool(re.search(r"\b(seed|determin|nondetermin)\w*\b", seed_text))
        ),
        "freeze timestamp is a real ISO-8601 UTC instant": _valid_iso_utc(
            freeze_timestamp
        ),
        "promotion criterion is frozen and substantive": _substantive(
            promotion, min_words=8
        ),
        "rejection criterion is frozen and substantive": _substantive(
            rejection, min_words=8
        ),
        "equivalence decision is explicit": (
            equivalence_text in {"not claimed", "not applicable"}
            or (
                _substantive(equivalence, min_words=8)
                and "margin" in equivalence_text
            )
        ),
        "falsifier is substantive": _substantive(falsifier, min_words=8),
        "control protocol is operational": _substantive(
            control_protocol, min_words=8
        ),
        "independent replication plan is substantive": _substantive(
            replication, min_words=8
        ),
        "cold-review objection is selected": _substantive(
            reviewer_objection, min_words=4
        ),
        "cold-review response records a repair": _substantive(
            reviewer_response, min_words=8
        ),
    }


_CAPSTONE_CONTROL_FAMILIES = {
    "observation": frozenset(
        {
            "identity / no-op",
            "simple-input or representation baseline",
            "label shuffle / shuffled pairing",
            "untrained or independent-seed model",
        }
    ),
    "prediction": frozenset(
        {
            "simple-input or representation baseline",
            "label shuffle / shuffled pairing",
            "untrained or independent-seed model",
            "positive control",
        }
    ),
    "intervention": frozenset(
        {
            "identity / no-op",
            "matched random direction/subgraph",
            "semantic placebo",
            "positive control",
        }
    ),
    "interchange": frozenset(
        {
            "identity / no-op",
            "matched random direction/subgraph",
            "positive control",
            "smaller and larger circuit alternatives",
        }
    ),
}


def capstone_control_families(operation: str) -> frozenset[str]:
    """Return controls that earn structural credit for an evidence operation."""

    try:
        return _CAPSTONE_CONTROL_FAMILIES[operation]
    except KeyError as exc:
        raise ValueError(f"Unknown capstone evidence operation: {operation}") from exc
