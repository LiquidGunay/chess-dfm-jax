"""Explicit, fail-closed evidence modes for the course notebooks."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from .build_snapshot import SCHEMA_VERSION, build_snapshot


SNAPSHOT_RELATIVE_PATH = Path(
    "curriculum/interpretability/artifacts/raw_hero_snapshot_v1.json"
)


class EvidenceMode(StrEnum):
    TOY = "toy"
    SNAPSHOT = "snapshot"
    SOURCE = "source"


@dataclass(frozen=True)
class CourseBundle:
    mode: EvidenceMode
    snapshot_id: str
    evidence_scope: str
    payload: Mapping[str, Any]
    sources: tuple[Mapping[str, Any], ...]
    integrity_sha256: str
    model_labels: tuple[str, str]

    def section(self, name: str) -> Mapping[str, Any]:
        value = self.payload.get(name)
        if not isinstance(value, Mapping):
            raise KeyError(f"Course bundle has no mapping section {name!r}")
        return value

    @property
    def is_empirical(self) -> bool:
        return self.mode in {EvidenceMode.SNAPSHOT, EvidenceMode.SOURCE}


def find_repo_root(start: Path | None = None) -> Path:
    candidates = [Path.cwd() if start is None else start]
    try:
        candidates.append(Path(__file__).resolve())
    except NameError:
        pass
    for candidate in candidates:
        for parent in (candidate, *candidate.parents):
            if (parent / "pyproject.toml").is_file() and (parent / "research").is_dir():
                return parent
    raise FileNotFoundError("Could not locate chess-dfm-jax repository root")


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_and_validate_snapshot(root: Path) -> dict[str, Any]:
    path = root / SNAPSHOT_RELATIVE_PATH
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Course snapshot schema drift: {value.get('schema_version')!r}")
    integrity = value.get("integrity")
    if not isinstance(integrity, Mapping):
        raise ValueError("Course snapshot has no integrity record")
    unsigned = dict(value)
    unsigned.pop("integrity", None)
    observed = hashlib.sha256(_canonical_bytes(unsigned)).hexdigest()
    if integrity.get("algorithm") != "sha256-canonical-json-without-integrity":
        raise ValueError("Unknown course snapshot integrity algorithm")
    if integrity.get("sha256") != observed:
        raise ValueError("Course snapshot integrity mismatch")
    sources = value.get("sources")
    payload = value.get("payload")
    if not isinstance(sources, list) or not sources or not isinstance(payload, Mapping):
        raise ValueError("Course snapshot sources/payload are malformed")
    ids: set[str] = set()
    for record in sources:
        if not isinstance(record, Mapping):
            raise ValueError("Course snapshot source record is malformed")
        source_id = record.get("id")
        if not isinstance(source_id, str) or source_id in ids:
            raise ValueError("Course snapshot source IDs are missing or duplicated")
        ids.add(source_id)
        if not isinstance(record.get("sha256"), str) or len(record["sha256"]) != 64:
            raise ValueError(f"Invalid source digest for {source_id}")
        if record.get("redistribution") != "derived_metrics_or_minimal_fen_only":
            raise ValueError(f"Unreviewed redistribution status for {source_id}")
    return value


def _verify_source_records(root: Path, snapshot: Mapping[str, Any]) -> None:
    root = root.resolve()
    for record in snapshot["sources"]:
        relative = Path(record["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Unsafe source path: {relative}")
        path = (root / relative).resolve(strict=True)
        if not path.is_relative_to(root):
            raise ValueError(f"Source escapes repository: {path}")
        if path.stat().st_size != int(record["size_bytes"]):
            raise ValueError(f"Source size drift: {relative}")
        if _sha256_file(path) != record["sha256"]:
            raise ValueError(f"Source SHA-256 drift: {relative}")
        if path.suffix == ".json" and record.get("schema_version") is not None:
            value = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(value, Mapping) or value.get("schema_version") != record[
                "schema_version"
            ]:
                raise ValueError(f"Source schema drift: {relative}")


def _toy_payload() -> dict[str, Any]:
    """Constructed lesson configuration; never presented as Raw/Hero evidence."""

    return {
        "architecture": {
            "input_planes": 8,
            "square_tokens": 64,
            "width": 8,
            "layers": 3,
            "heads": 2,
            "head_width": 4,
            "action_vocab": 12,
            "jepa_state_width": 8,
            "dfm_state_width": 4,
            "dfm_state_source": "trunk",
            "dfm_condition_on_current_jepa_state": False,
            "dfm_jepa_fusion_mode": "none",
            "dfm_closed_loop_mode": "none",
            "constructed": True,
        },
        "design": {"seed": 7, "constructed": True},
        "behavior": {"seed": 11, "constructed": True},
        "geometry": {"seed": 13, "constructed": True},
        "probes": {"seed": 17, "constructed": True},
        "attribution": {"seed": 19, "constructed": True},
        "intervention": {"seed": 23, "constructed": True},
        "lookahead": {"seed": 29, "constructed": True},
        "sparse": {"seed": 31, "constructed": True},
        "lorsa": {"seed": 37, "constructed": True},
        "circuits": {"seed": 41, "constructed": True},
    }


def load_course_bundle(
    mode: EvidenceMode | str,
    *,
    root: Path | None = None,
) -> CourseBundle:
    """Load an explicitly selected evidence mode; there is deliberately no auto mode."""

    selected = EvidenceMode(mode)
    root = find_repo_root(root).resolve()
    if selected is EvidenceMode.TOY:
        payload = _toy_payload()
        digest = hashlib.sha256(_canonical_bytes(payload)).hexdigest()
        return CourseBundle(
            mode=selected,
            snapshot_id="constructed-course-toy-v1",
            evidence_scope="Constructed mechanism for teaching only; no Raw/Hero claim permitted.",
            payload=payload,
            sources=(),
            integrity_sha256=digest,
            model_labels=("Constructed A", "Constructed B"),
        )

    snapshot = _load_and_validate_snapshot(root)
    if selected is EvidenceMode.SOURCE:
        _verify_source_records(root, snapshot)
        rebuilt = build_snapshot(root)
        if rebuilt["payload"] != snapshot["payload"]:
            raise ValueError("Source extraction does not reproduce the frozen payload")
        if rebuilt["sources"] != snapshot["sources"]:
            raise ValueError("Source inventory does not reproduce the frozen snapshot")

    return CourseBundle(
        mode=selected,
        snapshot_id=str(snapshot["snapshot_id"]),
        evidence_scope=str(snapshot["evidence_scope"]),
        payload=snapshot["payload"],
        sources=tuple(snapshot["sources"]),
        integrity_sha256=str(snapshot["integrity"]["sha256"]),
        model_labels=("Raw BT4", "Hero"),
    )
