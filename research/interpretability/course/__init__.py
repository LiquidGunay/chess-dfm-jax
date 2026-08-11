"""Shared scientific and UI primitives for the interpretability course."""

from .catalog import MODULES, ModuleSpec, get_module
from .data import CourseBundle, EvidenceMode, load_course_bundle
from .evidence import ClaimCard, EvidenceOperation, Endpoint, TargetLevel

__all__ = [
    "MODULES",
    "ClaimCard",
    "CourseBundle",
    "Endpoint",
    "EvidenceMode",
    "EvidenceOperation",
    "ModuleSpec",
    "TargetLevel",
    "get_module",
    "load_course_bundle",
]
