"""Shared scientific and UI primitives for the interpretability course."""

from .activations import (
    ActivationGallery,
    load_activation_gallery,
    load_optional_activation_gallery,
)
from .catalog import MODULES, ModuleSpec, get_module
from .data import CourseBundle, EvidenceMode, load_course_bundle
from .evidence import ClaimCard, EvidenceOperation, Endpoint, TargetLevel

__all__ = [
    "MODULES",
    "ActivationGallery",
    "ClaimCard",
    "CourseBundle",
    "Endpoint",
    "EvidenceMode",
    "EvidenceOperation",
    "ModuleSpec",
    "TargetLevel",
    "get_module",
    "load_course_bundle",
    "load_activation_gallery",
    "load_optional_activation_gallery",
]
