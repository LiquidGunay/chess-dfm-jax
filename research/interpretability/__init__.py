"""Torch-native interpretability tools for raw BT4 and Hero comparisons.

The public model exports are loaded lazily so infrastructure-only modules (for
example, cloud cost guards) can be imported without importing Torch locally.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "ARM_IDS",
    "BT4ComparisonModels",
    "ComparisonPolicyOutput",
    "load_bt4_comparison_models",
]


def __getattr__(name: str) -> Any:
    if name not in __all__:
        raise AttributeError(name)
    from . import models

    return getattr(models, name)
