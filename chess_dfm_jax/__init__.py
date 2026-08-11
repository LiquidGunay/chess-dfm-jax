"""Standalone LC0 BT4 training scaffold."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .paths import project_root, resolve_models_dir

if TYPE_CHECKING:
    from .nnx_bt4 import BT4Model

_JAX_EXPORTS = {
    "BT4Model",
    "bt4_forward",
    "bt4_forward_fp16",
    "bt4_forward_fp32",
    "make_bt4_model",
}


def __getattr__(name: str) -> Any:
    """Load legacy Flax exports only when a caller explicitly requests them."""

    if name in _JAX_EXPORTS:
        from . import nnx_bt4

        return getattr(nnx_bt4, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "BT4Model",
    "bt4_forward",
    "bt4_forward_fp16",
    "bt4_forward_fp32",
    "make_bt4_model",
    "project_root",
    "resolve_models_dir",
]
