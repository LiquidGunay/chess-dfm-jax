"""Fail-closed import boundary for the checksum-pinned legacy checkpoint.

The historical ``state.npz`` stores its model and optimizer trees as scalar
NumPy object arrays.  NumPy therefore cannot read those members with
``allow_pickle=False``.  This module still opens the archive through NumPy with
pickle disabled, validates its exact envelope and step, and only then decodes
the two object members with a restricted unpickler.  The restricted decoder
can reconstruct NumPy arrays, but cannot import arbitrary Python globals.

This is intentionally a one-time compatibility boundary.  New research
checkpoints use the strict research checkpoint format in ``research.train``.
"""

from __future__ import annotations

import dataclasses
import hashlib
import hmac
import os
import pickle
import re
import stat
import zipfile
from pathlib import Path
from typing import Any, BinaryIO, Literal

import ml_dtypes
import numpy as np
import optax
from flax import nnx

from chess_dfm_jax.nnx_bt4 import TrainableParam
from research.prepare import require_within_workspace


LegacyInitMode = Literal["exact", "model-only"]

_PAYLOAD_KEYS = frozenset({"step", "model_trainable", "optimizer_state"})
_NPZ_MEMBERS = frozenset(f"{key}.npy" for key in _PAYLOAD_KEYS)
_SHA256_PATTERN = re.compile(r"^[0-9a-fA-F]{64}$")
_HASH_CHUNK_BYTES = 8 * 1024 * 1024


@dataclasses.dataclass(frozen=True)
class LegacyImportResult:
    """Auditable outcome of importing one verified legacy source checkpoint."""

    source_path: Path
    source_size_bytes: int
    source_sha256: str
    source_step: int
    init_mode: LegacyInitMode
    optimizer_restored: bool
    model_abi_sha256: str
    optimizer_abi_sha256: str


_ALLOWED_PICKLE_GLOBALS = {
    ("numpy", "ndarray"): np.ndarray,
    ("numpy", "dtype"): np.dtype,
    ("numpy._core.multiarray", "_reconstruct"): np._core.multiarray._reconstruct,
    ("numpy.core.multiarray", "_reconstruct"): np._core.multiarray._reconstruct,
    ("numpy._core.multiarray", "scalar"): np._core.multiarray.scalar,
    ("numpy.core.multiarray", "scalar"): np._core.multiarray.scalar,
    # The pinned BT4 source contains bfloat16 model leaves.  NumPy serializes
    # that dtype by its extension scalar class rather than by ``numpy.dtype``.
    ("ml_dtypes", "bfloat16"): ml_dtypes.bfloat16,
    # Muon/AdamW masking uses this immutable, zero-field Optax sentinel in
    # otherwise plain optimizer trees.
    ("optax.transforms._masking", "MaskedNode"): optax.MaskedNode,
}


class _RestrictedNumpyUnpickler(pickle.Unpickler):
    """Unpickler that can only rebuild ordinary NumPy arrays and scalars."""

    def find_class(self, module: str, name: str) -> Any:
        allowed = _ALLOWED_PICKLE_GLOBALS.get((module, name))
        if allowed is None:
            raise pickle.UnpicklingError(
                f"Forbidden global in legacy NumPy object member: {module}.{name}"
            )
        return allowed

    def persistent_load(self, pid: object) -> Any:
        raise pickle.UnpicklingError(
            f"Persistent pickle references are forbidden in legacy state: {pid!r}"
        )


def _validate_expected_file(
    *,
    expected_size_bytes: int,
    expected_sha256: str,
    expected_step: int,
) -> tuple[int, str, int]:
    if isinstance(expected_size_bytes, bool) or not isinstance(
        expected_size_bytes, (int, np.integer)
    ):
        raise TypeError("expected_size_bytes must be an integer")
    size = int(expected_size_bytes)
    if size < 1:
        raise ValueError(f"expected_size_bytes must be positive, got {size}")

    if not isinstance(expected_sha256, str) or not _SHA256_PATTERN.fullmatch(
        expected_sha256
    ):
        raise ValueError("expected_sha256 must be exactly 64 hexadecimal characters")
    digest = expected_sha256.lower()

    if isinstance(expected_step, bool) or not isinstance(expected_step, (int, np.integer)):
        raise TypeError("expected_step must be an integer")
    step = int(expected_step)
    if step < 0:
        raise ValueError(f"expected_step must be non-negative, got {step}")
    return size, digest, step


def _sha256_open_file(handle: BinaryIO) -> str:
    digest = hashlib.sha256()
    handle.seek(0)
    while chunk := handle.read(_HASH_CHUNK_BYTES):
        digest.update(chunk)
    return digest.hexdigest()


def _validate_step(value: Any, *, expected_step: int) -> np.ndarray:
    step = np.asarray(value)
    if step.shape != () or not np.issubdtype(step.dtype, np.integer):
        raise ValueError(
            "Legacy optimizer step must be an integer scalar, "
            f"got shape={step.shape}, dtype={step.dtype}"
        )
    observed = int(step)
    if observed < 0:
        raise ValueError(f"Legacy optimizer step must be non-negative, got {observed}")
    if observed != expected_step:
        raise ValueError(
            f"Legacy optimizer step mismatch: expected {expected_step}, found {observed}"
        )
    return np.asarray(observed, dtype=np.int64)


def _validate_payload_envelope(payload: dict[str, Any], *, expected_step: int) -> int:
    if type(payload) is not dict:
        raise TypeError(f"Legacy state payload must be a plain dict, got {type(payload).__name__}")
    observed_keys = set(payload)
    if observed_keys != _PAYLOAD_KEYS or len(payload) != len(_PAYLOAD_KEYS):
        missing = sorted(_PAYLOAD_KEYS - observed_keys)
        extra = sorted(observed_keys - _PAYLOAD_KEYS)
        raise ValueError(f"Legacy state payload keys differ: missing={missing}, extra={extra}")
    return int(_validate_step(payload["step"], expected_step=expected_step))


def _validate_plain_state_tree(value: Any, *, label: str, path: str = "<root>") -> None:
    if type(value) is optax.MaskedNode:
        return
    if type(value) is dict:
        for key, child in value.items():
            if type(key) not in {bool, int, str}:
                raise TypeError(
                    f"{label} state key at {path} must be bool, int, or str; "
                    f"got {type(key).__name__}"
                )
            _validate_plain_state_tree(child, label=label, path=f"{path}[{key!r}]")
        return
    if type(value) is list or type(value) is tuple:
        for index, child in enumerate(value):
            _validate_plain_state_tree(child, label=label, path=f"{path}[{index}]")
        return

    array = np.asarray(value)
    if array.dtype.hasobject:
        raise TypeError(f"{label} state contains an object-valued leaf at {path}")


def _read_object_npy_member(archive: zipfile.ZipFile, member_name: str) -> dict[str, Any]:
    with archive.open(member_name, "r") as member:
        version = np.lib.format.read_magic(member)
        if version != (1, 0):
            raise ValueError(
                f"Legacy object member {member_name} must use NPY v1.0, found {version}"
            )
        shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(member)
        if shape != () or fortran_order or dtype != np.dtype(object):
            raise ValueError(
                f"Legacy object member {member_name} has invalid header: "
                f"shape={shape}, fortran_order={fortran_order}, dtype={dtype}"
            )
        try:
            wrapper = _RestrictedNumpyUnpickler(
                member,
                fix_imports=False,
                encoding="ASCII",
                errors="strict",
            ).load()
        except (AttributeError, EOFError, ImportError, IndexError, pickle.PickleError) as exc:
            raise ValueError(f"Failed restricted decode of legacy member {member_name}") from exc

    if (
        type(wrapper) is not np.ndarray
        or wrapper.shape != ()
        or wrapper.dtype != np.dtype(object)
    ):
        raise ValueError(
            f"Legacy object member {member_name} did not decode to a scalar object array"
        )
    tree = wrapper.item()
    if type(tree) is not dict:
        raise TypeError(
            f"Legacy object member {member_name} must contain a plain dict, "
            f"got {type(tree).__name__}"
        )
    return tree


def load_verified_legacy_payload(
    state_npz: str | os.PathLike[str],
    *,
    expected_size_bytes: int,
    expected_sha256: str,
    expected_step: int,
) -> dict[str, Any]:
    """Load a legacy payload only after file identity and envelope validation.

    The size and SHA-256 are checked on the same open file descriptor later
    passed to ``np.load(..., allow_pickle=False)``.  The object members are not
    accessed through NumPy's general pickle loader.
    """

    source_path = require_within_workspace(state_npz)
    expected_size, expected_digest, expected_source_step = _validate_expected_file(
        expected_size_bytes=expected_size_bytes,
        expected_sha256=expected_sha256,
        expected_step=expected_step,
    )

    with source_path.open("rb") as handle:
        source_stat = os.fstat(handle.fileno())
        if not stat.S_ISREG(source_stat.st_mode):
            raise ValueError(f"Legacy checkpoint is not a regular file: {source_path}")
        if source_stat.st_size != expected_size:
            raise ValueError(
                "Legacy checkpoint size mismatch: "
                f"expected {expected_size}, found {source_stat.st_size}"
            )
        observed_digest = _sha256_open_file(handle)
        if not hmac.compare_digest(observed_digest, expected_digest):
            raise ValueError(
                "Legacy checkpoint SHA-256 mismatch: "
                f"expected {expected_digest}, found {observed_digest}"
            )

        handle.seek(0)
        loaded = np.load(handle, allow_pickle=False)
        if not isinstance(loaded, np.lib.npyio.NpzFile):
            raise ValueError(f"Legacy checkpoint is not an NPZ archive: {source_path}")
        try:
            archive_keys = list(loaded.files)
            observed_keys = set(archive_keys)
            if observed_keys != _PAYLOAD_KEYS or len(archive_keys) != len(_PAYLOAD_KEYS):
                missing = sorted(_PAYLOAD_KEYS - observed_keys)
                extra = sorted(observed_keys - _PAYLOAD_KEYS)
                raise ValueError(
                    f"Legacy state payload keys differ: missing={missing}, extra={extra}"
                )
            step = _validate_step(
                loaded["step"],
                expected_step=expected_source_step,
            )
        finally:
            loaded.close()

        handle.seek(0)
        with zipfile.ZipFile(handle, mode="r") as archive:
            member_names = [info.filename for info in archive.infolist()]
            observed_members = set(member_names)
            if (
                observed_members != _NPZ_MEMBERS
                or len(member_names) != len(_NPZ_MEMBERS)
            ):
                missing = sorted(_NPZ_MEMBERS - observed_members)
                extra = sorted(observed_members - _NPZ_MEMBERS)
                raise ValueError(
                    f"Legacy NPZ members differ: missing={missing}, extra={extra}"
                )
            model_trainable = _read_object_npy_member(
                archive,
                "model_trainable.npy",
            )
            optimizer_state = _read_object_npy_member(
                archive,
                "optimizer_state.npy",
            )

    _validate_plain_state_tree(model_trainable, label="model")
    _validate_plain_state_tree(optimizer_state, label="optimizer")
    payload = {
        "step": step,
        "model_trainable": model_trainable,
        "optimizer_state": optimizer_state,
    }
    _validate_payload_envelope(payload, expected_step=expected_source_step)
    return payload


def restore_legacy_payload(
    payload: dict[str, Any],
    *,
    model: nnx.Module,
    optimizer: nnx.Optimizer,
    expected_step: int,
    init_mode: LegacyInitMode,
) -> int:
    """Preflight both typed ABIs, then restore according to ``init_mode``."""

    # This local import lets ``research.train`` use the importer without a
    # module-import cycle while keeping the checkpoint ABI implementation in
    # its single canonical location.
    from research.train import (
        assert_research_state_compatible,
        strict_restore_research_payload,
    )

    if init_mode not in {"exact", "model-only"}:
        raise ValueError(f"Unknown legacy init mode: {init_mode!r}")
    source_step = _validate_payload_envelope(payload, expected_step=expected_step)
    _validate_plain_state_tree(payload["model_trainable"], label="model")
    _validate_plain_state_tree(payload["optimizer_state"], label="optimizer")

    model_state = nnx.state(model, TrainableParam)
    optimizer_state = nnx.state(optimizer.opt_state)
    model_current = dict(nnx.to_pure_dict(model_state))
    optimizer_current = dict(nnx.to_pure_dict(optimizer_state))

    # Both checks intentionally happen before either target tree is changed.
    assert_research_state_compatible(
        model_current,
        payload["model_trainable"],
        label="model",
    )
    assert_research_state_compatible(
        optimizer_current,
        payload["optimizer_state"],
        label="optimizer",
    )

    if init_mode == "exact":
        return strict_restore_research_payload(payload, model, optimizer)

    # Model-only initialization deliberately ignores the already-validated
    # source optimizer values and source step.
    nnx.replace_by_pure_dict(model_state, payload["model_trainable"])
    nnx.update(model, model_state)
    return source_step


def import_legacy_checkpoint(
    state_npz: str | os.PathLike[str],
    *,
    model: nnx.Module,
    optimizer: nnx.Optimizer,
    expected_size_bytes: int,
    expected_sha256: str,
    expected_step: int,
    init_mode: LegacyInitMode,
) -> LegacyImportResult:
    """Verify, decode, preflight, and restore one historical source checkpoint."""

    from research.train import research_state_abi

    source_path = require_within_workspace(state_npz)
    expected_size, expected_digest, expected_source_step = _validate_expected_file(
        expected_size_bytes=expected_size_bytes,
        expected_sha256=expected_sha256,
        expected_step=expected_step,
    )
    payload = load_verified_legacy_payload(
        source_path,
        expected_size_bytes=expected_size,
        expected_sha256=expected_digest,
        expected_step=expected_source_step,
    )
    source_step = restore_legacy_payload(
        payload,
        model=model,
        optimizer=optimizer,
        expected_step=expected_source_step,
        init_mode=init_mode,
    )
    return LegacyImportResult(
        source_path=source_path,
        source_size_bytes=expected_size,
        source_sha256=expected_digest,
        source_step=source_step,
        init_mode=init_mode,
        optimizer_restored=init_mode == "exact",
        model_abi_sha256=research_state_abi(payload["model_trainable"])["sha256"],
        optimizer_abi_sha256=research_state_abi(payload["optimizer_state"])["sha256"],
    )


__all__ = [
    "LegacyImportResult",
    "LegacyInitMode",
    "import_legacy_checkpoint",
    "load_verified_legacy_payload",
    "restore_legacy_payload",
]
