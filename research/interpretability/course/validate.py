"""Offline release gates for the chess interpretability course."""

from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from typing import Iterable

from .activations import (
    describe_activation_gallery,
    load_optional_activation_gallery,
)
from .build_snapshot import SOURCE_DATA_COMMIT, build_snapshot
from .catalog import MODULES, validate_catalog
from .data import SNAPSHOT_RELATIVE_PATH, find_repo_root, load_course_bundle


FORBIDDEN_NOTEBOOK_IMPORTS = {
    "aiohttp",
    "boto3",
    "gcsfs",
    "httpx",
    "jax",
    "modal",
    "requests",
    "socket",
    "torch",
    "urllib",
}
FORBIDDEN_CALL_NAMES = {
    "from_pretrained",
    "load_state_dict",
    "snapshot_download",
    "urlopen",
}
REQUIRED_SOURCE_MARKERS = (
    "load_course_bundle",
    "mode_badge",
    "module_header",
    "resource_card",
    "ClaimCard",
    "mo.ui.run_button",
    "mo.stop",
    "module.transfer_assessment",
    "Retrieval",
    "**Source",
    "def prerequisite",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


_VTT_TIMING = re.compile(
    r"^(\d{2}):(\d{2})\.(\d{3}) --> "
    r"(\d{2}):(\d{2})\.(\d{3})$",
    flags=re.MULTILINE,
)


def _vtt_seconds(minutes: str, seconds: str, milliseconds: str) -> float:
    return int(minutes) * 60 + int(seconds) + int(milliseconds) / 1000


def _validate_vtt_timing(text: str, *, duration: float, animation_id: str) -> None:
    timings = _VTT_TIMING.findall(text)
    if not timings:
        raise ValueError(f"Animation captions have no timed cues: {animation_id}")
    previous_end = 0.0
    for fields in timings:
        start = _vtt_seconds(*fields[:3])
        end = _vtt_seconds(*fields[3:])
        if start < previous_end - 1e-9 or end <= start:
            raise ValueError(f"Animation captions overlap or reverse: {animation_id}")
        if start - previous_end > 0.25:
            raise ValueError(f"Animation captions contain an uncovered gap: {animation_id}")
        previous_end = end
    if previous_end > duration + 0.05:
        raise ValueError(f"Animation captions exceed video duration: {animation_id}")
    if previous_end < duration - 0.25:
        raise ValueError(f"Animation captions end before the video: {animation_id}")


@dataclass(frozen=True)
class ValidationReport:
    notebook_count: int
    source_count: int
    media_count: int
    activation_position_count: int
    activation_bytes: int
    activation_materialized: bool
    activation_sources_verified: bool
    snapshot_sha256: str
    exported_bytes: int
    source_verified: bool


def notebook_paths(root: Path) -> tuple[Path, ...]:
    notebook_dir = root / "curriculum/interpretability/notebooks"
    observed = tuple(sorted(notebook_dir.glob("*.py")))
    expected_names = tuple(module.notebook_name for module in MODULES)
    observed_names = tuple(path.name for path in observed)
    if observed_names != expected_names:
        raise ValueError(
            f"Notebook catalog drift: expected {expected_names}, observed {observed_names}"
        )
    return observed


def _import_roots(tree: ast.AST) -> set[str]:
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".", 1)[0])
    return roots


def _call_name(node: ast.Call) -> str | None:
    function = node.func
    if isinstance(function, ast.Name):
        return function.id
    if isinstance(function, ast.Attribute):
        return function.attr
    return None


def validate_notebook_source(path: Path, *, expected_index: int) -> None:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    forbidden_imports = _import_roots(tree) & FORBIDDEN_NOTEBOOK_IMPORTS
    if forbidden_imports:
        raise ValueError(f"{path.name} imports forbidden runtime modules: {forbidden_imports}")
    forbidden_calls = {
        name
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        if (name := _call_name(node)) in FORBIDDEN_CALL_NAMES
    }
    if forbidden_calls:
        raise ValueError(f"{path.name} contains forbidden loading/network calls: {forbidden_calls}")
    missing = tuple(marker for marker in REQUIRED_SOURCE_MARKERS if marker.lower() not in source.lower())
    if missing:
        raise ValueError(f"{path.name} misses course-contract markers: {missing}")
    if "_unparsable_cell" in source:
        raise ValueError(f"{path.name} contains an unparsable marimo cell")
    if f"get_module({expected_index})" not in source:
        raise ValueError(f"{path.name} does not bind catalog module {expected_index}")
    if 'value="Snapshot' not in source:
        raise ValueError(f"{path.name} does not default visibly to snapshot mode")
    if '"toy"' not in source or '"source"' not in source:
        raise ValueError(f"{path.name} does not expose all explicit evidence modes")
    if "http://" in source or "https://" in source:
        raise ValueError(f"{path.name} embeds a live URL; sources belong in SOURCES.md")
    if "mo.ui.slider" in source:
        raise ValueError(
            f"{path.name} uses a slider; use a labeled bounded number control so "
            "the value control has an accessible name"
        )


def validate_snapshot(root: Path, *, verify_source: bool) -> tuple[str, int]:
    toy = load_course_bundle("toy", root=root)
    if toy.is_empirical or toy.model_labels != ("Constructed A", "Constructed B"):
        raise ValueError("Toy evidence boundary is malformed")
    snapshot = load_course_bundle("snapshot", root=root)
    if not snapshot.is_empirical or snapshot.model_labels != ("Raw BT4", "Hero"):
        raise ValueError("Snapshot empirical labels are malformed")
    if set(snapshot.payload) != {
        "activation_gallery",
        "architecture",
        "attribution",
        "behavior",
        "circuits",
        "design",
        "geometry",
        "intervention",
        "lenses",
        "lookahead",
        "lorsa",
        "probes",
        "sparse",
    }:
        raise ValueError("Snapshot payload section drift")
    state_interface_keys = {
        "jepa_state_width",
        "dfm_state_width",
        "dfm_state_source",
        "dfm_condition_on_current_jepa_state",
        "dfm_jepa_fusion_mode",
        "dfm_closed_loop_mode",
    }
    for bundle in (toy, snapshot):
        missing_interfaces = state_interface_keys - set(bundle.section("architecture"))
        if missing_interfaces:
            raise ValueError(
                f"{bundle.mode.value} architecture misses state interfaces: "
                f"{sorted(missing_interfaces)}"
            )

    snapshot_path = root / SNAPSHOT_RELATIVE_PATH
    if snapshot_path.stat().st_size >= 1_000_000:
        raise ValueError("Course snapshot exceeds the compact-artifact budget")
    value = json.loads(snapshot_path.read_text(encoding="utf-8"))
    if value.get("source_repository_commit") != SOURCE_DATA_COMMIT:
        raise ValueError("Snapshot source-commit identity drift")
    allowed_redistribution = {
        "derived_metrics_or_minimal_fen_only",
        "internal_derived_tensor_release_review_required",
    }
    for record in value["sources"]:
        if Path(record["path"]).suffix not in {".json", ".jsonl", ".md", ".npz"}:
            raise ValueError(f"Snapshot references a non-derived source: {record['path']}")
        if record["redistribution"] not in allowed_redistribution:
            raise ValueError(f"Unreviewed redistribution status: {record['id']}")
        if Path(record["path"]).suffix == ".npz" and record["redistribution"] != (
            "internal_derived_tensor_release_review_required"
        ):
            raise ValueError("Tensor artifacts must retain the explicit release-review gate")
    if build_snapshot(root) != value:
        raise ValueError("Committed snapshot is not a deterministic source extraction")
    if verify_source:
        source = load_course_bundle("source", root=root)
        if source.payload != snapshot.payload or source.integrity_sha256 != snapshot.integrity_sha256:
            raise ValueError("Source mode does not reproduce snapshot mode")
        for record in value["sources"]:
            # The activation gallery is a new, content-bound derivative of the
            # historical sources. Its own manifest and strict loader verify the
            # current checkpoint/corpus/probe/code hashes below; it is deliberately
            # not asserted to have existed at SOURCE_DATA_COMMIT.
            if record["id"] in {
                "activation_gallery_manifest",
                "activation_gallery_tensors",
            }:
                continue
            completed = subprocess.run(
                ("git", "show", f"{SOURCE_DATA_COMMIT}:{record['path']}"),
                cwd=root,
                check=False,
                capture_output=True,
            )
            if completed.returncode:
                raise ValueError(
                    f"Declared source commit lacks {record['path']}: "
                    f"{completed.stderr.decode(errors='replace').strip()}"
                )
            committed_sha = hashlib.sha256(completed.stdout).hexdigest()
            if committed_sha != record["sha256"]:
                raise ValueError(
                    f"Source {record['path']} does not match declared commit "
                    f"{SOURCE_DATA_COMMIT}"
                )
    return snapshot.integrity_sha256, len(snapshot.sources)


def validate_activation_gallery(
    root: Path,
    *,
    verify_source: bool,
) -> tuple[int, int, bool]:
    """Validate the real artifact, with or without its license-gated local NPZ."""

    root = root.resolve()
    descriptor = describe_activation_gallery(root)
    manifest = descriptor.manifest
    gallery = load_optional_activation_gallery(root, verify_sources=verify_source)
    materialized = gallery is not None
    if materialized != descriptor.is_materialized:
        raise ValueError("Activation-gallery materialization state changed during validation")
    data_record = manifest["data"]
    position_count = int(manifest["selection"]["position_count"])
    data_bytes = int(data_record["size_bytes"])
    if position_count != 8:
        raise ValueError("Activation gallery must retain the frozen eight-position sample")
    selection = manifest["selection"]
    position_ids = selection.get("position_ids", ())
    if (
        selection.get("split") != "development"
        or len(position_ids) != position_count
        or len(set(position_ids)) != position_count
    ):
        raise ValueError("Activation-gallery development selection drift")
    if not re.fullmatch(r"[0-9a-f]{64}", str(data_record.get("sha256", ""))):
        raise ValueError("Activation-gallery data identity is malformed")
    expected_shape = (8, 15, 64, 1024)
    expected_products = {
        "raw_resid_post_f16": (expected_shape, "float16"),
        "hero_resid_post_f16": (expected_shape, "float16"),
        "lens_logits_f16": ((4, 8, 15, 1858), "float16"),
        "raw_probe_legal_destination_f16": ((8, 15, 64), "float16"),
        "hero_probe_legal_destination_f16": ((8, 15, 64), "float16"),
        "raw_probe_opponent_attack_count_f16": ((8, 15, 64), "float16"),
        "hero_probe_opponent_attack_count_f16": ((8, 15, 64), "float16"),
        "raw_probe_piece_code_f16": ((8, 15, 64, 13), "float16"),
        "hero_probe_piece_code_f16": ((8, 15, 64, 13), "float16"),
        "raw_target_gradient_square_norm_f32": ((8, 3, 64), "float32"),
        "hero_target_gradient_square_norm_f32": ((8, 3, 64), "float32"),
        "raw_target_gradient_x_state_f32": ((8, 3, 64), "float32"),
        "hero_target_gradient_x_state_f32": ((8, 3, 64), "float32"),
        "raw_target_gradient_rms_f32": ((8, 3), "float32"),
        "hero_target_gradient_rms_f32": ((8, 3), "float32"),
        "raw_centered_singular_values_f32": ((3, 512), "float32"),
        "hero_centered_singular_values_f32": ((3, 512), "float32"),
    }
    inventory = data_record.get("arrays", {})
    for name, (shape, dtype) in expected_products.items():
        record = inventory.get(name, {})
        if record.get("shape") != list(shape) or record.get("dtype") != dtype:
            raise ValueError(f"Activation-gallery manifest ABI drift for {name}")
        if gallery is not None and gallery.arrays[name].shape != shape:
            raise ValueError(f"Activation-gallery semantic shape drift for {name}")
    if gallery is not None:
        if tuple(gallery.arrays["arm_ids"].astype(str)) != (
            "RR",
            "HR",
            "RH",
            "HH",
        ):
            raise ValueError("Activation-gallery policy-arm identity drift")
        if not bool((gallery.arrays["split_u8"] == 1).all()):
            raise ValueError("Activation gallery contains a non-development position")
        if tuple(int(value) for value in gallery.arrays["gradient_layer_indices_i16"]) != (
            0,
            7,
            14,
        ):
            raise ValueError("Activation-gallery VJP layer contract drift")
        if tuple(int(value) for value in gallery.arrays["spectrum_layer_indices_i16"]) != (
            0,
            7,
            14,
        ):
            raise ValueError("Activation-gallery spectrum layer contract drift")
    if manifest["representation"].get("boundary") != "resid_post_after_ln":
        raise ValueError("Activation-gallery representation boundary drift")
    if manifest["target_gradient"].get("layers") != [0, 7, 14]:
        raise ValueError("Activation-gallery VJP layer contract drift")
    if manifest["activation_spectrum"].get("layers") != [0, 7, 14]:
        raise ValueError("Activation-gallery spectrum layer contract drift")
    parity = manifest["policy_lens"]
    if any(
        float(error) > float(parity["final_parity_atol"])
        for error in parity["final_parity_max_abs_error"].values()
    ):
        raise ValueError("Activation-gallery final policy-lens parity failed")
    if data_bytes >= 40_000_000:
        raise ValueError("Activation gallery exceeds the 40 MB course tensor budget")
    if manifest["redistribution"].get("standalone_release_gate") is None:
        raise ValueError("Activation gallery lacks its standalone redistribution gate")
    probe_record = manifest["probes"]
    if probe_record.get("compatibility_gate", {}).get("passed") is not True:
        raise ValueError("Activation gallery lacks a passing probe compatibility gate")
    if probe_record.get("test_split_opened") is not False:
        raise ValueError("Activation-gallery probe contract opened the test split")
    probe_validation = probe_record.get("independent_validation", {})
    if (
        probe_validation.get("evaluation_rows_all_development") is not True
        or probe_validation.get("test_rows_evaluated") is not False
        or float(probe_validation.get("maximum_absolute_metric_difference", 1.0)) > 1e-8
    ):
        raise ValueError("Activation-gallery independent probe validation drift")
    required_sources = {
        "gallery_builder",
        "comparison_model_loader",
        "torch_model_definition",
        "bt4_weight_loader",
        "bt4_proto_schema",
        "board_encoding",
        "action_codec",
        "action_codec_move_table",
        "policy_attention_map",
        "corpus_loader",
        "concept_labels",
        "artifact_integrity",
        "deterministic_npz",
        "policy_lens",
        "attribution_objective",
        "legal_policy_objective",
        "probe_result_validator",
        "probe_run_selector",
        "probe_metrics_code",
        "lookahead_probe",
        "public_corpus_manifest",
        "public_corpus_checksums",
        "public_corpus_puzzles",
        "public_corpus_engine_positions",
        "raw_bt4",
        "hero_checkpoint_manifest",
        "hero_checkpoint_state",
        "probe_run_manifest",
        "probe_metadata",
        "probe_checksums",
        "probe_weights",
        "probe_run_metrics",
        "probe_predictions",
    }
    missing_sources = required_sources - set(manifest["sources"])
    if missing_sources:
        raise ValueError(
            f"Activation-gallery scientific source closure drift: {sorted(missing_sources)}"
        )
    source_record = manifest["sources"]
    if probe_record["source_manifest_sha256"] != source_record["probe_run_manifest"][
        "sha256"
    ]:
        raise ValueError("Activation-gallery probe manifest binding drift")
    if probe_record["source_weights_sha256"] != source_record["probe_weights"]["sha256"]:
        raise ValueError("Activation-gallery probe weight binding drift")
    if probe_record["public_corpus_integrity"] != manifest[
        "source_public_corpus_integrity"
    ]:
        raise ValueError("Activation-gallery probe/corpus binding drift")
    for key, value in probe_record["model_identities"].items():
        if manifest["models"].get(key) != value:
            raise ValueError(f"Activation-gallery probe/model binding drift: {key}")
    data_path = descriptor.data_path
    ignored = subprocess.run(
        ("git", "check-ignore", "--quiet", str(data_path.relative_to(root))),
        cwd=root,
        check=False,
    )
    if ignored.returncode != 0:
        raise ValueError(
            "The license-gated activation NPZ must remain gitignored until human review"
        )
    return position_count, data_bytes, materialized


def validate_media_assets(root: Path) -> int:
    manifest_path = root / "curriculum/interpretability/assets/animations/manifest.json"
    value = json.loads(manifest_path.read_text(encoding="utf-8"))
    if value.get("schema_version") != "chess-interp-animation-manifest-v1":
        raise ValueError("Unknown animation manifest schema")
    source = value.get("source")
    if not isinstance(source, dict):
        raise ValueError("Animation manifest has no source record")
    source_path = root / str(source["path"])
    if _sha256_file(source_path) != source.get("sha256"):
        raise ValueError("Animation source hash drift")
    animations = value.get("animations")
    if not isinstance(animations, list) or {row.get("id") for row in animations} != {
        "superposition",
        "johnson_lindenstrauss",
    }:
        raise ValueError("Animation manifest inventory drift")
    total_size = 0
    for animation in animations:
        duration = float(animation["duration_seconds"])
        if not 1.0 <= duration <= 30.0:
            raise ValueError(f"Animation duration is out of bounds: {animation['id']}")
        files = animation.get("files")
        if not isinstance(files, dict) or set(files) != {"video", "poster", "captions"}:
            raise ValueError(f"Animation files are incomplete: {animation['id']}")
        for kind, record in files.items():
            path = root / str(record["path"])
            if not path.is_file() or path.stat().st_size != int(record["size_bytes"]):
                raise ValueError(f"Animation {kind} size drift: {path}")
            if _sha256_file(path) != record["sha256"]:
                raise ValueError(f"Animation {kind} hash drift: {path}")
            total_size += path.stat().st_size
        video = root / str(files["video"]["path"])
        poster = root / str(files["poster"]["path"])
        captions = root / str(files["captions"]["path"])
        if video.read_bytes()[4:8] != b"ftyp":
            raise ValueError(f"Animation is not an MP4 container: {video}")
        if not poster.read_bytes().startswith(b"\x89PNG\r\n\x1a\n"):
            raise ValueError(f"Animation poster is not PNG: {poster}")
        captions_text = captions.read_text(encoding="utf-8")
        if not captions_text.startswith("WEBVTT"):
            raise ValueError(f"Animation captions are not WEBVTT: {captions}")
        _validate_vtt_timing(
            captions_text,
            duration=duration,
            animation_id=str(animation["id"]),
        )
    if total_size > 2_000_000:
        raise ValueError("Animation assets exceed the 2 MB course budget")
    return len(animations)


def _marimo_executable(root: Path) -> str:
    local = root / ".venv/bin/marimo"
    if local.is_file():
        return str(local)
    resolved = shutil.which("marimo")
    if resolved is None:
        raise FileNotFoundError("marimo executable is unavailable")
    return resolved


def _run(command: Iterable[str], *, root: Path, env: dict[str, str]) -> None:
    completed = subprocess.run(
        tuple(command),
        cwd=root,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode:
        details = (completed.stdout + "\n" + completed.stderr).strip()
        raise RuntimeError(f"Command failed ({' '.join(command)}):\n{details}")


def validate_marimo(
    root: Path,
    paths: tuple[Path, ...],
    *,
    export: bool,
) -> int:
    executable = _marimo_executable(root)
    environment = dict(os.environ)
    prior_path = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = str(root) + (os.pathsep + prior_path if prior_path else "")
    notebook_dir = root / "curriculum/interpretability/notebooks"
    _run((executable, "check", "--strict", str(notebook_dir)), root=root, env=environment)
    if not export:
        return 0

    exported_bytes = 0
    with tempfile.TemporaryDirectory(prefix="chess-interp-course-") as temporary:
        output_dir = Path(temporary)
        for module, path in zip(MODULES, paths, strict=True):
            output = output_dir / f"{path.stem}.html"
            _run(
                (executable, "export", "html", str(path), "-o", str(output)),
                root=root,
                env=environment,
            )
            html = output.read_text(encoding="utf-8")
            required_static_markers = (
                module.title,
                "SNAPSHOT",
                "ci-limit",
                "Resource / provenance",
            )
            missing_markers = [
                marker for marker in required_static_markers if marker not in html
            ]
            if missing_markers:
                raise ValueError(f"Static export lacks title/provenance badge: {path.name}")
            if output.stat().st_size < 50_000:
                raise ValueError(f"Static export is unexpectedly incomplete: {path.name}")
            exported_bytes += output.stat().st_size
    return exported_bytes


_NOTEBOOK_CODE_PATTERN = re.compile(
    r'notebookCode:\s*("(?:\\.|[^"\\])*")',
)


def embedded_notebook_source(html: str) -> str:
    """Decode the exact Python source embedded in a marimo static export."""

    match = _NOTEBOOK_CODE_PATTERN.search(html)
    if match is None:
        raise ValueError("Static export does not embed notebookCode")
    decoded = json.loads(match.group(1))
    if not isinstance(decoded, str):
        raise ValueError("Static export notebookCode is not a string")
    return decoded


def validate_committed_site(root: Path, paths: tuple[Path, ...]) -> int:
    """Fail closed when a committed blog is stale relative to its notebook."""

    site = root / "curriculum/interpretability/site"
    total_bytes = 0
    for module, path in zip(MODULES, paths, strict=True):
        export = site / f"{path.stem}.html"
        if not export.is_file():
            raise ValueError(f"Missing committed static export: {export.name}")
        html = export.read_text(encoding="utf-8")
        if embedded_notebook_source(html) != path.read_text(encoding="utf-8"):
            raise ValueError(f"Committed static export is stale: {export.name}")
        required = (module.title, "SNAPSHOT", "ci-limit", "Resource / provenance")
        if any(marker not in html for marker in required):
            raise ValueError(f"Committed static export lacks course markers: {export.name}")
        if export.stat().st_size < 50_000:
            raise ValueError(f"Committed static export is incomplete: {export.name}")
        total_bytes += export.stat().st_size
    return total_bytes


def validate_course(
    root: Path | None = None,
    *,
    verify_source: bool = True,
    export: bool = True,
) -> ValidationReport:
    selected_root = find_repo_root(root).resolve()
    validate_catalog()
    paths = notebook_paths(selected_root)
    for module, path in zip(MODULES, paths, strict=True):
        validate_notebook_source(path, expected_index=module.index)
    snapshot_sha256, source_count = validate_snapshot(
        selected_root,
        verify_source=verify_source,
    )
    (
        activation_position_count,
        activation_bytes,
        activation_materialized,
    ) = validate_activation_gallery(
        selected_root,
        verify_source=verify_source,
    )
    media_count = validate_media_assets(selected_root)
    exported_bytes = validate_marimo(selected_root, paths, export=export)
    validate_committed_site(selected_root, paths)
    return ValidationReport(
        notebook_count=len(paths),
        source_count=source_count,
        media_count=media_count,
        activation_position_count=activation_position_count,
        activation_bytes=activation_bytes,
        activation_materialized=activation_materialized,
        activation_sources_verified=verify_source and activation_materialized,
        snapshot_sha256=snapshot_sha256,
        exported_bytes=exported_bytes,
        source_verified=verify_source,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--skip-source", action="store_true")
    parser.add_argument("--skip-export", action="store_true")
    args = parser.parse_args()
    report = validate_course(
        args.root,
        verify_source=not args.skip_source,
        export=not args.skip_export,
    )
    print(
        f"course validation passed: notebooks={report.notebook_count} "
        f"sources={report.source_count} source_verified={report.source_verified} "
        f"real_positions={report.activation_position_count} "
        f"activation_gallery={report.activation_bytes}B "
        f"activation_materialized={report.activation_materialized} "
        f"activation_sources_verified={report.activation_sources_verified} "
        f"media={report.media_count} exports={report.exported_bytes}B "
        f"snapshot={report.snapshot_sha256}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
