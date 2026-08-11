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


@dataclass(frozen=True)
class ValidationReport:
    notebook_count: int
    source_count: int
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


def validate_snapshot(root: Path, *, verify_source: bool) -> tuple[str, int]:
    toy = load_course_bundle("toy", root=root)
    if toy.is_empirical or toy.model_labels != ("Constructed A", "Constructed B"):
        raise ValueError("Toy evidence boundary is malformed")
    snapshot = load_course_bundle("snapshot", root=root)
    if not snapshot.is_empirical or snapshot.model_labels != ("Raw BT4", "Hero"):
        raise ValueError("Snapshot empirical labels are malformed")
    if set(snapshot.payload) != {
        "architecture",
        "attribution",
        "behavior",
        "circuits",
        "design",
        "geometry",
        "intervention",
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
    for record in value["sources"]:
        if Path(record["path"]).suffix not in {".json", ".jsonl", ".md"}:
            raise ValueError(f"Snapshot references a non-derived source: {record['path']}")
        if record["redistribution"] != "derived_metrics_or_minimal_fen_only":
            raise ValueError(f"Unreviewed redistribution status: {record['id']}")
    if build_snapshot(root) != value:
        raise ValueError("Committed snapshot is not a deterministic source extraction")
    if verify_source:
        source = load_course_bundle("source", root=root)
        if source.payload != snapshot.payload or source.integrity_sha256 != snapshot.integrity_sha256:
            raise ValueError("Source mode does not reproduce snapshot mode")
        for record in value["sources"]:
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
    exported_bytes = validate_marimo(selected_root, paths, export=export)
    validate_committed_site(selected_root, paths)
    return ValidationReport(
        notebook_count=len(paths),
        source_count=source_count,
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
        f"exports={report.exported_bytes}B snapshot={report.snapshot_sha256}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
