# /// script
# dependencies = ["marimo"]
# requires-python = ">=3.11"
# ///

import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium")


@app.cell
def imports():
    from pathlib import Path
    import hashlib
    import html
    import json
    import marimo as mo

    return Path, hashlib, html, json, mo


@app.cell
def title(mo):
    mo.vstack(
        [
            mo.Html(
                """
                <style>
                  :root { --ink:#162235; --muted:#65758b; --raw:#5470c6; --hero:#d96c3f;
                          --good:#16805d; --warn:#b06b13; --bad:#b54747; --panel:#f5f7fb; }
                  .hero-title { padding: 1.2rem 1.4rem; border-radius: 18px;
                    background: linear-gradient(120deg,#162235,#334a70 62%,#8c4d38); color:white; }
                  .hero-title h1 { margin:0; font-size:2rem; line-height:1.12; }
                  .hero-title p { margin:.55rem 0 0; color:#e7edf7; max-width:76ch; }
                  .cards { display:grid; grid-template-columns:repeat(auto-fit,minmax(155px,1fr)); gap:.65rem; }
                  .card { background:var(--panel); border:1px solid #dce3ee; border-radius:12px; padding:.8rem .9rem; }
                  .card .value { font-size:1.45rem; font-weight:720; color:var(--ink); }
                  .card .label { font-size:.82rem; color:var(--muted); margin-top:.2rem; }
                  .insight { border-left:4px solid var(--good); padding:.7rem .9rem; background:#edf8f4; border-radius:8px; }
                  .null { border-left:4px solid var(--warn); padding:.7rem .9rem; background:#fff8eb; border-radius:8px; }
                  .danger { border-left:4px solid var(--bad); padding:.7rem .9rem; background:#fff1f1; border-radius:8px; }
                  .method { border:1px solid #dce3ee; padding:.8rem 1rem; border-radius:12px; background:white; }
                  table.rh { width:100%; border-collapse:collapse; font-size:.88rem; }
                  table.rh th { text-align:left; color:#4f6075; background:#edf1f7; }
                  table.rh th, table.rh td { padding:.5rem .58rem; border-bottom:1px solid #e2e7ef; vertical-align:top; }
                  code.path { font-size:.78rem; color:#334a70; overflow-wrap:anywhere; }
                  .raw { color:var(--raw); font-weight:700; } .hero { color:var(--hero); font-weight:700; }
                  .caption { color:var(--muted); font-size:.82rem; margin-top:.35rem; }
                </style>
                <div class="hero-title">
                  <h1>Raw BT4 → Hero: what changed inside?</h1>
                  <p>A source-bound, executable tour of model diffing, chess probes,
                  published LoRSA transfer, exact token associations, matched causal
                  interventions, encoder-update diagnostics, negative results, and compute cost.</p>
                </div>
                """
            ),
            mo.md(
                """
    This notebook is a **map of evidence, not a victory lap**. Scientific values and timings are
    loaded from retained repository artifacts; price projections are explicitly labeled estimates.
    It runs on CPU, opens no checkpoint, and performs no network requests. Positive, unresolved,
    and failed tests are shown together.
    """
            ),
        ]
    )
    return


@app.cell
def paths(Path, json):
    def find_repo_root():
        candidates = [Path.cwd(), Path(__file__).resolve().parent]
        for candidate in candidates:
            for parent in (candidate, *candidate.parents):
                if (parent / "research" / "analysis").is_dir() and (
                    parent / "pyproject.toml"
                ).exists():
                    return parent
        raise FileNotFoundError(
            "Could not locate the chess-dfm-jax repository root"
        )


    def load_json(relative_path):
        path = REPO_ROOT / relative_path
        return json.loads(path.read_text(encoding="utf-8"))


    REPO_ROOT = find_repo_root()
    ARTIFACT_PATHS = {
        "synthesis": "research/analysis/raw_hero_interpretability_report_v1/synthesis.json",
        "validation": "research/analysis/raw_hero_interpretability_report_v1/validation.json",
        "cost": "research/analysis/raw_hero_interpretability_report_v1/cost_audit.json",
        "model_diff": "research/analysis/raw_hero_pilot_v1/metrics.json",
        "literature": "research/analysis/raw_hero_literature_pilot_v2/metrics.json",
        "probes": "research/analysis/modal_probe_all_layers_dev_v1/metrics.json",
        "parameters": "research/analysis/raw_hero_parameter_diff_20260807.json",
        "spectra": "research/analysis/raw_hero_matrix_spectra_20260807.json",
        "transcoder_dev64": "research/analysis/published_tc_l14_transfer_dev64_v1/metrics.json",
        "lorsa_transfer": "research/analysis/published_lorsa_l14_transfer_dev512_v3/metrics.json",
        "lorsa_transfer_manifest": "research/analysis/published_lorsa_l14_transfer_dev512_v3/manifest.json",
        "lorsa_bootstrap": "research/analysis/published_lorsa_l14_transfer_dev512_paired_bootstrap_v3/metrics.json",
        "lorsa_bootstrap_manifest": "research/analysis/published_lorsa_l14_transfer_dev512_paired_bootstrap_v3/manifest.json",
        "lorsa_compact": "research/analysis/published_lorsa_l14_compact_peak_semantics_dev512_v4/report.json",
        "lorsa_compact_manifest": "research/analysis/published_lorsa_l14_compact_peak_semantics_dev512_v4/manifest.json",
        "lorsa_token": "research/analysis/published_lorsa_l14_token_semantics_dev512_v3/metrics.json",
        "lorsa_token_selection": "research/analysis/published_lorsa_l14_token_semantics_dev512_v3/selection.json",
        "lorsa_token_manifest": "research/analysis/published_lorsa_l14_token_semantics_dev512_v3/manifest.json",
        "lorsa_token_v1_metrics": "research/analysis/published_lorsa_l14_token_semantics_dev512_v1/metrics.json",
        "lorsa_token_v1_selection": "research/analysis/published_lorsa_l14_token_semantics_dev512_v1/selection.json",
        "lorsa_modal_t4": "research/analysis/modal_lorsa_smoke_t4_v2/manifest.json",
        "lorsa_modal_l4": "research/analysis/modal_lorsa_smoke_l4_v2/manifest.json",
        "lorsa_causal": "research/analysis/published_lorsa_l14_causal_dev256_v2/metrics.json",
        "lorsa_causal_manifest": "research/analysis/published_lorsa_l14_causal_dev256_v2/manifest.json",
        "hero_lr_audit": "research/analysis/hero_encoder_lr_audit_20260808.json",
    }
    LORSA_RUN_DIRS = {
        "LoRSA transfer v3 (final)": "research/analysis/published_lorsa_l14_transfer_dev512_v3",
        "Paired bootstrap v3 (final)": "research/analysis/published_lorsa_l14_transfer_dev512_paired_bootstrap_v3",
        "Compact peak semantics v4": "research/analysis/published_lorsa_l14_compact_peak_semantics_dev512_v4",
        "Exact token semantics v3 (final)": "research/analysis/published_lorsa_l14_token_semantics_dev512_v3",
        "Token selector v1 (diagnostic only)": "research/analysis/published_lorsa_l14_token_semantics_dev512_v1",
        "Modal T4 LoRSA smoke": "research/analysis/modal_lorsa_smoke_t4_v2",
        "Modal L4 LoRSA smoke": "research/analysis/modal_lorsa_smoke_l4_v2",
        "Definitive LoRSA causal v2": "research/analysis/published_lorsa_l14_causal_dev256_v2",
    }
    return ARTIFACT_PATHS, LORSA_RUN_DIRS, REPO_ROOT, load_json


@app.cell
def load_artifacts(ARTIFACT_PATHS, REPO_ROOT, load_json):
    artifacts = {name: load_json(path) for name, path in ARTIFACT_PATHS.items()}
    source_rows = [
        {
            "artifact": name,
            "path": path,
            "bytes": (REPO_ROOT / path).stat().st_size,
            "schema": artifacts[name].get("schema_version", "aggregate"),
        }
        for name, path in ARTIFACT_PATHS.items()
    ]
    return artifacts, source_rows


@app.cell
def helpers(html, mo):
    def fmt(value, digits=3):
        if value is None:
            return "—"
        if isinstance(value, bool):
            return "yes" if value else "no"
        if isinstance(value, int):
            return f"{value:,}"
        if not isinstance(value, (float, int)):
            return str(value)
        magnitude = abs(value)
        if magnitude and (magnitude < 0.001 or magnitude >= 10_000):
            return f"{value:.{digits}e}"
        return f"{value:.{digits}f}"


    def html_table(rows, columns=None):
        if not rows:
            return '<div class="caption">No rows.</div>'
        selected = columns or list(rows[0])
        header = "".join(
            f"<th>{html.escape(str(label))}</th>" for _, label in selected
        )
        body = []
        for row in rows:
            body.append(
                "<tr>"
                + "".join(
                    f"<td>{html.escape(str(row.get(key, '—')))}</td>"
                    for key, _ in selected
                )
                + "</tr>"
            )
        return f'<table class="rh"><thead><tr>{header}</tr></thead><tbody>{"".join(body)}</tbody></table>'


    def metric_cards(cards):
        blocks = "".join(
            f'<div class="card"><div class="value">{html.escape(str(value))}</div>'
            f'<div class="label">{html.escape(str(label))}</div></div>'
            for value, label in cards
        )
        return mo.Html(f'<div class="cards">{blocks}</div>')


    def line_chart_html(series, y_label="", y_domain=None, width=760, height=270):
        left, right, top, bottom = 58, 18, 25, 42
        plot_w, plot_h = width - left - right, height - top - bottom
        points_all = [point for item in series for point in item["points"]]
        xs = [point[0] for point in points_all]
        ys = [point[1] for point in points_all]
        x_min, x_max = min(xs), max(xs)
        if y_domain is None:
            y_min, y_max = min(ys), max(ys)
            padding = (y_max - y_min) * 0.12 or 0.1
            y_min, y_max = y_min - padding, y_max + padding
        else:
            y_min, y_max = y_domain

        def sx(x):
            return left + (x - x_min) / max(x_max - x_min, 1e-12) * plot_w

        def sy(y):
            return top + (y_max - y) / max(y_max - y_min, 1e-12) * plot_h

        parts = [
            f'<svg viewBox="0 0 {width} {height}" style="width:100%;max-height:{height}px" role="img">'
        ]
        for tick in range(5):
            value = y_min + (y_max - y_min) * tick / 4
            y = sy(value)
            parts.append(
                f'<line x1="{left}" x2="{width - right}" y1="{y:.1f}" y2="{y:.1f}" stroke="#dfe5ee"/>'
            )
            parts.append(
                f'<text x="{left - 8}" y="{y + 4:.1f}" text-anchor="end" fill="#65758b" font-size="11">{fmt(value, 2)}</text>'
            )
        for x in sorted(set(xs)):
            if len(set(xs)) <= 16:
                parts.append(
                    f'<text x="{sx(x):.1f}" y="{height - 18}" text-anchor="middle" fill="#65758b" font-size="10">{x}</text>'
                )
        parts.append(
            f'<text x="14" y="{height / 2:.1f}" transform="rotate(-90 14 {height / 2:.1f})" text-anchor="middle" fill="#65758b" font-size="11">{html.escape(y_label)}</text>'
        )
        for item in series:
            color = item.get("color", "#5470c6")
            coords = " ".join(
                f"{sx(x):.1f},{sy(y):.1f}" for x, y in item["points"]
            )
            parts.append(
                f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="2.5"/>'
            )
            for x, y in item["points"]:
                parts.append(
                    f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="3" fill="{color}"/>'
                )
        legend_x = left
        for item in series:
            color = item.get("color", "#5470c6")
            parts.append(f'<circle cx="{legend_x}" cy="12" r="4" fill="{color}"/>')
            parts.append(
                f'<text x="{legend_x + 8}" y="16" fill="#33445b" font-size="11">{html.escape(item["name"])}</text>'
            )
            legend_x += 30 + 8 * len(item["name"])
        parts.append("</svg>")
        return mo.Html("".join(parts))


    def horizontal_bars_html(
        rows, value_key, label_key, color="#5470c6", percent=False
    ):
        maximum = max(abs(float(row[value_key])) for row in rows) or 1.0
        blocks = []
        for row in rows:
            value = float(row[value_key])
            width = 100 * abs(value) / maximum
            shown = f"{100 * value:.2f}%" if percent else fmt(value)
            blocks.append(
                f'<div style="display:grid;grid-template-columns:minmax(145px,2fr) 5fr 72px;gap:.55rem;align-items:center;margin:.38rem 0">'
                f'<div style="font-size:.82rem;overflow-wrap:anywhere">{html.escape(str(row[label_key]))}</div>'
                f'<div style="height:12px;background:#edf1f6;border-radius:8px"><div style="height:12px;width:{width:.2f}%;background:{color};border-radius:8px"></div></div>'
                f'<div style="font-variant-numeric:tabular-nums;font-size:.8rem">{shown}</div></div>'
            )
        return mo.Html("".join(blocks))

    return horizontal_bars_html, html_table, line_chart_html, metric_cards


@app.cell
def integrity(LORSA_RUN_DIRS, REPO_ROOT, artifacts, hashlib, json):
    def sha256_file(path):
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
        return digest.hexdigest()


    def canonical_identity(payload):
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


    source_checks = []
    for _label, _metadata in artifacts["synthesis"]["source_files"].items():
        _source_path = REPO_ROOT / _metadata["path"]
        _actual_sha = sha256_file(_source_path)
        source_checks.append(
            {
                "source": _label,
                "path": _metadata["path"],
                "hash match": _actual_sha == _metadata["sha256"],
                "sha256": _actual_sha[:12] + "…",
            }
        )
    checksum_pass_count = sum(row["hash match"] for row in source_checks)

    lorsa_ledger_checks = []
    for _label, _relative_dir in LORSA_RUN_DIRS.items():
        _run_dir = REPO_ROOT / _relative_dir
        _ledger_path = _run_dir / "checksums.sha256"
        _expected = {}
        for _line in _ledger_path.read_text(encoding="utf-8").splitlines():
            _digest, _relative = _line.split("  ", 1)
            _expected[_relative] = _digest
        _actual_names = {
            _path.relative_to(_run_dir).as_posix()
            for _path in _run_dir.rglob("*")
            if _path.is_file()
            and _path.name != "checksums.sha256"
            and not _path.name.endswith(".partial")
        }
        _inventory_exact = set(_expected) == _actual_names
        _hashes_match = _inventory_exact and all(
            sha256_file(_run_dir / _relative) == _digest
            for _relative, _digest in _expected.items()
        )
        lorsa_ledger_checks.append(
            {
                "artifact": _label,
                "files": len(_expected),
                "inventory exact": _inventory_exact,
                "hashes match": _hashes_match,
                "ledger SHA": sha256_file(_ledger_path)[:12] + "…",
            }
        )
    _bad_ledgers = [row["artifact"] for row in lorsa_ledger_checks if not row["hashes match"]]
    if _bad_ledgers:
        raise RuntimeError("LoRSA checksum validation failed: " + ", ".join(_bad_ledgers))
    lorsa_ledger_pass_count = sum(row["hashes match"] for row in lorsa_ledger_checks)

    _transfer_manifest = artifacts["lorsa_transfer_manifest"]
    _bootstrap_manifest = artifacts["lorsa_bootstrap_manifest"]
    _compact_manifest = artifacts["lorsa_compact_manifest"]
    _token_manifest = artifacts["lorsa_token_manifest"]
    _causal_manifest = artifacts["lorsa_causal_manifest"]
    _transfer_run_id = _transfer_manifest["run_id"]
    _transfer_payloads = _transfer_manifest["payload_identities"]
    _token_payloads = _token_manifest["payload_identities"]
    _bootstrap_payloads = _bootstrap_manifest["payload_identities"]
    _causal_payloads = _causal_manifest["payload_identities"]
    _lr_audit_payload = dict(artifacts["hero_lr_audit"])
    _lr_audit_claimed_identity = _lr_audit_payload.pop("content_identity")
    lorsa_chain_checks = [
        {
            "contract": "transfer logical ID recomputes",
            "passed": _transfer_run_id == "sha256:" + canonical_identity(_transfer_payloads),
            "evidence": _transfer_run_id[7:19] + "…",
        },
        {
            "contract": "51-file source map is bound in transfer ID",
            "passed": len(_transfer_manifest["source_files"]) == 51
            and canonical_identity(_transfer_manifest["source_files"])
            == _transfer_payloads["source_files"],
            "evidence": _transfer_payloads["source_files"][:12] + "…",
        },
        {
            "contract": "paired bootstrap points to final transfer",
            "passed": _bootstrap_manifest["input_artifact"]["run_id"] == _transfer_run_id
            and _bootstrap_manifest["run_id"]
            == "sha256:" + canonical_identity(_bootstrap_payloads),
            "evidence": _bootstrap_manifest["run_id"][7:19] + "…",
        },
        {
            "contract": "compact report points to final transfer",
            "passed": _compact_manifest["source_run"]["run_id"] == _transfer_run_id
            and canonical_identity(artifacts["lorsa_compact"])
            == _compact_manifest["report_identity_sha256"],
            "evidence": _compact_manifest["report_identity_sha256"][:12] + "…",
        },
        {
            "contract": "exact-token run points to final transfer",
            "passed": _token_manifest["source_transfer_gate"]["source_transfer_run_id"]
            == _transfer_run_id
            and _token_manifest["run_id"]
            == "sha256:" + canonical_identity(_token_payloads),
            "evidence": _token_manifest["run_id"][7:19] + "…",
        },
        {
            "contract": "causal v2 binds the final transfer and semantic audit",
            "passed": _causal_manifest["run_id"]
            == "sha256:" + canonical_identity(_causal_payloads)
            and canonical_identity(artifacts["lorsa_causal"])
            == _causal_payloads["metrics"]
            and _causal_manifest["dependency_gates"]["source_transfer"][
                "source_transfer_run_id"
            ]
            == _transfer_run_id
            and _causal_manifest["dependency_gates"]["semantic_source"][
                "semantic_run_id"
            ]
            == _token_manifest["run_id"]
            and not artifacts["lorsa_causal"]["claim_gates"][
                "reserved_test_split_selected_or_evaluated"
            ],
            "evidence": _causal_manifest["run_id"][7:19] + "…",
        },
        {
            "contract": "Hero encoder audit identity recomputes",
            "passed": canonical_identity(_lr_audit_payload)
            == _lr_audit_claimed_identity,
            "evidence": _lr_audit_claimed_identity[:12] + "…",
        },
        {
            "contract": "reserved test rows selected/evaluated",
            "passed": _token_manifest["test_split_contract"]["test_rows_selected"] == 0
            and not artifacts["lorsa_token"]["claim_gates"][
                "test_split_selected_or_evaluated"
            ]
            and not artifacts["lorsa_causal"]["claim_gates"][
                "reserved_test_split_selected_or_evaluated"
            ],
            "evidence": "0 rows",
        },
    ]
    _bad_chain = [row["contract"] for row in lorsa_chain_checks if not row["passed"]]
    if _bad_chain:
        raise RuntimeError("LoRSA provenance chain failed: " + ", ".join(_bad_chain))
    lorsa_chain_pass_count = sum(row["passed"] for row in lorsa_chain_checks)

    lorsa_source_checks = [
        sha256_file(REPO_ROOT / _relative) == _digest
        for _relative, _digest in _transfer_manifest["source_files"].items()
    ]
    lorsa_source_current_pass_count = sum(lorsa_source_checks)
    return (
        checksum_pass_count,
        lorsa_chain_checks,
        lorsa_chain_pass_count,
        lorsa_ledger_checks,
        lorsa_ledger_pass_count,
        lorsa_source_checks,
        lorsa_source_current_pass_count,
        source_checks,
    )


@app.cell
def integrity_view(
    checksum_pass_count,
    html_table,
    lorsa_chain_checks,
    lorsa_chain_pass_count,
    lorsa_ledger_checks,
    lorsa_ledger_pass_count,
    lorsa_source_checks,
    lorsa_source_current_pass_count,
    metric_cards,
    mo,
    source_checks,
):
    mo.vstack(
        [
            mo.md(
                "## Reading the evidence safely\n\nEvery referenced LoRSA run is checked against its exact on-disk inventory and SHA-256 ledger. Logical IDs and final-run dependencies are recomputed separately."
            ),
            metric_cards(
                [
                    (
                        f"{lorsa_ledger_pass_count}/{len(lorsa_ledger_checks)}",
                        "LoRSA ledgers pass",
                    ),
                    (
                        f"{lorsa_chain_pass_count}/{len(lorsa_chain_checks)}",
                        "final-chain contracts pass",
                    ),
                    (
                        f"{lorsa_source_current_pass_count}/{len(lorsa_source_checks)}",
                        "current source files match map",
                    ),
                    (
                        f"{checksum_pass_count}/{len(source_checks)}",
                        "earlier report-source hashes match",
                    ),
                    ("0", "test rows selected/evaluated"),
                    ("CPU / offline", "notebook runtime"),
                ]
            ),
            mo.Html(
                html_table(
                    lorsa_ledger_checks,
                    [
                        ("artifact", "Sealed artifact"),
                        ("files", "Payload files"),
                        ("inventory exact", "Exact inventory"),
                        ("hashes match", "Hashes match"),
                        ("ledger SHA", "Ledger SHA-256"),
                    ],
                )
            ),
            mo.Html(
                html_table(
                    lorsa_chain_checks,
                    [
                        ("contract", "Provenance contract"),
                        ("passed", "Pass"),
                        ("evidence", "Identity / evidence"),
                    ],
                )
            ),
            mo.md("### Earlier synthesis source identity (separate from scientific validity)"),
            mo.Html(
                html_table(
                    source_checks,
                    [
                        ("source", "Source"),
                        ("hash match", "Hash match"),
                        ("sha256", "SHA-256 prefix"),
                        ("path", "Path"),
                    ],
                )
            ),
            mo.md(
                "Matching bytes establish **artifact identity and recoverability**, not that an interpretation is causal or correct. Those claim gates remain part of the result."
            ),
        ]
    )
    return


@app.cell
def method_notes():
    method_notes = {
        "Model diff": (
            "Swap encoder and policy head independently",
            "Localizes the behavioral delta to a module; does not identify a circuit.",
        ),
        "Parameters": (
            "Measure exact BF16 deltas and matrix spectra",
            "Shows where update energy landed; weight distance is not functional importance.",
        ),
        "Activations": (
            "Compare paired residual streams with CKA and exact norms",
            "Shows representation divergence; similarity metrics are basis- and sample-dependent.",
        ),
        "Residual patching": (
            "Replace one model's whole residual with the other's",
            "Tests mediation at a layer; whole-residual patches do not isolate heads or features.",
        ),
        "Concept probes": (
            "Decode chess labels with nested, group-disjoint linear probes",
            "Decodability is not evidence the policy uses the decoded direction.",
        ),
        "Lookahead probes": (
            "Decode a public puzzle's third-ply continuation",
            "A continuation label is not a new engine principal-variation benchmark.",
        ),
        "Causal steering": (
            "Move along a learned probe direction and measure policy derivative",
            "Small-n directional evidence; off-manifold and multiple-testing risks remain.",
        ),
        "Sparse transfer": (
            "Freeze the published layer-14 LoRSA attention replacement, test Raw/Hero fidelity, then carry Raw-fit token associations unchanged to group-disjoint Raw and Hero heldout positions",
            "Fidelity and association are not causal use, semantic equivalence, or evidence of internal search; all results are exploratory development analyses.",
        ),
        "Sparse causal validation": (
            "Ablate or insert frozen LoRSA decoder directions with exact scalar, token-support, and decoder-norm matched random controls",
            "Causality is measured inside the LoRSA replacement path, not for a native dense BT4 feature or the chess input concept itself.",
        ),
        "Encoder update audit": (
            "Join the LR schedule, optimizer precision contract, parameter deltas, and retained validation trajectory",
            "The 30x LR reduction and BF16 direct updates are confounded; this audit motivates, but cannot replace, a controlled sweep.",
        ),
        "Attribution": (
            "Compare board-square saliency maps",
            "Map agreement can coexist with a failed completeness diagnostic.",
        ),
        "Cost audit": (
            "Reconcile provider billing with staged pipeline telemetry",
            "Attempt-level T4/L4 prices below are frozen-rate estimates, not exact billing attribution.",
        ),
    }
    return (method_notes,)


@app.cell
def method_picker(method_notes, mo):
    method_select = mo.ui.dropdown(
        options=list(method_notes),
        value="Model diff",
        label="Method to unpack",
    )
    method_select
    return (method_select,)


@app.cell
def method_guide(html, method_notes, method_select, mo):
    _selected_method = method_notes[method_select.value]
    mo.Html(
        f'<div class="method"><b>{html.escape(method_select.value)}</b><br>'
        f'<span style="color:#16805d">Question:</span> {html.escape(_selected_method[0])}<br>'
        f'<span style="color:#b06b13">Boundary:</span> {html.escape(_selected_method[1])}</div>'
    )
    return


@app.cell
def headline_data(artifacts):
    headline_cards = [
        ("0.107%", "encoder-trunk relative L2 update"),
        ("0.256", "encoder-swap policy JS"),
        ("~1e-5", "policy-head-swap JS"),
        ("120 / 120", "probe scores beat controls"),
        ("1 / 30", "probe-steering CIs excluding zero"),
        ("6 / 12", "LoRSA causal gates pass"),
        ("3 / 6", "features pass in both models"),
        (
            f"${artifacts['cost']['metered_cost_dollars']:.3f}",
            "earlier Modal work metered",
        ),
    ]
    return (headline_cards,)


@app.cell
def headline_view(headline_cards, metric_cards, mo):
    mo.vstack(
        [
            mo.md(
                "## Executive read\n\nHero is behaviorally different and overwhelmingly encoder-mediated, but its dense encoder update was numerically constrained. Three frozen LoRSA decoder directions now pass strict matched-control causal gates in both models. These results deepen the model comparison without establishing a native dense circuit or internal search."
            ),
            metric_cards(headline_cards),
            mo.Html(
                '<div class="insight"><b>Supported:</b> Hero’s changed policy is overwhelmingly encoder-mediated; representations progressively diverge; and LoRSA features 10843, 5429, and 10784 have direction-specific policy effects in both models.</div>'
            ),
            mo.Html(
                '<div class="null"><b>Not supported:</b> systematic Hero lookahead superiority, a native dense-feature circuit, input-concept causality, or the claim that the encoder LR alone caused update starvation.</div>'
            ),
        ]
    )
    return


@app.cell
def arms_data(artifacts):
    arm_order = ["RR", "RH", "HR", "HH"]
    arm_descriptions = {
        "RR": "Raw encoder + Raw native head",
        "RH": "Raw encoder + Hero native head",
        "HR": "Hero encoder + Raw native head",
        "HH": "Hero encoder + Hero native head",
    }
    arm_rows = []
    for _arm in arm_order:
        _entry = artifacts["model_diff"]["behavior"]["arms"][_arm]
        arm_rows.append(
            {
                "arm": _arm,
                "composition": arm_descriptions[_arm],
                "target NLL ↓": f"{_entry['target_nll']['estimate']:.3f}",
                "top-1 ↑": f"{100 * _entry['top1_accuracy']['estimate']:.1f}%",
            }
        )
    swap_keys = ["RR__HR", "RR__RH", "RH__HH", "HR__HH"]
    swap_rows = []
    for _pair in swap_keys:
        _pair_entry = artifacts["model_diff"]["behavior"]["pairs"][_pair]
        swap_rows.append(
            {
                "comparison": artifacts["model_diff"]["behavior"][
                    "pair_semantics"
                ][_pair],
                "JS divergence": f"{_pair_entry['js_divergence']['estimate']:.6f}",
                "top-1 agreement": f"{100 * _pair_entry['top1_agreement']['estimate']:.1f}%",
            }
        )
    return arm_rows, swap_rows


@app.cell
def arms_view(arm_rows, html_table, mo, swap_rows):
    mo.vstack(
        [
            mo.md(
                "## The 2×2 intervention: read the letters first\n\nThe **first letter is the encoder** and the **second is the native policy head**. RR and HH are the natural models; RH and HR are crossed interventions."
            ),
            mo.Html(
                html_table(
                    arm_rows,
                    [
                        ("arm", "Arm"),
                        ("composition", "Composition"),
                        ("target NLL ↓", "Target NLL ↓"),
                        ("top-1 ↑", "Top-1 ↑"),
                    ],
                )
            ),
            mo.Html(
                html_table(
                    swap_rows,
                    [
                        ("comparison", "One-component swap"),
                        ("JS divergence", "Policy JS"),
                        ("top-1 agreement", "Top-1 agreement"),
                    ],
                )
            ),
            mo.Html(
                '<div class="insight"><b>Localization:</b> swapping encoders changes the policy strongly and often changes the top move; swapping native heads is almost inert. This is unusually clean module-level evidence.</div>'
            ),
            mo.Html(
                '<div class="danger"><b>Sampling warning:</b> these 128 positions belong to one placeholder game cluster, so cluster-bootstrap intervals collapse. Treat the point estimates as a development localization pilot.</div>'
            ),
        ]
    )
    return


@app.cell
def parameter_data(artifacts):
    parameter_groups = artifacts["parameters"]["group_metrics"]
    parameter_total_energy = parameter_groups["all"]["delta_l2"] ** 2
    parameter_component_rows = []
    for _component in [
        "smolgen",
        "embedding",
        "attention",
        "layer_norm",
        "mlp",
        "policy_head",
    ]:
        _group = parameter_groups[_component]
        parameter_component_rows.append(
            {
                "component": _component.replace("_", " ").title(),
                "energy_share": _group["delta_l2"] ** 2 / parameter_total_energy,
                "relative_l2": _group["relative_delta_l2"],
                "unchanged": _group["unchanged_fraction"],
            }
        )
    parameter_top_rows = artifacts["parameters"]["ranked_views"][
        "largest_delta_energy_contributors"
    ][:8]
    embedding_projection_spectrum = next(
        row
        for row in artifacts["spectra"]["matrices"]
        if row["name"] == "embedding.proj.w"
    )
    embedding_top_direction_energy = (
        embedding_projection_spectrum["delta"]["singular_values"][0]
        / embedding_projection_spectrum["delta"]["frobenius_l2"]
    ) ** 2
    return (
        embedding_projection_spectrum,
        embedding_top_direction_energy,
        parameter_component_rows,
        parameter_groups,
        parameter_top_rows,
    )


@app.cell
def parameter_view(
    embedding_projection_spectrum,
    embedding_top_direction_energy,
    horizontal_bars_html,
    html_table,
    metric_cards,
    mo,
    parameter_component_rows,
    parameter_groups,
    parameter_top_rows,
):
    _parameter_cards = [
        (
            f"{100 * parameter_groups['trunk']['relative_delta_l2']:.3f}%",
            "encoder-trunk relative L2",
        ),
        (f"{parameter_groups['trunk']['cosine']:.7f}", "encoder-trunk cosine"),
        (
            f"{100 * parameter_groups['all']['unchanged_fraction']:.2f}%",
            "bit-identical aligned entries",
        ),
        (
            f"{100 * embedding_projection_spectrum['delta_energy_fraction_global']:.2f}%",
            "update energy in embedding.proj.w",
        ),
        (
            f"{embedding_projection_spectrum['delta']['stable_rank']:.2f}",
            "embedding.proj.w delta stable rank",
        ),
        (
            f"{100 * embedding_top_direction_energy:.1f}%",
            "energy in its top singular direction",
        ),
    ]
    _top_parameter_table = [
        {
            "tensor": row["name"],
            "energy": f"{100 * row['delta_energy_fraction']:.2f}%",
            "relative L2": f"{100 * row['relative_delta_l2']:.3f}%",
        }
        for row in parameter_top_rows
    ]
    mo.vstack(
        [
            mo.md(
                "## Parameters: small globally, structured locally\n\nEnergy means squared Frobenius norm of `Hero − Raw`. Component shares are disjoint and sum to the aligned BT4 encoder plus native head."
            ),
            metric_cards(_parameter_cards),
            horizontal_bars_html(
                parameter_component_rows,
                "energy_share",
                "component",
                color="#8c4d38",
                percent=True,
            ),
            mo.Html(
                html_table(
                    _top_parameter_table,
                    [
                        ("tensor", "Largest contributors"),
                        ("energy", "Global delta energy"),
                        ("relative L2", "Tensor relative L2"),
                    ],
                )
            ),
            mo.Html(
                '<div class="insight"><b>Positive result:</b> update energy concentrates in SmolGen and the embedding, with one embedding matrix showing an approximately rank-2 update.</div>'
            ),
            mo.Html(
                '<div class="null"><b>Interpretation boundary:</b> About 97% of aligned BF16 entries being bit-identical does not imply intentional sparse optimization. A low learning rate can leave many updates below a BF16 representable step.</div>'
            ),
        ]
    )
    return


@app.cell
def activation_data(artifacts):
    activation_depth_rows = artifacts["synthesis"]["datasets"]["model_diff"][
        "activation_depth"
    ]
    residual_layer_metrics = artifacts["model_diff"]["activation"]["hooks"][
        "resid_post_after_ln"
    ]["layers"]
    return activation_depth_rows, residual_layer_metrics


@app.cell
def layer_picker(mo):
    layer_select = mo.ui.slider(
        start=0, stop=14, step=1, value=14, label="Inspect layer"
    )
    layer_select
    return (layer_select,)


@app.cell
def activation_view(
    activation_depth_rows,
    layer_select,
    line_chart_html,
    metric_cards,
    mo,
    residual_layer_metrics,
):
    _selected_depth = next(
        row for row in activation_depth_rows if row["layer"] == layer_select.value
    )
    _selected_exact = residual_layer_metrics[str(layer_select.value)][
        "exact_elementwise"
    ]
    _activation_series = [
        {
            "name": "Linear CKA",
            "color": "#5470c6",
            "points": [
                (row["layer"], row["corresponding_layer_cka"])
                for row in activation_depth_rows
            ],
        },
        {
            "name": "Symmetric relative L2",
            "color": "#d96c3f",
            "points": [
                (row["layer"], row["symmetric_relative_l2"])
                for row in activation_depth_rows
            ],
        },
    ]
    mo.vstack(
        [
            mo.md(
                "## Activations: divergence accumulates with depth\n\nThe same 128 development positions are passed through both encoders. CKA describes shared multivariate geometry; symmetric relative L2 describes scale-sensitive distance."
            ),
            line_chart_html(
                _activation_series,
                y_label="similarity / distance",
                y_domain=(0.0, 1.2),
            ),
            metric_cards(
                [
                    (
                        f"{_selected_depth['corresponding_layer_cka']:.3f}",
                        f"layer {layer_select.value} CKA",
                    ),
                    (
                        f"{_selected_depth['symmetric_relative_l2']:.3f}",
                        "symmetric relative L2",
                    ),
                    (
                        f"{_selected_exact['cosine_similarity']:.3f}",
                        "exact elementwise cosine",
                    ),
                    (f"{_selected_exact['delta_rms']:.3f}", "exact delta RMS"),
                    (f"{_selected_exact['raw_rms']:.3f}", "Raw residual RMS"),
                    (f"{_selected_exact['hero_rms']:.3f}", "Hero residual RMS"),
                ]
            ),
            mo.Html(
                '<div class="insight"><b>Depth trend:</b> corresponding-layer CKA falls from 0.819 to 0.236 while relative distance grows to 1.144. Tiny parameter movement need not mean tiny function movement.</div>'
            ),
            mo.Html(
                '<div class="null"><b>Metric boundary:</b> representation similarity is descriptive. It does not say which dimensions matter to the policy, and the multivariate CKA uses a bounded sketch.</div>'
            ),
        ]
    )
    return


@app.cell
def literature_data(artifacts):
    literature_summary = artifacts["synthesis"]["datasets"]["literature"]
    literature_anchor_layers = [0, 7, 14]
    return literature_anchor_layers, literature_summary


@app.cell
def anchor_picker(literature_anchor_layers, mo):
    anchor_select = mo.ui.dropdown(
        options=literature_anchor_layers,
        value=14,
        label="Literature-method anchor layer",
    )
    anchor_select
    return (anchor_select,)


@app.cell
def literature_view(
    anchor_select,
    html_table,
    line_chart_html,
    literature_summary,
    metric_cards,
    mo,
):
    _anchor_delta = next(
        row
        for row in literature_summary["paired_delta"]
        if row["layer"] == anchor_select.value
    )
    _anchor_attention = next(
        row
        for row in literature_summary["attention_pattern_js"]
        if row["layer"] == anchor_select.value
    )
    _anchor_patches = [
        row
        for row in literature_summary["patching"]
        if row["layer"] == anchor_select.value
    ]
    _lens_rows = [
        row
        for row in literature_summary["logit_lens"]
        if row["layer"] == anchor_select.value
    ]
    _patch_series = [
        {
            "name": direction,
            "color": color,
            "points": [
                (row["layer"], row["distribution_delta_projection"])
                for row in literature_summary["patching"]
                if row["direction"] == direction
            ],
        }
        for direction, color in [
            ("Raw → Hero", "#d96c3f"),
            ("Hero → Raw", "#5470c6"),
        ]
    ]
    _delta_series = [
        {
            "name": "Delta RMS",
            "color": "#d96c3f",
            "points": [
                (row["layer"], row["mean_position_delta_rms"])
                for row in literature_summary["paired_delta"]
            ],
        },
        {
            "name": "Top-16 variance",
            "color": "#16805d",
            "points": [
                (row["layer"], row["top16_variance_share"])
                for row in literature_summary["paired_delta"]
            ],
        },
    ]
    mo.vstack(
        [
            mo.md(
                "## Literature-style diagnostics: where does the delta become actionable?\n\nThe expensive pilot examines three preregistered anchor layers on eight positions. Projection 1.0 means the patched distribution moved by the full baseline-to-reference delta along that direction."
            ),
            line_chart_html(
                _patch_series,
                y_label="distribution-delta projection",
                y_domain=(0.0, 1.08),
            ),
            line_chart_html(
                _delta_series, y_label="RMS / variance share", y_domain=(0.0, 0.9)
            ),
            metric_cards(
                [
                    (
                        f"{_anchor_delta['mean_position_delta_rms']:.3f}",
                        f"layer {anchor_select.value} paired delta RMS",
                    ),
                    (
                        f"{_anchor_delta['entropy_effective_rank']:.1f}",
                        "delta entropy effective rank",
                    ),
                    (
                        f"{100 * _anchor_delta['top16_variance_share']:.1f}%",
                        "top-16 delta variance",
                    ),
                    (
                        f"{_anchor_attention['pattern_js']:.4f}",
                        "attention-pattern JS",
                    ),
                    (
                        f"{_anchor_patches[0]['distribution_delta_projection']:.3f}",
                        _anchor_patches[0]["direction"] + " patch",
                    ),
                    (
                        f"{_anchor_patches[1]['distribution_delta_projection']:.3f}",
                        _anchor_patches[1]["direction"] + " patch",
                    ),
                ]
            ),
            mo.Html(
                html_table(
                    [
                        {
                            "arm": row["arm"],
                            "target probability": f"{row['target_probability']:.4f}",
                        }
                        for row in _lens_rows
                    ],
                    [
                        ("arm", "Logit-lens arm"),
                        (
                            "target probability",
                            f"Layer {anchor_select.value} target probability",
                        ),
                    ],
                )
            ),
            mo.Html(
                '<div class="insight"><b>Causal pilot:</b> at layer 14, bidirectional whole-residual patches reproduce essentially all of the fixed-head policy-distribution delta.</div>'
            ),
            mo.Html(
                '<div class="null"><b>Negative/detail result:</b> attention patterns change only modestly at the three anchors. The large residual divergence is not simply a wholesale attention-pattern rewrite.</div>'
            ),
            mo.Html(
                '<div class="null"><b>Late crystallization:</b> the logit lens stays near 6% target probability until late and jumps at layer 14 for every arm; Hero is only slightly higher in this eight-position pilot.</div>'
            ),
        ]
    )
    return


@app.cell
def attribution_view(literature_summary, metric_cards, mo):
    _attrib = literature_summary["attribution"]
    mo.vstack(
        [
            mo.md("### Attribution: a useful failure is still a result"),
            metric_cards(
                [
                    (
                        f"{_attrib['sarfa_cosine']['mean']:.3f}",
                        "Raw/Hero SARFA cosine",
                    ),
                    (
                        f"{_attrib['sarfa_pearson']['mean']:.3f}",
                        "Raw/Hero SARFA Pearson",
                    ),
                    (
                        f"{_attrib['sarfa_top8_jaccard']['mean']:.3f}",
                        "SARFA top-8 Jaccard",
                    ),
                    (
                        f"{_attrib['integrated_gradients_relative_completeness_error']['mean']:.2f}",
                        "IG relative completeness error",
                    ),
                ]
            ),
            mo.Html(
                '<div class="null"><b>Do not over-read the saliency agreement.</b> SARFA has only two examples, while integrated gradients failed its own completeness diagnostic (mean relative error 4.50 across four arm-examples). IG maps are method-debugging evidence here, not a model claim.</div>'
            ),
        ]
    )
    return


@app.cell
def probe_data(artifacts):
    concept_metric_rows = artifacts["synthesis"]["datasets"]["concept_metrics"]
    concept_delta_rows = artifacts["synthesis"]["datasets"]["concept_deltas"]
    concept_names = ["Piece code", "Legal destination", "Opponent attack count"]
    return concept_delta_rows, concept_metric_rows, concept_names


@app.cell
def concept_picker(concept_names, mo):
    concept_select = mo.ui.dropdown(
        options=concept_names, value="Legal destination", label="Probe concept"
    )
    concept_select
    return (concept_select,)


@app.cell
def probe_view(
    concept_delta_rows,
    concept_metric_rows,
    concept_select,
    html_table,
    layer_select,
    line_chart_html,
    metric_cards,
    mo,
):
    _concept_metrics = [
        row
        for row in concept_metric_rows
        if row["concept"] == concept_select.value
    ]
    _concept_deltas = [
        row for row in concept_delta_rows if row["concept"] == concept_select.value
    ]
    _concept_metric_name = _concept_metrics[0]["metric"]
    _concept_series = [
        {
            "name": model,
            "color": color,
            "points": [
                (row["layer"], row["score"])
                for row in _concept_metrics
                if row["model"] == model
            ],
        }
        for model, color in [("Raw BT4", "#5470c6"), ("Hero", "#d96c3f")]
    ]
    _concept_selected_delta = next(
        row for row in _concept_deltas if row["layer"] == layer_select.value
    )
    _direction_counts = {
        direction: sum(row["direction"] == direction for row in _concept_deltas)
        for direction in ["Hero higher", "Raw higher", "unresolved"]
    }
    _concept_selected_scores = [
        row for row in _concept_metrics if row["layer"] == layer_select.value
    ]
    mo.vstack(
        [
            mo.md(
                "## Linear concept probes: what is decodable?\n\nFit=1,024, selection=256, evaluation=512; groups are disjoint and the frozen test split remains unopened. The curve is the held-out primary metric. The selected-layer delta uses a paired per-example metric, so its scale may differ from the curve."
            ),
            line_chart_html(_concept_series, y_label=_concept_metric_name),
            metric_cards(
                [
                    (
                        f"{_concept_selected_delta['hero_minus_raw']:+.4f}",
                        f"layer {layer_select.value} Hero − Raw ({_concept_selected_delta['paired_metric']})",
                    ),
                    (
                        f"[{_concept_selected_delta['ci_low']:+.4f}, {_concept_selected_delta['ci_high']:+.4f}]",
                        "paired 95% interval",
                    ),
                    (
                        _concept_selected_delta["direction"],
                        "selected-layer direction",
                    ),
                    (_direction_counts["Hero higher"], "layers Hero higher"),
                    (_direction_counts["Raw higher"], "layers Raw higher"),
                    (_direction_counts["unresolved"], "layers unresolved"),
                ]
            ),
            mo.Html(
                html_table(
                    [
                        {
                            "model": row["model"],
                            _concept_metric_name: f"{row['score']:.4f}",
                        }
                        for row in _concept_selected_scores
                    ],
                    [
                        ("model", "Model"),
                        (
                            _concept_metric_name,
                            f"Layer {layer_select.value} {_concept_metric_name}",
                        ),
                    ],
                )
            ),
            mo.Html(
                '<div class="insight"><b>Reliability gate:</b> all 120 primary model/layer/concept scores beat both a label-permutation control and the train-only frequency or mean control.</div>'
            ),
            mo.Html(
                '<div class="null"><b>Mixed semantic picture:</b> legal destinations usually favor Hero; opponent attack count is mixed; piece identity resolves at only four layers and favors Raw at three of them. Better policy does not imply every concept becomes more linearly accessible.</div>'
            ),
        ]
    )
    return


@app.cell
def lookahead_data(artifacts):
    lookahead_metric_rows = artifacts["synthesis"]["datasets"]["lookahead_metrics"]
    lookahead_delta_rows = artifacts["synthesis"]["datasets"]["lookahead_deltas"]
    lookahead_targets = [
        "Third-ply destination",
        "Third-ply source | predicted destination",
        "Third-ply joint move",
    ]
    return lookahead_delta_rows, lookahead_metric_rows, lookahead_targets


@app.cell
def lookahead_picker(lookahead_targets, mo):
    lookahead_select = mo.ui.dropdown(
        options=lookahead_targets,
        value="Third-ply destination",
        label="Lookahead target",
    )
    lookahead_select
    return (lookahead_select,)


@app.cell
def lookahead_view(
    layer_select,
    line_chart_html,
    lookahead_delta_rows,
    lookahead_metric_rows,
    lookahead_select,
    metric_cards,
    mo,
):
    _lookahead_metrics = [
        row
        for row in lookahead_metric_rows
        if row["target"] == lookahead_select.value
    ]
    _lookahead_deltas = [
        row
        for row in lookahead_delta_rows
        if row["target"] == lookahead_select.value
    ]
    _lookahead_series = [
        {
            "name": model,
            "color": color,
            "points": [
                (row["layer"], row["accuracy"])
                for row in _lookahead_metrics
                if row["model"] == model
            ],
        }
        for model, color in [("Raw BT4", "#5470c6"), ("Hero", "#d96c3f")]
    ]
    _lookahead_selected_delta = next(
        row for row in _lookahead_deltas if row["layer"] == layer_select.value
    )
    _lookahead_resolved = [
        row for row in _lookahead_deltas if row["interval_excludes_zero"]
    ]
    _lookahead_peaks = {
        model: max(
            (row for row in _lookahead_metrics if row["model"] == model),
            key=lambda row: row["accuracy"],
        )
        for model in ["Raw BT4", "Hero"]
    }
    mo.vstack(
        [
            mo.md(
                "## Lookahead probes: a strong null against the exciting story\n\nThese probes decode the third ply of a public puzzle continuation (`future_ply_zero_indexed = 2`). Both models improve late in the stack."
            ),
            line_chart_html(
                _lookahead_series,
                y_label="held-out accuracy",
                y_domain=(0.0, 0.82),
            ),
            metric_cards(
                [
                    (
                        f"{_lookahead_selected_delta['hero_minus_raw']:+.4f}",
                        f"layer {layer_select.value} Hero − Raw",
                    ),
                    (
                        f"[{_lookahead_selected_delta['ci_low']:+.4f}, {_lookahead_selected_delta['ci_high']:+.4f}]",
                        "paired 95% interval",
                    ),
                    (
                        _lookahead_selected_delta["direction"],
                        "selected-layer direction",
                    ),
                    (
                        f"{_lookahead_peaks['Raw BT4']['accuracy']:.3f} @ L{_lookahead_peaks['Raw BT4']['layer']}",
                        "Raw peak",
                    ),
                    (
                        f"{_lookahead_peaks['Hero']['accuracy']:.3f} @ L{_lookahead_peaks['Hero']['layer']}",
                        "Hero peak",
                    ),
                    (
                        len(_lookahead_resolved),
                        "resolved layer deltas for this target",
                    ),
                ]
            ),
            mo.Html(
                '<div class="null"><b>Result:</b> there is no systematic Hero advantage. Destination accuracy resolves only at layer 6, where Hero is worse; source-conditional and joint accuracy resolve only at layer 14, where Hero is better.</div>'
            ),
            mo.Html(
                '<div class="danger"><b>Benchmark boundary:</b> puzzle continuation is not a fresh Stockfish principal variation and cannot by itself demonstrate internal search.</div>'
            ),
        ]
    )
    return


@app.cell
def causal_probe_data(artifacts):
    causal_probe_rows = []
    for _layer in artifacts["probes"]["layers"]:
        for _model in ["raw", "hero"]:
            _causal = artifacts["probes"]["causal_lookahead"][str(_layer)][_model][
                "probe_direction_policy_derivative"
            ]
            causal_probe_rows.append(
                {
                    "layer": _layer,
                    "model": _model,
                    "mean": _causal["mean"],
                    "lower": _causal["lower"],
                    "upper": _causal["upper"],
                    "resolved": _causal["lower"] > 0 or _causal["upper"] < 0,
                }
            )
    return (causal_probe_rows,)


@app.cell
def causal_probe_view(
    causal_probe_rows,
    html_table,
    layer_select,
    line_chart_html,
    metric_cards,
    mo,
):
    _causal_series = [
        {
            "name": model.title(),
            "color": color,
            "points": [
                (row["layer"], row["mean"])
                for row in causal_probe_rows
                if row["model"] == model
            ],
        }
        for model, color in [("raw", "#5470c6"), ("hero", "#d96c3f")]
    ]
    _causal_selected = [
        row for row in causal_probe_rows if row["layer"] == layer_select.value
    ]
    _causal_resolved = [row for row in causal_probe_rows if row["resolved"]]
    mo.vstack(
        [
            mo.md(
                "## Causal steering: decodable does not mean used\n\nAt each layer and model, the residual is moved one unit along the learned third-ply probe direction. We measure the derivative of the current policy target's log-odds."
            ),
            line_chart_html(_causal_series, y_label="policy log-odds derivative"),
            mo.Html(
                html_table(
                    [
                        {
                            "model": row["model"].title(),
                            "mean": f"{row['mean']:+.6f}",
                            "95% interval": f"[{row['lower']:+.6f}, {row['upper']:+.6f}]",
                            "resolved": row["resolved"],
                        }
                        for row in _causal_selected
                    ],
                    [
                        ("model", "Model"),
                        ("mean", f"Layer {layer_select.value} mean"),
                        ("95% interval", "95% interval"),
                        ("resolved", "Excludes zero"),
                    ],
                )
            ),
            metric_cards(
                [
                    (
                        f"{len(_causal_resolved)} / {len(causal_probe_rows)}",
                        "policy intervals excluding zero",
                    ),
                    ("30 / 30", "probe objective moved correctly"),
                    ("26 / 30", "random policy controls contained zero"),
                    ("16", "positions per model/layer"),
                ]
            ),
            mo.Html(
                '<div class="null"><b>Weak causal-use evidence:</b> only Raw layer 2 resolves (mean +0.00252, 95% interval +0.00031 to +0.00558). Moving the probe objective is easy; reliably moving the policy is not.</div>'
            ),
            mo.Html(
                '<div class="danger"><b>Control caveat:</b> four of 30 random-direction policy intervals also excluded zero. This is not evidence for an internal search algorithm.</div>'
            ),
        ]
    )
    return


@app.cell
def sparse_data(artifacts):
    transcoder = artifacts["transcoder_dev64"]
    transcoder_rows = []
    for _model in ["raw", "hero"]:
        _reconstruction = transcoder["reconstruction"][_model]
        _policy = transcoder["policy_replacement"][_model]
        transcoder_rows.append(
            {
                "model": _model.title(),
                "normalized MSE": f"{_reconstruction['normalized_mse']:.4f}",
                "cosine": f"{_reconstruction['cosine']:.4f}",
                "policy JS": f"{_policy['js_divergence']['mean']:.6f}",
                "top-1 agreement": f"{100 * _policy['top1_agreement']['mean']:.2f}%",
            }
        )

    lorsa_transfer = artifacts["lorsa_transfer"]
    lorsa_bootstrap = artifacts["lorsa_bootstrap"]
    lorsa_compact = artifacts["lorsa_compact"]
    lorsa_compact_summary = lorsa_compact["notebook_summary"]
    lorsa_token = artifacts["lorsa_token"]
    lorsa_token_pairs = artifacts["lorsa_token_selection"]["pairs"]
    lorsa_diagnostic_pairs = artifacts["lorsa_token_v1_selection"]["pairs"]

    _raw_stage = lorsa_transfer["source_stage"]
    _hero_stage = lorsa_transfer["transfer_stage"]["hero"]
    lorsa_transfer_rows = []
    for _model, _stage in [("Raw", _raw_stage), ("Hero", _hero_stage)]:
        lorsa_transfer_rows.append(
            {
                "model": _model,
                "normalized MSE": f"{_stage['reconstruction']['normalized_mse']:.6f}",
                "cosine": f"{_stage['reconstruction']['cosine']:.6f}",
                "policy JS": f"{_stage['policy_replacement']['js_divergence']['mean']:.6f}",
                "top-1": f"{_stage['policy_replacement']['top1_agreement']['true_count']}/512",
            }
        )


    def _paired_row(metric, payload, interpretation):
        interval = payload["hero_minus_raw_mean_percentile_ci95"]
        return {
            "metric": metric,
            "Raw mean": f"{payload['raw_mean']:.6f}",
            "Hero mean": f"{payload['hero_mean']:.6f}",
            "Hero − Raw": f"{payload['hero_minus_raw_mean']:+.6f}",
            "95% paired CI": f"[{interval['lower']:+.6f}, {interval['upper']:+.6f}]",
            "reading": interpretation,
        }


    lorsa_bootstrap_rows = [
        _paired_row(
            "per-position normalized MSE",
            lorsa_bootstrap["reconstruction_normalized_mse"]["mean_per_position"],
            "unresolved",
        ),
        _paired_row(
            "reconstruction cosine",
            lorsa_bootstrap["reconstruction_cosine_mean_per_position"],
            "unresolved",
        ),
        _paired_row(
            "policy JS after replacement",
            lorsa_bootstrap["policy_js_mean_per_position"],
            "small resolved degradation",
        ),
        _paired_row(
            "native↔LoRSA head-mean pattern JS",
            lorsa_bootstrap["native_vs_lorsa_head_mean_pattern_js_mean_per_position"],
            "smaller on Hero",
        ),
    ]
    return (
        lorsa_bootstrap,
        lorsa_bootstrap_rows,
        lorsa_compact,
        lorsa_compact_summary,
        lorsa_diagnostic_pairs,
        lorsa_token,
        lorsa_token_pairs,
        lorsa_transfer,
        lorsa_transfer_rows,
        transcoder,
        transcoder_rows,
    )


@app.cell
def sparse_view(
    html_table,
    lorsa_bootstrap,
    lorsa_bootstrap_rows,
    lorsa_transfer,
    lorsa_transfer_rows,
    metric_cards,
    mo,
    transcoder,
    transcoder_rows,
):
    _transcoder_support = transcoder["support_transfer"]
    _lorsa_diff = lorsa_transfer["transfer_stage"]["raw_hero"]
    _lorsa_support = _lorsa_diff["support_transfer"]
    _top1 = lorsa_bootstrap["policy_top1_contingency"]
    mo.vstack(
        [
            mo.md(
                "## Sparse replacement as a model diff\n\nTwo distinct published artifacts are kept separate: an earlier **MLP transcoder** result, then the final **LoRSA attention replacement** at layer 14."
            ),
            mo.md("### MLP transcoder — earlier 64-position compatibility result"),
            mo.Html(
                html_table(
                    transcoder_rows,
                    [
                        ("model", "Model"),
                        ("normalized MSE", "Normalized MSE"),
                        ("cosine", "Reconstruction cosine"),
                        ("policy JS", "Mean policy JS"),
                        ("top-1 agreement", "Top-1 agreement"),
                    ],
                )
            ),
            mo.Html(
                f'<div class="null"><b>Boundary:</b> this is an MLP transcoder, not LoRSA. Its small sample had {100 * _transcoder_support["first_dead_feature_fraction"]:.1f}% dead Raw features, so it is compatibility evidence only.</div>'
            ),
            mo.md("### Published LoRSA L14 — 512 development positions"),
            mo.Html(
                html_table(
                    lorsa_transfer_rows,
                    [
                        ("model", "Model"),
                        ("normalized MSE", "Normalized MSE"),
                        ("cosine", "Reconstruction cosine"),
                        ("policy JS", "Mean policy JS"),
                        ("top-1", "Top-1 preserved"),
                    ],
                )
            ),
            metric_cards(
                [
                    (f"{_lorsa_diff['degradation']['normalized_mse_ratio_hero_over_raw']:.4f}×", "Hero / Raw global NMSE"),
                    (f"{_lorsa_support['mean_support_jaccard']:.3f}", "Raw/Hero support Jaccard"),
                    (f"{_lorsa_support['mean_resolved_activation_correlation']:.3f}", "resolved activation correlation"),
                    (f"{_lorsa_support['resolved_activation_correlations']:,}", "resolved features"),
                    ("passed", "Raw source gate"),
                    ("$0", "local provider cost"),
                ]
            ),
            mo.md("#### Paired position-level uncertainty (20,000 group bootstraps)"),
            mo.Html(
                html_table(
                    lorsa_bootstrap_rows,
                    [
                        ("metric", "Metric"),
                        ("Raw mean", "Raw"),
                        ("Hero mean", "Hero"),
                        ("Hero − Raw", "Δ"),
                        ("95% paired CI", "95% CI"),
                        ("reading", "Reading"),
                    ],
                )
            ),
            mo.Html(
                f'<div class="insight"><b>Positive result:</b> the frozen Raw-trained LoRSA remains high-fidelity on Hero; both arms preserve 502/512 top moves. Paired failures still move: {_top1["raw_only_preserved"]} Raw-only and {_top1["hero_only_preserved"]} Hero-only, with exact McNemar p={_top1["paired_exact_test"]["p_value"]:.1f}.</div>'
            ),
            mo.Html(
                '<div class="null"><b>Resolved negative result:</b> replacement policy JS is slightly worse on Hero (Δ +0.000612; 95% CI +0.000261 to +0.000999). NMSE and cosine deltas remain unresolved. Support Jaccard 0.584 and correlation 0.673 also reject a simplistic “nothing changed” story.</div>'
            ),
            mo.md(
                "Native BT4 uses 32 attention heads while LoRSA learns a different head count; native↔LoRSA pattern comparisons therefore use head-mean distributions. Fidelity is not feature meaning."
            ),
        ]
    )
    return


@app.cell
def compact_semantics_view(
    html_table,
    lorsa_compact,
    lorsa_compact_summary,
    metric_cards,
    mo,
):
    _compact_overall = lorsa_compact_summary["overall"]
    _compact_directions = []
    for _direction in ["positive", "negative"]:
        _row = lorsa_compact_summary["by_selection_direction"][_direction]
        _compact_directions.append(
            {
                "fit direction": _direction,
                "selected": _row["selected"],
                "Raw replicated": f"{_row['raw_fit_direction_replicated']}/{_row['raw_supported']}",
                "Hero transferred": f"{_row['hero_fit_direction_transferred']}/{_row['hero_supported']}",
                "Raw/Hero agree": f"{_row['raw_hero_direction_agreement']}/{_row['paired_supported']}",
            }
        )
    _compact_hero_joint_one = sum(
        _row["selection_direction"] == "positive"
        and _row["hero_direction_transferred_from_raw_fit"]
        and _row["hero_evaluation_joint_peak_support"] == 1
        for _row in lorsa_compact["semantic_analysis"]["pairs"]
    )
    _compact_coverage = lorsa_compact_summary["coverage"]
    mo.vstack(
        [
            mo.md(
                "## Compact peak semantics — useful, but censored\n\nThis zero-rerun adapter analyzes the one retained peak square for each reported top feature. It was the right cheap diagnostic before the exact-token pass, but it cannot recover inactive features or non-peak active tokens."
            ),
            metric_cards(
                [
                    (_compact_overall["selected"], "selected pairs"),
                    (f"{_compact_overall['raw_fit_direction_replicated']}/{_compact_overall['raw_supported']}", "Raw replicated / supported"),
                    (f"{_compact_overall['hero_fit_direction_transferred']}/{_compact_overall['hero_supported']}", "Hero transferred / supported"),
                    (f"{_compact_overall['raw_hero_direction_agreement']}/{_compact_overall['paired_supported']}", "Raw/Hero direction agreement"),
                    (f"{100 * _compact_coverage['raw_evaluation']['observed_feature_fraction']:.2f}%", "Raw observed feature coverage"),
                    (f"{100 * _compact_coverage['hero_evaluation']['observed_feature_fraction']:.2f}%", "Hero observed feature coverage"),
                ]
            ),
            mo.Html(
                html_table(
                    _compact_directions,
                    [
                        ("fit direction", "Raw-fit direction"),
                        ("selected", "Selected"),
                        ("Raw replicated", "Raw heldout"),
                        ("Hero transferred", "Hero heldout"),
                        ("Raw/Hero agree", "Paired agreement"),
                    ],
                )
            ),
            mo.Html(
                f'<div class="danger"><b>Do not headline 97.4% as inferential.</b> The evaluation gate is feature support ≥8 only; it does not require heldout direction-appropriate joint/opportunity evidence. {_compact_hero_joint_one} replicated Hero-positive rows have joint peak count exactly one.</div>'
            ),
            mo.Html(
                '<div class="null"><b>Permitted claim:</b> descriptive peak-location association under heavy censoring. Not token-level semantics, feature causality, or Raw/Hero semantic equivalence.</div>'
            ),
        ]
    )
    return


@app.cell
def token_semantics_view(
    html_table,
    lorsa_token,
    lorsa_token_pairs,
    metric_cards,
    mo,
):
    def _token_pair(feature, concept, class_name):
        return next(
            _row
            for _row in lorsa_token_pairs
            if _row["feature_index"] == feature
            and _row["concept"] == concept
            and _row["class_name"] == class_name
        )


    _positive_specs = [
        (10843, "piece_code", "our_knight"),
        (5429, "piece_code", "our_queen"),
        (10784, "attacked_undefended_ours", "present"),
    ]
    _positive_rows = []
    for _feature, _concept, _class_name in _positive_specs:
        _row = _token_pair(_feature, _concept, _class_name)
        _positive_rows.append(
            {
                "feature": _feature,
                "association": f"{_concept} = {_class_name}",
                "Raw log2 lift": f"{_row['raw_heldout']['evaluation_log2_lift']:+.3f}",
                "Hero log2 lift": f"{_row['hero_heldout']['evaluation_log2_lift']:+.3f}",
                "Raw/Hero joint": f"{_row['raw_heldout']['evaluation_joint_support']}/{_row['hero_heldout']['evaluation_joint_support']}",
            }
        )

    _raw_sem = lorsa_token["raw_heldout"]
    _hero_sem = lorsa_token["hero_heldout"]
    _model_sem = lorsa_token["raw_hero_model_diff"]
    mo.vstack(
        [
            mo.md(
                "## Exact token semantics — authoritative descriptive pass\n\nSelection uses 256 Raw-fit development positions. The same 430 balanced pairs (215 positive, 215 negative) are then evaluated without reselection on 256 group-disjoint Raw and Hero heldout positions."
            ),
            metric_cards(
                [
                    (f"{_raw_sem['direction_replicated_count']}/{_raw_sem['supported_pair_count']}", "Raw directions replicate"),
                    (f"{_hero_sem['direction_replicated_count']}/{_hero_sem['supported_pair_count']}", "Hero directions transfer"),
                    (f"{_model_sem['raw_hero_direction_agreement_count']}/{_model_sem['both_supported_pair_count']}", "Raw/Hero directions agree"),
                    (f"{_model_sem['raw_hero_heldout_lift_pearson_supported']:.4f}", "heldout lift Pearson r"),
                    (f"{_model_sem['mean_absolute_hero_minus_raw_lift_supported']:.3f}", "mean |Hero − Raw| lift"),
                    (f"{_raw_sem['unsupported_pair_count']}/{_hero_sem['unsupported_pair_count']}", "Raw/Hero unsupported"),
                ]
            ),
            mo.md("### Strong replicated associations"),
            mo.Html(
                html_table(
                    _positive_rows,
                    [
                        ("feature", "Feature"),
                        ("association", "Token association"),
                        ("Raw log2 lift", "Raw heldout"),
                        ("Hero log2 lift", "Hero heldout"),
                        ("Raw/Hero joint", "Observed joint support"),
                    ],
                )
            ),
        ]
    )
    return


@app.cell
def token_semantics_failures_view(
    html_table,
    lorsa_diagnostic_pairs,
    lorsa_token,
    lorsa_token_pairs,
    mo,
):
    def _token_failure_pair(feature, concept, class_name):
        return next(
            _row
            for _row in lorsa_token_pairs
            if _row["feature_index"] == feature
            and _row["concept"] == concept
            and _row["class_name"] == class_name
        )


    _failure_specs = [
        (6661, "pinned_ours", "present", "supported direction disagreement"),
        (4516, "attack_count_theirs", "two", "supported direction disagreement"),
        (5851, "pinned_theirs", "absent", "near-zero nonreplication"),
    ]
    _failure_rows = []
    for _feature, _concept, _class_name, _reading in _failure_specs:
        _row = _token_failure_pair(_feature, _concept, _class_name)
        _failure_rows.append(
            {
                "feature": _feature,
                "association": f"{_concept} = {_class_name}",
                "Raw log2 lift": f"{_row['raw_heldout']['evaluation_log2_lift']:+.3f}",
                "Hero log2 lift": f"{_row['hero_heldout']['evaluation_log2_lift']:+.3f}",
                "reading": _reading,
            }
        )
    _diagnostic_negative = [
        _row for _row in lorsa_diagnostic_pairs if _row["fit_direction"] == "negative"
    ]
    _diagnostic_ungated = sum(
        (_row.get("fit_expected_joint_support") or 0) < 5
        for _row in _diagnostic_negative
    )
    _raw_failure_summary = lorsa_token["raw_heldout"]
    _hero_failure_summary = lorsa_token["hero_heldout"]
    _split_failure_summary = lorsa_token["position_split"]
    mo.vstack(
        [
            mo.md("### Explicit nonreplications and model disagreements"),
            mo.Html(
                html_table(
                    _failure_rows,
                    [
                        ("feature", "Feature"),
                        ("association", "Token association"),
                        ("Raw log2 lift", "Raw heldout"),
                        ("Hero log2 lift", "Hero heldout"),
                        ("reading", "Result"),
                    ],
                )
            ),
            mo.Html(
                f'<div class="danger"><b>Why the first selector was rejected:</b> {_diagnostic_ungated}/{len(lorsa_diagnostic_pairs)} v1 pairs were negative exclusions chosen without the expected-joint opportunity gate. Its apparent perfect replication was a selection pathology, so v1 is retained only as diagnostic evidence.</div>'
            ),
            mo.Html(
                f'<div class="insight"><b>What survived the fix:</b> {_raw_failure_summary["direction_replicated_count"]}/{_raw_failure_summary["supported_pair_count"]} Raw and {_hero_failure_summary["direction_replicated_count"]}/{_hero_failure_summary["supported_pair_count"]} Hero directions replicate/transfer under direction-appropriate evidence gates. Fit/heldout groups are disjoint: {_split_failure_summary["fit_group_count"]} + {_split_failure_summary["heldout_group_count"]}.</div>'
            ),
        ]
    )
    return


@app.cell
def token_claim_gates_view(html_table, lorsa_token, mo):
    _token_false_claim_rows = [
        {"claim": "feature causality", "permitted": lorsa_token["claim_gates"]["feature_causality_permitted"], "why not": "no ablation/insertion or matched controls"},
        {"claim": "Raw/Hero semantic equivalence", "permitted": lorsa_token["claim_gates"]["raw_hero_semantic_equivalence_permitted"], "why not": "direction agreement is descriptive"},
        {"claim": "internal search", "permitted": lorsa_token["claim_gates"]["internal_search_claim_permitted"], "why not": "token concepts do not identify an algorithm"},
        {"claim": "confirmatory test result", "permitted": lorsa_token["claim_gates"]["confirmatory_test_claim_permitted"], "why not": "development-only; zero test rows"},
        {"claim": "multiplicity-adjusted discovery", "permitted": lorsa_token["interpretation_contract"]["multiple_testing_adjusted"], "why not": "430 frozen exploratory pairs"},
    ]
    mo.vstack(
        [
            mo.md("### Hard claim gates"),
            mo.Html(
                html_table(
                    _token_false_claim_rows,
                    [
                        ("claim", "Claim"),
                        ("permitted", "Permitted"),
                        ("why not", "Why not"),
                    ],
                )
            ),
            mo.Html(
                '<div class="null"><b>Estimand:</b> class prevalence among active feature tokens divided by the token baseline. Activation-weighted lift is also descriptive and depends on LoRSA decoder amplitude units.</div>'
            ),
        ]
    )
    return


@app.cell
def causal_feature_data(artifacts):
    causal = artifacts["lorsa_causal"]
    _summary = causal["summary"]
    _pair_order = [
        "piece_code:2:feature_10843",
        "piece_code:5:feature_5429",
        "attacked_undefended_ours:1:feature_10784",
        "pinned_ours:1:feature_6661",
        "pinned_theirs:0:feature_5851",
        "attack_count_theirs:2:feature_4516",
    ]
    _labels = {
        "piece_code:2:feature_10843": "own knight",
        "piece_code:5:feature_5429": "own queen",
        "attacked_undefended_ours:1:feature_10784": "attacked + undefended ours",
        "pinned_ours:1:feature_6661": "pinned ours",
        "pinned_theirs:0:feature_5851": "pinned theirs absent (null)",
        "attack_count_theirs:2:feature_4516": "two enemy attackers",
    }
    causal_rows = []
    causal_delta_rows = []
    for _pair in _pair_order:
        _raw = _summary["arms"]["RR"][_pair]
        _hero = _summary["arms"]["HH"][_pair]
        _raw_effect = _raw["active_position_ablation_specificity"]["paired_difference"]
        _hero_effect = _hero["active_position_ablation_specificity"]["paired_difference"]
        _raw_pass = _raw["exploratory_causal_feature_policy_gate"]["passed"]
        _hero_pass = _hero["exploratory_causal_feature_policy_gate"]["passed"]
        if _raw_pass and _hero_pass:
            _result = "passes both"
        elif _pair.endswith("feature_6661"):
            _result = "falsified: target < controls"
        elif _pair.endswith("feature_5851"):
            _result = "null retained"
        else:
            _result = "underpowered (n=2)"
        causal_rows.append(
            {
                "concept": _labels[_pair],
                "feature": _raw["target"]["feature_index"],
                "Raw n": _raw["natural_feature_active_position_count"],
                "Raw target-control TV": f"{_raw_effect['mean_difference']:+.6f} [{_raw_effect['lower']:+.6f}, {_raw_effect['upper']:+.6f}]",
                "Hero n": _hero["natural_feature_active_position_count"],
                "Hero target-control TV": f"{_hero_effect['mean_difference']:+.6f} [{_hero_effect['lower']:+.6f}, {_hero_effect['upper']:+.6f}]",
                "Holm p max": f"{max(_raw['active_position_ablation_specificity']['sign_flip_pvalue_holm'], _hero['active_position_ablation_specificity']['sign_flip_pvalue_holm']):.6f}",
                "result": _result,
            }
        )
        if _raw_pass and _hero_pass:
            _cross = _summary["cross_model"][_pair]
            _delta = _cross[
                "hero_minus_raw_ablation_total_variation_active_union"
            ]
            causal_delta_rows.append(
                {
                    "concept": _labels[_pair],
                    "feature": _raw["target"]["feature_index"],
                    "Hero-Raw effect": f"{_delta['mean_difference']:+.6f} [{_delta['lower']:+.6f}, {_delta['upper']:+.6f}]",
                    "median delta cosine": f"{_cross['raw_hero_policy_delta_cosine_active_union']['median']:.4f}",
                }
            )
    return causal, causal_delta_rows, causal_rows


@app.cell
def causal_feature_view(
    artifacts,
    causal,
    causal_delta_rows,
    causal_rows,
    html_table,
    metric_cards,
    mo,
):
    _manifest = artifacts["lorsa_causal_manifest"]
    mo.vstack(
        [
            mo.md(
                "## Causal sparse-feature validation\n\nFor each target, three seeded random decoder directions receive the **same target activation scalar, token support/insertion token, and decoder norm**. Only direction changes. Natural ablation is the multiplicity-corrected primary test; insertion and dose curves are supporting diagnostics."
            ),
            metric_cards(
                [
                    (
                        f"{causal['claim_gates']['exploratory_specific_causal_gate_pass_count']}/12",
                        "Raw/Hero gates pass",
                    ),
                    ("3/6", "features pass bilaterally"),
                    (causal["position_count"], "held-out dev roots"),
                    ("20,000", "paired resamples"),
                    (
                        f"{_manifest['resource_usage']['wall_seconds'] / 60:.1f} min",
                        "local wall time",
                    ),
                    (
                        f"{_manifest['resource_usage']['cuda_peak_allocated_bytes'] / 2**30:.2f} GiB",
                        "peak CUDA allocated",
                    ),
                ]
            ),
            mo.Html(
                html_table(
                    causal_rows,
                    [
                        ("concept", "Exploratory concept"),
                        ("feature", "Feature"),
                        ("Raw n", "Raw active n"),
                        ("Raw target-control TV", "Raw specificity [95% CI]"),
                        ("Hero n", "Hero active n"),
                        ("Hero target-control TV", "Hero specificity [95% CI]"),
                        ("Holm p max", "max Holm p"),
                        ("result", "Strict result"),
                    ],
                )
            ),
            mo.Html(
                '<div class="insight"><b>Positive result:</b> features 10843 (own knight), 5429 (own queen), and 10784 (attacked/undefended ours) pass in Raw and Hero. Dose-1 insertion is target-direction-specific in both models.</div>'
            ),
            mo.Html(
                '<div class="danger"><b>Falsification:</b> feature 6661’s descriptive pinned-piece sign reversal does not survive intervention—matched random directions move policy more in both models. Feature 4516 is not estimable with two active positions.</div>'
            ),
            mo.md("### Same direction, changed strength"),
            mo.Html(
                html_table(
                    causal_delta_rows,
                    [
                        ("concept", "Concept"),
                        ("feature", "Feature"),
                        ("Hero-Raw effect", "Hero − Raw magnitude [95% CI]"),
                        ("median delta cosine", "Median policy-delta cosine"),
                    ],
                )
            ),
            mo.Html(
                '<div class="null"><b>Claim boundary:</b> this is causal use inside a frozen LoRSA attention-replacement path. It does not identify a native dense BT4 feature, prove the chess label is monosemantic, or establish input-concept causality. The reserved test split remains unopened.</div>'
            ),
        ]
    )
    return


@app.cell
def hero_lr_data(artifacts):
    hero_lr_audit = artifacts["hero_lr_audit"]
    _optimizers = hero_lr_audit["optimizer_movement"]["by_optimizer"]
    optimizer_update_rows = [
        {
            "optimizer": _name,
            "encoder leaves": _entry["leaf_count"],
            "parameters": f"{_entry['parameter_count']:,}",
            "unchanged": f"{100 * _entry['unchanged_fraction']:.3f}%",
            "delta energy": f"{100 * _entry['global_delta_energy_fraction']:.3f}%",
        }
        for _name, _entry in _optimizers.items()
    ]
    _movement = hero_lr_audit["component_movement"]
    _total_energy = _movement["all"]["delta_l2"] ** 2
    _component_order = [
        ("smolgen", "SmolGen"),
        ("embedding", "embedding"),
        ("attention", "attention"),
        ("mlp", "MLP"),
        ("layer_norm", "layer norms"),
        ("policy_head", "policy head"),
    ]
    encoder_component_rows = [
        {
            "component": _label,
            "parameters": f"{_movement[_key]['parameter_count']:,}",
            "unchanged": f"{100 * _movement[_key]['unchanged_fraction']:.3f}%",
            "relative L2": f"{100 * _movement[_key]['relative_delta_l2']:.4f}%",
            "delta energy": f"{100 * _movement[_key]['delta_l2'] ** 2 / _total_energy:.3f}%",
        }
        for _key, _label in _component_order
    ]
    _delta = hero_lr_audit["validation"]["ten_to_hundred_percent_delta"]
    validation_delta_rows = [
        {"metric": "root legal conditional CE", "change": f"{_delta['root_legal_conditional_ce']:+.4f}", "better": "down"},
        {"metric": "root legal top-1", "change": f"{_delta['root_legal_top1_accuracy']:+.4f}", "better": "up"},
        {"metric": "first legal mass", "change": f"{_delta['first_legal_mass']:+.4f}", "better": "up"},
        {"metric": "DFM CE", "change": f"{_delta['dfm_ce_loss']:+.4f}", "better": "down"},
        {"metric": "JEPA positive loss", "change": f"{_delta['jepa_positive_loss']:+.4f}", "better": "down"},
        {"metric": "WDL loss", "change": f"{_delta['wdl_loss']:+.4f}", "better": "down"},
    ]
    return (
        encoder_component_rows,
        hero_lr_audit,
        optimizer_update_rows,
        validation_delta_rows,
    )


@app.cell
def hero_lr_view(
    encoder_component_rows,
    hero_lr_audit,
    html_table,
    metric_cards,
    mo,
    optimizer_update_rows,
    validation_delta_rows,
):
    _schedule = hero_lr_audit["schedule"]
    _all = hero_lr_audit["component_movement"]["all"]
    _muon = hero_lr_audit["optimizer_movement"]["by_optimizer"]["muon"]
    mo.vstack(
        [
            mo.md(
                "## Was Hero’s encoder learning rate too low?\n\n**Probably directionally, but LR alone is not identified.** Hero used a 30× lower encoder peak LR while updating BF16 parameters and same-dtype optimizer moments directly, with no recorded FP32 master weights. Precision and LR are therefore confounded."
            ),
            metric_cards(
                [
                    (f"{_schedule['main_to_encoder_peak_ratio']:.0f}×", "main / encoder peak LR"),
                    (f"{100 * _all['unchanged_fraction']:.3f}%", "encoder values unchanged"),
                    (f"{100 * _muon['unchanged_fraction']:.3f}%", "Muon values unchanged"),
                    (f"{100 * _muon['global_delta_energy_fraction']:.3f}%", "delta energy in Muon leaves"),
                    (f"{100 * _all['relative_delta_l2']:.4f}%", "encoder relative L2 change"),
                    (_schedule["updates"], "one-epoch updates"),
                ]
            ),
            mo.md("### Where updates survived"),
            mo.Html(
                html_table(
                    optimizer_update_rows,
                    [
                        ("optimizer", "Optimizer"),
                        ("encoder leaves", "Encoder leaves"),
                        ("parameters", "Parameters"),
                        ("unchanged", "Bit-identical"),
                        ("delta energy", "Encoder delta energy"),
                    ],
                )
            ),
            mo.Html(
                html_table(
                    encoder_component_rows,
                    [
                        ("component", "Component"),
                        ("parameters", "Parameters"),
                        ("unchanged", "Bit-identical"),
                        ("relative L2", "Relative L2"),
                        ("delta energy", "Encoder delta energy"),
                    ],
                )
            ),
            mo.Html(
                '<div class="danger"><b>Key diagnosis:</b> 99.707% of Muon-managed encoder values are unchanged, and those matrices carry only 0.251% of encoder delta energy. Small BF16 in-place updates can round away instead of accumulating.</div>'
            ),
            mo.md("### Hero still learned during the retained trajectory"),
            mo.Html(
                html_table(
                    validation_delta_rows,
                    [
                        ("metric", "Metric, 10% → 100%"),
                        ("change", "Change"),
                        ("better", "Desired direction"),
                    ],
                )
            ),
            mo.Html(
                '<div class="null"><b>Do not overread this:</b> validation begins at 10%, not Raw step zero. Hero v1 is useful and learned; it is simply not a clean test of broad encoder adaptation. Raising LR under the same BF16 direct-update contract would not isolate the problem. FP32 masters/moments add at least 1.268 GiB, so 24-GiB training must pass a memory smoke with a 48-GiB fallback.</div>'
            ),
        ]
    )
    return


@app.cell
def next_steps_view(mo):
    mo.md("""
    ## Broader next steps

    1. **Fix precision first:** add FP32 master parameters and FP32 moments for BF16 compute weights, exact-resume serialization, and update-to-ULP telemetry.
    2. **Run a controlled successive-halving sweep:** BF16-direct `1/30` control versus FP32-master encoder ratios `1/30`, `1/10`, and `1/3`; add `1/1` only if `1/3` is stable and still starved. Smoke all arms, take all to ~2% epoch, then the best two to ~10%.
    3. **Promote on a Pareto gate:** update health, Raw-policy retention, legal CE/top-1, DFM, JEPA, WDL, legality, and paired arena—not training loss alone.
    4. **Route by lifecycle:** local 1660 Ti for tests and the full current LoRSA causal tier; Modal for short restartable training arms, with a memory-gated 24-GiB smoke and 48-GiB fallback; Runpod only for a selected long run after measured utilization and checkpoint recovery justify a warm pod.
    5. **Interpret the winner with the same frozen stack:** RR/HR/RH/HH model diff, update audit, L14 LoRSA transfer, exact semantics, and strict causal validation. Then localize features 10843, 5429, and 10784 through attention/MLP/policy paths and matched board counterfactuals.
    6. **Open test data once:** freeze model selection and analysis code before the multi-seed, game-clustered, multiplicity-corrected confirmation and licensed Stockfish-PV benchmark.

    The immediate engineering order is **FP32 master weights → short LR sweep → winner → repeated causal/model-diff audit**.
    """)
    return


@app.cell
def cost_data(artifacts):
    cost_audit = artifacts["cost"]
    cost_rows = [
        {
            "item": "Pre-powered setup",
            "dollars": cost_audit["pre_powered_run_metered_cost_dollars"],
        },
        {
            "item": "Powered staged pipeline",
            "dollars": cost_audit[
                "powered_pipeline_incremental_metered_cost_dollars"
            ],
        },
        {"item": "Total metered", "dollars": cost_audit["metered_cost_dollars"]},
        {
            "item": "Billed after credits",
            "dollars": cost_audit["billed_cost_after_credits_dollars"],
        },
    ]

    _transfer_manifest_cost = artifacts["lorsa_transfer_manifest"]
    _token_manifest_cost = artifacts["lorsa_token_manifest"]
    _causal_manifest_cost = artifacts["lorsa_causal_manifest"]
    _t4_manifest = artifacts["lorsa_modal_t4"]
    _l4_manifest = artifacts["lorsa_modal_l4"]
    _t4_rate = 0.00026968
    _l4_rate = 0.00032768
    lorsa_compute_rows = [
        {
            "workload": "Local final transfer (512 positions)",
            "device": "GTX 1660 Ti",
            "wall": f"{_transfer_manifest_cost['resource_usage']['wall_seconds']:.1f}s",
            "peak allocated": f"{_transfer_manifest_cost['resource_usage']['cuda_peak_allocated_bytes'] / 2**30:.2f} GiB",
            "cost": "$0 exact",
        },
        {
            "workload": "Local exact-token audit (3 streamed arms)",
            "device": "GTX 1660 Ti",
            "wall": f"{_token_manifest_cost['resource_usage']['wall_seconds']:.1f}s",
            "peak allocated": f"{_token_manifest_cost['resource_usage']['cuda_peak_allocated_bytes'] / 2**30:.2f} GiB",
            "cost": "$0 exact",
        },
        {
            "workload": "Local causal LoRSA v2 (256 roots, both models)",
            "device": "GTX 1660 Ti",
            "wall": f"{_causal_manifest_cost['resource_usage']['wall_seconds']:.1f}s",
            "peak allocated": f"{_causal_manifest_cost['resource_usage']['cuda_peak_allocated_bytes'] / 2**30:.2f} GiB",
            "cost": "$0 exact",
        },
        {
            "workload": "Remote LoRSA smoke (8 positions)",
            "device": "T4",
            "wall": f"{_t4_manifest['elapsed_seconds']:.1f}s",
            "peak allocated": f"{_t4_manifest['resource_usage']['cuda_peak_allocated_bytes'] / 2**30:.2f} GiB",
            "cost": f"${_t4_manifest['elapsed_seconds'] * _t4_rate:.4f} estimate",
        },
        {
            "workload": "Remote LoRSA smoke (8 positions)",
            "device": "L4",
            "wall": f"{_l4_manifest['elapsed_seconds']:.1f}s",
            "peak allocated": f"{_l4_manifest['resource_usage']['cuda_peak_allocated_bytes'] / 2**30:.2f} GiB",
            "cost": f"${_l4_manifest['elapsed_seconds'] * _l4_rate:.4f} estimate",
        },
    ]
    return cost_audit, cost_rows, lorsa_compute_rows


@app.cell
def cost_view(
    artifacts,
    cost_audit,
    cost_rows,
    horizontal_bars_html,
    html_table,
    lorsa_compute_rows,
    metric_cards,
    mo,
):
    _gpu = cost_audit["pipeline"]["t4_all_layer_run"]
    _t4_elapsed = artifacts["lorsa_modal_t4"]["elapsed_seconds"]
    _l4_elapsed = artifacts["lorsa_modal_l4"]["elapsed_seconds"]
    _speedup = _t4_elapsed / _l4_elapsed
    _relative_cost = (_l4_elapsed * 0.00032768) / (_t4_elapsed * 0.00026968)
    mo.vstack(
        [
            mo.md(
                "## Compute and cost: stage data before GPU allocation\n\nThe remote design downloads and hashes on CPU, verifies immutable Volume bundles, then starts the requested GPU. Result upload is CPU-only."
            ),
            metric_cards(
                [
                    (f"${cost_audit['metered_cost_dollars']:.3f}", "earlier workstream metered"),
                    (f"${cost_audit['billed_cost_after_credits_dollars']:.2f}", "earlier billed after credits"),
                    (f"{_gpu['gpu_elapsed_seconds'] / 60:.1f} min", "all-layer T4 elapsed"),
                    (f"{_gpu['peak_allocated_bytes'] / 2**30:.2f} GiB", "all-layer peak allocation"),
                    ("0 bytes", "GPU-stage network download"),
                    ("local", "bounded LoRSA default"),
                ]
            ),
            horizontal_bars_html(cost_rows, "dollars", "item", color="#16805d"),
            mo.md("### LoRSA scheduling measurements"),
            mo.Html(
                html_table(
                    lorsa_compute_rows,
                    [
                        ("workload", "Workload"),
                        ("device", "Device"),
                        ("wall", "Wall time"),
                        ("peak allocated", "Peak GPU allocation"),
                        ("cost", "Provider cost"),
                    ],
                )
            ),
            mo.Html(
                f'<div class="insight"><b>Scheduling result:</b> use the local 1660 Ti for bounded inference and semantic audits. In matched 8-position remote smokes, L4 was {_speedup:.3f}× faster but {_relative_cost:.3f}× the estimated T4 cost; use T4 for heavier restartable maps and L4 only when latency matters.</div>'
            ),
            mo.Html(
                '<div class="null"><b>Cost boundary:</b> T4/L4 dollar figures use frozen combined rates and are estimates, not exact attempt-level billing attribution. The smokes compare startup-dominated eight-position jobs, not full training throughput.</div>'
            ),
        ]
    )
    return


@app.cell
def evidence_view(artifacts, html_table, mo):
    mo.vstack(
        [
            mo.md(
                "## Evidence ladder\n\nDifferent methods answer different questions. `Evidence rank` below is the report's own calibrated label, not a universal statistical scale."
            ),
            mo.Html(
                html_table(
                    artifacts["synthesis"]["datasets"]["evidence_matrix"],
                    [
                        ("method", "Method"),
                        ("evidence_rank", "Evidence rank"),
                        ("finding", "Finding"),
                        ("scope", "Scope"),
                        ("main_limit", "Main limit"),
                    ],
                )
            ),
        ]
    )
    return


@app.cell
def does_not_show(artifacts):
    required_caveats = artifacts["synthesis"]["validation"]["required_caveats"]
    extra_boundaries = [
        "The 2×2 model-diff pilot has one placeholder game cluster; its collapsed cluster-bootstrap intervals are not inferential uncertainty.",
        "Whole-residual patching localizes mediation to a depth, not to an attention head, SmolGen component, MLP neuron, or sparse feature.",
        "High reconstruction fidelity and direction-stable token associations alone do not establish feature causality or Raw/Hero semantic equivalence.",
        "The causal v2 result applies inside the frozen LoRSA replacement path; it does not identify native dense features or establish input-concept causality.",
        "The compact peak report is censored: it sees one reported peak token, not every active token, and its evaluation gate is feature support only.",
        "The exact-token analysis is exploratory, has no multiplicity adjustment, and treats correlated board-square observations descriptively.",
        "Two supported Raw/Hero direction disagreements and two supported nonreplications are part of the result, not noise to hide.",
        "No reserved test row was selected or evaluated; confirmatory claims are not permitted.",
        "No result here demonstrates an internal search algorithm.",
    ]
    all_boundaries = required_caveats + extra_boundaries
    return (all_boundaries,)


@app.cell
def does_not_show_view(all_boundaries, html, mo):
    _boundary_items = "".join(
        f"<li style='margin:.42rem 0'>{html.escape(item)}</li>"
        for item in all_boundaries
    )
    mo.vstack(
        [
            mo.md("## What this does **not** show"),
            mo.Html(f'<div class="danger"><ol>{_boundary_items}</ol></div>'),
            mo.md(
                "### The shortest honest conclusion\n\nHero is a functionally different but severely update-constrained encoder whose late residual representation mediates its policy change. A frozen Raw-trained LoRSA remains high-fidelity, and three decoder directions have matched-control causal policy effects in both models. Sparse support geometry and effect strength still change, and explicit falsifications remain. This is **not yet a native dense-circuit explanation, input-concept causality, semantic equivalence, or evidence that Hero learned search**."
            ),
        ]
    )
    return


@app.cell
def sources_view(REPO_ROOT, html_table, mo, source_rows):
    _source_display = [
        {
            "artifact": row["artifact"],
            "schema": row["schema"],
            "size": f"{row['bytes'] / 1024:.1f} KiB",
            "path": row["path"],
        }
        for row in source_rows
    ]
    mo.vstack(
        [
            mo.md(
                "## Reproduce this view\n\nAll inputs are repository-relative JSON/JSONL artifacts. No large checkpoint is opened."
            ),
            mo.Html(
                html_table(
                    _source_display,
                    [
                        ("artifact", "Artifact"),
                        ("schema", "Schema"),
                        ("size", "Size"),
                        ("path", "Repo-relative path"),
                    ],
                )
            ),
            mo.md(
                f"""
    Notebook root resolved to `{REPO_ROOT}`.

    Launch from the repository root:

    ```bash
    uvx --from marimo==0.23.16 marimo edit research/notebooks/raw_hero_interpretability.py --no-token --sandbox
    ```

    Canonical private Railway bundles:

    - transfer v3: `dee7536374cd85c4573b8399f1d8696947bd63f98784c8b39729b161d4e6a55f`
    - paired bootstrap v3: `86e373d08e3dd464c3171fdce8adb970c178b4e94457d574e51d765e61ee4062`
    - compact peak semantics v4: `5ad94b63df56bba029424f6b127d05e95501f49e926ce19af818f4f282bb795e`
    - exact-token semantics v3: `a3f1f50b2ccabd7794368a2b3a9e30d8201fcd67aba92f1e5b898535b462100f`
    - causal LoRSA v2: `12e42f17402bfa1a05ee9e32e47f9e4afebab446cfc5403e90daf19a0abcf1c8`
    - exact 51-file source snapshot: `c53390977c6dad107ca20614dcb18bcfa44ea3e477987b9d58bc357f35ccdfa8`

    The checkpoint came from the published LoRSA work, but its upstream repository declares no license;
    checkpoint and derived artifact redistribution therefore stays private. Runtime dependency: **marimo only**.
    GPU and network are unnecessary after the isolated notebook runner is available.
    """
            ),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
