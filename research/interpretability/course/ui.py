"""Accessible HTML/SVG components for marimo lessons."""

from __future__ import annotations

import html
from typing import Iterable, Mapping, Sequence

from .catalog import ModuleSpec
from .evidence import ClaimCard


COURSE_CSS = """
<style>
:root { --ink:#162235; --muted:#5d6b7d; --blue:#3366a3; --orange:#a95f19;
  --green:#14735a; --purple:#7450a8; --red:#a33d43; --paper:#fbfcfe;
  --panel:#f2f5f9; --line:#d5dde8; }
.ci-hero {padding:1.15rem 1.3rem;border-radius:16px;color:white;
  background:linear-gradient(125deg,#162235,#304b71 65%,#75482e);}
.ci-hero h1{margin:0;font-size:1.8rem;line-height:1.15}.ci-hero p{margin:.5rem 0 0;color:#e9eef6}
.ci-badge{display:inline-block;border:2px solid currentColor;border-radius:999px;padding:.25rem .65rem;
  font-size:.78rem;font-weight:800;letter-spacing:.04em;margin:.3rem .35rem .3rem 0}
.ci-toy{color:#7450a8;background:#f3edfb}.ci-snapshot{color:#14735a;background:#eaf7f2}
.ci-source{color:#3366a3;background:#eaf2fa}.ci-card-grid{display:grid;
  grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:.65rem}
.ci-card{background:var(--panel);border:1px solid var(--line);border-radius:11px;padding:.72rem .82rem}
.ci-card strong{display:block;font-size:1.3rem;color:var(--ink)}.ci-card span{font-size:.8rem;color:var(--muted)}
.ci-callout{border-left:5px solid var(--blue);background:#edf4fb;border-radius:9px;padding:.72rem .9rem;margin:.55rem 0}
.ci-positive{border-color:var(--green);background:#eaf7f2}.ci-limit{border-color:var(--orange);background:#fff5e8}
.ci-danger{border-color:var(--red);background:#fff0f1}.ci-neutral{border-color:var(--purple);background:#f3edfb}
.ci-claim{border:1px solid var(--line);border-radius:12px;overflow:hidden;margin:.6rem 0;background:white}
.ci-claim h4{margin:0;padding:.6rem .8rem;background:#e9eef5}.ci-claim dl{display:grid;
  grid-template-columns:minmax(125px,22%) 1fr;margin:0}.ci-claim dt,.ci-claim dd{margin:0;padding:.42rem .65rem;border-top:1px solid #e5eaf1}
.ci-claim dt{font-weight:700;color:#42526a}.ci-table{width:100%;border-collapse:collapse;font-size:.86rem}
.ci-table th{background:#e9eef5;text-align:left}.ci-table th,.ci-table td{padding:.45rem .55rem;border-bottom:1px solid var(--line);vertical-align:top}
.ci-board{border-collapse:collapse;border:2px solid #5f4937}.ci-board td{position:relative;width:2.35rem;height:2.35rem;text-align:center;
  font-size:1.65rem;line-height:1}.ci-light{background:#eee3cb}.ci-dark{background:#9a795c;color:#111}
.ci-coord{font-size:.62rem;color:#111;background:rgba(255,255,255,.72);padding:0 .1rem;vertical-align:bottom}
.ci-highlight-marker{position:absolute;right:.08rem;top:.08rem;border:1px solid #111;border-radius:50%;
  background:#ffeb3b;color:#111;font:800 .68rem/1rem sans-serif;width:1rem;height:1rem}
.ci-caption{font-size:.8rem;color:var(--muted);margin-top:.35rem}
.ci-visually-hidden{position:absolute;width:1px;height:1px;padding:0;margin:-1px;overflow:hidden;
  clip:rect(0,0,0,0);white-space:nowrap;border:0}
</style>
"""


def committed_form(
    mo,
    fields: Mapping[str, object],
    *,
    submit_label: str,
    min_words: int = 0,
    allow_empty: Iterable[str] = (),
):
    """Create an immutable submit-before-reveal dictionary form.

    The returned form has value ``None`` until a valid submission. Subsequent
    edits do not alter the submitted value until the learner submits again,
    which makes prediction feedback auditable instead of merely hidden.
    """

    allowed = set(allow_empty)

    def validate(value):
        if not isinstance(value, Mapping):
            return "Complete and submit the form before revealing feedback."
        missing = set(fields) - set(value)
        unexpected = set(value) - set(fields)
        if missing or unexpected:
            names = ", ".join(sorted(missing or unexpected))
            return f"Form field mismatch: {names}."
        for name in fields:
            response = value[name]
            if name in allowed:
                continue
            if response is None or response == "" or response == []:
                return f"Complete {name.replace('_', ' ')} before submitting."
            if min_words and len(str(response).split()) < min_words:
                return (
                    f"Write at least {min_words} words for "
                    f"{name.replace('_', ' ')}."
                )
        return None

    return mo.ui.dictionary(dict(fields)).form(
        submit_button_label=submit_label,
        validate=validate,
        show_clear_button=True,
        clear_button_label="Clear draft",
    )


def accessible_text(mo, *, label: str, **kwargs):
    """Build a text input whose placeholder supplies a fallback AX name.

    Marimo 0.23.3 renders a visible label but does not associate it with the
    nested input. Mirroring the label into the placeholder gives the control
    a programmatic name until upstream connects the label to an input id.
    """

    kwargs.setdefault("placeholder", label)
    return mo.ui.text(label=label, **kwargs)


def accessible_text_area(mo, *, label: str, **kwargs):
    """Build a text area with the same accessible-name fallback."""

    kwargs.setdefault("placeholder", label)
    return mo.ui.text_area(label=label, **kwargs)
def module_header(spec: ModuleSpec, *, subtitle: str | None = None) -> str:
    track = "Core" if spec.track == "core" else "Advanced"
    return COURSE_CSS + (
        '<section class="ci-hero">'
        f"<div>{track} · Module {spec.index:02d}</div>"
        f"<h1>{html.escape(spec.title)}</h1>"
        f"<p>{html.escape(subtitle or spec.mission)}</p>"
        "</section>"
    )


def mode_badge(mode: str, detail: str) -> str:
    normalized = mode.lower()
    if normalized not in {"toy", "snapshot", "source"}:
        raise ValueError(f"Unknown evidence mode: {mode}")
    labels = {
        "toy": "TOY · ILLUSTRATION ONLY",
        "snapshot": "SNAPSHOT · FROZEN FIGURE",
        "source": "SOURCE · VERIFIED EXTRACTION",
    }
    return (
        f'<div role="status"><span class="ci-badge ci-{normalized}">'
        f"{labels[normalized]}</span><span>{html.escape(detail)}</span></div>"
    )


def callout(kind: str, title: str, body: str) -> str:
    if kind not in {"positive", "limit", "danger", "neutral"}:
        raise ValueError(f"Unknown callout kind: {kind}")
    return (
        f'<aside class="ci-callout ci-{kind}"><strong>{html.escape(title)}</strong> '
        f"{body}</aside>"
    )


def metric_cards(cards: Iterable[tuple[object, str]]) -> str:
    blocks = []
    for value, label in cards:
        blocks.append(
            '<div class="ci-card"><strong>'
            + html.escape(str(value))
            + "</strong><span>"
            + html.escape(label)
            + "</span></div>"
        )
    return '<div class="ci-card-grid">' + "".join(blocks) + "</div>"


def table(rows: Sequence[Mapping[str, object]], columns: Sequence[tuple[str, str]]) -> str:
    header = "".join(
        f'<th scope="col">{html.escape(label)}</th>' for _, label in columns
    )
    body = []
    for row in rows:
        body.append(
            "<tr>"
            + "".join(
                f"<td>{html.escape(str(row.get(key, '—')))}</td>" for key, _ in columns
            )
            + "</tr>"
        )
    return (
        '<div role="region" aria-label="Scrollable data table" tabindex="0" '
        'style="overflow-x:auto"><table class="ci-table"><thead><tr>'
        + header
        + "</tr></thead><tbody>"
        + "".join(body)
        + "</tbody></table></div>"
    )


def claim_card_html(card: ClaimCard) -> str:
    card.validate()
    rows = [
        ("Operation × target", f"{card.operation.value} × {card.target.value}"),
        ("Scope", card.scope),
        ("Endpoint / estimand", f"{card.endpoint.value}: {card.estimand}"),
        ("Unit / grouping", f"{card.unit}; grouped by {card.grouping}"),
        ("Intervention", card.intervention or "none"),
        ("Controls", "; ".join(card.controls) or "none"),
        ("Assumptions", "; ".join(card.assumptions) or "none stated"),
        ("Uncertainty", f"{card.uncertainty}; multiplicity: {card.multiplicity}"),
        ("Allowed", card.allowed),
        ("Excluded", card.excluded),
        ("Falsifier", card.falsifier),
        ("Status", card.status.value),
    ]
    terms = "".join(
        f"<dt>{html.escape(name)}</dt><dd>{html.escape(value)}</dd>" for name, value in rows
    )
    return '<section class="ci-claim"><h4>Claim card</h4><dl>' + terms + "</dl></section>"


def line_chart(
    series: Sequence[Mapping[str, object]],
    *,
    x_label: str,
    y_label: str,
    alt_text: str,
    y_domain: tuple[float, float] | None = None,
    width: int = 760,
    height: int = 280,
) -> str:
    points = [tuple(point) for item in series for point in item["points"]]  # type: ignore[index]
    if not points:
        raise ValueError("A chart needs at least one point")
    xs = [float(point[0]) for point in points]
    ys = [float(point[1]) for point in points]
    x_min, x_max = min(xs), max(xs)
    if y_domain is None:
        padding = (max(ys) - min(ys)) * 0.1 or 0.1
        y_min, y_max = min(ys) - padding, max(ys) + padding
    else:
        y_min, y_max = y_domain
    left, right, top, bottom = 58, 18, 24, 40
    plot_w, plot_h = width - left - right, height - top - bottom

    def sx(value: float) -> float:
        return left + (value - x_min) / max(x_max - x_min, 1e-12) * plot_w

    def sy(value: float) -> float:
        return top + (y_max - value) / max(y_max - y_min, 1e-12) * plot_h

    parts = [
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(alt_text)}" '
        'style="width:100%;max-height:280px">',
        f"<title>{html.escape(alt_text)}</title>",
        f"<desc>x axis: {html.escape(x_label)}; y axis: {html.escape(y_label)}. "
        + html.escape(
            "; ".join(
                f"{item['name']}: "
                + ", ".join(f"({float(x):g}, {float(y):g})" for x, y in item["points"])  # type: ignore[index]
                for item in series
            )
        )
        + "</desc>",
    ]
    for tick in range(5):
        value = x_min + (x_max - x_min) * tick / 4
        x = sx(value)
        label = f"{value:.0f}" if abs(value - round(value)) < 1e-9 else f"{value:.2g}"
        parts.append(
            f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{top}" y2="{height-bottom}" '
            f'stroke="#edf0f4"/><text x="{x:.1f}" y="{height-bottom+16}" '
            f'text-anchor="middle" fill="#46566d" font-size="11">{label}</text>'
        )
    for tick in range(5):
        value = y_min + (y_max - y_min) * tick / 4
        y = sy(value)
        parts.append(
            f'<line x1="{left}" x2="{width-right}" y1="{y:.1f}" y2="{y:.1f}" '
            'stroke="#d5dde8"/><text x="50" y="{:.1f}" text-anchor="end" '.format(y + 4)
            + f'fill="#46566d" font-size="11">{value:.2f}</text>'
        )
    patterns = ("", "5 3", "2 2", "8 3 2 3")
    for index, item in enumerate(series):
        color = str(item.get("color", ("#3366a3", "#a95f19")[index % 2]))
        coordinates = " ".join(
            f"{sx(float(x)):.1f},{sy(float(y)):.1f}" for x, y in item["points"]  # type: ignore[index]
        )
        dash = patterns[index % len(patterns)]
        parts.append(
            f'<polyline points="{coordinates}" fill="none" stroke="{color}" '
            f'stroke-width="2.6" stroke-dasharray="{dash}"/>'
        )
        for x, y in item["points"]:  # type: ignore[index]
            parts.append(
                f'<circle cx="{sx(float(x)):.1f}" cy="{sy(float(y)):.1f}" r="3" '
                f'fill="{color}" stroke="#fff"/>'
            )
    parts.append(
        f'<text x="14" y="{height/2:.1f}" transform="rotate(-90 14 {height/2:.1f})" '
        f'text-anchor="middle" fill="#46566d" font-size="11">{html.escape(y_label)}</text>'
        f'<text x="{left + plot_w/2:.1f}" y="{height-5}" text-anchor="middle" '
        f'fill="#46566d" font-size="11">{html.escape(x_label)}</text></svg>'
    )
    legend = " · ".join(html.escape(str(item["name"])) for item in series)
    return "".join(parts) + f'<div class="ci-caption">{legend}</div>'


def chessboard(fen: str, *, highlighted: Iterable[str] = ()) -> str:
    import chess

    board = chess.Board(fen)
    selected = {chess.parse_square(square) for square in highlighted}
    symbols = {
        "P": "♙", "N": "♘", "B": "♗", "R": "♖", "Q": "♕", "K": "♔",
        "p": "♟", "n": "♞", "b": "♝", "r": "♜", "q": "♛", "k": "♚",
    }
    piece_names = {
        chess.PAWN: "pawn",
        chess.KNIGHT: "knight",
        chess.BISHOP: "bishop",
        chess.ROOK: "rook",
        chess.QUEEN: "queen",
        chess.KING: "king",
    }
    rows = []
    for rank in range(7, -1, -1):
        cells = []
        for file in range(8):
            square = chess.square(file, rank)
            piece = board.piece_at(square)
            name = chess.square_name(square)
            color = "ci-light" if (file + rank) % 2 else "ci-dark"
            is_highlighted = square in selected
            outline = (
                "outline:4px solid #005fcc;outline-offset:-4px;"
                "box-shadow:inset 0 0 0 2px #fff;"
                if is_highlighted
                else ""
            )
            occupant = (
                f"{'white' if piece.color == chess.WHITE else 'black'} "
                f"{piece_names[piece.piece_type]}"
                if piece
                else "empty"
            )
            label = f"{name}: {occupant}" + (", highlighted" if is_highlighted else "")
            marker = (
                '<span class="ci-highlight-marker" aria-hidden="true">★</span>'
                if is_highlighted
                else ""
            )
            cells.append(
                f'<td class="{color}" style="{outline}" aria-label="{html.escape(label)}">'
                f'{symbols.get(piece.symbol(), "") if piece else ""}'
                f'<span class="ci-coord">{name}</span>{marker}</td>'
            )
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return (
        '<table class="ci-board" aria-label="Chess board"><caption class="ci-visually-hidden">'
        + html.escape(fen)
        + "</caption><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )


def square_heatmap(
    values: Sequence[float],
    *,
    alt_text: str,
    signed: bool = True,
) -> str:
    """Render 64 square values with color, sign, magnitude, and accessible text."""

    if len(values) != 64:
        raise ValueError("A square heatmap needs exactly 64 values")
    numeric = [float(value) for value in values]
    scale = max((abs(value) for value in numeric), default=1.0) or 1.0
    rows = []
    ranked = sorted(range(64), key=lambda index: abs(numeric[index]), reverse=True)
    for rank in range(7, -1, -1):
        cells = []
        for file in range(8):
            index = rank * 8 + file
            value = numeric[index]
            magnitude = min(abs(value) / scale, 1.0)
            if signed and value < 0.0:
                rgb = "169,95,25"
            else:
                rgb = "51,102,163"
            background = f"rgba({rgb},{0.12 + 0.78 * magnitude:.3f})"
            foreground = "#fff" if magnitude > 0.57 else "#162235"
            square = chr(ord("a") + file) + str(rank + 1)
            display = f"{value:+.2f}" if signed else f"{value:.2f}"
            cells.append(
                f'<td style="width:3rem;height:3rem;text-align:center;background:{background};'
                f'color:{foreground};font-size:.72rem;border:1px solid #fff" '
                f'aria-label="{square}: {value:+.6g}"><strong>{square}</strong><br>{display}</td>'
            )
        rows.append("<tr>" + "".join(cells) + "</tr>")
    top = ", ".join(
        f"{chr(ord('a') + (index % 8))}{index // 8 + 1} ({numeric[index]:+.3g})"
        for index in ranked[:5]
    )
    return (
        f'<figure><table aria-label="{html.escape(alt_text)}" style="border-collapse:collapse">'
        + "<tbody>"
        + "".join(rows)
        + "</tbody></table>"
        + f'<figcaption class="ci-caption">{html.escape(alt_text)} Top magnitude: '
        + html.escape(top)
        + ". Blue denotes positive/nonnegative; orange denotes negative; every cell also prints a signed value.</figcaption></figure>"
    )


def resource_card(
    *,
    mode: str,
    runtime: str,
    device: str,
    determinism: str,
    sources: str,
    limitations: str,
) -> str:
    rows = [
        {"field": "Evidence mode", "value": mode},
        {"field": "Expected runtime", "value": runtime},
        {"field": "Device", "value": device},
        {"field": "Seed / dtype", "value": determinism},
        {"field": "Sources", "value": sources},
        {"field": "Limitations", "value": limitations},
    ]
    return table(rows, (("field", "Resource / provenance"), ("value", "Value")))
