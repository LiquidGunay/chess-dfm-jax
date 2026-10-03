"""Create bounded W&B comparison views without altering existing user views."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from research.analysis.import_wandb_logs import ROOT, ensure_private_project, inventory


ENTITY = "gunays-independent"
ARCHIVE_PROJECT = "chess-dfm-torch"
HISTORY_PROJECT = "chess-dfm-history"
COHORTS = {
    "hero-554-current-only": "Archive | Hero architecture and optimizer | 554 updates",
    "hero-1024-current-only": "Archive | Hero closed-loop comparison | 1,024 updates",
    "proposal-a-lr": "Archive | Proposal A learning-rate sweep",
    "proposal-a-decay": "Archive | Proposal A weight-decay controls",
    "PA-001-full-epoch": "Archive | Proposal A full epoch",
    "diagnostics": "Archive | Diagnostics only | not strength evidence",
}
HISTORY_IDS = ["hh002-full-edf0a158db54", "hh002-old-edf0a158db54"]


def line(metric, title):
    import wandb_workspaces.reports.v2 as wr
    return wr.LinePlot(title=title, x="examples", y=[metric], aggregate=False,
                       smoothing_type="none", max_runs_to_show=20,
                       title_x="Training examples processed")


def section(name, panels, *, opened=True):
    import wandb_workspaces.workspaces as ws
    import wandb_workspaces.reports.v2 as wr
    for index, panel in enumerate(panels):
        panel.layout = wr.Layout(x=12 * (index % 2), y=6 * (index // 2), w=12, h=6)
    return ws.Section(name=name.replace(" | ", " - "), panels=panels, is_open=opened)


def make_views(runs):
    import wandb_workspaces.workspaces as ws
    import wandb_workspaces.reports.v2 as wr
    from wandb_workspaces import expr
    views = {}
    for group, title in COHORTS.items():
        members = [run for run in runs if run["group"] == group]
        ids = [run["id"] for run in members]
        pools = sorted({pool for run in members for pool in run["config"]["validation_pools"]})
        note = ("Historical logs only. Runs are selected explicitly; diagnostics and other training "
                "budgets are excluded. The x-axis is examples processed, not upload time. "
                "No averaging across architecture variants. Total weighted loss is not a fair "
                "cross-recipe ranking. Validation panels use the original pool identified in their metric name; "
                "these are not the HH-002 matched-history validation results. Loss is not playing strength.")
        sections = [section("Read before comparing", [wr.MarkdownPanel(markdown=note)]),
                    section("Training | comparable component definitions", [
                        line("train/dfm_ce_loss", "DFM cross-entropy"),
                        line("train/root_legal_conditional_ce", "Root legal-move cross-entropy")])]
        for pool in pools:
            sections.append(section("Original held-out pool | " + pool, [
                wr.BarPlot(title="Latest held-out DFM cross-entropy", metrics=[pool + "/dfm_ce_loss"],
                           aggregate=False, max_runs_to_show=20, max_bars_to_show=20),
                wr.BarPlot(title="Latest held-out root legal-move cross-entropy", metrics=[pool + "/root_legal_conditional_ce"],
                           aggregate=False, max_runs_to_show=20, max_bars_to_show=20)]))
        sections.append(section("Diagnostics | objectives vary; do not rank total loss", [
            line("train/loss", "Total weighted training loss (recipe-dependent)"),
            line("train/jepa_positive_loss", "JEPA positive loss (targets depend on encoder)"),
            line("train/learning_rate", "Main learning rate"),
            line("train/bt4_learning_rate", "Encoder learning rate")], opened=False))
        views[group] = ws.Workspace(entity=ENTITY, project=ARCHIVE_PROJECT, name=title.replace(" | ", " - "),
            auto_generate_panels=False, sections=sections,
            settings=ws.WorkspaceSettings(x_axis="examples", max_runs=20, smoothing_type="none"),
            runset_settings=ws.RunsetSettings(filters=[expr.Config("comparison_cohort") == group,
                                                      expr.Metric("name").isin(ids)],
                pinned_columns=["config:history_mode.value", "config:training_updates.value"]))
    note = ("HH-002: full-history Hero recipe versus historical Hero 1. Only these two run IDs are included. "
            "Training examples differ slightly because unreconstructable histories and minibatch tails were excluded; "
            "the old training curve is descriptive, not a perfectly matched data control. "
            "The 1,024-position validation curve uses shared retained positions with each model's training-matched "
            "history inputs. The 8,128-position terminal evaluation is shown separately, never joined to that curve. "
            "Old endpoint: 554 updates; new endpoint: 553 updates. JEPA targets change with history/encoder; "
            "a lower JEPA loss alone does not establish improvement. No playing-strength conclusion follows from these plots.")
    views["HH-002"] = ws.Workspace(entity=ENTITY, project=HISTORY_PROJECT,
        name="HH-002 - History versus no history - matched validation",
        auto_generate_panels=False,
        settings=ws.WorkspaceSettings(x_axis="examples", max_runs=2, smoothing_type="none"),
        runset_settings=ws.RunsetSettings(filters=[expr.Metric("name").isin(HISTORY_IDS)]),
        sections=[section("Comparison contract", [wr.MarkdownPanel(markdown=note)]),
            section("Shared 1,024-position validation curve", [
                line("val/dfm_ce_loss", "Validation DFM cross-entropy"),
                line("val/root_legal_conditional_ce", "Validation root legal-move cross-entropy"),
                line("val/loss", "Validation total loss (same recipe; targets differ)"),
                line("val/jepa_positive_loss", "Validation JEPA loss (targets differ)")]),
            section("Shared 8,128-position terminal evaluation | separate pool", [
                wr.BarPlot(title="Terminal DFM cross-entropy", metrics=["val_full/dfm_ce_loss"], aggregate=False),
                wr.BarPlot(title="Terminal root legal-move cross-entropy", metrics=["val_full/root_legal_conditional_ce"], aggregate=False)]),
            section("Training | slightly different example sets", [
                line("train/dfm_ce_loss", "Training DFM cross-entropy"),
                line("train/root_legal_conditional_ce", "Training root legal-move cross-entropy"),
                line("train/loss", "Training total weighted loss"),
                line("train/jepa_positive_loss", "Training JEPA positive loss")])])
    return views


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--dotenv", type=Path)
    parser.add_argument("--receipt", type=Path,
                        default=ROOT / "research/analysis/wandb_sync_20261003/workspaces.json")
    args = parser.parse_args()
    views = make_views(inventory(ROOT))
    if not args.save:
        print(json.dumps({key: {"name": view.name, "project": view.project,
                         "sections": len(view.sections)} for key, view in views.items()}, indent=2))
        return
    if args.dotenv:
        from dotenv import dotenv_values
        os.environ["WANDB_API_KEY"] = dotenv_values(args.dotenv)["WANDB_API_KEY"]
    import wandb_workspaces.workspaces as ws
    saved = json.loads(args.receipt.read_text()) if args.receipt.exists() else {}
    for project in (ARCHIVE_PROJECT, HISTORY_PROJECT):
        ensure_private_project(ENTITY, project)
    for key, view in views.items():
        if key in saved:
            # Only mutate a view previously created by this script, never the user's default.
            existing = ws.Workspace.from_url(saved[key]["url"])
            existing.sections, existing.settings = view.sections, view.settings
            existing.runset_settings = view.runset_settings
            existing.auto_generate_panels = False
            view = existing
        view.save()
        saved[key] = {"url": view.url, "name": view.name, "project": view.project,
                      "sections": len(view.sections), "readback_verified": False}
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.write_text(json.dumps(saved, indent=2) + "\n")
        remote = ws.Workspace.from_url(view.url)
        if (remote.name != view.name or remote.auto_generate_panels is not False
                or len(remote.sections) != len(view.sections)
                or remote.runset_settings.filters != view.runset_settings.filters):
            raise RuntimeError("Saved workspace read-back differs from requested view")
        saved[key]["readback_verified"] = True
        args.receipt.write_text(json.dumps(saved, indent=2) + "\n")
        print(json.dumps({key: saved[key]}), flush=True)


if __name__ == "__main__":
    main()
