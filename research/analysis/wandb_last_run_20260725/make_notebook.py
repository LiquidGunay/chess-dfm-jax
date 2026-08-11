#!/usr/bin/env python3
"""Create and execute the reproducible lineage-analysis notebook."""

from __future__ import annotations

import os
import textwrap
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR", "/mountpoint/.exp/chess-dfm-jax/.local/cache/matplotlib"
)
os.environ.setdefault("TMPDIR", "/mountpoint/.exp/chess-dfm-jax/.local/tmp")

import nbformat as nbf
from nbclient import NotebookClient


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "wandb_last_run_reconstruction.ipynb"


def code(source: str):
    return nbf.v4.new_code_cell(textwrap.dedent(source).strip())


def markdown(source: str):
    return nbf.v4.new_markdown_cell(textwrap.dedent(source).strip())


def build_notebook():
    notebook = nbf.v4.new_notebook()
    notebook["metadata"]["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    notebook["metadata"]["language_info"] = {"name": "python", "version": "3"}
    notebook["cells"] = [
        markdown(
            """
            # May 2026 joint-training lineage reconstruction

            This notebook rebuilds the disconnected W&B runs into one checkpoint
            lineage, reconciles examples seen against the 28,343,296-example
            training set, audits validation comparability, and fits explicitly
            conditional 100-epoch projections. It reads only the bounded local W&B
            export and checked-in repository artifacts.
            """
        ),
        code(
            """
            import os
            import sys
            from pathlib import Path

            os.environ["MPLCONFIGDIR"] = "/mountpoint/.exp/chess-dfm-jax/.local/cache/matplotlib"
            os.environ["TMPDIR"] = "/mountpoint/.exp/chess-dfm-jax/.local/tmp"

            analysis_dir = Path("/mountpoint/.exp/chess-dfm-jax/research/analysis/wandb_last_run_20260725")
            repo_root = Path("/mountpoint/.exp/chess-dfm-jax")
            sys.path.insert(0, str(analysis_dir))

            import pandas as pd
            from IPython.display import Image, Markdown, display
            import analyze_lineage

            results = analyze_lineage.analyze()
            derived = analysis_dir / "derived"
            figures = analysis_dir / "figures"
            """
        ),
        markdown(
            """
            ## Reconstructed exposure

            “Epoch” means one training-set equivalent (28,343,296 examples), not a
            W&B run or optimizer phase. Only updates inherited by the retained
            checkpoint count toward its lineage; discarded branches and retries are
            tracked separately.
            """
        ),
        code(
            """
            summary = results["summary"]
            display(Markdown(
                f"**Retained step-265k model:** {summary['retained_lineage_epochs']:.3f} "
                f"epochs / {summary['retained_lineage_examples']:,} examples.  "
                f"**W&B endpoint:** {summary['wandb_endpoint_lineage_epochs']:.3f} "
                f"epochs, but it is not the retained checkpoint.  "
                f"**Actual compute:** at least "
                f"{summary['actual_compute_epochs_lower_bound']:.3f} epochs and likely "
                f"about {summary['retry_adjusted_compute_epochs_estimate']:.3f} after "
                f"high-confidence checkpoint replays (medium-confidence estimate)."
            ))
            lineage = pd.read_csv(derived / "lineage_segments.csv")
            display(lineage[[
                "segment", "inherited_updates", "global_batch",
                "examples_in_final_lineage", "epochs_in_final_lineage",
                "optimizer_start", "jepa_sigreg_coefficient"
            ]])
            """
        ),
        markdown(
            """
            ## Validation protocol audit

            Cross-run validation is not apples-to-apples. Run C evaluated only
            process 0's 96-shard slice; its first two validations used 1,024
            examples, then later validations used 16,384. Projection fits therefore
            use only the internally consistent run-C protocol from step 15k onward.
            """
        ),
        code(
            """
            validation = pd.read_csv(derived / "validation_points.csv")
            protocol_summary = (
                validation.groupby(
                    ["source_run", "validation_protocol", "validation_examples",
                     "validation_visible_shards"],
                    dropna=False
                )
                .agg(points=("source_step", "count"),
                     first_step=("source_step", "min"),
                     last_step=("source_step", "max"))
                .reset_index()
            )
            display(protocol_summary)
            """
        ),
        markdown("## Validation trajectory and conditional projection"),
        code(
            """
            display(Image(filename=str(figures / "validation_lineage_projection.png")))
            """
        ),
        code(
            """
            fits = pd.read_csv(derived / "projection_fits.csv")
            display(fits[[
                "scenario", "model", "start_step", "cutoff_step", "fit_points",
                "rmse", "holdout_mae", "projected_val_dfm_ce_at_epoch_100"
            ]].sort_values(["scenario", "start_step", "model"]))

            projection = summary["conditional_epoch_100_projection"]
            display(Markdown(
                f"Stable-phase saturation fits give **"
                f"{projection['range_min']:.3f}–{projection['range_max']:.3f}** "
                f"at epoch 100. Tail-inclusive fits give **"
                f"{projection['tail_inclusive_range_min']:.3f}–"
                f"{projection['tail_inclusive_range_max']:.3f}**. "
                "These are scenarios, not confidence intervals."
            ))
            """
        ),
        markdown(
            """
            ## Late-run deterioration

            The best comparable validation DFM CE was at step 255k. Both DFM CE and
            JEPA diagnostics then worsened; the target SIGReg spike at the endpoint
            makes a literal “continue unchanged to epoch 100” extrapolation unsafe.
            """
        ),
        code(
            """
            display(Image(filename=str(figures / "tail_instability.png")))
            """
        ),
        markdown("## Data-quality findings and clean-run implications"),
        code(
            """
            quality = results["data_quality"]
            display(pd.DataFrame(quality["issues"])[
                ["severity", "confidence", "issue", "impact", "remediation"]
            ])
            check = quality["profile"]["local_checkpoint_crosscheck"]
            display(Markdown(
                f"The retained local metrics and W&B match exactly at "
                f"{len(check['common_validation_steps'])} common validation steps: "
                f"**{check['all_checked_values_exact']}**."
            ))
            """
        ),
        markdown(
            """
            ## Bottom line

            The final retained model is a roughly 76-epoch model assembled across
            three optimizer phases, not an eight-epoch run. Its best observed
            comparable DFM CE was 4.503 at about 73.17 epochs; the retained step-265k
            checkpoint had worsened to 4.541. A clean run should make validation
            topology-invariant, version the objective, log monotonic examples seen
            and optimizer updates, and bind every evaluation to an explicit
            checkpoint artifact before attempting to calibrate loss against Elo.
            """
        ),
    ]
    return notebook


def main() -> None:
    notebook = build_notebook()
    client = NotebookClient(
        notebook,
        timeout=600,
        kernel_name="python3",
        resources={"metadata": {"path": str(HERE)}},
    )
    executed = client.execute()
    nbf.write(executed, OUTPUT)
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
