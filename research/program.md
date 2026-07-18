# Autoresearch program

The goal is to improve the fixed held-out chess metrics of the local-GPU
BT4/DFM/JEPA model while preserving latent diversity and making training and
inference faster.

## Editable surface

During automated architecture research, edit only `research/train.py`.

Treat these as immutable:

- `research/prepare.py`
- `research/program.md`
- the fixed data splits and arena FENs
- the recovered baseline checkpoint
- metric definitions and promotion gates

Changes to immutable support require a separate human-reviewed commit and a new
baseline version.

## Runtime contract

- Run entirely under `/mountpoint/.exp`.
- Source `research/env.sh` or use `research/run_gpu.sh`.
- Use one NVIDIA A10G.
- Measure compilation separately.
- Give each accepted quick experiment 30 minutes of steady-state training.
- Fix seeds and data order for the first comparison.
- Default to local JSONL/TSV logging; external tracking is opt-in.

## Primary result

Optimize fixed validation DFM cross-entropy. Also report:

- first-action and per-horizon accuracy;
- legal mass;
- JEPA loss against zero, identity, shuffled-target, and action-shuffled
  baselines;
- per-horizon latent RMS, feature variance, and effective rank;
- examples processed and examples/s;
- compile time and peak HBM; and
- initialization lineage.

The weighted training loss is not by itself a promotion metric.

Reject a run if:

- any metric is non-finite;
- prediction variance or effective rank crosses the collapse threshold;
- the model fails to beat the trivial JEPA baselines;
- legality regresses beyond the fixed tolerance;
- the run writes outside the workspace; or
- validation data order or metric code changed.

## Experiment loop

1. Inspect the current code, previous result, and baseline noise.
2. State one testable hypothesis.
3. Make one coherent change to `research/train.py`.
4. Run correctness and short GPU smoke tests.
5. Run the fixed 30-minute experiment.
6. Append exactly one row to `research/results.tsv`.
7. Keep improvements that exceed baseline noise and pass every gate.
8. Revert rejected changes without rewriting the result history.
9. Promote only confirmed candidates to repeated runs and relative Elo.

Prefer simple changes whose effects can be explained. Record surprises and
negative results; they are part of the research output.
