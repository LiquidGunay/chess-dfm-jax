# Autoresearch program

The goal is to improve the fixed held-out chess metrics of the local-GPU
BT4/DFM/JEPA model while preserving latent diversity and making training and
inference faster.

Current gate: `AUTORESEARCH_READY = False`. Do not launch unattended
architecture search, append calibration runs to `research/results.tsv`, run a
promotion arena, or describe a checkpoint as promoted until a repeated,
noise-qualified baseline passes the offline gates below.

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
- Fix validation batch size independently of physical training batch size.
- Default to local JSONL/TSV logging; external tracking is opt-in.

## Primary result

Optimize fixed, seeded, globally permuted validation DFM cross-entropy. Also
report:

- first-action and per-horizon accuracy;
- legal mass;
- JEPA loss against zero, identity, shuffled-target, and action-shuffled
  baselines;
- per-horizon latent RMS, feature variance, and effective rank;
- examples processed and examples/s;
- compile time and peak HBM; and
- initialization lineage.

First-shard slices are correctness and calibration instruments only. Every
quality comparison must use identical `global_permutation` validation slots
for control and candidate, with the effective seed, schedule, batch size, batch
count, and finite-sample partition recorded. A validation sampler, batch
partition, or metric change requires rerunning the matched control.

Before readiness is opened, the comparison protocol is 64 validation batches
of 64 examples. Use seed 10,000 for development and seed 20,000 for the first
independent confirmation. Training-batch changes must retain those same 4,096
positions and partitions through `--eval-batch-size 64`.

The weighted training loss is not by itself a promotion metric.

The strength gate consumes complete color-reversed pairs from the frozen
promotion pool. Only the normalized-Elo GSPRT may promote a checkpoint:
`H0=0`, `H1=+20`, `alpha=beta=0.05`, checked after complete pairs and capped at
2,048 pairs. Descriptive logistic Elo never authorizes promotion.

Reject a run if:

- any metric is non-finite;
- prediction variance or effective rank crosses the collapse threshold;
- the model fails to beat the trivial JEPA baselines;
- legality regresses beyond the fixed tolerance;
- an apparent improvement does not exceed repeated-run noise;
- the run writes outside the workspace; or
- validation data order or metric code changed without a matched control rerun.

## Experiment loop

1. Inspect the current code, previous result, and baseline noise.
2. State one testable hypothesis.
3. Make one coherent change to `research/train.py`.
4. Run correctness and short GPU smoke tests.
5. Repeat the baseline/candidate when bitwise repeatability is absent.
6. Outside the unattended loop, complete a matched 30-minute baseline
   qualification and open readiness only if its repeats pass every gate.
7. Once readiness is open, run the fixed 30-minute experiment.
8. Append exactly one row to `research/results.tsv` for that accepted
   30-minute experiment; leave smoke and pre-baseline calibration runs out.
9. Keep improvements that exceed baseline noise and pass every gate.
10. Revert rejected changes without rewriting the result history.
11. Promote only confirmed candidates to relative Elo.

Prefer simple changes whose effects can be explained. Record surprises and
negative results; they are part of the research output.
