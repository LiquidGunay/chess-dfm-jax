# Autoresearch program

The goal is to improve the fixed held-out chess metrics of the local-GPU
BT4/DFM/JEPA model while preserving latent diversity and making training and
inference faster.

Current gate: `AUTORESEARCH_READY = True`. The dense source-parity gate and
first-experiment preregistration are complete, and the frozen 30-minute
autoresearch loop is active. Promotion still requires every offline and arena
gate below; readiness alone is not a strength claim.

The first accepted experiment samples two of eight future BT4 targets during
training while preserving all eight DFM/JEPA prediction horizons and
full-horizon evaluation. Its update-800 checkpoint improves two-pool DFM CE
from `4.5102692712` to `4.5054920968`, improves accuracy and legal mass, passes
every latent gate, and raises fixed-time throughput from `41.35` to `92.97`
examples/s. It is the current offline incumbent. Its direct 128-pair arena
against corrected v2/update 400 scored `50.586%` (`+4.1` descriptive logistic
Elo, pair-aware 95% interval `[-80.8,+89.4]`), which is positive but
inconclusive. An exact v2 repeat again selects update 800 and independently
passes every gate at CE `4.5077912323`, but its selected CE is `0.0022991356`
worse than v1, beyond the old `0.000637` repeat envelope. The direction is
replicated; the effect size is not tightly repeat-stable, and the model is not
Elo-promoted.

The repeat-qualified norm-on compatibility baseline is v2/update 300. An identical v1 run also
selected update 300, and their four-pool DFM CE gains differ by only
`0.000210253`; the selected v2 checkpoint also passes the recorded
effective-rank, low-variance-tail, prediction/target RMS-ratio, and
trivial-baseline checks. This repeat-qualified baseline is not an accepted
autoresearch experiment, an Elo result, or a promoted checkpoint. Readiness
was opened later by the separate source-parity and preregistration gates.

The baseline-length prediction-SIGReg `0.57` experiment is rejected. Its best
two-pool CE is `4.511721` versus incumbent v2/u300 `4.510334`; a matched
update-400 audit also regresses accuracy, legal mass, and JEPA MSE for only
tiny rank/variance gains. Keep prediction-SIGReg at `0.0` for the norm-on
compatibility objective. The separately qualified corrected no-norm objective
uses prediction-SIGReg `1.0`.

The corrected no-norm target-SIGReg-5.76/prediction-SIGReg-1.0 baseline is
repeat-qualified at v2/update 400. Real-checkpoint GPU parity, sealed-history
timing, and both 128-pair cap-256 strength anchors are complete. It scored
`49.61%` against the recovered step-265,000 model and `38.09%` against raw
BT4. These are descriptive model-pool-relative results, not promotion or
absolute Elo. The fixed four-model representation comparison may proceed.

Keep searchless inference fixed at eight DFM refinement passes during the
initial stronger-model experiments. Although the sampled-target checkpoint
improves the frozen offline gates twice, its effect size varies and its arena
interval remains unresolved. Pass-count ablation stays deferred until a larger
offline improvement gives less ambiguous strength evidence. Changing
refinement compute must not be mixed into an architecture comparison.

The promotion assets are repaired and available. The source-derived v3 pool
replaces the old claimable-threefold root before selection is frozen, has zero
selected validation/test overlap, and passes production replay for all 2,048
histories. Runtime skipping or substitution remains forbidden. This asset
repair did not by itself open readiness. Dense representation drift is complete in
`artifacts/representations/dense-stage1-four-model-v2`. Official-epsilon
PyTorch/JAX source parity passes in
`artifacts/representations/upstream-bt4-source-parity-v1`; the pinned
TransformerLens constructor-default epsilon fails and must not be used as the
source oracle. The immutable preregistration and completed outcome of the first
post-baseline experiment are in
`research/experiment_future_target_sampling_k2.md`.

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
Select checkpoints on the matched mean of both pools. For a near tie, add new
predeclared matched pools to every tied checkpoint; the current tie-break uses
seeds 30,000 and 40,000. Never choose an extra seed after inspecting only one
candidate.

The weighted training loss is not by itself a promotion metric.

The strength gate consumes complete color-reversed pairs from the repinned
promotion pool. Only the normalized-Elo GSPRT may promote a checkpoint:
`H0=0`, `H1=+20`, `alpha=beta=0.05`, checked after complete pairs and capped
at 2,048 pairs. Descriptive logistic Elo never authorizes promotion.

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
