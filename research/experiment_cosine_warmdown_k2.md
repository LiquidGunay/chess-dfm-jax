# Experiment 006: fixed-time cosine learning-rate warmdown

Status: completed and accepted as the offline incumbent on 2026-07-21. The
contract below was preregistered before implementation or candidate GPU
measurement.

## Question and hypothesis

Both exact K=2 runs select update 800 under the constant main/BT4 learning
rates `3e-5/1e-6`. Their update-1200 checkpoints regress sharply and their
terminal checkpoints remain worse than update 800:

| run | update 400 CE | update 800 CE | update 1200 CE | terminal CE |
| --- | ---: | ---: | ---: | ---: |
| K=2 v1 | 4.5099766254 | **4.5054920968** | 4.5286737829 | 4.5071463250 |
| K=2 v2 | 4.5138616990 | **4.5077912323** | 4.5376210772 | 4.5146793593 |

This experiment asks whether late fixed-time degradation is reduced by
warming both learning rates down after the reproducible early-improvement
phase. The primary hypothesis is that a single preregistered cosine schedule
improves selected full-horizon CE beyond the K=2 repeat envelope while
preserving policy, legality, latent health, and throughput. The secondary
hypothesis is that update 1200 no longer exhibits the large policy/legal
regression seen in both constant-rate runs.

This is an optimizer-schedule experiment only. It restores the complete K=2
architecture: both JEPA projector blocks, all four DFM blocks, two sampled
future targets, eight recurrent predictions, and eight searchless refinement
passes.

## Candidate contract

Relative to Experiment 001 K=2, the sole coherent schedule override is:

```python
"lr_decay_start_steps": 400
"lr_decay_steps": 800
"lr_min_ratio": 0.1
```

Apply it identically as a multiplicative schedule to the main and BT4 peak
rates:

- optimizer updates 0 through 399: ratio `1.0`;
- updates 400 through 1199: Optax cosine decay from `1.0` to `0.1` over 800
  schedule steps;
- update 800: ratio `0.55` exactly;
- update 1200 and later: ratio `0.1`;
- no warmup (`lr_warmup_steps=0`).

The implementation must satisfy all of the following:

- Keep the existing constant-schedule path bitwise exact when
  `lr_decay_steps=0`, with `lr_decay_start_steps=0` and `lr_min_ratio=1.0`.
- Require integer non-negative start/decay steps and finite minimum ratio in
  `(0, 1]`. A positive start or non-unit minimum is inert and rejected when
  decay is disabled. The first implementation rejects simultaneous warmup and
  decay rather than guessing their composition.
- Use one optimizer update counter for both schedules. The BT4/main ratio
  remains exactly `1e-6 / 3e-5`; weight decay, Muon/Adam partitioning,
  clipping, and fresh optimizer initialization are unchanged.
- Restore every source model parameter exactly and start from a fresh
  optimizer. The schedule must not change model or optimizer-state ABI and
  must be bound into config serialization and strict resume contracts.
- Keep full four-layer DFM and two-layer JEPA projector execution, K=2 target
  sampling, no anchors, target/prediction SIGReg counts `576/512`, fixed
  64-example normalized V-statistic SIGReg, and norm/target/prediction
  coefficients `0.0/5.76/1.0`.
- Preserve every model forward value, RNG stream, loss scalar, and gradient at
  optimizer update 0 relative to K=2. Only post-gradient parameter updates may
  diverge according to the frozen schedule.
- Record peak rates, schedule family, boundaries, floor ratio, and exact
  boundary values in reports and resume contracts.

## Frozen control and run configuration

The matched control remains K=2 v1/update 800:

- state SHA-256:
  `f42af6bef64c532376e0532d2370b1cf2bd6356d494e08738355359bb506abd2`;
- two-pool DFM CE `4.5054920968`;
- accuracy `0.1095886230`;
- legal mass `0.6455246028`;
- cached profile `92.1149 examples/s`;
- 30-minute throughput `92.9745 examples/s`; and
- exact-repeat selected-checkpoint CE gap `0.0022991356`.

The candidate starts from the recovered step-265,000 model with a fresh
optimizer. Freeze one NVIDIA A10G and no concurrent workload;
physical/evaluation batches `128/64`; train/data seed 0 with
`global_permutation`; peak main/BT4 rates `3e-5/1e-6`; 1,800 seconds of
steady-state training after compile/first update; validation seeds 10,000 and
20,000, each 64 batches of 64 examples; and saves at updates 400, 800, 1200,
and terminal with at most four temporary states.

The update-based schedule is deliberately independent of measured wall time.
If hardware throughput changes enough that terminal occurs before update 1200,
the run is invalid for this hypothesis and must be diagnosed rather than
silently redefining the schedule.

## Correctness and performance gates

Before the 30-minute run:

1. CPU tests prove exact schedule ratios at updates 0, 399, 400, 800, 1199,
   1200, and later; proportional main/BT4 values; default constant-path
   exactness; validation of inert/unsupported combinations; config and resume
   binding; unchanged optimizer-state ABI; and unchanged update-0 model/loss/
   gradient values.
2. A real-checkpoint A10G smoke has finite loss and gradients, all architecture
   depths at their full defaults, three encoded boards per example, eight
   predictions, SIGReg counts `576/512`, no clipping/skipped update, and no
   write outside `/mountpoint/.exp`.
3. A cached 30-update profile must reach at least `87.5092 examples/s`, 95% of
   K=2. Schedule evaluation should have no material throughput cost; report
   the measured delta.

## Checkpoint selection and decision

Evaluate every saved checkpoint on the identical seed-10,000 and seed-20,000
full-horizon pools and select minimum mean DFM CE. Acceptance requires:

- two-pool DFM CE below `4.5031929612`, more than `0.0022991356` better than
  retained K=2 v1;
- accuracy at least `0.1075886230`;
- legal mass at least `0.6425246028`;
- every metric and gradient finite;
- positive JEPA prediction beats zero and action-shuffled predictions at every
  horizon;
- mean/minimum prediction effective rank at least `30.0/28.5`;
- mean/minimum prediction feature-std p05 at least `0.63/0.61`;
- mean target RMS at least `0.90`; and
- prediction/target RMS ratio in `[0.94, 1.02]`.

Separately report update-1200 CE/accuracy/legal deltas against both constant-
rate runs to test the warmdown hypothesis without using that secondary
analysis for checkpoint selection. If the first run passes, repeat it exactly
and require the repeat to beat K=2 CE while passing every non-CE gate before
the frozen 128-pair arena. A failure gets no repeat, arena, pass-count sweep,
or SAE refit. Retain only a qualifying selected state; otherwise delete all
candidate states after preserving compact evidence.

## Outcome

Commits `dfe3dd4` and `984101c` preregister and implement the schedule. The
implementation preserves the default constant/warmup optimizer ABI, binds the
schedule into checkpoint contracts and reports, and passed 53 focused CPU
tests plus Ruff. The one-update A10G smoke had finite metrics, encoded three
boards per example, retained all eight predictions, reported target/prediction
SIGReg counts `576/512`, did not clip or skip the update, and peaked at
`9,482,849,280` bytes of HBM. The cached 30-update profile reached `93.2987`
end-to-end examples/s, `1.29%` above K=2 and above the `87.5092` gate.

The first 1,800-second run processed `161,408` examples in 1,261 updates at
`92.8796` steady end-to-end examples/s and peaked at `9,581,214,976` bytes.
The frozen two-pool scan was:

| update | DFM CE | accuracy | legal mass |
| ---: | ---: | ---: | ---: |
| 400 | 4.5098696686 | 0.1086425781 | 0.6431957292 |
| 800 | 4.5032504834 | 0.1087341309 | 0.6439909316 |
| 1200 | 4.5014958344 | 0.1093139648 | 0.6468938608 |
| 1261 | **4.5007607210** | **0.1094207764** | **0.6473310636** |

The selected terminal state improves the retained K=2 incumbent by
`0.0047313757`, exceeding its `0.0022991356` repeat separation, while passing
all policy and latent gates. Its mean/min prediction effective rank is
`30.9710/29.0574`, mean/min feature-std p05 is `0.65062/0.63730`, mean target
RMS is `0.92754`, and prediction/target RMS ratio is `0.97044`. Positive JEPA
MSE beats both zero and action-shuffled controls at every horizon.

The update-1200 result directly confirms the secondary hypothesis. Relative
to constant-rate K=2 v1/v2 at update 1200, warmdown v1 changes CE by
`-0.0271779485/-0.0361252427`, accuracy by `+0.0016174316/+0.0026550293`, and
legal mass by `+0.0109306378/+0.0151808239`. The large replicated late-policy
regression is removed.

An exact second run processed `160,384` examples in 1,253 updates at `92.4980`
examples/s and independently selected update 800 at CE `4.5022441577`,
accuracy `0.1088714600`, and legal mass `0.6433968488`. Its selected-checkpoint
latent audit also passes: rank `31.0751/29.2224`, feature p05
`0.65108/0.63779`, target RMS `0.92711`, RMS ratio `0.97137`, and both trivial
controls beaten at all horizons. The two selected CEs differ by `0.0014834367`,
inside the preregistered K=2 repeat separation. All v2 states were deleted
after the audit.

The primary v1/update-1261 checkpoint then completed the frozen eight-pass
128-pair cap-256 arena against K=2 v1/update 800. It scored `0.50390625` with
pentanomial `[0,3,120,5,0]`, descriptive logistic Elo `+2.7144`, and pair-aware
95% interval `[-82.1983,+87.9592]`. Seven games were cap draws. One candidate
game was a fail-closed `no_representable_move` loss from the known incomplete
legacy promotion codec; the incumbent had no fault. Mean policy-call time was
`41.70 ms` for the candidate and `43.01 ms` for K=2. This is an
inconclusive-positive development screen, not an Elo promotion.

Accept cosine warmdown as the new offline incumbent. Retain only the selected
v1/update-1261 candidate state, SHA-256
`7bac3c49875ab35c3121c4471b65a40c08ce169a10915d4ebbb55661617cdc1a`;
all other candidate and repeat states are deleted. The arena state is
`artifacts/arena/cosine-warmdown-k2-u1261-vs-future-target-k2-u800-development-128pairs-cap256-v1/state.json`
with SHA-256
`342d41d70a97a5575a9cf055cc37c660ffe48eb812f3e1ad837b97e0d9d15ba7`.
