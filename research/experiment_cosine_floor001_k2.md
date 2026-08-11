# Experiment 007: one-percent cosine floor

Status: completed and rejected on 2026-07-21. The 1% floor improved the
incumbent point estimate but missed the frozen repeat-noise-aware CE ceiling
by `0.0000760853`; no repeat or arena was run and no candidate state remains.

## Question and hypothesis

Experiment 006 establishes that fixed-time cosine warmdown removes the severe
constant-rate update-1200 regression. Its two exact runs do not completely
agree on the low-rate tail: v1 improves from CE `4.5032504834` at update 800 to
`4.5007607210` at terminal, while v2 selects update 800 at `4.5022441577` and
is nearly flat thereafter (`4.5027645919` at update 1200 and `4.5023701042` at
terminal).

This experiment asks whether the retained 10% floor still permits avoidable
late drift. The sole change is a 1% floor. The primary hypothesis is that the
lower floor improves selected two-pool DFM CE beyond the accepted
cosine-warmdown repeat envelope without damaging policy, legality, latent
health, or throughput. The secondary hypothesis is that update-1200 and
terminal quality are at least as stable as the 10%-floor repeats.

This is an optimizer hyperparameter experiment only. It keeps both JEPA
projector blocks, all four DFM blocks, K=2 future-target sampling, all eight
JEPA predictions, eight searchless refinement passes, and the normalized
loss coefficients `0.0/5.76/1.0` for norm/target-SIGReg/prediction-SIGReg.

## Candidate contract

Relative to the accepted Experiment 006 implementation, change only:

```python
"lr_min_ratio": 0.01
```

The shared multiplicative main/BT4 schedule is therefore:

- optimizer updates 0 through 399: ratio `1.0`;
- updates 400 through 1199: cosine decay from `1.0` to `0.01` over 800
  schedule steps;
- update 800: ratio `0.505`;
- update 1200 and later: ratio `0.01`;
- floor main/BT4 rates `3e-7/1e-8`; and
- no warmup.

Every other model, objective, optimizer, data, checkpoint, validation, and
inference setting remains byte-for-byte configured as in Experiment 006.
Reports and strict resume contracts must identify the new floor and exact
boundary values. The default-off constant and historical warmup paths must
remain unchanged.

## Frozen control and run configuration

The offline incumbent is cosine-warmdown v1/update 1261:

- state SHA-256:
  `7bac3c49875ab35c3121c4471b65a40c08ce169a10915d4ebbb55661617cdc1a`;
- two-pool DFM CE `4.5007607210`;
- accuracy `0.1094207764`;
- legal mass `0.6473310636`;
- cached profile `93.2987` examples/s;
- 30-minute throughput `92.8796` examples/s; and
- exact-repeat selected-checkpoint CE separation `0.0014834367`.

Start from the recovered step-265,000 model with a fresh optimizer. Freeze one
NVIDIA A10G and no concurrent workload; physical/evaluation batches `128/64`;
train/data seed 0 with `global_permutation`; peak main/BT4 rates
`3e-5/1e-6`; 1,800 seconds of steady-state training after compile/first
update; validation seeds 10,000 and 20,000, each 64 batches of 64 examples;
and saves at updates 400, 800, 1200, and terminal with at most four temporary
states.

## Correctness and performance gates

Before the 30-minute run:

1. CPU tests confirm active override precedence and exact schedule values at
   updates 0, 399, 400, 800, 1199, and 1200 while preserving optimizer ABI and
   update-0 exactness.
2. A one-update real-checkpoint A10G smoke has finite loss/gradients, full
   architecture depths, three encoded boards per example, eight predictions,
   SIGReg counts `576/512`, no clipping/skipped update, and no write outside
   `/mountpoint/.exp`.
3. A cached 30-update profile must reach at least `88.6337` examples/s, 95% of
   the accepted cosine profile.

## Checkpoint selection and decision

Evaluate every saved checkpoint on the identical seed-10,000 and seed-20,000
full-horizon pools and select minimum mean DFM CE. Acceptance requires:

- two-pool DFM CE below `4.4992772844`, more than the accepted
  `0.0014834367` repeat separation better than the incumbent;
- accuracy at least `0.1074207764`;
- legal mass at least `0.6443310636`;
- every metric and gradient finite;
- positive JEPA prediction beats zero and action-shuffled predictions at every
  horizon;
- mean/minimum prediction effective rank at least `30.0/28.5`;
- mean/minimum prediction feature-std p05 at least `0.63/0.61`;
- mean target RMS at least `0.90`; and
- prediction/target RMS ratio in `[0.94, 1.02]`.

Separately compare update-1200 and terminal CE/accuracy/legal mass with both
10%-floor runs. If the first run passes, repeat it exactly and require the
repeat to beat the cosine incumbent CE while passing every non-CE gate before
the frozen 128-pair arena. A failure gets no repeat, arena, pass-count sweep,
or SAE refit. Retain only a qualifying selected state; otherwise delete all
candidate states after preserving compact evidence.

## Outcome

Commits `f832edf` and `181c7e9` preregister and activate the 1% floor. The
implementation passed 54 focused CPU tests plus Ruff. The real-checkpoint
one-update A10G smoke stayed finite, used the full two-projector/four-DFM
graph, encoded three boards per example, retained all eight predictions,
reported target/prediction SIGReg counts `576/512`, did not clip or skip the
update, and peaked at `9,474,440,192` bytes of HBM. The cached 30-update
profile reached `91.3181` end-to-end examples/s, above the `88.6337` gate and
`2.12%` below the accepted 10%-floor profile.

The fixed 1,800-second run processed `158,720` examples in 1,240 updates at
`91.6046` steady end-to-end examples/s and peaked at `9,585,710,848` bytes.
Every recorded value remained finite and no loss clip activated. The frozen
two-pool scan was:

| update | DFM CE | accuracy | legal mass |
| ---: | ---: | ---: | ---: |
| 400 | 4.5114891175 | 0.1085662842 | 0.6424590521 |
| 800 | 4.5022034571 | 0.1091766357 | 0.6452634027 |
| 1200 | **4.4993533697** | **0.1100006104** | **0.6472349875** |
| 1240 | 4.4993873630 | 0.1098632812 | 0.6471266039 |

Update 1200 wins and improves the 10%-floor incumbent by `0.0014073513` CE.
Acceptance required an improvement greater than the accepted repeat
separation, or CE below `4.4992772844`; the candidate misses that strict
ceiling by `0.0000760853`. Accuracy and legal mass pass their gates, so this
is a primary-CE-only rejection rather than a policy or legality regression.

The mandatory seed-10,000 free-rollout audit also passes every latent gate.
Mean/minimum prediction effective rank is `30.9780/29.0780`, mean/minimum
feature-std p05 is `0.65079/0.63758`, mean target RMS is `0.92941`, and the
prediction/target RMS ratio is `0.96882`. Genuine JEPA predictions beat zero
and action-shuffled controls at all eight horizons. Prediction SIGReg remains
sufficient to prevent collapse without RMS norm matching in this trial.

The secondary tail-stability result is positive. At update 1200 the 1% floor
improves CE versus the two 10%-floor runs by `0.0021424647/0.0034112222`, and
its terminal CE is only `0.0000339933` above its selected update-1200 value.
The terminal also improves the two accepted-run endpoints by
`0.0013733581/0.0029827412`. One run is not enough to establish that smaller
floor as a repeat-stable gain, and the frozen primary gate deliberately
requires more than this point estimate.

Per the preregistration, the experiment gets no exact repeat, arena,
pass-count sweep, or SAE refit. All four 1.85 GB candidate states were
deleted; compact manifests, training metrics, profile, paired scan, latent
audit, and this decision remain. The active override returns to the accepted
10% cosine floor.
