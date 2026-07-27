# Experiment: SIGReg sample count 256

## Question

Does reducing finite-sample noise in both target and prediction SIGReg improve
early fresh-initialization learning relative to the frozen 64-example
estimator?

## Preregistered comparison

Run two independent processes sequentially on the single A10G. Both start from
the same raw BT4 parameters plus identically seeded fresh DFM/JEPA modules and
consume the same examples in the same order.

| Setting | Control | Candidate |
|---|---:|---:|
| SIGReg example count | `64` | `256` |
| Target SIGReg coefficient | `2.0` | `2.0` |
| Prediction SIGReg coefficient | `2.0` | `2.0` |
| Physical batch | `1,024` | `1,024` |
| Optimizer updates | `1,024` | `1,024` |
| Examples | `1,048,576` | `1,048,576` |
| Seed / data start | `0 / 0` | `0 / 0` |

All other model, loss, optimizer, example-based LR schedule, canonical action
codec, eight-pass inference, and systems settings remain exactly equal. Use
the selected `eager-fused-backward` LayerNorm backward, regional
`torch.compile`, and after-forward prefetch runtime. Write every loss
component to `metrics.jsonl`, one immutable `run_config.json`, and only one
terminal model-only checkpoint per run.

The 1,024-update budget is a discovery screen, not a promotion run. It covers
3.70% of one epoch and extends 481 updates beyond the 566,866-example warmup;
a 30-minute run would remain almost entirely inside warmup.

## Evaluation and decision

Evaluate both checkpoints on the same frozen fast validation pool with the
fixed count-64 evaluation estimator. Then run the same 128 color-reversed
opening pairs against raw BT4, with eight refinement passes, inference batch
cap 16, and additional-ply cap 256.

Rank the experiment primarily by frozen validation DFM CE, action accuracy,
legal mass, JEPA MSE, target/prediction rank diagnostics, and WDL metrics.
Arena score is a secondary, low-power chess-strength screen at this exposure.
Record all results even if the candidate loses.

Advance count 256 only if it improves the frozen validation profile without a
collapse/rank regression and is directionally non-worse in Arena. A positive
candidate must be repeated and extended to at least 10% of an epoch before it
can replace count 64. Do not rescale the coefficient as part of this
experiment.

After evaluation, retain the winner's model checkpoint. Delete the loser's
712 MB model only after the comparison report records its checksum and all
scalar/evaluation evidence is complete.

## Result

Complete and rejected on 2026-07-27. Both guarded runs processed exactly
1,048,576 matched examples in 1,024 updates with no non-finite skips. Count
256 is systems-neutral: it changes end-to-end throughput from `254.570` to
`254.330` examples/s (`-0.094%`) and adds 124 MiB peak allocated HBM.

On the identical 8,192-example frozen validation pool, evaluated with the
fixed count-64 estimator, count 256 changes:

- DFM CE from `5.706299` to `5.702968` (`-0.003330`, or `-0.058%`);
- action accuracy from `7.2433%` to `7.4875%` (`+0.2441` percentage points);
- JEPA MSE from `0.133190` to `0.117179` (`-12.02%`);
- WDL CE from `0.771990` to `0.788461` (`+2.13%`); and
- horizon-8 prediction/target effective rank from
  `11.1652/10.6859` to `10.9265/10.4867`.

There is no latent collapse: feature standard deviations remain healthy and
prediction-target cosine increases. The candidate nevertheless has a modest
rank and WDL regression.

The preregistered same-opening Arena screen is decisive in the opposite
direction. Count 64 scores `0.908203` against raw BT4, while count 256 scores
`0.812500`; the paired candidate-minus-control delta is `-0.095703`.
Count 256 is worse on 58 opening pairs, equal on 51, and better on 19. The
descriptive paired-t 95% interval is `[-0.133370, -0.058036]`, with zero
faults and zero cap draws in either run. The corresponding descriptive
logistic-Elo estimates differ by `-143.41`, but remain relative to this raw
BT4 implementation and frozen pool.

Reject count 256 and do not extend it to 10% of an epoch. Retain count 64 and
coefficients `2.0/2.0` as the fresh-research baseline. The result does not
prove finite-sample noise is generally beneficial; it shows that reducing it
is not a useful change for this exact short-run recipe. The rejected model
weights were deleted after checksum
`c5bd59891b14b9edd2a6b221d06efe792901c6486b03ce2f99ab76fdee22a495`
and all evidence were sealed. The compact machine-readable comparison and
loss plot are:

- `research/analysis/sigreg_sample_count_256_experiment_20260727.json`
- `research/analysis/sigreg_sample_count_256_experiment_20260727.png`
