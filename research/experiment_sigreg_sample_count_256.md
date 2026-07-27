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
