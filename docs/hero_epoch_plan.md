# Clean BT4 + DFM + JEPA Hero Epoch

Status: approved for execution on 2026-07-25.

This document is the immutable scientific and systems contract for the first
clean one-epoch local-GPU run. It supersedes the continuation-checkpoint
recipe for this run, but it does not erase the historical experiment record in
`docs/local_gpu_autoresearch_plan.md`.

## Goal

Train exactly one pass over the 28,343,296-example training split from:

- the original raw BT4 encoder and native policy head;
- fresh state-projector, DFM, JEPA, residual-policy, and WDL modules; and
- no parameters from the joint step-265,000 continuation checkpoint.

The run should answer:

1. Can a clean BT4 + DFM + JEPA model beat raw searchless BT4?
2. How do frozen validation losses and paired Elo change during the epoch?
3. What sustained training throughput and approximate MFU can the A10G
   achieve after deliberate PyTorch systems optimization?

No action-only pre-run is allowed. Action-only, closed-loop JEPA feedback,
alternative DFM probability paths, pass-count changes, and SAE work remain
post-hero ablations.

## Model contract

- Use the complete `lc0_canonical_1858` action codec for training,
  validation, and inference.
- Keep horizon 8 and eight inference refinement passes fixed.
- Use projector/DFM/JEPA depths `2/4/4`.
- Use balanced one-future-per-example JEPA target sampling while predicting
  all eight horizons.
- Retain the one-trainable-block future BT4 tail unless a correctness gate
  rejects it.
- Add the DFM root output as a residual over native BT4 policy logits:

  ```text
  root_logits = bt4_policy_logits + dfm_root_residual
  ```

- Zero-initialize the residual output so update zero is exactly raw BT4.
  Horizons 2--8 use the DFM outputs without a BT4 future-state anchor.
- Fine-tune the BT4 encoder and native policy branch at a lower learning rate
  than all fresh modules.
- JEPA remains a joint training auxiliary in this run. Predicted states do
  not feed back into later DFM passes.

The update-zero canonical arena must show exact raw-BT4 policy parity and zero
unrepresentable-move faults before training can start.

## Loss contract

Use:

```text
L =
    L_DFM_CE_H1_to_H8
  + alpha_root * L_root_legal_conditional_CE
  + 2.0 * L_root_illegal_mass
  + 1.0 * L_JEPA_raw_MSE
  + lambda_sig * SIGReg(z_target)
  + lambda_sig * SIGReg(z_pred)
  + 0.25 * L_WDL(z_pred)
```

Details:

- `L_DFM_CE_H1_to_H8` is uniform-horizon, masked-action, full-vocabulary CE.
- Calibrate `alpha_root` once on the deterministic fresh initialization so
  the weighted root legal-conditional term initially contributes `0.25`.
  Freeze it for the epoch.
- Remove RMS norm matching.
- Apply categorical WDL CE to every valid predicted horizon. Log expected
  value, value MSE, WDL accuracy, Brier score, entropy, and calibration, but
  do not add a separate scalar-value loss.
- The WDL labels are final-game outcomes, not LC0 root-Q or counterfactual
  action values. Keep the WDL term auxiliary.
- Remove scalar total-loss clipping. Retain non-finite fail-closed behavior
  and global gradient-norm clipping at `1.0`.

### SIGReg and batch size

Use the normalized V-statistic estimator. The provisional shared coefficient
is:

```text
lambda_sig = 2.0
```

Both target and prediction coefficients must always be identical. Before the
run, perform a deterministic no-update scalar and parameter-group gradient
audit. If the provisional coefficient is rejected, adjust the single shared
coefficient once and record the evidence; do not tune the two terms
independently.

Physical training batch size and SIGReg estimator sample size are separate:

- Changing physical batch size does **not** change the expectation or required
  coefficient of the normalized estimator.
- Keep `sigreg_example_count` fixed during systems batch-size comparisons so
  estimator variance and the fraction of directly regularized examples do
  not confound throughput results.
- If a later isolated profile shows that a larger SIGReg sample materially
  reduces gradient variance at negligible cost, treat that as a separately
  recorded scientific change. Do not rescale `lambda_sig` merely because the
  sample count changes; re-run the scalar/gradient audit instead.
- Never restore the old count-scaled `5.76` convention.

## Optimizer and schedule

- Retain the validated Muon/Nesterov-AdamW partition for the hero run.
- Provisional peak rates are `3e-4` for fresh modules and `1e-5` for raw-BT4
  modules.
- Before launch, run one resettable LR-range calibration on training data,
  sweeping the fresh rate from `1e-5` to `1e-3` while preserving the
  fresh/BT4 ratio. Use no validation data, reset all state afterward, and
  freeze the selected peaks.
- Express the schedule in examples, not updates:
  - linear warmup over 2% of the training examples;
  - cosine decay over the remaining 98%; and
  - final rate equal to `peak / 1000`.
- Use decoupled weight decay only on fresh matrix weights. Exclude biases,
  normalization scales, and embeddings. Apply no decay to raw-BT4 parameters
  in this first epoch.
- Choose the fresh-module decay by integrated shrink:

  ```text
  shrink = exp(-weight_decay * sum_t learning_rate_t)
  ```

  Target 5--10% shrink over the epoch. At peak `3e-4` with the planned cosine
  schedule, `weight_decay ~= 0.01` is the provisional point.

If the systems profile selects a different physical batch, preserve warmup
and decay boundaries in examples. Do not apply automatic linear learning-rate
scaling; repeat the short LR calibration at the selected batch.

## Frozen evaluation contract

Create checksummed, immutable, game-disjoint manifests before any
calibration:

- fast validation: 8,192 positions;
- primary validation: 65,536 disjoint positions;
- blind test: untouched during recipe selection; and
- paired arena openings with color reversal.

Run fast validation at 0%, 10%, ..., 100% of examples. Run primary validation
at initialization and completion. Record at least:

- H1 legal-conditional CE and top-1 accuracy;
- H1 full CE, all-horizon CE, and root legal mass;
- JEPA raw MSE by horizon and action-shuffle/identity/zero controls;
- target and prediction SIGReg, RMS, standard deviation, and effective rank;
- WDL CE, accuracy, Brier score, entropy, calibration, and expected-value MSE;
- component and parameter-group gradient norms; and
- examples/s, step timing, HBM, host RSS, power, utilization, and approximate
  MFU.

Run a small paired arena in process at 0%, 25%, 50%, and 75% of examples.
Run at least 1,024 color-reversed pairs at 100%, continuing if the confidence
interval does not resolve the raw-BT4 comparison. Report paired score,
descriptive logistic Elo delta, and a paired confidence interval.

A within-run loss/Elo trajectory is descriptive. It does not establish that
the relationship transfers across architectures or objectives.

Persist exactly:

- one halfway crash-recovery checkpoint;
- one terminal checkpoint;
- the fixed manifests, configs, metric curves, profiles, and reports needed
  to reproduce the result.

Delete the halfway checkpoint after the terminal checkpoint and its
cross-framework audit pass. Do not retain periodic checkpoints.

## Systems optimization contract

Performance engineering is a dedicated pre-launch phase. The accepted eager
batch-512 control is `160.310` examples/s and `49.11` hours per epoch, with
`12.571 GB` peak allocated HBM. Batch size alone is not considered an
adequate optimization.

All GPU work must use `research/run_gpu.sh`. Every compiler experiment must
retain the two-CPU affinity and fail-closed host-RAM, process-group-RSS, disk,
timeout, and 50 ms polling guards. All source, compiler caches, temporary
files, profiles, and outputs must remain under `/mountpoint/.exp`.

Establish a profiler-backed baseline and then test, one change at a time:

1. Production timing without per-update CUDA synchronization, CUDA events,
   scalar `.cpu()` transfers, or JSON writes. Keep detailed CUDA-event timing
   in bounded profile windows and log compact asynchronous training metrics
   every 10--20 updates.
2. Full rematerialization versus no/selective rematerialization at
   shard-compatible physical batches, selecting by sustained end-to-end
   examples/s within safe HBM.
3. PyTorch regional compilation of repeated BT4/projector/DFM/JEPA blocks,
   followed by broader compilation only if regional compilation is healthy.
   Start with one Inductor compiler worker, no max-autotune, static production
   shapes, workspace-local caches, and the existing resource guard.
4. Existing PyTorch fused primitives or maintained open-source kernels where
   numerical semantics match:
   - scaled-dot-product/Flash attention where XSA/QK-normalization semantics
     permit it;
   - fused/foreach optimizer operations;
   - fused normalization, SwiGLU, and projection epilogues; and
   - CUDA graphs only after the static step is correct and compiler-stable.
5. Use Nsight Systems/Compute or PyTorch profiler only in short guarded
   windows to identify launch gaps, tensor-core eligibility, memory-bound
   kernels, graph breaks, recompilations, and recomputation.

For every candidate, record cold compile time/RSS, warm examples/s, peak HBM,
numerical agreement, graph breaks/recompiles, and projected epoch time.
Reject any optimization that changes the frozen loss/evaluation semantics,
crosses a resource guard, or fails to provide sustained end-to-end benefit.

The rough 14% headline-BF16 MFU estimate is a diagnostic, not a conclusion.
The immediate target is at least `200 examples/s` (39.4 hours/epoch), with
`220 examples/s` (35.8 hours/epoch) as the first stretch target. Continue
profiler-guided work beyond those points when a measured bottleneck and a
maintained kernel path remain.

## Launch gates and execution order

1. Complete the remaining PyTorch checkpoint/JAX oracle verification.
2. Implement and audit canonical action conversion and raw-BT4 policy parity.
3. Implement deterministic fresh initialization and the residual root policy.
4. Implement the approved WDL, shared-SIGReg, root-action, optimizer, and
   example-based schedule contracts.
5. Freeze and checksum evaluation manifests.
6. Pass CPU unit/parity tests and guarded GPU forward/update/evaluation smoke.
7. Complete the guarded systems optimization matrix and freeze the fastest
   healthy runtime/batch.
8. Run the no-update gradient audit and resettable LR calibration at that
   batch.
9. Record the final immutable config and launch exactly one epoch.
10. Complete terminal validation, paired Elo, cross-framework checkpoint
    audit, compact retention, and the loss/Elo report before any ablation.
