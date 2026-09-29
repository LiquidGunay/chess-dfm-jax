# Pending Torch epoch on Modal

Prepared 2026-09-29 on `research/modal-experiments` (base commit `cee8775`).
The active model is PyTorch in `research/train_torch.py`. The experiment
worktree is `.local/worktrees/modal-experiments` in the original checkout;
uncommitted course work remains on `research/interpretability-blog-lenses`.

The completed LR/decay screens selected Proposal A with equal encoder/main
LR `5e-5`, FP32 optimizer masters, BF16 compute, and decoupled decay. The
pending experiment is the frozen 7,774-update epoch over 7,960,576 examples.
The Modal wrapper uses the exact training command from
`research/analysis/proposal_a_one_epoch_preregister_20260810.json`; only its
provider execution changes. This retains the six recovery checkpoints,
update-6219 pre-decay state, terminal model, and 8,192-position validations at
updates 6,219 and 7,774. The full corpus is distinct from the tuning pilot.

Verify the full local corpus, then upload it to the existing training Volume:

```bash
/home/ubuntu/chess-dfm-jax/.venv/bin/python -m research.infra.proposal_a_epoch_data \
  --data-root /home/ubuntu/chess-dfm-jax/data/lc0_sequential/test80_20240401_0117
/home/ubuntu/chess-dfm-jax/.venv/bin/python -m research.infra.proposal_a_epoch_data \
  --data-root /home/ubuntu/chess-dfm-jax/data/lc0_sequential/test80_20240401_0117 --upload
```

Staging checks all 11,431 files (8,635,794,032 bytes) against the frozen
inventory hash. It uploads without overwrite and publishes a completion
marker only after a successful directory upload. The GPU worker verifies
the full inventory and train/validation scheduler identities, then copies
the hot corpus to container-local storage. No additional dataset conversion
or raw-model download is necessary.

When the live budget check permits:

```bash
modal run --detach --env chess-dfm-research research/infra/modal_train_app.py \
  --mode proposal_epoch --arm proposal_a_epoch --gpu L40S
```

This requests one L40S, four CPU cores, and 32 GiB host memory with a
nine-hour execution timeout, 15-minute startup timeout, and zero configured
retries. At the September 29 verified rates, the declared-attempt maximum is
$22.159152, under the $22.20 job cap. The earlier measured 2.591 seconds per
update suggests about 6.25 hours / $14.97 including ordinary overhead;
this is an estimate, not a measured full-epoch result. Modal can reschedule
crashed containers independently of configured retries, so the existing
workspace budget reserve still applies. Recovery uses the newest
checksum-valid state, preserves the original schedule, commits checkpoints
to the Volume immediately, and stitches interrupted metric segments.

No GPU was launched during preparation. The September 29 account summary
reported $22.71684919 gross metered usage, including $19.61684919 of fully
offset storage, and $0 billed. The corrected budget accounting excludes only
the provider's explicitly waived free-storage adjustment. Compute remains
counted before plan or promotional credits; both the $34 stop line and $42.50
outer budget remain unchanged. Budget usage is therefore $3.10, and the
declared-attempt projected total is $25.259152. Missing allowance metadata
falls back to gross usage, and inconsistent adjustments refuse launch.
The next calendar billing cycle begins October 1 at 00:00 UTC; query live
billing again before any launch.

The read-only storage inventory found about 233.53 GiB: 220.14 GiB training
results (211.43 GiB recovery states and 8.63 GiB final states), 10.98 GiB
training inputs, 1.16 GiB compilation cache, 1.18 GiB interpretability
inputs, and 0.07 GiB interpretability results. Archived attempts account for
35.73 GiB within the training-results total. No stored artifacts were deleted.
Modal currently includes 1 TiB/month of Volume storage; the listed rate beyond
the allowance is $0.09/GiB-month.

References: [pricing](https://modal.com/pricing),
[billing cycles](https://modal.com/docs/sdk/py/latest/Workspace#billingsummary),
[Volume accounting](https://modal.com/docs/guide/volumes).
