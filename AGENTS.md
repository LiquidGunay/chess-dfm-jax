# Working on the DFM experiment branch

The active training implementation is PyTorch in `research/train_torch.py`.
The repository name and legacy JAX/TPU files are historical; do not route new
experiments to JAX by default.

Start with `research/experiments/registry.json` and
`docs/dfm_research_program.md`. The current order is the selected full epoch,
searchless strength evaluation, conditional matched-search evaluation against
LC0, then strength-qualified architecture/representation comparisons.
Continuous diffusion in embedding space is a recorded hypothesis, not a
scheduled run. The user will choose the architecture strength threshold after
the first searchless evaluation; freeze it before inspecting architecture
comparisons.

Keep one stable experiment ID and immutable specification per scientific
question. Bind remote runs to the source commit/content hashes, data identity,
seed, configuration, and declared compute budget. Preserve prior evidence;
do not rewrite a preregistration after seeing results. Separate tooling,
scientific changes, plans, and run receipts into reviewable commits.

Use the budget-gated Modal launchers. Count compute before promotional
credits; exclude only explicitly provider-waived storage with validated
accounting. Refresh usage before launches, account for concurrent jobs, and
retain the existing $34 stop line and $42.50 outer budget. GPU work must not
start until full input verification succeeds.

Follow the storage review thresholds and retention rules in
`research/experiments/README.md`. Preserve selected models, active recovery
states, pre-decay continuation states, source identities, and compact evidence.
Never delete an entire Volume as routine cleanup.

This worktree is separate from uncommitted interpretability-course work in
the original checkout. Do not absorb or discard those changes. The original
checkout's Torch-only `.venv` and dataset may be reused explicitly without
copying multi-gigabyte local assets. Use targeted Torch/infrastructure tests;
the historical project dependencies include JAX and are not the active
environment contract.
