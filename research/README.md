# Local GPU research path

The implementation contract is
[docs/local_gpu_autoresearch_plan.md](../docs/local_gpu_autoresearch_plan.md).

Before running local commands:

```bash
source research/env.sh
```

This creates and exports workspace-local locations for temporary files,
dependency caches, JAX compilation artifacts, data, checkpoints, profiles, and
experiment runs. The script refuses to configure itself if the repository is
outside `/mountpoint/.exp`.

The intended stable interface is:

```text
prepare.py   fixed data, checkpoint, and evaluation support
train.py     the single editable architecture/training surface
program.md   rules for automated research
results.tsv  compact append-only experiment ledger
```

Those files are introduced incrementally after the legacy checkpoint has a
verified local-GPU baseline.
