# Phase 2: TPU Scaling Sweep Error Log

This document tracks technical hurdles and resolutions encountered during the massively parallel JEPA scaling sweep on Spot TPUs.

## 1. Multi-host Concurrency & Handshaking

### GCS Write Collisions (`IndexError`)
- **Issue**: Parallel workers in a TPU pod all attempted to validate/create the same GCS checkpoint directory simultaneously.
- **Root Cause**: Race condition in `google-cloud-storage` during bucket/folder validation.
- **Resolution**: Gated all GCS checkpoint and W&B operations to `jax.process_index() == 0`.

### Distributed Handshake Deadlock (`SIGABRT` / `DEADLINE_EXCEEDED`)
- **Issue**: JAX workers hanging indefinitely during `jax.distributed.initialize()`.
- **Root Cause**: Standard 5-minute timeout is insufficient when pods are performing heavy `uv pip` installs or source syncing concurrently across 4+ hosts.
- **Resolution**: Increased `initialization_timeout` to 1200s (20 mins).

### Orbax Manager Deadlock
- **Issue**: TPU VM process stalling at startup with `futex_wait`.
- **Root Cause**: Gating the `CheckpointManager` constructor to Process 0. In a distributed JAX environment, all hosts must initialize the manager to satisfy internal barriers, even if only one host performs the actual I/O.
- **Resolution**: Removed the Process 0 guard from the constructor; only gated the `.save()` call.

### Hidden Distributed Barriers (`CheckpointManager`)
- **Issue**: Even after gating saves to Process 0, training still stalled at `futex_wait` during checkpoint steps.
- **Root Cause**: Orbax's `CheckpointManager` internally attempts to coordinate directory creation and metadata updates across all hosts in a JAX cluster unless a specific `multiprocess_context` is provided to disable it.
- **Resolution**: Removed `CheckpointManager` entirely. Implemented manual Step/Directory management and cleanup on Process 0 using a bare `Checkpointer(PyTreeCheckpointHandler())`, which is guaranteed to be local and synchronous.

### Orbax `sync_global_devices` Deadlock
- **Issue**: Training crashing during checkpointing with `sync_global_devices name mismatch`.
- **Root Cause**: Gating the entire `save_training_checkpoint` call to Process 0. Even with async saving, Orbax's internal handler may trigger global JAX barriers.
- **Resolution**: Refactored to ensure all workers call the save function, but only Process 0 executes the actual `checkpointer.save()` and GCS I/O.

---

## 2. Environment & Dependencies

### TPU Boot Failures (`HOME` unbound)
- **Issue**: Startup script failing with `HOME: unbound variable`.
- **Root Cause**: TPU VM base images sometimes have minimal environments where `$HOME` isn't exported to the root shell during systemd/startup-script execution.
- **Resolution**: Explicitly added `export HOME=/root` to the top of the startup sequence.

### Missing `gcloud` in Startup Script
- **Issue**: Startup script failing with `gcloud: command not found` during the final status upload.
- **Root Cause**: Minimal shell environments during systemd/startup execution may not include `/snap/bin` in the default `PATH`. On Ubuntu-based TPU VMs, `gcloud` is often installed via Snap, but symlinks in `/snap/bin` may not be initialized during the initial boot phase.
- **Resolution**: Refactored the template to use the absolute raw snap path `/snap/google-cloud-cli/current/bin/gcloud` for every storage command.

### Missing GCS Support (`ModuleNotFoundError: gcsfs`)
- **Issue**: Orbax failing to save to `gs://` paths.
- **Root Cause**: Orbax requires the `gcsfs` backend for GCS interaction, which wasn't in the base dependencies.
- **Resolution**: Added `gcsfs` to `pyproject.toml`.

---

## 3. Persistent Recovery (Checkpointing)

### GCS Directory Discovery (`FileNotFoundError`)
- **Issue**: `latest_checkpoint_step` returning `None` despite checkpoints existing in GCS.
- **Root Cause**: `etils.epath` using `iterdir()` or shallow `glob("step*")` on GCS, which fails because GCS is a flat object store. These calls often fail to find "directories" unless they have explicit marker blobs.
- **Resolution**: Switched to a deep glob `base.glob("step*/state.npz")` to robustly discover completed checkpoint files.

### Resume Initialization Order (`UnboundLocalError`)
- **Issue**: Training script crashing during resume with `cannot access local variable 'model'`.
- **Root Cause**: The script attempted to load weights into the `model` object before `create_jepa_components()` was called.
- **Resolution**: Refactored `scripts/train_jepa.py` to initialize all model/optimizer components before entering the resume/checkpoint block.

### Checkpoint Modulo Skipping
- **Issue**: Checkpoints not being saved at expected step counts (e.g., 100, 200).
- **Root Cause**: Simple `completed_step % save_every == 0` check fails if the loop's logging or control flow skips that exact step (e.g., jumping from 91 to 101).
- **Resolution**: Implemented a "boundary-crossing" check: `(completed_step // save_every) > (step // save_every)`. This guarantees exactly one save per interval even if the exact multiple is bypassed.

### Public IP Quota Exhaustion (`IN_USE_ADDRESSES`)
- **Issue**: New TPU requests failing immediately with `You have reached IN_USE_ADDRESSES limit`.
- **Root Cause**: By default, each TPU VM worker is assigned a public IP. Rapidly cycling through multi-host pods (e.g., 4 hosts for `v5litepod-16`) exhausts the project's regional static IP quota.
- **Resolution**: Explicitly set `enable_external_ips: False` in the job spec, which forces the use of the `--internal-ips` flag during creation.

### NPZ Payload Mismatch
- **Issue**: Resume failing with `expected Mapping; got ndarray`.
- **Root Cause**: When saving a nested dictionary via `np.savez(**payload)`, NumPy sometimes pickles the entire dict into a single `arr_0` file if the structure is complex. The loading logic was incorrectly attempting to unwrap this even when it was already a valid mapping.
- **Resolution**: Refined the unwrap logic to only call `.item()` if `arr_0` is the *only* file in the NPZ archive, ensuring the resulting payload is always the original dictionary.

---

## 4. Performance & Infrastructure

### Synchronous GCS Latency (Training Hitching)
- **Issue**: Training steps spiking from ~30ms to ~500ms during checkpoint steps.
- **Root Cause**: Synchronous GCS I/O blocking the main XLA execution thread.
- **Resolution**: Pivoted to saving all checkpoints to a local VM directory first, then using `subprocess.run()` to sync to GCS.

### Background Sync Failure (`Popen`)
- **Issue**: Local checkpoints existed on disk but never appeared in GCS prefix.
- **Root Cause**: `subprocess.Popen` background sync is not guaranteed to survive a rapid VM preemption and lacks visibility into error states (like auth failures or path mismatches).
- **Resolution**: Switched back to synchronous `subprocess.run()` for the final verification phase to ensure the upload is finalized before the training loop continues.

### Shell Pipe Deadlock (Silent Logs)
- **Issue**: Worker 0 logs appearing "stuck" for 10+ minutes despite training being active.
- **Root Cause**: Shell redirection (`tee -a`) combined with Python's internal buffering filling up the pipe.
- **Resolution**: Added explicit `sys.stdout.flush()` after all major logging and checkpointing events.

### DFM Checkpoint Serialization (`AssertionError: expected Mapping; got ndarray`)
- **Issue**: Evaluating the DFM checkpoint failed because `opt_state` was restored as a 0-D object array of tuples instead of a dict, causing `flax.nnx` traversals to crash.
- **Root Cause**: `np.savez` flattens dictionaries with complex keys (like tuples) into `arr_0` 0-D object arrays, breaking the expected Mapping interface.
- **Resolution**: Modified `chess_dfm_jax/training/checkpoints.py` to correctly unwrap 0-D object arrays back into native Python dictionaries upon loading. Also made the `optimizer` state restoration optional in `restore_train_state` since inference scripts (like `uci_dfm.py`) do not require the AdamW/Muon optimizer momentum tensors to run forward passes.

### Policy Head Dimensionality (`ValueError: operands could not be broadcast together with shapes (1858,) (1024,) ()`)
- **Issue**: `evaluate_elo.py` crashed when falling back to the raw BT4 policy because the output logits had shape `[1024]` instead of `[1858]`.
- **Root Cause**: The `PolicyHead` in `nnx_bt4.py` was misimplemented as a simple linear projection. In the actual BT4 architecture, the Policy Head is a complex bilinear attention mechanism (`q @ k^T`) combined with promotion mappings.
- **Resolution**: Refactored `PolicyHead` in `nnx_bt4.py` to exactly match the bilinear attention and `mapping_table` slicing logic found in the official `reference_bt4.py` implementation. Verified the 1858-dim logits via the UCI tournament manager.
- **Root Cause**: Architectural refactoring renamed the underlying transformer component to `EncoderLayer` in `nnx_bt4.py`, but the JEPA training script was still referencing the old name.
- **Resolution**: Updated `chess_dfm_jax/training/jepa.py` to correctly import and instantiate `EncoderLayer`.

### GCS Metadata Caching & Stale Source Sync
- **Issue**: TPU instances repeatedly failing with old `KeyError` exceptions even after code was fixed locally and re-uploaded.
- **Root Cause**: `gcloud` and TPU VMs cache the `startup-script` and downloaded tarballs. If an experiment ID is reused or the `override_source_uri` points to the same object name (e.g., `verified_source_v1.tar.gz`), GCP fetches the old, cached code instead of the newly uploaded fixes.
- **Resolution**: 
  1. Implemented strict versioning for the experiment IDs (e.g., `v5`) and the immutable snapshot names (`verified_source_v5.tar.gz`). Every code change now requires a bump to both to break the cache.
  2. Added a pre-flight test script (`tests/test_training_step.py`) to `start_sweep.sh` that validates the model shapes locally on the controller before any TPUs are requested.

### Shape Mismatch in BT4 Encoder (`Smolgen` & Heads)
- **Issue**: Training failing with `TypeError: dot_general requires contracting dimensions to have the same shape`.
- **Root Cause**: The `Smolgen` block and head layers (`ValueHead`, `MovesLeftHead`) were oversimplified as single linear layers, whereas the LC0 BT4 architecture uses multi-stage projections and shared generator matrices.
- **Resolution**: Re-implemented `Smolgen` as a multi-stage projection inside `EncoderLayer` and added missing intermediate dense layers to the heads, verified via local forward pass.

### Data Pipeline `StopIteration` (Shuffled vs Sequential Data)
- **Issue**: `LeelaChunkDataLoader` returned 0 valid sequences and the training script crashed immediately with `StopIteration`.
- **Root Cause**: The official Lc0 `.gz` chunk archives contain shuffled, single-position board states. Our JEPA and DFM architectures require unrolling sequential trajectories (e.g., `horizon=8`). You cannot extract an 8-move trajectory from a bag of shuffled independent positions.
- **Resolution**: Reverted to processing high-strength engine games (TCEC PGNs) via `scripts/process_tcec_to_gcs.py`. This script slices the games into contiguous 8-ply horizons, encodes the boards, extracts the true game outcomes for Value/WDL targets, and saves them as `.npz` chunks.

### DFM Formulation Mix-up (Masked Diffusion vs Discrete Flow Matching)
- **Issue**: The initial DFM implementation used a heuristic cosine masking schedule, which is standard Masked Diffusion, not true Discrete Flow Matching (DFM).
- **Root Cause**: Conflation of discrete diffusion methodologies.
- **Resolution**: Refactored `dfm_loss_fn` in `chess_dfm_jax/training/dfm.py` to use the rigorous DFM continuous-time linear probability path ($P(x_t = M | x_1) = 1 - t$ and $P(x_t = x_1 | x_1) = t$) and transition rate objective (categorical cross-entropy evaluated on masked tokens).

### DFM Checkpoint Serialization (`AssertionError: expected Mapping; got ndarray`)
- **Issue**: Evaluating the DFM checkpoint failed because `opt_state` was restored as a 0-D object array of tuples instead of a dict, causing `flax.nnx` traversals to crash.
- **Root Cause**: `np.savez` flattens dictionaries with complex keys (like tuples) into `arr_0` 0-D object arrays, breaking the expected Mapping interface.
- **Resolution**: Modified `chess_dfm_jax/training/checkpoints.py` to correctly unwrap 0-D object arrays back into native Python dictionaries upon loading. Also made the `optimizer` state restoration optional in `restore_train_state` since inference scripts (like `uci_dfm.py`) do not require the AdamW/Muon optimizer momentum tensors to run forward passes.

### Policy Head Dimensionality (`ValueError: operands could not be broadcast together with shapes (1858,) (1024,) ()`)
- **Issue**: `evaluate_elo.py` crashed when falling back to the raw BT4 policy because the output logits had shape `[1024]` instead of `[1858]`.
- **Root Cause**: The `PolicyHead` in `nnx_bt4.py` was misimplemented as a simple linear projection. In the actual BT4 architecture, the Policy Head is a complex bilinear attention mechanism (`q @ k^T`) combined with promotion mappings.
- **Resolution**: Refactored `PolicyHead` in `nnx_bt4.py` to exactly match the bilinear attention and `mapping_table` slicing logic found in the official `reference_bt4.py` implementation. Verified the 1858-dim logits via the UCI tournament manager.
