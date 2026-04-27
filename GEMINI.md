# Gemini CLI Foundation Mandates

## Core Principles
- **Credential Protection**: Never log, print, or commit secrets.
- **Contextual Precedence**: Instructions in this file take absolute precedence.
- **Surgical Updates**: Prefer targeted edits to large-scale refactors.
- **Validation is Finality**: A task is only complete once verified with automated tests.
- **No Fake Code**: NEVER fake any part of the pipeline or replace a complex component with a simplified or placeholder implementation. All mathematical, optimization, and training logic must be rigorous and empirically verified against the Lc0 reference logic.

## Research & Scaling Guidelines
- **Trial Quota Management**: We use Spot TPUs on GCP Trial Quota (64 chips).
- **Region Awareness**: Multi-host pods must stay within regional buckets for efficient data/checkpoint syncing.
- **Verified Deployments**: **ALWAYS** verify model architecture, weight mapping, and data pipelines with a local forward pass on the controller VM via `tests/test_pipeline_e2e.py` *before* provisioning expensive TPU resources. 
- **Infinite Retry Loop**: The orchestrator (`sweep_manager.py`) is designed to poll for capacity indefinitely; do not interrupt unless changing the experiment matrix.

## Technical Standards
- **JAX/Flax NNX**: Use `nnx.Module` for all new model components.
- **Checkpointing**: Use raw NumPy (`np.savez`) for Process 0 saves and synchronous GCS uploads to bypass Orbax deadlocks.
- **Data Loading**: Stream Lichess ZST chunks directly to GCS `.npz` files for zero-copy training.

## Current Project Status (chess-dfm-jax)
- **Scope Shift**: The repository has transitioned from standard JEPA pretraining into **Discrete Flow Matching (DFM)** applied to Categorical Diffusion for searchless chess planning.
- **Model Horizon**: We have empirically reduced the unrolled prediction horizon to **K=4 plies**, which dramatically reduces the combinatorial memory burden on the 4-layer and 12-layer models. The data pipeline correctly truncates the unrolled target actions to match this configuration.
- **Muon Optimizer**: We have integrated and rigidly tested `optax.contrib.muon` (Newton-Schulz orthogonalization) specifically for the 2D linear matrices, falling back to AdamW for 1D biases and layer norms.
- **Legality Penalty**: The training loss is successfully utilizing a heavily weighted legality penalty (currently set to 2.0) applied exclusively to the first step of the unrolled horizon. This has resulted in the model naturally dropping its legality loss from ~0.9 to ~0.2 within 200,000 steps without manual hard-masking.
- **Spot Orchestration**: To combat aggressive Spot preemption on Google Cloud, training jobs are managed by an autonomous `sweep_manager.py` daemon running on a persistent VM (`dfm-orchestrator-vm-7`). The orchestrator supports **cross-region floating**, allowing preempted jobs in US datacenters to seamlessly jump to idle quotas in Europe while continuing to write checkpoints to the same central GCS bucket.
- **Parity Verified**: The NNX architecture has been meticulously verified against the official LC0 BT4 reference implementation. Zero "fake" code or simplifications are allowed.
