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
