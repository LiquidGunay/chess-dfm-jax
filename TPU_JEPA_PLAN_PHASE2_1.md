# Phase 2.1: Multi-Task JEPA with SigReg and Value Prediction

## Objective
Train the JEPA predictor using real BT4 self-play chunks (`.zst`) with a multi-task loss. The goal is to perfectly align the predicted latent state with both the future encoder embedding (via SigReg) and the future game outcome (via Value/WDL prediction). 

## 1. Data Pipeline Updates
- **Source Data**: Transition from synthetic or PGN-derived `.npz` files to raw Leela/BT4 `.zst` chunks.
- **Float Extraction**: Update `chess_dfm_jax/data/leela.py` to extract the 15 floats stored in the `TrainingRecord` (specifically Q-value, WDL Win, Draw, Loss probabilities).
- **Target Formatting**: Provide these parsed floats as targets (`value_target`, `wdl_target`) in the transition batch.

## 2. Architectural Updates
- **Action Conditioning**: Replace the simple action embedding addition with an `ActionMLP`. It will project the discrete action index into a continuous embedding (e.g., 128 dims), then through a 2-layer MLP with Swish activation to match the JEPA hidden dimension, and finally add it to the latent state. This provides a richer transition condition.
- **Value Prediction Head**: Attach a new MLP head to the predicted latent state ($z_{t+k}$) to output a single Q-value and a 3-class WDL probability distribution.

## 3. Loss Function & Curriculum (SigReg)
- **SigReg Loss**: Implement Sketched Isotropic Gaussian Regularization (SIGReg) as described in the Le World Model paper. This computes the empirical variance/covariance along random 1D projections to enforce an isotropic Gaussian distribution, preventing representation collapse without needing stop-gradients or EMA.
- **Composite Loss**:
  $$\mathcal{L} = \mathcal{L}_{sim}(z_{pred}, z_{target}) + \lambda_{sig} \mathcal{L}_{sigreg}(z_{pred}) + \lambda_v \mathcal{L}_{value}(v_{pred}, v_{target}) + \lambda_{wdl} \mathcal{L}_{wdl}(w_{pred}, w_{target})$$
- **Curriculum Stage 1 (Frozen Backbone)**: Train the `ActionMLP`, JEPA Predictor, and `ValuePredictionHead` while keeping the `BT4Encoder` perfectly frozen.
- **Curriculum Stage 2 (End-to-End)**: Once the composite loss saturates, unfreeze the `BT4Encoder` to fine-tune the representations for multi-step planning.

## 4. Pre-Flight Testing & Deployment
- Update `tests/test_training_step.py` to test the new architecture, ensuring the composite loss, ActionMLP, and Value Heads compile correctly.
- Use `start_sweep.sh` to run the pre-flight check automatically.
- Re-enable Weights & Biases for proper run tracking across the distributed TPU pod.
