# Phase 3: DFM (Categorical Diffusion) Planner

## Objective
The goal is to test if the combination of a JEPA dynamics model and a Discrete Flow Matching / Categorical Diffusion Model (DFM) can create a stronger **searchless** chess engine than the raw BT4 policy head alone. 

Since unfreezing the encoder without predicting actual action trajectories isn't maximizing its potential as a planner, we will now build the DFM policy head on top of the JEPA latents.

## 1. The DFM Architecture
The DFM acts as the "Planner". It learns the reverse diffusion process to denoise a sequence of future actions $a_{t:t+k}$ conditioned on the current board state $z_t$.

- **Conditioning**: The frozen BT4 encoder provides the rich initial state embedding $z_t$.
- **Action Trajectory**: The sequence of $K$ future actions $A = (a_t, a_{t+1}, \dots, a_{t+K-1})$.
- **Forward Process (Noising)**: We corrupt the action sequence by uniformly replacing discrete actions with a `[MASK]` token or random actions according to a categorical diffusion schedule (e.g., D3PM / Masked Diffusion).
- **Reverse Process (Denoising Network)**: A small Transformer or deep MLP that takes:
  1. The latent condition $z_t$
  2. The noisy action trajectory $A_{\tau}$ 
  3. The diffusion timestep $\tau$
  And predicts the clean, optimal action trajectory $A_0$ (or the marginal probabilities of the next denoising step).

## 2. Training Strategy
- **Data**: We will reuse the exact same TCEC / Lc0 `.zst` high-strength dataset we collected for the JEPA training, parsing the `action_indices` of length $K=8$.
- **Loss**: Cross-Entropy loss between the DFM's predicted clean categorical action logits and the true Lc0 engine actions from the dataset.
- **Independence**: The DFM can be trained concurrently with or immediately after the JEPA. Because the BT4 encoder is frozen (or separately fine-tuned via JEPA), the DFM simply learns to interpret $z_t$ as a starting point for generating a highly probable winning sequence of moves.

## 3. Inference (The "Searchless" Engine)
Once trained, we will build an interactive script (`notebooks/play_dfm.py`):
1. User provides a FEN.
2. BT4 encodes the board into $z_t$.
3. We initialize a completely random/masked sequence of $K$ actions.
4. We iteratively pass it through the DFM for $N$ timesteps until we get a clean sequence of $K$ actions.
5. The engine plays the first action $a_t$ of the denoised trajectory. 
6. We benchmark this against the raw 1-ply BT4 policy output to definitively prove if JEPA+DFM improves searchless Elo!

## Next Steps
1. Scaffold the Categorical Diffusion schedule (masking, transition matrices) in `chess_dfm_jax/training/dfm.py`.
2. Build the Denoising Transformer architecture in `chess_dfm_jax/nnx_bt4.py` or a dedicated `nnx_dfm.py`.
3. Update `train_dfm.py` to leverage the existing `LeelaChunkDataLoader` and launch the Phase 3 sweep.

## 4. Evaluation Harness & Illegal Move Handling
To safely evaluate the model and calculate its Elo without relying on external dependencies like `cutechess-cli`, we use a custom, pure Python tournament manager (`scripts/evaluate_elo.py`).

### Handled Edge Cases:
- **JIT Compilation Timeouts**: Engines respond to the `isready` UCI command by triggering a dummy forward pass and fully compiling the JAX graph, guaranteeing they respond instantly during timed matches.
- **Engine Crashes & Timeouts**: The `evaluate_elo.py` script wraps the `await engine.play()` call in a try/except block. Any crash, timeout, or bad UCI string results in an immediate forfeit, preventing deadlocks.
- **Illegal Moves**: Currently, if the DFM generates an illegal move, the UCI script (`scripts/uci_dfm.py`) catches it via the python-chess `legal_move_mask` and falls back to the highest-confidence raw BT4 policy move.

### Proposals for Incorporating Legal Moves into DFM
Instead of falling back to BT4 at inference time, we could explicitly teach or constrain the DFM to only produce legal moves. Potential approaches (awaiting confirmation before implementation):

1.  **Inference-Time Masking (Masked DFM)**:
    During the $N$ steps of the denoising diffusion process, we can zero out the logits for illegal moves at every step $\tau$. Since we are using JAX, we can pass the legal move mask (which is pre-computed using python-chess on the CPU) directly into the `dfm_infer` loop as an extra argument. This guarantees the DFM *only* distributes probability mass over valid moves without needing retraining.
    
2.  **Training-Time Loss Penalty (Negative Reward)**:
    We can add an auxiliary loss during training. In addition to minimizing cross-entropy against the true engine action, we can compute a "Legality Loss". We generate the `legal_move_mask` for the $t=0$ board state and add a large penalty if the DFM's predicted logits place high probability on masked (illegal) indices. 
    
3.  **State-Conditioned Action Space (JEPA Dynamics Verification)**:
    Because we are building a JEPA, we theoretically have a dynamics model that predicts future states $z_{t+1}$. We could incorporate the JEPA predictor into the DFM: if the DFM suggests an action $a_t$, we unroll the JEPA to see if $a_t$ leads to a catastrophic state or an invalid board representation, feeding that signal back as a conditioning vector. (This is computationally heavy but the most "World Model" approach).
