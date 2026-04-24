# Phase 3 DFM Experiment Design

## Objective
We aim to determine whether a combination of a frozen pre-trained dynamics model (the JEPA encoder/representation) and a Discrete Flow Matching (DFM) planner can result in a stronger **searchless** chess engine compared to the raw 1-ply policy head of the original BT4 (Leela Chess Zero) model.

## The Architecture
1. **Frozen BT4 Encoder (Representation Model)**
   - The model takes the standard Lc0 112-plane board representation as input.
   - We pass the planes through a rigorously verified, mathematically identical JAX/Flax NNX implementation of the BT4 network.
   - The encoder is kept **completely frozen**. It outputs a sequence of 64 spatial latents $z_t \in \mathbb{R}^{64 \times 512}$ that represent the current board state.

2. **The DFM Planner (Categorical Diffusion)**
   - The planner is an autoregressive-free sequence denoiser built using the rigorous mathematical formulation of Discrete Flow Matching (Gat et al., 2024).
   - **Input**: The model receives the frozen board latents $z_t$ and a sequence of $K=8$ discrete actions (moves). During training, these actions are corrupted by a time-dependent masking process.
   - **Action Conditioning**: Actions are passed through an `ActionMLP` (an embedding layer followed by a 2-layer Swish MLP) which maps the discrete 1858-vocabulary actions to the 512-dim latent space.
   - **Noising Process**: A continuous timestep $t \sim \mathcal{U}[0, 1]$ is sampled. We use the linear probability path $P(x_t = M | x_1) = 1 - t$ and $P(x_t = x_1 | x_1) = t$. The target action sequence $x_1$ is corrupted into the noisy state $x_t$ by replacing tokens with the `[MASK]` token (id=1858) according to this probability.
   - **Denoising Network**: A Transformer encoder (without Cross-Square Attention, to allow action self-attention). It takes the concatenated sequence `[Z_1...Z_64, A_1...A_K]`, injects the timestep embedding, and predicts the clean action tokens (the vector field target).

3. **Multi-Task Heads**
   - **Policy Head**: Predicts the discrete moves.
   - **Value Head**: Evaluates the outcome (Q-value and WDL probabilities).

## The Training Data (TCEC/CCRL High-Strength Sequences)
Instead of training on single, shuffled Lichess states, we process high-quality engine games (e.g., Stockfish vs. Leela from TCEC).
- The dataset consists of `.npz` chunks.
- Each sample contains the input planes, a sequence of 8 future actions, the true game outcome (Win, Draw, Loss probabilities and continuous Q-value), and a boolean `legal_move_mask` representing the valid moves for the starting state.

## The Loss Objectives
The model minimizes a composite loss function during training:
1. **DFM Loss (Categorical Cross-Entropy)**: The model predicts the true $x_1$ target tokens. The loss is strictly evaluated *only* on the tokens that are currently masked at timestep $t$.
2. **Legality Loss Penalty**: Because DFM is generative, it can propose illegal moves. We extract the predicted probabilities for the very first step ($k=0$), mask out all legally valid moves, and heavily penalize the model for placing any probability mass on the remaining illegal indices.

## Evaluation Harness (Searchless Elo Benchmark)
To prove the architecture works, we have built a custom, zero-dependency Python UCI tournament manager (`scripts/evaluate_elo.py`):
1. Two UCI engines are initialized: `uci_bt4.py` (the raw 1-ply baseline) and `uci_dfm.py` (the multi-step flow matching planner).
2. The DFM engine uses $N=8$ iterative refinement steps to denoise a sequence of `[MASK]` tokens into a definitive action trajectory, playing the first generated move.
3. The tournament manager rapidly forces the engines to play hundreds of games against each other.
4. Based on the Win/Draw/Loss outcomes, the exact Bayesian Elo difference is calculated, directly answering the core research question.
