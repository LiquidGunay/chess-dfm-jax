import marimo

__generated_with = "0.23.2"
app = marimo.App(width="medium")

@app.cell
def _():
    import sys
    from pathlib import Path
    import jax
    import jax.numpy as jnp
    import numpy as np
    import chess
    import chess.svg
    import functools
    import marimo as mo

    REPO_ROOT = Path().resolve()
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    from chess_dfm_jax.encoding import encode_board
    from chess_dfm_jax.policy import legal_move_mask, policy_index_to_move
    from chess_dfm_jax.analysis.profile_targets import load_mapped_bt4_params
    from chess_dfm_jax.training.dfm import create_dfm_components, DFMConfig
    from chess_dfm_jax.training.checkpoints import load_training_checkpoint

    return (
        REPO_ROOT, jax, jnp, np, chess, functools, mo,
        encode_board, legal_move_mask, policy_index_to_move,
        load_mapped_bt4_params, create_dfm_components, DFMConfig,
        load_training_checkpoint
    )


@app.cell
def _(REPO_ROOT, load_mapped_bt4_params, create_dfm_components, DFMConfig, load_training_checkpoint, mo):
    bt4_pb = REPO_ROOT / "models" / "BT4-1024x15x32h-swa-6147500-policytune-332.pb.gz"
    dfm_ckpt = REPO_ROOT / "artifacts" / "dfm-sweep-lr1e4-muon-k4"

    with mo.status.spinner("Loading BT4 parameters and DFM components..."):
        params = load_mapped_bt4_params(pb=str(bt4_pb))
        config = DFMConfig(token_dim=512, num_layers=4, use_qk_gain=True, use_muon=True, horizon=4)
        model, _ = create_dfm_components(params, config)

        step = load_training_checkpoint(dfm_ckpt, model=model)
    return bt4_pb, dfm_ckpt, params, config, model, step


@app.cell
def _(functools, jax, jnp):
    @functools.partial(jax.jit, static_argnames=['refinement_steps'])
    def dfm_infer_history(model, planes, legal_mask, refinement_steps: int):
        batch_size = planes.shape[0]
        K = model.config.horizon
        MASK = model.config.action_vocab_size
        
        x = jnp.full((batch_size, K), MASK, dtype=jnp.int32)
        
        def step_fn(val, i):
            x_curr = val
            t = jnp.full((batch_size,), i / refinement_steps, dtype=jnp.float32)
            logits = model(planes, x_curr, t)
            
            mask_expanded = legal_mask[:, None, :]
            logits_0 = jnp.where(mask_expanded, logits[:, 0:1, :], -1e9)
            if K > 1:
                logits = jnp.concatenate([logits_0, logits[:, 1:, :]], axis=1)
            else:
                logits = logits_0
                
            probs = jax.nn.softmax(logits, axis=-1)
            max_probs = jnp.max(probs, axis=-1)
            preds = jnp.argmax(logits, axis=-1)
            
            num_unmasked_target = (K * (i + 1)) // refinement_steps
            max_probs_all = jnp.where(x_curr == MASK, max_probs, 2.0)
            kth_idx = jnp.maximum(0, K - num_unmasked_target)
            sorted_probs = jnp.sort(max_probs_all, axis=-1)
            thresholds = jax.lax.dynamic_slice_in_dim(sorted_probs, kth_idx, 1, axis=-1)
            
            unmask_now = max_probs_all >= thresholds
            x_next = jnp.where(unmask_now, preds, x_curr)
            return x_next, x_next

        x_final, history = jax.lax.scan(step_fn, x, jnp.arange(refinement_steps))
        return x_final, history
    return (dfm_infer_history,)


@app.cell
def _(mo, chess):
    # State management for the game
    get_board, set_board = mo.state(chess.Board())
    get_history, set_history = mo.state(None)
    return get_board, set_board, get_history, set_history


@app.cell
def _(mo, get_board, set_board):
    board = get_board()
    
    # UI Elements
    move_input = mo.ui.text(placeholder="e.g. e2e4", label="Your Move (UCI)")
    play_human_btn = mo.ui.button(label="Play Human Move")
    play_dfm_btn = mo.ui.button(label="Play DFM Move")
    reset_btn = mo.ui.button(label="Reset Game")
    refinement_steps = mo.ui.slider(start=1, stop=16, step=1, value=8, label="Refinement Steps")

    return board, move_input, play_human_btn, play_dfm_btn, reset_btn, refinement_steps


@app.cell
def _(chess, board, move_input, play_human_btn, set_board, get_history, set_history):
    # Human move logic
    if play_human_btn.value:
        try:
            move = chess.Move.from_uci(move_input.value)
            if move in board.legal_moves:
                new_board = board.copy()
                new_board.push(move)
                set_board(new_board)
                set_history(None) # clear history since human played
        except ValueError:
            pass
    return


@app.cell
def _(mo, chess, board, play_dfm_btn, set_board, model, dfm_infer_history, encode_board, legal_move_mask, refinement_steps, jnp, np, policy_index_to_move, set_history):
    is_script_mode = mo.app_meta().mode == "script"
    
    # DFM Move logic
    if (play_dfm_btn.value or is_script_mode) and not board.is_game_over():
        planes = encode_board(board, [], input_format="INPUT_CLASSICAL_112_PLANE")
        planes_jnp = jnp.asarray(planes, dtype=jnp.float16)[None, ...]
        mask = legal_move_mask(board, "lc0_1858")
        mask_jnp = jnp.asarray(mask)[None, ...]

        actions, _history = dfm_infer_history(model, planes_jnp, mask_jnp, refinement_steps.value)
        
        best_idx = int(np.asarray(actions[0])[0])
        best_move = policy_index_to_move(best_idx, "lc0_1858")
        
        if best_move in board.legal_moves:
            _new_board = board.copy()
            _new_board.push(best_move)
            set_board(_new_board)
            
            # format history
            hist_np = np.asarray(_history[:, 0, :]) # [steps, K]
            formatted_history = []
            for step_idx in range(len(hist_np)):
                step_moves = []
                for k in range(hist_np.shape[1]):
                    idx = hist_np[step_idx, k]
                    if idx == model.config.action_vocab_size:
                        step_moves.append("[MASK]")
                    else:
                        step_moves.append(policy_index_to_move(int(idx), "lc0_1858").uci())
                formatted_history.append(f"Step {step_idx+1}: " + " -> ".join(step_moves))
            set_history(formatted_history)
            
            if is_script_mode:
                print(f"DFM Played: {best_move.uci()}")
                for line in formatted_history:
                    print(line)
    return


@app.cell
def _(chess, reset_btn, set_board, set_history):
    if reset_btn.value:
        set_board(chess.Board())
        set_history(None)
    return


@app.cell
def _(mo, board, move_input, play_human_btn, play_dfm_btn, reset_btn, refinement_steps, get_history, chess):
    board_html = mo.Html(chess.svg.board(board, size=400))
    
    controls = mo.vstack([
        mo.hstack([move_input, play_human_btn]),
        mo.hstack([play_dfm_btn, refinement_steps]),
        reset_btn
    ])
    
    history = get_history()
    if history:
        history_ui = mo.callout(mo.md("**DFM Token Refinement History:**\n\n" + "\n\n".join(history)), kind="info")
    else:
        history_ui = mo.md("*Play a DFM move to see the refinement history.*")
        
    layout = mo.vstack([
        mo.md("## Play vs DFM (Discrete Flow Matching)"),
        mo.hstack([board_html, controls]),
        history_ui
    ])
    
    layout
    return (board_html, controls, history, history_ui, layout)

if __name__ == "__main__":
    app.run()