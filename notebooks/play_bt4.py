import marimo

__generated_with = "0.20.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import chess
    import chess.svg
    import jax
    import jax.numpy as jnp
    import marimo as mo
    import numpy as np

    from chess_dfm_jax.analysis.profile_targets import load_mapped_bt4_params, parse_compute_dtype
    from chess_dfm_jax.encoding import encode_board
    from chess_dfm_jax.nnx_bt4 import jit_bt4_forward, make_bt4_model
    from chess_dfm_jax.paths import default_bt4_paths
    from chess_dfm_jax.policy import legal_move_mask, policy_index_to_move

    return (
        chess,
        default_bt4_paths,
        encode_board,
        jax,
        jit_bt4_forward,
        legal_move_mask,
        load_mapped_bt4_params,
        make_bt4_model,
        mo,
        np,
        parse_compute_dtype,
        policy_index_to_move,
    )


@app.cell
def _(marimo):
    marimo.md(
        "# Play Against BT4\n\n"
        "This notebook runs the current BT4 policy head directly on GPU, masks to legal moves, and plays the greedy legal argmax. There is no search."
    )
    return


@app.cell
def _(chess, mo):
    board_fen, set_board_fen = mo.state(chess.STARTING_FEN)
    status_text, set_status_text = mo.state("New game.")
    return board_fen, set_board_fen, set_status_text, status_text


@app.cell
def _(default_bt4_paths, load_mapped_bt4_params, make_bt4_model, parse_compute_dtype):
    MODEL_PATHS = default_bt4_paths()
    params = load_mapped_bt4_params(models_dir=MODEL_PATHS["models_dir"])
    models = {
        "float16": make_bt4_model(params, dtype=parse_compute_dtype("float16")),
        "float32": make_bt4_model(params, dtype=parse_compute_dtype("float32")),
    }
    return MODEL_PATHS, models, params


@app.cell
def _(jax, marimo):
    marimo.md(f"## Runtime\n\nBackend: `{jax.default_backend()}` | Device: `{jax.devices()[0]}`")
    return


@app.cell
def _(board_fen, chess, mo):
    board = chess.Board(board_fen())
    human_color = mo.ui.dropdown(options={"White": "white", "Black": "black"}, value="white", label="Human plays")
    compute_dtype = mo.ui.dropdown(options={"FP16": "float16", "FP32": "float32"}, value="float16", label="Compute dtype")
    legal_options = {board.san(move): move.uci() for move in board.legal_moves} if not board.is_game_over() else {}
    move_dropdown = mo.ui.dropdown(
        options=legal_options,
        value=next(iter(legal_options.values()), None),
        allow_select_none=True,
        label="Your move",
    )
    return board, compute_dtype, human_color, move_dropdown


@app.cell
def _(
    board_fen,
    chess,
    compute_dtype,
    encode_board,
    human_color,
    jax,
    jnp,
    jit_bt4_forward,
    legal_move_mask,
    mo,
    models,
    move_dropdown,
    np,
    parse_compute_dtype,
    policy_index_to_move,
    set_board_fen,
    set_status_text,
):
    gpu = next((device for device in jax.devices() if device.platform == "gpu"), jax.devices()[0])

    def greedy_model_move(board: chess.Board) -> chess.Move:
        dtype = parse_compute_dtype(compute_dtype.value)
        model = models[compute_dtype.value]
        planes = encode_board(board, [], input_format="INPUT_CLASSICAL_112_PLANE")[None, ...]
        planes = jax.device_put(jnp.asarray(planes, dtype=dtype), gpu)
        policy, _wdl, _moves_left = jit_bt4_forward(model, planes)
        policy = np.asarray(policy[0], dtype=np.float32)
        mask = legal_move_mask(board, "lc0_1858")
        masked = np.where(mask, policy, -1e9)
        return policy_index_to_move(int(masked.argmax()), "lc0_1858")

    def maybe_autoplay_from(board: chess.Board) -> chess.Board:
        human_turn = (board.turn == chess.WHITE and human_color.value == "white") or (
            board.turn == chess.BLACK and human_color.value == "black"
        )
        if board.is_game_over() or human_turn:
            return board
        model_move = greedy_model_move(board)
        model_san = board.san(model_move)
        board.push(model_move)
        set_status_text(f"Model played {model_san}.")
        return board

    def on_new_game(_value):
        board = chess.Board()
        board = maybe_autoplay_from(board)
        set_board_fen(board.fen())
        if board.move_stack:
            return
        set_status_text("New game.")

    def on_play(_value):
        if move_dropdown.value is None:
            set_status_text("Select a legal move first.")
            return
        board = chess.Board(board_fen())
        move = chess.Move.from_uci(move_dropdown.value)
        if move not in board.legal_moves:
            set_status_text("Illegal move selected.")
            return
        human_san = board.san(move)
        board.push(move)
        if board.is_game_over():
            set_board_fen(board.fen())
            set_status_text(f"You played {human_san}. Result: {board.result()}")
            return
        model_move = greedy_model_move(board)
        model_san = board.san(model_move)
        board.push(model_move)
        set_board_fen(board.fen())
        if board.is_game_over():
            set_status_text(f"You played {human_san}. Model replied {model_san}. Result: {board.result()}")
        else:
            set_status_text(f"You played {human_san}. Model replied {model_san}.")

    new_game_button = mo.ui.button(on_click=on_new_game, label="New game")
    play_button = mo.ui.button(on_click=on_play, label="Play selected move", disabled=move_dropdown.value is None)
    return gpu, new_game_button, play_button


@app.cell
def _(board, chess, mo, new_game_button, play_button, move_dropdown, human_color, compute_dtype, status_text):
    board_svg = chess.svg.board(board=board, size=440, lastmove=board.peek() if board.move_stack else None)
    mo.vstack(
        [
            mo.md(
                f"""
### Controls

- Human plays: `{human_color.value}`
- Compute dtype: `{compute_dtype.value}`
"""
            ),
            human_color,
            compute_dtype,
            new_game_button,
            move_dropdown,
            play_button,
            mo.Html(board_svg),
            mo.md(f"**Status:** {status_text()}"),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
