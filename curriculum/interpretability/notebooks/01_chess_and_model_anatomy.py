# /// script
# dependencies = ["marimo", "numpy", "python-chess"]
# requires-python = ">=3.11"
# ///

import marimo

__generated_with = "0.23.3"
app = marimo.App(width="full")


@app.cell
def imports():
    import sys
    from pathlib import Path

    import chess
    import marimo as mo
    import numpy as np

    repo_root = Path(__file__).resolve().parents[3]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from research.interpretability.course import get_module, load_course_bundle
    from research.interpretability.course.evidence import (
        ClaimCard,
        Endpoint,
        EvidenceOperation,
        ResultStatus,
        TargetLevel,
    )
    from research.interpretability.course.stats import stable_softmax
    from research.interpretability.course.ui import (
        callout,
        chessboard,
        claim_card_html,
        committed_form,
        metric_cards,
        mode_badge,
        module_header,
        resource_card,
        table,
    )

    return (
        ClaimCard,
        Endpoint,
        EvidenceOperation,
        ResultStatus,
        TargetLevel,
        callout,
        chess,
        chessboard,
        claim_card_html,
        committed_form,
        get_module,
        load_course_bundle,
        metric_cards,
        mo,
        mode_badge,
        module_header,
        np,
        resource_card,
        stable_softmax,
        table,
    )


@app.cell
def title(get_module, mo, module_header):
    module = get_module(1)
    mo.Html(module_header(module))
    return (module,)


@app.cell
def opening(mo, module, table):
    objectives = [
        {"success": objective, "evidence": evidence}
        for objective, evidence in zip(
            module.objectives,
            (
                "Explain a legal move in UCI and SAN on an unseen board.",
                "Recover every batch/token/width dimension without guessing.",
                "Choose a hook whose input and output ABI matches a proposed replacement.",
            ),
            strict=True,
        )
    ]
    mo.vstack(
        [
            mo.md(
                r"""
                ## The specimen before the microscope

                A chess model is especially useful for learning interpretability:
                the environment has exact rules, the board is fully observed, and
                generated actions can be replayed. Those conveniences remove some
                ambiguity—but they do not make an activation self-explanatory.

                We will trace two related systems. **BT4** encodes one position and
                exposes policy, win/draw/loss, and moves-left outputs. The **DFM**
                generates a multi-ply action sequence from the encoded state. The
                **JEPA** predicts future state embeddings conditioned on a state and
                actions. First, make every noun and tensor boundary concrete.
                """
            ),
            mo.Html(table(objectives, (("success", "Objective"), ("evidence", "Mastery evidence")))),
        ]
    )
    return


@app.cell
def prerequisite(committed_form, mo):
    prereq_answer = mo.ui.radio(
        options=[
            "A probability distribution over legal actions",
            "The move with the largest unmasked logit",
            "An attention pattern over squares",
        ],
        value=None,
        label="Quick diagnostic: after legality masking and softmax, a policy is…",
    )
    prereq_form = committed_form(
        mo, {"policy_definition": prereq_answer}, submit_label="Check prerequisite"
    )
    prereq_form
    return (prereq_form,)


@app.cell
def prerequisite_feedback(callout, mo, prereq_form):
    mo.stop(prereq_form.value is None, mo.md("Commit before revealing the explanation."))
    prereq_ok = prereq_form.value["policy_definition"] == "A probability distribution over legal actions"
    mo.Html(
        callout(
            "positive" if prereq_ok else "limit",
            "Ready" if prereq_ok else "One distinction to repair",
            "A logit is an unconstrained score. Mask illegal actions, then normalize the remaining scores; the result is the policy distribution.",
        )
    )
    return


@app.cell
def mode_controls(mo):
    evidence_mode = mo.ui.radio(
        options={
            "Snapshot — frozen real architecture": "snapshot",
            "Toy — reduced constructed architecture": "toy",
            "Source — verify retained architecture": "source",
        },
        value="Snapshot — frozen real architecture",
        label="Evidence mode",
    )
    verify_source = mo.ui.run_button(label="Verify retained sources", kind="warn")
    mo.hstack([evidence_mode, verify_source], justify="start", gap=1)
    return evidence_mode, verify_source


@app.cell
def load_mode(evidence_mode, load_course_bundle, mo, verify_source):
    selected_mode = evidence_mode.value
    mo.stop(
        selected_mode == "source" and not verify_source.value,
        mo.callout("Press **Verify retained sources** to enter source mode.", kind="warn"),
    )
    bundle = load_course_bundle(selected_mode)
    architecture = bundle.section("architecture")
    return architecture, bundle, selected_mode


@app.cell
def mode_view(bundle, mo, mode_badge, selected_mode):
    mo.Html(mode_badge(selected_mode, bundle.evidence_scope))
    return


@app.cell
def chess_glossary(mo, table):
    glossary_rows = [
        {"term": "square", "definition": "file a–h plus rank 1–8; e4 is file e, rank 4"},
        {"term": "side to move", "definition": "the player whose action is legal now"},
        {"term": "attack", "definition": "a square a piece could capture on, with pawn and king rules respected"},
        {"term": "check", "definition": "the side-to-move king is attacked; every legal action must resolve it"},
        {"term": "pseudo-legal", "definition": "obeys the piece's movement/occupancy rules but may leave its own king attacked; legal moves also preserve king safety"},
        {"term": "pin", "definition": "moving a piece would expose a more valuable piece; an absolute pin exposes the king"},
        {"term": "material", "definition": "the pieces a side retains, often summarized by conventional values such as pawn 1, knight/bishop 3, rook 5, queen 9; it is a feature, not the game objective"},
        {"term": "UCI / SAN", "definition": "coordinate action such as e2e4 / contextual notation such as e4 or Nf3"},
        {"term": "special moves", "definition": "castling moves king+rook; promotion changes a pawn; en passant is a conditional pawn capture"},
    ]
    mo.vstack([mo.md("## Minimal chess vocabulary"), mo.Html(table(glossary_rows, (("term", "Term"), ("definition", "Operational definition"))))])
    return


@app.cell
def chess_primer(chess, chessboard, mo, table):
    movement_rows = [
        {"piece": "king", "ordinary movement": "one square in any direction", "important constraint": "may not move onto an attacked square"},
        {"piece": "queen", "ordinary movement": "any distance along rank, file, or diagonal", "important constraint": "cannot jump over a piece"},
        {"piece": "rook", "ordinary movement": "any distance along rank or file", "important constraint": "cannot jump over a piece"},
        {"piece": "bishop", "ordinary movement": "any distance along a diagonal", "important constraint": "cannot jump over a piece"},
        {"piece": "knight", "ordinary movement": "two squares in one axis and one in the other", "important constraint": "may jump over pieces"},
        {"piece": "pawn", "ordinary movement": "one square forward; captures one diagonal forward", "important constraint": "direction depends on color; first double-step, promotion, and en passant use state"},
    ]
    pinned_fen = "4k3/4r3/8/8/8/8/4R3/4K3 w - - 0 1"
    pinned_board = chess.Board(pinned_fen)
    pinned_move = chess.Move.from_uci("e2d2")
    primer_checks = [
        {"test": "rook e2→d2 follows rook movement and lands on an empty square", "result": pinned_board.is_pseudo_legal(pinned_move)},
        {"test": "after e2→d2, White's king on e1 remains safe from the black rook on e7", "result": pinned_board.is_legal(pinned_move)},
    ]
    mo.vstack([
        mo.md(r"""
        ## Chess primer: enough rules to audit legality

        The diagrams use **White's viewpoint**: file `a` is at the left, file
        `h` at the right, rank 1 at the bottom, and rank 8 at the top. White
        pawns normally move toward increasing ranks; Black pawns toward
        decreasing ranks. A move is legal only if (1) its piece can make that
        geometric move, (2) occupancy and special-state rules allow it, and
        (3) the mover's king is not attacked afterward.
        """),
        mo.Html(table(movement_rows, (("piece", "Piece"), ("ordinary movement", "Movement"), ("important constraint", "Constraint")))),
        mo.hstack([
            mo.Html(chessboard(pinned_fen, highlighted=("e2", "d2", "e1", "e7"))),
            mo.vstack([
                mo.md("**Worked legality trace — White to move.** The white rook on e2 blocks the black rook's line to the white king on e1."),
                mo.Html(table(primer_checks, (("test", "Question"), ("result", "True?")))),
                mo.md("So `e2d2` is **pseudo-legal but illegal**: moving the pinned rook exposes its own king. In check, the same final king-safety test removes every response that fails to capture the checker, block the line, or move the king safely."),
            ]),
        ], widths=[1, 2], gap=1.2),
    ])
    return


@app.cell
def position_control(chess, mo):
    position = mo.ui.dropdown(
        options={
            "Ordinary opening position": chess.STARTING_FEN,
            "Black king is in check": "4k3/8/8/8/8/8/4R3/4K3 b - - 0 1",
            "White rook is absolutely pinned": "4k3/4r3/8/8/8/8/4R3/4K3 w - - 0 1",
            "Castling rights are part of state": "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
            "A pawn can promote": "7k/P7/8/8/8/8/8/7K w - - 0 1",
            "En passant target is part of state": "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1",
        },
        value="Ordinary opening position",
        label="Choose a position",
    )
    position
    return (position,)


@app.cell
def position_state(chess, position):
    board = chess.Board(position.value)
    legal_moves = list(board.legal_moves)
    move_options = {
        f"{board.san(move)} · {move.uci()}": move.uci()
        for move in legal_moves
    }
    return board, legal_moves, move_options


@app.cell
def move_control(mo, move_options):
    first_move_label = next(iter(move_options))
    selected_move = mo.ui.dropdown(
        options=move_options,
        value=first_move_label,
        label="Inspect one legal action (SAN · UCI)",
    )
    selected_move
    return (selected_move,)


@app.cell
def board_view(
    board,
    chess,
    chessboard,
    legal_moves,
    metric_cards,
    mo,
    selected_move,
):
    move = chess.Move.from_uci(selected_move.value)
    highlighted = [chess.square_name(move.from_square), chess.square_name(move.to_square)]
    mo.hstack(
        [
            mo.Html(chessboard(board.fen(), highlighted=highlighted)),
            mo.vstack(
                [
                    mo.Html(metric_cards([
                        ("White" if board.turn else "Black", "side to move"),
                        (len(legal_moves), "legal actions"),
                        ("yes" if board.is_check() else "no", "king in check"),
                        (selected_move.value, "selected UCI action"),
                    ])),
                    mo.md(
                        "A FEN string stores piece placement, side to move, castling rights, an en-passant square, and two clocks. The picture alone is not always the full state. In the en-passant example, White's e5d6 move lands on d6 but captures the black pawn on d5; that exceptional capture is available only immediately after the qualifying two-square pawn advance."
                    ),
                ]
            ),
        ],
        widths=[1, 2],
        gap=1.2,
    )
    return


@app.cell
def temperature_control(mo):
    temperature = mo.ui.number(0.25, 3.0, step=0.25, value=1.0, label="Policy temperature T")
    temperature
    return (temperature,)


@app.cell
def policy_math(board, legal_moves, np, stable_softmax, temperature):
    move_names = [move.uci() for move in legal_moves]
    raw_logits = np.asarray([
        1.2 * np.cos(index * 1.7) + 0.35 * np.sin(index * 0.41)
        for index in range(len(move_names))
    ])
    probabilities = stable_softmax(raw_logits / temperature.value)
    order = np.argsort(probabilities)[::-1][: min(8, len(move_names))]
    policy_rows = [
        {
            "uci": move_names[index],
            "san": board.san(legal_moves[index]),
            "logit": f"{raw_logits[index]:+.3f}",
            "probability": f"{probabilities[index]:.3f}",
        }
        for index in order
    ]
    entropy = float(-np.sum(probabilities * np.log(np.clip(probabilities, 1e-12, None))))
    return entropy, policy_rows


@app.cell
def policy_view(entropy, mo, policy_rows, table, temperature):
    mo.vstack(
        [
            mo.md(
                rf"""
                ## From logits to a legal policy

                For legal-action set \(L(s)\),

                \[
                p(a\mid s)=\frac{{\exp(\ell_a/T)}}
                {{\sum_{{b\in L(s)}}\exp(\ell_b/T)}}\,\mathbf 1[a\in L(s)].
                \]

                At \(T={temperature.value:.2f}\), entropy is **{entropy:.3f} nats**.
                Temperature changes the estimand whenever distributional metrics
                are computed after rescaling; it is not merely a visual filter.
                """
            ),
            mo.Html(table(policy_rows, (("uci", "UCI"), ("san", "SAN"), ("logit", "Logit"), ("probability", "Probability")))),
        ]
    )
    return


@app.cell
def architecture_trace(architecture, mo, table):
    trace = [
        {"stage": "LC0 input at Torch boundary", "shape": f"[B, {architecture['input_planes']}, 8, 8]", "meaning": "channel-first rule/history planes"},
        {"stage": "embedding scratch layout", "shape": f"[B, 8, 8, {architecture['input_planes']}] → [B, {architecture['square_tokens']}, {architecture['width']}]", "meaning": "the encoder transposes internally, then creates one token per square"},
        {"stage": "BT4 trunk Z_trunk", "shape": f"[B, {architecture['square_tokens']}, {architecture['width']}]", "meaning": f"{architecture['square_tokens']} contextual square tokens after {architecture['layers']} bidirectional blocks"},
        {"stage": "native attention head", "shape": f"[B, {architecture['heads']}, {architecture['square_tokens']}, {architecture['square_tokens']}]", "meaning": f"QK weights; head width {architecture['head_width']}"},
        {"stage": "policy logits", "shape": f"[B, {architecture['action_vocab']}]", "meaning": "scores over the fixed move vocabulary"},
        {"stage": "Torch Hero value / WDL outputs", "shape": "[B] and [B, 3]", "meaning": "scalar value and win/draw/loss logits read from pooled JEPA state; the Torch Hero graph does not invoke a moves-left head"},
        {"stage": "DFM state projection Z_DFM", "shape": f"[B, {architecture['square_tokens']}, {architecture['width']}] → [B, {architecture['square_tokens']}, {architecture['dfm_state_width']}]", "meaning": f"project each trunk square token; these {architecture['square_tokens']} tokens condition the action denoiser"},
        {"stage": "DFM action trajectory", "shape": "[B, 8] plus denoising state", "meaning": "eight discrete action slots, refined jointly"},
        {"stage": "JEPA StateProjector z_JEPA", "shape": f"[B, {architecture['square_tokens']}, {architecture['width']}] → [B, {architecture['jepa_state_width']}]", "meaning": f"internally form {architecture['square_tokens']} projected squares plus CLS, then return only the pooled CLS state"},
        {"stage": "JEPA latent trajectory", "shape": f"[B, 8, {architecture['jepa_state_width']}]", "meaning": "eight recurrent predictions of pooled state vectors, not square-token grids"},
    ]
    mo.vstack([mo.md("## Shape trace: state → representation → actions → states"), mo.Html(table(trace, (("stage", "Stage"), ("shape", "Tensor shape"), ("meaning", "What the axes mean"))))])
    return


@app.cell
def shape_prediction(committed_form, mo):
    shape_answer = mo.ui.dropdown(
        options=["[B, 64, 1024]", "[B, 32, 64, 64]", "[B, 1858]", "[B, 8]"],
        value=None,
        label="Predict: native attention weights after QK softmax",
    )
    shape_form = committed_form(
        mo, {"attention_shape": shape_answer}, submit_label="Lock shape prediction"
    )
    shape_form
    return (shape_form,)


@app.cell
def shape_feedback(callout, mo, shape_form):
    mo.stop(shape_form.value is None, mo.md("Predict before revealing."))
    shape_ok = shape_form.value["attention_shape"] == "[B, 32, 64, 64]"
    mo.Html(callout(
        "positive" if shape_ok else "limit",
        "Axes recovered" if shape_ok else "Count the axes again",
        "Each of 32 heads has one query row and one key column for all 64 squares. The width-32 values are consumed before the projected branch returns to [B,64,1024].",
    ))
    return


@app.cell
def attention_control(mo):
    similarity = mo.ui.number(-4.0, 4.0, step=0.5, value=1.0, label="One square-pair Q·K similarity")
    attention_scale = mo.ui.number(0.25, 2.0, step=0.25, value=1.0, label="logit scale")
    mo.hstack([similarity, attention_scale], justify="start")
    return attention_scale, similarity


@app.cell
def attention_pass(attention_scale, mo, np, similarity, stable_softmax, table):
    qk = np.asarray([similarity.value, 1.0, 0.5, -0.5]) * attention_scale.value
    weights = stable_softmax(qk)
    values = np.asarray([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 0.5]])
    mixed = weights @ values
    attention_rows = [
        {"key square": name, "QK score": f"{score:+.2f}", "weight": f"{weight:.3f}", "value": str(value.tolist())}
        for name, score, weight, value in zip(("a1", "b1", "c1", "d1"), qk, weights, values, strict=True)
    ]
    mo.vstack([
        mo.md(r"""
        ## A tiny attention pass

        Attention first chooses a convex weighting with QK scores, then mixes
        **value vectors**. A striking weight is not yet an explanation: the
        value, output projection, residual addition, and downstream use matter.
        """),
        mo.Html(table(attention_rows, (("key square", "Key"), ("QK score", "Q·K"), ("weight", "Softmax weight"), ("value", "Value")))),
        mo.md(f"Weighted value at this query: **[{mixed[0]:+.3f}, {mixed[1]:+.3f}]**"),
    ])
    return


@app.cell
def hook_control(architecture, mo):
    hook_labels = {f"{name} — {description}": name for name, description in architecture.get("hooks", {}).items()}
    if not hook_labels:
        hook_labels = {
            "hook_attn_in — layer input before Q/K/V": "hook_attn_in",
            "hook_attn_out — projected attention branch": "hook_attn_out",
            "hook_mlp_out — MLP branch": "hook_mlp_out",
            "resid_post_after_ln — next-layer input": "resid_post_after_ln",
        }
    hook_choice = mo.ui.dropdown(options=hook_labels, value=next(iter(hook_labels)), label="Inspect an intervention boundary")
    replacement_goal = mo.ui.radio(
        options=[
            "Observe the layer input only",
            "Replace the attention branch while preserving the residual path",
            "Replace the whole post-block representation",
        ],
        value="Replace the attention branch while preserving the residual path",
        label="Proposed experiment",
    )
    mo.vstack([hook_choice, replacement_goal])
    return hook_choice, replacement_goal


@app.cell
def hook_feedback(architecture, callout, hook_choice, mo, replacement_goal):
    chosen_hook = hook_choice.value
    expected_hook = {
        "Observe the layer input only": "hook_attn_in",
        "Replace the attention branch while preserving the residual path": "hook_attn_out",
        "Replace the whole post-block representation": "resid_post_after_ln",
    }[replacement_goal.value]
    hook_ok = chosen_hook == expected_hook
    description = architecture.get("hooks", {}).get(chosen_hook, "constructed hook with the displayed ABI")
    mo.Html(callout(
        "positive" if hook_ok else "limit",
        "Boundary matches the estimand" if hook_ok else "Boundary changes the intervention",
        f"Selected `{chosen_hook}`: {description}. For the stated experiment, `{expected_hook}` is the precise boundary. Equal width does not make a residual SAE and an attention-branch replacement interchangeable.",
    ))
    return


@app.cell
def system_relationships(mo, table):
    relationship_rows = [
        {"object": "BT4 encoder", "input": "board planes", "output": "64 contextual square embeddings", "inference role": "state representation"},
        {"object": "DFM", "input": "Z_DFM: 64 projected trunk tokens + partially masked action slots", "output": "distribution over action tokens in all slots", "inference role": "iterative discrete trajectory completion"},
        {"object": "JEPA", "input": "z_JEPA: one pooled state + actions + DFM action hidden states", "output": "future pooled state embeddings (and future value in the proposed extension)", "inference role": "conditional world-model prediction only when explicitly invoked"},
        {"object": "Torch Hero value/WDL head", "input": "pooled current or predicted JEPA state", "output": "scalar value + 3 WDL logits", "inference role": "trained state-quality readout; not a moves-left head or value-guided DFM selector"},
    ]
    mo.vstack([
        mo.md("## DFM and JEPA are coupled objectives, not synonyms"),
        mo.Html(table(relationship_rows, (("object", "Object"), ("input", "Consumes"), ("output", "Produces"), ("inference role", "Role")))),
        mo.md(r"""
        The sealed Hero and selected open-loop Proposal-A family have **two**
        state interfaces after the shared trunk (Z_{\text{trunk}}=E(s_0)):

        \[
        Z_{\text{DFM}}=P_{\text{DFM}}(Z_{\text{trunk}})
          \in\mathbb R^{64\times256},\qquad
        z_{\text{JEPA}}=P_{\text{JEPA}}(Z_{\text{trunk}})
          \in\mathbb R^{1024}.
        \]

        The action model samples
        \((a_0,\ldots,a_7)\sim D(\cdot\mid Z_{\text{DFM}})\); it does **not**
        consume the pooled JEPA state. The world model predicts
        \((\hat z_1,\ldots,\hat z_8)=J(z_{\text{JEPA}},a_{0:7},h_{0:7})\),
        where (h) denotes DFM action hidden states.
        Training gradients from the JEPA objective can shape the shared encoder.
        That fact alone does **not** feed JEPA state predictions back into DFM
        denoising at inference. In the sealed Hero manifest the DFM source is the
        trunk, current-JEPA conditioning is disabled, and fusion/closed-loop modes
        are `none`.

        The upstream LC0/BT4 checkpoint lineage also contains legacy value and
        moves-left head concepts. The Torch Hero graph studied here binds the
        native policy readout and its own pooled value/WDL head; it does not
        invoke a moves-left head. Keeping source-checkpoint contents separate
        from the executed Torch graph prevents a common anatomy error.
        """),
    ])
    return


@app.cell
def anatomy_claim(
    ClaimCard,
    Endpoint,
    EvidenceOperation,
    ResultStatus,
    TargetLevel,
    architecture,
    bundle,
    claim_card_html,
    mo,
):
    label = "constructed architecture" if not bundle.is_empirical else "retained Raw BT4 / Hero architecture records"
    card = ClaimCard(
        operation=EvidenceOperation.OBSERVATION,
        target=TargetLevel.REPRESENTATION,
        scope=label,
        endpoint=Endpoint.POLICY,
        estimand="recorded tensor ABI from LC0 planes through the policy action vocabulary",
        unit="one architecture specification",
        grouping="stage and named hook",
        intervention=None,
        controls=("shape consistency", "named hook-boundary audit"),
        assumptions=("the retained specification matches the runtime selected for later experiments",),
        uncertainty="deterministic shape audit; no sampling interval",
        multiplicity="not applicable",
        allowed=(
            f"The selected record maps {architecture['input_planes']} planes to "
            f"{architecture['square_tokens']} tokens of width {architecture['width']} and "
            f"a {architecture['action_vocab']}-entry action vocabulary."
        ),
        excluded="Matching dimensions do not establish semantic equivalence or a valid cross-boundary replacement.",
        falsifier="A runtime hook trace with any incompatible named dimension or stage ordering.",
        status=ResultStatus.DESCRIPTIVE,
    )
    mo.Html(claim_card_html(card))
    return


@app.cell
def analogy(mo, table):
    analogy_rows = [
        {"chess": "one square token", "language analogy": "one sequence token", "where it breaks": "all 64 squares exist at once; empty squares are meaningful"},
        {"chess": "BT4 bidirectional attention", "language analogy": "encoder-style contextualization", "where it breaks": "no causal next-token mask inside the board encoder"},
        {"chess": "1858 move vocabulary", "language analogy": "word-piece vocabulary", "where it breaks": "legality depends exactly on simulator state"},
        {"chess": "DFM action slots", "language analogy": "masked span generation", "where it breaks": "slots jointly refine a trajectory rather than append one token"},
        {"chess": "JEPA pooled future-state vector [B,H,1024]", "language analogy": "predicting a future sequence summary", "where it breaks": "it is not a per-square latent grid; chess offers exact replay targets and legality checks"},
    ]
    mo.vstack([mo.md("## Language-model analogy—with failure boundaries"), mo.Html(table(analogy_rows, (("chess", "Chess object"), ("language analogy", "Useful analogy"), ("where it breaks", "Failure boundary"))))])
    return


@app.cell
def assessments(committed_form, mo, module):
    chess_transfer = mo.ui.text_area(label="Transfer 1 — legality", placeholder="In FEN 4k3/4r3/8/8/8/8/4R3/4K3 w - - 0 1, explain why e2d2 is pseudo-legal but illegal.")
    shape_transfer = mo.ui.text_area(label="Transfer 2 — dimensions", placeholder="An unseen hook has [B,64,1024]. Name two objects it could be and one object it cannot be.")
    abi_transfer = mo.ui.text_area(label="Transfer 3 — replacement", placeholder="A module consumes [B,64,1024] and returns [B,64,1024]. What else must match before replacing hook_attn_out?")
    assessment_form = committed_form(
        mo,
        {"legality": chess_transfer, "dimensions": shape_transfer, "replacement": abi_transfer},
        submit_label="Commit responses and reveal formative rubric",
        min_words=5,
    )
    mo.vstack([mo.md(f"## Unseen transfer assessment\n\n{module.transfer_assessment}"), assessment_form])
    return (assessment_form,)


@app.cell
def assessment_feedback(assessment_form, mo):
    mo.stop(assessment_form.value is None, mo.md("Submit all three responses before revealing the rubric."))
    counts = [len(assessment_form.value[key].split()) for key in ("legality", "dimensions", "replacement")]
    mo.md(f"""
    **Rubric (response lengths {counts}):**

    1. Legality answer names side to move, king safety, and state fields beyond piece placement when relevant.
    2. Shape answer distinguishes a residual/branch activation from per-head attention `[B,H,64,64]` and logits `[B,A]`.
    3. ABI answer names semantics, normalization point, layer, dtype/device, token ordering, and whether replacement or residual addition is intended.
    """)
    return


@app.cell
def close(bundle, callout, mo, resource_card):
    mo.vstack([
        mo.Html(callout(
            "limit",
            "What this anatomy does not show",
            "A shape trace verifies compatibility, not meaning. A visually interpretable attention map does not identify the values carried or establish downstream use.",
        )),
        mo.md("""
        ## Retrieval before module 02

        1. Why is the policy not simply the largest raw logit?
        2. Which tensor leaves one attention branch before residual addition?
        3. Does a JEPA training gradient imply JEPA feedback during DFM inference?

        **Sources:** Searchless Chess (NeurIPS 2024) for the BT4 specimen and
        McGrath et al. (PNAS 2022) for chess-network representation analysis.
        Exact status and links are in the [course source register](../SOURCES.md).
        """),
        mo.Html(resource_card(
            mode=bundle.mode.value,
            runtime="< 2 seconds in toy/snapshot; source mode hashes retained files",
            device="CPU; no checkpoint or network",
            determinism="no RNG in visible lab; NumPy float64; deterministic python-chess rules",
            sources=f"snapshot/source identity {bundle.integrity_sha256[:16]}…",
            limitations="Architecture metadata and a hand-built attention pass; no live model activation.",
        )),
    ])
    return


if __name__ == "__main__":
    app.run()
