"""Bounded diagnostic: are longer omitted trajectories uniquely recoverable?

This does not modify the reconstruction rules or emit training inputs.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import chess
import numpy as np

from research.hero_history_data import sha256, write_json_once, initial_board
from chess_dfm_jax.encoding import encode_board


def legal_paths(start, end_fen, plies, *, node_limit=500000, seconds=20):
    board = start.copy(stack=False)
    target = chess.Board(end_fen, chess960=board.chess960)
    target_map = target.piece_map()
    solutions, nodes, began = [], 0, time.monotonic()
    if not 1 <= plies <= 9:
        return {"status": "outside_depth_bound", "plies": plies}

    def visit(left, path):
        nonlocal nodes
        nodes += 1
        if nodes > node_limit or time.monotonic()-began > seconds:
            raise TimeoutError
        if not left:
            if board.fen() == end_fen:
                solutions.append(path[:])
            return
        # Each move can establish at most one target-colored piece on a target
        # square, or two if castling remains possible. Opponent captures cannot
        # establish a target-colored piece. This is a safe necessary bound.
        for color in (chess.WHITE, chess.BLACK):
            slots = (left+1)//2 if board.turn == color else left//2
            needed = sum(piece.color == color and board.piece_at(square) != piece
                         for square,piece in target_map.items())
            multiplier = 2 if board.has_castling_rights(color) else 1
            if needed > slots*multiplier:
                return
        for move in list(board.legal_moves):
            board.push(move)
            visit(left-1, path+[move.uci()])
            board.pop()
            if len(solutions) > 1:
                return

    try:
        visit(plies, [])
        status = "unique" if len(solutions) == 1 else "ambiguous" if solutions else "no_path"
    except TimeoutError:
        status = "bound_exhausted"
    return {"status": status, "plies": plies, "nodes": nodes,
            "seconds": time.monotonic()-began, "paths": solutions,
            "start_fen": start.fen(), "end_fen": end_fen}


def audit_shard(path):
    with np.load(path, allow_pickle=False) as data:
        plies = data["ply"]
        gaps = np.flatnonzero((plies[1:] != 0) & (plies[1:] > plies[:-1]+9))+1
        records = []
        for row in gaps:
            board = chess.Board(str(data["fen_t"][row-1]), chess960=True)
            for uci in data["actions_uci"][row-1]:
                board.push_uci(str(uci))
            length = int(plies[row]-plies[row-1]-8)
            records.append({"row": int(row), "missing_from_ply": int(plies[row-1]+8),
                            **legal_paths(board, str(data["fen_t"][row]), length)})
    return {"file": str(path), "sha256": sha256(path), "gaps": records}


def ambiguity_witness(path, row):
    """Show distinct history planes consistent with the same retained row."""
    with np.load(path, allow_pickle=False) as source:
        data = {key: source[key] for key in ("ply","fen_t","actions_uci","planes_t_u8")}
    begin = int(np.flatnonzero(data["ply"][:row] == 0)[-1])
    moves, fens = {}, {}
    for index in range(begin,row):
        ply = int(data["ply"][index])
        fens[ply] = str(data["fen_t"][index])
        for offset, uci in enumerate(data["actions_uci"][index]):
            if ply+offset in moves and moves[ply+offset] != str(uci):
                raise ValueError("Conflicting witness prefix")
            moves[ply+offset] = str(uci)
    board = initial_board(fens[0])
    history = [board.copy(stack=False)]
    for ply in range(max(moves)+1):
        if ply in fens and board.fen() != fens[ply]:
            raise ValueError("Witness prefix FEN mismatch")
        board.push_uci(moves[ply])
        history.append(board.copy(stack=False))
    end_fen = str(data["fen_t"][row])
    result = legal_paths(board, end_fen, int(data["ply"][row])-len(history)+1)
    if result["status"] != "ambiguous":
        raise ValueError("Witness did not find multiple legal histories")
    encodes = []
    for suffix in result["paths"][:2]:
        replay = board.copy(stack=False)
        prefix = history.copy()
        for uci in suffix:
            replay.push_uci(uci)
            prefix.append(replay.copy(stack=False))
        assert replay.fen() == end_fen
        encoded = encode_board(replay,prefix).astype(np.uint8)
        stripped = encoded.copy();stripped[12:104] = 0
        np.testing.assert_array_equal(stripped,data["planes_t_u8"][row])
        encodes.append(encoded)
    different = np.flatnonzero(np.any(encodes[0] != encodes[1],axis=(1,2))).tolist()
    if not different or any(not 12 <= i < 104 for i in different):
        raise ValueError("Witness must differ only in actual history planes")
    return {"source":str(path),"source_sha256":sha256(path),"row":row,
            "row_ply":int(data["ply"][row]),"prefix_replayed_from_ply_zero":True,
            "nonhistory_inputs_identical_to_original":True,
            "different_history_plane_indices":different,
            "full_history_input_sha256":[hashlib.sha256(value.tobytes()).hexdigest() for value in encodes],
            "finding":"Retained records do not uniquely determine the full-history model input",
            **result}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--witness-shard", type=Path)
    parser.add_argument("--witness-row", type=int,default=819)
    args = parser.parse_args()
    if args.witness_shard:
        result = ambiguity_witness(args.witness_shard,args.witness_row)
        write_json_once(args.output,result)
        print(json.dumps(result,indent=2))
        return
    records = []
    for path in sorted(args.fixtures.glob("*/*.npz")):
        record = audit_shard(path)
        if record["gaps"]:
            records.append(record)
            print(json.dumps(record), flush=True)
    write_json_once(args.output, {"schema_version": "hero-missing-move-uniqueness-v1", "records": records,
                                  "scope": "Bounded legal-path diagnostic; no training data generated"})


if __name__ == "__main__":
    main()
