import chess

from research.analysis.audit_missing_hero_moves import legal_paths


def test_exact_fen_can_hide_distinct_legal_promotion_histories():
    board = chess.Board("8/8/8/8/8/3k2K1/4pp2/8 b - - 1 101")
    end = "8/8/8/8/8/3k3K/8/4qr2 w - - 0 103"
    result = legal_paths(board,end,3)
    assert result["status"] == "ambiguous"
    assert len(result["paths"]) == 2
    for path in result["paths"]:
        replay=board.copy()
        for uci in path:replay.push_uci(uci)
        assert replay.fen() == end


def test_search_bound_is_not_a_uniqueness_certificate():
    board=chess.Board()
    result=legal_paths(board,board.fen(),4,node_limit=1)
    assert result["status"] == "bound_exhausted"


def test_verified_unique_five_ply_gap():
    board=chess.Board("8/5k2/5p2/p1pPpP2/P1P5/6K1/1p5p/8 b - - 3 49")
    end="8/5k2/5p2/p1pPpP2/P1P5/7q/4K3/1q6 w - - 0 52"
    result=legal_paths(board,end,5)
    assert result["status"] == "unique"
    assert result["paths"] == [["h2h1q","g3f2","h1h3","f2e2","b2b1q"]]
