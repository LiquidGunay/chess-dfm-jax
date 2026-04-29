#!/usr/bin/env python3
import os
import sys
import tempfile
import io
import chess
import chess.pgn
import numpy as np
import jax
import jax.numpy as jnp
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.process_tcec_to_gcs import slice_game
from chess_dfm_jax.data.leela import LeelaChunkDataLoader
from chess_dfm_jax.analysis.profile_targets import load_mapped_bt4_params
from chess_dfm_jax.training.dfm import create_dfm_components, train_dfm_step, DFMConfig

from scripts.compare_logits import check_bt4_parity
from tests.test_muon import test_muon

def test_end_to_end():
    print("Running Strict Component Checks...")
    check_bt4_parity()
    test_muon()
    
    print("\nTesting End-to-End Data and Training Pipeline...")

    # 1. Create a dummy PGN
    pgn_text = """[Event "FIDE World Cup 2017"]
[Site "Tbilisi GEO"]
[Date "2017.09.09"]
[Round "3.1"]
[White "Carlsen,M"]
[Black "Bu Xiangzhi"]
[Result "0-1"]
[WhiteElo "2827"]
[BlackElo "2710"]
[EventDate "2017.09.03"]
[ECO "C55"]

1. e4 e5 2. Nf3 Nc6 3. Bc4 Nf6 4. d3 h6 5. O-O d6 6. c3 g6 7. Re1 Bg7 8. h3 O-O 9. Nbd2 Re8 10. b4 a6 11. a4 Be6 12. Bxe6 Rxe6 13. Bb2 d5 14. b5 Ne7 15. c4 axb5 16. axb5 d4 17. Rxa8 Qxa8 18. Qb3 Nd7 19. Ra1 Qb8 20. Ba3 Nc8 21. c5 c6 22. Nc4 Na7 23. b6 Nxb6 24. cxb6 Nb5 25. Bb2 Re8 26. Ra5 Qd8 27. Rxb5 cxb5 28. Qxb5 Re6 29. Ba3 Kh7 30. Nfd2 h5 31. Nb3 Qg5 32. Bc1 Qe7 33. Nc5 Rc6 34. Ba3 Qg5 35. g3 h4 36. Kg2 hxg3 37. fxg3 Qh5 38. g4 Qg5 39. Nxb7 Rf6 40. Qb2 Qh4 41. Qe2 Bh6 42. Be7 Rf2+ 43. Qxf2 Qxe7 44. Nbd6 Bf4 45. b7 Qd8 46. Qb2 Qh4 47. b8=Q Qg3+ 48. Kf1 Qxd3+ 49. Qe2 Qxh3+ 50. Qg2 Qd3+ 51. Kg1 Qd1+ 52. Qf1 Qxg4+ 53. Kf2 Qg3+ 54. Ke2 d3+ 55. Kd1 Qg4+ 0-1
"""
    game = chess.pgn.read_game(io.StringIO(pgn_text))
    
    print("Slicing game into chunks...")
    samples = list(slice_game(game, horizon=8))
    assert len(samples) > 0, "No samples generated from game!"
    
    with tempfile.TemporaryDirectory() as tmpdir:
        chunk_path = os.path.join(tmpdir, "chunk_000000.npz")
        
        planes_t = np.stack([s["planes_t"] for s in samples])
        actions = np.stack([s["actions"] for s in samples])
        planes_future = np.stack([s["planes_future"] for s in samples])
        future_valid = np.stack([s["future_valid"] for s in samples])
        legal_masks = np.stack([s["legal_masks"] for s in samples])
        value_targets = np.stack([s["value_targets"] for s in samples])
        wdl_targets = np.stack([s["wdl_targets"] for s in samples])
        
        np.savez_compressed(
            chunk_path,
            schema_version=np.asarray("trajectory-v2"),
            planes_t=planes_t,
            actions=actions,
            planes_future=planes_future,
            future_valid=future_valid,
            legal_masks=legal_masks,
            value_targets=value_targets,
            wdl_targets=wdl_targets,
        )
        
        print(f"Saved dummy chunk to {chunk_path}. Loading with LeelaChunkDataLoader...")
        loader = LeelaChunkDataLoader([chunk_path], batch_size=8, seed=42, horizon=8)
        batch_iter = iter(loader)
        
        batch = next(batch_iter)
        assert batch["current_planes"].shape == (8, 112, 8, 8), f"Wrong planes shape: {batch['current_planes'].shape}"
        assert batch["action_indices"].shape == (8, 8), f"Wrong actions shape: {batch['action_indices'].shape}"
        assert batch["future_planes"].shape == (8, 8, 112, 8, 8), f"Wrong future planes shape: {batch['future_planes'].shape}"
        
        print("Data Loading successful. Initializing model...")
        models_dir = REPO_ROOT / "models"
        pb_path = None
        if models_dir.exists():
            for f in models_dir.iterdir():
                if f.name.endswith(".pb.gz") and "exported" not in f.name:
                    pb_path = f
                    break
        
        if pb_path is None:
            print(f"Warning: No .pb.gz found in {models_dir}. Cannot run training pass.")
            return
            
        params = load_mapped_bt4_params(pb=str(pb_path))
        config = DFMConfig(token_dim=512, num_layers=2, use_muon=False)
        model, optimizer = create_dfm_components(params, config)
        
        print("Running training steps...")
        rng = jax.random.PRNGKey(0)
        
        for i in range(3):
            rng, step_rng = jax.random.split(rng)
            try:
                batch = next(batch_iter)
            except StopIteration:
                batch_iter = iter(loader)
                batch = next(batch_iter)
                
            loss, aux = train_dfm_step(model, optimizer, batch, step_rng)
            print(f"Step {i} | Loss: {loss:.4f} | Legality Loss: {aux.get('legality_loss', 0.0):.6f} | Mask Prob: {aux['mask_prob']:.4f}")
            
        print("End-to-End Pipeline test PASSED!")

if __name__ == "__main__":
    test_end_to_end()
