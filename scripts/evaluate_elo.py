#!/usr/bin/env python3
import chess
import chess.engine
import asyncio
import sys
import math

async def play_game(engine1, engine2, time_limit):
    board = chess.Board()
    engines = [engine1, engine2]
    turn = 0
    while not board.is_game_over(claim_draw=True):
        try:
            result = await engines[turn].play(board, chess.engine.Limit(time=time_limit))
            if result.move is None:
                break
            board.push(result.move)
        except Exception as e:
            print(f"Engine {turn} crashed: {e}")
            return "1-0" if turn == 1 else "0-1"
        turn = 1 - turn
    return board.result(claim_draw=True)

async def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--e1", type=str, required=True, help="Command to run engine 1")
    parser.add_argument("--e2", type=str, required=True, help="Command to run engine 2")
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--time", type=float, default=1.0)
    args = parser.parse_args()

    e1_wins = 0
    e2_wins = 0
    draws = 0

    for i in range(args.games):
        _, engine1 = await chess.engine.popen_uci(args.e1.split())
        _, engine2 = await chess.engine.popen_uci(args.e2.split())
        
        # Alternate colors
        if i % 2 == 0:
            res = await play_game(engine1, engine2, args.time)
            if res == "1-0": e1_wins += 1
            elif res == "0-1": e2_wins += 1
            else: draws += 1
        else:
            res = await play_game(engine2, engine1, args.time)
            if res == "1-0": e2_wins += 1
            elif res == "0-1": e1_wins += 1
            else: draws += 1

        print(f"Game {i+1}: {res} | E1: {e1_wins} E2: {e2_wins} Draws: {draws}")
        await engine1.quit()
        await engine2.quit()

    total = args.games
    score = e1_wins + 0.5 * draws
    expected = score / total if total > 0 else 0
    
    if expected > 0 and expected < 1:
        elo_diff = -400 * math.log10(1/expected - 1)
        print(f"\nMatch Finished! Elo difference (E1 - E2): {elo_diff:+.2f}")
    else:
        print("\nMatch Finished! Elo difference infinite (one engine scored 100%).")

    import os
    try:
        with open("/tmp/illegal_moves.log", "r") as f:
            illegal_count = len(f.readlines())
        print(f"\nTotal illegal move fallbacks by E1: {illegal_count}")
        os.remove("/tmp/illegal_moves.log")
    except FileNotFoundError:
        print("\nTotal illegal move fallbacks by E1: 0")

if __name__ == "__main__":
    asyncio.run(main())
