#!/usr/bin/env python3
"""Generate a supervised chess-position dataset using python-chess."""

import argparse
import random

try:
    import chess
except ImportError as exc:
    raise SystemExit("python-chess is required: pip install python-chess") from exc


def label_for(board):
    if board.is_checkmate():
        loser = "White" if board.turn == chess.WHITE else "Black"
        return f"Checkmate {loser}"
    if board.is_check():
        side = "White" if board.turn == chess.WHITE else "Black"
        return f"Check {side}"
    return "Nothing"


def random_position(max_plies):
    board = chess.Board()
    for _ in range(random.randint(1, max_plies)):
        legal = list(board.legal_moves)
        if not legal:
            break
        board.push(random.choice(legal))
    return board


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--positions", type=int, default=50000)
    parser.add_argument("--max-plies", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="data/chess_positions.txt")
    args = parser.parse_args()

    random.seed(args.seed)
    seen, rows = set(), []
    while len(rows) < args.positions:
        board = random_position(args.max_plies)
        fen = board.fen()
        if fen in seen:
            continue
        seen.add(fen)
        rows.append(f"{fen} {label_for(board)}")

    with open(args.output, "w", encoding="utf-8") as fh:
        fh.write("\n".join(rows) + "\n")
    print(f"wrote {len(rows)} positions to {args.output}")


if __name__ == "__main__":
    main()
