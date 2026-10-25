"""FEN encoding and chess-dataset utilities."""

import numpy as np

PIECES = "PNBRQKpnbrqk"
LABELS = [
    "Nothing",
    "Check White",
    "Check Black",
    "Checkmate White",
    "Checkmate Black",
]
LABEL_TO_INDEX = {label: i for i, label in enumerate(LABELS)}


def encode_fen(fen):
    """Encode a FEN string into a fixed-length floating-point vector."""
    fields = fen.split()
    if len(fields) < 4:
        raise ValueError(f"Invalid FEN: {fen!r}")
    board, active, castling, en_passant = fields[:4]

    features = np.zeros(781, dtype=np.float32)
    square = 0
    for char in board:
        if char == "/":
            continue
        if char.isdigit():
            square += int(char)
        else:
            features[PIECES.index(char) * 64 + square] = 1.0
            square += 1
    if square != 64:
        raise ValueError(f"Invalid FEN board: {board!r}")

    features[768] = 1.0 if active == "w" else 0.0
    for i, right in enumerate("KQkq"):
        features[769 + i] = 1.0 if right in castling else 0.0
    if en_passant != "-":
        features[773 + "abcdefgh".index(en_passant[0])] = 1.0
    return features


def encode_batch(fens):
    return np.vstack([encode_fen(fen) for fen in fens])


def load_chess_dataset(path):
    """Load `FEN LABEL` lines from a chess dataset."""
    fens, labels = [], []
    with open(path, "r", encoding="utf-8") as fh:
        for raw in fh:
            row = raw.rstrip("\n\r")
            if not row:
                continue
            parts = row.split()

            if len(parts) < 7:
                raise ValueError("Expected six FEN fields followed by a label")

            fen = " ".join(parts[:6])
            label = " ".join(parts[6:])
            if label not in LABEL_TO_INDEX:
                raise ValueError(f"Unknown label {label!r}")
            fens.append(fen.strip())
            labels.append(LABEL_TO_INDEX[label])
    return encode_batch(fens), np.array(labels, dtype=np.int64)
