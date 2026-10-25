"""my_torch: a small supervised-learning neural-network library."""

from .network import NeuralNetwork
from .config import NetworkConfig
from .data import encode_fen, load_chess_dataset, LABELS

__all__ = ["NeuralNetwork", "NetworkConfig", "encode_fen", "load_chess_dataset", "LABELS"]
