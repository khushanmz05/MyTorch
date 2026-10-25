#!/usr/bin/env python3
"""Benchmark prediction latency and accuracy for a trained network."""

import argparse
import time

import numpy as np

from my_torch import NeuralNetwork, load_chess_dataset, LABELS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("network")
    parser.add_argument("dataset")
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    network = NeuralNetwork.load(args.network)
    x, y = load_chess_dataset(args.dataset)
    start = time.perf_counter()
    predictions = network.predict(x)
    elapsed = time.perf_counter() - start
    print(f"samples={len(y)}")
    print(f"accuracy={np.mean(predictions == y):.4f}")
    print(f"latency_ms_per_sample={1000.0 * elapsed / len(y):.4f}")
    for index, label in enumerate(LABELS):
        print(f"{label}: {int(np.sum(y == index))}")


if __name__ == "__main__":
    main()
