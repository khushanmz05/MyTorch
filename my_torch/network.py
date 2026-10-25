"""The core feed-forward neural-network implementation."""

import json
import os

import numpy as np

from .layers import Activation, Dense, Dropout
from .losses import SoftmaxCrossEntropy
from .optimizers import SGD, Adam


class NeuralNetwork:
    def __init__(self, layer_sizes, hidden_activation="relu", dropout=0.0, seed=42):
        if len(layer_sizes) < 3:
            raise ValueError("A network needs input, hidden and output layers")
        self.layer_sizes = list(layer_sizes)
        self.hidden_activation = hidden_activation
        self.dropout_rate = dropout
        self.rng = np.random.default_rng(seed)
        self.layers = []
        self.loss_fn = SoftmaxCrossEntropy()

        for i in range(len(layer_sizes) - 1):
            self.layers.append(Dense(layer_sizes[i], layer_sizes[i + 1], self.rng))
            is_output = i == len(layer_sizes) - 2
            self.layers.append(Activation("softmax" if is_output else hidden_activation))
            if not is_output and dropout > 0.0:
                self.layers.append(Dropout(dropout, self.rng))

    @classmethod
    def from_config(cls, config):
        return cls(
            config.layer_sizes(),
            hidden_activation=config.hidden_activation,
            dropout=config.dropout,
            seed=config.seed,
        )

    def forward(self, x, training=False):
        for layer in self.layers:
            x = layer.forward(x, training=training)
        return x

    def predict(self, x):
        logits = self.forward(np.atleast_2d(x), training=False)
        return np.argmax(logits, axis=1)

    def predict_proba(self, x):
        return self.forward(np.atleast_2d(x), training=False)

    def train_batch(self, x, targets, optimizer):
        logits = self.forward(x, training=True)
        loss = self.loss_fn.forward(logits, targets)
        grad = self.loss_fn.backward(targets)
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        for i, layer in enumerate(self.layers):
            optimizer.step(layer, i)
        return loss

    def evaluate(self, x, targets, batch_size=256):
        correct = 0
        total_loss = 0.0
        for start in range(0, len(x), batch_size):
            xb = x[start:start + batch_size]
            tb = targets[start:start + batch_size]
            logits = self.forward(xb, training=False)
            total_loss += self.loss_fn.forward(logits, tb) * len(xb)
            correct += int(np.sum(np.argmax(logits, axis=1) == tb))
        return total_loss / len(x), correct / len(x)

    def save(self, path):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        arrays = {}
        for i, layer in enumerate(self.layers):
            if isinstance(layer, Dense):
                arrays[f"layer_{i}_weights"] = layer.weights
                arrays[f"layer_{i}_biases"] = layer.biases
        metadata = {
            "format": "my_torch-npz-v1",
            "layer_sizes": self.layer_sizes,
            "hidden_activation": self.hidden_activation,
            "dropout": self.dropout_rate,
        }
        with open(path, "wb") as fh:
            np.savez_compressed(
                fh,
                __metadata__=np.array(json.dumps(metadata)),
                **arrays,
            )

    @classmethod
    def load(cls, path):
        data = np.load(path, allow_pickle=False)
        metadata = json.loads(str(data["__metadata__"]))
        network = cls(
            metadata["layer_sizes"],
            hidden_activation=metadata["hidden_activation"],
            dropout=metadata["dropout"],
        )
        for i, layer in enumerate(network.layers):
            if isinstance(layer, Dense):
                layer.weights = data[f"layer_{i}_weights"]
                layer.biases = data[f"layer_{i}_biases"]
        return network
