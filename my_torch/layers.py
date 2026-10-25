"""Trainable and regularized layers used by my_torch."""

import numpy as np

from .activations import get_activation


class Dense:
    def __init__(self, input_size, output_size, rng):
        limit = np.sqrt(6.0 / (input_size + output_size))
        self.weights = rng.uniform(-limit, limit, size=(input_size, output_size))
        self.biases = np.zeros(output_size)
        self.input = None
        self.grad_weights = None
        self.grad_biases = None

    def forward(self, x, training=False):
        self.input = x
        return x @ self.weights + self.biases

    def backward(self, grad):
        self.grad_weights = self.input.T @ grad
        self.grad_biases = np.sum(grad, axis=0)
        return grad @ self.weights.T

    def parameters(self):
        return [(self.weights, "weights"), (self.biases, "biases")]

    def gradients(self):
        return [(self.grad_weights, "weights"), (self.grad_biases, "biases")]


class Activation:
    def __init__(self, name):
        self.name = name
        self.forward_fn, self.backward_fn = get_activation(name)
        self.input = None

    def forward(self, x, training=False):
        self.input = x
        return self.forward_fn(x)

    def backward(self, grad):
        if self.backward_fn is None:
            return grad
        return self.backward_fn(self.input, grad)

    def parameters(self):
        return []

    def gradients(self):
        return []


class Dropout:
    """Inverted dropout: active only during training."""

    def __init__(self, rate, rng):
        if not 0.0 <= rate < 1.0:
            raise ValueError("Dropout rate must be in [0, 1)")
        self.rate = rate
        self.rng = rng
        self.mask = None

    def forward(self, x, training=False):
        if not training or self.rate == 0.0:
            return x
        keep = 1.0 - self.rate
        self.mask = (self.rng.random(x.shape) < keep) / keep
        return x * self.mask

    def backward(self, grad):
        if self.mask is None:
            return grad
        return grad * self.mask

    def parameters(self):
        return []

    def gradients(self):
        return []
