"""Activation functions and their derivatives."""

import numpy as np


def relu(x):
    return np.maximum(0.0, x)


def relu_backward(x, grad):
    return grad * (x > 0)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60, 60)))


def sigmoid_backward(x, grad):
    s = sigmoid(x)
    return grad * s * (1.0 - s)


def tanh(x):
    return np.tanh(x)


def tanh_backward(x, grad):
    return grad * (1.0 - np.tanh(x) ** 2)


def softmax(x):
    shifted = x - np.max(x, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=1, keepdims=True)


ACTIVATIONS = {
    "relu": (relu, relu_backward),
    "sigmoid": (sigmoid, sigmoid_backward),
    "tanh": (tanh, tanh_backward),
    "softmax": (softmax, None),
}


def get_activation(name):
    try:
        return ACTIVATIONS[name]
    except KeyError as exc:
        raise ValueError(f"Unknown activation: {name}") from exc
