"""Gradient-descent optimizers."""

import numpy as np


class SGD:
    def __init__(self, learning_rate=0.01, momentum=0.0, weight_decay=0.0):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.velocity = {}

    def step(self, layer, layer_id):
        for grad, name in layer.gradients():
            if grad is None:
                continue
            if self.weight_decay and name == "weights":
                grad = grad + self.weight_decay * layer.weights
            key = (layer_id, name)
            if self.momentum:
                self.velocity[key] = self.momentum * self.velocity.get(key, 0.0) - self.learning_rate * grad
                update = self.velocity[key]
            else:
                update = -self.learning_rate * grad
            setattr(layer, name, getattr(layer, name) + update)


class Adam:
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, weight_decay=0.0):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.m = {}
        self.v = {}
        self.t = 0

    def step(self, layer, layer_id):
        self.t += 1
        for grad, name in layer.gradients():
            if grad is None:
                continue
            if self.weight_decay and name == "weights":
                grad = grad + self.weight_decay * layer.weights
            key = (layer_id, name)
            param = getattr(layer, name)
            self.m[key] = self.beta1 * self.m.get(key, 0.0) + (1 - self.beta1) * grad
            self.v[key] = self.beta2 * self.v.get(key, 0.0) + (1 - self.beta2) * grad**2
            m_hat = self.m[key] / (1 - self.beta1**self.t)
            v_hat = self.v[key] / (1 - self.beta2**self.t)
            setattr(layer, name, param - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon))
