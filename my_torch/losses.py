"""Loss functions for classification."""

import numpy as np


class SoftmaxCrossEntropy:
    """Combines a softmax output layer with categorical cross-entropy."""

    def forward(self, logits, targets):
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        exp = np.exp(shifted)
        self.probs = exp / np.sum(exp, axis=1, keepdims=True)
        n = logits.shape[0]
        picked = self.probs[np.arange(n), targets]
        return float(np.mean(-np.log(np.clip(picked, 1e-12, None))))

    def backward(self, targets):
        n = self.probs.shape[0]
        grad = self.probs.copy()
        grad[np.arange(n), targets] -= 1.0
        return grad / n
