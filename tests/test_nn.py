import numpy as np

from my_torch import NeuralNetwork
from my_torch.activations import softmax
from my_torch.data import encode_fen
from my_torch.optimizers import Adam


def test_forward_shape():
    network = NeuralNetwork([4, 8, 3], seed=1)
    assert network.forward(np.zeros((5, 4))).shape == (5, 3)


def test_softmax_sums_to_one():
    assert np.allclose(softmax(np.array([[1.0, 2.0, 3.0]])).sum(axis=1), 1.0)


def test_training_reduces_loss():
    rng = np.random.default_rng(7)
    x = rng.normal(size=(200, 4))
    y = (x[:, 0] > 0).astype(np.int64)
    network = NeuralNetwork([4, 16, 2], seed=7)
    optimizer = Adam(0.01)
    first = network.train_batch(x[:32], y[:32], optimizer)
    last = first
    for _ in range(50):
        last = network.train_batch(x, y, optimizer)
    assert last < first


def test_save_and_load_roundtrip(tmp_path):
    network = NeuralNetwork([4, 8, 3], seed=3)
    path = tmp_path / "roundtrip.nn"
    network.save(str(path))
    loaded = NeuralNetwork.load(str(path))
    x = np.random.default_rng(4).normal(size=(6, 4))
    assert np.allclose(network.predict_proba(x), loaded.predict_proba(x))


def test_fen_encoding():
    features = encode_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    assert features.shape == (781,)
    assert features[768] == 1.0
    assert features.sum() == 32 + 1 + 4
