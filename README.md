# my_torch

`my_torch` is a dependency-light neural-network library and chessboard analyzer
implemented from scratch with NumPy. It provides dense layers, ReLU/sigmoid/tanh,
softmax cross-entropy, mini-batch SGD, Adam, inverted dropout, L2 weight decay,
and early stopping. No PyTorch, TensorFlow, or other neural-network framework is used.

## Executables

```text
./my_torch_analyzer [--predict | --train [--save SAVEFILE]] LOADFILE CHESSFILE
./my_torch_generator config_file nb_1 [config_file_2 nb_2...]
```

The analyzer reads one FEN position per line and prints, in input order:

- `Nothing`
- `Check White`
- `Check Black`
- `Checkmate White`
- `Checkmate Black`

Training input uses `FEN LABEL` lines.

## Input representation

Each FEN becomes a 781-value vector: 12 one-hot piece planes (768), active color
(1), castling rights (4), and en-passant file (8). The output layer has five
softmax units corresponding to the labels above.

## Quick start

```bash
make install
make dataset
make generate
make train
./my_torch_analyzer --predict my_torch_network_full.nn examples/chessboards.txt
```

Generate blank networks with:

```bash
./my_torch_generator configs/basic.conf 3
```

## Overfitting controls

Training uses a validation split, mini-batch optimization, dropout, L2 weight
decay, and early stopping on validation loss.

## Layout

- `my_torch/` — reusable neural-network module
- `my_torch_analyzer` — training and prediction executable
- `my_torch_generator` — blank-network generator
- `configs/` — architecture configurations
- `scripts/` — dataset generation and benchmarking
- `tests/` — unit tests
- `docs/` — benchmark documentation
- `models/` — network artifacts

## Delivery checklist

- [x] Neural-network module with creation, training, saving, and loading
- [x] Analyzer with `--predict` and `--train` modes
- [x] Network generator and configuration files
- [x] Supervised dataset-generation workflow
- [x] Overfitting controls and benchmark tooling
- [x] `my_torch_network_full.nn` artifact at repository root
