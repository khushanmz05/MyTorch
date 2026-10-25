# my_torch

`my_torch` is a neural-network library and chessboard analyzer implemented
from scratch with NumPy. It provides dense layers, ReLU/sigmoid/tanh,
softmax cross-entropy, mini-batch SGD, Adam, inverted dropout, L2 weight
decay, and early stopping.

No PyTorch, TensorFlow, or other neural-network framework is used.

## Executables

```text
./my_torch_analyzer [--predict | --train [--save SAVEFILE]] LOADFILE CHESSFILE
./my_torch_generator config_file nb_1 [config_file_2 nb_2...]
```

The analyzer reads one FEN position per line and prints predictions in
the same order as the input positions.

Supported labels:

- `Nothing`
- `Check White`
- `Check Black`
- `Checkmate White`
- `Checkmate Black`

The color identifies the attacking side for check and the winning side
for checkmate, following the supplied dataset's convention.

Training input uses six FEN fields followed by the expected label:

```text
8/8/R2k4/4r1p1/8/5K2/5P2/8 b - - 7 59 Check White
```

## Trained model

The trained model is provided at the repository root:

```text
my_torch_network_basic_v2.nn
```

Its held-out test accuracy is 85.43% on 143109 positions.

Predict with the supplied model:

```bash
python3 my_torch_analyzer --predict my_torch_network_basic_v2.nn examples/chessboards.txt
```

On Windows PowerShell, using the project's virtual environment:

```powershell
.\.venv\Scripts\python.exe my_torch_analyzer --predict my_torch_network_basic_v2.nn examples/chessboards.txt
```

## Input representation

Each FEN becomes a 781-value vector:

- 12 one-hot piece planes over 64 squares: 768 values
- Active color: 1 value
- Castling rights: 4 values
- En-passant file: 8 values

The encoder does not use the halfmove clock or fullmove number.

The output layer produces five raw scores (logits), one per label.
The cross-entropy loss applies softmax internally during training.
`predict_proba()` applies softmax to return class probabilities.

## Installation

Install the dependencies into your chosen Python environment:

```bash
python3 -m pip install -r requirements.txt
```

On Windows PowerShell, with an existing project virtual environment:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

NumPy is used for the neural-network implementation. Pytest is used for
tests. The optional dataset-generation workflow uses python-chess.

## Dataset preparation

Place the supplied labeled dataset in this structure:

```text
dataset/
├── check/
│   ├── 10_pieces.txt
│   ├── 20_pieces.txt
│   └── many_pieces.txt
├── checkmate/
│   ├── 10_pieces.txt
│   ├── 20_pieces.txt
│   └── many_pieces.txt
└── nothing/
    ├── 10_pieces.txt
    ├── 20_pieces.txt
    └── many_pieces.txt
```

From the project root, run:

```bash
python3 scripts/prepare_dataset.py
```

The preparation script preserves the existing labels, removes duplicate
model inputs, and creates a reproducible, approximately 80/20 split
within each class using seed 42.

Generated files:

```text
data/train.txt
data/test.txt
```

For the reported run:

- Training-file positions: 572450
- Test positions: 143109
- Duplicate positions removed: 7460

The raw dataset and generated splits are excluded from Git. Keep local
copies available for reproducing the training run.

## Reproduce training

Generate a fresh blank network:

```bash
python3 my_torch_generator configs/basic.conf 1
```

Train it and save the best validation-loss checkpoint:

```bash
python3 my_torch_analyzer --train --epochs 30 --batch-size 64 --learning-rate 0.001 --save my_torch_network_basic_v2.nn configs/basic_1.nn data/train.txt
```

This command overwrites the supplied model file when an improved
checkpoint is saved. Use a different `--save` filename if you want to
preserve the supplied model.

The analyzer reserves 15% of the training file for internal validation.
The separate test file is not used by this training command.

Blank networks can also be generated in batches:

```bash
python3 my_torch_generator configs/basic.conf 3
```

## Overfitting controls

Training provides:

- An internal validation split
- Dropout
- L2 weight decay
- Early stopping based on validation loss
- Saving the checkpoint with the lowest validation loss

Mini-batch training and shuffling are also used.

## Results

Training configuration:

- Architecture: `configs/basic.conf`
- Hidden layers: 128 and 64 units
- Hidden activation: ReLU
- Dropout rate: 0.05
- Optimizer: Adam
- Learning rate: 0.001
- Batch size: 64
- Weight decay: 0.00001
- Completed epochs: 30
- Best validation-loss checkpoint: epoch 28

Measured results:

| Metric | Result |
|---|---:|
| Best validation loss | 0.35971 |
| Validation accuracy at best checkpoint | 85.53% |
| Held-out test positions | 143109 |
| Held-out test accuracy | 85.43% |
| Average batched prediction time per position | 0.0029 ms |

The prediction benchmark processes the test inputs as one batch.
Its time per position is total prediction time divided by sample count,
not independently measured single-request latency.

The reported class counts describe the dataset distribution, not
per-class accuracy. Overall accuracy alone does not establish equal
performance across all classes.

See `docs/BENCHMARKS.md` and `docs/benchmark-results.txt` for details.

## Benchmark command

Run from the project root:

```bash
python3 -m scripts.benchmark my_torch_network_basic_v2.nn data/test.txt
```

## Testing

```bash
python3 -m pytest -q
```

On Windows, if pytest cannot access its default temporary directory:

```powershell
.\.venv\Scripts\python.exe -m pytest -q --basetemp=.pytest_tmp
```

## Layout

- `my_torch/` — reusable neural-network module and FEN encoder
- `my_torch_analyzer` — training and prediction executable
- `my_torch_generator` — blank-network generator
- `configs/` — architecture configurations
- `scripts/prepare_dataset.py` — supplied-dataset preparation
- `scripts/generate_dataset.py` — alternative position generator
- `scripts/benchmark.py` — accuracy and prediction-time benchmark
- `tests/` — unit tests
- `docs/` — benchmark documentation and measured results
- `dataset/` — local supplied dataset, excluded from Git
- `data/` — generated train/test splits, excluded from Git
- `my_torch_network_basic_v2.nn` — supplied trained model

The alternative dataset generator was not used for the reported run.
Its label-color convention must be aligned with the supplied dataset
before mixing generated positions into this training workflow.

## Delivery checklist

- [x] Neural-network creation, training, saving, and loading
- [x] Analyzer with prediction and training modes
- [x] Network generator and configuration files
- [x] Reproducible supplied-dataset preparation script
- [x] Overfitting controls and benchmark tooling
- [x] Trained `my_torch_network_basic_v2.nn` at repository root
- [x] Measured held-out test results