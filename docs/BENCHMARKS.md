# Benchmarks

Generate a dataset, train each architecture, then benchmark it:

### Held-out test results

- Samples: 143109
- Accuracy: 85.43%
- Average inference time per position: 0.0029 ms
- Model: my_torch_network_basic_v2.nn

Test-set class counts:
- Nothing: 59512
- Check White: 26998
- Check Black: 22628
- Checkmate White: 18466
- Checkmate Black: 15505

The benchmark predicts the test set as one batch. The reported
time per position is total prediction time divided by sample count,
not independently measured single-position request latency.

The class counts describe the test labels, not per-class accuracy.

```bash
make dataset
python3 my_torch_generator configs/basic.conf 1
python3 my_torch_analyzer --train --epochs 40 --save my_torch_network_basic.nn my_torch_network_basic.nn data/chess_positions.txt
python3 scripts/benchmark.py my_torch_network_basic.nn data/chess_positions.txt
```

For every model, record validation loss/accuracy per epoch, final accuracy,
prediction latency, parameter count, and training time. Include learning-curve
plots or CSV exports in `docs/` so architectural choices can be compared.
