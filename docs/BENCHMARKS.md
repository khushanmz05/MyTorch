# Benchmarks

Generate a dataset, train each architecture, then benchmark it:

```bash
make dataset
python3 my_torch_generator configs/basic.conf 1
python3 my_torch_analyzer --train --epochs 40 --save my_torch_network_basic.nn my_torch_network_basic.nn data/chess_positions.txt
python3 scripts/benchmark.py my_torch_network_basic.nn data/chess_positions.txt
```

For every model, record validation loss/accuracy per epoch, final accuracy,
prediction latency, parameter count, and training time. Include learning-curve
plots or CSV exports in `docs/` so architectural choices can be compared.
