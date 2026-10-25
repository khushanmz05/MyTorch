PYTHON ?= python3
PIP ?= $(PYTHON) -m pip

.PHONY: all install test generate train predict benchmark dataset clean fclean

all: install

install:
	$(PIP) install -r requirements.txt

test:
	$(PYTHON) -m pytest -q

dataset:
	$(PYTHON) scripts/generate_dataset.py --positions 50000 --output data/chess_positions.txt

generate:
	$(PYTHON) my_torch_generator configs/full.conf 1

train:
	$(PYTHON) my_torch_analyzer --train --save my_torch_network_full.nn my_torch_network_full.nn data/chess_positions.txt

predict:
	$(PYTHON) my_torch_analyzer --predict my_torch_network_full.nn examples/chessboards.txt

benchmark:
	$(PYTHON) scripts/benchmark.py my_torch_network_full.nn data/chess_positions.txt

clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .pytest_cache

fclean: clean
	rm -f *.nn
