PYTHON ?= python3

.PHONY: run test clean

run:
	$(PYTHON) agent/main.py

test:
	PYTHONPATH=. pytest

clean:
	rm -rf __pycache__ .pytest_cache
