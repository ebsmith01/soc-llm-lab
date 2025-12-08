PYTHON ?= python3

.PHONY: run ingest test clean

run:
	uvicorn service.api:app --reload --app-dir .

ingest:
	$(PYTHON) scripts/ingest.py

test:
	PYTHONPATH=. pytest

clean:
	rm -rf __pycache__ */__pycache__
	rm -rf .pytest_cache
