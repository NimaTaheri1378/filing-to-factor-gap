.PHONY: smoke test lint figures tables docs full clean

PYTHON ?= python

smoke:
	$(PYTHON) -m ftfg.cli --config configs/sample_period.yml smoke

test:
	$(PYTHON) -m pytest

lint:
	$(PYTHON) -m ruff check src tests

figures:
	$(PYTHON) -m ftfg.cli --config configs/sample_period.yml make-figures

tables:
	$(PYTHON) -m ftfg.cli --config configs/sample_period.yml make-tables

docs:
	$(PYTHON) -m mkdocs build

full:
	$(PYTHON) -m ftfg.cli --config configs/sample_period.yml full

clean:
	$(PYTHON) -m ftfg.cli clean
