PY ?= python

.PHONY: install test run monitor ablation clean

install:
	$(PY) -m pip install -e ".[dev]"

test:
	$(PY) -m pytest

run:
	$(PY) -m idxf run

monitor:
	$(PY) -m idxf monitor

ablation:
	$(PY) scripts/run_ablation.py

clean:
	rm -rf output/*
