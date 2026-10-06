PYTHON_VERSION := $(shell cat .python-version)
VENV := .venv
PY := $(VENV)/bin/python

.PHONY: setup data test

setup:
	python$(PYTHON_VERSION) -m venv $(VENV)
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements.txt

data:
	PYTHONPATH=src $(PY) -m fraud.data

test:
	$(PY) -m pytest
