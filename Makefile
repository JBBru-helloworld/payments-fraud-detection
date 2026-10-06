PYTHON_VERSION := $(shell cat .python-version)
VENV := .venv
PY := $(VENV)/bin/python

.PHONY: setup data eda split baselines features test

setup:
	python$(PYTHON_VERSION) -m venv $(VENV)
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements.txt

data:
	PYTHONPATH=src $(PY) -m fraud.data

eda:
	PYTHONPATH=src $(PY) -m fraud.eda

split:
	PYTHONPATH=src $(PY) -m fraud.split

baselines: split
	PYTHONPATH=src $(PY) -m fraud.train

features:
	PYTHONPATH=src $(PY) -m fraud.features
	PYTHONPATH=src $(PY) -m fraud.train --engineered-check

test:
	$(PY) -m pytest
