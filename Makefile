PYTHON_VERSION := $(shell cat .python-version)
VENV := .venv
PY := $(VENV)/bin/python

.PHONY: setup data eda split baselines features models threshold final-eval test

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

models:
	PYTHONPATH=src $(PY) -m fraud.train --models

threshold:
	PYTHONPATH=src $(PY) -m fraud.threshold

final-eval:
	PYTHONPATH=src $(PY) -m fraud.final_eval

test:
	$(PY) -m pytest
