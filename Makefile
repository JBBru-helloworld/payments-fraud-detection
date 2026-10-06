PYTHON_VERSION := $(shell cat .python-version)
VENV := .venv
PY := $(VENV)/bin/python

.PHONY: setup data eda split baselines features models threshold final-eval explain sample-request serve docker-build docker-run test

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

explain:
	PYTHONPATH=src $(PY) -m fraud.explain

sample-request:
	PYTHONPATH=src $(PY) -m fraud.payload

serve:
	PYTHONPATH=src:. $(PY) -m uvicorn app.main:app --host 127.0.0.1 --port 8000

docker-build:
	docker build -t payments-fraud-api .

docker-run:
	docker run --rm -p 8000:8000 payments-fraud-api

test:
	$(PY) -m pytest
