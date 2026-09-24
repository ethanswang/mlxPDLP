# Included only by an explicitly invoked reference solver Makefile.
PYTHON ?= python3
.DEFAULT_GOAL := help
.PHONY: help setup run

help:
	@echo 'make setup [PYTHON=python3.12]   Install this reference in its own .venv'
	@echo 'make run ARGS="--help"          Set up on demand, then run the benchmark'

.venv/bin/python:
	$(PYTHON) -c 'import sys; sys.exit(0 if sys.version_info >= (3, 12) else "Python 3.12+ required; set PYTHON to a suitable interpreter")'
	$(PYTHON) -m venv .venv

.venv/.requirements: requirements.txt .venv/bin/python
	.venv/bin/python -m pip install --cache-dir .venv/pip-cache -r requirements.txt
	touch $@

setup: .venv/.requirements

run: setup
	.venv/bin/python run.py $(ARGS)
