SHELL=/bin/bash
C_RED = $(shell echo -e "\\033[0;31m")
C_RESET = $(shell echo -e "\\033[0m")

.PHONY: build venv nv-venv clean clean-test clean-pyc clean-build docs help
.DEFAULT_GOAL := help

define BROWSER_PYSCRIPT
import os, webbrowser, sys

from urllib.request import pathname2url

webbrowser.open("file://" + pathname2url(os.path.abspath(sys.argv[1])))
endef
export BROWSER_PYSCRIPT

define PRINT_HELP_PYSCRIPT
import re, sys

for line in sys.stdin:
	match = re.match(r'^([a-zA-Z_-]+):.*?## (.*)$$', line)
	if match:
		target, help = match.groups()
		print("%-20s %s" % (target, help))
endef
export PRINT_HELP_PYSCRIPT

BROWSER := python -c "$$BROWSER_PYSCRIPT"

help:
	@python -c "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)

clean: clean-env clean-build clean-pyc clean-test ## remove all build, test, coverage, and Python artifacts

clean-env: ## remove env artifacts
	rm -fr env/
	rm -fr venv/

clean-build: ## remove build artifacts
	rm -fr build/
	rm -fr dist/
	rm -fr .eggs/
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -f {} +

clean-pyc: ## remove Python file artifacts
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

clean-test: ## remove test and coverage artifacts
	rm -fr .tox/
	rm -f .coverage
	rm -fr htmlcov/
	rm -fr .pytest_cache

clean-logs: ## remove *.csv, *.log and *.json artifacts
	rm -fr *.csv
	rm -fr *.json
	rm -fr *.log

lint: ## check style with flake8
	flake8 genre tests

test: ## run tests quickly with the default Python
	python setup.py test

test-all: ## run tests on every Python version with tox
	tox

coverage: ## check code coverage quickly with the default Python
	coverage run --source genre setup.py test
	coverage report -m
	coverage html
	$(BROWSER) htmlcov/index.html

docs: ## generate Sphinx HTML documentation, including API docs
	rm -f docs/modelscope.rst
	rm -f docs/modules.rst
	sphinx-apidoc -o docs/ modelscope
	$(MAKE) -C docs clean
	$(MAKE) -C docs html
	$(BROWSER) docs/_build/html/index.html

servedocs: docs ## compile the docs watching for changes
	watchmedo shell-command -p '*.rst' -c '$(MAKE) -C docs html' -R -D .

build:  ## builds source and wheel package
	python -m build
	ls -l dist

venv: clean ## create and initialize venv environment
	@python3 -m venv ./venv
	source venv/bin/activate; \
	pip install --no-input --upgrade pip; \
	pip install --no-input -r requirements.txt; \
	pip install --no-input -e .; \
	./setenv.sh
	@echo
	@echo "********************* Activate your environment ***********************"
	@echo
	@echo "Please activate your environment with: "
	@echo "$(C_RED)source venv/bin/activate$(C_RESET)"
	@echo
	@echo "***********************************************************************"

nv-venv: clean ## create and initialize venv environment
	@python3 -m venv ./venv
	source venv/bin/activate; \
	pip install --no-input --upgrade pip; \
	pip install --no-input -r requirements.txt; \
	pip install --no-input -r requirements_dev.txt; \
	pip install --no-input -e .
	@echo
	@echo "********************* Activate your environment ***********************"
	@echo
	@echo "Please activate your environment with: "
	@echo "$(C_RED)source venv/bin/activate$(C_RESET)"
	@echo
	@echo "***********************************************************************"
