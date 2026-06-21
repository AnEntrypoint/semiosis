.PHONY: install install-full lint format type test test-full build ci clean

install:
	pip install -e ".[dev]"

install-full:
	pip install torch --index-url https://download.pytorch.org/whl/cpu
	pip install -e ".[hyperbolic,encoder,serving,eval,dev]"

lint:
	ruff check .

format:
	ruff format .

type:
	mypy core

test:
	pytest core/ -q

test-full: install-full test

build:
	python -m build
	twine check dist/*

ci: lint type test
	ruff format --check .

clean:
	rm -rf dist build *.egg-info .mypy_cache .pytest_cache .ruff_cache
