format:
	ruff format

lint:
	ruff check

pre-commit: format lint type-check test

test:
	pytest

type-check:
	mypy main.py
