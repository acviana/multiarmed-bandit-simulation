format:
	ruff format

lint:
	ruff check

pre-commit: format lint type-check

type-check:
	mypy main.py
