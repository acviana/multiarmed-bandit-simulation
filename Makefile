format:
	ruff format

lint:
	ruff check

pre-commit: format lint type-check test

test:
	pytest test_main.py

type-check:
	mypy main.py
