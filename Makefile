.PHONY: install test lint typecheck verify backtest snooping_check

install:
	pip install -r requirements.txt -r requirements-dev.txt
	pre-commit install

test:
	rtk pytest tests/ -v --tb=short

lint:
	ruff check app/ tests/ scripts/

typecheck:
	mypy app/

snooping_check:
	python scripts/verify_no_snooping.py

verify: lint typecheck test snooping_check
	@echo "✅ All quality gates passed."

backtest:
	@echo "Lance manuellement un script run_*.py spécifique."
