.PHONY: install test lint typecheck verify backtest snooping_check pipeline_check pipeline_check_extended

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

pipeline_check:
	rtk pytest tests/integration/test_pipeline_integrity.py -v

pipeline_check_extended:
	rtk pytest tests/integration/test_pipeline_integrity_extended.py -v

verify: lint typecheck test snooping_check pipeline_check pipeline_check_extended
	@echo "✅ All quality gates passed (including pipeline integrity extended)."

backtest:
	@echo "Lance manuellement un script run_*.py spécifique."
