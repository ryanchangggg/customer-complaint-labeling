.PHONY: run test lint format clean install

# --- Run ---

run:
	python3 -m src.main

run-custom:
	python3 -m src.main --input $(INPUT) --output $(OUTPUT)

# --- Testing ---

test:
	python3 -m pytest tests/ -v --cov=src --cov-report=term

test-quiet:
	python3 -m pytest tests/ --tb=short -q

# --- Linting & Formatting ---

lint:
	ruff check src/ tests/

format:
	ruff format src/ tests/

check: lint format-check

format-check:
	ruff format --check src/ tests/

# --- Type Checking ---

typecheck:
	python3 -m mypy src/ --ignore-missing-imports

# --- Cleaning ---

clean:
	rm -rf output/*.csv output/*.json output/*.txt output/report.txt
	rm -rf logs/*.log
	rm -rf .pytest_cache __pycache__
	rm -rf .mypy_cache
	rm -rf .coverage *.coverage coverage.xml htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete

# --- Installation ---

install:
	pip install -r requirements.txt
	pip install ruff mypy

install-dev: install
	pip install pytest pytest-cov

# --- Info ---

tree:
	find . -not -path './.git/*' -not -path './node_modules/*' \
	       -not -path './__pycache__/*' -not -path '*/__pycache__/*' \
	       -not -path './.pytest_cache/*' -not -path './.mypy_cache/*' \
	       -not -name '*.pyc' -not -name '.DS_Store' \
	       | sort | head -60
