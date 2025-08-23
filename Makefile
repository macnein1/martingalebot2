# Martingale Lab CLI Makefile
# Commands for running optimization, linting, and cleanup

PYTHON := python3
MODULE := martingale_lab.cli.optimize
DB_PATH := db_results/experiments.db

.PHONY: run lint clean-ui clean-db test help

# Default target
help:
	@echo "Martingale Lab CLI Commands:"
	@echo "  make run        - Run optimization with example parameters"
	@echo "  make lint       - Run code linting (ruff + mypy)"
	@echo "  make clean-ui   - Clean up any remaining UI artifacts"
	@echo "  make clean-db   - Clean database and results"
	@echo "  make test       - Run quick smoke test"
	@echo "  make help       - Show this help message"

# Run optimization with example parameters
run:
	$(PYTHON) -m $(MODULE) \
		--overlap-min 15 --overlap-max 30 \
		--orders-min 5 --orders-max 20 \
		--batches 100 --batch-size 3000 --workers 4 \
		--disable-early-stop \
		--prune-threshold 2.0 --grace-batches 5 --patience 20 \
		--db $(DB_PATH) --seed 42 --notes "no-ui run"

# Run with smaller parameters for quick testing
test:
	$(PYTHON) -m $(MODULE) \
		--overlap-min 15 --overlap-max 25 \
		--orders-min 5 --orders-max 10 \
		--batches 5 --batch-size 100 --workers 2 \
		--prune-threshold 3.0 --grace-batches 2 --patience 5 \
		--db $(DB_PATH) --seed 42 --notes "quick test"

# Run linting tools
lint:
	@echo "Running ruff..."
	-ruff check . --fix
	@echo "Running mypy..."
	-mypy martingale_lab --ignore-missing-imports --no-strict-optional

# Clean up UI artifacts (should be none after refactoring)
clean-ui:
	@echo "Cleaning up UI artifacts..."
	-rm -rf ui/ ui_bridge/ .streamlit/ pages/
	-rm -f main.py streamlit_app.py
	-find . -name "*.pyc" -delete
	-find . -name "__pycache__" -type d -exec rm -rf {} +
	@echo "UI cleanup complete"

# Clean database and results
clean-db:
	@echo "Cleaning database and results..."
	-rm -rf db_results/
	@echo "Database cleanup complete"

# Install dependencies
install:
	pip install -r requirements.txt

# Check if CLI module can be imported
check:
	$(PYTHON) -c "from martingale_lab.cli.optimize import main; print('CLI module imports successfully')"