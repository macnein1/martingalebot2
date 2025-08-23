# Martingale Lab CLI Makefile
# Commands for running optimization, linting, and cleanup

PYTHON := python3
MODULE := martingale_lab.cli.optimize
DB_PATH := db_results/experiments.db
LOG_DIR := logs

.PHONY: run lint clean-ui clean-db test help logs-dir

# Default target
help:
	@echo "Martingale Lab CLI Commands:"
	@echo "  make run        - Run optimization with production logging"
	@echo "  make run-dev    - Run with development logging (detailed file logs)"
	@echo "  make run-debug  - Run with debug logging and evaluation sampling"
	@echo "  make test       - Run quick smoke test with minimal logging"
	@echo "  make lint       - Run code linting (ruff + mypy)"
	@echo "  make clean-ui   - Clean up any remaining UI artifacts"
	@echo "  make clean-db   - Clean database and results"
	@echo "  make clean-logs - Clean log files"
	@echo "  make logs-dir   - Create logs directory"
	@echo "  make help       - Show this help message"

# Create logs directory
logs-dir:
	@mkdir -p $(LOG_DIR)

# Production run with enterprise logging - clean console, detailed file logs
run: logs-dir
	$(PYTHON) -m $(MODULE) \
		--overlap-min 15 --overlap-max 30 \
		--orders-min 5 --orders-max 20 \
		--batches 100 --batch-size 3000 --workers 4 \
		--disable-early-stop \
		--prune-threshold 2.0 --grace-batches 5 --patience 20 \
		--db $(DB_PATH) --seed 42 --notes "production run" \
		--log-level INFO \
		--log-file $(LOG_DIR)/run-$(shell date +%Y%m%d-%H%M%S).json \
		--log-eval-sample 0.0 \
		--log-every-batch 1

# Development run with more detailed logging
run-dev: logs-dir
	$(PYTHON) -m $(MODULE) \
		--overlap-min 15 --overlap-max 25 \
		--orders-min 5 --orders-max 15 \
		--batches 50 --batch-size 2000 --workers 4 \
		--prune-threshold 2.0 --grace-batches 3 --patience 15 \
		--db $(DB_PATH) --seed 42 --notes "development run" \
		--log-level DEBUG \
		--log-file $(LOG_DIR)/dev-$(shell date +%Y%m%d-%H%M%S).json \
		--log-eval-sample 0.001 \
		--log-every-batch 1

# Debug run with evaluation sampling and shorter batches
run-debug: logs-dir
	$(PYTHON) -m $(MODULE) \
		--overlap-min 15 --overlap-max 25 \
		--orders-min 5 --orders-max 10 \
		--batches 10 --batch-size 500 --workers 2 \
		--prune-threshold 3.0 --grace-batches 2 --patience 5 \
		--db $(DB_PATH) --seed 42 --notes "debug run with sampling" \
		--log-level DEBUG \
		--log-file $(LOG_DIR)/debug-$(shell date +%Y%m%d-%H%M%S).json \
		--log-eval-sample 0.05 \
		--log-every-batch 1 \
		--max-time-sec 300

# Quick test with minimal logging
test: logs-dir
	$(PYTHON) -m $(MODULE) \
		--overlap-min 15 --overlap-max 25 \
		--orders-min 5 --orders-max 10 \
		--batches 5 --batch-size 100 --workers 2 \
		--prune-threshold 3.0 --grace-batches 2 --patience 5 \
		--db $(DB_PATH) --seed 42 --notes "quick test" \
		--log-level INFO \
		--log-eval-sample 0.0 \
		--log-every-batch 5

# Test with timeout constraint
test-timeout: logs-dir
	$(PYTHON) -m $(MODULE) \
		--overlap-min 15 --overlap-max 20 \
		--orders-min 5 --orders-max 8 \
		--batches 100 --batch-size 1000 --workers 2 \
		--db $(DB_PATH) --seed 42 --notes "timeout test" \
		--log-level INFO \
		--log-file $(LOG_DIR)/timeout-test-$(shell date +%Y%m%d-%H%M%S).json \
		--max-time-sec 60

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

# Clean log files
clean-logs:
	@echo "Cleaning log files..."
	-rm -rf $(LOG_DIR)/
	@echo "Log cleanup complete"

# Clean everything
clean-all: clean-ui clean-db clean-logs
	@echo "Full cleanup complete"

# Install dependencies
install:
	pip install -r requirements.txt

# Check if CLI module can be imported
check:
	$(PYTHON) -c "from martingale_lab.cli.optimize import main; print('CLI module imports successfully')"

# Show recent log files
logs:
	@echo "Recent log files:"
	@if [ -d "$(LOG_DIR)" ]; then ls -la $(LOG_DIR)/ | tail -10; else echo "No logs directory found. Run 'make logs-dir' first."; fi

# Tail latest log file
tail-log:
	@if [ -d "$(LOG_DIR)" ]; then \
		LATEST=$$(ls -t $(LOG_DIR)/*.json 2>/dev/null | head -1); \
		if [ -n "$$LATEST" ]; then \
			echo "Tailing: $$LATEST"; \
			tail -f "$$LATEST"; \
		else \
			echo "No log files found in $(LOG_DIR)"; \
		fi; \
	else \
		echo "No logs directory found"; \
	fi
