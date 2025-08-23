"""
Constants and configuration for DCA/Martingale UI
"""
import os
from pathlib import Path

# Database path - single source of truth
DB_PATH = "db_results/experiments.db"

# Ensure db_results directory exists
DB_DIR = Path("db_results")
DB_DIR.mkdir(exist_ok=True)

# Crash snapshots directory
CRASH_SNAPSHOTS_DIR = DB_DIR / "crash_snapshots"
CRASH_SNAPSHOTS_DIR.mkdir(exist_ok=True)

# Logging configuration
RING_BUFFER_SIZE = int(os.getenv('MLAB_TRACE_N', '1000'))
DEBUG_MODE = os.getenv('MLAB_DEBUG', '0') == '1'

# UI Configuration
DEFAULT_TOP_K = 50
DEFAULT_BATCH_SIZE = 1000
DEFAULT_MAX_BATCHES = 100

# Performance thresholds
MIN_EVALS_PER_SECOND = 10
MAX_BATCH_TIME_MS = 30000  # 30 seconds

# Optimization defaults
DEFAULT_CONFIG = {
    "overlap_min": 10.0,
    "overlap_max": 30.0,
    "orders_min": 5,
    "orders_max": 15,
    "alpha": 0.5,
    "beta": 0.3,
    "gamma": 0.2,
    "lambda_penalty": 0.1,
    "wave_pattern": True,
    "tail_cap": 0.40,
    "min_indent_step": 0.05,
    "softmax_temp": 1.0
}

# Event categories for filtering
EVENT_CATEGORIES = {
    "APP": ["APP.START", "APP.STOP"],
    "CONFIG": ["BUILD.CONFIG"],
    "ORCH": ["ORCH.START", "ORCH.BATCH", "ORCH.SAVE_OK", "ORCH.PRUNE", "ORCH.EARLY_STOP", "ORCH.DONE", "ORCH.ERROR"],
    "EVAL": ["EVAL.CALL", "EVAL.RETURN", "EVAL.ERROR"],
    "DB": ["DB.INIT", "DB.UPSERT_EXP", "DB.UPSERT_RES", "DB.VERIFY", "DB.ERROR"],
    "UI": ["UI.CLICK_START", "UI.CLICK_STOP", "UI.RESULTS_LOAD", "UI.EXPORT"]
}

# Status constants
class Status:
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    STOPPED = "STOPPED"


