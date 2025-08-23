"""
Structured Logging System for DCA/Martingale Optimization
Provides JSON logging with event constants, context management, and traceability.
"""
from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime
from typing import Dict, Any, Optional, List
import threading
from pathlib import Path


# Event Constants
class Events:
    """Standardized event names for structured logging."""
    # Application lifecycle
    APP_START = "APP.START"
    APP_STOP = "APP.STOP"
    
    # Configuration
    BUILD_CONFIG = "BUILD.CONFIG"
    
    # Orchestrator events
    ORCH_START = "ORCH.START"
    ORCH_BATCH = "ORCH.BATCH"
    ORCH_SAVE_OK = "ORCH.SAVE_OK"
    ORCH_PRUNE = "ORCH.PRUNE"
    ORCH_EARLY_STOP = "ORCH.EARLY_STOP"
    ORCH_DONE = "ORCH.DONE"
    ORCH_ERROR = "ORCH.ERROR"
    
    # Evaluation events
    EVAL_CALL = "EVAL.CALL"
    EVAL_RETURN = "EVAL.RETURN"
    EVAL_ERROR = "EVAL.ERROR"
    
    # Database events
    DB_INIT = "DB.INIT"
    DB_UPSERT_EXP = "DB.UPSERT_EXP"
    DB_UPSERT_RES = "DB.UPSERT_RES"
    DB_VERIFY = "DB.VERIFY"
    DB_ERROR = "DB.ERROR"
    
    # UI events
    UI_CLICK_START = "UI.CLICK_START"
    UI_CLICK_STOP = "UI.CLICK_STOP"
    UI_RESULTS_LOAD = "UI.RESULTS_LOAD"
    UI_EXPORT = "UI.EXPORT"


class LogContext:
    """Thread-local context for logging."""
    _local = threading.local()
    
    @classmethod
    def set_run_id(cls, run_id: str):
        """Set run ID for current thread."""
        cls._local.run_id = run_id
    
    @classmethod
    def get_run_id(cls) -> Optional[str]:
        """Get run ID for current thread."""
        return getattr(cls._local, 'run_id', None)
    
    @classmethod
    def set_exp_id(cls, exp_id: int):
        """Set experiment ID for current thread."""
        cls._local.exp_id = exp_id
    
    @classmethod
    def get_exp_id(cls) -> Optional[int]:
        """Get experiment ID for current thread."""
        return getattr(cls._local, 'exp_id', None)
    
    @classmethod
    def set_batch_idx(cls, batch_idx: int):
        """Set batch index for current thread."""
        cls._local.batch_idx = batch_idx
    
    @classmethod
    def get_batch_idx(cls) -> Optional[int]:
        """Get batch index for current thread."""
        return getattr(cls._local, 'batch_idx', None)
    
    @classmethod
    def clear(cls):
        """Clear all context."""
        for attr in ['run_id', 'exp_id', 'batch_idx']:
            if hasattr(cls._local, attr):
                delattr(cls._local, attr)


class JSONFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging."""
    
    def __init__(self, include_extra_fields: bool = True):
        """Initialize JSON formatter."""
        super().__init__()
        self.include_extra_fields = include_extra_fields
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        # Base log structure
        log_entry = {
            "ts": datetime.fromtimestamp(record.created).isoformat(),
            "lvl": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        
        # Add context if available
        run_id = LogContext.get_run_id()
        if run_id:
            log_entry["run_id"] = run_id
        
        exp_id = LogContext.get_exp_id()
        if exp_id is not None:
            log_entry["exp_id"] = exp_id
        
        batch_idx = LogContext.get_batch_idx()
        if batch_idx is not None:
            log_entry["batch_idx"] = batch_idx
            log_entry["span_id"] = f"batch-{batch_idx}"
        
        # Add extra fields from record
        if self.include_extra_fields:
            extra_fields = [
                'event', 'adapter', 'overlap', 'orders', 'score', 'duration_ms',
                'eval_count', 'rows', 'best_score', 'total_evals', 'elapsed_ms',
                'pruned', 'saved_rows', 'evals_ok', 'evals_failed'
            ]
            
            for field in extra_fields:
                if hasattr(record, field):
                    log_entry[field] = getattr(record, field)
        
        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        
        return json.dumps(log_entry, separators=(',', ':'), ensure_ascii=False)


class StructuredLogger:
    """Structured logger with event support."""
    
    def __init__(self, name: str):
        """Initialize structured logger."""
        self.logger = logging.getLogger(name)
        self._setup_json_handler()
    
    def _setup_json_handler(self):
        """Setup JSON handler if not already present."""
        # Check if JSON handler already exists
        has_json_handler = any(
            isinstance(handler.formatter, JSONFormatter) 
            for handler in self.logger.handlers
        )
        
        if not has_json_handler:
            # Create JSON handler
            json_handler = logging.StreamHandler()
            json_handler.setFormatter(JSONFormatter())
            self.logger.addHandler(json_handler)
            
            # Set level based on environment
            debug_mode = os.getenv('MLAB_DEBUG', '0') == '1'
            self.logger.setLevel(logging.DEBUG if debug_mode else logging.INFO)
    
    def event(self, event_name: str, **kwargs):
        """Log an event with structured data."""
        extra_data = {'event': event_name}
        extra_data.update(kwargs)
        
        # Create log record with extra fields
        record = self.logger.makeRecord(
            self.logger.name,
            logging.INFO,
            __file__,
            0,
            f"Event: {event_name}",
            (),
            None,
            extra=extra_data
        )
        
        # Add extra fields as attributes
        for key, value in extra_data.items():
            setattr(record, key, value)
        
        self.logger.handle(record)
    
    def info(self, msg: str, **kwargs):
        """Log info message with optional extra fields."""
        extra_data = kwargs
        record = self.logger.makeRecord(
            self.logger.name,
            logging.INFO,
            __file__,
            0,
            msg,
            (),
            None,
            extra=extra_data
        )
        
        for key, value in extra_data.items():
            setattr(record, key, value)
        
        self.logger.handle(record)
    
    def error(self, msg: str, exc_info: bool = False, **kwargs):
        """Log error message with optional exception info."""
        extra_data = kwargs
        record = self.logger.makeRecord(
            self.logger.name,
            logging.ERROR,
            __file__,
            0,
            msg,
            (),
            exc_info if exc_info else None,
            extra=extra_data
        )
        
        for key, value in extra_data.items():
            setattr(record, key, value)
        
        self.logger.handle(record)
    
    def debug(self, msg: str, **kwargs):
        """Log debug message with optional extra fields."""
        extra_data = kwargs
        record = self.logger.makeRecord(
            self.logger.name,
            logging.DEBUG,
            __file__,
            0,
            msg,
            (),
            None,
            extra=extra_data
        )
        
        for key, value in extra_data.items():
            setattr(record, key, value)
        
        self.logger.handle(record)


def generate_run_id() -> str:
    """Generate unique run ID: YYYYMMDD-HHMMSS-<6hex>"""
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d-%H%M%S")
    hex_suffix = hex(int(time.time() * 1000000) % 0xFFFFFF)[2:].upper().zfill(6)
    return f"{timestamp}-{hex_suffix}"


def create_crash_snapshot(run_id: str, payload: Dict[str, Any], error_msg: str):
    """Create crash snapshot for debugging."""
    # Ensure crash_snapshots directory exists
    crash_dir = Path("db_results/crash_snapshots")
    crash_dir.mkdir(parents=True, exist_ok=True)
    
    # Create snapshot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{run_id}_{timestamp}.json"
    filepath = crash_dir / filename
    
    snapshot = {
        "run_id": run_id,
        "timestamp": timestamp,
        "error": error_msg,
        "payload": payload
    }
    
    try:
        with open(filepath, 'w') as f:
            json.dump(snapshot, f, indent=2, ensure_ascii=False)
        return str(filepath)
    except Exception as e:
        # If we can't write crash snapshot, at least log it
        logging.getLogger(__name__).error(f"Failed to write crash snapshot: {e}")
        return None


# Global logger instances
def get_structured_logger(name: str) -> StructuredLogger:
    """Get or create structured logger."""
    return StructuredLogger(name)


# Pre-configured loggers for different components
app_logger = get_structured_logger("mlab.app")
orch_logger = get_structured_logger("mlab.orchestrator")
eval_logger = get_structured_logger("mlab.evaluation")
db_logger = get_structured_logger("mlab.storage")
ui_logger = get_structured_logger("mlab.ui")


def setup_structured_logging():
    """Setup structured logging for the entire application."""
    # Configure root logger
    root_logger = logging.getLogger("mlab")
    
    # Remove existing handlers to avoid duplicates
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Add JSON handler
    json_handler = logging.StreamHandler()
    json_handler.setFormatter(JSONFormatter())
    root_logger.addHandler(json_handler)
    
    # Set level based on environment
    debug_mode = os.getenv('MLAB_DEBUG', '0') == '1'
    root_logger.setLevel(logging.DEBUG if debug_mode else logging.INFO)
    
    # Prevent propagation to avoid duplicate logs
    root_logger.propagate = False
    
    return root_logger


# Initialize on import
setup_structured_logging()
