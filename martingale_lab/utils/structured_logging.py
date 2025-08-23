"""
Structured JSON Logging Infrastructure for Martingale Lab

Provides:
- JSONFormatter for automatic JSON formatting of all log messages
- Standardized event names as constants
- Common fields and utilities for consistent logging
- Integration with existing RingBuffer system
"""

import json
import logging
import time
from datetime import datetime
from typing import Dict, Any, Optional
import uuid


class JSONFormatter(logging.Formatter):
    """Custom formatter that converts all log records to JSON format"""
    
    def __init__(self, include_extra=True):
        super().__init__()
        self.include_extra = include_extra
    
    def format(self, record: logging.LogRecord) -> str:
        # Base log entry
        log_entry = {
            "ts": datetime.fromtimestamp(record.created).isoformat(),
            "lvl": record.levelname,
            "msg": record.getMessage(),
            "module": record.module,
            "func": record.funcName,
            "line": record.lineno
        }
        
        # Add extra fields if they exist
        if self.include_extra and hasattr(record, 'extra_fields'):
            log_entry.update(record.extra_fields)
        
        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        
        return json.dumps(log_entry, default=str)


class EventNames:
    """Standardized event names for consistent logging"""
    
    # Application lifecycle
    APP_START = "APP.START"
    APP_STOP = "APP.STOP"
    
    # Configuration
    BUILD_CONFIG = "BUILD.CONFIG"
    
    # Orchestrator events
    ORCH_START = "ORCH.START"
    ORCH_BATCH = "ORCH.BATCH"
    ORCH_PRUNE = "ORCH.PRUNE"
    ORCH_SAVE_OK = "ORCH.SAVE_OK"
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


def generate_run_id() -> str:
    """Generate a run_id in format YYYYMMDD-HHMMSS-<6hex>"""
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d-%H%M%S")
    hex_suffix = uuid.uuid4().hex[:6].upper()
    return f"{timestamp}-{hex_suffix}"


def generate_span_id(batch_idx: int) -> str:
    """Generate a span_id for a batch"""
    return f"batch-{batch_idx:04d}"


class StructuredLogger:
    """Enhanced logger with structured JSON logging capabilities"""
    
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self.name = name
    
    def _log_with_context(self, level: int, event: str, msg: str = "", 
                         run_id: Optional[str] = None, exp_id: Optional[int] = None,
                         span_id: Optional[str] = None, **kwargs):
        """Log with structured context"""
        extra_fields = {
            "event": event,
            "adapter": kwargs.get("adapter", ""),
            "run_id": run_id or "",
            "exp_id": exp_id or 0,
            "span_id": span_id or "",
            "batch_idx": kwargs.get("batch_idx", 0),
            "eval_count": kwargs.get("eval_count", 0),
            "overlap": kwargs.get("overlap", 0.0),
            "orders": kwargs.get("orders", 0),
            "score": kwargs.get("score", 0.0),
            "duration_ms": kwargs.get("duration_ms", 0.0),
        }
        
        # Add any additional kwargs
        for key, value in kwargs.items():
            if key not in extra_fields:
                extra_fields[key] = value
        
        # Create log record with extra fields
        record = self.logger.makeRecord(
            self.logger.name, level, "", 0, msg, (), None
        )
        record.extra_fields = extra_fields
        
        self.logger.handle(record)
    
    def info(self, event: str, msg: str = "", **kwargs):
        self._log_with_context(logging.INFO, event, msg, **kwargs)
    
    def error(self, event: str, msg: str = "", **kwargs):
        self._log_with_context(logging.ERROR, event, msg, **kwargs)
    
    def warning(self, event: str, msg: str = "", **kwargs):
        self._log_with_context(logging.WARNING, event, msg, **kwargs)
    
    def debug(self, event: str, msg: str = "", **kwargs):
        self._log_with_context(logging.DEBUG, event, msg, **kwargs)


def setup_structured_logging(logger_name: str = "mlab", 
                           use_json_formatter: bool = True,
                           level: int = logging.INFO) -> StructuredLogger:
    """Setup structured logging with JSON formatter and ring buffer"""
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    
    # Remove existing handlers to avoid duplicates
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Add console handler with JSON formatter
    if use_json_formatter:
        handler = logging.StreamHandler()
        handler.setFormatter(JSONFormatter())
        logger.addHandler(handler)
    
    # Add ring buffer handler for live trace
    from ui.utils.logging_buffer import ensure_ring_handler
    ring_handler = ensure_ring_handler(logger_name, level=level, use_json=True)
    
    # Return structured logger wrapper
    return StructuredLogger(logger_name)


def get_structured_logger(name: str = "mlab") -> StructuredLogger:
    """Get or create a structured logger instance"""
    return StructuredLogger(name)


class Timer:
    """Context manager for timing operations"""
    
    def __init__(self):
        self.start_time = None
        self.duration_ms = 0.0
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            self.duration_ms = (time.time() - self.start_time) * 1000.0


def ensure_json_serializable(obj: Any) -> Any:
    """Ensure an object is JSON serializable, converting numpy arrays to lists"""
    import numpy as np
    
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: ensure_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [ensure_json_serializable(item) for item in obj]
    else:
        return obj
