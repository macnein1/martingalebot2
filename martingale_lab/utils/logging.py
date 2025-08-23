"""
Clean JSONL logging system for CLI operations.
Fields: ts (iso), lvl, logger, msg, optional run_id, exp_id, batch_idx, span_id
"""
import json
import logging
import sys
from datetime import datetime
from typing import Optional


class JSONLFormatter(logging.Formatter):
    """JSON Lines formatter for structured logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "ts": datetime.utcnow().isoformat() + "Z",
            "lvl": record.levelname,
            "logger": record.name,
            "msg": record.getMessage()
        }
        
        # Add extra fields if present
        if hasattr(record, 'run_id') and record.run_id:
            log_entry["run_id"] = record.run_id
        if hasattr(record, 'exp_id') and record.exp_id:
            log_entry["exp_id"] = record.exp_id
        if hasattr(record, 'batch_idx') and record.batch_idx is not None:
            log_entry["batch_idx"] = record.batch_idx
        if hasattr(record, 'span_id') and record.span_id:
            log_entry["span_id"] = record.span_id
        if hasattr(record, 'event') and record.event:
            log_entry["event"] = record.event
            
        # Add any other extra fields
        if hasattr(record, '__dict__'):
            for key, value in record.__dict__.items():
                if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname', 'filename',
                               'module', 'lineno', 'funcName', 'created', 'msecs', 'relativeCreated',
                               'thread', 'threadName', 'processName', 'process', 'getMessage',
                               'run_id', 'exp_id', 'batch_idx', 'span_id', 'event', 'exc_info', 'exc_text',
                               'stack_info', 'taskName'] and not key.startswith('_'):
                    if isinstance(value, (str, int, float, bool, type(None))):
                        log_entry[key] = value
                    elif hasattr(value, '__dict__'):
                        continue  # Skip complex objects
                    else:
                        log_entry[key] = str(value)
        
        return json.dumps(log_entry, separators=(',', ':'))


def get_logger(name: str) -> logging.Logger:
    """Get a logger with JSONL formatting."""
    logger = logging.getLogger(name)
    
    # Only add handler if not already configured
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = JSONLFormatter()
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    
    return logger


def log_with_context(logger: logging.Logger, level: str, message: str, 
                    run_id: Optional[str] = None, exp_id: Optional[int] = None, 
                    batch_idx: Optional[int] = None, event: Optional[str] = None, **kwargs):
    """Log a message with context fields."""
    extra = {}
    if run_id:
        extra['run_id'] = run_id
    if exp_id:
        extra['exp_id'] = exp_id
    if batch_idx is not None:
        extra['batch_idx'] = batch_idx
    if event:
        extra['event'] = event
    
    # Add any additional fields
    extra.update(kwargs)
    
    level_func = getattr(logger, level.lower())
    level_func(message, extra=extra)


# Pre-configured loggers
cli_logger = get_logger("mlab.cli")
orchestrator_logger = get_logger("mlab.orchestrator") 
eval_logger = get_logger("mlab.eval")
db_logger = get_logger("mlab.db")
