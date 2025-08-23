"""
Enterprise-level centralized logging system for Martingale Lab CLI operations.
Provides modular, configurable logging with file/console separation and sampling support.
"""
import json
import logging
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured file logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "ts": datetime.utcnow().isoformat() + "Z",
            "lvl": record.levelname,
            "logger": record.name,
            "msg": record.getMessage()
        }
        
        # Add extra fields if present
        extra_fields = ["run_id", "exp_id", "batch_idx", "span_id", "event"]
        for field in extra_fields:
            if hasattr(record, field) and getattr(record, field) is not None:
                log_entry[field] = getattr(record, field)
        
        # Add any other custom fields from extra dict
        if hasattr(record, '__dict__'):
            for key, value in record.__dict__.items():
                if (key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname', 
                               'filename', 'module', 'lineno', 'funcName', 'created', 'msecs', 
                               'relativeCreated', 'thread', 'threadName', 'processName', 'process', 
                               'getMessage', 'exc_info', 'exc_text', 'stack_info', 'taskName'] 
                    and not key.startswith('_') 
                    and key not in extra_fields):
                    if isinstance(value, (str, int, float, bool, type(None))):
                        log_entry[key] = value
                    elif isinstance(value, (list, dict)):
                        log_entry[key] = value
                    else:
                        log_entry[key] = str(value)
        
        return json.dumps(log_entry, separators=(',', ':'))


class HumanFormatter(logging.Formatter):
    """Human-readable formatter for console output."""
    
    def __init__(self):
        super().__init__("[%(asctime)s] %(levelname)s %(name)s: %(message)s", "%H:%M:%S")


def configure_logging(
    level: str = "INFO",
    *,
    json_console: bool = False,
    log_file: Optional[str] = None
) -> logging.Logger:
    """
    Configure enterprise-level centralized logging system.
    
    Args:
        level: Log level for console output (DEBUG, INFO, WARNING, ERROR)
        json_console: If True, use JSON format for console; if False, use human-readable
        log_file: Optional file path for detailed JSON logging
    
    Returns:
        Root logger instance
    """
    # Clear any existing handlers
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(getattr(logging, level.upper()))
    
    # Console handler - brief output for human readability
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(root.level)
    
    if json_console:
        console_formatter = JSONFormatter()
    else:
        console_formatter = HumanFormatter()
    
    console_handler.setFormatter(console_formatter)
    root.addHandler(console_handler)
    
    # File handler - detailed JSON output for analysis
    if log_file:
        # Ensure log directory exists
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)  # File gets everything
        file_handler.setFormatter(JSONFormatter())
        root.addHandler(file_handler)
    
    # Configure logger hierarchy for noise reduction
    _configure_logger_hierarchy()
    
    return root


def _configure_logger_hierarchy():
    """Configure logger hierarchy to reduce noise and control verbosity."""
    # Per-eval logs: Default to WARNING (will be set to DEBUG with sampling)
    logging.getLogger("mlab.eval").setLevel(logging.WARNING)
    
    # Other subsystem defaults
    logging.getLogger("mlab.db").setLevel(logging.INFO)
    logging.getLogger("mlab.orchestrator").setLevel(logging.INFO)
    logging.getLogger("mlab.cli").setLevel(logging.INFO)
    logging.getLogger("mlab.optimizer").setLevel(logging.INFO)


class EvaluationSampler:
    """Handles sampling for per-evaluation logging to prevent log overflow."""
    
    def __init__(self, sample_rate: float = 0.0):
        """
        Initialize evaluation sampler.
        
        Args:
            sample_rate: Float between 0.0 and 1.0 for sampling percentage
        """
        self.sample_rate = max(0.0, min(1.0, sample_rate))
        self._random = random.Random()  # Use separate random instance for deterministic behavior
    
    def should_log(self) -> bool:
        """Determine if this evaluation should be logged based on sample rate."""
        if self.sample_rate <= 0.0:
            return False
        return self._random.random() < self.sample_rate
    
    def set_seed(self, seed: Optional[int] = None):
        """Set seed for deterministic sampling behavior."""
        if seed is not None:
            self._random.seed(seed)


class BatchAggregator:
    """Handles batch-level logging aggregation and summary generation."""
    
    @staticmethod
    def log_batch_summary(
        batch_idx: int,
        total_batches: int,
        best_score: float,
        evaluations: int,
        candidates_kept: int,
        prune_mode: str,
        evaluations_per_second: float,
        log_every_batch: int = 1,
        logger: Optional[logging.Logger] = None,
        kept_in_this_batch: Optional[int] = None,
        kept_total: Optional[int] = None
    ):
        """
        Log a concise batch summary with both batch-specific and cumulative statistics.
        
        Args:
            batch_idx: Current batch index (0-based)
            total_batches: Total number of batches
            best_score: Best score so far
            evaluations: Number of evaluations in this batch
            candidates_kept: Number of candidates kept after pruning (deprecated, use kept_in_this_batch)
            prune_mode: Pruning mode description
            evaluations_per_second: Current evaluation speed
            log_every_batch: Log every N batches
            logger: Logger instance to use
            kept_in_this_batch: Number of candidates kept in this specific batch
            kept_total: Total number of candidates kept across all batches
        """
        if logger is None:
            logger = logging.getLogger("mlab.orchestrator")
        
        # Only log according to the frequency setting
        if (batch_idx + 1) % max(1, log_every_batch) != 0:
            return
        
        # Use new parameters if provided, otherwise fall back to old behavior
        if kept_in_this_batch is not None and kept_total is not None:
            # New single-line format with both batch and cumulative stats
            logger.info(
                "BATCH %d/%d | best=%.6f evals=%d kept_batch=%d kept_total=%d mode=%s speed=%.1f/s",
                batch_idx + 1, total_batches, best_score, evaluations, 
                kept_in_this_batch, kept_total, prune_mode, evaluations_per_second,
                extra={
                    "event": "BATCH_SUMMARY",
                    "batch_idx": batch_idx,
                    "total_batches": total_batches,
                    "best_score": best_score,
                    "evaluations": evaluations,
                    "kept_in_this_batch": kept_in_this_batch,
                    "kept_total": kept_total,
                    "prune_mode": prune_mode,
                    "evaluations_per_second": evaluations_per_second
                }
            )
        else:
            # Legacy format for backward compatibility
            logger.info(
                "BATCH %d/%d | best=%.6f evals=%d kept=%d mode=%s speed=%.1f/s",
                batch_idx + 1, total_batches, best_score, evaluations, 
                candidates_kept, prune_mode, evaluations_per_second,
                extra={
                    "event": "BATCH_SUMMARY",
                    "batch_idx": batch_idx,
                    "total_batches": total_batches,
                    "best_score": best_score,
                    "evaluations": evaluations,
                    "candidates_kept": candidates_kept,
                    "prune_mode": prune_mode,
                    "evaluations_per_second": evaluations_per_second
                }
            )


def log_with_context(
    logger: logging.Logger, 
    level: str, 
    message: str, 
    run_id: Optional[str] = None, 
    exp_id: Optional[int] = None, 
    batch_idx: Optional[int] = None, 
    event: Optional[str] = None, 
    **kwargs
):
    """
    Log a message with contextual information.
    
    Args:
        logger: Logger instance
        level: Log level (debug, info, warning, error)
        message: Log message
        run_id: Optional run identifier
        exp_id: Optional experiment identifier
        batch_idx: Optional batch index
        event: Optional event type
        **kwargs: Additional context fields
    """
    extra = {}
    if run_id:
        extra['run_id'] = run_id
    if exp_id is not None:
        extra['exp_id'] = exp_id
    if batch_idx is not None:
        extra['batch_idx'] = batch_idx
    if event:
        extra['event'] = event
    
    # Add any additional fields
    extra.update(kwargs)
    
    level_func = getattr(logger, level.lower())
    level_func(message, extra=extra)


# Pre-configured loggers for different subsystems
def get_cli_logger() -> logging.Logger:
    """Get CLI subsystem logger."""
    return logging.getLogger("mlab.cli")


def get_orchestrator_logger() -> logging.Logger:
    """Get orchestrator subsystem logger."""
    return logging.getLogger("mlab.orchestrator")


def get_eval_logger() -> logging.Logger:
    """Get evaluation subsystem logger."""
    return logging.getLogger("mlab.eval")


def get_db_logger() -> logging.Logger:
    """Get database subsystem logger."""
    return logging.getLogger("mlab.db")


def get_optimizer_logger() -> logging.Logger:
    """Get optimizer subsystem logger."""
    return logging.getLogger("mlab.optimizer")


# Module-level instances for backward compatibility
cli_logger = get_cli_logger()
orchestrator_logger = get_orchestrator_logger()
eval_logger = get_eval_logger()
db_logger = get_db_logger()
optimizer_logger = get_optimizer_logger()


# Global sampler instance (will be configured by CLI)
_global_eval_sampler = EvaluationSampler()


def configure_eval_sampling(sample_rate: float = 0.0, seed: Optional[int] = None):
    """
    Configure global evaluation logging sampling.
    
    Args:
        sample_rate: Sampling rate between 0.0 and 1.0
        seed: Optional seed for deterministic sampling
    """
    global _global_eval_sampler
    _global_eval_sampler = EvaluationSampler(sample_rate)
    if seed is not None:
        _global_eval_sampler.set_seed(seed)
    
    # Enable DEBUG level for eval logger if sampling is enabled
    if sample_rate > 0.0:
        logging.getLogger("mlab.eval").setLevel(logging.DEBUG)
    else:
        logging.getLogger("mlab.eval").setLevel(logging.WARNING)


def should_log_eval() -> bool:
    """Check if current evaluation should be logged based on global sampling configuration."""
    return _global_eval_sampler.should_log()
