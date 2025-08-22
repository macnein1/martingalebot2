"""
Structured logging module for martingale optimization.
Provides JSON logging with rotation and real-time UI integration.
"""
import logging
import json
import sys
import os
from pathlib import Path
from logging.handlers import RotatingFileHandler
from typing import Dict, Any, Optional
from datetime import datetime


def setup_logging(run_id: str, log_dir: str = "logs") -> logging.Logger:
    """
    Set up structured logging with file rotation and console output.
    
    Args:
        run_id: Unique run identifier
        log_dir: Directory for log files
        
    Returns:
        Configured logger instance
    """
    # Create log directory if it doesn't exist
    Path(log_dir).mkdir(exist_ok=True)
    
    # Create logger
    logger = logging.getLogger("mlab")
    logger.setLevel(logging.INFO)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Custom formatter for clean JSON output
    formatter = logging.Formatter('%(message)s')
    
    # File handler with rotation (10MB files, keep 3 backups)
    log_file = os.path.join(log_dir, f"{run_id}.log")
    file_handler = RotatingFileHandler(
        log_file, 
        maxBytes=10_000_000,  # 10MB
        backupCount=3,
        encoding='utf-8'
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Console handler for real-time monitoring
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger


def jlog(logger: logging.Logger, **kwargs: Any) -> None:
    """
    Log structured data as JSON.
    
    Args:
        logger: Logger instance
        **kwargs: Key-value pairs to log as JSON
    """
    # Add timestamp if not present
    if 'timestamp' not in kwargs:
        kwargs['timestamp'] = datetime.utcnow().isoformat()
    
    # Serialize to JSON with proper encoding
    log_entry = json.dumps(kwargs, ensure_ascii=False, default=str)
    logger.info(log_entry)


class LogContext:
    """Context manager for consistent logging with run context."""
    
    def __init__(self, logger: logging.Logger, run_id: str, 
                 batch_id: Optional[str] = None, candidate_id: Optional[str] = None):
        self.logger = logger
        self.base_context = {
            'run_id': run_id,
            'batch_id': batch_id,
            'candidate_id': candidate_id
        }
        # Remove None values
        self.base_context = {k: v for k, v in self.base_context.items() if v is not None}
    
    def log(self, event: str, **kwargs: Any) -> None:
        """Log event with base context."""
        log_data = {**self.base_context, 'event': event, **kwargs}
        jlog(self.logger, **log_data)
    
    def error(self, event: str, error: Exception, **kwargs: Any) -> None:
        """Log error with context."""
        log_data = {
            **self.base_context, 
            'event': event,
            'error': str(error),
            'error_type': type(error).__name__,
            **kwargs
        }
        jlog(self.logger, **log_data)
    
    def timing(self, event: str, duration_ms: float, **kwargs: Any) -> None:
        """Log timing information."""
        log_data = {
            **self.base_context,
            'event': event,
            'duration_ms': round(duration_ms, 3),
            **kwargs
        }
        jlog(self.logger, **log_data)


class LogReader:
    """Utility for reading and filtering log files for UI display."""
    
    @staticmethod
    def read_recent_logs(log_file: str, max_lines: int = 100) -> list:
        """
        Read recent log entries from file.
        
        Args:
            log_file: Path to log file
            max_lines: Maximum number of lines to return
            
        Returns:
            List of parsed log entries (newest first)
        """
        if not os.path.exists(log_file):
            return []
        
        entries = []
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                
            # Take last max_lines and reverse for newest first
            recent_lines = lines[-max_lines:] if len(lines) > max_lines else lines
            
            for line in reversed(recent_lines):
                line = line.strip()
                if line:
                    try:
                        entry = json.loads(line)
                        entries.append(entry)
                    except json.JSONDecodeError:
                        # Handle non-JSON lines gracefully
                        entries.append({'message': line, 'timestamp': 'unknown'})
                        
        except Exception as e:
            entries.append({'error': f'Failed to read log: {str(e)}'})
        
        return entries
    
    @staticmethod
    def filter_logs(entries: list, event_filter: Optional[str] = None, 
                   run_id_filter: Optional[str] = None) -> list:
        """
        Filter log entries by event type or run ID.
        
        Args:
            entries: List of log entries
            event_filter: Filter by event type (e.g., 'candidate_error', 'batch_end')
            run_id_filter: Filter by run ID
            
        Returns:
            Filtered list of entries
        """
        filtered = entries
        
        if event_filter:
            filtered = [e for e in filtered if e.get('event') == event_filter]
        
        if run_id_filter:
            filtered = [e for e in filtered if e.get('run_id') == run_id_filter]
        
        return filtered
    
    @staticmethod
    def get_log_summary(log_file: str) -> Dict[str, Any]:
        """
        Get summary statistics from log file.
        
        Args:
            log_file: Path to log file
            
        Returns:
            Dictionary with log statistics
        """
        entries = LogReader.read_recent_logs(log_file, max_lines=1000)
        
        if not entries:
            return {'total_entries': 0, 'events': {}, 'errors': 0}
        
        event_counts = {}
        error_count = 0
        
        for entry in entries:
            event = entry.get('event', 'unknown')
            event_counts[event] = event_counts.get(event, 0) + 1
            
            if 'error' in entry or event.endswith('_error'):
                error_count += 1
        
        return {
            'total_entries': len(entries),
            'events': event_counts,
            'errors': error_count,
            'latest_timestamp': entries[0].get('timestamp', 'unknown') if entries else None
        }


# Convenience function for quick setup
def get_logger_for_run(run_id: str) -> tuple[logging.Logger, LogContext]:
    """
    Quick setup for run-specific logging.
    
    Args:
        run_id: Unique run identifier
        
    Returns:
        Tuple of (logger, log_context)
    """
    logger = setup_logging(run_id)
    log_ctx = LogContext(logger, run_id)
    return logger, log_ctx
