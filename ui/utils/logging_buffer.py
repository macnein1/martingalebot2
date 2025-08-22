import logging
import threading
import json
import os
from collections import deque
from typing import Deque, Iterable, Optional, Dict, Any, List
from martingale_lab.utils.structured_logging import JSONFormatter


class RingBufferLogHandler(logging.Handler):
    """A thread-safe ring buffer handler that stores recent log records as formatted strings."""

    def __init__(self, name: str, capacity: int = 500):
        super().__init__()
        self.name = name
        self.capacity = capacity
        self._buffer: Deque[str] = deque(maxlen=capacity)
        self._json_buffer: Deque[Dict[str, Any]] = deque(maxlen=capacity)
        self._lock = threading.Lock()

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
        except Exception:
            msg = record.getMessage()
            
        # Also store JSON representation if available
        json_record = None
        try:
            if hasattr(record, 'extra_fields'):
                # This is a structured log record
                json_record = {
                    "ts": record.created,
                    "lvl": record.levelname,
                    "msg": record.getMessage(),
                    "module": record.module,
                    "func": record.funcName,
                    "line": record.lineno,
                    **record.extra_fields
                }
        except Exception:
            pass
            
        with self._lock:
            self._buffer.append(msg)
            if json_record:
                self._json_buffer.append(json_record)

    def tail(self, last_n: Optional[int] = None) -> Iterable[str]:
        with self._lock:
            if last_n is None or last_n >= len(self._buffer):
                return list(self._buffer)
            return list(list(self._buffer)[-last_n:])
    
    def tail_json(self, last_n: Optional[int] = None) -> List[Dict[str, Any]]:
        """Return recent logs as JSON objects"""
        with self._lock:
            if last_n is None or last_n >= len(self._json_buffer):
                return list(self._json_buffer)
            return list(list(self._json_buffer)[-last_n:])


_handlers_registry = {}
_registry_lock = threading.Lock()


def ensure_ring_handler(logger_name: str = "mlab", level: int = logging.INFO, 
                       use_json: bool = False) -> RingBufferLogHandler:
    """Attach (or return existing) ring buffer handler to the given logger."""
    with _registry_lock:
        if logger_name in _handlers_registry:
            return _handlers_registry[logger_name]

        # Get capacity from environment variable
        capacity = int(os.getenv("MLAB_TRACE_N", "1000"))
        handler = RingBufferLogHandler(name=f"{logger_name}.ring", capacity=capacity)
        
        if use_json:
            handler.setFormatter(JSONFormatter())
        else:
            fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
            handler.setFormatter(fmt)
            
        handler.setLevel(level)

        logger = logging.getLogger(logger_name)
        logger.setLevel(level)
        logger.addHandler(handler)

        _handlers_registry[logger_name] = handler
        return handler


def tail_logs(logger_name: str = "mlab", last_n: int = 200, as_json: bool = False) -> Iterable:
    """Yield the most recent logs from the ring buffer. Intended for UI refresh loops."""
    handler = ensure_ring_handler(logger_name, use_json=as_json)
    
    if as_json:
        return handler.tail_json(last_n)
    else:
        for line in handler.tail(last_n):
            yield line


def get_live_trace(logger_name: str = "mlab", event_filter: Optional[str] = None, 
                  last_n: int = 50) -> List[Dict[str, Any]]:
    """Get live trace logs for UI display, optionally filtered by event type"""
    logs = tail_logs(logger_name, last_n=last_n, as_json=True)
    
    if event_filter:
        return [log for log in logs if log.get("event", "").startswith(event_filter)]
    
    return list(logs)


def clear_logs(logger_name: str = "mlab"):
    """Clear the ring buffer for a logger"""
    with _registry_lock:
        if logger_name in _handlers_registry:
            handler = _handlers_registry[logger_name]
            with handler._lock:
                handler._buffer.clear()
                handler._json_buffer.clear()


