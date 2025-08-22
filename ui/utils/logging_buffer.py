import logging
import threading
from collections import deque
from typing import Deque, Iterable, Optional


class RingBufferLogHandler(logging.Handler):
    """A thread-safe ring buffer handler that stores recent log records as formatted strings."""

    def __init__(self, name: str, capacity: int = 500):
        super().__init__()
        self.name = name
        self.capacity = capacity
        self._buffer: Deque[str] = deque(maxlen=capacity)
        self._lock = threading.Lock()

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
        except Exception:
            msg = record.getMessage()
        with self._lock:
            self._buffer.append(msg)

    def tail(self, last_n: Optional[int] = None) -> Iterable[str]:
        with self._lock:
            if last_n is None or last_n >= len(self._buffer):
                return list(self._buffer)
            return list(list(self._buffer)[-last_n:])


_handlers_registry = {}
_registry_lock = threading.Lock()


def ensure_ring_handler(logger_name: str = "mlab", level: int = logging.INFO) -> RingBufferLogHandler:
    """Attach (or return existing) ring buffer handler to the given logger."""
    with _registry_lock:
        if logger_name in _handlers_registry:
            return _handlers_registry[logger_name]

        handler = RingBufferLogHandler(name=f"{logger_name}.ring", capacity=1000)
        fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        handler.setFormatter(fmt)
        handler.setLevel(level)

        logger = logging.getLogger(logger_name)
        logger.setLevel(level)
        logger.addHandler(handler)

        _handlers_registry[logger_name] = handler
        return handler


def tail_logs(logger_name: str = "mlab", last_n: int = 200) -> Iterable[str]:
    """Yield the most recent logs from the ring buffer. Intended for UI refresh loops."""
    handler = ensure_ring_handler(logger_name)
    for line in handler.tail(last_n):
        yield line


