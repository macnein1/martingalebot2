"""
Thread-safe write queue manager for database operations.
Implements single-writer pattern with batch processing.
"""
from __future__ import annotations

import sqlite3
import threading
import queue
import time
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Callable
from contextlib import contextmanager
from enum import Enum

logger = logging.getLogger(__name__)


class WriteOperation(Enum):
    """Types of write operations."""
    INSERT = "INSERT"
    UPDATE = "UPDATE"
    DELETE = "DELETE"
    EXECUTE = "EXECUTE"
    EXECUTE_MANY = "EXECUTE_MANY"


@dataclass
class WriteRequest:
    """A single write request."""
    operation: WriteOperation
    query: str
    params: Any
    callback: Optional[Callable] = None
    retry_count: int = 0
    max_retries: int = 3


class WriteQueueManager:
    """
    Manages database writes through a single writer thread.
    
    Features:
    - Single writer thread pattern
    - Batch processing for efficiency
    - Automatic retries with backoff
    - Transaction management
    - Performance metrics
    """
    
    def __init__(
        self,
        db_path: str,
        batch_size: int = 500,
        flush_interval_ms: int = 100,
        max_queue_size: int = 10000
    ):
        """
        Initialize write queue manager.
        
        Args:
            db_path: Path to SQLite database
            batch_size: Max items per batch transaction
            flush_interval_ms: Max time between flushes
            max_queue_size: Maximum queue size before blocking
        """
        self.db_path = db_path
        self.batch_size = batch_size
        self.flush_interval_ms = flush_interval_ms
        
        # Queue for write requests
        self._queue = queue.Queue(maxsize=max_queue_size)
        
        # Writer thread
        self._writer_thread = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        
        # Metrics
        self._metrics = {
            'total_writes': 0,
            'successful_writes': 0,
            'failed_writes': 0,
            'retries': 0,
            'batches_processed': 0,
            'avg_batch_size': 0,
            'total_time_ms': 0
        }
        
        # Start writer thread
        self._start_writer()
    
    def _start_writer(self):
        """Start the writer thread."""
        with self._lock:
            if self._writer_thread is None or not self._writer_thread.is_alive():
                self._writer_thread = threading.Thread(
                    target=self._writer_loop,
                    daemon=True,
                    name="DBWriterThread"
                )
                self._writer_thread.start()
                logger.info(f"Writer thread started for {self.db_path}")
    
    def _writer_loop(self):
        """Main writer thread loop."""
        conn = None
        last_flush = time.time()
        batch = []
        
        try:
            # Create connection with optimized pragmas
            conn = self._create_optimized_connection()
            
            while not self._stop_event.is_set():
                try:
                    # Calculate timeout
                    timeout = (self.flush_interval_ms / 1000.0) - (time.time() - last_flush)
                    timeout = max(0.001, timeout)
                    
                    # Get item from queue
                    try:
                        request = self._queue.get(timeout=timeout)
                        batch.append(request)
                    except queue.Empty:
                        pass
                    
                    # Check if we should flush
                    should_flush = (
                        len(batch) >= self.batch_size or
                        (time.time() - last_flush) * 1000 >= self.flush_interval_ms or
                        (len(batch) > 0 and self._stop_event.is_set())
                    )
                    
                    if should_flush and batch:
                        self._process_batch(conn, batch)
                        batch = []
                        last_flush = time.time()
                        
                except Exception as e:
                    logger.error(f"Error in writer loop: {e}", exc_info=True)
                    # Don't lose the batch
                    for req in batch:
                        if req.retry_count < req.max_retries:
                            req.retry_count += 1
                            self._queue.put(req)
                    batch = []
                    
                    # Recreate connection if needed
                    if conn:
                        try:
                            conn.close()
                        except:
                            pass
                    conn = self._create_optimized_connection()
                    
        finally:
            # Final flush
            if batch and conn:
                try:
                    self._process_batch(conn, batch)
                except Exception as e:
                    logger.error(f"Error in final flush: {e}")
            
            # Close connection
            if conn:
                try:
                    conn.close()
                except:
                    pass
            
            logger.info("Writer thread stopped")
    
    def _create_optimized_connection(self) -> sqlite3.Connection:
        """Create SQLite connection with optimized pragmas."""
        conn = sqlite3.connect(self.db_path, isolation_level=None)
        
        # Optimize for write performance
        pragmas = [
            "PRAGMA journal_mode = WAL",
            "PRAGMA synchronous = NORMAL",
            "PRAGMA temp_store = MEMORY",
            "PRAGMA cache_size = -20000",  # 20MB cache
            "PRAGMA mmap_size = 30000000000",  # 30GB mmap
            "PRAGMA page_size = 4096",
            "PRAGMA wal_checkpoint(TRUNCATE)",
        ]
        
        for pragma in pragmas:
            try:
                conn.execute(pragma)
            except sqlite3.Error as e:
                logger.warning(f"Failed to set {pragma}: {e}")
        
        return conn
    
    def _process_batch(self, conn: sqlite3.Connection, batch: List[WriteRequest]):
        """
        Process a batch of write requests.
        
        Args:
            conn: Database connection
            batch: List of write requests
        """
        if not batch:
            return
        
        start_time = time.time()
        success_count = 0
        
        # Start transaction
        try:
            conn.execute("BEGIN IMMEDIATE")
            
            for request in batch:
                try:
                    self._execute_request(conn, request)
                    success_count += 1
                    
                    # Call callback if provided
                    if request.callback:
                        try:
                            request.callback(True, None)
                        except Exception as e:
                            logger.warning(f"Callback error: {e}")
                            
                except sqlite3.Error as e:
                    logger.error(f"Failed to execute {request.operation}: {e}")
                    self._metrics['failed_writes'] += 1
                    
                    # Retry logic
                    if request.retry_count < request.max_retries:
                        request.retry_count += 1
                        self._metrics['retries'] += 1
                        # Re-queue for retry
                        self._queue.put(request)
                    else:
                        # Call callback with error
                        if request.callback:
                            try:
                                request.callback(False, e)
                            except:
                                pass
            
            # Commit transaction
            conn.execute("COMMIT")
            
        except sqlite3.Error as e:
            logger.error(f"Batch transaction failed: {e}")
            try:
                conn.execute("ROLLBACK")
            except:
                pass
            
            # Re-queue all requests for retry
            for request in batch:
                if request.retry_count < request.max_retries:
                    request.retry_count += 1
                    self._queue.put(request)
        
        # Update metrics
        elapsed_ms = (time.time() - start_time) * 1000
        self._update_metrics(len(batch), success_count, elapsed_ms)
        
        if success_count > 0:
            logger.debug(f"Processed batch: {success_count}/{len(batch)} in {elapsed_ms:.1f}ms")
    
    def _execute_request(self, conn: sqlite3.Connection, request: WriteRequest):
        """Execute a single write request."""
        if request.operation == WriteOperation.INSERT:
            conn.execute(request.query, request.params)
        elif request.operation == WriteOperation.UPDATE:
            conn.execute(request.query, request.params)
        elif request.operation == WriteOperation.DELETE:
            conn.execute(request.query, request.params)
        elif request.operation == WriteOperation.EXECUTE:
            conn.execute(request.query, request.params or ())
        elif request.operation == WriteOperation.EXECUTE_MANY:
            conn.executemany(request.query, request.params)
        else:
            raise ValueError(f"Unknown operation: {request.operation}")
        
        self._metrics['successful_writes'] += 1
    
    def _update_metrics(self, batch_size: int, success_count: int, elapsed_ms: float):
        """Update performance metrics."""
        with self._lock:
            self._metrics['total_writes'] += batch_size
            self._metrics['successful_writes'] += success_count
            self._metrics['batches_processed'] += 1
            self._metrics['total_time_ms'] += elapsed_ms
            
            # Update average batch size
            total_batches = self._metrics['batches_processed']
            prev_avg = self._metrics['avg_batch_size']
            self._metrics['avg_batch_size'] = (
                (prev_avg * (total_batches - 1) + batch_size) / total_batches
            )
    
    # Public API
    
    def write(
        self,
        query: str,
        params: Any = None,
        operation: WriteOperation = WriteOperation.EXECUTE,
        callback: Optional[Callable] = None,
        block: bool = True
    ) -> bool:
        """
        Queue a write operation.
        
        Args:
            query: SQL query
            params: Query parameters
            operation: Type of operation
            callback: Optional callback(success, error)
            block: Whether to block if queue is full
            
        Returns:
            True if queued successfully
        """
        request = WriteRequest(
            operation=operation,
            query=query,
            params=params,
            callback=callback
        )
        
        try:
            self._queue.put(request, block=block)
            return True
        except queue.Full:
            logger.error("Write queue is full")
            return False
    
    def insert(self, table: str, data: Dict[str, Any], callback: Optional[Callable] = None):
        """Insert a row into table."""
        columns = list(data.keys())
        placeholders = ['?' for _ in columns]
        query = f"INSERT INTO {table} ({','.join(columns)}) VALUES ({','.join(placeholders)})"
        params = list(data.values())
        return self.write(query, params, WriteOperation.INSERT, callback)
    
    def insert_many(self, table: str, data: List[Dict[str, Any]], callback: Optional[Callable] = None):
        """Insert multiple rows."""
        if not data:
            return True
        
        columns = list(data[0].keys())
        placeholders = ['?' for _ in columns]
        query = f"INSERT INTO {table} ({','.join(columns)}) VALUES ({','.join(placeholders)})"
        params = [list(row.values()) for row in data]
        return self.write(query, params, WriteOperation.EXECUTE_MANY, callback)
    
    def update(self, table: str, data: Dict[str, Any], where: str, where_params: List[Any], 
               callback: Optional[Callable] = None):
        """Update rows in table."""
        set_clause = ','.join([f"{k}=?" for k in data.keys()])
        query = f"UPDATE {table} SET {set_clause} WHERE {where}"
        params = list(data.values()) + where_params
        return self.write(query, params, WriteOperation.UPDATE, callback)
    
    def flush(self, timeout: float = 5.0) -> bool:
        """
        Flush pending writes.
        
        Args:
            timeout: Max time to wait for flush
            
        Returns:
            True if queue is empty
        """
        start = time.time()
        while not self._queue.empty():
            if time.time() - start > timeout:
                return False
            time.sleep(0.01)
        
        # Wait a bit more for processing
        time.sleep(self.flush_interval_ms / 1000.0 * 2)
        return self._queue.empty()
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics."""
        with self._lock:
            metrics = self._metrics.copy()
            metrics['queue_size'] = self._queue.qsize()
            metrics['writer_alive'] = self._writer_thread.is_alive() if self._writer_thread else False
            
            # Calculate rates
            if metrics['total_time_ms'] > 0:
                metrics['writes_per_second'] = (
                    metrics['successful_writes'] / (metrics['total_time_ms'] / 1000.0)
                )
            else:
                metrics['writes_per_second'] = 0
            
            return metrics
    
    def stop(self, timeout: float = 10.0):
        """
        Stop the writer thread.
        
        Args:
            timeout: Max time to wait for thread to stop
        """
        logger.info("Stopping write queue manager...")
        
        # Signal stop
        self._stop_event.set()
        
        # Flush remaining items
        self.flush(timeout=timeout/2)
        
        # Wait for thread to stop
        if self._writer_thread:
            self._writer_thread.join(timeout=timeout/2)
            if self._writer_thread.is_alive():
                logger.warning("Writer thread did not stop gracefully")
        
        # Log final metrics
        metrics = self.get_metrics()
        logger.info(f"Final metrics: {metrics}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
        return False


# Global instance management
_write_queue_instances: Dict[str, WriteQueueManager] = {}
_instances_lock = threading.Lock()


def get_write_queue(db_path: str, **kwargs) -> WriteQueueManager:
    """
    Get or create a write queue manager for a database.
    
    Args:
        db_path: Database path
        **kwargs: Additional arguments for WriteQueueManager
        
    Returns:
        WriteQueueManager instance
    """
    with _instances_lock:
        if db_path not in _write_queue_instances:
            _write_queue_instances[db_path] = WriteQueueManager(db_path, **kwargs)
        return _write_queue_instances[db_path]


def close_all_queues():
    """Close all write queue managers."""
    with _instances_lock:
        for queue_mgr in _write_queue_instances.values():
            try:
                queue_mgr.stop()
            except Exception as e:
                logger.error(f"Error stopping queue: {e}")
        _write_queue_instances.clear()