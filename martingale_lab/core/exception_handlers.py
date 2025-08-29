"""
Robust exception handling utilities.
Provides decorators and context managers for better error handling.
"""
from __future__ import annotations

import functools
import logging
import sqlite3
import time
import traceback
from contextlib import contextmanager
from typing import Any, Callable, Optional, TypeVar, Union

logger = logging.getLogger(__name__)

T = TypeVar('T')


class RetryableError(Exception):
    """Error that can be retried."""
    pass


class NonRetryableError(Exception):
    """Error that should not be retried."""
    pass


def classify_sqlite_error(error: sqlite3.Error) -> tuple[bool, str]:
    """
    Classify SQLite error and determine if retryable.
    
    Returns:
        Tuple of (is_retryable, error_category)
    """
    error_msg = str(error).lower()
    
    # Retryable errors
    if isinstance(error, sqlite3.OperationalError):
        if 'locked' in error_msg or 'busy' in error_msg:
            return True, "database_locked"
        elif 'disk' in error_msg:
            return False, "disk_error"
        elif 'corrupt' in error_msg:
            return False, "database_corrupt"
        else:
            return True, "operational"
    
    elif isinstance(error, sqlite3.IntegrityError):
        if 'unique' in error_msg or 'constraint' in error_msg:
            return False, "constraint_violation"
        else:
            return False, "integrity"
    
    elif isinstance(error, sqlite3.DatabaseError):
        return False, "database_error"
    
    else:
        return False, "unknown_sqlite_error"


def retry_on_error(
    max_retries: int = 3,
    initial_delay: float = 0.1,
    backoff_factor: float = 2.0,
    max_delay: float = 10.0,
    exceptions: tuple = (RetryableError, sqlite3.OperationalError)
):
    """
    Decorator for retrying functions on specific errors.
    
    Args:
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay between retries in seconds
        backoff_factor: Multiplier for delay on each retry
        max_delay: Maximum delay between retries
        exceptions: Tuple of exceptions to retry on
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            delay = initial_delay
            last_error = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                    
                except exceptions as e:
                    last_error = e
                    
                    # Check if SQLite error is retryable
                    if isinstance(e, sqlite3.Error):
                        is_retryable, category = classify_sqlite_error(e)
                        if not is_retryable:
                            logger.error(
                                f"{func.__name__} failed with non-retryable {category}: {e}"
                            )
                            raise NonRetryableError(str(e)) from e
                    
                    if attempt < max_retries:
                        logger.warning(
                            f"{func.__name__} failed (attempt {attempt + 1}/{max_retries + 1}): {e}. "
                            f"Retrying in {delay:.2f}s..."
                        )
                        time.sleep(delay)
                        delay = min(delay * backoff_factor, max_delay)
                    else:
                        logger.error(
                            f"{func.__name__} failed after {max_retries + 1} attempts: {e}"
                        )
                        raise
                    
                except Exception as e:
                    # Non-retryable error
                    logger.error(
                        f"{func.__name__} failed with non-retryable error: {e}",
                        exc_info=True
                    )
                    raise
            
            # Should not reach here, but just in case
            if last_error:
                raise last_error
            
        return wrapper
    return decorator


@contextmanager
def safe_transaction(conn: sqlite3.Connection, isolation_level: str = "DEFERRED"):
    """
    Context manager for safe database transactions.
    
    Args:
        conn: Database connection
        isolation_level: Transaction isolation level
        
    Yields:
        Cursor object
    """
    cursor = None
    transaction_started = False
    
    try:
        # Start transaction
        cursor = conn.cursor()
        cursor.execute(f"BEGIN {isolation_level}")
        transaction_started = True
        
        yield cursor
        
        # Commit on success
        cursor.execute("COMMIT")
        transaction_started = False
        
    except sqlite3.Error as e:
        # Database error - rollback
        logger.error(f"Database error in transaction: {e}", exc_info=True)
        if transaction_started:
            try:
                cursor.execute("ROLLBACK")
            except sqlite3.Error as rollback_error:
                logger.error(f"Failed to rollback: {rollback_error}")
        raise
        
    except Exception as e:
        # Other error - rollback
        logger.error(f"Unexpected error in transaction: {e}", exc_info=True)
        if transaction_started:
            try:
                cursor.execute("ROLLBACK")
            except sqlite3.Error as rollback_error:
                logger.error(f"Failed to rollback: {rollback_error}")
        raise
        
    finally:
        if cursor:
            try:
                cursor.close()
            except:
                pass


@contextmanager
def safe_file_operation(filepath: str, mode: str = 'r', encoding: str = 'utf-8'):
    """
    Context manager for safe file operations.
    
    Args:
        filepath: Path to file
        mode: File open mode
        encoding: Text encoding
        
    Yields:
        File object
    """
    file_handle = None
    
    try:
        file_handle = open(filepath, mode, encoding=encoding)
        yield file_handle
        
    except IOError as e:
        logger.error(f"IO error with file {filepath}: {e}", exc_info=True)
        raise
        
    except Exception as e:
        logger.error(f"Unexpected error with file {filepath}: {e}", exc_info=True)
        raise
        
    finally:
        if file_handle:
            try:
                file_handle.close()
            except:
                pass


def log_and_suppress(
    exceptions: tuple = (Exception,),
    default_return: Any = None,
    log_level: str = "error"
):
    """
    Decorator to log and suppress exceptions.
    
    Args:
        exceptions: Tuple of exceptions to catch
        default_return: Value to return on exception
        log_level: Logging level for errors
    """
    def decorator(func: Callable[..., T]) -> Callable[..., Optional[T]]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Optional[T]:
            try:
                return func(*args, **kwargs)
            except exceptions as e:
                log_func = getattr(logger, log_level)
                log_func(
                    f"{func.__name__} failed: {e}",
                    exc_info=(log_level == "error")
                )
                return default_return
        return wrapper
    return decorator


def measure_execution_time(func: Callable[..., T]) -> Callable[..., T]:
    """
    Decorator to measure and log execution time.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> T:
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            elapsed = (time.time() - start_time) * 1000
            logger.debug(f"{func.__name__} completed in {elapsed:.2f}ms")
            return result
        except Exception as e:
            elapsed = (time.time() - start_time) * 1000
            logger.error(f"{func.__name__} failed after {elapsed:.2f}ms: {e}")
            raise
    return wrapper


class ErrorCollector:
    """
    Collects errors for batch reporting.
    """
    
    def __init__(self, max_errors: int = 100):
        """
        Initialize error collector.
        
        Args:
            max_errors: Maximum errors to keep
        """
        self.errors = []
        self.max_errors = max_errors
        self.error_counts = {}
    
    def add(self, error: Exception, context: Optional[str] = None):
        """Add an error to the collection."""
        error_type = type(error).__name__
        error_msg = str(error)
        
        # Count by type
        self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
        
        # Store if under limit
        if len(self.errors) < self.max_errors:
            self.errors.append({
                'type': error_type,
                'message': error_msg,
                'context': context,
                'traceback': traceback.format_exc()
            })
    
    def get_summary(self) -> dict:
        """Get error summary."""
        return {
            'total_errors': sum(self.error_counts.values()),
            'error_types': self.error_counts,
            'sample_errors': self.errors[:10]
        }
    
    def clear(self):
        """Clear collected errors."""
        self.errors.clear()
        self.error_counts.clear()
    
    def log_summary(self, log_level: str = "error"):
        """Log error summary."""
        summary = self.get_summary()
        if summary['total_errors'] > 0:
            log_func = getattr(logger, log_level)
            log_func(f"Error summary: {summary['total_errors']} errors")
            for error_type, count in summary['error_types'].items():
                log_func(f"  {error_type}: {count}")


def validate_required_params(**required):
    """
    Decorator to validate required parameters.
    
    Usage:
        @validate_required_params(x=int, y=float, z=str)
        def my_func(x, y, z):
            ...
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get all arguments
            import inspect
            sig = inspect.signature(func)
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            
            # Validate required parameters
            for param_name, expected_type in required.items():
                if param_name not in bound.arguments:
                    raise ValueError(f"Missing required parameter: {param_name}")
                
                value = bound.arguments[param_name]
                if value is not None and not isinstance(value, expected_type):
                    raise TypeError(
                        f"Parameter {param_name} must be {expected_type.__name__}, "
                        f"got {type(value).__name__}"
                    )
            
            return func(*args, **kwargs)
        return wrapper
    return decorator