"""
Error boundaries and fallback mechanisms for robust optimization.
Provides layered exception handling with automatic recovery strategies.
"""
import functools
import time
import traceback
from typing import Any, Callable, Optional, Dict, Union
from enum import Enum
from dataclasses import dataclass

from .logging import LogContext


class ErrorSeverity(Enum):
    """Error severity levels for different handling strategies."""
    LOW = "low"           # Continue with fallback
    MEDIUM = "medium"     # Log and skip current item
    HIGH = "high"         # Stop current batch, checkpoint
    CRITICAL = "critical" # Stop entire run


@dataclass
class ErrorResult:
    """Result of error handling with recovery information."""
    success: bool
    result: Any = None
    error: Optional[Exception] = None
    fallback_used: bool = False
    recovery_strategy: Optional[str] = None
    duration_ms: float = 0.0


class SafeEvaluator:
    """Safe evaluation wrapper with multiple fallback strategies."""
    
    def __init__(self, log_ctx: LogContext):
        self.log_ctx = log_ctx
        self.error_counts = {
            'numba_errors': 0,
            'nan_results': 0,
            'timeout_errors': 0,
            'validation_errors': 0,
            'unknown_errors': 0
        }
    
    def safe_eval_candidate(self, candidate: Any, 
                           primary_evaluator: Callable,
                           fallback_evaluator: Optional[Callable] = None,
                           timeout_seconds: float = 30.0) -> ErrorResult:
        """
        Safely evaluate a candidate with fallback mechanisms.
        
        Args:
            candidate: Candidate to evaluate
            primary_evaluator: Primary evaluation function (e.g., Numba)
            fallback_evaluator: Fallback function (e.g., NumPy)
            timeout_seconds: Maximum evaluation time
            
        Returns:
            ErrorResult with evaluation outcome
        """
        start_time = time.perf_counter()
        candidate_id = getattr(candidate, 'id', 'unknown')
        
        try:
            # Try primary evaluator first
            self.log_ctx.log('candidate_eval_start', candidate_id=candidate_id, method='primary')
            
            result = self._with_timeout(primary_evaluator, candidate, timeout_seconds)
            
            # Validate result
            if self._is_invalid_result(result):
                raise ValueError(f"Invalid result: {result}")
            
            duration_ms = (time.perf_counter() - start_time) * 1000
            self.log_ctx.timing('candidate_eval_success', duration_ms, 
                              candidate_id=candidate_id, method='primary')
            
            return ErrorResult(
                success=True,
                result=result,
                duration_ms=duration_ms
            )
            
        except Exception as primary_error:
            self._log_error('primary_evaluator_error', primary_error, candidate_id)
            
            # Try fallback if available
            if fallback_evaluator:
                try:
                    self.log_ctx.log('candidate_fallback_start', candidate_id=candidate_id)
                    
                    result = self._with_timeout(fallback_evaluator, candidate, timeout_seconds * 2)
                    
                    if self._is_invalid_result(result):
                        raise ValueError(f"Invalid fallback result: {result}")
                    
                    duration_ms = (time.perf_counter() - start_time) * 1000
                    self.log_ctx.timing('candidate_fallback_success', duration_ms,
                                      candidate_id=candidate_id, method='fallback')
                    
                    return ErrorResult(
                        success=True,
                        result=result,
                        fallback_used=True,
                        recovery_strategy='fallback_evaluator',
                        duration_ms=duration_ms
                    )
                    
                except Exception as fallback_error:
                    self._log_error('fallback_evaluator_error', fallback_error, candidate_id)
            
            # Both evaluators failed
            duration_ms = (time.perf_counter() - start_time) * 1000
            self.log_ctx.log('candidate_eval_failed', 
                           candidate_id=candidate_id,
                           primary_error=str(primary_error),
                           duration_ms=duration_ms)
            
            return ErrorResult(
                success=False,
                error=primary_error,
                duration_ms=duration_ms
            )
    
    def _with_timeout(self, func: Callable, *args, timeout_seconds: float) -> Any:
        """Execute function with timeout protection."""
        # Simple timeout implementation - for more complex cases, use threading/multiprocessing
        start_time = time.perf_counter()
        result = func(*args)
        
        elapsed = time.perf_counter() - start_time
        if elapsed > timeout_seconds:
            self.error_counts['timeout_errors'] += 1
            raise TimeoutError(f"Evaluation timed out after {elapsed:.2f}s")
        
        return result
    
    def _is_invalid_result(self, result: Any) -> bool:
        """Check if result contains invalid values."""
        if result is None:
            return True
        
        # Check for NaN/Inf in numeric results
        try:
            import numpy as np
            if hasattr(result, '__iter__') and not isinstance(result, str):
                if np.any(np.isnan(list(result))) or np.any(np.isinf(list(result))):
                    self.error_counts['nan_results'] += 1
                    return True
            elif isinstance(result, (int, float)):
                if np.isnan(result) or np.isinf(result):
                    self.error_counts['nan_results'] += 1
                    return True
        except Exception:
            pass
        
        return False
    
    def _log_error(self, event: str, error: Exception, candidate_id: str):
        """Log error with classification."""
        error_type = type(error).__name__
        
        # Classify error for statistics
        if 'numba' in str(error).lower():
            self.error_counts['numba_errors'] += 1
        elif isinstance(error, ValueError):
            self.error_counts['validation_errors'] += 1
        elif isinstance(error, TimeoutError):
            self.error_counts['timeout_errors'] += 1
        else:
            self.error_counts['unknown_errors'] += 1
        
        self.log_ctx.error(event, error, 
                          candidate_id=candidate_id,
                          error_type=error_type,
                          traceback=traceback.format_exc())
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of errors encountered."""
        total_errors = sum(self.error_counts.values())
        return {
            'total_errors': total_errors,
            'error_breakdown': self.error_counts.copy(),
            'error_rate_pct': 0.0 if total_errors == 0 else 
                             (total_errors / (total_errors + 1)) * 100  # Rough estimate
        }


class BatchProcessor:
    """Process batches with error isolation and recovery."""
    
    def __init__(self, log_ctx: LogContext):
        self.log_ctx = log_ctx
        self.safe_evaluator = SafeEvaluator(log_ctx)
    
    def process_batch_safe(self, candidates: list, 
                          evaluator: Callable,
                          fallback_evaluator: Optional[Callable] = None,
                          max_failures_pct: float = 50.0) -> Dict[str, Any]:
        """
        Process a batch of candidates with error isolation.
        
        Args:
            candidates: List of candidates to evaluate
            evaluator: Primary evaluation function
            fallback_evaluator: Optional fallback function
            max_failures_pct: Stop batch if failure rate exceeds this
            
        Returns:
            Dictionary with results and statistics
        """
        batch_start = time.perf_counter()
        results = []
        failed_candidates = []
        
        self.log_ctx.log('batch_start', 
                        candidate_count=len(candidates),
                        max_failure_rate=max_failures_pct)
        
        for i, candidate in enumerate(candidates):
            try:
                eval_result = self.safe_evaluator.safe_eval_candidate(
                    candidate, evaluator, fallback_evaluator
                )
                
                if eval_result.success:
                    results.append({
                        'candidate': candidate,
                        'result': eval_result.result,
                        'fallback_used': eval_result.fallback_used,
                        'duration_ms': eval_result.duration_ms
                    })
                else:
                    failed_candidates.append({
                        'candidate': candidate,
                        'error': str(eval_result.error),
                        'index': i
                    })
                
                # Check failure rate
                current_failure_rate = len(failed_candidates) / (i + 1) * 100
                if current_failure_rate > max_failures_pct and i > 10:  # At least 10 attempts
                    self.log_ctx.log('batch_stopped_high_failure_rate',
                                   failure_rate=current_failure_rate,
                                   processed=i + 1,
                                   failed=len(failed_candidates))
                    break
                    
            except Exception as e:
                # Unexpected error in batch processing itself
                self.log_ctx.error('batch_processing_error', e, candidate_index=i)
                failed_candidates.append({
                    'candidate': candidate,
                    'error': f"Batch processing error: {str(e)}",
                    'index': i
                })
        
        batch_duration = (time.perf_counter() - batch_start) * 1000
        
        summary = {
            'successful_results': results,
            'failed_candidates': failed_candidates,
            'success_count': len(results),
            'failure_count': len(failed_candidates),
            'total_processed': len(results) + len(failed_candidates),
            'success_rate_pct': len(results) / len(candidates) * 100 if candidates else 0,
            'batch_duration_ms': batch_duration,
            'error_summary': self.safe_evaluator.get_error_summary()
        }
        
        self.log_ctx.log('batch_complete', **{k: v for k, v in summary.items() 
                                            if k not in ['successful_results', 'failed_candidates']})
        
        return summary


def resilient_operation(max_retries: int = 3, 
                       backoff_seconds: float = 1.0,
                       log_ctx: Optional[LogContext] = None):
    """
    Decorator for resilient operations with retry logic.
    
    Args:
        max_retries: Maximum number of retry attempts
        backoff_seconds: Base delay between retries (exponential backoff)
        log_ctx: Optional logging context
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    
                    if log_ctx:
                        log_ctx.error(f'{func.__name__}_retry_attempt', e, 
                                    attempt=attempt, max_retries=max_retries)
                    
                    if attempt < max_retries:
                        delay = backoff_seconds * (2 ** attempt)  # Exponential backoff
                        time.sleep(delay)
                    else:
                        # Final attempt failed
                        if log_ctx:
                            log_ctx.error(f'{func.__name__}_final_failure', e,
                                        total_attempts=max_retries + 1)
                        raise last_exception
            
            raise last_exception  # Should never reach here
        
        return wrapper
    return decorator


class DatabaseFailover:
    """Failover mechanism for database operations."""
    
    def __init__(self, log_ctx: LogContext, failover_file: str = "results_failover.jsonl"):
        self.log_ctx = log_ctx
        self.failover_file = failover_file
    
    @resilient_operation(max_retries=3)
    def safe_db_write(self, write_func: Callable, data: Any) -> bool:
        """
        Safely write to database with failover to file.
        
        Args:
            write_func: Database write function
            data: Data to write
            
        Returns:
            True if successful, False if failed over to file
        """
        try:
            write_func(data)
            return True
        except Exception as e:
            self.log_ctx.error('database_write_failed', e)
            
            # Failover to file
            try:
                import json
                with open(self.failover_file, 'a', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, default=str)
                    f.write('\n')
                
                self.log_ctx.log('database_failover_success', 
                               failover_file=self.failover_file)
                return False
                
            except Exception as failover_error:
                self.log_ctx.error('database_failover_failed', failover_error)
                raise failover_error
