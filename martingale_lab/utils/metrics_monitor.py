"""
Performance metrics and monitoring for optimization runs.
Provides micro-profiling and observability for system performance.
"""
import time
import psutil
import threading
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque
from contextlib import contextmanager

from .logging import LogContext


@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""
    evaluations_per_second: float = 0.0
    accept_ratio: float = 0.0
    nan_count: int = 0
    pruned_count: int = 0
    promoted_count: int = 0
    best_score: Optional[float] = None
    avg_eval_time_ms: float = 0.0
    memory_usage_mb: float = 0.0
    cpu_usage_pct: float = 0.0
    numba_compile_time_ms: float = 0.0
    python_overhead_ms: float = 0.0
    batch_success_rate: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'evaluations_per_second': self.evaluations_per_second,
            'accept_ratio': self.accept_ratio,
            'nan_count': self.nan_count,
            'pruned_count': self.pruned_count,
            'promoted_count': self.promoted_count,
            'best_score': self.best_score,
            'avg_eval_time_ms': self.avg_eval_time_ms,
            'memory_usage_mb': self.memory_usage_mb,
            'cpu_usage_pct': self.cpu_usage_pct,
            'numba_compile_time_ms': self.numba_compile_time_ms,
            'python_overhead_ms': self.python_overhead_ms,
            'batch_success_rate': self.batch_success_rate
        }


@dataclass
class TimingRecord:
    """Record for timing measurements."""
    operation: str
    start_time: float
    end_time: Optional[float] = None
    duration_ms: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def finish(self):
        """Mark timing as finished and calculate duration."""
        self.end_time = time.perf_counter()
        self.duration_ms = (self.end_time - self.start_time) * 1000


class MicroProfiler:
    """Micro-profiler for detailed performance analysis."""
    
    def __init__(self, max_records: int = 1000):
        self.max_records = max_records
        self.timing_records: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_records))
        self.active_timings: Dict[str, TimingRecord] = {}
        self._lock = threading.Lock()
    
    @contextmanager
    def time_operation(self, operation: str, **metadata):
        """Context manager for timing operations."""
        timing_id = f"{operation}_{int(time.time() * 1000000)}"
        
        with self._lock:
            record = TimingRecord(
                operation=operation,
                start_time=time.perf_counter(),
                metadata=metadata
            )
            self.active_timings[timing_id] = record
        
        try:
            yield record
        finally:
            with self._lock:
                if timing_id in self.active_timings:
                    record = self.active_timings.pop(timing_id)
                    record.finish()
                    self.timing_records[operation].append(record)
    
    def get_timing_stats(self, operation: str) -> Dict[str, float]:
        """Get timing statistics for an operation."""
        with self._lock:
            records = list(self.timing_records[operation])
        
        if not records:
            return {}
        
        durations = [r.duration_ms for r in records if r.duration_ms is not None]
        
        if not durations:
            return {}
        
        return {
            'count': len(durations),
            'avg_ms': sum(durations) / len(durations),
            'min_ms': min(durations),
            'max_ms': max(durations),
            'total_ms': sum(durations),
            'p95_ms': sorted(durations)[int(0.95 * len(durations))] if len(durations) > 1 else durations[0],
            'p99_ms': sorted(durations)[int(0.99 * len(durations))] if len(durations) > 1 else durations[0]
        }
    
    def get_all_stats(self) -> Dict[str, Dict[str, float]]:
        """Get timing statistics for all operations."""
        return {op: self.get_timing_stats(op) for op in self.timing_records.keys()}
    
    def reset(self):
        """Clear all timing records."""
        with self._lock:
            self.timing_records.clear()
            self.active_timings.clear()


class SystemMonitor:
    """System resource monitoring."""
    
    def __init__(self):
        self.process = psutil.Process()
        self._baseline_memory = self.process.memory_info().rss / 1024 / 1024  # MB
    
    def get_system_metrics(self) -> Dict[str, float]:
        """Get current system metrics."""
        try:
            memory_info = self.process.memory_info()
            cpu_percent = self.process.cpu_percent()
            
            return {
                'memory_usage_mb': memory_info.rss / 1024 / 1024,
                'memory_delta_mb': (memory_info.rss / 1024 / 1024) - self._baseline_memory,
                'cpu_usage_pct': cpu_percent,
                'num_threads': self.process.num_threads(),
                'num_fds': self.process.num_fds() if hasattr(self.process, 'num_fds') else 0
            }
        except Exception:
            return {
                'memory_usage_mb': 0.0,
                'memory_delta_mb': 0.0,
                'cpu_usage_pct': 0.0,
                'num_threads': 0,
                'num_fds': 0
            }


class PerformanceTracker:
    """Main performance tracking and monitoring class."""
    
    def __init__(self, log_ctx: Optional[LogContext] = None):
        self.log_ctx = log_ctx
        self.profiler = MicroProfiler()
        self.system_monitor = SystemMonitor()
        
        # Counters
        self.evaluation_count = 0
        self.successful_evaluations = 0
        self.failed_evaluations = 0
        self.nan_count = 0
        self.pruned_count = 0
        self.promoted_count = 0
        self.fallback_count = 0
        
        # Timing
        self.batch_start_time: Optional[float] = None
        self.last_metrics_time = time.perf_counter()
        
        # Best score tracking
        self.best_score: Optional[float] = None
        
        # Time series data (last 100 points for sparklines)
        self.eval_rate_history = deque(maxlen=100)
        self.accept_rate_history = deque(maxlen=100)
        self.memory_history = deque(maxlen=100)
    
    def start_batch(self):
        """Mark start of a new batch."""
        self.batch_start_time = time.perf_counter()
        self.evaluation_count = 0
        self.successful_evaluations = 0
        self.failed_evaluations = 0
        self.nan_count = 0
        self.pruned_count = 0
        self.promoted_count = 0
        self.fallback_count = 0
        
        if self.log_ctx:
            self.log_ctx.log('performance_batch_start')
    
    def record_evaluation(self, success: bool, score: Optional[float] = None,
                         fallback_used: bool = False, nan_result: bool = False):
        """Record an evaluation result."""
        self.evaluation_count += 1
        
        if success:
            self.successful_evaluations += 1
            if score is not None and (self.best_score is None or score < self.best_score):
                self.best_score = score
        else:
            self.failed_evaluations += 1
        
        if fallback_used:
            self.fallback_count += 1
        
        if nan_result:
            self.nan_count += 1
    
    def record_pruning(self, pruned: int, promoted: int):
        """Record pruning/promotion counts."""
        self.pruned_count += pruned
        self.promoted_count += promoted
    
    def get_current_metrics(self) -> PerformanceMetrics:
        """Get current performance metrics."""
        current_time = time.perf_counter()
        
        # Calculate evaluation rate
        if self.batch_start_time:
            batch_duration = current_time - self.batch_start_time
            eval_rate = self.evaluation_count / batch_duration if batch_duration > 0 else 0.0
        else:
            eval_rate = 0.0
        
        # Calculate accept ratio
        accept_ratio = (
            self.successful_evaluations / self.evaluation_count 
            if self.evaluation_count > 0 else 0.0
        )
        
        # Get system metrics
        system_metrics = self.system_monitor.get_system_metrics()
        
        # Get timing stats
        eval_stats = self.profiler.get_timing_stats('candidate_evaluation')
        avg_eval_time = eval_stats.get('avg_ms', 0.0)
        
        numba_stats = self.profiler.get_timing_stats('numba_kernel')
        numba_time = numba_stats.get('avg_ms', 0.0)
        
        python_stats = self.profiler.get_timing_stats('python_overhead')
        python_time = python_stats.get('avg_ms', 0.0)
        
        # Calculate batch success rate
        batch_success_rate = accept_ratio * 100
        
        metrics = PerformanceMetrics(
            evaluations_per_second=eval_rate,
            accept_ratio=accept_ratio,
            nan_count=self.nan_count,
            pruned_count=self.pruned_count,
            promoted_count=self.promoted_count,
            best_score=self.best_score,
            avg_eval_time_ms=avg_eval_time,
            memory_usage_mb=system_metrics['memory_usage_mb'],
            cpu_usage_pct=system_metrics['cpu_usage_pct'],
            numba_compile_time_ms=numba_time,
            python_overhead_ms=python_time,
            batch_success_rate=batch_success_rate
        )
        
        # Update time series
        self.eval_rate_history.append(eval_rate)
        self.accept_rate_history.append(accept_ratio)
        self.memory_history.append(system_metrics['memory_usage_mb'])
        
        return metrics
    
    def log_metrics(self, batch_idx: Optional[int] = None):
        """Log current metrics."""
        metrics = self.get_current_metrics()
        
        if self.log_ctx:
            log_data = {
                'batch_idx': batch_idx,
                **metrics.to_dict()
            }
            self.log_ctx.log('performance_metrics', **log_data)
    
    def get_sparkline_data(self) -> Dict[str, List[float]]:
        """Get data for sparkline charts."""
        return {
            'eval_rate': list(self.eval_rate_history),
            'accept_rate': list(self.accept_rate_history),
            'memory_usage': list(self.memory_history)
        }
    
    def get_profiling_summary(self) -> Dict[str, Any]:
        """Get detailed profiling summary."""
        all_stats = self.profiler.get_all_stats()
        system_metrics = self.system_monitor.get_system_metrics()
        
        return {
            'timing_breakdown': all_stats,
            'system_metrics': system_metrics,
            'counters': {
                'total_evaluations': self.evaluation_count,
                'successful_evaluations': self.successful_evaluations,
                'failed_evaluations': self.failed_evaluations,
                'nan_count': self.nan_count,
                'pruned_count': self.pruned_count,
                'promoted_count': self.promoted_count,
                'fallback_count': self.fallback_count
            },
            'best_score': self.best_score
        }
    
    @contextmanager
    def time_evaluation(self, candidate_id: str, method: str = 'primary'):
        """Time a candidate evaluation."""
        with self.profiler.time_operation('candidate_evaluation', 
                                         candidate_id=candidate_id, method=method):
            yield
    
    @contextmanager
    def time_numba_kernel(self, kernel_name: str):
        """Time a Numba kernel execution."""
        with self.profiler.time_operation('numba_kernel', kernel=kernel_name):
            yield
    
    @contextmanager
    def time_python_overhead(self, operation: str):
        """Time Python overhead operations."""
        with self.profiler.time_operation('python_overhead', operation=operation):
            yield
    
    def reset_batch_metrics(self):
        """Reset batch-specific metrics."""
        self.evaluation_count = 0
        self.successful_evaluations = 0
        self.failed_evaluations = 0
        self.nan_count = 0
        self.pruned_count = 0
        self.promoted_count = 0
        self.fallback_count = 0
        self.batch_start_time = None


class MetricsCollector:
    """Collects and aggregates metrics across multiple batches."""
    
    def __init__(self, log_ctx: Optional[LogContext] = None):
        self.log_ctx = log_ctx
        self.batch_metrics: List[PerformanceMetrics] = []
        self.run_start_time = time.perf_counter()
    
    def add_batch_metrics(self, metrics: PerformanceMetrics, batch_idx: int):
        """Add metrics for a completed batch."""
        self.batch_metrics.append(metrics)
        
        if self.log_ctx:
            self.log_ctx.log('batch_metrics_collected', 
                           batch_idx=batch_idx, **metrics.to_dict())
    
    def get_run_summary(self) -> Dict[str, Any]:
        """Get summary statistics for the entire run."""
        if not self.batch_metrics:
            return {}
        
        run_duration = time.perf_counter() - self.run_start_time
        
        # Aggregate metrics
        total_evals = sum(m.evaluations_per_second * run_duration / len(self.batch_metrics) 
                         for m in self.batch_metrics)
        avg_eval_rate = sum(m.evaluations_per_second for m in self.batch_metrics) / len(self.batch_metrics)
        avg_accept_rate = sum(m.accept_ratio for m in self.batch_metrics) / len(self.batch_metrics)
        total_nan_count = sum(m.nan_count for m in self.batch_metrics)
        best_overall_score = min(m.best_score for m in self.batch_metrics if m.best_score is not None)
        
        return {
            'run_duration_minutes': run_duration / 60,
            'total_batches': len(self.batch_metrics),
            'estimated_total_evaluations': int(total_evals),
            'avg_evaluations_per_second': avg_eval_rate,
            'avg_accept_rate': avg_accept_rate,
            'total_nan_count': total_nan_count,
            'best_overall_score': best_overall_score,
            'performance_trend': self._calculate_trend()
        }
    
    def _calculate_trend(self) -> str:
        """Calculate performance trend over batches."""
        if len(self.batch_metrics) < 3:
            return 'insufficient_data'
        
        recent_avg = sum(m.evaluations_per_second for m in self.batch_metrics[-3:]) / 3
        early_avg = sum(m.evaluations_per_second for m in self.batch_metrics[:3]) / 3
        
        if recent_avg > early_avg * 1.1:
            return 'improving'
        elif recent_avg < early_avg * 0.9:
            return 'degrading'
        else:
            return 'stable'