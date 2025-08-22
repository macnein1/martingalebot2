"""
Performance monitoring utilities for the Streamlit UI.
"""
import psutil
import time
from typing import Dict, Any
import streamlit as st

class PerformanceMonitor:
    """Monitor system performance metrics."""
    
    @staticmethod
    def get_system_metrics() -> Dict[str, Any]:
        """Get current system performance metrics."""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            return {
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'memory_available_gb': memory.available / (1024**3),
                'memory_total_gb': memory.total / (1024**3),
                'disk_percent': disk.percent,
                'disk_free_gb': disk.free / (1024**3),
                'timestamp': time.time()
            }
        except Exception as e:
            return {
                'error': str(e),
                'timestamp': time.time()
            }
    
    @staticmethod
    def format_bytes(bytes_value: float) -> str:
        """Format bytes to human readable format."""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if bytes_value < 1024.0:
                return f"{bytes_value:.1f} {unit}"
            bytes_value /= 1024.0
        return f"{bytes_value:.1f} PB"
    
    @staticmethod
    def get_performance_status(metrics: Dict[str, Any]) -> str:
        """Get performance status based on metrics."""
        if 'error' in metrics:
            return "error"
        
        cpu = metrics.get('cpu_percent', 0)
        memory = metrics.get('memory_percent', 0)
        
        if cpu > 80 or memory > 85:
            return "critical"
        elif cpu > 60 or memory > 70:
            return "warning"
        else:
            return "good"
    
    @staticmethod
    def create_performance_summary(metrics: Dict[str, Any]) -> Dict[str, str]:
        """Create a formatted performance summary."""
        if 'error' in metrics:
            return {'status': 'Error', 'details': metrics['error']}
        
        status = PerformanceMonitor.get_performance_status(metrics)
        
        return {
            'status': status.title(),
            'cpu': f"{metrics.get('cpu_percent', 0):.1f}%",
            'memory': f"{metrics.get('memory_percent', 0):.1f}%",
            'memory_available': f"{metrics.get('memory_available_gb', 0):.1f} GB",
            'disk_free': f"{metrics.get('disk_free_gb', 0):.1f} GB"
        }


def display_performance_metrics():
    """Display performance metrics in Streamlit."""
    monitor = PerformanceMonitor()
    metrics = monitor.get_system_metrics()
    summary = monitor.create_performance_summary(metrics)
    
    # Create columns for metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        status = summary['status']
        if status == "Good":
            st.success(f"🟢 {status}")
        elif status == "Warning":
            st.warning(f"🟡 {status}")
        elif status == "Critical":
            st.error(f"🔴 {status}")
        else:
            st.info(f"ℹ️ {status}")
    
    with col2:
        st.metric("CPU", summary['cpu'])
    
    with col3:
        st.metric("Memory", summary['memory'])
    
    with col4:
        st.metric("Available RAM", summary['memory_available'])
    
    return metrics, summary


class OptimizationProfiler:
    """Profile optimization performance."""
    
    def __init__(self):
        self.start_time = None
        self.metrics_history = []
    
    def start_profiling(self):
        """Start performance profiling."""
        self.start_time = time.time()
        self.metrics_history = []
    
    def record_metric(self, metric_name: str, value: float):
        """Record a performance metric."""
        if self.start_time is None:
            return
        
        self.metrics_history.append({
            'metric': metric_name,
            'value': value,
            'timestamp': time.time() - self.start_time
        })
    
    def get_profile_summary(self) -> Dict[str, Any]:
        """Get profiling summary."""
        if not self.metrics_history:
            return {}
        
        total_time = max(m['timestamp'] for m in self.metrics_history)
        
        return {
            'total_time': total_time,
            'metrics_count': len(self.metrics_history),
            'metrics': self.metrics_history
        }


# Global profiler instance
optimization_profiler = OptimizationProfiler()
