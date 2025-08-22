"""
Advanced Diagnostics Module for Martingale Lab
Provides crash snapshot functionality, error analysis, and system diagnostics
"""
import os
import json
import time
import traceback
import psutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict

from martingale_lab.utils.structured_logging import get_structured_logger, EventNames
from ui.utils.constants import CRASH_SNAPSHOTS_DIR


# Initialize logger for diagnostics
logger = get_structured_logger("mlab.diagnostics")


@dataclass
class SystemStats:
    """System performance statistics"""
    cpu_percent: float
    memory_percent: float
    memory_available_mb: float
    disk_usage_percent: float
    process_count: int
    timestamp: float


@dataclass
class ErrorContext:
    """Error context information"""
    error_type: str
    error_message: str
    traceback_str: str
    timestamp: float
    module: str
    function: str
    line_number: int


@dataclass
class CrashSnapshot:
    """Complete crash snapshot"""
    snapshot_id: str
    run_id: Optional[str]
    exp_id: Optional[int]
    timestamp: float
    error_context: ErrorContext
    system_stats: SystemStats
    application_state: Dict[str, Any]
    recent_logs: List[Dict[str, Any]]
    configuration: Dict[str, Any]


class DiagnosticsManager:
    """Manager for advanced diagnostics and crash snapshots"""
    
    def __init__(self, snapshots_dir: Path = CRASH_SNAPSHOTS_DIR):
        self.snapshots_dir = Path(snapshots_dir)
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize system monitoring
        self._last_system_stats: Optional[SystemStats] = None
        
        logger.info(
            EventNames.APP_START,
            "Diagnostics manager initialized",
            snapshots_dir=str(self.snapshots_dir)
        )
    
    def get_system_stats(self) -> SystemStats:
        """Get current system statistics"""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=0.1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_available_mb = memory.available / (1024 * 1024)
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_usage_percent = (disk.used / disk.total) * 100
            
            # Process count
            process_count = len(psutil.pids())
            
            stats = SystemStats(
                cpu_percent=cpu_percent,
                memory_percent=memory_percent,
                memory_available_mb=memory_available_mb,
                disk_usage_percent=disk_usage_percent,
                process_count=process_count,
                timestamp=time.time()
            )
            
            self._last_system_stats = stats
            return stats
            
        except Exception as e:
            logger.error(
                EventNames.ORCH_ERROR,
                f"Failed to get system stats: {str(e)}",
                error=str(e)
            )
            # Return default stats
            return SystemStats(
                cpu_percent=0.0,
                memory_percent=0.0,
                memory_available_mb=0.0,
                disk_usage_percent=0.0,
                process_count=0,
                timestamp=time.time()
            )
    
    def extract_error_context(self, error: Exception) -> ErrorContext:
        """Extract error context from exception"""
        tb = traceback.extract_tb(error.__traceback__)
        
        if tb:
            # Get the last frame (where error occurred)
            frame = tb[-1]
            module = frame.filename
            function = frame.name
            line_number = frame.lineno
        else:
            module = "unknown"
            function = "unknown"
            line_number = 0
        
        return ErrorContext(
            error_type=type(error).__name__,
            error_message=str(error),
            traceback_str=traceback.format_exc(),
            timestamp=time.time(),
            module=module,
            function=function,
            line_number=line_number
        )
    
    def get_recent_logs(self, count: int = 100) -> List[Dict[str, Any]]:
        """Get recent logs for crash snapshot"""
        try:
            from ui.utils.logging_buffer import get_live_trace
            return list(get_live_trace("mlab", last_n=count))
        except Exception as e:
            logger.error(
                EventNames.ORCH_ERROR,
                f"Failed to get recent logs: {str(e)}",
                error=str(e)
            )
            return []
    
    def create_crash_snapshot(self, 
                            error: Exception,
                            run_id: Optional[str] = None,
                            exp_id: Optional[int] = None,
                            application_state: Optional[Dict[str, Any]] = None,
                            configuration: Optional[Dict[str, Any]] = None) -> str:
        """
        Create comprehensive crash snapshot
        
        Returns:
            str: Path to the crash snapshot file
        """
        try:
            # Generate snapshot ID
            timestamp = datetime.now()
            snapshot_id = f"crash_{timestamp.strftime('%Y%m%d_%H%M%S')}_{int(time.time() * 1000) % 10000:04d}"
            
            # Gather diagnostics data
            system_stats = self.get_system_stats()
            error_context = self.extract_error_context(error)
            recent_logs = self.get_recent_logs(200)
            
            # Create crash snapshot
            snapshot = CrashSnapshot(
                snapshot_id=snapshot_id,
                run_id=run_id,
                exp_id=exp_id,
                timestamp=time.time(),
                error_context=error_context,
                system_stats=system_stats,
                application_state=application_state or {},
                recent_logs=recent_logs,
                configuration=configuration or {}
            )
            
            # Save snapshot to file
            snapshot_path = self.snapshots_dir / f"{snapshot_id}.json"
            
            with open(snapshot_path, 'w') as f:
                json.dump(asdict(snapshot), f, indent=2, default=str)
            
            # Log crash snapshot creation
            logger.error(
                EventNames.ORCH_ERROR,
                f"Crash snapshot created: {snapshot_id}",
                snapshot_id=snapshot_id,
                snapshot_path=str(snapshot_path),
                error_type=error_context.error_type,
                error_message=error_context.error_message,
                run_id=run_id,
                exp_id=exp_id,
                system_cpu=system_stats.cpu_percent,
                system_memory=system_stats.memory_percent
            )
            
            return str(snapshot_path)
            
        except Exception as e:
            logger.error(
                EventNames.ORCH_ERROR,
                f"Failed to create crash snapshot: {str(e)}",
                error=str(e)
            )
            return ""
    
    def get_crash_snapshots(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get list of recent crash snapshots"""
        try:
            snapshots = []
            
            # Find all snapshot files
            for snapshot_file in sorted(self.snapshots_dir.glob("crash_*.json"), reverse=True):
                if len(snapshots) >= limit:
                    break
                
                try:
                    with open(snapshot_file, 'r') as f:
                        snapshot_data = json.load(f)
                    
                    # Add file info
                    snapshot_data['file_path'] = str(snapshot_file)
                    snapshot_data['file_size'] = snapshot_file.stat().st_size
                    
                    snapshots.append(snapshot_data)
                    
                except Exception as e:
                    logger.warning(
                        EventNames.ORCH_ERROR,
                        f"Failed to load snapshot {snapshot_file}: {str(e)}",
                        file_path=str(snapshot_file),
                        error=str(e)
                    )
            
            return snapshots
            
        except Exception as e:
            logger.error(
                EventNames.ORCH_ERROR,
                f"Failed to get crash snapshots: {str(e)}",
                error=str(e)
            )
            return []
    
    def analyze_crash_pattern(self, snapshots: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """Analyze crash patterns from snapshots"""
        if snapshots is None:
            snapshots = self.get_crash_snapshots(50)
        
        if not snapshots:
            return {"analysis": "No crash snapshots available"}
        
        # Analyze error types
        error_types = {}
        modules = {}
        functions = {}
        
        for snapshot in snapshots:
            error_context = snapshot.get('error_context', {})
            
            # Count error types
            error_type = error_context.get('error_type', 'Unknown')
            error_types[error_type] = error_types.get(error_type, 0) + 1
            
            # Count modules
            module = error_context.get('module', 'unknown')
            module_name = os.path.basename(module) if module != 'unknown' else 'unknown'
            modules[module_name] = modules.get(module_name, 0) + 1
            
            # Count functions
            function = error_context.get('function', 'unknown')
            functions[function] = functions.get(function, 0) + 1
        
        return {
            "total_crashes": len(snapshots),
            "most_common_errors": sorted(error_types.items(), key=lambda x: x[1], reverse=True)[:5],
            "most_problematic_modules": sorted(modules.items(), key=lambda x: x[1], reverse=True)[:5],
            "most_problematic_functions": sorted(functions.items(), key=lambda x: x[1], reverse=True)[:5],
            "analysis_timestamp": time.time()
        }
    
    def cleanup_old_snapshots(self, max_age_days: int = 30, max_count: int = 100):
        """Clean up old crash snapshots"""
        try:
            snapshot_files = list(self.snapshots_dir.glob("crash_*.json"))
            
            # Sort by modification time (newest first)
            snapshot_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
            
            current_time = time.time()
            max_age_seconds = max_age_days * 24 * 3600
            
            deleted_count = 0
            
            for i, snapshot_file in enumerate(snapshot_files):
                # Delete if too old or beyond max count
                file_age = current_time - snapshot_file.stat().st_mtime
                
                if file_age > max_age_seconds or i >= max_count:
                    try:
                        snapshot_file.unlink()
                        deleted_count += 1
                    except Exception as e:
                        logger.warning(
                            EventNames.ORCH_ERROR,
                            f"Failed to delete snapshot {snapshot_file}: {str(e)}",
                            file_path=str(snapshot_file),
                            error=str(e)
                        )
            
            if deleted_count > 0:
                logger.info(
                    EventNames.APP_STOP,
                    f"Cleaned up {deleted_count} old crash snapshots",
                    deleted_count=deleted_count,
                    max_age_days=max_age_days,
                    max_count=max_count
                )
                
        except Exception as e:
            logger.error(
                EventNames.ORCH_ERROR,
                f"Failed to cleanup old snapshots: {str(e)}",
                error=str(e)
            )
    
    def generate_diagnostics_report(self) -> Dict[str, Any]:
        """Generate comprehensive diagnostics report"""
        try:
            # Get system stats
            system_stats = self.get_system_stats()
            
            # Get crash analysis
            crash_analysis = self.analyze_crash_pattern()
            
            # Get recent logs summary
            recent_logs = self.get_recent_logs(50)
            log_events = {}
            for log in recent_logs:
                event = log.get('event', 'UNKNOWN')
                log_events[event] = log_events.get(event, 0) + 1
            
            # Check disk space in snapshots directory
            try:
                disk_usage = psutil.disk_usage(str(self.snapshots_dir))
                snapshots_disk_mb = sum(f.stat().st_size for f in self.snapshots_dir.glob("*.json")) / (1024 * 1024)
            except:
                snapshots_disk_mb = 0
            
            report = {
                "timestamp": time.time(),
                "system_stats": asdict(system_stats),
                "crash_analysis": crash_analysis,
                "recent_log_events": dict(sorted(log_events.items(), key=lambda x: x[1], reverse=True)[:10]),
                "snapshots_disk_usage_mb": snapshots_disk_mb,
                "snapshots_count": len(list(self.snapshots_dir.glob("crash_*.json"))),
                "health_status": self._assess_health_status(system_stats, crash_analysis)
            }
            
            return report
            
        except Exception as e:
            logger.error(
                EventNames.ORCH_ERROR,
                f"Failed to generate diagnostics report: {str(e)}",
                error=str(e)
            )
            return {"error": str(e), "timestamp": time.time()}
    
    def _assess_health_status(self, system_stats: SystemStats, crash_analysis: Dict[str, Any]) -> str:
        """Assess overall system health status"""
        issues = []
        
        # Check system resources
        if system_stats.cpu_percent > 90:
            issues.append("high_cpu")
        if system_stats.memory_percent > 90:
            issues.append("high_memory")
        if system_stats.disk_usage_percent > 90:
            issues.append("low_disk")
        
        # Check crash frequency
        total_crashes = crash_analysis.get("total_crashes", 0)
        if total_crashes > 10:
            issues.append("frequent_crashes")
        elif total_crashes > 5:
            issues.append("moderate_crashes")
        
        # Determine overall status
        if not issues:
            return "healthy"
        elif len(issues) == 1 and issues[0] in ["moderate_crashes"]:
            return "warning"
        else:
            return "critical"


# Global diagnostics manager instance
_diagnostics_manager: Optional[DiagnosticsManager] = None


def get_diagnostics_manager() -> DiagnosticsManager:
    """Get or create global diagnostics manager"""
    global _diagnostics_manager
    if _diagnostics_manager is None:
        _diagnostics_manager = DiagnosticsManager()
    return _diagnostics_manager


def create_crash_snapshot(error: Exception, 
                         run_id: Optional[str] = None,
                         exp_id: Optional[int] = None,
                         application_state: Optional[Dict[str, Any]] = None,
                         configuration: Optional[Dict[str, Any]] = None) -> str:
    """Convenience function to create crash snapshot"""
    manager = get_diagnostics_manager()
    return manager.create_crash_snapshot(error, run_id, exp_id, application_state, configuration)


def get_recent_crash_snapshots(limit: int = 10) -> List[Dict[str, Any]]:
    """Convenience function to get recent crash snapshots"""
    manager = get_diagnostics_manager()
    return manager.get_crash_snapshots(limit)


def generate_diagnostics_report() -> Dict[str, Any]:
    """Convenience function to generate diagnostics report"""
    manager = get_diagnostics_manager()
    return manager.generate_diagnostics_report()