"""
Bridge module between Streamlit UI and background optimization runner.
Starts/stops background orchestrator, exposes validation and status helpers.
"""
import sys
import os
import json
import time
import logging
from typing import Dict, Any, Optional, Callable
from pathlib import Path
from uuid import uuid4
from datetime import datetime
from threading import Thread, Event

from ui.utils.constants import DB_PATH
from ui.utils.logging_buffer import ensure_ring_handler
from ui.utils.orchestrator_runner import BackgroundOrchestrator, RunnerConfig

logger = logging.getLogger("mlab")
ensure_ring_handler("mlab")


class OptimizationBridge:
    """Bridge owning the lifecycle of a background optimization job."""
    
    def __init__(self):
        self._thread: Optional[Thread] = None
        self._stop_event: Optional[Event] = None
        self._run_id: Optional[str] = None
        self._last_error: Optional[str] = None
        self.progress_callback: Optional[Callable] = None
    
    def set_progress_callback(self, callback: Callable):
        """Set callback function for progress updates."""
        self.progress_callback = callback
    
    def create_optimization_session(self, parameters: Dict[str, Any], 
                                  max_iterations: int = 1000,
                                  time_limit: float = 300.0) -> Dict[str, Any]:
        """For compatibility with existing UI. No-op that returns a dummy session id."""
        sid = f"session_{int(time.time())}"
        return {'success': True, 'session_id': sid, 'message': 'Session stub created'}
    
    def start_optimization(self, parameters: Optional[Dict[str, Any]] = None, db_path: str = DB_PATH) -> Dict[str, Any]:
        """Start background orchestrator and return run id."""
        if self._thread and self._thread.is_alive():
            return {'success': False, 'error': 'Already running', 'message': 'A job is already running'}
        try:
            params = parameters or {}
            self._run_id = datetime.utcnow().strftime("%Y%m%d-%H%M%S") + "-" + uuid4().hex[:6]
            self._stop_event = Event()
            self._last_error = None

            cfg = RunnerConfig(
                min_overlap=float(params.get('min_overlap', 1.0)),
                max_overlap=float(params.get('max_overlap', 10.0)),
                min_order=int(params.get('min_order', 3)),
                max_order=int(params.get('max_order', 8)),
                db_path=db_path,
            )
            runner = BackgroundOrchestrator(cfg, stop_event=self._stop_event)

            def _target():
                logger.info("THREAD.START run_id=%s", self._run_id)
                try:
                    runner.run(run_id=self._run_id)
                except Exception:
                    import traceback
                    self._last_error = traceback.format_exc()
                finally:
                    logger.info("THREAD.END run_id=%s", self._run_id)

            self._thread = Thread(target=_target, daemon=True)
            self._thread.start()
            return {'success': True, 'run_id': self._run_id}
        except Exception as e:
            logger.exception("start_optimization failed")
            return {'success': False, 'error': str(e)}
    
    def stop_optimization(self) -> Dict[str, Any]:
        """Signal stop and wait for the background thread to finish."""
        if not self._thread:
            return {'success': False, 'error': 'No job running'}
        try:
            assert self._stop_event is not None
            self._stop_event.set()
            self._thread.join(timeout=10.0)
            stopped = not self._thread.is_alive()
            return {'success': True, 'stopped': stopped}
        except Exception as e:
            logger.exception("stop_optimization failed")
            return {'success': False, 'error': str(e)}
    
    def get_optimization_status(self) -> Dict[str, Any]:
        """Report background thread status."""
        try:
            status = 'idle'
            if self._thread:
                if self._last_error is not None:
                    status = 'error'
                else:
                    status = 'running' if self._thread.is_alive() else 'completed'
            data = {'status': status, 'run_id': self._run_id}
            if self._last_error is not None:
                data['error'] = self._last_error
            return {'success': True, 'data': data}
        except Exception as e:
            logger.exception("get_optimization_status failed")
            return {'success': False, 'error': str(e)}
    
    def get_results(self) -> Dict[str, Any]:
        """Results are persisted in DB; this returns status only."""
        try:
            completed = self._thread is not None and not self._thread.is_alive()
            return {'success': True, 'results': {}, 'statistics': {}, 'completed': completed}
        except Exception as e:
            logger.exception("get_results failed")
            return {'success': False, 'error': str(e)}
    
    def cleanup_session(self) -> Dict[str, Any]:
        self._thread = None
        self._stop_event = None
        self._run_id = None
        return {'success': True}
    
    def validate_parameters(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Validate optimization parameters."""
        try:
            # Basic validation
            if parameters['min_overlap'] >= parameters['max_overlap']:
                return {
                    'success': False,
                    'error': 'Min overlap must be less than max overlap',
                    'message': 'Invalid overlap parameters'
                }
            
            if parameters['min_order'] >= parameters['max_order']:
                return {
                    'success': False,
                    'error': 'Min order must be less than max order',
                    'message': 'Invalid order parameters'
                }
            
            if not (0 <= parameters['min_overlap'] <= 100):
                return {
                    'success': False,
                    'error': 'Min overlap must be between 0 and 100',
                    'message': 'Invalid min overlap value'
                }
            
            if not (0 <= parameters['max_overlap'] <= 100):
                return {
                    'success': False,
                    'error': 'Max overlap must be between 0 and 100',
                    'message': 'Invalid max overlap value'
                }
            
            if not (1 <= parameters['min_order'] <= 50):
                return {
                    'success': False,
                    'error': 'Min order must be between 1 and 50',
                    'message': 'Invalid min order value'
                }
            
            if not (1 <= parameters['max_order'] <= 50):
                return {
                    'success': False,
                    'error': 'Max order must be between 1 and 50',
                    'message': 'Invalid max order value'
                }
            
            return {
                'success': True,
                'message': 'Parameters validated successfully'
            }
            
        except KeyError as e:
            return {
                'success': False,
                'error': f'Missing parameter: {e}',
                'message': 'Required parameter missing'
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'message': 'Parameter validation failed'
            }


# Global bridge instance
optimization_bridge = OptimizationBridge()
