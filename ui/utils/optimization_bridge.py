"""
Optimization Bridge for UI Integration
Provides background threading and proper logging for DCA optimization.
"""
from __future__ import annotations

import threading
import time
from typing import Dict, Any, Optional, Callable
from concurrent.futures import ThreadPoolExecutor

from martingale_lab.orchestrator.dca_orchestrator import DCAOrchestrator, DCAConfig
from martingale_lab.storage.experiments_store import ExperimentsStore
from ui.utils.structured_logging import Events, ui_logger, LogContext, generate_run_id
from ui.utils.constants import DB_PATH, Status


class OptimizationBridge:
    """Bridge between UI and optimization backend."""
    
    def __init__(self):
        """Initialize optimization bridge."""
        self.current_run_id: Optional[str] = None
        self.current_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        self.is_running = False
        
    def start_optimization(self, params: Dict[str, Any], db_path: str = DB_PATH) -> Dict[str, Any]:
        """
        Start optimization in background thread.
        
        Args:
            params: Optimization parameters
            db_path: Database path
            
        Returns:
            Dict with success status and run_id
        """
        if self.is_running:
            return {"success": False, "error": "Optimization already running"}
        
        try:
            # Generate run ID
            run_id = generate_run_id()
            self.current_run_id = run_id
            
            # Log UI start event
            ui_logger.event(
                Events.UI_CLICK_START,
                run_id=run_id,
                overlap_range=f"{params.get('overlap_min', 0)}-{params.get('overlap_max', 0)}",
                orders_range=f"{params.get('orders_min', 0)}-{params.get('orders_max', 0)}",
                wave_pattern=params.get('wave_pattern', False)
            )
            
            # Create configuration
            config = DCAConfig(
                overlap_min=params.get('overlap_min', 10.0),
                overlap_max=params.get('overlap_max', 30.0),
                orders_min=params.get('orders_min', 5),
                orders_max=params.get('orders_max', 15),
                alpha=params.get('alpha', 0.5),
                beta=params.get('beta', 0.3),
                gamma=params.get('gamma', 0.2),
                lambda_penalty=params.get('lambda_penalty', 0.1),
                wave_pattern=params.get('wave_pattern', True),
                tail_cap=params.get('tail_cap', 0.40),
                n_candidates_per_batch=params.get('n_candidates_per_batch', 1000),
                max_batches=params.get('max_batches', 100),
                n_workers=params.get('n_workers', 4),
                random_seed=params.get('random_seed', None)
            )
            
            # Start optimization in background thread
            def run_optimization():
                try:
                    LogContext.set_run_id(run_id)
                    
                    store = ExperimentsStore(db_path)
                    orchestrator = DCAOrchestrator(config, store, run_id)
                    
                    # Run optimization
                    results = orchestrator.run_optimization(notes=params.get('notes', ''))
                    
                    self.is_running = False
                    
                except Exception as e:
                    ui_logger.event(
                        Events.ORCH_ERROR,
                        run_id=run_id,
                        error=str(e)
                    )
                    self.is_running = False
            
            self.current_thread = threading.Thread(target=run_optimization, daemon=True)
            self.current_thread.start()
            self.is_running = True
            
            return {"success": True, "run_id": run_id}
            
        except Exception as e:
            ui_logger.event(
                Events.UI_CLICK_START,
                error=str(e),
                status="failed"
            )
            return {"success": False, "error": str(e)}
    
    def stop_optimization(self) -> Dict[str, Any]:
        """
        Stop current optimization.
        
        Returns:
            Dict with success status
        """
        try:
            if not self.is_running:
                return {"success": False, "error": "No optimization running"}
            
            # Set stop event
            self.stop_event.set()
            
            # Wait for thread to finish (with timeout)
            if self.current_thread and self.current_thread.is_alive():
                self.current_thread.join(timeout=5.0)
            
            self.is_running = False
            
            ui_logger.event(
                Events.UI_CLICK_STOP,
                run_id=self.current_run_id,
                status="stopped"
            )
            
            return {"success": True}
            
        except Exception as e:
            ui_logger.event(
                Events.UI_CLICK_STOP,
                error=str(e),
                status="failed"
            )
            return {"success": False, "error": str(e)}
    
    def get_optimization_status(self) -> Dict[str, Any]:
        """
        Get current optimization status.
        
        Returns:
            Dict with status information
        """
        if not self.is_running:
            return {"success": True, "data": {"status": "completed"}}
        
        return {"success": True, "data": {"status": "running", "run_id": self.current_run_id}}
    
    def validate_parameters(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate optimization parameters.
        
        Args:
            params: Parameters to validate
            
        Returns:
            Dict with validation result
        """
        try:
            # Basic validation
            overlap_min = params.get('overlap_min', 0)
            overlap_max = params.get('overlap_max', 0)
            orders_min = params.get('orders_min', 0)
            orders_max = params.get('orders_max', 0)
            
            if overlap_min >= overlap_max:
                return {"success": False, "error": "overlap_min must be < overlap_max"}
            
            if orders_min >= orders_max:
                return {"success": False, "error": "orders_min must be < orders_max"}
            
            if overlap_min <= 0 or overlap_max > 100:
                return {"success": False, "error": "Overlap percentages must be in (0, 100]"}
            
            if orders_min < 2 or orders_max > 50:
                return {"success": False, "error": "Orders must be in [2, 50]"}
            
            # Weight validation
            alpha = params.get('alpha', 0.5)
            beta = params.get('beta', 0.3)
            gamma = params.get('gamma', 0.2)
            
            if not (0 <= alpha <= 1 and 0 <= beta <= 1 and 0 <= gamma <= 1):
                return {"success": False, "error": "Weights must be in [0, 1]"}
            
            return {"success": True}
            
        except Exception as e:
            return {"success": False, "error": str(e)}


# Global bridge instance
optimization_bridge = OptimizationBridge()
