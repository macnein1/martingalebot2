"""
Optimization Bridge for Background Thread Control and Live Trace Streaming
Provides start/stop optimization control with structured logging integration
"""
import threading
import time
from typing import Dict, Any, Optional, Callable
from dataclasses import asdict
import traceback

from martingale_lab.utils.structured_logging import (
    get_structured_logger, EventNames, generate_run_id
)
from martingale_lab.orchestrator.adaptive_orchestrator import AdaptiveOrchestrator, OrchConfig
from martingale_lab.storage.experiments_store import ExperimentsStore
from ui.utils.logging_buffer import get_live_trace
from ui.utils.constants import DB_PATH, Status
from ui.utils.config import make_auto_config, get_system_info

# Initialize structured logger for optimization bridge
logger = get_structured_logger("mlab.bridge")


class OptimizationBridge:
    """
    Bridge for controlling optimization in background threads with live trace streaming
    """
    
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self.current_thread: Optional[threading.Thread] = None
        self.current_orchestrator: Optional[AdaptiveOrchestrator] = None
        self.stop_event = threading.Event()
        self.is_running = False
        self.current_run_id: Optional[str] = None
        self.progress_callback: Optional[Callable] = None
        self._start_time: Optional[float] = None
        
    def _orch_config_from_params(self, run_id: str, params: Dict[str, Any]) -> OrchConfig:
        """Build OrchConfig from either auto or manual params."""
        use_auto = str(params.get('config_mode', '')).lower().startswith('auto') or params.get('auto', False)
        if use_auto:
            search_space = params.get('search_space') or {
                'overlap_min': params.get('overlap_min', 10.0),
                'overlap_max': params.get('overlap_max', 30.0),
                'orders_min': params.get('orders_min', 5),
                'orders_max': params.get('orders_max', 15),
            }
            sys_info = params.get('sys_info') or get_system_info()
            dca_cfg = make_auto_config(search_space, sys_info)
            # Map DCAConfig -> OrchConfig
            return OrchConfig(
                run_id=run_id,
                overlap_min=dca_cfg.overlap_min,
                overlap_max=dca_cfg.overlap_max,
                orders_min=dca_cfg.orders_min,
                orders_max=dca_cfg.orders_max,
                alpha=dca_cfg.alpha,
                beta=dca_cfg.beta,
                gamma=dca_cfg.gamma,
                lambda_penalty=dca_cfg.lambda_penalty,
                wave_pattern=dca_cfg.wave_pattern,
                tail_cap=dca_cfg.tail_cap,
                min_indent_step=dca_cfg.min_indent_step,
                softmax_temp=dca_cfg.softmax_temp,
                batch_size=dca_cfg.n_candidates_per_batch,
                max_batches=dca_cfg.max_batches,
                patience=dca_cfg.early_stop_patience,
                top_k=min(50, int(params.get('top_k', 50)))
            )
        # Manual config mapping
        return OrchConfig(
            run_id=run_id,
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
            min_indent_step=params.get('min_indent_step', 0.05),
            softmax_temp=params.get('softmax_temp', 1.0),
            batch_size=params.get('batch_size', params.get('n_candidates_per_batch', 100)),
            max_batches=params.get('max_batches', 50),
            patience=params.get('patience', params.get('early_stop_patience', 5)),
            top_k=params.get('top_k', 50),
            base_price=params.get('base_price', 100.0)
        )
        
    def start_optimization(self, params: Dict[str, Any], 
                          progress_callback: Optional[Callable] = None) -> str:
        """
        Start optimization in background thread
        
        Steps:
        1) Build AutoConfig from search_space + sys_info if Auto selected
        2) Run AdaptiveOrchestrator in background
        3) Provide live logs via tail_logs('mlab') through get_live_logs()
        4) Enrich progress with batches, rows inserted, best_score, evals/sec
        5) On completion, ensure experiments row best_score/total_evals updated
        """
        if self.is_running:
            raise RuntimeError("Optimization is already running")
        
        # Generate run ID
        run_id = generate_run_id()
        self.current_run_id = run_id
        self.progress_callback = progress_callback
        self._start_time = time.time()
        
        # Build orchestrator config
        config = self._orch_config_from_params(run_id, params)
        
        # Log UI click start
        logger.info(
            EventNames.UI_CLICK_START,
            f"Starting optimization with run_id {run_id}",
            run_id=run_id,
            config_snapshot=asdict(config)
        )
        
        # Create orchestrator
        self.current_orchestrator = AdaptiveOrchestrator(config, self.db_path)
        
        # Reset stop event
        self.stop_event.clear()
        
        # Start background thread
        self.current_thread = threading.Thread(
            target=self._run_optimization_thread,
            args=(self.current_orchestrator,),
            daemon=True
        )
        self.is_running = True
        self.current_thread.start()
        
        return run_id
    
    def stop_optimization(self) -> bool:
        """
        Stop current optimization gracefully
        
        Returns:
            bool: True if stop signal was sent successfully
        """
        if not self.is_running or not self.current_orchestrator:
            return False
        
        # Log UI click stop
        logger.info(
            EventNames.UI_CLICK_STOP,
            "Stop signal sent by user",
            run_id=self.current_run_id
        )
        
        # Signal orchestrator to stop
        self.current_orchestrator.stop()
        self.stop_event.set()
        
        # Wait for thread to finish (with timeout)
        if self.current_thread:
            self.current_thread.join(timeout=10.0)
            if self.current_thread.is_alive():
                logger.warning(
                    EventNames.UI_CLICK_STOP,
                    "Thread did not stop gracefully within timeout",
                    run_id=self.current_run_id
                )
        
        # Log app stop
        logger.info(
            EventNames.APP_STOP,
            "Optimization stopped by user",
            run_id=self.current_run_id
        )
        
        self.is_running = False
        self.current_thread = None
        self.current_orchestrator = None
        
        return True
    
    def _run_optimization_thread(self, orchestrator: AdaptiveOrchestrator):
        """
        Run optimization in background thread with error handling
        """
        try:
            # Define progress callback wrapper
            def progress_wrapper(progress_data: Dict[str, Any]):
                # Enrich with evals/sec and rows inserted
                try:
                    now = time.time()
                    elapsed = max(1e-6, (now - (self._start_time or now)))
                    eval_count = int(progress_data.get('eval_count', 0))
                    batch_stats = progress_data.get('batch_stats', {}) or {}
                    rows_inserted = int(batch_stats.get('evaluated', 0))
                    enriched = {
                        'batch_idx': progress_data.get('batch_idx', 0),
                        'batches_total': orchestrator.config.max_batches,
                        'best_score': progress_data.get('best_score'),
                        'eval_count': eval_count,
                        'rows_inserted': rows_inserted,
                        'evals_per_sec': eval_count / elapsed,
                    }
                    if self.progress_callback:
                        self.progress_callback(enriched)
                except Exception as e:
                    logger.error(
                        EventNames.ORCH_ERROR,
                        f"Progress enrichment error: {str(e)}",
                        run_id=self.current_run_id,
                        error=str(e)
                    )
            
            # Run optimization
            result = orchestrator.run_optimization(progress_callback=progress_wrapper)
            
            # Ensure DB row has final stats (best_score, total_evals)
            try:
                store = ExperimentsStore(self.db_path)
                store.update_experiment_summary(
                    orchestrator.exp_id, result.get('best_score', float('inf')), result.get('total_evals', 0), result.get('elapsed_s', 0.0)
                )
            except Exception:
                pass
            
            # Log successful completion
            logger.info(
                EventNames.ORCH_DONE,
                f"Background optimization completed successfully",
                run_id=self.current_run_id,
                exp_id=result.get('exp_id'),
                best_score=result.get('best_score'),
                total_evals=result.get('total_evals')
            )
            
        except Exception as e:
            # Log error
            logger.error(
                EventNames.ORCH_ERROR,
                f"Background optimization failed: {str(e)}",
                run_id=self.current_run_id,
                error=str(e),
                traceback=traceback.format_exc()
            )
        
        finally:
            # Clean up
            self.is_running = False
            self.current_thread = None
            self.current_orchestrator = None
    
    def get_status(self) -> Dict[str, Any]:
        """Get current optimization status"""
        return {
            'is_running': self.is_running,
            'run_id': self.current_run_id,
            'thread_alive': self.current_thread.is_alive() if self.current_thread else False,
            'has_orchestrator': self.current_orchestrator is not None
        }
    
    def get_live_logs(self, event_filter: Optional[str] = None, 
                     last_n: int = 50) -> list:
        """Get live trace logs for UI display"""
        return get_live_trace("mlab", event_filter=event_filter, last_n=last_n)
    
    def get_orchestrator_logs(self, last_n: int = 20) -> list:
        """Get orchestrator-specific logs"""
        return self.get_live_logs(event_filter="ORCH", last_n=last_n)
    
    def get_evaluation_logs(self, last_n: int = 10) -> list:
        """Get evaluation-specific logs"""
        return self.get_live_logs(event_filter="EVAL", last_n=last_n)
    
    def get_database_logs(self, last_n: int = 10) -> list:
        """Get database-specific logs"""
        return self.get_live_logs(event_filter="DB", last_n=last_n)


# Global bridge instance
_bridge_instance: Optional[OptimizationBridge] = None


def get_optimization_bridge(db_path: str = DB_PATH) -> OptimizationBridge:
    """Get or create global optimization bridge instance"""
    global _bridge_instance
    if _bridge_instance is None:
        _bridge_instance = OptimizationBridge(db_path)
    return _bridge_instance


def start_optimization(params: Dict[str, Any], db_path: str = DB_PATH,
                      progress_callback: Optional[Callable] = None) -> str:
    """Convenience function to start optimization"""
    bridge = get_optimization_bridge(db_path)
    return bridge.start_optimization(params, progress_callback)


def stop_optimization() -> bool:
    """Convenience function to stop optimization"""
    bridge = get_optimization_bridge()
    return bridge.stop_optimization()


def get_optimization_status() -> Dict[str, Any]:
    """Convenience function to get optimization status"""
    bridge = get_optimization_bridge()
    return bridge.get_status()


def get_live_trace_logs(event_filter: Optional[str] = None, last_n: int = 50) -> list:
    """Convenience function to get live trace logs"""
    bridge = get_optimization_bridge()
    return bridge.get_live_logs(event_filter, last_n)
