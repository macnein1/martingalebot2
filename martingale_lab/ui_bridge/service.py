"""
Service layer for UI-Martingale Lab communication.
"""
import asyncio
import threading
import time
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass
from datetime import datetime

from ..core.types import Params, Schedule, ScoreBreakdown
from ..adapters.numba_adapter import NumbaAdapter, NumbaAdapterConfig
from ..optimizer.evaluation_engine import run_experiment, ExperimentConfig
from .payloads import UIPayloadConverter, UIResponseBuilder, UIPayloadValidator


@dataclass
class OptimizationConfig:
    """Configuration for optimization runs."""
    params: Params
    max_iterations: int
    time_limit: float
    batch_size: int = 100
    early_stopping: bool = True
    progress_callback: Optional[Callable] = None


@dataclass
class OptimizationSession:
    """Represents an active optimization session."""
    session_id: str
    config: OptimizationConfig
    start_time: datetime
    status: str = "initialized"
    results: List[tuple] = None
    statistics: Dict[str, Any] = None
    error: Optional[Exception] = None
    
    def __post_init__(self):
        if self.results is None:
            self.results = []
        if self.statistics is None:
            self.statistics = {}


class OptimizationService:
    """Service for managing optimization sessions and UI communication."""
    
    def __init__(self):
        self.active_sessions: Dict[str, OptimizationSession] = {}
        self._session_counter = 0
    
    def create_session(self, config: OptimizationConfig) -> str:
        """Create a new optimization session."""
        session_id = f"session_{self._session_counter}_{int(time.time())}"
        self._session_counter += 1
        
        session = OptimizationSession(
            session_id=session_id,
            config=config,
            start_time=datetime.now()
        )
        
        self.active_sessions[session_id] = session
        return session_id
    
    def start_optimization(self, session_id: str) -> Dict[str, Any]:
        """Start optimization for a session."""
        if session_id not in self.active_sessions:
            return UIResponseBuilder.error_response(
                ValueError("Session not found"), 
                "Invalid session ID"
            )
        
        session = self.active_sessions[session_id]
        session.status = "running"
        
        # Start optimization in a separate thread
        thread = threading.Thread(
            target=self._run_optimization,
            args=(session_id,)
        )
        thread.daemon = True
        thread.start()
        
        return UIResponseBuilder.optimization_start_response(
            session_id, 
            session.config.max_iterations
        )
    
    def _run_optimization(self, session_id: str):
        """Run optimization in background thread."""
        session = self.active_sessions[session_id]
        
        try:
            # Create experiment configuration
            experiment_config = ExperimentConfig(
                base_price=1.0,
                overlap_min=session.config.params.min_overlap,
                overlap_max=session.config.params.max_overlap,
                orders_min=session.config.params.min_order,
                orders_max=session.config.params.max_order,
                n_candidates_per_M=session.config.batch_size,
                seed=42,
                top_k_global=100
            )
            
            # Run optimization with progress tracking
            start_time = time.time()
            
            # Run the experiment
            experiment_result = run_experiment(experiment_config)
            
            # Convert results to the expected format
            results = []
            if experiment_result['top']:
                for result in experiment_result['top']:
                    # Create a tuple format: (score, params, breakdown)
                    params = Params(
                        min_overlap=result.get('overlap_pct', 0),
                        max_overlap=result.get('overlap_pct', 0),
                        min_order=result.get('orders', 0),
                        max_order=result.get('orders', 0),
                        risk_factor=session.config.params.risk_factor,
                        smoothing_factor=session.config.params.smoothing_factor,
                        tail_weight=session.config.params.tail_weight
                    )
                    
                    breakdown = ScoreBreakdown(
                        total_score=result.get('score', 0),
                        max_score=result.get('max_need', 0),
                        variance_score=result.get('var_need', 0),
                        tail_score=result.get('tail', 0),
                        gini_penalty=0.0,
                        entropy_penalty=0.0,
                        monotone_penalty=0.0,
                        smoothness_penalty=0.0
                    )
                    
                    results.append((result.get('score', 0), params, breakdown))
            
            # Calculate statistics
            elapsed_time = time.time() - start_time
            session.statistics = {
                'total_evaluations': len(results),
                'total_time': elapsed_time,
                'average_time_per_evaluation': elapsed_time / len(results) if results else 0.0,
                'evaluations_per_second': len(results) / elapsed_time if elapsed_time > 0 else 0.0,
                'best_score': max(r[0] for r in results) if results else 0.0,
                'elapsed_time': elapsed_time
            }
            
            session.results = results
            session.status = "completed"
            
        except Exception as e:
            session.error = e
            session.status = "error"
    
    def stop_optimization(self, session_id: str) -> Dict[str, Any]:
        """Stop an active optimization session."""
        if session_id not in self.active_sessions:
            return UIResponseBuilder.error_response(
                ValueError("Session not found"), 
                "Invalid session ID"
            )
        
        session = self.active_sessions[session_id]
        session.status = "stopped"
        
        return UIResponseBuilder.success_response(
            {"session_id": session_id, "status": "stopped"},
            "Optimization stopped successfully"
        )
    
    def get_session_status(self, session_id: str) -> Dict[str, Any]:
        """Get status of an optimization session."""
        if session_id not in self.active_sessions:
            return UIResponseBuilder.error_response(
                ValueError("Session not found"), 
                "Invalid session ID"
            )
        
        session = self.active_sessions[session_id]
        
        response_data = {
            'session_id': session_id,
            'status': session.status,
            'start_time': session.start_time.isoformat(),
            'elapsed_time': (datetime.now() - session.start_time).total_seconds()
        }
        
        if session.results:
            response_data['results'] = UIPayloadConverter.batch_results_to_json(session.results)
        
        if session.statistics:
            response_data['statistics'] = UIPayloadConverter.statistics_to_json(session.statistics)
        
        if session.error:
            response_data['error'] = UIPayloadConverter.error_to_json(session.error)
        
        return UIResponseBuilder.success_response(response_data)
    
    def get_all_sessions(self) -> Dict[str, Any]:
        """Get all active sessions."""
        sessions_data = {}
        for session_id, session in self.active_sessions.items():
            sessions_data[session_id] = {
                'status': session.status,
                'start_time': session.start_time.isoformat(),
                'elapsed_time': (datetime.now() - session.start_time).total_seconds(),
                'results_count': len(session.results) if session.results else 0
            }
        
        return UIResponseBuilder.success_response(sessions_data)
    
    def cleanup_session(self, session_id: str) -> Dict[str, Any]:
        """Clean up a completed session."""
        if session_id not in self.active_sessions:
            return UIResponseBuilder.error_response(
                ValueError("Session not found"), 
                "Invalid session ID"
            )
        
        session = self.active_sessions[session_id]
        if session.status in ["running", "initialized"]:
            return UIResponseBuilder.error_response(
                ValueError("Cannot cleanup active session"), 
                "Session is still active"
            )
        
        del self.active_sessions[session_id]
        
        return UIResponseBuilder.success_response(
            {"session_id": session_id},
            "Session cleaned up successfully"
        )
    
    def validate_ui_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Validate UI payload and return validation result."""
        if not UIPayloadValidator.validate_optimization_config(payload):
            return UIResponseBuilder.error_response(
                ValueError("Invalid payload format"),
                "Payload validation failed"
            )
        
        return UIResponseBuilder.success_response(
            {"valid": True},
            "Payload validation successful"
        )


# Global service instance
optimization_service = OptimizationService()
