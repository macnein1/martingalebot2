"""
Payload converters for UI communication.
"""
import json
import numpy as np
from typing import Dict, Any, List, Optional
from datetime import datetime

from ..core.types import Params, Schedule, ScoreBreakdown


class UIPayloadConverter:
    """Convert optimization models to UI-friendly JSON payloads."""
    
    @staticmethod
    def params_to_json(params: Params) -> Dict[str, Any]:
        """Convert Params to JSON-serializable dict."""
        return {
            'min_overlap': float(params.min_overlap),
            'max_overlap': float(params.max_overlap),
            'min_order': int(params.min_order),
            'max_order': int(params.max_order),
            'risk_factor': float(params.risk_factor),
            'smoothing_factor': float(params.smoothing_factor),
            'tail_weight': float(params.tail_weight)
        }
    
    @staticmethod
    def schedule_to_json(schedule: Schedule) -> Dict[str, Any]:
        """Convert Schedule to JSON-serializable dict."""
        return {
            'orders': schedule.orders.tolist(),
            'volumes': schedule.volumes.tolist(),
            'overlaps': schedule.overlaps.tolist(),
            'num_levels': schedule.num_levels,
            'total_volume': float(schedule.total_volume()),
            'max_exposure': float(schedule.max_exposure())
        }
    
    @staticmethod
    def score_breakdown_to_json(breakdown: ScoreBreakdown) -> Dict[str, Any]:
        """Convert ScoreBreakdown to JSON-serializable dict."""
        return {
            'total_score': float(breakdown.total_score),
            'max_score': float(breakdown.max_score),
            'variance_score': float(breakdown.variance_score),
            'tail_score': float(breakdown.tail_score),
            'gini_penalty': float(breakdown.gini_penalty),
            'entropy_penalty': float(breakdown.entropy_penalty),
            'monotone_penalty': float(breakdown.monotone_penalty),
            'smoothness_penalty': float(breakdown.smoothness_penalty),
            'penalty_total': float(breakdown.penalty_total),
            'base_score': float(breakdown.base_score)
        }
    
    @staticmethod
    def optimization_result_to_json(score: float, params: Params, 
                                  breakdown: ScoreBreakdown, 
                                  schedule: Optional[Schedule] = None) -> Dict[str, Any]:
        """Convert optimization result to JSON payload."""
        return {
            'score': float(score),
            'params': UIPayloadConverter.params_to_json(params),
            'breakdown': UIPayloadConverter.score_breakdown_to_json(breakdown),
            'schedule': UIPayloadConverter.schedule_to_json(schedule) if schedule else None,
            'timestamp': datetime.now().isoformat()
        }
    
    @staticmethod
    def batch_results_to_json(results: List[tuple]) -> Dict[str, Any]:
        """Convert batch results to JSON payload."""
        converted_results = []
        for score, params, breakdown in results:
            converted_results.append(
                UIPayloadConverter.optimization_result_to_json(score, params, breakdown)
            )
        
        return {
            'results': converted_results,
            'count': len(converted_results),
            'timestamp': datetime.now().isoformat()
        }
    
    @staticmethod
    def progress_update_to_json(iteration: int, best_score: float, 
                              current_score: float, total_iterations: int) -> Dict[str, Any]:
        """Convert progress update to JSON payload."""
        progress_percentage = (iteration / total_iterations * 100) if total_iterations > 0 else 0
        
        return {
            'iteration': int(iteration),
            'best_score': float(best_score),
            'current_score': float(current_score),
            'total_iterations': int(total_iterations),
            'progress_percentage': float(progress_percentage),
            'timestamp': datetime.now().isoformat()
        }
    
    @staticmethod
    def statistics_to_json(stats: Dict[str, Any]) -> Dict[str, Any]:
        """Convert statistics to JSON payload."""
        return {
            'total_evaluations': int(stats.get('total_evaluations', 0)),
            'total_time': float(stats.get('total_time', 0.0)),
            'average_time_per_evaluation': float(stats.get('average_time_per_evaluation', 0.0)),
            'evaluations_per_second': float(stats.get('evaluations_per_second', 0.0)),
            'best_score': float(stats.get('best_score', 0.0)),
            'elapsed_time': float(stats.get('elapsed_time', 0.0)),
            'remaining_time': float(stats.get('remaining_time', 0.0)) if stats.get('remaining_time') else None,
            'remaining_evaluations': int(stats.get('remaining_evaluations', 0)) if stats.get('remaining_evaluations') else None,
            'timestamp': datetime.now().isoformat()
        }
    
    @staticmethod
    def error_to_json(error: Exception, context: str = "") -> Dict[str, Any]:
        """Convert error to JSON payload."""
        return {
            'error': True,
            'error_type': type(error).__name__,
            'error_message': str(error),
            'context': context,
            'timestamp': datetime.now().isoformat()
        }


class UIResponseBuilder:
    """Build structured UI responses."""
    
    @staticmethod
    def success_response(data: Any, message: str = "Success") -> Dict[str, Any]:
        """Build success response."""
        return {
            'success': True,
            'message': message,
            'data': data,
            'timestamp': datetime.now().isoformat()
        }
    
    @staticmethod
    def error_response(error: Exception, message: str = "Error occurred") -> Dict[str, Any]:
        """Build error response."""
        return {
            'success': False,
            'message': message,
            'error': UIPayloadConverter.error_to_json(error),
            'timestamp': datetime.now().isoformat()
        }
    
    @staticmethod
    def optimization_start_response(session_id: str, total_iterations: int) -> Dict[str, Any]:
        """Build optimization start response."""
        return UIResponseBuilder.success_response({
            'session_id': session_id,
            'total_iterations': total_iterations,
            'status': 'started'
        }, "Optimization started successfully")
    
    @staticmethod
    def optimization_complete_response(results: List[tuple], 
                                     statistics: Dict[str, Any]) -> Dict[str, Any]:
        """Build optimization complete response."""
        return UIResponseBuilder.success_response({
            'results': UIPayloadConverter.batch_results_to_json(results),
            'statistics': UIPayloadConverter.statistics_to_json(statistics),
            'status': 'completed'
        }, "Optimization completed successfully")
    
    @staticmethod
    def progress_response(progress_data: Dict[str, Any]) -> Dict[str, Any]:
        """Build progress response."""
        return UIResponseBuilder.success_response({
            'progress': UIPayloadConverter.progress_update_to_json(**progress_data),
            'status': 'in_progress'
        }, "Progress update")


class UIDataFormatter:
    """Format data for UI display."""
    
    @staticmethod
    def format_score(score: float, decimal_places: int = 4) -> str:
        """Format score for display."""
        return f"{score:.{decimal_places}f}"
    
    @staticmethod
    def format_percentage(value: float, decimal_places: int = 2) -> str:
        """Format percentage for display."""
        return f"{value:.{decimal_places}f}%"
    
    @staticmethod
    def format_time(seconds: float) -> str:
        """Format time duration for display."""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.1f}m"
        else:
            hours = seconds / 3600
            return f"{hours:.1f}h"
    
    @staticmethod
    def format_volume(volume: float) -> str:
        """Format volume for display."""
        if volume >= 1e6:
            return f"{volume/1e6:.2f}M"
        elif volume >= 1e3:
            return f"{volume/1e3:.2f}K"
        else:
            return f"{volume:.2f}"
    
    @staticmethod
    def create_display_data(result: tuple) -> Dict[str, Any]:
        """Create formatted display data for UI."""
        score, params, breakdown = result
        
        return {
            'score_display': UIDataFormatter.format_score(score),
            'params_display': {
                'min_overlap': UIDataFormatter.format_percentage(params.min_overlap),
                'max_overlap': UIDataFormatter.format_percentage(params.max_overlap),
                'min_order': str(params.min_order),
                'max_order': str(params.max_order)
            },
            'breakdown_display': {
                'max_score': UIDataFormatter.format_score(breakdown.max_score),
                'variance_score': UIDataFormatter.format_score(breakdown.variance_score),
                'tail_score': UIDataFormatter.format_score(breakdown.tail_score),
                'penalty_total': UIDataFormatter.format_score(breakdown.penalty_total)
            }
        }


class UIPayloadValidator:
    """Validate UI payloads."""
    
    @staticmethod
    def validate_params_payload(payload: Dict[str, Any]) -> bool:
        """Validate parameters payload."""
        required_fields = ['min_overlap', 'max_overlap', 'min_order', 'max_order']
        
        for field in required_fields:
            if field not in payload:
                return False
        
        # Validate ranges
        if not (0 <= payload['min_overlap'] <= payload['max_overlap'] <= 100):
            return False
        
        if not (1 <= payload['min_order'] <= payload['max_order'] <= 50):
            return False
        
        return True
    
    @staticmethod
    def validate_optimization_config(payload: Dict[str, Any]) -> bool:
        """Validate optimization configuration payload."""
        required_fields = ['params', 'max_iterations', 'time_limit']
        
        for field in required_fields:
            if field not in payload:
                return False
        
        if not UIPayloadValidator.validate_params_payload(payload['params']):
            return False
        
        if payload['max_iterations'] <= 0 or payload['time_limit'] <= 0:
            return False
        
        return True
