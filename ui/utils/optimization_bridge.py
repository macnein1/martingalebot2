"""
Bridge module for connecting Streamlit UI with Martingale Lab optimization service.
"""
import sys
import os
import json
import time
from typing import Dict, Any, Optional, Callable
from pathlib import Path

# Add martingale_lab to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from martingale_lab.core.types import Params
    from martingale_lab.ui_bridge.service import OptimizationService, OptimizationConfig
    from martingale_lab.ui_bridge.payloads import UIPayloadConverter, UIResponseBuilder
except ImportError as e:
    print(f"Warning: Could not import martingale_lab modules: {e}")
    # Create mock classes for development
    class MockParams:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)
    
    class MockOptimizationConfig:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)
    
    class MockOptimizationService:
        def __init__(self):
            self.active_sessions = {}
        
        def create_session(self, config):
            return "mock_session_123"
        
        def start_optimization(self, session_id):
            return {"success": True, "message": "Mock optimization started"}
        
        def get_session_status(self, session_id):
            return {"success": True, "data": {"status": "completed"}}
    
    Params = MockParams
    OptimizationConfig = MockOptimizationConfig
    OptimizationService = MockOptimizationService
    UIPayloadConverter = None
    UIResponseBuilder = None


class OptimizationBridge:
    """Bridge between Streamlit UI and Martingale Lab optimization service."""
    
    def __init__(self):
        self.service = OptimizationService()
        self.current_session_id: Optional[str] = None
        self.progress_callback: Optional[Callable] = None
    
    def set_progress_callback(self, callback: Callable):
        """Set callback function for progress updates."""
        self.progress_callback = callback
    
    def create_optimization_session(self, parameters: Dict[str, Any], 
                                  max_iterations: int = 1000,
                                  time_limit: float = 300.0) -> Dict[str, Any]:
        """Create a new optimization session."""
        try:
            # Create Params object from UI parameters
            params = Params(
                min_overlap=parameters['min_overlap'],
                max_overlap=parameters['max_overlap'],
                min_order=parameters['min_order'],
                max_order=parameters['max_order'],
                risk_factor=1.0,  # Default values
                smoothing_factor=0.1,
                tail_weight=0.2
            )
            
            # Create optimization config
            config = OptimizationConfig(
                params=params,
                max_iterations=max_iterations,
                time_limit=time_limit,
                batch_size=100,
                early_stopping=True,
                progress_callback=self.progress_callback
            )
            
            # Create session
            session_id = self.service.create_session(config)
            self.current_session_id = session_id
            
            return {
                'success': True,
                'session_id': session_id,
                'message': 'Optimization session created successfully'
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'message': 'Failed to create optimization session'
            }
    
    def start_optimization(self) -> Dict[str, Any]:
        """Start optimization for the current session."""
        if not self.current_session_id:
            return {
                'success': False,
                'error': 'No active session',
                'message': 'Please create a session first'
            }
        
        try:
            response = self.service.start_optimization(self.current_session_id)
            return response
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'message': 'Failed to start optimization'
            }
    
    def stop_optimization(self) -> Dict[str, Any]:
        """Stop the current optimization session."""
        if not self.current_session_id:
            return {
                'success': False,
                'error': 'No active session',
                'message': 'No session to stop'
            }
        
        try:
            response = self.service.stop_optimization(self.current_session_id)
            return response
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'message': 'Failed to stop optimization'
            }
    
    def get_optimization_status(self) -> Dict[str, Any]:
        """Get status of the current optimization session."""
        if not self.current_session_id:
            return {
                'success': False,
                'error': 'No active session',
                'message': 'No session to check'
            }
        
        try:
            response = self.service.get_session_status(self.current_session_id)
            return response
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'message': 'Failed to get optimization status'
            }
    
    def get_results(self) -> Dict[str, Any]:
        """Get results from the current optimization session."""
        if not self.current_session_id:
            return {
                'success': False,
                'error': 'No active session',
                'message': 'No session to get results from'
            }
        
        try:
            response = self.service.get_session_status(self.current_session_id)
            
            if response.get('success') and response.get('data', {}).get('results'):
                return {
                    'success': True,
                    'results': response['data']['results'],
                    'statistics': response['data'].get('statistics', {}),
                    'message': 'Results retrieved successfully'
                }
            else:
                return {
                    'success': False,
                    'error': 'No results available',
                    'message': 'Optimization may still be running or failed'
                }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'message': 'Failed to get results'
            }
    
    def cleanup_session(self) -> Dict[str, Any]:
        """Clean up the current optimization session."""
        if not self.current_session_id:
            return {
                'success': False,
                'error': 'No active session',
                'message': 'No session to cleanup'
            }
        
        try:
            response = self.service.cleanup_session(self.current_session_id)
            if response.get('success'):
                self.current_session_id = None
            return response
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'message': 'Failed to cleanup session'
            }
    
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
