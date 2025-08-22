"""
Constraints and normalization functions for martingale optimization.
"""
import numpy as np
from typing import Tuple, List
from .types import Params, Schedule


class ConstraintValidator:
    """Validates and enforces constraints on optimization parameters."""
    
    @staticmethod
    def validate_params(params: Params) -> bool:
        """Validate parameter constraints."""
        return (
            0 <= params.min_overlap <= params.max_overlap <= 100 and
            1 <= params.min_order <= params.max_order <= 50 and
            params.risk_factor > 0 and
            0 <= params.smoothing_factor <= 1 and
            0 <= params.tail_weight <= 1
        )
    
    @staticmethod
    def validate_schedule(schedule: Schedule) -> bool:
        """Validate schedule constraints."""
        if schedule.num_levels == 0:
            return False
        
        # Check for positive values
        if np.any(schedule.orders <= 0) or np.any(schedule.volumes <= 0):
            return False
        
        # Check overlap constraints
        if np.any(schedule.overlaps < 0) or np.any(schedule.overlaps > 100):
            return False
        
        # Check monotonicity of orders
        if not np.all(np.diff(schedule.orders) >= 0):
            return False
        
        return True


class VolumeConstraints:
    """Volume-related constraints and calculations."""
    
    @staticmethod
    def calculate_martingale_volumes(base_volume: float, multiplier: float, 
                                   num_levels: int) -> np.ndarray:
        """Calculate martingale volume progression."""
        volumes = np.zeros(num_levels)
        for i in range(num_levels):
            volumes[i] = base_volume * (multiplier ** i)
        return volumes
    
    @staticmethod
    def normalize_volumes(volumes: np.ndarray, target_total: float) -> np.ndarray:
        """Normalize volumes to target total."""
        current_total = np.sum(volumes)
        if current_total == 0:
            return volumes
        return volumes * (target_total / current_total)
    
    @staticmethod
    def apply_volume_limits(volumes: np.ndarray, min_volume: float, 
                           max_volume: float) -> np.ndarray:
        """Apply minimum and maximum volume constraints."""
        return np.clip(volumes, min_volume, max_volume)


class OverlapConstraints:
    """Overlap-related constraints and calculations."""
    
    @staticmethod
    def generate_overlaps(min_overlap: float, max_overlap: float, 
                         num_levels: int, distribution: str = 'linear') -> np.ndarray:
        """Generate overlap values based on distribution type."""
        if distribution == 'linear':
            return np.linspace(min_overlap, max_overlap, num_levels)
        elif distribution == 'exponential':
            return np.logspace(np.log10(min_overlap), np.log10(max_overlap), num_levels)
        elif distribution == 'random':
            return np.random.uniform(min_overlap, max_overlap, num_levels)
        else:
            raise ValueError(f"Unknown distribution: {distribution}")
    
    @staticmethod
    def validate_overlap_sequence(overlaps: np.ndarray) -> bool:
        """Validate overlap sequence constraints."""
        # Check bounds
        if np.any(overlaps < 0) or np.any(overlaps > 100):
            return False
        
        # Check monotonicity (optional - depends on strategy)
        # if not np.all(np.diff(overlaps) >= 0):
        #     return False
        
        return True


class OrderConstraints:
    """Order-related constraints and calculations."""
    
    @staticmethod
    def generate_orders(min_order: int, max_order: int, 
                       num_levels: int, distribution: str = 'linear') -> np.ndarray:
        """Generate order values based on distribution type."""
        if distribution == 'linear':
            return np.linspace(min_order, max_order, num_levels, dtype=int)
        elif distribution == 'exponential':
            orders = np.logspace(np.log10(min_order), np.log10(max_order), num_levels)
            return np.round(orders).astype(int)
        elif distribution == 'random':
            return np.random.randint(min_order, max_order + 1, num_levels)
        else:
            raise ValueError(f"Unknown distribution: {distribution}")
    
    @staticmethod
    def validate_order_sequence(orders: np.ndarray) -> bool:
        """Validate order sequence constraints."""
        # Check bounds
        if np.any(orders < 1):
            return False
        
        # Check monotonicity
        if not np.all(np.diff(orders) >= 0):
            return False
        
        return True


class Normalizer:
    """Normalization utilities for optimization parameters."""
    
    @staticmethod
    def normalize_params(params: Params) -> np.ndarray:
        """Normalize parameters to [0, 1] range for optimization."""
        return np.array([
            params.min_overlap / 100.0,
            params.max_overlap / 100.0,
            (params.min_order - 1) / 49.0,  # Assuming max_order <= 50
            (params.max_order - 1) / 49.0,
            params.risk_factor / 10.0,  # Assuming max risk_factor <= 10
            params.smoothing_factor,
            params.tail_weight
        ])
    
    @staticmethod
    def denormalize_params(normalized: np.ndarray) -> Params:
        """Denormalize parameters from [0, 1] range."""
        return Params(
            min_overlap=normalized[0] * 100.0,
            max_overlap=normalized[1] * 100.0,
            min_order=int(round(normalized[2] * 49.0)) + 1,
            max_order=int(round(normalized[3] * 49.0)) + 1,
            risk_factor=normalized[4] * 10.0,
            smoothing_factor=normalized[5],
            tail_weight=normalized[6]
        )
