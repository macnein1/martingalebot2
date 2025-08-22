"""
Constraints and normalization functions for martingale optimization.
"""
import numpy as np
from typing import Tuple, List, Dict, Any
from .types import Params, Schedule


def validate_search_space(cfg: Dict[str, Any]) -> None:
    """
    Validate search space configuration parameters.
    
    Args:
        cfg: Configuration dictionary with optimization parameters
        
    Raises:
        AssertionError: If any constraint is violated
    """
    assert 0 < cfg.get('overlap_min', 0) < cfg.get('overlap_max', 100) <= 100, \
        f"Invalid overlap range: {cfg.get('overlap_min')} - {cfg.get('overlap_max')}"
    
    assert 1 <= cfg.get('orders_min', 1) <= cfg.get('orders_max', 100) <= 100, \
        f"Invalid orders range: {cfg.get('orders_min')} - {cfg.get('orders_max')}"
    
    # Penalty weight validation
    alpha = cfg.get('alpha', 0)
    beta = cfg.get('beta', 0) 
    gamma = cfg.get('gamma', 0)
    assert 0 <= alpha <= 1 and 0 <= beta <= 1 and 0 <= gamma <= 1, \
        f"Invalid penalty weights: α={alpha}, β={beta}, γ={gamma}"
    
    # At least one objective should be active
    assert alpha + beta + gamma > 0, "At least one objective weight must be positive"


def assert_schedule_invariants(schedule: Schedule) -> None:
    """
    Assert critical schedule invariants for early error detection.
    
    Args:
        schedule: Schedule object to validate
        
    Raises:
        AssertionError: If any invariant is violated
    """
    if hasattr(schedule, 'indent_pct'):
        indent = schedule.indent_pct
        # Indent should be non-negative and monotonically increasing
        assert indent[0] >= 0, f"First indent must be non-negative: {indent[0]}"
        for i in range(len(indent) - 1):
            assert indent[i] <= indent[i + 1], \
                f"Indent not monotonic at position {i}: {indent[i]} > {indent[i+1]}"
    
    if hasattr(schedule, 'volume_pct'):
        vol = schedule.volume_pct
        # Volumes should sum to 1 and be non-negative
        vol_sum = sum(vol)
        assert abs(vol_sum - 1.0) < 1e-6, f"Volume sum not 1.0: {vol_sum}"
        assert all(v >= 0 for v in vol), f"Negative volumes found: {[v for v in vol if v < 0]}"


def validate_martingale_bounds(martingale_pct: float) -> None:
    """
    Validate martingale percentage bounds.
    
    Args:
        martingale_pct: Martingale percentage to validate
        
    Raises:
        AssertionError: If bounds are violated
    """
    assert 0 <= martingale_pct <= 100, f"Martingale % out of bounds: {martingale_pct}"


def validate_need_pct_sanity(need_pct_values: np.ndarray) -> None:
    """
    Validate NeedPct values are within reasonable bounds.
    
    Args:
        need_pct_values: Array of NeedPct values
        
    Raises:
        AssertionError: If values are unreasonable
    """
    assert np.all(need_pct_values >= 0), "NeedPct values must be non-negative"
    assert np.all(need_pct_values <= 50), f"NeedPct values too high (>50%): {np.max(need_pct_values)}"
    assert not np.any(np.isnan(need_pct_values)), "NaN values in NeedPct"
    assert not np.any(np.isinf(need_pct_values)), "Infinite values in NeedPct"


def validate_physical_constraints(schedule: Schedule) -> None:
    """
    Validate physical/business constraints on schedule.
    
    Args:
        schedule: Schedule to validate
        
    Raises:
        AssertionError: If constraints are violated
    """
    # Volume constraints
    if hasattr(schedule, 'volumes'):
        volumes = schedule.volumes
        assert np.all(volumes >= 0), "All volumes must be non-negative"
        assert np.sum(volumes) > 0, "Total volume must be positive"
    
    # Order constraints  
    if hasattr(schedule, 'orders'):
        orders = schedule.orders
        assert np.all(orders >= 1), "All orders must be at least 1"
        assert np.all(orders <= 1000), "Orders too large (>1000)"
    
    # Overlap constraints
    if hasattr(schedule, 'overlaps'):
        overlaps = schedule.overlaps
        assert np.all(overlaps >= 0), "Overlaps must be non-negative"
        assert np.all(overlaps <= 100), "Overlaps must be <= 100%"


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
    def validate_params_strict(params: Params) -> None:
        """Strict parameter validation with detailed error messages."""
        if not (0 <= params.min_overlap <= params.max_overlap <= 100):
            raise ValueError(f"Invalid overlap range: {params.min_overlap} - {params.max_overlap}")
        
        if not (1 <= params.min_order <= params.max_order <= 50):
            raise ValueError(f"Invalid order range: {params.min_order} - {params.max_order}")
        
        if params.risk_factor <= 0:
            raise ValueError(f"Risk factor must be positive: {params.risk_factor}")
        
        if not (0 <= params.smoothing_factor <= 1):
            raise ValueError(f"Smoothing factor out of range [0,1]: {params.smoothing_factor}")
        
        if not (0 <= params.tail_weight <= 1):
            raise ValueError(f"Tail weight out of range [0,1]: {params.tail_weight}")
    
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

    @staticmethod
    def validate_schedule_strict(schedule: Schedule) -> None:
        """Strict schedule validation with detailed error messages."""
        if schedule.num_levels == 0:
            raise ValueError("Schedule must have at least one level")
        
        # Apply all invariant checks
        assert_schedule_invariants(schedule)
        validate_physical_constraints(schedule)


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
