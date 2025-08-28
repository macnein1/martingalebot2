"""
Two-phase constraint enforcement for perfect slope control.
Phase 1: Normalize to sum=100
Phase 2: Enforce slopes while maintaining sum
"""
from __future__ import annotations

import numpy as np
from numba import njit
from typing import Tuple


@njit(cache=True, fastmath=True)
def enforce_slopes_with_sum_preservation(
    volumes: np.ndarray,
    slope_cap: float,
    m2_target: float,
    m_min: float = 0.05,
    m_max: float = 1.0,
    max_iterations: int = 50,
    tolerance: float = 1e-6
) -> Tuple[np.ndarray, int, bool]:
    """
    Enforce slope constraints while preserving sum=100.
    
    This uses an iterative projection method that alternates between:
    1. Enforcing slope constraints
    2. Renormalizing to sum=100
    
    Special handling for m2 to ensure it's preserved.
    
    Args:
        volumes: Input volume array (should sum to ~100)
        slope_cap: Maximum slope change allowed
        m2_target: Target value for m[1] (v[1]/v[0] - 1)
        m_min: Minimum martingale value
        m_max: Maximum martingale value
        max_iterations: Maximum iterations for convergence
        tolerance: Convergence tolerance
        
    Returns:
        Tuple of (adjusted_volumes, iterations_used, converged)
    """
    n = len(volumes)
    if n <= 1:
        return volumes, 0, True
    
    result = volumes.copy()
    v0 = result[0]  # Keep first volume fixed
    
    for iteration in range(max_iterations):
        prev_result = result.copy()
        
        # Step 1: Enforce m2 exactly
        if n > 1:
            result[1] = v0 * (1.0 + m2_target)
        
        # Step 2: Enforce slope constraints for rest
        if n > 2:
            # Calculate current martingales
            prev_m = m2_target
            
            for i in range(2, n):
                if result[i-1] > 1e-12:
                    current_m = result[i] / result[i-1] - 1.0
                    
                    # Apply slope constraint (with tiny tolerance for numerical stability)
                    max_allowed = prev_m + slope_cap - 1e-9
                    min_allowed = prev_m - slope_cap + 1e-9
                    
                    # Also apply global bounds
                    min_allowed = max(min_allowed, m_min)
                    max_allowed = min(max_allowed, m_max)
                    
                    # Adjust if needed
                    target_m = max(min_allowed, min(current_m, max_allowed))
                    result[i] = result[i-1] * (1.0 + target_m)
                    
                    prev_m = target_m
        
        # Step 3: Renormalize to sum=100 (keeping v0 and v1 fixed)
        if n > 2:
            current_sum = np.sum(result)
            if abs(current_sum - 100.0) > tolerance:
                # Only scale orders 2 and beyond
                tail_sum = np.sum(result[2:])
                if tail_sum > 1e-12:
                    target_tail = 100.0 - v0 - result[1]
                    if target_tail > 0:
                        result[2:] *= target_tail / tail_sum
        
        # Check convergence
        max_change = np.max(np.abs(result - prev_result))
        if max_change < tolerance:
            return result, iteration + 1, True
    
    return result, max_iterations, False


@njit(cache=True, fastmath=True)
def calculate_feasible_v0(
    num_orders: int,
    slope_cap: float,
    m2: float,
    target_sum: float = 100.0
) -> float:
    """
    Calculate the maximum feasible v0 that allows sum=target_sum
    with given slope constraints.
    
    This assumes geometric growth at maximum rate.
    
    Args:
        num_orders: Number of orders
        slope_cap: Maximum slope cap
        m2: Value for m[1]
        target_sum: Target sum (usually 100)
        
    Returns:
        Maximum feasible v0
    """
    if num_orders <= 1:
        return target_sum
    
    # Calculate sum of growth factors
    # v[0] = v0
    # v[1] = v0 * (1 + m2)
    # v[2] = v0 * (1 + m2) * (1 + m2 + slope_cap)
    # v[3] = v0 * (1 + m2) * (1 + m2 + slope_cap) * (1 + m2 + 2*slope_cap)
    # ...
    
    sum_factors = 1.0  # for v[0]
    
    if num_orders > 1:
        sum_factors += (1.0 + m2)  # for v[1]
        
        if num_orders > 2:
            current_factor = 1.0 + m2
            current_m = m2
            
            for i in range(2, num_orders):
                # Next martingale can be at most current_m + slope_cap
                next_m = min(current_m + slope_cap, 1.0)  # Cap at 100% martingale
                current_factor *= (1.0 + next_m)
                sum_factors += current_factor
                current_m = next_m
    
    # v0 * sum_factors = target_sum
    return target_sum / sum_factors


@njit(cache=True, fastmath=True)
def apply_two_phase_enforcement(
    volumes: np.ndarray,
    v0_target: float,
    m2_target: float,
    slope_cap: float,
    m_min: float = 0.05,
    m_max: float = 1.0
) -> Tuple[np.ndarray, bool]:
    """
    Apply two-phase enforcement:
    Phase 1: Set v0 and v1, normalize rest to sum=100
    Phase 2: Enforce slopes while maintaining sum
    
    Args:
        volumes: Initial volume distribution
        v0_target: Target for first volume
        m2_target: Target for m[1] (v1/v0 - 1)
        slope_cap: Maximum slope change
        m_min: Minimum martingale
        m_max: Maximum martingale
        
    Returns:
        Tuple of (enforced_volumes, success)
    """
    n = len(volumes)
    if n == 0:
        return volumes, False
    
    result = np.zeros(n)
    
    # Phase 1: Set v0 and v1, distribute rest
    result[0] = v0_target
    
    if n > 1:
        # Ensure m2 doesn't violate slope cap (since m[0] = 0)
        m2_capped = min(m2_target, slope_cap)
        result[1] = v0_target * (1.0 + m2_capped)
        
        if n > 2:
            # Distribute remaining sum to other orders
            remaining_sum = 100.0 - result[0] - result[1]
            
            # Use input volumes as weights for distribution
            weights = volumes[2:].copy()
            weight_sum = np.sum(weights)
            
            if weight_sum > 1e-12:
                result[2:] = weights * (remaining_sum / weight_sum)
            else:
                # Uniform distribution if no weights
                result[2:] = remaining_sum / (n - 2)
    
    # Phase 2: Enforce slopes while maintaining sum
    result, iterations, converged = enforce_slopes_with_sum_preservation(
        result, slope_cap, m2_capped, m_min, m_max
    )
    
    return result, converged


def validate_slope_constraints(volumes: np.ndarray, slope_cap: float, 
                              tolerance: float = 1e-6) -> Tuple[bool, int, float]:
    """
    Validate that slope constraints are satisfied.
    
    Args:
        volumes: Volume array to validate
        slope_cap: Maximum allowed slope change
        tolerance: Numerical tolerance
        
    Returns:
        Tuple of (all_valid, num_violations, max_violation)
    """
    n = len(volumes)
    if n <= 2:
        return True, 0, 0.0
    
    violations = 0
    max_violation = 0.0
    
    # Calculate martingales
    martingales = np.zeros(n)
    for i in range(1, n):
        if volumes[i-1] > 1e-12:
            martingales[i] = volumes[i] / volumes[i-1] - 1.0
    
    # Check slopes
    for i in range(1, n):
        slope = abs(martingales[i] - martingales[i-1])
        if slope > slope_cap + tolerance:
            violations += 1
            max_violation = max(max_violation, slope - slope_cap)
    
    return violations == 0, violations, max_violation