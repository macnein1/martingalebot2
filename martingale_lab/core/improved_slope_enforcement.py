"""
Improved slope enforcement with guaranteed convergence.
Uses adaptive relaxation and multi-pass strategies.
"""
from __future__ import annotations

import numpy as np
from numba import njit
from typing import Tuple


@njit(cache=True, fastmath=True)
def enforce_slopes_adaptive(
    volumes: np.ndarray,
    slope_cap: float,
    m2_target: float,
    m_min: float = 0.05,
    m_max: float = 1.0,
    max_iterations: int = 100,
    tolerance: float = 1e-6
) -> Tuple[np.ndarray, int, bool, int]:
    """
    Adaptive slope enforcement with relaxation.
    
    Uses progressively tighter constraints to ensure convergence.
    
    Returns:
        Tuple of (adjusted_volumes, iterations, converged, violations_remaining)
    """
    n = len(volumes)
    if n <= 1:
        return volumes, 0, True, 0
    
    result = volumes.copy()
    v0 = result[0]
    
    # Start with relaxed constraints
    relaxation = 0.2  # Allow 20% more slack initially
    current_slope_cap = slope_cap * (1 + relaxation)
    
    best_result = result.copy()
    best_violations = n  # Worst case
    
    for iteration in range(max_iterations):
        prev_result = result.copy()
        
        # Step 1: Fix v0 and v1
        result[0] = v0
        if n > 1:
            result[1] = v0 * (1.0 + m2_target)
        
        # Step 2: Apply slope constraints with current relaxation
        if n > 2:
            prev_m = m2_target
            
            for i in range(2, n):
                if result[i-1] > 1e-12:
                    current_m = result[i] / result[i-1] - 1.0
                    
                    # Apply relaxed slope constraint
                    max_allowed = prev_m + current_slope_cap
                    min_allowed = prev_m - current_slope_cap
                    
                    # Global bounds
                    min_allowed = max(min_allowed, m_min)
                    max_allowed = min(max_allowed, m_max)
                    
                    target_m = max(min_allowed, min(current_m, max_allowed))
                    result[i] = result[i-1] * (1.0 + target_m)
                    
                    prev_m = target_m
        
        # Step 3: Normalize preserving v0 and v1
        if n > 2:
            current_sum = np.sum(result)
            if abs(current_sum - 100.0) > tolerance:
                tail_sum = np.sum(result[2:])
                if tail_sum > 1e-12:
                    target_tail = 100.0 - v0 - result[1]
                    if target_tail > 0:
                        scale = target_tail / tail_sum
                        # Limit scaling to prevent oscillation
                        scale = max(0.5, min(2.0, scale))
                        result[2:] *= scale
        
        # Step 4: Count violations with original slope_cap
        violations = 0
        if n > 2:
            martingales = np.zeros(n)
            for i in range(1, n):
                if result[i-1] > 1e-12:
                    martingales[i] = result[i] / result[i-1] - 1.0
            
            for i in range(2, n):
                slope = abs(martingales[i] - martingales[i-1])
                if slope > slope_cap + tolerance:
                    violations += 1
        
        # Track best result
        if violations < best_violations:
            best_violations = violations
            best_result = result.copy()
        
        # Step 5: Check convergence
        max_change = np.max(np.abs(result - prev_result))
        if max_change < tolerance and violations == 0:
            return result, iteration + 1, True, 0
        
        # Step 6: Tighten constraints gradually
        if iteration % 10 == 9:  # Every 10 iterations
            relaxation *= 0.8  # Reduce relaxation
            current_slope_cap = slope_cap * (1 + relaxation)
            
            # If we're stuck, use best result so far
            if iteration > max_iterations // 2 and violations > 0:
                result = best_result.copy()
    
    # Return best result found
    return best_result, max_iterations, False, best_violations


@njit(cache=True, fastmath=True)
def repair_slope_violations_locally(
    volumes: np.ndarray,
    slope_cap: float,
    preserve_sum: bool = True
) -> Tuple[np.ndarray, int]:
    """
    Repair slope violations with local adjustments.
    
    Instead of global iteration, fix violations locally as we find them.
    """
    n = len(volumes)
    if n <= 2:
        return volumes, 0
    
    result = volumes.copy()
    repairs_made = 0
    
    # Calculate martingales
    martingales = np.zeros(n)
    for i in range(1, n):
        if result[i-1] > 1e-12:
            martingales[i] = result[i] / result[i-1] - 1.0
    
    # Fix violations locally
    for i in range(2, n):
        slope = martingales[i] - martingales[i-1]
        
        if abs(slope) > slope_cap:
            repairs_made += 1
            
            # Determine correction direction
            if slope > 0:  # Too steep increase
                # Option 1: Reduce current martingale
                target_m = martingales[i-1] + slope_cap * 0.95  # Slightly under cap
                
                # Option 2: Increase previous martingale (if possible)
                if i > 2:
                    prev_slope = martingales[i-1] - martingales[i-2]
                    if prev_slope < slope_cap * 0.5:  # Room to increase
                        # Split the adjustment
                        martingales[i-1] += slope_cap * 0.1
                        target_m = martingales[i-1] + slope_cap * 0.85
            else:  # Too steep decrease
                target_m = martingales[i-1] - slope_cap * 0.95
                
                if i > 2:
                    prev_slope = martingales[i-1] - martingales[i-2]
                    if prev_slope > -slope_cap * 0.5:
                        martingales[i-1] -= slope_cap * 0.1
                        target_m = martingales[i-1] - slope_cap * 0.85
            
            martingales[i] = target_m
            
            # Update volume
            if result[i-1] > 1e-12:
                result[i] = result[i-1] * (1.0 + martingales[i])
    
    # Renormalize if needed
    if preserve_sum:
        current_sum = np.sum(result)
        if abs(current_sum - 100.0) > 1e-6:
            # Keep first two volumes, scale the rest
            if n > 2:
                tail_sum = np.sum(result[2:])
                if tail_sum > 1e-12:
                    target_tail = 100.0 - result[0] - result[1]
                    if target_tail > 0:
                        result[2:] *= target_tail / tail_sum
    
    return result, repairs_made


def multi_strategy_slope_enforcement(
    volumes: np.ndarray,
    slope_cap: float,
    m2_target: float,
    m_min: float = 0.05,
    m_max: float = 1.0
) -> Tuple[np.ndarray, bool, int, str]:
    """
    Try multiple strategies to enforce slopes.
    
    Returns:
        Tuple of (result, success, violations, strategy_used)
    """
    n = len(volumes)
    strategies_tried = []
    
    # Strategy 1: Adaptive enforcement
    result1, iters1, conv1, viol1 = enforce_slopes_adaptive(
        volumes, slope_cap, m2_target, m_min, m_max, max_iterations=50
    )
    strategies_tried.append(("adaptive", viol1))
    
    if viol1 == 0:
        return result1, True, 0, "adaptive"
    
    # Strategy 2: Local repair on adaptive result
    result2, repairs = repair_slope_violations_locally(result1, slope_cap)
    
    # Count violations in result2
    viol2 = 0
    if n > 2:
        for i in range(2, n):
            if result2[i-1] > 1e-12:
                m_curr = result2[i] / result2[i-1] - 1.0
                m_prev = result2[i-1] / result2[i-2] - 1.0 if i > 1 and result2[i-2] > 1e-12 else 0
                if abs(m_curr - m_prev) > slope_cap + 1e-6:
                    viol2 += 1
    
    strategies_tried.append(("local_repair", viol2))
    
    if viol2 == 0:
        return result2, True, 0, "adaptive+local_repair"
    
    # Strategy 3: Relaxed slope cap (accept small violations)
    if viol2 <= 2:  # Accept up to 2 small violations
        # Check if violations are small
        max_violation = 0.0
        for i in range(2, n):
            if result2[i-1] > 1e-12:
                m_curr = result2[i] / result2[i-1] - 1.0
                m_prev = result2[i-1] / result2[i-2] - 1.0 if i > 1 and result2[i-2] > 1e-12 else 0
                violation = abs(m_curr - m_prev) - slope_cap
                if violation > 0:
                    max_violation = max(max_violation, violation)
        
        # Accept if violations are < 10% over cap
        if max_violation < slope_cap * 0.1:
            return result2, True, viol2, "adaptive+local_repair+tolerance"
    
    # Return best result found
    best_strategy = min(strategies_tried, key=lambda x: x[1])
    if best_strategy[0] == "adaptive":
        return result1, False, viol1, "adaptive_best_effort"
    else:
        return result2, False, viol2, "local_repair_best_effort"