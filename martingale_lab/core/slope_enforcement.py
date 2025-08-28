"""
Slope enforcement utilities for martingale sequences.
Ensures smooth transitions between martingale values.
"""
from __future__ import annotations

import numpy as np
from numba import njit
from typing import Tuple


@njit(cache=True, fastmath=True)
def enforce_martingale_slopes(volumes: np.ndarray, slope_cap: float = 0.25,
                              m_min: float = 0.05, m_max: float = 1.0) -> np.ndarray:
    """
    Enforce slope constraints on martingale sequence derived from volumes.
    
    This function adjusts volumes to ensure that consecutive martingale
    values don't change by more than slope_cap.
    
    Args:
        volumes: Volume percentages array
        slope_cap: Maximum allowed change between consecutive martingales
        m_min: Minimum martingale value
        m_max: Maximum martingale value
        
    Returns:
        Adjusted volumes array with slope constraints enforced
    """
    n = len(volumes)
    if n <= 1:
        return volumes
    
    # Calculate current martingales
    martingales = np.zeros(n)
    martingales[0] = 0.0  # First order has no martingale
    
    for i in range(1, n):
        if volumes[i-1] > 1e-12:
            martingales[i] = volumes[i] / volumes[i-1] - 1.0
        else:
            martingales[i] = m_min
    
    # Enforce slope constraints
    adjusted_martingales = np.zeros(n)
    adjusted_martingales[0] = 0.0
    
    for i in range(1, n):
        # Target martingale based on input
        target_m = martingales[i]
        
        # Previous martingale
        prev_m = adjusted_martingales[i-1]
        
        # Apply slope constraint
        max_allowed = prev_m + slope_cap
        min_allowed = prev_m - slope_cap
        
        # Also apply global bounds
        min_allowed = max(min_allowed, m_min)
        max_allowed = min(max_allowed, m_max)
        
        # Clamp to allowed range
        adjusted_martingales[i] = max(min_allowed, min(target_m, max_allowed))
    
    # Reconstruct volumes from adjusted martingales
    adjusted_volumes = np.zeros(n)
    adjusted_volumes[0] = volumes[0]  # Keep first volume fixed
    
    for i in range(1, n):
        growth = 1.0 + adjusted_martingales[i]
        adjusted_volumes[i] = adjusted_volumes[i-1] * growth
    
    return adjusted_volumes


@njit(cache=True, fastmath=True)
def smooth_martingale_sequence(martingales: np.ndarray, slope_cap: float = 0.25,
                               window_size: int = 3) -> np.ndarray:
    """
    Smooth a martingale sequence to reduce slope violations.
    
    Uses local averaging to smooth out sharp transitions while
    maintaining the overall shape.
    
    Args:
        martingales: Martingale percentages (as fractions)
        slope_cap: Maximum allowed slope change
        window_size: Size of smoothing window
        
    Returns:
        Smoothed martingale sequence
    """
    n = len(martingales)
    if n <= 2:
        return martingales
    
    smoothed = martingales.copy()
    
    # Apply smoothing passes until slope constraints are met
    max_iterations = 10
    for iteration in range(max_iterations):
        violations = False
        
        # Check for violations and smooth
        for i in range(1, n):
            diff = abs(smoothed[i] - smoothed[i-1])
            if diff > slope_cap:
                violations = True
                
                # Apply local smoothing
                start = max(0, i - window_size // 2)
                end = min(n, i + window_size // 2 + 1)
                
                # Average over window
                local_mean = np.mean(smoothed[start:end])
                
                # Blend current value with local mean
                blend_factor = 0.5
                smoothed[i] = blend_factor * smoothed[i] + (1 - blend_factor) * local_mean
        
        if not violations:
            break
    
    # Final pass: hard enforce slope constraints
    for i in range(1, n):
        prev = smoothed[i-1]
        if smoothed[i] > prev + slope_cap:
            smoothed[i] = prev + slope_cap
        elif smoothed[i] < prev - slope_cap:
            smoothed[i] = prev - slope_cap
    
    return smoothed


def project_to_slope_feasible(volumes: np.ndarray, slope_cap: float,
                              preserve_sum: bool = True) -> Tuple[np.ndarray, int]:
    """
    Project volumes to the feasible region with slope constraints.
    
    Uses iterative projection to find volumes that satisfy:
    1. Monotonic increase
    2. Slope constraints on martingales
    3. Sum preservation (if requested)
    
    Args:
        volumes: Input volume array
        slope_cap: Maximum martingale slope change
        preserve_sum: Whether to preserve the sum of volumes
        
    Returns:
        Tuple of (projected_volumes, num_iterations)
    """
    n = len(volumes)
    if n <= 1:
        return volumes, 0
    
    original_sum = np.sum(volumes)
    result = volumes.copy()
    
    max_iterations = 20
    for iteration in range(max_iterations):
        # Step 1: Enforce slope constraints
        result = enforce_martingale_slopes(result, slope_cap)
        
        # Step 2: Ensure monotonic increase
        for i in range(1, n):
            if result[i] < result[i-1] + 1e-6:
                result[i] = result[i-1] + 1e-6
        
        # Step 3: Preserve sum if requested
        if preserve_sum:
            current_sum = np.sum(result)
            if current_sum > 1e-9:
                # Scale everything except first element
                if n > 1:
                    tail_sum = np.sum(result[1:])
                    if tail_sum > 1e-9:
                        target_tail = original_sum - result[0]
                        result[1:] *= target_tail / tail_sum
                else:
                    result *= original_sum / current_sum
        
        # Check convergence
        if iteration > 0:
            # Calculate martingale slopes
            all_within_slope = True
            for i in range(2, n):
                if result[i-1] > 1e-12:
                    m_curr = result[i] / result[i-1] - 1.0
                    m_prev = result[i-1] / result[i-2] - 1.0 if i > 1 and result[i-2] > 1e-12 else 0.0
                    if abs(m_curr - m_prev) > slope_cap + 1e-6:
                        all_within_slope = False
                        break
            
            if all_within_slope:
                break
    
    return result, iteration + 1