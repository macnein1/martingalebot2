from __future__ import annotations

import numpy as np
from typing import Tuple, Dict


def tail_only_rescale_keep_first_two(v: np.ndarray) -> float:
    """
    Keep v[0], v[1] fixed; rescale v[2:] with a single factor so that sum(v) = 100.
    
    Args:
        v: Volume array to rescale
        
    Returns:
        Scaling factor applied to tail (v[2:])
    """
    n = len(v)
    if n <= 2:
        return 1.0
    
    # Calculate current sum of first two elements
    sum_first_two = v[0] + v[1]
    
    # Calculate current sum of tail
    sum_tail = np.sum(v[2:])
    
    # If tail sum is zero or near zero, create a small geometric ramp
    if sum_tail < 1e-12:
        # Create a small increasing geometric sequence for tail
        for i in range(2, n):
            v[i] = 1e-6 * (1.01 ** (i - 2))
        sum_tail = np.sum(v[2:])
    
    # Calculate needed tail sum to reach 100 total
    needed_tail_sum = 100.0 - sum_first_two
    
    # Calculate scaling factor
    if sum_tail > 1e-12 and needed_tail_sum > 0:
        f = needed_tail_sum / sum_tail
        v[2:] *= f
        return f
    
    return 1.0


def compute_m_from_v(v: np.ndarray) -> np.ndarray:
    """
    Compute martingale ratios from volumes.
    m[0] = 0; m[i] = v[i]/max(v[i-1], 1e-12) - 1 for i >= 1
    
    Args:
        v: Volume array
        
    Returns:
        Martingale ratio array (m)
    """
    n = len(v)
    if n == 0:
        return np.array([])
    
    m = np.zeros(n, dtype=np.float64)
    m[0] = 0.0  # First order has no martingale
    
    for i in range(1, n):
        prev = max(v[i-1], 1e-12)
        m[i] = v[i] / prev - 1.0
    
    return m


def rechain_v_from_m(v0: float, v1: float, m: np.ndarray) -> np.ndarray:
    """
    Reconstruct volume array from v0, v1 and martingale ratios.
    v[0] = v0; v[1] = v1; v[i] = v[i-1] * (1 + m[i]) for i >= 2
    
    Args:
        v0: First volume value
        v1: Second volume value  
        m: Martingale ratio array
        
    Returns:
        Reconstructed volume array
    """
    n = len(m)
    if n == 0:
        return np.array([])
    
    v = np.zeros(n, dtype=np.float64)
    v[0] = v0
    
    if n > 1:
        v[1] = v1
    
    for i in range(2, n):
        v[i] = v[i-1] * (1.0 + m[i])
    
    return v


def longest_plateau_run(m: np.ndarray, center: float = 1.0, tol: float = 0.02, start_idx: int = 2) -> Tuple[int, int]:
    """
    Find the longest run where |m - center| < tol.
    
    Args:
        m: Martingale ratio array
        center: Center value to check around (default 1.0)
        tol: Tolerance for plateau detection
        start_idx: Starting index to check from
        
    Returns:
        Tuple of (max_run_length, start_position_of_max_run)
    """
    n = len(m)
    if n <= start_idx:
        return 0, -1
    
    max_run = 0
    max_run_start = -1
    current_run = 0
    current_run_start = -1
    
    for i in range(start_idx, n):
        if abs(m[i] - center) < tol:
            if current_run == 0:
                current_run_start = i
            current_run += 1
            if current_run > max_run:
                max_run = current_run
                max_run_start = current_run_start
        else:
            current_run = 0
            current_run_start = -1
    
    return max_run, max_run_start


def bootstrap_tail_from_bands(v0: float, v1: float, N: int,
                              m2_min: float, m2_max: float,
                              m_min: float,
                              m_head: float, m_tail: float, tau_scale: float,
                              eps: float = 1e-12) -> np.ndarray:
    """
    Bootstrap feasible tail volumes using geometric growth within m2 bounds.
    
    Creates a volume array with v0, v1 fixed and v[2:] following a geometric
    progression that satisfies m2 band constraints. The sum is normalized to 100
    while preserving v0 and v1.
    
    Args:
        v0: Fixed first volume
        v1: Fixed second volume  
        N: Total number of orders
        m2_min/m2_max: Bounds for m[2] = v[2]/v[1] - 1
        m_min: Minimum growth rate for tail
        m_head/m_tail/tau_scale: Parameters for decaying ceiling
        eps: Numerical tolerance
        
    Returns:
        Volume array of length N with sum = 100
    """
    if N <= 0:
        return np.array([])
    
    if N == 1:
        # Single order, must sum to 100
        return np.array([100.0])
    
    if N == 2:
        # Two orders, keep v0 fixed but adjust v1 to sum to 100
        v = np.zeros(2, dtype=np.float64)
        v[0] = v0
        v[1] = 100.0 - v0
        return v
    
    # For N >= 3
    # Initialize volume array
    v = np.zeros(N, dtype=np.float64)
    v[0] = v0
    v[1] = v1
    
    # Use middle of m2 range for v[2]
    # This ensures m2 constraint is always satisfied
    # Use below middle to leave room for rescaling which increases m2
    m2_use = 0.4 * (m2_min + m2_max)
    v[2] = v1 * (1.0 + m2_use)
    
    if N == 3:
        # For N=3, apply tail_only_rescale to get sum=100
        # This will increase v[2] proportionally
        tail_only_rescale_keep_first_two(v)
        # Check if m2 is still within bounds (with some tolerance)
        m2_final = (v[2] / v[1]) - 1.0
        if m2_final > m2_max * 1.1:  # Allow 10% overshoot
            # m2 is too large after rescaling, cap it
            v[2] = v[1] * (1.0 + m2_max)
            # Accept that sum might not be exactly 100
        return v
    
    # For N > 3, build a geometric progression for v[3:]
    # Use a moderate growth rate within bounds
    tau = max(1.0, N * tau_scale)
    
    # Calculate growth rate bounds
    r_min = 1.0 + m_min
    r_max_ceiling = 1.0 + m_tail + (m_head - m_tail) * np.exp(-1.0 / tau)
    r_max = min(1.5, r_max_ceiling)  # Cap growth for stability
    
    # Use a moderate growth rate
    r = 0.5 * (r_min + r_max)
    
    # Build geometric tail
    for i in range(3, N):
        v[i] = v[i-1] * r
    
    # Now rescale the tail to sum to 100
    # This preserves v0, v1 and the relative proportions of v[2:]
    tail_only_rescale_keep_first_two(v)
    
    # Verify m2 is still within bounds after rescaling
    m2_final = (v[2] / v[1]) - 1.0
    if m2_final < m2_min or m2_final > m2_max:
        # This shouldn't happen with tail_only_rescale, but just in case
        # Clip v[2] and rescale again
        v2_min = v1 * (1.0 + m2_min)
        v2_max = v1 * (1.0 + m2_max)
        v[2] = np.clip(v[2], v2_min, v2_max)
        tail_only_rescale_keep_first_two(v)
    
    return v


def hard_clip_local_growth(volumes: np.ndarray, g_min: float, g_max: float) -> Tuple[np.ndarray, Dict[str, int]]:
    """
    Apply hard clipping to local growth ratios for i>=2 and rebuild the chain.

    Keeps v0 as-is and ensures v1 <= v0. For i>=2, enforces g_i in [g_min, g_max].

    Args:
        volumes: Volume percentages array (sum can be arbitrary)
        g_min: Minimum allowed local growth ratio (>=1.0 for monotone)
        g_max: Maximum allowed local growth ratio

    Returns:
        Tuple of (new_volumes, stats) where new_volumes preserves v0 and clamps ratios,
        and stats contains counts of hi/lo clips.
    """
    if volumes.size == 0:
        return volumes.copy(), {"hi": 0, "lo": 0}

    v = volumes.astype(np.float64).copy()
    M = v.size

    out = np.zeros_like(v)
    out[0] = v[0]
    if M >= 2:
        out[1] = min(v[1], out[0])
    hi = 0
    lo = 0
    for i in range(2, M):
        prev = max(out[i-1], 1e-12)
        g = v[i] / max(v[i-1], 1e-12)
        if g > g_max:
            g = g_max
            hi += 1
        elif g < g_min:
            g = g_min
            lo += 1
        out[i] = prev * g

    return out, {"hi": int(hi), "lo": int(lo)}


def isotonic_non_decreasing(y: np.ndarray) -> np.ndarray:
    """
    Pool Adjacent Violators Algorithm (PAVA) for isotonic regression (non-decreasing).
    Equal weights are assumed.

    Args:
        y: 1D array to be made non-decreasing

    Returns:
        Isotonic-regressed array of same shape
    """
    n = int(y.size)
    if n <= 1:
        return y.astype(np.float64).copy()

    # Initialize blocks
    levels = []  # block means
    counts = []  # block sizes (ints)

    for i in range(n):
        levels.append(float(y[i]))
        counts.append(1)
        # Merge while decreasing
        while len(levels) >= 2 and levels[-2] > levels[-1]:
            total_count = counts[-2] + counts[-1]
            new_level = (levels[-2] * counts[-2] + levels[-1] * counts[-1]) / total_count
            levels[-2] = new_level
            counts[-2] = total_count
            del levels[-1]
            del counts[-1]

    # Expand blocks back to full array
    out = np.empty(n, dtype=np.float64)
    idx = 0
    for lvl, cnt in zip(levels, counts):
        out[idx: idx + cnt] = lvl
        idx += cnt
    return out