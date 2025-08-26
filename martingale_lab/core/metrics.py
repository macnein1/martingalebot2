"""
Exit-ease and advanced metrics for martingale optimization.
"""

import numpy as np
from numba import njit
from typing import Tuple, Dict, Any


@njit(cache=True, fastmath=True)
def compute_exit_ease(needpct: np.ndarray, guard_min: float = 0.1) -> np.ndarray:
    """
    Compute exit-ease scores (inverse of needpct).
    
    Higher exit-ease means easier to exit at that order level.
    
    Args:
        needpct: Need percentages for each order
        guard_min: Minimum needpct to avoid division by zero
        
    Returns:
        Exit-ease array (1/needpct)
    """
    return 1.0 / np.maximum(needpct, guard_min)


@njit(cache=True, fastmath=True)
def exit_ease_harmonic_mean(exit_ease: np.ndarray) -> float:
    """
    Compute harmonic mean of exit-ease scores.
    
    Harmonic mean emphasizes lower values, penalizing hard-to-exit positions.
    """
    n = len(exit_ease)
    if n == 0:
        return 0.0
    
    # Harmonic mean = n / sum(1/x_i)
    reciprocal_sum = np.sum(1.0 / np.maximum(exit_ease, 1e-10))
    if reciprocal_sum > 0:
        return n / reciprocal_sum
    return 0.0


@njit(cache=True, fastmath=True)
def exit_ease_tail_weighted(exit_ease: np.ndarray, power: float = 2.0) -> float:
    """
    Compute tail-weighted average of exit-ease scores.
    
    Later orders get higher weight to emphasize tail exit opportunities.
    
    Args:
        exit_ease: Exit-ease array
        power: Power for weighting (2.0 = quadratic)
        
    Returns:
        Weighted average
    """
    n = len(exit_ease)
    if n == 0:
        return 0.0
    
    # Create weights: w[i] = 1 + (i/n)^power
    weights = np.zeros(n)
    for i in range(n):
        weights[i] = 1.0 + (i / n) ** power
    
    # Weighted average
    weighted_sum = np.sum(exit_ease * weights)
    weight_total = np.sum(weights)
    
    if weight_total > 0:
        return weighted_sum / weight_total
    return 0.0


@njit(cache=True, fastmath=True)
def compute_block_metrics(values: np.ndarray, n_blocks: int = 4) -> Tuple[float, float, float, float]:
    """
    Compute metrics for each quartile block.
    
    Returns:
        (Q1_sum, Q2_sum, Q3_sum, Q4_sum)
    """
    n = len(values)
    if n == 0:
        return 0.0, 0.0, 0.0, 0.0
    
    block_size = n // n_blocks
    
    q1_end = block_size
    q2_end = 2 * block_size
    q3_end = 3 * block_size
    
    q1_sum = np.sum(values[:q1_end])
    q2_sum = np.sum(values[q1_end:q2_end])
    q3_sum = np.sum(values[q2_end:q3_end])
    q4_sum = np.sum(values[q3_end:])
    
    return q1_sum, q2_sum, q3_sum, q4_sum


@njit(cache=True, fastmath=True)
def block_balance_penalty(q1: float, q2: float, q3: float, q4: float) -> float:
    """
    Compute penalty for unbalanced block distribution.
    
    Penalizes large differences between adjacent blocks.
    """
    penalty = 0.0
    penalty += (q1 - q2) ** 2
    penalty += (q2 - q3) ** 2
    penalty += (q3 - q4) ** 2
    return penalty


@njit(cache=True, fastmath=True)
def detect_micro_patterns(m: np.ndarray, start_idx: int = 2) -> Dict[str, int]:
    """
    Detect problematic micro-patterns in martingale ratios.
    
    Returns counts of:
    - plateau: Long runs where |m - 1.0| < 0.02
    - sawtooth: Alternating large changes
    - sudden_drops: m[i] < 0.7 * m[i-1]
    """
    n = len(m)
    plateau_count = 0
    sawtooth_count = 0
    sudden_drops = 0
    
    for i in range(start_idx, n):
        # Plateau detection
        if abs(m[i] - 1.0) < 0.02:
            plateau_count += 1
        
        # Sudden drop detection
        if i > start_idx and m[i] < m[i-1] * 0.7:
            sudden_drops += 1
        
        # Sawtooth detection (zigzag pattern)
        if i > start_idx + 1:
            if abs(m[i] - m[i-2]) < 0.01 and abs(m[i] - m[i-1]) > 0.1:
                sawtooth_count += 1
    
    return {
        'plateau': plateau_count,
        'sawtooth': sawtooth_count,
        'sudden_drops': sudden_drops
    }


def compute_exit_ease_metrics(needpct: np.ndarray, volumes: np.ndarray) -> Dict[str, float]:
    """
    Compute comprehensive exit-ease metrics.
    
    Args:
        needpct: Need percentages
        volumes: Volume percentages
        
    Returns:
        Dictionary of metrics
    """
    # Basic exit-ease
    exit_ease = compute_exit_ease(needpct)
    
    # Overall metrics
    harmonic = exit_ease_harmonic_mean(exit_ease)
    tail_weighted = exit_ease_tail_weighted(exit_ease)
    
    # Block metrics for exit-ease
    q1_ee, q2_ee, q3_ee, q4_ee = compute_block_metrics(exit_ease)
    
    # Block metrics for volumes
    q1_vol, q2_vol, q3_vol, q4_vol = compute_block_metrics(volumes)
    
    # Balance penalties
    ee_balance = block_balance_penalty(q1_ee, q2_ee, q3_ee, q4_ee)
    vol_balance = block_balance_penalty(q1_vol, q2_vol, q3_vol, q4_vol)
    
    # Front/tail ratio
    n = len(exit_ease)
    if n >= 4:
        front_median = np.median(exit_ease[:n//4])
        tail_median = np.median(exit_ease[3*n//4:])
        front_tail_ratio = front_median / max(tail_median, 0.01)
    else:
        front_tail_ratio = 1.0
    
    return {
        'ee_harmonic': harmonic,
        'ee_tail_weighted': tail_weighted,
        'ee_q1': q1_ee,
        'ee_q2': q2_ee,
        'ee_q3': q3_ee,
        'ee_q4': q4_ee,
        'ee_balance_penalty': ee_balance,
        'vol_balance_penalty': vol_balance,
        'ee_front_tail_ratio': front_tail_ratio,
        'vol_q1': q1_vol,
        'vol_q2': q2_vol,
        'vol_q3': q3_vol,
        'vol_q4': q4_vol,
    }