"""
JIT Kernels for DCA/Martingale Optimization
High-performance numba-compiled functions for Need% calculation and core computations.
"""
from __future__ import annotations

import numpy as np
from numba import njit
import math
from typing import Tuple


@njit(cache=True, fastmath=True)
def normalize_volumes_softmax(raw_volumes: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """
    Normalize raw volume logits using softmax to ensure sum = 100%.
    
    Args:
        raw_volumes: Raw volume logits
        temperature: Softmax temperature for controlling sharpness
        
    Returns:
        Normalized volume percentages (sum = 100.0)
    """
    if raw_volumes.size == 0:
        return np.array([], dtype=np.float64)
    
    # Apply temperature scaling
    scaled = raw_volumes / temperature
    
    # Stabilize by subtracting max
    max_val = np.max(scaled)
    exp_vals = np.exp(scaled - max_val)
    
    # Softmax normalization
    sum_exp = np.sum(exp_vals)
    if sum_exp <= 1e-12:
        # Fallback to uniform distribution
        return np.full(raw_volumes.size, 100.0 / raw_volumes.size)
    
    probabilities = exp_vals / sum_exp
    return probabilities * 100.0


@njit(cache=True, fastmath=True)
def monotonic_indents_softplus(raw_indents: np.ndarray, overlap_pct: float) -> np.ndarray:
    """
    Convert raw indent logits to monotonically increasing cumulative indents.
    
    Args:
        raw_indents: Raw indent logits
        overlap_pct: Total overlap percentage to scale to
        
    Returns:
        Cumulative indent percentages [0, i1, i2, ..., overlap_pct]
    """
    if raw_indents.size == 0:
        return np.array([0.0], dtype=np.float64)
    
    # Apply softplus to ensure positive steps
    steps = np.empty(raw_indents.size)
    for i in range(raw_indents.size):
        x = raw_indents[i]
        # Softplus: log(1 + exp(x))
        if x > 20:  # Avoid overflow
            steps[i] = x
        elif x < -20:  # Avoid underflow
            steps[i] = math.exp(x)
        else:
            steps[i] = math.log1p(math.exp(x))
    
    # Normalize steps to sum to overlap_pct
    steps_sum = np.sum(steps)
    if steps_sum <= 1e-12:
        # Fallback to uniform steps
        step_size = overlap_pct / raw_indents.size
        steps = np.full(raw_indents.size, step_size)
    else:
        steps = steps * (overlap_pct / steps_sum)
    
    # Create cumulative indents starting from 0
    cumulative = np.empty(raw_indents.size + 1)
    cumulative[0] = 0.0
    for i in range(raw_indents.size):
        cumulative[i + 1] = cumulative[i] + steps[i]
    
    return cumulative


@njit(cache=True, fastmath=True)
def martingale_percentages(volumes: np.ndarray) -> np.ndarray:
    """
    Calculate martingale percentages from volume distribution.
    
    Args:
        volumes: Volume percentages
        
    Returns:
        Martingale percentages (first element = 0, rest calculated from volume ratios)
    """
    if volumes.size <= 1:
        return np.array([0.0], dtype=np.float64)
    
    martingales = np.empty(volumes.size)
    martingales[0] = 0.0  # First order has no martingale
    
    for i in range(1, volumes.size):
        prev_vol = volumes[i - 1]
        curr_vol = volumes[i]
        
        if prev_vol <= 1e-12:
            # Avoid division by zero
            martingales[i] = 100.0  # Maximum martingale
        else:
            ratio = curr_vol / prev_vol
            martingale_pct = (ratio - 1.0) * 100.0
            # Clamp to reasonable range [1, 100]
            martingales[i] = max(1.0, min(100.0, martingale_pct))
    
    return martingales


@njit(cache=True, fastmath=True)
def order_prices_from_indents(base_price: float, indents: np.ndarray) -> np.ndarray:
    """
    Calculate order prices from base price and indent percentages.
    
    Args:
        base_price: Base entry price
        indents: Cumulative indent percentages [0, i1, i2, ..., iM]
        
    Returns:
        Order prices [base_price, p1, p2, ..., pM]
    """
    prices = np.empty(indents.size)
    
    for i in range(indents.size):
        if i == 0:
            prices[i] = base_price  # Base price
        else:
            # Price decreases by indent percentage (for long positions)
            indent_fraction = indents[i] / 100.0
            prices[i] = base_price * (1.0 - indent_fraction)
            # Ensure price doesn't go negative or too small
            prices[i] = max(prices[i], base_price * 0.01)
    
    return prices


@njit(cache=True, fastmath=True)
def need_curve_calculation(volumes: np.ndarray, prices: np.ndarray) -> np.ndarray:
    """
    Calculate Need% curve - percentage needed to return to entry from each order.
    
    This is the core calculation for "İşlemden En Hızlı Çıkış" optimization.
    
    Args:
        volumes: Volume percentages for each order
        prices: Order prices [base_price, p1, p2, ..., pM]
        
    Returns:
        Need percentages [n1, n2, ..., nM] for each order
    """
    if volumes.size == 0 or prices.size != volumes.size + 1:
        return np.array([], dtype=np.float64)
    
    need_pct = np.empty(volumes.size)
    
    # Running totals for weighted average calculation
    total_volume = 0.0
    total_value = 0.0
    
    for k in range(volumes.size):
        # Add current order to totals
        total_volume += volumes[k]
        total_value += volumes[k] * prices[k + 1]  # prices[k+1] is k-th order price
        
        # Calculate weighted average entry price up to this point
        if total_volume <= 1e-12:
            avg_entry_price = prices[k + 1]
        else:
            avg_entry_price = total_value / total_volume
        
        # Current order price
        current_price = prices[k + 1]
        
        # Percentage needed to return to average entry price from current price
        if current_price <= 1e-12:
            need_pct[k] = 100.0  # Maximum need if price is too small
        else:
            need_ratio = avg_entry_price / current_price
            need_pct[k] = (need_ratio - 1.0) * 100.0
            
            # Ensure reasonable bounds
            need_pct[k] = max(-50.0, min(200.0, need_pct[k]))
    
    return need_pct


@njit(cache=True, fastmath=True)
def evaluate_single_candidate(raw_indents: np.ndarray, raw_volumes: np.ndarray,
                             base_price: float, overlap_pct: float,
                             softmax_temp: float = 1.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluate a single DCA candidate and return all components.
    
    Args:
        raw_indents: Raw indent logits
        raw_volumes: Raw volume logits
        base_price: Base entry price
        overlap_pct: Total overlap percentage
        softmax_temp: Softmax temperature for volume normalization
        
    Returns:
        Tuple of (indents, volumes, martingales, prices, need_pct)
    """
    # Normalize volumes
    volumes = normalize_volumes_softmax(raw_volumes, softmax_temp)
    
    # Create monotonic indents
    indents = monotonic_indents_softplus(raw_indents, overlap_pct)
    
    # Calculate martingale percentages
    martingales = martingale_percentages(volumes)
    
    # Calculate order prices
    prices = order_prices_from_indents(base_price, indents)
    
    # Calculate Need% curve
    need_pct = need_curve_calculation(volumes, prices)
    
    return indents, volumes, martingales, prices, need_pct


@njit(cache=True, fastmath=True)
def batch_evaluate_candidates(raw_indents_batch: np.ndarray, raw_volumes_batch: np.ndarray,
                             base_price: float, overlap_pct: float,
                             softmax_temp: float = 1.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Batch evaluate multiple candidates for core metrics.
    
    Args:
        raw_indents_batch: Batch of raw indent logits (n_candidates, n_orders)
        raw_volumes_batch: Batch of raw volume logits (n_candidates, n_orders)
        base_price: Base entry price
        overlap_pct: Total overlap percentage
        softmax_temp: Softmax temperature
        
    Returns:
        Tuple of (max_need_batch, var_need_batch, tail_penalty_batch)
    """
    n_candidates = raw_indents_batch.shape[0]
    
    max_need_batch = np.empty(n_candidates)
    var_need_batch = np.empty(n_candidates)
    tail_penalty_batch = np.empty(n_candidates)
    
    for i in range(n_candidates):
        # Extract candidate
        raw_indents = raw_indents_batch[i, :]
        raw_volumes = raw_volumes_batch[i, :]
        
        # Evaluate candidate
        indents, volumes, martingales, prices, need_pct = evaluate_single_candidate(
            raw_indents, raw_volumes, base_price, overlap_pct, softmax_temp
        )
        
        # Calculate core metrics
        if need_pct.size > 0:
            max_need_batch[i] = np.max(need_pct)
            var_need_batch[i] = np.var(need_pct)
        else:
            max_need_batch[i] = 0.0
            var_need_batch[i] = 0.0
        
        # Simple tail penalty (concentration in last 20% of orders)
        n_orders = volumes.size
        if n_orders > 0:
            tail_start = max(0, int(0.8 * n_orders))
            tail_volume = np.sum(volumes[tail_start:])
            tail_penalty_batch[i] = tail_volume / 100.0  # Normalize to [0,1]
        else:
            tail_penalty_batch[i] = 0.0
    
    return max_need_batch, var_need_batch, tail_penalty_batch


@njit(cache=True, fastmath=True)
def constraint_violations(indents: np.ndarray, volumes: np.ndarray, 
                         martingales: np.ndarray, overlap_pct: float) -> Tuple[float, float, float]:
    """
    Calculate constraint violations for penalty system.
    
    Args:
        indents: Cumulative indent percentages
        volumes: Volume percentages
        martingales: Martingale percentages
        overlap_pct: Maximum allowed overlap
        
    Returns:
        Tuple of (monotone_violation, volume_sum_violation, bounds_violation)
    """
    # Monotonicity violation (indents should be non-decreasing)
    monotone_viol = 0.0
    if indents.size > 1:
        for i in range(1, indents.size):
            if indents[i] < indents[i - 1]:
                monotone_viol += indents[i - 1] - indents[i]
    
    # Volume sum violation (should sum to 100)
    volume_sum = np.sum(volumes)
    volume_sum_viol = abs(volume_sum - 100.0)
    
    # Bounds violations
    bounds_viol = 0.0
    
    # Indent bounds [0, overlap_pct]
    for indent in indents:
        if indent < 0:
            bounds_viol += abs(indent)
        elif indent > overlap_pct:
            bounds_viol += indent - overlap_pct
    
    # Volume bounds [0, 100]
    for vol in volumes:
        if vol < 0:
            bounds_viol += abs(vol)
        elif vol > 100:
            bounds_viol += vol - 100
    
    # Martingale bounds [0, 100] (first should be 0)
    if martingales.size > 0:
        if martingales[0] != 0.0:
            bounds_viol += abs(martingales[0])
        
        for i in range(1, martingales.size):
            if martingales[i] < 1.0:
                bounds_viol += 1.0 - martingales[i]
            elif martingales[i] > 100.0:
                bounds_viol += martingales[i] - 100.0
    
    return monotone_viol, volume_sum_viol, bounds_viol


@njit(cache=True, fastmath=True)
def quick_dominance_check(score_a: float, max_need_a: float, var_need_a: float,
                         score_b: float, max_need_b: float, var_need_b: float) -> int:
    """
    Quick dominance check between two candidates.
    
    Args:
        score_a, max_need_a, var_need_a: Metrics for candidate A
        score_b, max_need_b, var_need_b: Metrics for candidate B
        
    Returns:
        1 if A dominates B, -1 if B dominates A, 0 if non-dominated
    """
    # A dominates B if A is better or equal in all objectives and strictly better in at least one
    a_better_score = score_a <= score_b
    a_better_max = max_need_a <= max_need_b
    a_better_var = var_need_a <= var_need_b
    
    a_strictly_better = (score_a < score_b) or (max_need_a < max_need_b) or (var_need_a < var_need_b)
    
    if a_better_score and a_better_max and a_better_var and a_strictly_better:
        return 1  # A dominates B
    
    # Check if B dominates A
    b_better_score = score_b <= score_a
    b_better_max = max_need_b <= max_need_a
    b_better_var = var_need_b <= var_need_a
    
    b_strictly_better = (score_b < score_a) or (max_need_b < max_need_a) or (var_need_b < var_need_a)
    
    if b_better_score and b_better_max and b_better_var and b_strictly_better:
        return -1  # B dominates A
    
    return 0  # Non-dominated


@njit(cache=True, fastmath=True)
def calculate_diversity_distance(volumes_a: np.ndarray, volumes_b: np.ndarray) -> float:
    """
    Calculate diversity distance between two volume distributions.
    
    Args:
        volumes_a, volumes_b: Volume distributions to compare
        
    Returns:
        Distance measure (0 = identical, higher = more different)
    """
    if volumes_a.size != volumes_b.size:
        return 1.0  # Maximum distance for different sizes
    
    # Normalize to probabilities
    sum_a = np.sum(volumes_a)
    sum_b = np.sum(volumes_b)
    
    if sum_a <= 1e-12 or sum_b <= 1e-12:
        return 1.0
    
    prob_a = volumes_a / sum_a
    prob_b = volumes_b / sum_b
    
    # Calculate L2 distance
    diff = prob_a - prob_b
    l2_dist = math.sqrt(np.sum(diff * diff))
    
    return l2_dist
