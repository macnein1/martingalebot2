"""
Constraints and Normalization for DCA/Martingale Optimization
Implements soft penalty approach with proper bounds checking and normalization functions.
"""
from __future__ import annotations

import numpy as np
from numba import njit
from typing import Dict, Any, Tuple
from martingale_lab.core.repair import hard_clip_local_growth, isotonic_non_decreasing
import math


# helpers
def _clip_local_band_forward(vol, start_idx, gmin, gmax):
    """i>=start_idx için v[i] ∈ [v[i-1]*gmin, v[i-1]*gmax] ileri tarama."""
    n = len(vol)
    for i in range(max(1, start_idx), n):
        lo = vol[i-1] * gmin
        hi = vol[i-1] * gmax
        if vol[i] < lo:
            vol[i] = lo
        elif vol[i] > hi:
            vol[i] = hi
    return vol


def _rescale_block(vol, start_idx, target_sum):
    """v[start_idx:] toplamını tek faktörle target_sum yap; prefix'i dokunma."""
    cur = sum(vol[start_idx:])
    if cur <= 0:
        return vol
    f = target_sum / cur
    for i in range(start_idx, len(vol)):
        vol[i] *= f
    return vol


@njit(cache=True, fastmath=True)
def normalize_volumes_softmax(raw_volumes: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """
    Normalize volumes using softmax to ensure sum = 100%.
    
    Args:
        raw_volumes: Raw volume logits
        temperature: Softmax temperature (higher = more uniform)
        
    Returns:
        Normalized volume percentages summing to 100.0
    """
    if raw_volumes.size == 0:
        return np.zeros(1, dtype=np.float64)  # Return single element for numba compatibility
    
    # Apply temperature scaling
    scaled = raw_volumes / max(temperature, 1e-6)
    
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
def monotonic_softplus_for_indents(raw_indents: np.ndarray, overlap_pct: float, 
                                  min_step: float = 0.01) -> np.ndarray:
    """
    Convert raw indents to monotonically increasing cumulative indents using softplus.
    
    Args:
        raw_indents: Raw indent logits
        overlap_pct: Total overlap percentage to scale to
        min_step: Minimum step size percentage
        
    Returns:
        Cumulative indent percentages [0, i1, i2, ..., overlap_pct]
    """
    if raw_indents.size == 0:
        return np.array([0.0], dtype=np.float64)
    
    # Apply softplus to ensure positive steps
    steps = np.empty(raw_indents.size)
    for i in range(raw_indents.size):
        x = raw_indents[i]
        # Softplus with numerical stability
        if x > 20:
            steps[i] = x
        elif x < -20:
            steps[i] = math.exp(x)
        else:
            steps[i] = math.log1p(math.exp(x))
        
        # Ensure minimum step size
        steps[i] = max(steps[i], min_step)
    
    # Normalize steps to sum to overlap_pct
    steps_sum = np.sum(steps)
    if steps_sum <= 1e-12:
        # Fallback to uniform steps
        step_size = overlap_pct / raw_indents.size
        steps = np.full(raw_indents.size, max(step_size, min_step))
    else:
        steps = steps * (overlap_pct / steps_sum)
        # Ensure all steps meet minimum
        for i in range(steps.size):
            steps[i] = max(steps[i], min_step)
    
    # Create cumulative indents starting from 0
    cumulative = np.empty(raw_indents.size + 1)
    cumulative[0] = 0.0
    for i in range(raw_indents.size):
        cumulative[i + 1] = cumulative[i] + steps[i]
    
    return cumulative


@njit(cache=True, fastmath=True)
def sigmoid_martingales(raw_martingales: np.ndarray, min_mart: float = 1.0, 
                       max_mart: float = 100.0) -> np.ndarray:
    """
    Convert raw martingale logits to bounded percentages using sigmoid.
    
    Args:
        raw_martingales: Raw martingale logits
        min_mart: Minimum martingale percentage
        max_mart: Maximum martingale percentage
        
    Returns:
        Martingale percentages with first element = 0, rest in [min_mart, max_mart]
    """
    if raw_martingales.size == 0:
        return np.array([], dtype=np.float64)
    
    martingales = np.empty(raw_martingales.size)
    martingales[0] = 0.0  # First order has no martingale
    
    for i in range(1, raw_martingales.size):
        # Sigmoid to [0, 1]
        sigmoid_val = 1.0 / (1.0 + math.exp(-raw_martingales[i]))
        # Scale to [min_mart, max_mart]
        martingales[i] = min_mart + (max_mart - min_mart) * sigmoid_val
    
    return martingales


@njit(cache=True, fastmath=True)
def assert_increasing(sequence: np.ndarray, tolerance: float = 1e-6) -> float:
    """
    Calculate penalty for non-increasing sequences (soft constraint).
    
    Args:
        sequence: Array that should be non-decreasing
        tolerance: Tolerance for small violations
        
    Returns:
        Penalty value (0 if perfectly increasing)
    """
    if sequence.size <= 1:
        return 0.0
    
    violation = 0.0
    for i in range(1, sequence.size):
        diff = sequence[i-1] - sequence[i]
        if diff > tolerance:
            violation += diff
    
    return violation


@njit(cache=True, fastmath=True)
def assert_bounds(values: np.ndarray, min_val: float, max_val: float) -> float:
    """
    Calculate penalty for values outside bounds (soft constraint).
    
    Args:
        values: Array of values to check
        min_val: Minimum allowed value
        max_val: Maximum allowed value
        
    Returns:
        Penalty value (0 if all values within bounds)
    """
    if values.size == 0:
        return 0.0
    
    violation = 0.0
    for val in values:
        if val < min_val:
            violation += min_val - val
        elif val > max_val:
            violation += val - max_val
    
    return violation


@njit(cache=True, fastmath=True)
def assert_sum_constraint(values: np.ndarray, target_sum: float, tolerance: float = 1e-6) -> float:
    """
    Calculate penalty for sum constraint violation (soft constraint).
    
    Args:
        values: Array of values
        target_sum: Target sum value
        tolerance: Tolerance for small violations
        
    Returns:
        Penalty value (0 if sum matches target within tolerance)
    """
    if values.size == 0:
        return 0.0
    
    actual_sum = np.sum(values)
    diff = abs(actual_sum - target_sum)
    
    if diff <= tolerance:
        return 0.0
    
    return diff


@njit(cache=True, fastmath=True)
def volume_normalization_penalty(volumes: np.ndarray) -> float:
    """
    Calculate penalty for volume normalization deviation from 100%.
    
    Args:
        volumes: Volume percentages
        
    Returns:
        Normalization penalty
    """
    if volumes.size == 0:
        return 0.0
    
    total = np.sum(volumes)
    return abs(total - 100.0)


@njit(cache=True, fastmath=True)
def indent_monotonicity_penalty(indents: np.ndarray) -> float:
    """
    Calculate penalty for non-monotonic indent sequence.
    
    Args:
        indents: Cumulative indent percentages
        
    Returns:
        Monotonicity penalty
    """
    return assert_increasing(indents)


@njit(cache=True, fastmath=True)
def martingale_bounds_penalty(martingales: np.ndarray, min_mart: float = 1.0, 
                             max_mart: float = 100.0) -> float:
    """
    Calculate penalty for martingale values outside bounds.
    
    Args:
        martingales: Martingale percentages
        min_mart: Minimum allowed martingale (for orders > 1)
        max_mart: Maximum allowed martingale
        
    Returns:
        Bounds penalty
    """
    if martingales.size <= 1:
        return 0.0
    
    penalty = 0.0
    
    # First martingale should be 0
    if martingales[0] != 0.0:
        penalty += abs(martingales[0])
    
    # Rest should be in [min_mart, max_mart]
    for i in range(1, martingales.size):
        if martingales[i] < min_mart:
            penalty += min_mart - martingales[i]
        elif martingales[i] > max_mart:
            penalty += martingales[i] - max_mart
    
    return penalty


@njit(cache=True, fastmath=True)
def volume_bounds_penalty(volumes: np.ndarray, min_vol: float = 0.1, 
                         max_vol: float = 80.0) -> float:
    """
    Calculate penalty for volume values outside reasonable bounds.
    
    Args:
        volumes: Volume percentages
        min_vol: Minimum reasonable volume percentage
        max_vol: Maximum reasonable volume percentage
        
    Returns:
        Bounds penalty
    """
    return assert_bounds(volumes, min_vol, max_vol)


@njit(cache=True, fastmath=True)
def indent_bounds_penalty(indents: np.ndarray, overlap_pct: float) -> float:
    """
    Calculate penalty for indent values outside [0, overlap_pct] bounds.
    
    Args:
        indents: Cumulative indent percentages
        overlap_pct: Maximum allowed overlap
        
    Returns:
        Bounds penalty
    """
    return assert_bounds(indents, 0.0, overlap_pct)


@njit(cache=True, fastmath=True)
def tail_cap_penalty(volumes: np.ndarray, max_last_pct: float = 40.0) -> float:
    """
    Calculate penalty for excessive volume concentration in last order.
    
    Args:
        volumes: Volume percentages
        max_last_pct: Maximum allowed percentage for last order
        
    Returns:
        Tail cap penalty
    """
    if volumes.size == 0:
        return 0.0
    
    last_volume = volumes[-1]
    if last_volume > max_last_pct:
        return (last_volume - max_last_pct) * 2.0  # Strong penalty
    
    return 0.0


@njit(cache=True, fastmath=True)
def head_cap_penalty(volumes: np.ndarray, max_first_pct: float = 60.0) -> float:
    """
    Calculate penalty for excessive volume concentration in first order.
    
    Args:
        volumes: Volume percentages
        max_first_pct: Maximum allowed percentage for first order
        
    Returns:
        Head cap penalty
    """
    if volumes.size == 0:
        return 0.0
    
    first_volume = volumes[0]
    if first_volume > max_first_pct:
        return (first_volume - max_first_pct) * 1.5  # Moderate penalty
    
    return 0.0


def compute_all_constraint_penalties(indents: np.ndarray, volumes: np.ndarray, 
                                   martingales: np.ndarray, overlap_pct: float,
                                   config: Dict[str, Any]) -> Dict[str, float]:
    """
    Compute all constraint penalties for a candidate solution.
    
    Args:
        indents: Cumulative indent percentages
        volumes: Volume percentages
        martingales: Martingale percentages
        overlap_pct: Maximum overlap percentage
        config: Configuration dictionary
        
    Returns:
        Dictionary of constraint penalties
    """
    # Extract configuration with defaults
    min_vol = config.get('min_volume_pct', 0.1)
    max_vol = config.get('max_volume_pct', 80.0)
    min_mart = config.get('min_martingale_pct', 1.0)
    max_mart = config.get('max_martingale_pct', 100.0)
    max_last_vol = config.get('tail_cap_pct', 40.0)
    max_first_vol = config.get('head_cap_pct', 60.0)
    
    penalties = {}
    
    # Volume constraints
    penalties['volume_normalization'] = volume_normalization_penalty(volumes)
    penalties['volume_bounds'] = volume_bounds_penalty(volumes, min_vol, max_vol)
    penalties['tail_cap'] = tail_cap_penalty(volumes, max_last_vol)
    penalties['head_cap'] = head_cap_penalty(volumes, max_first_vol)
    
    # Indent constraints
    penalties['indent_monotonicity'] = indent_monotonicity_penalty(indents)
    penalties['indent_bounds'] = indent_bounds_penalty(indents, overlap_pct)
    
    # Martingale constraints
    penalties['martingale_bounds'] = martingale_bounds_penalty(martingales, min_mart, max_mart)
    
    return penalties


def apply_soft_constraints(raw_indents: np.ndarray, raw_volumes: np.ndarray, 
                          raw_martingales: np.ndarray, overlap_pct: float,
                          config: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]:
    """
    Apply soft constraints and normalization to raw parameters.
    
    Args:
        raw_indents: Raw indent logits
        raw_volumes: Raw volume logits
        raw_martingales: Raw martingale logits
        overlap_pct: Maximum overlap percentage
        config: Configuration dictionary
        
    Returns:
        Tuple of (normalized_indents, normalized_volumes, normalized_martingales, penalties)
    """
    # Extract configuration
    softmax_temp = config.get('softmax_temperature', 1.0)
    min_step = config.get('min_indent_step', 0.01)
    min_mart = config.get('min_martingale_pct', 1.0)
    max_mart = config.get('max_martingale_pct', 100.0)
    
    # Apply normalization
    volumes = normalize_volumes_softmax(raw_volumes, softmax_temp)
    indents = monotonic_softplus_for_indents(raw_indents, overlap_pct, min_step)
    martingales = sigmoid_martingales(raw_martingales, min_mart, max_mart)
    
    # Calculate constraint penalties
    penalties = compute_all_constraint_penalties(indents, volumes, martingales, overlap_pct, config)
    
    return indents, volumes, martingales, penalties


@njit(cache=True, fastmath=True)
def total_constraint_penalty(penalties: Dict[str, float], weights: Dict[str, float]) -> float:
    """
    Calculate total weighted constraint penalty.
    
    Args:
        penalties: Dictionary of individual penalties
        weights: Dictionary of penalty weights
        
    Returns:
        Total weighted penalty
    """
    total = 0.0
    
    # Apply weights to penalties
    for key in penalties:
        weight = weights.get(key, 1.0)
        total += weight * penalties[key]
    
    return total


# Default constraint weights
DEFAULT_CONSTRAINT_WEIGHTS = {
    'volume_normalization': 10.0,    # Critical: volumes must sum to 100
    'volume_bounds': 2.0,            # Important: reasonable volume ranges
    'tail_cap': 3.0,                 # Important: prevent tail concentration
    'head_cap': 1.0,                 # Moderate: prevent head concentration
    'indent_monotonicity': 5.0,      # Critical: indents must be increasing
    'indent_bounds': 2.0,            # Important: indents within overlap
    'martingale_bounds': 1.0,        # Moderate: martingale ranges
}


def validate_candidate_hard(indents: np.ndarray, volumes: np.ndarray, 
                           martingales: np.ndarray, overlap_pct: float,
                           tolerance: float = 1e-3) -> Tuple[bool, str]:
    """
    Hard validation check for candidate feasibility.
    
    Args:
        indents: Cumulative indent percentages
        volumes: Volume percentages
        martingales: Martingale percentages
        overlap_pct: Maximum overlap percentage
        tolerance: Tolerance for numerical errors
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    # Check volume sum
    volume_sum = float(np.sum(volumes))
    if abs(volume_sum - 100.0) > tolerance:
        return False, f"Volume sum {volume_sum:.3f} != 100.0"
    
    # Check indent monotonicity
    for i in range(1, len(indents)):
        if indents[i] < indents[i-1] - tolerance:
            return False, f"Non-monotonic indents at position {i}"
    
    # Check indent bounds
    if np.any(indents < -tolerance) or np.any(indents > overlap_pct + tolerance):
        return False, f"Indents outside [0, {overlap_pct}] bounds"
    
    # Check martingale first element
    if abs(martingales[0]) > tolerance:
        return False, f"First martingale {martingales[0]:.3f} != 0.0"
    
    # Check martingale bounds (for orders > 1)
    for i in range(1, len(martingales)):
        if martingales[i] < 1.0 - tolerance or martingales[i] > 100.0 + tolerance:
            return False, f"Martingale {i} value {martingales[i]:.3f} outside [1, 100] bounds"
    
    return True, "Valid"


def enforce_schedule_shape_fixed(
    indent_pct: list,
    volume_pct: list,
    base_price: float,
    first_volume_target: float = 1.0,  # Changed from 0.01 to 1.0
    first_indent_target: float = 0.0,
    k_front: int = 3,
    front_cap: float = 5.0,
    g_min: float = 1.01,
    g_max: float = 1.20,
    # New parameters for HC constraints
    m2_min: float = 0.10,
    m2_max: float = 1.00,
    m_min: float = 0.05,
    m_max: float = 1.00,
    firstK_min: float = 1.0,
    eps_inc: float = 1e-5,
    # Additional new parameters
    second_upper_c2: float = 2.0,
    m_head: float = 0.40,
    m_tail: float = 0.20,
    tau_scale: float = 1/3,
    slope_cap: float = 0.25,
    q1_cap: float = 22.0,
    tail_floor: float = 32.0,
    # Head budget parameters (new)
    head_budget_pct: float = 2.0,
    use_head_budget: bool = False,
    use_hc0_bootstrap: bool = True,
) -> Tuple[list, list, list, list, list, list, Dict[str, Any]]:
    """
    Enforce fixed-first-order and shaped martingale band on a schedule with HC0-HC7 pipeline.
    
    Pipeline steps:
    0. HC0: Bootstrap feasible tail if needed
    1. Initialize: v0=0.01, indent0=0.00, normalize with tail_only_rescale
    2. Compute m = compute_m_from_v(v)
    3. HC1: v1 band [1.10*v0, second_upper_c2*v0], then tail_only_rescale
    4. HC3: martingale bands with decaying ceiling, then rechain and tail_only_rescale
    5. HC4: slope limits for smoothness, then rechain and tail_only_rescale
    6. HC2: strict monotonicity v[i] >= v[i-1] + eps_inc
    7. Tail-only rescale
    8. HC5: mass control (Q1 cap, Q4 floor)
    9. HC6: plateau breaker
    10. Compute derivatives

    Args:
        indent_pct: cumulative indent percentages per order (length M)
        volume_pct: volume percentages per order (length M, sums ~100)
        base_price: base price to compute order_prices and needpct
        first_volume_target: fixed first order volume percentage
        first_indent_target: fixed first order indent percentage (usually 0.0)
        k_front: number of initial orders considered "front"
        front_cap: maximum total volume percentage allowed in the first k_front orders
        g_min: minimum allowed growth ratio for i>=2 (vol[i]/vol[i-1])
        g_max: maximum allowed growth ratio for i>=2
        second_upper_c2: upper bound multiplier for v1 (v1 <= c2*v0)
        m2_min/m2_max: bounds for m[2]
        m_min/m_max: general bounds for m[i] where i>=3
        m_head/m_tail: parameters for decaying ceiling
        tau_scale: decay rate scale (tau = N * tau_scale)
        slope_cap: maximum |Δm| for i>=3
        q1_cap: first quartile volume cap
        tail_floor: last quartile volume floor
        eps_inc: minimum increment for strict monotonicity
        head_budget_pct: target head budget percentage when use_head_budget=True
        use_head_budget: whether to apply head budget redistribution
        use_hc0_bootstrap: whether to apply HC0 bootstrap for feasibility

    Returns:
        (repaired_indent_pct, repaired_volume_pct, martingale_pct, needpct,
         order_prices, price_step_pct, diagnostics)
    """
    from martingale_lab.core.repair import (
        tail_only_rescale_keep_first_two,
        compute_m_from_v,
        rechain_v_from_m,
        longest_plateau_run,
        bootstrap_tail_from_bands
    )
    
    # Initialize logger
    import logging
    logger = logging.getLogger(__name__)

    M = int(len(volume_pct))
    if M == 0:
        return [], [], [], [], [], [], {
            "clipped_frac": 0.0,
            "front_excess_before": 0.0,
            "front_excess_after": 0.0,
            "tv_before": 0.0,
            "tv_after": 0.0,
            "l1_change": 0.0,
            "band_clips": 0,
        }

    ind = np.asarray(indent_pct, dtype=np.float64).copy()
    vol_in = np.asarray(volume_pct, dtype=np.float64).copy()

    # Diagnostics baselines
    def _total_variation(v: np.ndarray) -> float:
        return float(np.sum(np.abs(np.diff(v)))) if v.size > 1 else 0.0

    tv_before = _total_variation(vol_in)
    front_sum_before = float(np.sum(vol_in[: min(k_front, M)]))

    # Helper: decaying ceiling function
    def _decay_ceiling(i: int, N: int, m_head: float, m_tail_cap: float, tau_scale: float) -> float:
        """Compute decaying ceiling for martingale at position i."""
        tau = max(1.0, N * tau_scale)
        return m_tail_cap + (m_head - m_tail_cap) * np.exp(-(i - 2) / tau)

    # Initialize volumes
    vol = vol_in.copy()
    
    # Check if initial volumes are extremely unbalanced
    if M > 2:
        initial_sum = np.sum(vol)
        if initial_sum > 0:
            vol_normalized = vol / initial_sum * 100.0
            # If first two orders take more than 50% of total, we have a problem
            if vol_normalized[0] + vol_normalized[1] > 50.0:
                logger.warning(f"Initial volumes extremely unbalanced: v0+v1={vol_normalized[0]+vol_normalized[1]:.1f}%")
                # Create a reasonable geometric progression
                vol = np.zeros(M)
                vol[0] = first_volume_target
                vol[1] = first_volume_target * 1.5  # v1 = 1.5 * v0
                # Geometric growth for the rest
                growth_rate = 1.15
                for i in range(2, M):
                    vol[i] = vol[i-1] * growth_rate
                # Normalize to sum=100
                vol = vol / np.sum(vol) * 100.0
                # Fix v0
                vol[0] = first_volume_target
                tail_only_rescale_keep_first_two(vol)
                logger.info(f"Using fallback geometric progression: v[:5]={vol[:5].round(3).tolist()}")
    
    # Step 0: HC0 - Bootstrap feasible tail if needed
    hc0_applied = False
    if use_hc0_bootstrap and M > 2:
        # Check if current volumes would create infeasible m2
        v0_target = first_volume_target
        v1_current = vol[1] if M > 1 else v0_target * 2.0
        
        # Apply HC0 bootstrap to get feasible starting point
        vol_bootstrap = bootstrap_tail_from_bands(
            v0_target, v1_current, M,
            m2_min, m2_max, m_min,
            m_head, m_tail, tau_scale
        )
        
        # Use bootstrapped volumes as starting point
        vol = vol_bootstrap
        hc0_applied = True
        
        # Debug log for HC0
        logger.debug(f"HC0: Applied bootstrap, v[:5]={vol[:5].round(3).tolist() if M >= 5 else vol.round(3).tolist()}")
    
    # Step 1: Initialize v0 and indent0, normalize with tail_only_rescale
    if M >= 1:
        ind[0] = first_indent_target
        vol[0] = first_volume_target
    
    if M > 1:
        # Keep v1 from input initially (or from bootstrap), will adjust in HC1
        tail_only_rescale_keep_first_two(vol)
        
        # Immediately enforce v[2] bounds if it exists
        if M > 2:
            v2_min = vol[1] * (1.0 + m2_min)
            v2_max = vol[1] * (1.0 + m2_max)
            if vol[2] < v2_min or vol[2] > v2_max:
                vol[2] = np.clip(vol[2], v2_min, v2_max)
                tail_only_rescale_keep_first_two(vol)
        
        logger.debug(f"HC1: After init, v[:5]={vol[:5].round(3).tolist() if M >= 5 else vol.round(3).tolist()}")
    
    # Apply head budget if requested
    head_budget_applied = False
    if use_head_budget and M > 2:
        # Calculate current head sum
        head_sum = vol[0] + vol[1]
        target_head = min(head_budget_pct, head_sum + 1.0)  # Allow small increase
        
        if target_head > head_sum:
            # Redistribute small amount to v[2] to ease m2 constraint
            delta = target_head - head_sum
            vol[2] += delta
            # Rescale rest of tail to maintain sum = 100
            tail_only_rescale_keep_first_two(vol)
            head_budget_applied = True

    # Tracking variables for diagnostics
    band_clips = 0
    v1_band_applied = False
    m2_clip_applied = False
    decaying_clips = 0
    slope_clips = 0
    
    # Step 2: Compute initial m
    m = compute_m_from_v(vol)
    
    # Step 3: HC1 - v1 band constraint
    if M > 1:
        v1_min = 1.10 * vol[0]
        v1_max = second_upper_c2 * vol[0]
        v1_original = vol[1]
        vol[1] = np.clip(vol[1], v1_min, v1_max)
        if vol[1] != v1_original:
            v1_band_applied = True
            band_clips += 1
        tail_only_rescale_keep_first_two(vol)
        m = compute_m_from_v(vol)
        
        logger.debug(f"HC1: v1 band applied, v[:5]={vol[:5].round(3).tolist() if M >= 5 else vol.round(3).tolist()}")
    
    # Step 4: HC3 - Martingale bands with decaying ceiling
    if M > 2:
        # First ensure v[2] is within m2 bounds
        v2_min = vol[1] * (1.0 + m2_min)
        v2_max = vol[1] * (1.0 + m2_max)
        vol[2] = np.clip(vol[2], v2_min, v2_max)
        
        # Compute m after v[2] adjustment
        m = compute_m_from_v(vol)
        
        # Track if we clipped m[2]
        if abs(m[2] - np.clip(m[2], m2_min, m2_max)) > 1e-6:
            m2_clip_applied = True
            band_clips += 1
        
        # Ensure m[2] is within bounds (defensive)
        m[2] = np.clip(m[2], m2_min, m2_max)
        
        # Apply decaying ceiling for i>=3
        for i in range(3, M):
            m_max_i = _decay_ceiling(i, M, m_head, m_tail, tau_scale)
            m_original = m[i]
            # Ensure m[i] is not negative and within bounds
            m[i] = np.clip(m[i], m_min, min(m_max_i, m_max))
            if abs(m[i] - m_original) > 1e-6:
                decaying_clips += 1
                band_clips += 1
        
        # Rechain volumes from adjusted m
        vol = rechain_v_from_m(vol[0], vol[1], m)
        
        # Now rescale carefully to maintain sum=100 while preserving v0, v1
        tail_only_rescale_keep_first_two(vol)
        
        # CRITICAL FIX: After rescaling, m[2] might be out of bounds again
        # Force v[2] to respect m2 bounds relative to the fixed v[1]
        m_check = compute_m_from_v(vol)
        if M > 2 and (m_check[2] < m2_min - 0.01 or m_check[2] > m2_max + 0.01):
            logger.warning(f"HC3: m[2]={m_check[2]:.3f} out of bounds after rescale, fixing...")
            # Directly set v[2] based on m2 bounds
            # Use the midpoint of m2 range if current is out of bounds
            if m_check[2] < m2_min:
                m2_target = m2_min
            elif m_check[2] > m2_max:
                m2_target = m2_max
            else:
                m2_target = m_check[2]
            
            # Set v[2] directly
            vol[2] = vol[1] * (1.0 + m2_target)
            
            # Now we need to rescale v[3:] to maintain sum=100
            if M > 3:
                tail_sum_target = 100.0 - vol[0] - vol[1] - vol[2]
                tail_sum_current = np.sum(vol[3:])
                if tail_sum_current > 0 and tail_sum_target > 0:
                    scale_factor = tail_sum_target / tail_sum_current
                    vol[3:] *= scale_factor
                elif tail_sum_target > 0:
                    # Distribute evenly if tail is zero
                    vol[3:] = tail_sum_target / (M - 3)
        
        # Final m computation
        m = compute_m_from_v(vol)
        logger.debug(f"HC3: After martingale bands, v[:5]={vol[:5].round(3).tolist() if M >= 5 else vol.round(3).tolist()}, m2={m[2]:.3f}")
    
    # Step 5: HC4 - Slope limits for smoothness
    if M > 3:
        # m should already be computed from HC3
        
        logger.debug(f"HC4: Before slope limit, m[2:6]={m[2:6].round(3).tolist() if len(m) >= 6 else m[2:].round(3).tolist()}")
        
        # Apply slope limits more carefully
        for i in range(3, M):
            m_prev = m[i-1]
            # Ensure we don't create negative or huge m values
            m_min_slope = max(m_min, m_prev - slope_cap)
            m_max_slope = min(m_max, m_prev + slope_cap)
            m_original = m[i]
            m[i] = np.clip(m[i], m_min_slope, m_max_slope)
            if abs(m[i] - m_original) > 1e-6:
                slope_clips += 1
        
        logger.debug(f"HC4: After slope limit, m[2:6]={m[2:6].round(3).tolist() if len(m) >= 6 else m[2:].round(3).tolist()}")
        
        # Rechain volumes from adjusted m
        vol = rechain_v_from_m(vol[0], vol[1], m)
        
        # Rescale to maintain sum=100
        tail_only_rescale_keep_first_two(vol)
        
        # Update m for next steps
        m = compute_m_from_v(vol)
        
        logger.debug(f"HC4: After slope limit, v[:5]={vol[:5].round(3).tolist() if M >= 5 else vol.round(3).tolist()}, m2={m[2]:.3f}")
    
    # Step 6: HC2 - Strict monotonicity (respecting m2_min for v[2])
    for i in range(1, M):
        if i == 2 and M > 2:
            # For v[2], ensure it respects both monotonicity and m2_min
            min_vol_mono = vol[1] + eps_inc
            min_vol_m2 = vol[1] * (1.0 + m2_min)
            min_vol = max(min_vol_mono, min_vol_m2)
        else:
            min_vol = vol[i-1] + eps_inc
        
        if vol[i] < min_vol:
            vol[i] = min_vol
    
    # Step 7: Final tail-only rescale
    tail_only_rescale_keep_first_two(vol)
    
    # CRITICAL: After HC2 rescale, ensure m[2] is still valid
    if M > 2:
        m_check = compute_m_from_v(vol)
        if m_check[2] < m2_min:
            logger.debug(f"HC2: m[2]={m_check[2]:.3f} too small after rescale, fixing...")
            # Force v[2] to respect m2_min
            vol[2] = vol[1] * (1.0 + m2_min)
            
            # Rescale the rest of the tail to maintain sum=100
            if M > 3:
                tail_sum_target = 100.0 - vol[0] - vol[1] - vol[2]
                tail_sum_current = np.sum(vol[3:])
                if tail_sum_current > 0 and tail_sum_target > 0:
                    scale_factor = tail_sum_target / tail_sum_current
                    vol[3:] *= scale_factor
                elif tail_sum_target > 0:
                    # Distribute evenly if tail is zero
                    vol[3:] = tail_sum_target / (M - 3)
    
    logger.debug(f"HC2: After strict monotonicity, v[:5]={vol[:5].round(3).tolist() if M >= 5 else vol.round(3).tolist()}")
    
    # Step 8: HC5 - Mass control (Q1 cap, Q4 floor)
    Q1 = min(M, max(1, int(np.ceil(M / 4.0))))
    Q4 = min(M, max(1, int(np.ceil(M / 4.0))))
    
    # Calculate current shares
    q1_sum = float(np.sum(vol[:Q1]))
    q4_start = max(0, M - Q4)
    q4_sum = float(np.sum(vol[q4_start:]))
    
    # Apply Q1 cap
    if q1_sum > q1_cap and Q1 > 2:
        # Scale down orders [2..Q1-1] to meet cap
        front2_sum = float(np.sum(vol[2:Q1]))
        if front2_sum > 1e-12:
            target_front2 = max(0.0, q1_cap - vol[0] - vol[1])
            scale_front2 = target_front2 / front2_sum
            vol[2:Q1] *= scale_front2
            # Add excess to tail
            excess = front2_sum - target_front2
            tail_sum = float(np.sum(vol[Q1:]))
            if tail_sum > 1e-12:
                vol[Q1:] *= (tail_sum + excess) / tail_sum
    
    # Apply Q4 floor
    if q4_sum < tail_floor and q4_start < M:
        # Scale up tail to meet floor
        deficit = tail_floor - q4_sum
        # Try to take from middle section
        if Q1 < q4_start:
            mid_sum = float(np.sum(vol[Q1:q4_start]))
            if mid_sum > deficit:
                vol[Q1:q4_start] *= (mid_sum - deficit) / mid_sum
                vol[q4_start:] *= tail_floor / q4_sum
        else:
            # Take from front2 if no middle section
            if Q1 > 2:
                front2_sum = float(np.sum(vol[2:Q1]))
                if front2_sum > deficit:
                    vol[2:Q1] *= (front2_sum - deficit) / front2_sum
                    vol[q4_start:] *= tail_floor / q4_sum
    
    # Final rescale
    tail_only_rescale_keep_first_two(vol)
    
    # Recalculate shares after mass control
    q1_share = float(np.sum(vol[:Q1])) if Q1 > 0 else 0.0
    q4_share = float(np.sum(vol[q4_start:])) if q4_start < M else 0.0
    
    logger.debug(f"HC5: After mass control, q1={q1_share:.1f}%, q4={q4_share:.1f}%")
    
    # Recalculate m after mass control
    m = compute_m_from_v(vol)

    # Step 9: HC6 - Plateau breaker
    plateau_max_run, plateau_start = longest_plateau_run(m, center=1.0, tol=0.02, start_idx=2)
    if plateau_max_run > 3 and plateau_start >= 2:
        # Apply alternating perturbations to break plateau
        delta = min(0.01, slope_cap / 2)  # Small perturbation within slope limits
        for i in range(plateau_start, min(plateau_start + plateau_max_run, M)):
            if (i - plateau_start) % 2 == 0:
                m[i] = min(m[i] + delta, _decay_ceiling(i, M, m_head, m_tail, tau_scale))
            else:
                m[i] = max(m[i] - delta, m_min)
        
        # Rechain and rescale
        vol = rechain_v_from_m(vol[0], vol[1], m)
        tail_only_rescale_keep_first_two(vol)
        m = compute_m_from_v(vol)
        
        # Recalculate turn count
        turn_count = 0
        for i in range(3, M):
            if (m[i] - 1.0) * (m[i-1] - 1.0) < 0:
                turn_count += 1

        logger.debug(f"HC6: After plateau breaker, plateau_max_run={plateau_max_run_final}")
    
    # Additional wave enforcement if not enough turns
    if M > 6:
        # Count current turns
        turn_count_current = 0
        for i in range(3, M):
            prev_sign = np.sign(m[i-1] - 1.0) if abs(m[i-1] - 1.0) > 0.01 else 0
            curr_sign = np.sign(m[i] - 1.0) if abs(m[i] - 1.0) > 0.01 else 0
            if prev_sign != 0 and curr_sign != 0 and prev_sign != curr_sign:
                turn_count_current += 1
        
        # If not enough turns, add some wave pattern
        if turn_count_current < 2:
            # Apply a gentle wave pattern
            wave_period = max(3, M // 4)
            for i in range(3, M):
                phase = ((i - 3) % wave_period) / float(wave_period)
                wave_val = np.sin(2 * np.pi * phase)
                # Add small wave perturbation
                delta = 0.05 * wave_val
                m_max_i = _decay_ceiling(i, M, m_head, m_tail, tau_scale)
                m[i] = np.clip(m[i] + delta, m_min, m_max_i)
            
            # Rechain and rescale
            vol = rechain_v_from_m(vol[0], vol[1], m)
            tail_only_rescale_keep_first_two(vol)
            m = compute_m_from_v(vol)

            # Recalculate turn count
            turn_count = 0
            for i in range(3, M):
                if (m[i] - 1.0) * (m[i-1] - 1.0) < 0:
                    turn_count += 1

            logger.debug(f"HC7: After wave enforcement, turn_count={turn_count}")

    # Final check: ensure v0 is still fixed
    vol[0] = first_volume_target
    
    # FINAL VALIDATION: Ensure m[2] is within bounds
    if M > 2:
        m_final = compute_m_from_v(vol)
        if m_final[2] < m2_min - 0.001 or m_final[2] > m2_max + 0.001:
            logger.warning(f"FINAL: m[2]={m_final[2]:.3f} out of bounds, forcing correction")
            # Force v[2] to be at the appropriate bound
            if m_final[2] < m2_min:
                # Set to m2_min
                target_m2 = m2_min
            else:
                # Set to m2_max
                target_m2 = m2_max
            
            vol[2] = vol[1] * (1.0 + target_m2)
            
            # Rescale tail to maintain sum=100
            if M > 3:
                tail_sum_target = 100.0 - vol[0] - vol[1] - vol[2]
                if tail_sum_target > 0:
                    tail_sum_current = np.sum(vol[3:])
                    if tail_sum_current > 0:
                        vol[3:] *= tail_sum_target / tail_sum_current
                    else:
                        # Initialize tail with geometric progression
                        for i in range(3, M):
                            vol[i] = vol[i-1] * (1.0 + m_min)
                        tail_sum_current = np.sum(vol[3:])
                        if tail_sum_current > 0:
                            vol[3:] *= tail_sum_target / tail_sum_current
            
            # Ensure monotonicity is maintained after rescaling
            for i in range(3, M):
                min_val = vol[i-1] + eps_inc
                if vol[i] < min_val:
                    vol[i] = min_val
            
            # Final normalization to ensure sum=100
            total_sum = np.sum(vol)
            if abs(total_sum - 100.0) > 0.01:
                # Keep v0 and v1 fixed, rescale the rest
                tail_sum = np.sum(vol[2:])
                if tail_sum > 0:
                    target_tail = 100.0 - vol[0] - vol[1]
                    vol[2:] *= target_tail / tail_sum
                    
            # Double-check m[2] after all adjustments
            m_final = compute_m_from_v(vol)
            if M > 2:
                logger.debug(f"FINAL: After correction, m[2]={m_final[2]:.3f}")
    
    # Repair indents to be non-decreasing
    for i in range(1, len(ind)):
        if ind[i] < ind[i-1]:
            ind[i] = ind[i-1]
    
    # Step 10: Compute derivatives
    # Recompute martingale percentages
    mart = np.zeros(M, dtype=np.float64)
    for i in range(1, M):
        mart[i] = m[i] * 100.0
    
    # Build indent cumulative (prepend 0)
    indent_cum = np.concatenate([[0.0], ind.astype(np.float64)])
    
    # Recompute order prices from base price and indents
    order_prices = np.empty(M + 1, dtype=np.float64)
    order_prices[0] = base_price
    for i in range(1, M + 1):
        if i < len(indent_cum):
            order_prices[i] = base_price * (1.0 - indent_cum[i] / 100.0)
        else:
            order_prices[i] = base_price * (1.0 - indent_cum[-1] / 100.0)
    
    # Price steps
    price_step_pct = np.diff(indent_cum)
    
    # Recompute needpct
    needpct = np.empty(M, dtype=np.float64)
    vol_acc = 0.0
    val_acc = 0.0
    for i in range(M):
        vol_acc += vol[i]
        val_acc += vol[i] * order_prices[i + 1]
        avg_price = val_acc / vol_acc if vol_acc > 1e-12 else base_price
        needpct[i] = ((base_price - avg_price) / base_price) * 100.0
    
    # Calculate diagnostic metrics
    tv_after = _total_variation(vol)
    front_sum_after = float(np.sum(vol[:min(k_front, M)]))
    l1_change = float(np.sum(np.abs(vol - vol_in)))
    clipped_frac = float(band_clips) / float(max(1, M - 1))
    
    # Calculate additional metrics
    first3_sum = float(np.sum(vol[:min(3, M)]))
    q1_share = float(np.sum(vol[:Q1]))
    q4_share = float(np.sum(vol[q4_start:]))
    
    # Calculate std_m for tail diversity
    std_m = 0.0
    if M > 2:
        m_tail = m[2:]
        if len(m_tail) > 0:
            std_m = float(np.std(m_tail))
    
    # Count sign changes (turns) in m
    turn_count = 0
    if M > 3:
        for i in range(3, M):
            prev_sign = np.sign(m[i-1] - 1.0)
            curr_sign = np.sign(m[i] - 1.0)
            if prev_sign != 0 and curr_sign != 0 and prev_sign != curr_sign:
                turn_count += 1
    
    # Update plateau info after potential breaking
    plateau_max_run_final, _ = longest_plateau_run(m, center=1.0, tol=0.02, start_idx=2)
    
    diagnostics = {
        "clipped_frac": clipped_frac,
        "band_clips": int(band_clips),
        "front_excess_before": float(max(0.0, front_sum_before - front_cap)),
        "front_excess_after": float(max(0.0, front_sum_after - front_cap)),
        "tv_before": float(tv_before),
        "tv_after": float(tv_after),
        "l1_change": float(l1_change),
        "first3_sum": first3_sum,
        "v1_band_applied": v1_band_applied,
        "m2_clip_applied": m2_clip_applied,
        "decaying_clips_count": decaying_clips,
        "slope_clips_count": slope_clips,
        "q1_share": q1_share,
        "q4_share": q4_share,
        "plateau_max_run": plateau_max_run_final,
        "std_m": std_m,
        "turn_count": turn_count,
        "v0": float(vol[0]),
        "v1": float(vol[1]),
        "m2": float(m[2]) if M > 2 else 0.0,
        "hc0_applied": hc0_applied,
        "head_budget_applied": head_budget_applied,
    }
    
    return (
        ind.tolist(),
        vol.tolist(),
        mart.tolist(),
        needpct.tolist(),
        order_prices.tolist(),
        price_step_pct.tolist(),
        diagnostics,
    )
