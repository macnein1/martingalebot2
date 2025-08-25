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
    first_volume_target: float = 0.01,
    first_indent_target: float = 0.0,
    k_front: int = 3,
    front_cap: float = 5.0,
    g_min: float = 1.01,
    g_max: float = 1.20,
    g_min_post: float = 1.01,
    g_max_post: float = 1.30,
    isotonic_tail: bool = True,
) -> Tuple[list, list, list, list, list, list, Dict[str, Any]]:
    """
    Enforce fixed-first-order and shaped martingale band on a schedule, then renormalize.

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

    Returns:
        (repaired_indent_pct, repaired_volume_pct, martingale_pct, needpct,
         order_prices, price_step_pct, diagnostics)
    """
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

    # Ensure first indent and first volume fixed
    if M >= 1:
        ind[0] = first_indent_target
    vol = np.zeros(M, dtype=np.float64)
    vol[0] = first_volume_target

    # Compute input ratios g_i from provided volumes to guide repair
    g_input = np.ones(M, dtype=np.float64)
    for i in range(1, M):
        if vol_in[i-1] > 1e-12:
            g_input[i] = vol_in[i] / max(vol_in[i-1], 1e-12)
        else:
            g_input[i] = 1.0

    # Apply hard constraints on ratios and rebuild volumes while keeping v0 fixed
    band_clips = 0
    for i in range(1, M):
        if i == 1:
            g = min(g_input[i], 1.0)  # second order must be <= 1.0 (<=100%)
            if g < 0.0:
                g = 0.0
            if not np.isclose(g, g_input[i]):
                band_clips += 1
        else:
            g = g_input[i]
            if g < g_min:
                g = g_min
                band_clips += 1
            elif g > g_max:
                g = g_max
                band_clips += 1
        vol[i] = vol[i-1] * g

    # Renormalize keeping vol[0] and vol[1] fixed exactly
    if M > 1:
        rest2_sum = float(np.sum(vol[2:]))
        target_rest2 = max(0.0, 100.0 - vol[0] - vol[1])
        if rest2_sum > 1e-12:
            scale_rest2 = target_rest2 / rest2_sum
            vol[2:] *= scale_rest2
        else:
            # Distribute remaining mass uniformly across 2..M-1
            if M > 2:
                vol[2:] = target_rest2 / (M - 2)
            else:
                # M == 2: assign remainder to v1 (cannot satisfy g2<=1 and sum=100 simultaneously)
                vol[1] = target_rest2

    # Enforce front cap while keeping vol[0] and vol[1] fixed when possible; scale indices 2..k_front-1
    kf = min(k_front, M)
    if kf >= 1:
        target_front_sum = min(front_cap, 100.0)
        if kf == 1:
            # only v0 in front; nothing to scale. If over cap, cannot reduce v0; tail will handle sum.
            pass
        else:
            front_sum = float(np.sum(vol[:kf]))
            if front_sum > target_front_sum + 1e-12:
                current_front_rest2 = float(np.sum(vol[2:kf])) if kf > 2 else 0.0
                target_front_rest2 = max(0.0, target_front_sum - vol[0] - vol[1])
                if current_front_rest2 > 1e-12:
                    s_front2 = target_front_rest2 / current_front_rest2
                    vol[2:kf] *= s_front2
                # Now scale tail (kf..M-1) to keep total 100
                tail_sum = float(np.sum(vol[kf:]))
                needed_tail = max(0.0, 100.0 - (vol[0] + vol[1] + float(np.sum(vol[2:kf]))))
                if tail_sum > 1e-12:
                    s_tail = needed_tail / tail_sum
                    vol[kf:] *= s_tail
                else:
                    # Push residual to last front component (prefer kf-1, but keep v0/v1 fixed)
                    if kf > 2:
                        vol[kf-1] = max(0.0, 100.0 - (vol[0] + vol[1] + float(np.sum(vol[2:kf-1]))))

    # Final small normalization to correct rounding drift
    total = float(np.sum(vol))
    if total > 1e-9:
        vol *= 100.0 / total

    # 0) v0 sabit, v1 = min(v1, v0) zaten yapılıyor
    vol[0] = first_volume_target
    if M > 1:
        vol[1] = min(vol[1], vol[0])

    # 1) tail hedef toplam (v2..n-1)
    S_tail = 100.0 - vol[0] - vol[1]

    # 2) önce i>=2 ham tail'i bu toplamla normalize et (tek faktör)
    if M > 2:
        vol = _rescale_block(vol, start_idx=2, target_sum=S_tail)

    # 3) g2'yi kesin sınırla: v2 ∈ [v1*gmin_post, v1*gmax_post]
    if M > 2:
        lo2 = vol[1] * g_min_post
        hi2 = vol[1] * g_max_post
        vol[2] = min(max(vol[2], lo2), hi2)

    # 4) v3..n-1'i S₃ = S_tail - v2 hedefine ölçekle (v2'ye dokunma → g2 korunur)
    if M > 3:
        S3 = S_tail - vol[2]
        if S3 > 0:
            vol = _rescale_block(vol, start_idx=3, target_sum=S3)

        # 5) ölçeklemeden bozulan yerel bantları i>=3 için tekrar kliple
        vol = _clip_local_band_forward(vol, start_idx=3, gmin=g_min_post, gmax=g_max_post)

        # 6) bir kez daha yalnızca i>=3 bloğunu S3 hedefine rescale et (genelde tek iter yeter)
        vol = _rescale_block(vol, start_idx=3, target_sum=S3)

    # (opsiyonel) isotonic tail istiyorsan burada i>=3'e uygula, ardından YİNE start_idx=3 rescale
    if isotonic_tail and M > 3:
        tail_iso = isotonic_non_decreasing(vol[3:])
        vol[3:] = tail_iso
        # Scale tail back to preserve sum 100 with fixed v0,v1,v2
        S3 = S_tail - vol[2]
        if S3 > 0:
            vol = _rescale_block(vol, start_idx=3, target_sum=S3)

    # Repair indents to be non-decreasing and anchored at first_indent_target
    for i in range(1, len(ind)):
        if ind[i] < ind[i-1]:
            ind[i] = ind[i-1]

    # Recompute martingale percentages from repaired volumes (no clamping)
    mart = np.zeros(M, dtype=np.float64)
    for i in range(1, M):
        denom = max(vol[i-1], 1e-12)
        mart[i] = (vol[i] / denom - 1.0) * 100.0

    # Build indent cumulative (prepend 0)
    indent_cum = np.concatenate([[0.0], ind.astype(np.float64)])

    # Recompute order prices from base price and indents
    order_prices = np.empty(M + 1, dtype=np.float64)
    order_prices[0] = base_price
    for i in range(1, M + 1):
        order_prices[i] = base_price * (1.0 - indent_cum[i] / 100.0)

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

    # Diagnostics
    tv_after = _total_variation(vol)
    front_sum_after = float(np.sum(vol[:kf]))
    l1_change = float(np.sum(np.abs(vol - vol_in)))
    clipped_frac = float(band_clips) / float(max(1, M - 1))

    diagnostics = {
        "clipped_frac": clipped_frac,
        "band_clips": int(band_clips),
        "front_excess_before": float(max(0.0, front_sum_before - front_cap)),
        "front_excess_after": float(max(0.0, front_sum_after - front_cap)),
        "tv_before": float(tv_before),
        "tv_after": float(tv_after),
        "l1_change": float(l1_change),
        "clip_hi_post": 0,  # Updated algorithm doesn't use clip_stats
        "clip_lo_post": 0,  # Updated algorithm doesn't use clip_stats
        "first3_sum": float(np.sum(vol[: min(3, M)])),
        "g2": float((vol[2] / max(vol[1], 1e-12)) if M > 2 else 0.0),
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
