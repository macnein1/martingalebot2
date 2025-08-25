"""
Constraints and Normalization for DCA/Martingale Optimization
Implements soft penalty approach with proper bounds checking and normalization functions.
"""
from __future__ import annotations

import numpy as np
from numba import njit
from typing import Dict, Any, Tuple
from martingale_lab.core.repair import hard_clip_local_growth, isotonic_non_decreasing, tail_only_rescale_keep_first_two, compute_m_from_v, rechain_v_from_m, longest_plateau_run
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


def _decay_ceiling(i, N, m_head, m_tail, tau_scale):
    """Decaying ceiling function for m_max(i)"""
    tau = max(1.0, N * tau_scale)
    return m_tail + (m_head - m_tail) * np.exp(-(i-2)/tau)


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
    # New parameters
    second_upper_c2: float = 2.0,     # v1 ≤ c2*v0 band
    m2_min: float = 0.10, m2_max: float = 0.80,   # i=2 band
    m_min: float = 0.05,              # i≥2 alt band
    m_head: float = 0.40, m_tail: float = 0.20,   # decaying tavan
    tau_scale: float = 1/3,           # tau = N*tau_scale
    slope_cap: float = 0.25,          # |Δm| ≤ cap (i≥3)
    q1_cap: float = 22.0,             # ilk çeyrek toplam üst sınır (%)
    tail_floor: float = 32.0,         # son çeyrek taban (%)
    eps_inc: float = 1e-6,
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

    # Initialize diagnostics counters
    v1_band_applied = 0
    m2_clip_applied = 0
    decaying_clips_count = 0
    slope_clips_count = 0
    plateau_max_run = 0
    turn_count = 0

    # 1. Başlat: v0=0.01, indent0=0.00 (zaten set). İlk normalize: tail_only_rescale_keep_first_two.
    vol = np.zeros(M, dtype=np.float64)
    vol[0] = first_volume_target
    if M > 1:
        vol[1] = vol_in[1] if len(vol_in) > 1 else first_volume_target
    if M > 2:
        vol[2:] = vol_in[2:] if len(vol_in) > 2 else np.zeros(M-2)
    
    # Initial tail-only rescale
    tail_only_rescale_keep_first_two(vol)

    # 2. m = compute_m_from_v(v).
    m = compute_m_from_v(vol)

    # 3. HC1 (v1 band): v1 ∈ [1.10*v0, second_upper_c2*v0]. Kliple; sonra tail-only rescale.
    if M > 1:
        v1_min = 1.10 * vol[0]
        v1_max = second_upper_c2 * vol[0]
        if vol[1] < v1_min:
            vol[1] = v1_min
            v1_band_applied = 1
        elif vol[1] > v1_max:
            vol[1] = v1_max
            v1_band_applied = 1
        tail_only_rescale_keep_first_two(vol)

    # 4. HC3 (m bantları):
    # i=2: klip [m2_min, m2_max]
    # i≥3: klip [m_min, m_max(i)], m_max(i)=_decay_ceiling(...)
    # Rechain v = rechain_v_from_m(v0,v1,m) → tail-only rescale.
    if M > 2:
        # Clip m[2]
        m[2] = max(m2_min, min(m[2], m2_max))
        if m[2] != (vol[2] / max(vol[1], 1e-12) - 1.0):
            m2_clip_applied = 1
        
        # Clip m[i] for i≥3 with decaying ceiling
        for i in range(3, M):
            m_max_i = _decay_ceiling(i, M, m_head, m_tail, tau_scale)
            m[i] = max(m_min, min(m[i], m_max_i))
            if m[i] != (vol[i] / max(vol[i-1], 1e-12) - 1.0):
                decaying_clips_count += 1
        
        # Rechain and rescale
        vol = rechain_v_from_m(vol[0], vol[1], m)
        tail_only_rescale_keep_first_two(vol)

    # 5. HC4 (eğim sınırı): i≥3 için bandı [m[i-1]-slope_cap, m[i-1]+slope_cap] ile kesiştirip kliple. → Rechain → tail-only.
    if M > 3:
        for i in range(3, M):
            m_prev = m[i-1]
            m_min_slope = m_prev - slope_cap
            m_max_slope = m_prev + slope_cap
            m[i] = max(m_min_slope, min(m[i], m_max_slope))
            if m[i] != (vol[i] / max(vol[i-1], 1e-12) - 1.0):
                slope_clips_count += 1
        
        # Rechain and rescale
        vol = rechain_v_from_m(vol[0], vol[1], m)
        tail_only_rescale_keep_first_two(vol)

    # 6. HC2 (katı artış): ∀i≥1: v[i] ≥ v[i-1]+eps_inc. Gerekli yerleri yükselt.
    for i in range(1, M):
        min_vol = vol[i-1] + eps_inc
        if vol[i] < min_vol:
            vol[i] = min_vol

    # 7. Tail-only rescale (v0,v1 sabit).
    tail_only_rescale_keep_first_two(vol)

    # 8. HC5 (kütle kontrolü): N'e göre Q1=ceil(N/4), Q4=ceil(N/4).
    # Front cap: S_F>q1_cap ise yalnız [2..Q1-1] blokunu f_F2 ile küçült; çıkan Δ'yı Tail'e oransal ekle.
    # Tail floor: S_T<tail_floor ise Tail'i f_T ile büyüt; dengeyi Mid'den (yoksa Front2) düş.
    # Ardından HC2 + tail-only.
    Q1 = max(1, int(np.ceil(M / 4.0)))
    Q4 = max(1, int(np.ceil(M / 4.0)))
    
    # Front cap
    front_sum = np.sum(vol[:Q1])
    if front_sum > q1_cap:
        front2_sum = np.sum(vol[2:Q1]) if Q1 > 2 else 0.0
        if front2_sum > 1e-12:
            target_front2 = max(0.0, q1_cap - vol[0] - vol[1])
            f_front2 = target_front2 / front2_sum
            vol[2:Q1] *= f_front2
            # Redistribute excess to tail
            excess = front_sum - q1_cap
            tail_sum = np.sum(vol[Q1:])
            if tail_sum > 1e-12:
                vol[Q1:] *= (tail_sum + excess) / tail_sum
    
    # Tail floor
    tail_sum = np.sum(vol[-Q4:])
    if tail_sum < tail_floor:
        deficit = tail_floor - tail_sum
        # Take from middle section
        mid_start = Q1
        mid_end = M - Q4
        if mid_end > mid_start:
            mid_sum = np.sum(vol[mid_start:mid_end])
            if mid_sum > deficit:
                vol[mid_start:mid_end] *= (mid_sum - deficit) / mid_sum
                vol[-Q4:] *= (tail_sum + deficit) / tail_sum
    
    # Re-apply HC2 and tail-only
    for i in range(1, M):
        min_vol = vol[i-1] + eps_inc
        if vol[i] < min_vol:
            vol[i] = min_vol
    tail_only_rescale_keep_first_two(vol)

    # 9. HC6 (plato kırıcı): i≥2'de |m−1|<0.02 koşulunda run uzunluğu L>3 ise, run içinde +/− δ alternasyonla ayarla; δ, bant/slope izinlerine göre min alınır. → Rechain → tail-only.
    m = compute_m_from_v(vol)
    plateau_max_run = longest_plateau_run(m, center=1.0, tol=0.02, start_idx=2)
    
    if plateau_max_run > 3:
        # Find plateau runs and break them
        current_run = 0
        run_start = 2
        for i in range(2, M):
            if abs(m[i] - 1.0) < 0.02:
                if current_run == 0:
                    run_start = i
                current_run += 1
            else:
                if current_run > 3:
                    # Break this plateau run
                    delta = min(0.05, slope_cap, m_head - 1.0)  # Conservative delta
                    for j in range(run_start, i):
                        if (j - run_start) % 2 == 0:
                            m[j] = 1.0 + delta
                        else:
                            m[j] = 1.0 - delta
                current_run = 0
        
        # Handle last run
        if current_run > 3:
            delta = min(0.05, slope_cap, m_head - 1.0)
            for j in range(run_start, M):
                if (j - run_start) % 2 == 0:
                    m[j] = 1.0 + delta
                else:
                    m[j] = 1.0 - delta
        
        # Rechain and rescale
        vol = rechain_v_from_m(vol[0], vol[1], m)
        tail_only_rescale_keep_first_two(vol)

    # 10. Türevler: martingale_pct, needpct, order_prices, price_step_pct hesaplarını mevcut formüllerle aynen üret.
    # Ensure v0 is fixed after all operations
    vol[0] = first_volume_target

    # Repair indents to be non-decreasing and anchored at first_indent_target
    if M >= 1:
        ind[0] = first_indent_target
    for i in range(1, len(ind)):
        if ind[i] < ind[i-1]:
            ind[i] = ind[i-1]

    # Recompute martingale percentages from repaired volumes
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

    # Final diagnostics
    tv_after = _total_variation(vol)
    front_sum_after = float(np.sum(vol[: min(k_front, M)]))
    l1_change = float(np.sum(np.abs(vol - vol_in)))
    
    # Calculate final metrics
    m_final = compute_m_from_v(vol)
    std_m = float(np.std(m_final[2:])) if M > 2 else 0.0
    
    # Count turns (sign changes in m around 1.0)
    turn_count = 0
    if M > 3:
        for i in range(3, M):
            if (m_final[i-1] - 1.0) * (m_final[i] - 1.0) < 0:
                turn_count += 1
    
    # Q1 and Q4 shares
    q1_share = float(np.sum(vol[:Q1])) if M > 0 else 0.0
    q4_share = float(np.sum(vol[-Q4:])) if M > 0 else 0.0

    diagnostics = {
        "clipped_frac": 0.0,  # Legacy field
        "band_clips": 0,      # Legacy field
        "front_excess_before": float(max(0.0, front_sum_before - front_cap)),
        "front_excess_after": float(max(0.0, front_sum_after - front_cap)),
        "tv_before": float(tv_before),
        "tv_after": float(tv_after),
        "l1_change": float(l1_change),
        "clip_hi_post": 0,    # Legacy field
        "clip_lo_post": 0,    # Legacy field
        "first3_sum": float(np.sum(vol[: min(3, M)])),
        "g2": float((vol[2] / max(vol[1], 1e-12)) if M > 2 else 0.0),
        "m2": float(m_final[2]) if M > 2 else 0.0,
        "std_m": std_m,
        "sign_changes": turn_count,
        "iter_fix_loops": 0,  # Legacy field
        # New diagnostics
        "v1_band_applied": v1_band_applied,
        "m2_clip_applied": m2_clip_applied,
        "decaying_clips_count": decaying_clips_count,
        "slope_clips_count": slope_clips_count,
        "q1_share": q1_share,
        "q4_share": q4_share,
        "plateau_max_run": plateau_max_run,
        "turn_count": turn_count,
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
