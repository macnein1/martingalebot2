"""
Penalties and Rewards for DCA/Martingale Optimization
Implements Gini, entropy, monotonicity, smoothness, and wave pattern calculations
using numba-compatible pure NumPy functions.
"""
from __future__ import annotations

import numpy as np
from numba import njit
from typing import Dict, Any
import math
from martingale_lab.core.repair import compute_m_from_v, longest_plateau_run


@njit(cache=True, fastmath=True)
def gini_coefficient(x: np.ndarray) -> float:
    """
    Calculate Gini coefficient for inequality measurement.
    
    Args:
        x: Array of values (e.g., volumes, martingales)
        
    Returns:
        Gini coefficient [0,1] where 0=perfect equality, 1=maximum inequality
    """
    if x.size == 0:
        return 0.0
    
    # Handle negative values by shifting
    x_shifted = x - np.min(x) + 1e-12
    
    # Sort values
    x_sorted = np.sort(x_shifted)
    n = len(x_sorted)
    
    # Calculate Gini using the formula:
    # G = (2 * sum(i * x_i) / (n * sum(x_i))) - (n + 1) / n
    cumsum_x = np.cumsum(x_sorted)
    total_sum = cumsum_x[-1]
    
    if total_sum <= 1e-12:
        return 0.0
    
    # Weighted sum: sum(i * x_i) where i starts from 1
    weighted_sum = 0.0
    for i in range(n):
        weighted_sum += (i + 1) * x_sorted[i]
    
    gini = (2.0 * weighted_sum) / (n * total_sum) - (n + 1.0) / n
    
    return max(0.0, min(1.0, gini))


@njit(cache=True, fastmath=True)
def entropy_normalized(x: np.ndarray, eps: float = 1e-12) -> float:
    """
    Calculate normalized entropy for diversity measurement.
    
    Args:
        x: Array of values (e.g., volumes)
        eps: Small epsilon to avoid log(0)
        
    Returns:
        Normalized entropy [0,1] where 0=no diversity, 1=maximum diversity
    """
    if x.size <= 1:
        return 0.0
    
    # Normalize to probabilities
    x_sum = np.sum(x)
    if x_sum <= eps:
        return 0.0
    
    p = x / x_sum
    
    # Calculate entropy
    entropy = 0.0
    for i in range(len(p)):
        if p[i] > eps:
            entropy -= p[i] * math.log(p[i])
    
    # Normalize by maximum possible entropy (log(n))
    max_entropy = math.log(len(p))
    if max_entropy <= eps:
        return 0.0
    
    return entropy / max_entropy


@njit(cache=True, fastmath=True)
def monotonicity_penalty(x: np.ndarray) -> float:
    """
    Calculate penalty for non-monotonic sequences.
    
    Args:
        x: Array that should be monotonically increasing
        
    Returns:
        Penalty value >= 0 (0 = perfectly monotonic)
    """
    if x.size <= 1:
        return 0.0
    
    penalty = 0.0
    for i in range(1, len(x)):
        if x[i] < x[i-1]:
            penalty += x[i-1] - x[i]
    
    # Normalize by the range of x
    x_range = np.max(x) - np.min(x)
    if x_range > 1e-12:
        penalty = penalty / x_range
    
    return penalty


@njit(cache=True, fastmath=True)
def smoothness_penalty(x: np.ndarray) -> float:
    """
    Calculate penalty for non-smooth (jumpy) sequences.
    
    Args:
        x: Array to check for smoothness
        
    Returns:
        Penalty value >= 0 (0 = perfectly smooth)
    """
    if x.size <= 2:
        return 0.0
    
    # Calculate second differences
    penalty = 0.0
    for i in range(2, len(x)):
        # Second difference: (x[i] - x[i-1]) - (x[i-1] - x[i-2])
        second_diff = x[i] - 2*x[i-1] + x[i-2]
        penalty += abs(second_diff)
    
    # Normalize by the range of x
    x_range = np.max(x) - np.min(x)
    if x_range > 1e-12:
        penalty = penalty / x_range
    
    return penalty


@njit(cache=True, fastmath=True)
def tail_penalty_combined(volumes: np.ndarray, martingales: np.ndarray, 
                         weight_vol: float = 0.7, weight_mart: float = 0.3) -> float:
    """
    Calculate combined tail penalty using Gini coefficients of volumes and martingales.
    
    Args:
        volumes: Volume percentages
        martingales: Martingale percentages
        weight_vol: Weight for volume Gini
        weight_mart: Weight for martingale Gini
        
    Returns:
        Combined tail penalty [0,1]
    """
    # Normalize volumes to [0,1]
    vol_normalized = volumes / 100.0
    
    # Normalize martingales to [0,1] (skip first element which is 0)
    mart_normalized = martingales / 100.0
    
    vol_gini = gini_coefficient(vol_normalized)
    mart_gini = gini_coefficient(mart_normalized)
    
    return weight_vol * vol_gini + weight_mart * mart_gini


@njit(cache=True, fastmath=True)
def shape_reward_late_surge(volumes: np.ndarray) -> float:
    """
    Calculate shape reward for late surge pattern.
    Rewards volume distributions that increase towards the middle-end, then slightly decrease.
    
    Args:
        volumes: Volume percentages
        
    Returns:
        Shape reward [0,1] where 1=perfect late surge pattern
    """
    n = len(volumes)
    if n <= 2:
        return 0.0
    
    # Create late surge template: low start, peak around 70-80%, slight decrease at end
    template = np.empty(n)
    for i in range(n):
        t = float(i) / (n - 1)  # normalized position [0,1]
        if t < 0.7:
            # Gradual increase to position 0.7
            template[i] = 0.3 + 0.7 * (t / 0.7)
        else:
            # Slight decrease after position 0.7
            template[i] = 1.0 - 0.2 * ((t - 0.7) / 0.3)
    
    # Normalize template
    template_sum = np.sum(template)
    if template_sum > 1e-12:
        template = template / template_sum
    
    # Normalize volumes
    vol_normalized = volumes / np.sum(volumes)
    
    # Calculate cosine similarity
    dot_product = np.sum(vol_normalized * template)
    norm_vol = math.sqrt(np.sum(vol_normalized * vol_normalized))
    norm_template = math.sqrt(np.sum(template * template))
    
    if norm_vol > 1e-12 and norm_template > 1e-12:
        cosine_sim = dot_product / (norm_vol * norm_template)
        return max(0.0, cosine_sim)
    
    return 0.0


@njit(cache=True, fastmath=True)
def shape_reward_double_hump(volumes: np.ndarray) -> float:
    """
    Calculate shape reward for double hump pattern.
    Rewards volume distributions with two peaks.
    
    Args:
        volumes: Volume percentages
        
    Returns:
        Shape reward [0,1] where 1=perfect double hump pattern
    """
    n = len(volumes)
    if n <= 4:
        return 0.0
    
    # Create double hump template
    template = np.empty(n)
    for i in range(n):
        t = float(i) / (n - 1)  # normalized position [0,1]
        # Two Gaussian-like peaks at t=0.25 and t=0.75
        peak1 = math.exp(-((t - 0.25) / 0.15)**2)
        peak2 = math.exp(-((t - 0.75) / 0.15)**2)
        template[i] = peak1 + peak2 + 0.1  # Add small baseline
    
    # Normalize template
    template_sum = np.sum(template)
    if template_sum > 1e-12:
        template = template / template_sum
    
    # Normalize volumes
    vol_normalized = volumes / np.sum(volumes)
    
    # Calculate cosine similarity
    dot_product = np.sum(vol_normalized * template)
    norm_vol = math.sqrt(np.sum(vol_normalized * vol_normalized))
    norm_template = math.sqrt(np.sum(template * template))
    
    if norm_vol > 1e-12 and norm_template > 1e-12:
        cosine_sim = dot_product / (norm_vol * norm_template)
        return max(0.0, cosine_sim)
    
    return 0.0


@njit(cache=True, fastmath=True)
def wave_pattern_reward(martingales: np.ndarray, strong_threshold: float = 50.0, 
                       weak_threshold: float = 10.0) -> float:
    """
    Calculate wave pattern reward for alternating strong-weak martingale patterns.
    
    Args:
        martingales: Martingale percentages (first element should be 0)
        strong_threshold: Threshold for "strong" martingale
        weak_threshold: Threshold for "weak" martingale
        
    Returns:
        Wave reward (can be negative for penalties)
    """
    if len(martingales) <= 2:
        return 0.0
    
    reward = 0.0
    penalty = 0.0
    
    # Skip first element (should be 0)
    for i in range(2, len(martingales)):
        prev_mart = martingales[i-1]
        curr_mart = martingales[i]
        
        # Reward alternating patterns
        if prev_mart >= strong_threshold and curr_mart <= weak_threshold:
            reward += 0.1  # Strong to weak
        elif prev_mart <= weak_threshold and curr_mart >= strong_threshold:
            reward += 0.1  # Weak to strong
        
        # Penalty for consecutive patterns
        if prev_mart >= strong_threshold and curr_mart >= strong_threshold:
            penalty += 0.15  # Consecutive strong
        elif prev_mart <= weak_threshold and curr_mart <= weak_threshold:
            penalty += 0.15  # Consecutive weak
    
    return reward - penalty


@njit(cache=True, fastmath=True)
def cvar_calculation(need_pct: np.ndarray, q: float = 0.8) -> float:
    """
    Calculate Conditional Value at Risk (CVaR) for Need% values.
    
    Args:
        need_pct: Array of Need% values
        q: Quantile level (default 0.8 for 80th percentile)
        
    Returns:
        CVaR value (mean of values above q-th percentile)
    """
    if need_pct.size == 0:
        return 0.0
    
    # Sort values
    sorted_need = np.sort(need_pct)
    n = len(sorted_need)
    
    # Find q-th percentile index
    q_index = int(q * n)
    if q_index >= n:
        q_index = n - 1
    
    # Calculate mean of values above q-th percentile
    if q_index < n - 1:
        tail_values = sorted_need[q_index:]
        return np.mean(tail_values)
    else:
        return sorted_need[-1]


@njit(cache=True, fastmath=True)
def penalty_first_fixed(volumes: np.ndarray, indents: np.ndarray, target_volume: float, target_indent: float) -> float:
    if volumes.size == 0:
        return 0.0
    v0 = volumes[0]
    i0 = indents[0] if indents.size > 0 else 0.0
    return abs(v0 - target_volume) + abs(i0 - target_indent)


@njit(cache=True, fastmath=True)
def penalty_g_band(volumes: np.ndarray, g_min: float, g_max: float) -> float:
    n = volumes.size
    if n <= 2:
        return 0.0
    pen = 0.0
    for i in range(2, n):
        denom = volumes[i-1] if volumes[i-1] > 1e-12 else 1e-12
        g = volumes[i] / denom
        if g < g_min:
            pen += (g_min - g)
        elif g > g_max:
            pen += (g - g_max)
    return pen


@njit(cache=True, fastmath=True)
def penalty_frontload(volumes: np.ndarray, k_front: int, front_cap: float) -> float:
    kf = k_front if k_front < volumes.size else volumes.size
    s = 0.0
    for i in range(kf):
        s += volumes[i]
    excess = s - front_cap
    if excess > 0.0:
        return excess * excess
    return 0.0


@njit(cache=True, fastmath=True)
def penalty_total_variation_vol(volumes: np.ndarray) -> float:
    n = volumes.size
    if n <= 1:
        return 0.0
    tv = 0.0
    for i in range(1, n):
        diff = volumes[i] - volumes[i-1]
        tv += abs(diff)
    return tv


@njit(cache=True, fastmath=True)
def penalty_second_leq(volumes: np.ndarray) -> float:
    n = volumes.size
    if n <= 1:
        return 0.0
    v0 = volumes[0]
    v1 = volumes[1]
    excess = v1 - v0
    return excess if excess > 0.0 else 0.0


@njit(cache=True, fastmath=True)
def penalty_wave(volumes: np.ndarray, lambda_v2: float = 0.1) -> float:
    n = volumes.size
    if n <= 3:
        return 0.0
    # Growth ratios for i>=2
    # g[i] corresponds to vol[i]/vol[i-1] for i in [2..n-1]
    prev = volumes[1] if volumes[1] > 1e-12 else 1e-12
    # Prepare first growth at i=2
    g_prev = volumes[2] / (volumes[1] if volumes[1] > 1e-12 else 1e-12)
    sum_abs_delta_g = 0.0
    for i in range(3, n):
        denom = volumes[i-1] if volumes[i-1] > 1e-12 else 1e-12
        g_curr = volumes[i] / denom
        diff_g = g_curr - g_prev
        sum_abs_delta_g += abs(diff_g)
        g_prev = g_curr
    # Second differences on volumes (i>=2)
    sum_abs_second_diff = 0.0
    for i in range(2, n):
        second = volumes[i] - 2.0 * volumes[i-1] + volumes[i-2]
        sum_abs_second_diff += abs(second)
    return sum_abs_delta_g + lambda_v2 * sum_abs_second_diff


@njit(cache=True, fastmath=True)
def penalty_uniform_martingale(martingales: np.ndarray, target_std: float = 0.10) -> float:
    """
    Calculate penalty for uniform martingale pattern (low diversity).
    
    Args:
        martingales: Martingale percentages (first element should be 0)
        target_std: Target standard deviation for diversity
        
    Returns:
        Penalty for low diversity (0 if std >= target_std)
    """
    if martingales.size <= 2:
        return 0.0
    
    # Calculate m_i = g_i - 1 (growth ratio minus 1)
    m_values = np.empty(martingales.size - 1)
    for i in range(1, martingales.size):
        m_values[i-1] = martingales[i] / 100.0  # Convert percentage to ratio
    
    # Calculate standard deviation
    mean_m = np.mean(m_values)
    var_m = 0.0
    for i in range(m_values.size):
        diff = m_values[i] - mean_m
        var_m += diff * diff
    var_m /= m_values.size
    std_m = math.sqrt(var_m)
    
    # Penalty for low diversity
    return max(0.0, target_std - std_m)


@njit(cache=True, fastmath=True)
def penalty_low_entropy(martingales: np.ndarray, target_entropy: float = 1.0) -> float:
    """
    Calculate penalty for low entropy in martingale pattern.
    
    Args:
        martingales: Martingale percentages (first element should be 0)
        target_entropy: Target entropy value
        
    Returns:
        Penalty for low entropy (0 if entropy >= target_entropy)
    """
    if martingales.size <= 2:
        return 0.0
    
    # Calculate m_i = g_i - 1 (growth ratio minus 1)
    m_values = np.empty(martingales.size - 1)
    for i in range(1, martingales.size):
        m_values[i-1] = martingales[i] / 100.0  # Convert percentage to ratio
    
    # Calculate entropy
    m_sum = np.sum(m_values)
    if m_sum <= 1e-12:
        return target_entropy  # Maximum penalty for zero sum
    
    entropy = 0.0
    for i in range(m_values.size):
        if m_values[i] > 1e-12:
            p = m_values[i] / m_sum
            entropy -= p * math.log(p)
    
    # Penalty for low entropy
    return max(0.0, target_entropy - entropy)


@njit(cache=True, fastmath=True)
def penalty_flat_blocks(volumes: np.ndarray, k_front: int = 3) -> float:
    """
    Calculate penalty for flat block patterns (blocks mode).
    
    Args:
        volumes: Volume percentages
        k_front: Number of front orders to consider as blocks
        
    Returns:
        Penalty for flat blocks (0 if good variation)
    """
    if volumes.size <= k_front:
        return 0.0
    
    # Calculate average martingales for each block
    # For simplicity, we'll use volume ratios as proxy for martingales
    block_means = []
    
    # Front block (first k_front orders)
    front_sum = 0.0
    for i in range(k_front):
        front_sum += volumes[i]
    front_mean = front_sum / k_front
    block_means.append(front_mean)
    
    # Tail block (remaining orders)
    tail_sum = 0.0
    for i in range(k_front, volumes.size):
        tail_sum += volumes[i]
    tail_mean = tail_sum / (volumes.size - k_front)
    block_means.append(tail_mean)
    
    # Count sign changes in block means
    sign_changes = 0
    for i in range(1, len(block_means)):
        if (block_means[i] - block_means[i-1]) * (block_means[i-1] - (block_means[i-2] if i > 1 else block_means[i-1])) < 0:
            sign_changes += 1
    
    # Penalty: require at least 1 sign change (1 up + 1 down)
    return max(0.0, 2.0 - sign_changes)


# New penalty functions (SP1-SP7)
def penalty_second_band(v0: float, v1: float) -> float:
    """SP1: v1'in [1.10v0, 2.0v0] dışına karesi."""
    v1_min = 1.10 * v0
    v1_max = 2.0 * v0
    if v1 < v1_min:
        return (v1_min - v1) ** 2
    elif v1 > v1_max:
        return (v1 - v1_max) ** 2
    return 0.0


def penalty_plateau(m_tail: np.ndarray, tol: float = 0.02, max_len: int = 3) -> float:
    """SP2: plato run'larının (|m−1|<tol) fazla uzun kısmının toplamı."""
    max_run = longest_plateau_run(m_tail, center=1.0, tol=tol, start_idx=0)
    return max(0.0, max_run - max_len)


def penalty_varm(m_tail: np.ndarray, target_std: float = 0.20) -> float:
    """SP3: max(0, target_std - std(m_tail))."""
    if m_tail.size <= 1:
        return target_std
    std_m = float(np.std(m_tail))
    return max(0.0, target_std - std_m)


def penalty_wave_shape(m_tail: np.ndarray, m_min: float, m_head: float, m_tail_cap: float, 
                      tau: float, phase: float = 0.0) -> float:
    """SP4: decaying tavana gömülü 1.5 dalgalı hedefe L2 uzaklık (N'e ölçekli)."""
    if m_tail.size <= 1:
        return 0.0
    
    # Create target wave pattern
    target = np.zeros_like(m_tail)
    for i in range(m_tail.size):
        # Decaying ceiling
        ceiling = m_tail_cap + (m_head - m_tail_cap) * np.exp(-i/tau)
        # Wave pattern: 1.5 cycles with phase
        wave = 0.5 * np.sin(2 * np.pi * 1.5 * i / m_tail.size + phase)
        target[i] = ceiling * (1.0 + wave)
    
    # L2 distance
    diff = m_tail - target
    return float(np.sum(diff * diff)) / m_tail.size


def penalty_front_share(volumes: np.ndarray, q1_cap: float = 22.0) -> float:
    """SP5: ilk çeyrek toplam cap aşımları."""
    if volumes.size == 0:
        return 0.0
    
    Q1 = max(1, int(np.ceil(volumes.size / 4.0)))
    front_sum = float(np.sum(volumes[:Q1]))
    return max(0.0, front_sum - q1_cap)


def penalty_tailweak(volumes: np.ndarray, tail_floor: float = 32.0) -> float:
    """SP6: son çeyrek taban altı eksikliği."""
    if volumes.size == 0:
        return 0.0
    
    Q4 = max(1, int(np.ceil(volumes.size / 4.0)))
    tail_sum = float(np.sum(volumes[-Q4:]))
    return max(0.0, tail_floor - tail_sum)


def penalty_slope(m_tail: np.ndarray, delta_soft: float = 0.20, delta_cap: float = 0.25) -> float:
    """SP7: |Δm| delta_soft'ı aşınca karesel maliyet."""
    if m_tail.size <= 1:
        return 0.0
    
    penalty = 0.0
    for i in range(1, m_tail.size):
        delta = abs(m_tail[i] - m_tail[i-1])
        if delta > delta_soft:
            # Quadratic penalty
            penalty += (delta - delta_soft) ** 2
        if delta > delta_cap:
            # Additional penalty for exceeding hard cap
            penalty += (delta - delta_cap) ** 2
    
    return penalty


def compute_shape_penalties(volumes: np.ndarray, indents: np.ndarray,
                            first_volume_target: float, first_indent_target: float,
                            g_min: float, g_max: float,
                            k_front: int, front_cap: float,
                            martingales: np.ndarray = None,
                            target_std: float = 0.10,
                            use_entropy: bool = False,
                            entropy_target: float = 1.0,
                            # New parameters for SP1-SP7
                            second_upper_c2: float = 2.0,
                            m2_min: float = 0.10, m2_max: float = 0.80,
                            m_min: float = 0.05, m_head: float = 0.40, m_tail: float = 0.20,
                            tau_scale: float = 1/3, slope_cap: float = 0.25,
                            q1_cap: float = 22.0, tail_floor: float = 32.0,
                            delta_soft: float = 0.20, delta_cap: float = 0.25,
                            # Weight presets
                            penalty_preset: str = "robust") -> Dict[str, float]:
    """Compute shape-specific penalties after repair."""
    penalties = {
        "penalty_first_fixed": penalty_first_fixed(volumes, indents, first_volume_target, first_indent_target),
        "penalty_second_leq": penalty_second_leq(volumes),
        "penalty_g_band": penalty_g_band(volumes, g_min, g_max),
        "penalty_frontload": penalty_frontload(volumes, k_front, front_cap),
        "penalty_tv_vol": penalty_total_variation_vol(volumes),
        "penalty_wave": penalty_wave(volumes, 0.1),
    }
    
    # Add new diversity penalties if martingales provided
    if martingales is not None:
        penalties["penalty_uniform"] = penalty_uniform_martingale(martingales, target_std)
        penalties["penalty_flat_blocks"] = penalty_flat_blocks(volumes, k_front)
        
        if use_entropy:
            penalties["penalty_low_entropy"] = penalty_low_entropy(martingales, entropy_target)
    
    # Add new SP1-SP7 penalties
    if volumes.size >= 2:
        v0, v1 = volumes[0], volumes[1]
        penalties["penalty_second_band"] = penalty_second_band(v0, v1)
    
    if martingales is not None and martingales.size > 2:
        m_tail = martingales[2:]  # Skip first two elements
        penalties["penalty_plateau"] = penalty_plateau(m_tail)
        penalties["penalty_varm"] = penalty_varm(m_tail, target_std=0.20)
        
        # Wave shape penalty with parameters
        tau = max(1.0, volumes.size * tau_scale)
        penalties["penalty_wave_shape"] = penalty_wave_shape(m_tail, m_min, m_head, m_tail, tau)
    
    penalties["penalty_front_share"] = penalty_front_share(volumes, q1_cap)
    penalties["penalty_tailweak"] = penalty_tailweak(volumes, tail_floor)
    
    if martingales is not None and martingales.size > 2:
        m_tail = martingales[2:]
        penalties["penalty_slope"] = penalty_slope(m_tail, delta_soft, delta_cap)
    
    # Weight presets
    if penalty_preset == "robust":
        weights = {
            "w_second": 3.0, "w_plateau": 2.0, "w_varm": 3.0, 
            "w_wave_shape": 2.0, "w_front": 2.0, "w_tail": 2.0, "w_slope": 1.0
        }
    elif penalty_preset == "tight":
        weights = {
            "w_second": 4.0, "w_plateau": 3.0, "w_varm": 4.0,
            "w_wave_shape": 3.0, "w_front": 3.0, "w_tail": 3.0, "w_slope": 2.0
        }
    elif penalty_preset == "explore":
        weights = {
            "w_second": 2.0, "w_plateau": 1.5, "w_varm": 2.0,
            "w_wave_shape": 1.5, "w_front": 1.5, "w_tail": 1.5, "w_slope": 0.5
        }
    else:
        weights = {
            "w_second": 3.0, "w_plateau": 2.0, "w_varm": 3.0,
            "w_wave_shape": 2.0, "w_front": 2.0, "w_tail": 2.0, "w_slope": 1.0
        }
    
    # Apply weights to new penalties
    for key, weight in weights.items():
        penalty_key = key.replace("w_", "penalty_")
        if penalty_key in penalties:
            penalties[penalty_key] *= weight
    
    return penalties


def compute_all_penalties(volumes: np.ndarray, martingales: np.ndarray, 
                         indents: np.ndarray, need_pct: np.ndarray,
                         config: Dict[str, Any]) -> Dict[str, float]:
    """
    Compute all penalty and reward components.
    
    Args:
        volumes: Volume percentages
        martingales: Martingale percentages
        indents: Indent percentages
        need_pct: Need percentages
        config: Configuration dictionary with weights and parameters
        
    Returns:
        Dictionary with all penalty/reward components
    """
    # Extract configuration
    gini_w_vol = config.get('gini_w_vol', 0.7)
    gini_w_mart = config.get('gini_w_mart', 0.3)
    shape_template = config.get('shape_template', 'late_surge')
    q_cvar = config.get('q_cvar', 0.8)
    strong_threshold = config.get('strong_threshold', 50.0)
    weak_threshold = config.get('weak_threshold', 10.0)
    
    penalties = {}
    
    # Basic metrics
    penalties['max_need'] = float(np.max(need_pct)) if len(need_pct) > 0 else 0.0
    penalties['var_need'] = float(np.var(need_pct)) if len(need_pct) > 0 else 0.0
    penalties['cvar_need'] = cvar_calculation(need_pct, q_cvar)
    
    # Tail penalty (Gini-based)
    penalties['tail_penalty'] = tail_penalty_combined(volumes, martingales, gini_w_vol, gini_w_mart)
    
    # Individual Gini coefficients
    penalties['gini_volumes'] = gini_coefficient(volumes / 100.0)
    penalties['gini_martingales'] = gini_coefficient(martingales / 100.0)
    
    # Entropy
    penalties['entropy_volumes'] = entropy_normalized(volumes)
    penalties['entropy_martingales'] = entropy_normalized(martingales[1:])  # Skip first 0
    
    # Monotonicity and smoothness
    penalties['monotone_penalty'] = monotonicity_penalty(indents)
    penalties['smooth_penalty_indents'] = smoothness_penalty(indents)
    penalties['smooth_penalty_need'] = smoothness_penalty(need_pct)
    
    # Shape reward
    if shape_template == 'late_surge':
        shape_reward = shape_reward_late_surge(volumes)
    elif shape_template == 'double_hump':
        shape_reward = shape_reward_double_hump(volumes)
    else:  # 'flat' or default
        shape_reward = 1.0 - gini_coefficient(volumes / 100.0)  # Reward flat distribution
    
    penalties['shape_reward'] = shape_reward
    penalties['shape_penalty'] = 1.0 - shape_reward
    
    # Wave pattern reward
    penalties['wave_reward'] = wave_pattern_reward(martingales, strong_threshold, weak_threshold)
    penalties['wave_penalty'] = max(0.0, -penalties['wave_reward'])
    
    return penalties


def compute_composite_score(penalties: Dict[str, float], weights: Dict[str, float]) -> float:
    """
    Compute composite score J from penalty components.
    
    Args:
        penalties: Dictionary of penalty/reward components
        weights: Dictionary of weights for each component
        
    Returns:
        Composite score J (to be minimized)
    """
    # Extract weights with defaults
    alpha = weights.get('alpha', 0.45)      # max_need
    beta = weights.get('beta', 0.20)        # var_need
    gamma = weights.get('gamma', 0.20)      # tail_penalty
    delta = weights.get('delta', 0.10)      # shape_penalty
    rho = weights.get('rho', 0.05)          # cvar_need
    eta = weights.get('eta', 0.02)          # monotone_penalty
    zeta = weights.get('zeta', 0.02)        # smooth_penalty
    
    # Compute composite score
    J = (alpha * penalties['max_need'] +
         beta * penalties['var_need'] +
         gamma * penalties['tail_penalty'] +
         delta * penalties['shape_penalty'] +
         rho * penalties['cvar_need'] +
         eta * penalties['monotone_penalty'] +
         zeta * (penalties['smooth_penalty_indents'] + penalties['smooth_penalty_need']))
    
    return float(J)
