"""
DCA/Martingale Evaluation Contract - "İşlemden En Hızlı Çıkış" Odaklı
Implements complete evaluation function with NeedPct calculation, wave patterns, sanity checks.
"""
from __future__ import annotations

import json
import math
import hashlib
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


@dataclass
class EvaluationKnobs:
    """Configuration for evaluation function."""
    # Core scoring weights
    alpha: float = 0.5  # max_need weight
    beta: float = 0.3   # var_need weight  
    gamma: float = 0.2  # tail weight
    lambda_penalty: float = 0.1  # penalty weight
    
    # Wave pattern settings
    wave_pattern: bool = False
    wave_strong_threshold: float = 50.0  # >= this is "strong" martingale
    wave_weak_threshold: float = 10.0    # <= this is "very weak" martingale
    
    # Tail and volume constraints
    tail_cap: float = 0.40  # max volume % for last order
    min_indent_step: float = 0.05  # minimum indent step %
    softmax_temp: float = 1.0  # softmax temperature for volume distribution
    
    # Sanity check thresholds
    max_need_tolerance: float = 1e-6
    collapse_threshold: float = 0.01  # minimum step size
    
    # Random seed
    random_seed: Optional[int] = None


def _softplus(x: np.ndarray) -> np.ndarray:
    """Softplus activation: log(1 + exp(x))"""
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)


def _softmax(x: np.ndarray, temperature: float = 1.0, eps: float = 1e-12) -> np.ndarray:
    """Softmax with temperature control."""
    x_scaled = x / temperature
    x_max = np.max(x_scaled)
    e = np.exp(x_scaled - x_max)
    s = np.sum(e) + eps
    return e / s


def _gini_coefficient(weights: np.ndarray) -> float:
    """Calculate Gini coefficient for volume distribution."""
    if weights.size == 0:
        return 0.0
    w = np.sort(weights)
    cum = np.cumsum(w)
    n = float(weights.size)
    denom = n * cum[-1] + 1e-12
    g = (n + 1.0 - 2.0 * np.sum(cum) / denom) / n
    return float(max(0.0, g))


def _entropy_normalized(weights: np.ndarray, eps: float = 1e-12) -> float:
    """Calculate normalized entropy (0-1 scale)."""
    if weights.size == 0:
        return 0.0
    p = weights / (np.sum(weights) + eps)
    h = -np.sum(p * np.log(p + eps))
    n = float(weights.size)
    if n <= 1:
        return 0.0
    return float(h / math.log(n))


def _weight_center_index(weights: np.ndarray) -> float:
    """Calculate Weight Center Index (0=early, 1=late load)."""
    n = weights.size
    if n <= 1:
        return 0.0
    indices = np.arange(n, dtype=np.float64)
    w_sum = float(np.sum(weights))
    if w_sum <= 0:
        return 0.0
    center = float(np.sum(weights * indices) / (w_sum * (n - 1)))
    return float(min(1.0, max(0.0, center)))


def _count_sign_flips(needpct: np.ndarray) -> int:
    """Count sign flips in NeedPct trend."""
    if needpct.size < 2:
        return 0
    diff = np.diff(needpct)
    sign_changes = np.sum(diff[:-1] * diff[1:] < 0)
    return int(sign_changes)


def build_schedule(
    base_price: float,
    overlap_pct: float, 
    num_orders: int,
    knobs: EvaluationKnobs
) -> Dict[str, Any]:
    """Build complete DCA schedule with all required components."""
    if num_orders < 1:
        raise ValueError("num_orders must be >= 1")
    if not (0.0 < overlap_pct <= 100.0):
        raise ValueError("overlap_pct must be in (0,100]")
    
    rng = np.random.default_rng(knobs.random_seed)
    
    # Generate random logits
    indent_logits = rng.normal(0.0, 1.0, size=num_orders).astype(np.float64)
    volume_logits = rng.normal(0.0, 1.0, size=num_orders).astype(np.float64)
    
    # 1. Build indent_pct with monotonic increasing steps
    # Use softplus to ensure positive steps, then normalize
    steps = _softplus(indent_logits)
    
    # Apply minimum step threshold
    steps = np.maximum(steps, knobs.min_indent_step / 100.0)
    
    # Normalize to sum to overlap_pct
    steps_sum = np.sum(steps)
    if steps_sum <= 1e-12:
        steps = np.full(num_orders, overlap_pct / (100.0 * num_orders))
    else:
        steps = steps * (overlap_pct / 100.0) / steps_sum
    
    # Cumulative indents (p0=0 implicit, p1, p2, ..., pM)
    indent_pct = np.cumsum(steps) * 100.0  # Convert back to percentage
    indent_pct = np.concatenate([[0.0], indent_pct])  # p0 = 0.0
    
    # 2. Build volume_pct using softmax (sums to 100)
    volume_raw = _softmax(volume_logits, temperature=knobs.softmax_temp)
    volume_pct = volume_raw * 100.0
    
    # Apply tail cap constraint
    if volume_pct[-1] > knobs.tail_cap * 100.0:
        excess = volume_pct[-1] - knobs.tail_cap * 100.0
        volume_pct[-1] = knobs.tail_cap * 100.0
        # Redistribute excess to other orders proportionally
        if num_orders > 1:
            other_sum = np.sum(volume_pct[:-1])
            if other_sum > 1e-9:
                volume_pct[:-1] += volume_pct[:-1] * (excess / other_sum)
    
    # 3. Build martingale_pct (m1=0, m2...mM calculated)
    martingale_pct = np.zeros(num_orders)
    for i in range(1, num_orders):
        prev_vol = volume_pct[i-1]
        curr_vol = volume_pct[i]
        if prev_vol > 1e-12:
            martingale_pct[i] = min(100.0, max(1.0, (curr_vol / prev_vol - 1.0) * 100.0))
    
    # 4. Calculate order prices
    order_prices = np.empty(num_orders + 1)  # Include base price
    order_prices[0] = base_price  # Base price
    for i in range(1, num_orders + 1):
        order_prices[i] = base_price * (1.0 - indent_pct[i] / 100.0)
    
    # 5. Calculate price step percentages
    price_step_pct = np.empty(num_orders)
    for i in range(num_orders):
        if i == 0:
            price_step_pct[i] = indent_pct[1]  # First step
        else:
            price_step_pct[i] = indent_pct[i+1] - indent_pct[i]
    
    # 6. Calculate NeedPct sequence (exit-to-entry percentages)
    needpct = np.empty(num_orders)
    vol_acc = 0.0
    val_acc = 0.0
    
    for k in range(num_orders):
        vol_acc += volume_pct[k]
        val_acc += volume_pct[k] * order_prices[k+1]  # k+1 because order_prices[0] is base
        
        # Weighted average entry price
        avg_entry_price = val_acc / max(vol_acc, 1e-12)
        
        # Percentage needed to return to entry from current order price
        current_price = order_prices[k+1]
        needpct[k] = (avg_entry_price / max(current_price, 1e-12) - 1.0) * 100.0
    
    return {
        "indent_pct": indent_pct[1:].tolist(),  # Exclude p0=0
        "volume_pct": volume_pct.tolist(),
        "martingale_pct": martingale_pct.tolist(),
        "needpct": needpct.tolist(),
        "order_prices": order_prices.tolist(),
        "price_step_pct": price_step_pct.tolist(),
    }


def calculate_wave_pattern_score(martingale_pct: np.ndarray, knobs: EvaluationKnobs) -> float:
    """Calculate wave pattern reward/penalty."""
    if not knobs.wave_pattern or len(martingale_pct) < 3:
        return 0.0
    
    reward = 0.0
    penalty = 0.0
    
    # Check for alternating strong-weak pattern
    for i in range(1, len(martingale_pct) - 1):
        curr = martingale_pct[i]
        next_val = martingale_pct[i + 1]
        
        # Reward alternating pattern
        if curr >= knobs.wave_strong_threshold and next_val <= knobs.wave_weak_threshold:
            reward += 0.1
        elif curr <= knobs.wave_weak_threshold and next_val >= knobs.wave_strong_threshold:
            reward += 0.1
            
        # Penalty for consecutive strong or very weak
        if (curr >= knobs.wave_strong_threshold and next_val >= knobs.wave_strong_threshold):
            penalty += 0.2
        elif (curr <= knobs.wave_weak_threshold and next_val <= knobs.wave_weak_threshold):
            penalty += 0.2
    
    return reward - penalty


def calculate_sanity_flags(
    schedule: Dict[str, Any], 
    max_need: float,
    knobs: EvaluationKnobs
) -> Dict[str, bool]:
    """Calculate sanity check flags."""
    needpct = np.array(schedule["needpct"])
    indent_pct = np.array(schedule["indent_pct"])
    volume_pct = np.array(schedule["volume_pct"])
    price_step_pct = np.array(schedule["price_step_pct"])
    
    flags = {}
    
    # Max need mismatch
    calculated_max = float(np.max(needpct)) if len(needpct) > 0 else 0.0
    flags["max_need_mismatch"] = abs(max_need - calculated_max) > knobs.max_need_tolerance
    
    # Collapsed indents (non-monotonic or too small steps)
    if len(price_step_pct) > 1:
        min_step = float(np.min(price_step_pct))
        non_monotonic = bool(np.any(np.diff(indent_pct) < 0))
        flags["collapse_indents"] = non_monotonic or min_step < knobs.collapse_threshold
    else:
        flags["collapse_indents"] = False
    
    # Tail overflow
    last_volume_pct = volume_pct[-1] if len(volume_pct) > 0 else 0.0
    flags["tail_overflow"] = last_volume_pct > (knobs.tail_cap * 100.0)
    
    return flags


def calculate_diagnostics(schedule: Dict[str, Any]) -> Dict[str, float]:
    """Calculate diagnostic metrics."""
    volume_pct = np.array(schedule["volume_pct"])
    needpct = np.array(schedule["needpct"])
    
    # Weight Center Index
    wci = _weight_center_index(volume_pct)
    
    # Sign flips in NeedPct trend
    sign_flips = _count_sign_flips(needpct)
    
    # Gini coefficient
    gini = _gini_coefficient(volume_pct)
    
    # Entropy
    entropy = _entropy_normalized(volume_pct)
    
    return {
        "wci": float(wci),
        "sign_flips": int(sign_flips),
        "gini": float(gini),
        "entropy": float(entropy),
    }


def calculate_penalties(
    schedule: Dict[str, Any],
    sanity: Dict[str, bool],
    knobs: EvaluationKnobs
) -> Dict[str, float]:
    """Calculate all penalty components (normalized to [0,1])."""
    volume_pct = np.array(schedule["volume_pct"])
    indent_pct = np.array(schedule["indent_pct"])
    price_step_pct = np.array(schedule["price_step_pct"])
    martingale_pct = np.array(schedule["martingale_pct"])
    
    penalties = {}
    
    # P_gini: High Gini penalty (volume concentration)
    gini = _gini_coefficient(volume_pct)
    penalties["P_gini"] = float(min(1.0, gini))
    
    # P_entropy: Low entropy penalty (lack of diversity)
    entropy = _entropy_normalized(volume_pct)
    penalties["P_entropy"] = float(max(0.0, 1.0 - entropy))
    
    # P_monotone: Non-monotonic indent steps penalty
    if len(indent_pct) > 1:
        violations = np.sum(np.maximum(0, -np.diff(indent_pct)))
        max_violation = np.sum(indent_pct[:-1])  # Worst case: all negative
        penalties["P_monotone"] = float(min(1.0, violations / max(max_violation, 1e-12)))
    else:
        penalties["P_monotone"] = 0.0
    
    # P_smooth: Price step smoothness penalty
    if len(price_step_pct) > 1:
        step_var = np.var(price_step_pct)
        max_var = np.var([0, 100])  # Normalize by max possible variance
        penalties["P_smooth"] = float(min(1.0, step_var / max(max_var, 1e-12)))
    else:
        penalties["P_smooth"] = 0.0
    
    # P_tailcap: Tail cap violation penalty
    last_vol = volume_pct[-1] if len(volume_pct) > 0 else 0.0
    cap_limit = knobs.tail_cap * 100.0
    if last_vol > cap_limit:
        penalties["P_tailcap"] = float(min(1.0, (last_vol - cap_limit) / cap_limit))
    else:
        penalties["P_tailcap"] = 0.0
    
    # P_need_mismatch: Sanity check penalty
    penalties["P_need_mismatch"] = 1.0 if sanity["max_need_mismatch"] else 0.0
    
    # Wave pattern penalty/reward
    wave_score = calculate_wave_pattern_score(martingale_pct, knobs)
    penalties["P_wave"] = float(max(0.0, -wave_score))  # Convert negative reward to penalty
    
    return penalties


def evaluation_function(
    base_price: float,
    overlap_pct: float, 
    num_orders: int,
    seed: Optional[int] = None,
    wave_pattern: bool = False,
    **kwargs
) -> Dict[str, Any]:
    """
    Main evaluation function implementing the complete DCA/Martingale contract.
    
    Returns dict with keys: score, max_need, var_need, tail, schedule, sanity, diagnostics
    """
    # Setup knobs
    knobs = EvaluationKnobs(
        alpha=kwargs.get("alpha", 0.5),
        beta=kwargs.get("beta", 0.3), 
        gamma=kwargs.get("gamma", 0.2),
        lambda_penalty=kwargs.get("lambda_penalty", 0.1),
        wave_pattern=wave_pattern,
        wave_strong_threshold=kwargs.get("wave_strong_threshold", 50.0),
        wave_weak_threshold=kwargs.get("wave_weak_threshold", 10.0),
        tail_cap=kwargs.get("tail_cap", 0.40),
        min_indent_step=kwargs.get("min_indent_step", 0.05),
        softmax_temp=kwargs.get("softmax_temp", 1.0),
        random_seed=seed
    )
    
    try:
        # Build schedule
        schedule = build_schedule(base_price, overlap_pct, num_orders, knobs)
        
        # Calculate core metrics
        needpct = np.array(schedule["needpct"])
        volume_pct = np.array(schedule["volume_pct"])
        
        max_need = float(np.max(needpct)) if len(needpct) > 0 else 0.0
        var_need = float(np.var(needpct)) if len(needpct) > 0 else 0.0
        
        # Calculate tail (last 20% of orders concentration)
        n = len(volume_pct)
        if n > 0:
            tail_start = max(0, int(0.8 * n))
            tail_volume = np.sum(volume_pct[tail_start:])
            tail = float(tail_volume / 100.0)  # Normalize to [0,1]
        else:
            tail = 0.0
        
        # Calculate sanity flags
        sanity = calculate_sanity_flags(schedule, max_need, knobs)
        
        # Calculate penalties
        penalties = calculate_penalties(schedule, sanity, knobs)
        
        # Calculate diagnostics
        diagnostics = calculate_diagnostics(schedule)
        
        # Calculate final score J
        penalty_sum = sum(penalties.values())
        score = (knobs.alpha * max_need + 
                knobs.beta * var_need + 
                knobs.gamma * tail + 
                knobs.lambda_penalty * penalty_sum)
        
        return {
            "score": float(score),
            "max_need": max_need,
            "var_need": var_need, 
            "tail": tail,
            "schedule": schedule,
            "sanity": sanity,
            "diagnostics": diagnostics,
            "penalties": penalties,
            "knobs": {
                "alpha": knobs.alpha,
                "beta": knobs.beta,
                "gamma": knobs.gamma,
                "lambda": knobs.lambda_penalty,
                "wave_pattern": knobs.wave_pattern
            }
        }
        
    except Exception as e:
        # Error fallback
        return {
            "score": float("inf"),
            "max_need": float("inf"),
            "var_need": float("inf"),
            "tail": float("inf"),
            "schedule": {
                "indent_pct": [],
                "volume_pct": [],
                "martingale_pct": [],
                "needpct": [],
                "order_prices": [],
                "price_step_pct": []
            },
            "sanity": {
                "max_need_mismatch": True,
                "collapse_indents": True,
                "tail_overflow": True
            },
            "diagnostics": {
                "wci": 0.0,
                "sign_flips": 0,
                "gini": 1.0,
                "entropy": 0.0
            },
            "penalties": {},
            "error": str(e)
        }