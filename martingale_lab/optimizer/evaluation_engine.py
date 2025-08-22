"""
Evaluation Engine for DCA/Martingale Optimization
Implements evaluation_function exactly as defined in README specification.
"""
from __future__ import annotations

import hashlib
import json
import numpy as np
import time
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple
import math

from martingale_lab.utils.structured_logging import (
    get_structured_logger, EventNames, Timer, ensure_json_serializable
)

# Initialize structured logger for evaluation
logger = get_structured_logger("mlab.eval")


def _ensure_json_serializable(obj):
    """Convert numpy arrays and scalars to JSON-serializable types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.generic):
        return obj.item()
    elif isinstance(obj, (list, tuple)):
        return [_ensure_json_serializable(x) for x in obj]
    elif isinstance(obj, dict):
        return {k: _ensure_json_serializable(v) for k, v in obj.items()}
    else:
        return obj


def _softplus(x: np.ndarray) -> np.ndarray:
    """Softplus activation: log(1 + exp(x))"""
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)


def _softmax(x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Softmax with temperature control."""
    x_scaled = x / temperature
    x_max = np.max(x_scaled)
    e = np.exp(x_scaled - x_max)
    return e / np.sum(e)


def _gini_coefficient(x: np.ndarray) -> float:
    """Calculate Gini coefficient for inequality measurement."""
    if len(x) == 0:
        return 0.0
    x_sorted = np.sort(x)
    n = len(x_sorted)
    cumsum_x = np.cumsum(x_sorted)
    total_sum = cumsum_x[-1]
    
    if total_sum <= 1e-12:
        return 0.0
    
    weighted_sum = np.sum((np.arange(n) + 1) * x_sorted)
    gini = (2.0 * weighted_sum) / (n * total_sum) - (n + 1.0) / n
    
    return max(0.0, min(1.0, gini))


def _entropy_normalized(x: np.ndarray) -> float:
    """Calculate normalized entropy for diversity measurement."""
    if len(x) <= 1:
        return 0.0
    
    x_sum = np.sum(x)
    if x_sum <= 1e-12:
        return 0.0
    
    p = x / x_sum
    p = p[p > 1e-12]  # Remove zeros for log calculation
    entropy = -np.sum(p * np.log(p))
    
    max_entropy = math.log(len(x))
    if max_entropy <= 1e-12:
        return 0.0
    
    return entropy / max_entropy


def _weight_center_index(weights: np.ndarray) -> float:
    """Calculate Weight Center Index (0=early, 1=late load)."""
    n = len(weights)
    if n <= 1:
        return 0.0
    
    indices = np.arange(n)
    w_sum = np.sum(weights)
    if w_sum <= 0:
        return 0.0
    
    center = np.sum(weights * indices) / (w_sum * (n - 1))
    return min(1.0, max(0.0, center))


def _count_sign_flips(needpct: np.ndarray) -> int:
    """Count sign flips in NeedPct trend."""
    if len(needpct) < 2:
        return 0
    
    diff = np.diff(needpct)
    if len(diff) < 2:
        return 0
    
    sign_changes = np.sum(diff[:-1] * diff[1:] < 0)
    return int(sign_changes)


def evaluation_function(
    base_price: float, 
    overlap_pct: float, 
    num_orders: int,
    seed: Optional[int] = None,
    wave_pattern: bool = True,
    alpha: float = 0.5,
    beta: float = 0.3,
    gamma: float = 0.2,
    lambda_penalty: float = 0.1,
    wave_strong_threshold: float = 50.0,
    wave_weak_threshold: float = 10.0,
    tail_cap: float = 0.40,
    min_indent_step: float = 0.05,
    softmax_temp: float = 1.0,
    **kwargs
) -> Dict[str, Any]:
    """
    DCA/Martingale evaluation function exactly as defined in README.
    
    Returns complete dict with all required outputs, always JSON-serializable.
    Never throws exceptions - returns error state in dict if needed.
    """
    start_time = time.time()
    
    # Log evaluation call
    logger.info(
        EventNames.EVAL_CALL,
        "Starting evaluation",
        overlap=overlap_pct,
        orders=num_orders,
        wave_pattern=wave_pattern,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        lambda_penalty=lambda_penalty
    )
    
    try:
        # Generate random parameters
        rng = np.random.default_rng(seed)
        raw_indents = rng.normal(0.0, 1.0, size=num_orders)
        raw_volumes = rng.normal(0.0, 1.0, size=num_orders)
        
        # 1. Build indent_pct with monotonic increasing steps
        steps = _softplus(raw_indents)
        steps = np.maximum(steps, min_indent_step / 100.0)
        steps_sum = np.sum(steps)
        if steps_sum <= 1e-12:
            steps = np.full(num_orders, overlap_pct / (100.0 * num_orders))
        else:
            steps = steps * (overlap_pct / 100.0) / steps_sum
        
        # Cumulative indents: [p0=0, p1, p2, ..., pM]
        indent_cumulative = np.concatenate([[0.0], np.cumsum(steps)]) * 100.0
        indent_pct = indent_cumulative[1:].tolist()  # [p1, p2, ..., pM] for output
        
        # 2. Build volume_pct using softmax (sums to 100)
        volume_raw = _softmax(raw_volumes, softmax_temp)
        volume_pct = volume_raw * 100.0
        
        # Apply tail cap constraint
        if volume_pct[-1] > tail_cap * 100.0:
            excess = volume_pct[-1] - tail_cap * 100.0
            volume_pct[-1] = tail_cap * 100.0
            if num_orders > 1:
                other_sum = np.sum(volume_pct[:-1])
                if other_sum > 1e-9:
                    volume_pct[:-1] += volume_pct[:-1] * (excess / other_sum)
        
        # 3. Build martingale_pct (m1=0, m2...mM calculated)
        martingale_pct = np.zeros(num_orders)
        for i in range(1, num_orders):
            if volume_pct[i-1] > 1e-12:
                ratio = volume_pct[i] / volume_pct[i-1]
                martingale_pct[i] = max(1.0, min(100.0, (ratio - 1.0) * 100.0))
        
        # 4. Calculate order prices
        order_prices = np.empty(num_orders + 1)
        order_prices[0] = base_price  # Base price
        for i in range(1, num_orders + 1):
            order_prices[i] = base_price * (1.0 - indent_cumulative[i] / 100.0)
        
        # 5. Calculate price step percentages
        price_step_pct = np.diff(indent_cumulative).tolist()
        
        # 6. Calculate NeedPct sequence using exact formula from README
        needpct = np.empty(num_orders)
        vol_acc = 0.0
        val_acc = 0.0
        
        for k in range(num_orders):
            vol_acc += volume_pct[k]
            val_acc += volume_pct[k] * order_prices[k+1]
            
            # Weighted average entry price
            avg_entry_price = val_acc / max(vol_acc, 1e-12)
            
            # Current order price
            current_price = order_prices[k+1]
            
            # Need percentage: (avg_entry / current_price - 1) * 100
            needpct[k] = (avg_entry_price / max(current_price, 1e-12) - 1.0) * 100.0
        
        # Calculate core metrics
        max_need = float(np.max(needpct))
        var_need = float(np.var(needpct))
        
        # Tail calculation (last 20% of orders)
        tail_start = max(0, int(0.8 * num_orders))
        tail = float(np.sum(volume_pct[tail_start:]) / 100.0)
        
        # Build schedule dict with all required fields
        schedule = {
            "indent_pct": _ensure_json_serializable(indent_pct),
            "volume_pct": _ensure_json_serializable(volume_pct.tolist()),
            "martingale_pct": _ensure_json_serializable(martingale_pct.tolist()),
            "needpct": _ensure_json_serializable(needpct.tolist()),
            "order_prices": _ensure_json_serializable(order_prices.tolist()),
            "price_step_pct": _ensure_json_serializable(price_step_pct)
        }
        
        # Calculate sanity checks
        calculated_max_need = float(np.max(needpct))
        sanity = {
            "max_need_mismatch": bool(abs(max_need - calculated_max_need) > 1e-6),
            "collapse_indents": bool(np.any(np.diff(indent_cumulative[1:]) < min_indent_step)),
            "tail_overflow": bool(volume_pct[-1] > tail_cap * 100.0)
        }
        
        # Calculate diagnostics
        diagnostics = {
            "wci": float(_weight_center_index(volume_pct)),
            "sign_flips": int(_count_sign_flips(needpct)),
            "gini": float(_gini_coefficient(volume_pct / 100.0)),
            "entropy": float(_entropy_normalized(volume_pct))
        }
        
        # Calculate all penalties (always present, even if zero)
        penalties = {}
        
        # P_gini: Volume concentration penalty
        penalties["P_gini"] = float(_gini_coefficient(volume_pct / 100.0))
        
        # P_entropy: Low diversity penalty
        penalties["P_entropy"] = float(max(0.0, 1.0 - _entropy_normalized(volume_pct)))
        
        # P_monotone: Non-monotonic indent penalty
        monotone_violations = np.sum(np.maximum(0, -np.diff(indent_cumulative[1:])))
        penalties["P_monotone"] = float(monotone_violations)
        
        # P_smooth: Price step smoothness penalty
        if len(price_step_pct) > 1:
            step_var = np.var(price_step_pct)
            penalties["P_smooth"] = float(step_var / 100.0)  # Normalize
        else:
            penalties["P_smooth"] = 0.0
        
        # P_tailcap: Tail cap violation penalty
        if volume_pct[-1] > tail_cap * 100.0:
            penalties["P_tailcap"] = float((volume_pct[-1] - tail_cap * 100.0) / (tail_cap * 100.0))
        else:
            penalties["P_tailcap"] = 0.0
        
        # P_need_mismatch: Sanity check penalty
        penalties["P_need_mismatch"] = float(1.0 if sanity["max_need_mismatch"] else 0.0)
        
        # P_wave: Wave pattern penalty/reward
        wave_score = 0.0
        if wave_pattern and len(martingale_pct) > 2:
            for i in range(2, len(martingale_pct)):
                prev_mart = martingale_pct[i-1]
                curr_mart = martingale_pct[i]
                
                # Reward alternating patterns
                if prev_mart >= wave_strong_threshold and curr_mart <= wave_weak_threshold:
                    wave_score += 0.1
                elif prev_mart <= wave_weak_threshold and curr_mart >= wave_strong_threshold:
                    wave_score += 0.1
                
                # Penalty for consecutive patterns
                if prev_mart >= wave_strong_threshold and curr_mart >= wave_strong_threshold:
                    wave_score -= 0.2
                elif prev_mart <= wave_weak_threshold and curr_mart <= wave_weak_threshold:
                    wave_score -= 0.2
        
        penalties["P_wave"] = float(max(0.0, -wave_score))  # Convert negative reward to penalty
        
        # Calculate final score using exact README formula: J = α·max_need + β·var_need + γ·tail + λ·Σ(penalties)
        penalty_sum = sum(penalties.values())
        score = alpha * max_need + beta * var_need + gamma * tail + lambda_penalty * penalty_sum
        
        # Log successful evaluation
        duration_ms = (time.time() - start_time) * 1000
        logger.info(
            EventNames.EVAL_RETURN,
            "Evaluation completed successfully",
            score=float(score),
            max_need=float(max_need),
            var_need=float(var_need),
            tail=float(tail),
            duration_ms=duration_ms,
            sanity_violations=sum(1 for v in sanity.values() if v),
            penalty_sum=penalty_sum
        )
        
        # Return complete dict exactly as specified in README
        return {
            "score": float(score),
            "max_need": float(max_need),
            "var_need": float(var_need),
            "tail": float(tail),
            
            "schedule": schedule,
            "sanity": sanity,
            "diagnostics": diagnostics,
            "penalties": penalties
        }
        
    except Exception as e:
        # Log evaluation error
        duration_ms = (time.time() - start_time) * 1000
        logger.error(
            EventNames.EVAL_ERROR,
            f"Evaluation failed: {str(e)}",
            error=str(e),
            duration_ms=duration_ms,
            overlap=overlap_pct,
            orders=num_orders
        )
        
        # Never throw - return error state as complete dict
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
            
            "penalties": {
                "P_gini": 1.0,
                "P_entropy": 1.0,
                "P_monotone": 1.0,
                "P_smooth": 1.0,
                "P_tailcap": 1.0,
                "P_need_mismatch": 1.0,
                "P_wave": 1.0
            },
            
            "_error": str(e)
        }


# Legacy compatibility wrapper
def evaluate_configuration(overlap_pct: float, num_orders: int, **kwargs) -> Tuple[float, Dict[str, Any]]:
    """
    Legacy wrapper that returns (score, metrics) tuple.
    """
    result = evaluation_function(
        base_price=kwargs.get('base_price', 1.0),
        overlap_pct=overlap_pct,
        num_orders=num_orders,
        **kwargs
    )
    
    score = result["score"]
    return score, result


@dataclass
class ExperimentConfig:
    """Configuration for experiment runs."""
    base_price: float = 1.0
    overlap_min: float = 10.0
    overlap_max: float = 20.0
    orders_min: int = 5
    orders_max: int = 20
    n_candidates_per_M: int = 50_000
    seed: int = 123
    top_k_global: int = 100
    
    # New DCA v2 parameters
    alpha: float = 0.5
    beta: float = 0.3
    gamma: float = 0.2
    lambda_penalty: float = 0.1
    wave_pattern: bool = True
    tail_cap: float = 0.40


def create_bullets_format(schedule: Dict[str, Any]) -> List[str]:
    """
    Create bullets format exactly as specified in README.
    
    Format:
    1. Emir: Indent %0.00 Volume %x.xx (no martingale, first order) — NeedPct %n1
    2. Emir: Indent %p2 Volume %v2 (Martingale %m2) — NeedPct %n2
    """
    indent_pct = schedule.get("indent_pct", [])
    volume_pct = schedule.get("volume_pct", [])
    martingale_pct = schedule.get("martingale_pct", [])
    needpct = schedule.get("needpct", [])
    
    bullets = []
    n = len(volume_pct)
    
    for i in range(n):
        indent = indent_pct[i] if i < len(indent_pct) else 0.0
        volume = volume_pct[i]
        martingale = martingale_pct[i]
        need = needpct[i] if i < len(needpct) else 0.0
        
        if i == 0:
            bullet = f"{i+1}. Emir: Indent %{indent:.2f} Volume %{volume:.2f} (no martingale, first order) — NeedPct %{need:.2f}"
        else:
            bullet = f"{i+1}. Emir: Indent %{indent:.2f} Volume %{volume:.2f} (Martingale %{martingale:.2f}) — NeedPct %{need:.2f}"
        
        bullets.append(bullet)
    
    return bullets


def run_experiment(cfg: ExperimentConfig) -> Dict[str, Any]:
    """
    Run experiment using the new evaluation_function.
    """
    all_results = []
    rng = np.random.default_rng(cfg.seed)
    
    for M in range(cfg.orders_min, cfg.orders_max + 1):
        for _ in range(min(cfg.n_candidates_per_M // (cfg.orders_max - cfg.orders_min + 1), 1000)):
            overlap = rng.uniform(cfg.overlap_min, cfg.overlap_max)
            eval_seed = int(rng.integers(0, 2**31 - 1))
            
            result = evaluation_function(
                base_price=cfg.base_price,
                overlap_pct=overlap,
                num_orders=M,
                seed=eval_seed,
                alpha=cfg.alpha,
                beta=cfg.beta,
                gamma=cfg.gamma,
                lambda_penalty=cfg.lambda_penalty,
                wave_pattern=cfg.wave_pattern,
                tail_cap=cfg.tail_cap
            )
            
            # Add metadata
            result["overlap_pct"] = overlap
            result["orders"] = M
            
            all_results.append(result)
    
    if not all_results:
        return {"top": [], "best": None}
    
    # Sort by score (lower is better)
    all_results.sort(key=lambda x: x["score"])
    top_results = all_results[:cfg.top_k_global]
    best_result = top_results[0]
    
    # Create bullets for best result
    bullets = create_bullets_format(best_result["schedule"])
    
    return {
        "top": top_results,
        "best": {
            **best_result,
            "bullets": bullets
        }
    }