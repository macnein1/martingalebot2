"""
Simple DCA Engine - Pure Python Implementation
Multi-objective evaluation without numba dependencies for immediate functionality.
"""
from __future__ import annotations

import hashlib
import json
import numpy as np
from typing import Dict, Any, List, Tuple
import math


def softplus(x):
    """Softplus activation function."""
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)


def softmax(x, temperature=1.0):
    """Softmax with temperature."""
    x_scaled = x / temperature
    x_max = np.max(x_scaled)
    e = np.exp(x_scaled - x_max)
    return e / np.sum(e)


def gini_coefficient(x):
    """Calculate Gini coefficient."""
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


def entropy_normalized(x):
    """Calculate normalized entropy."""
    if len(x) <= 1:
        return 0.0
    
    x_sum = np.sum(x)
    if x_sum <= 1e-12:
        return 0.0
    
    p = x / x_sum
    p = p[p > 1e-12]  # Remove zeros
    entropy = -np.sum(p * np.log(p))
    
    max_entropy = math.log(len(x))
    if max_entropy <= 1e-12:
        return 0.0
    
    return entropy / max_entropy


def weight_center_index(weights):
    """Calculate Weight Center Index."""
    n = len(weights)
    if n <= 1:
        return 0.0
    
    indices = np.arange(n)
    w_sum = np.sum(weights)
    if w_sum <= 0:
        return 0.0
    
    center = np.sum(weights * indices) / (w_sum * (n - 1))
    return min(1.0, max(0.0, center))


def count_sign_flips(needpct):
    """Count sign flips in NeedPct trend."""
    if len(needpct) < 2:
        return 0
    
    diff = np.diff(needpct)
    if len(diff) < 2:
        return 0
    
    sign_changes = np.sum(diff[:-1] * diff[1:] < 0)
    return int(sign_changes)


def evaluate_simple_dca(
    base_price: float = 1.0,
    overlap_pct: float = 20.0,
    num_orders: int = 5,
    seed: int = None,
    wave_pattern: bool = False,
    **kwargs
) -> Dict[str, Any]:
    """
    Simple DCA evaluation function using pure Python.
    """
    # Setup configuration
    config = {
        'alpha': kwargs.get('alpha', 0.45),
        'beta': kwargs.get('beta', 0.20),
        'gamma': kwargs.get('gamma', 0.20),
        'delta': kwargs.get('delta', 0.10),
        'rho': kwargs.get('rho', 0.05),
        'wave_pattern': wave_pattern,
        'strong_threshold': kwargs.get('wave_strong_threshold', 50.0),
        'weak_threshold': kwargs.get('wave_weak_threshold', 10.0),
        'shape_template': kwargs.get('shape_template', 'late_surge'),
        'q_cvar': kwargs.get('q_cvar', 0.8),
        'tail_cap': kwargs.get('tail_cap', 0.40),
        'softmax_temp': kwargs.get('softmax_temp', 1.0),
        'min_indent_step': kwargs.get('min_indent_step', 0.01),
    }
    
    try:
        # Generate random parameters
        rng = np.random.default_rng(seed)
        raw_indents = rng.normal(0.0, 1.0, size=num_orders)
        raw_volumes = rng.normal(0.0, 1.0, size=num_orders)
        
        # 1. Build indents (monotonic increasing)
        steps = softplus(raw_indents)
        steps = np.maximum(steps, config['min_indent_step'] / 100.0)
        steps = steps * (overlap_pct / 100.0) / np.sum(steps)
        
        indents = np.concatenate([[0.0], np.cumsum(steps)]) * 100.0
        
        # 2. Build volumes (softmax normalized)
        volumes = softmax(raw_volumes, config['softmax_temp']) * 100.0
        
        # Apply tail cap
        if volumes[-1] > config['tail_cap'] * 100.0:
            excess = volumes[-1] - config['tail_cap'] * 100.0
            volumes[-1] = config['tail_cap'] * 100.0
            if num_orders > 1:
                other_sum = np.sum(volumes[:-1])
                if other_sum > 1e-9:
                    volumes[:-1] += volumes[:-1] * (excess / other_sum)
        
        # 3. Build martingales
        martingales = np.zeros(num_orders)
        for i in range(1, num_orders):
            if volumes[i-1] > 1e-12:
                ratio = volumes[i] / volumes[i-1]
                martingales[i] = max(1.0, min(100.0, (ratio - 1.0) * 100.0))
        
        # 4. Calculate order prices
        order_prices = np.empty(num_orders + 1)
        order_prices[0] = base_price
        for i in range(1, num_orders + 1):
            order_prices[i] = base_price * (1.0 - indents[i] / 100.0)
        
        # 5. Calculate Need% curve
        needpct = np.empty(num_orders)
        vol_acc = 0.0
        val_acc = 0.0
        
        for k in range(num_orders):
            vol_acc += volumes[k]
            val_acc += volumes[k] * order_prices[k+1]
            
            avg_entry_price = val_acc / max(vol_acc, 1e-12)
            current_price = order_prices[k+1]
            needpct[k] = (avg_entry_price / max(current_price, 1e-12) - 1.0) * 100.0
        
        # 6. Calculate metrics
        max_need = float(np.max(needpct))
        var_need = float(np.var(needpct))
        
        # Tail (last 20% concentration)
        tail_start = max(0, int(0.8 * num_orders))
        tail = float(np.sum(volumes[tail_start:]) / 100.0)
        
        # CVaR calculation
        sorted_need = np.sort(needpct)
        q_index = int(config['q_cvar'] * len(sorted_need))
        if q_index < len(sorted_need):
            cvar_need = float(np.mean(sorted_need[q_index:]))
        else:
            cvar_need = float(sorted_need[-1])
        
        # Shape reward (late surge template)
        if config['shape_template'] == 'late_surge':
            # Simple late surge: reward increasing towards end
            n = len(volumes)
            template = np.array([0.3 + 0.7 * (i / (n-1)) for i in range(n)])
            template = template / np.sum(template)
            vol_norm = volumes / np.sum(volumes)
            
            # Cosine similarity
            dot_product = np.dot(vol_norm, template)
            norm_vol = np.linalg.norm(vol_norm)
            norm_template = np.linalg.norm(template)
            
            if norm_vol > 1e-12 and norm_template > 1e-12:
                shape_reward = max(0.0, dot_product / (norm_vol * norm_template))
            else:
                shape_reward = 0.0
        else:
            shape_reward = 0.5  # Default for other templates
        
        # Wave pattern reward
        wave_reward = 0.0
        if config['wave_pattern'] and len(martingales) > 2:
            for i in range(2, len(martingales)):
                prev_mart = martingales[i-1]
                curr_mart = martingales[i]
                
                # Reward alternating patterns
                if prev_mart >= config['strong_threshold'] and curr_mart <= config['weak_threshold']:
                    wave_reward += 0.1
                elif prev_mart <= config['weak_threshold'] and curr_mart >= config['strong_threshold']:
                    wave_reward += 0.1
                
                # Penalty for consecutive patterns
                if prev_mart >= config['strong_threshold'] and curr_mart >= config['strong_threshold']:
                    wave_reward -= 0.15
                elif prev_mart <= config['weak_threshold'] and curr_mart <= config['weak_threshold']:
                    wave_reward -= 0.15
        
        # Calculate sanity flags
        calculated_max = float(np.max(needpct))
        sanity = {
            "max_need_mismatch": abs(max_need - calculated_max) > 1e-6,
            "collapse_indents": np.any(np.diff(indents[1:]) < 0.01),
            "tail_overflow": volumes[-1] > config['tail_cap'] * 100.0
        }
        
        # Calculate diagnostics
        diagnostics = {
            "wci": weight_center_index(volumes),
            "sign_flips": count_sign_flips(needpct),
            "gini": gini_coefficient(volumes / 100.0),
            "entropy": entropy_normalized(volumes)
        }
        
        # Calculate final score
        shape_penalty = 1.0 - shape_reward
        wave_penalty = max(0.0, -wave_reward)
        
        # Simple penalties
        gini_penalty = gini_coefficient(volumes / 100.0)
        monotone_penalty = float(np.sum(np.maximum(0, -np.diff(indents[1:]))))
        
        score = (config['alpha'] * max_need +
                config['beta'] * var_need +
                config['gamma'] * tail +
                config['delta'] * shape_penalty +
                config['rho'] * cvar_need +
                0.1 * (gini_penalty + monotone_penalty + wave_penalty))
        
        # Build schedule
        schedule = {
            "indent_pct": indents[1:].tolist(),
            "volume_pct": volumes.tolist(),
            "martingale_pct": martingales.tolist(),
            "needpct": needpct.tolist(),
            "order_prices": order_prices.tolist(),
            "price_step_pct": np.diff(indents).tolist()
        }
        
        # Create parameters
        params = {
            "base_price": base_price,
            "overlap_pct": overlap_pct,
            "num_orders": num_orders,
            "alpha": config['alpha'],
            "beta": config['beta'],
            "gamma": config['gamma'],
            "delta": config['delta'],
            "rho": config['rho'],
            "wave_pattern": wave_pattern,
            "tail_cap": config['tail_cap'],
            "shape_template": config['shape_template']
        }
        
        # Generate stable ID
        stable_id = hashlib.sha1(json.dumps(params, sort_keys=True).encode()).hexdigest()[:16]
        
        return {
            "score": float(score),
            "max_need": max_need,
            "var_need": var_need,
            "tail": tail,
            "shape_reward": shape_reward,
            "cvar_need": cvar_need,
            
            "schedule": schedule,
            "sanity": sanity,
            "diagnostics": diagnostics,
            
            "penalties": {
                "gini_penalty": gini_penalty,
                "monotone_penalty": monotone_penalty,
                "wave_reward": wave_reward,
                "shape_penalty": shape_penalty
            },
            
            "params": params,
            "stable_id": stable_id,
            
            "knobs": {
                "alpha": config['alpha'],
                "beta": config['beta'],
                "gamma": config['gamma'],
                "delta": config['delta'],
                "rho": config['rho'],
                "wave_pattern": config['wave_pattern']
            }
        }
        
    except Exception as e:
        # Error fallback
        return {
            "score": float("inf"),
            "max_need": float("inf"),
            "var_need": float("inf"),
            "tail": float("inf"),
            "shape_reward": 0.0,
            "cvar_need": float("inf"),
            
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
            "params": {"error": str(e)},
            "stable_id": None,
            "knobs": {},
            "error": str(e)
        }


def create_bullets_format(schedule: Dict[str, Any]) -> List[str]:
    """Create bullets in exact specified format."""
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
            bullet = f"{i+1}. Emir: Indent %{indent:.2f}  Volume %{volume:.2f}  (no martingale, first order) — NeedPct %{need:.2f}"
        else:
            bullet = f"{i+1}. Emir: Indent %{indent:.2f}  Volume %{volume:.2f}  (Martingale %{martingale:.2f}) — NeedPct %{need:.2f}"
        
        bullets.append(bullet)
    
    return bullets


def validate_simple_result(result: Dict[str, Any]) -> Tuple[bool, str]:
    """Validate evaluation result."""
    # Check required keys
    required_keys = ['score', 'max_need', 'var_need', 'tail', 'schedule', 'sanity', 'diagnostics']
    for key in required_keys:
        if key not in result:
            return False, f"Missing required key: {key}"
    
    # Check finite score
    if not np.isfinite(result['score']):
        return False, "Score is not finite"
    
    # Check schedule
    schedule = result['schedule']
    if not schedule.get('volume_pct'):
        return False, "Empty volume_pct"
    
    volume_sum = sum(schedule['volume_pct'])
    if abs(volume_sum - 100.0) > 1e-3:
        return False, f"Volume sum {volume_sum:.3f} != 100.0"
    
    return True, "Valid"