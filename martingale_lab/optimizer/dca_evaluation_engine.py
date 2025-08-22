"""
DCA Evaluation Engine - Integrated Multi-Objective Evaluation
Combines all components: penalties, constraints, JIT kernels, and scoring.
"""
from __future__ import annotations

import hashlib
import json
import numpy as np
from typing import Dict, Any, List, Tuple
import math

from martingale_lab.core.penalties import compute_all_penalties, compute_composite_score
from martingale_lab.core.constraints import apply_soft_constraints, DEFAULT_CONSTRAINT_WEIGHTS
from martingale_lab.core.jit_kernels import evaluate_single_candidate, need_curve_calculation


def create_stable_id(params: Dict[str, Any]) -> str:
    """Create stable ID from parameters."""
    # Select key parameters for stable ID
    key_params = {
        "base_price": params.get("base_price", 1.0),
        "overlap_pct": params.get("overlap_pct", 20.0),
        "num_orders": params.get("num_orders", 5),
        "alpha": params.get("alpha", 0.45),
        "beta": params.get("beta", 0.20),
        "gamma": params.get("gamma", 0.20),
        "delta": params.get("delta", 0.10),
        "rho": params.get("rho", 0.05),
        "wave_pattern": params.get("wave_pattern", False),
        "tail_cap": params.get("tail_cap", 0.40),
    }
    
    params_str = json.dumps(key_params, sort_keys=True)
    return hashlib.sha1(params_str.encode()).hexdigest()[:16]


def evaluate_dca_candidate(
    base_price: float = 1.0,
    overlap_pct: float = 20.0,
    num_orders: int = 5,
    seed: int = None,
    wave_pattern: bool = False,
    **kwargs
) -> Dict[str, Any]:
    """
    Main DCA evaluation function with complete multi-objective scoring.
    
    Args:
        base_price: Base entry price
        overlap_pct: Total overlap percentage
        num_orders: Number of orders
        seed: Random seed for reproducibility
        wave_pattern: Enable wave pattern rewards
        **kwargs: Additional configuration parameters
        
    Returns:
        Complete evaluation result dictionary
    """
    # Setup configuration
    config = {
        # Scoring weights
        'alpha': kwargs.get('alpha', 0.45),
        'beta': kwargs.get('beta', 0.20),
        'gamma': kwargs.get('gamma', 0.20),
        'delta': kwargs.get('delta', 0.10),
        'rho': kwargs.get('rho', 0.05),
        'eta': kwargs.get('eta', 0.02),
        'zeta': kwargs.get('zeta', 0.02),
        
        # Wave pattern
        'wave_pattern': wave_pattern,
        'strong_threshold': kwargs.get('wave_strong_threshold', 50.0),
        'weak_threshold': kwargs.get('wave_weak_threshold', 10.0),
        
        # Shape template
        'shape_template': kwargs.get('shape_template', 'late_surge'),
        'q_cvar': kwargs.get('q_cvar', 0.8),
        
        # Gini weights for tail penalty
        'gini_w_vol': kwargs.get('gini_w_vol', 0.7),
        'gini_w_mart': kwargs.get('gini_w_mart', 0.3),
        
        # Constraint parameters
        'softmax_temperature': kwargs.get('softmax_temp', 1.0),
        'min_indent_step': kwargs.get('min_indent_step', 0.01),
        'min_martingale_pct': kwargs.get('min_martingale_pct', 1.0),
        'max_martingale_pct': kwargs.get('max_martingale_pct', 100.0),
        'tail_cap_pct': kwargs.get('tail_cap', 0.40) * 100,  # Convert to percentage
        'head_cap_pct': kwargs.get('head_cap', 0.60) * 100,
        'min_volume_pct': kwargs.get('min_volume_pct', 0.1),
        'max_volume_pct': kwargs.get('max_volume_pct', 80.0),
    }
    
    try:
        # Generate random parameters
        rng = np.random.default_rng(seed)
        raw_indents = rng.normal(0.0, 1.0, size=num_orders)
        raw_volumes = rng.normal(0.0, 1.0, size=num_orders)
        raw_martingales = rng.normal(0.0, 1.0, size=num_orders)
        
        # Apply soft constraints and normalization
        indents, volumes, martingales, constraint_penalties = apply_soft_constraints(
            raw_indents, raw_volumes, raw_martingales, overlap_pct, config
        )
        
        # Calculate order prices
        prices = np.empty(len(indents))
        for i in range(len(indents)):
            if i == 0:
                prices[i] = base_price
            else:
                indent_fraction = indents[i] / 100.0
                prices[i] = base_price * (1.0 - indent_fraction)
                prices[i] = max(prices[i], base_price * 0.01)  # Safety bound
        
        # Calculate Need% curve
        need_pct = need_curve_calculation(volumes, prices)
        
        # Build schedule dictionary
        schedule = {
            "indent_pct": indents[1:].tolist(),  # Exclude first 0
            "volume_pct": volumes.tolist(),
            "martingale_pct": martingales.tolist(),
            "needpct": need_pct.tolist(),
            "order_prices": prices.tolist(),
            "price_step_pct": np.diff(indents).tolist() if len(indents) > 1 else []
        }
        
        # Calculate all penalties and rewards
        penalties = compute_all_penalties(volumes, martingales, indents[1:], need_pct, config)
        
        # Add constraint penalties
        for key, value in constraint_penalties.items():
            penalties[f"constraint_{key}"] = value
        
        # Calculate core metrics
        max_need = penalties['max_need']
        var_need = penalties['var_need']
        tail_penalty = penalties['tail_penalty']
        shape_reward = penalties['shape_reward']
        cvar_need = penalties['cvar_need']
        
        # Calculate sanity flags
        sanity = {
            "max_need_mismatch": abs(max_need - np.max(need_pct)) > 1e-6 if len(need_pct) > 0 else False,
            "collapse_indents": penalties['monotone_penalty'] > 0.1,
            "tail_overflow": volumes[-1] > config['tail_cap_pct'] if len(volumes) > 0 else False
        }
        
        # Calculate diagnostics
        diagnostics = {
            "wci": penalties.get('wci', 0.0),
            "sign_flips": int(penalties.get('sign_flips', 0)),
            "gini": penalties['gini_volumes'],
            "entropy": penalties['entropy_volumes']
        }
        
        # Compute final composite score
        score_weights = {
            'alpha': config['alpha'],
            'beta': config['beta'], 
            'gamma': config['gamma'],
            'delta': config['delta'],
            'rho': config['rho'],
            'eta': config['eta'],
            'zeta': config['zeta']
        }
        
        composite_score = compute_composite_score(penalties, score_weights)
        
        # Add constraint penalty
        constraint_weight_total = sum(
            DEFAULT_CONSTRAINT_WEIGHTS.get(k.replace('constraint_', ''), 0.0) * v 
            for k, v in penalties.items() if k.startswith('constraint_')
        )
        
        final_score = composite_score + 0.1 * constraint_weight_total
        
        # Create parameters dictionary
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
            "tail_cap": kwargs.get('tail_cap', 0.40),
            "shape_template": config['shape_template'],
            "q_cvar": config['q_cvar']
        }
        
        # Generate stable ID
        stable_id = create_stable_id(params)
        
        # Return complete evaluation result
        return {
            "score": float(final_score),
            "max_need": float(max_need),
            "var_need": float(var_need),
            "tail": float(tail_penalty),
            "shape_reward": float(shape_reward),
            "cvar_need": float(cvar_need),
            
            "schedule": schedule,
            "sanity": sanity,
            "diagnostics": diagnostics,
            "penalties": {k: float(v) for k, v in penalties.items()},
            
            "params": params,
            "stable_id": stable_id,
            
            "knobs": {
                "alpha": config['alpha'],
                "beta": config['beta'],
                "gamma": config['gamma'],
                "delta": config['delta'],
                "rho": config['rho'],
                "wave_pattern": config['wave_pattern'],
                "shape_template": config['shape_template']
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


def batch_evaluate_dca_candidates(
    candidates_params: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Evaluate a batch of DCA candidates.
    
    Args:
        candidates_params: List of parameter dictionaries for each candidate
        
    Returns:
        List of evaluation results
    """
    results = []
    
    for params in candidates_params:
        result = evaluate_dca_candidate(**params)
        results.append(result)
    
    return results


def create_bullets_text(schedule: Dict[str, Any]) -> List[str]:
    """
    Create bullets text in the exact specified format.
    
    Args:
        schedule: Schedule dictionary
        
    Returns:
        List of bullet strings
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
            bullet = f"{i+1}. Emir: Indent %{indent:.2f}  Volume %{volume:.2f}  (no martingale, first order) — NeedPct %{need:.2f}"
        else:
            bullet = f"{i+1}. Emir: Indent %{indent:.2f}  Volume %{volume:.2f}  (Martingale %{martingale:.2f}) — NeedPct %{need:.2f}"
        
        bullets.append(bullet)
    
    return bullets


def validate_evaluation_result(result: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Validate evaluation result structure and values.
    
    Args:
        result: Evaluation result dictionary
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    # Check required keys
    required_keys = ['score', 'max_need', 'var_need', 'tail', 'schedule', 'sanity', 'diagnostics']
    for key in required_keys:
        if key not in result:
            return False, f"Missing required key: {key}"
    
    # Check schedule structure
    schedule = result['schedule']
    schedule_keys = ['indent_pct', 'volume_pct', 'martingale_pct', 'needpct', 'order_prices']
    for key in schedule_keys:
        if key not in schedule:
            return False, f"Missing schedule key: {key}"
    
    # Check dimensions consistency
    volume_pct = schedule['volume_pct']
    martingale_pct = schedule['martingale_pct']
    needpct = schedule['needpct']
    
    if not (len(volume_pct) == len(martingale_pct) == len(needpct)):
        return False, "Schedule dimensions are inconsistent"
    
    # Check volume sum
    if len(volume_pct) > 0:
        volume_sum = sum(volume_pct)
        if abs(volume_sum - 100.0) > 1e-3:
            return False, f"Volume sum {volume_sum:.3f} != 100.0"
    
    # Check martingale first element
    if len(martingale_pct) > 0 and abs(martingale_pct[0]) > 1e-6:
        return False, f"First martingale {martingale_pct[0]:.3f} != 0.0"
    
    # Check finite values
    if not np.isfinite(result['score']):
        return False, "Score is not finite"
    
    return True, "Valid"