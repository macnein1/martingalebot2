"""
Evaluation Engine for DCA/Martingale Optimization
Implements evaluation_function exactly as defined in README specification.
"""
from __future__ import annotations

import json
import numpy as np
import time
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple
import math

from martingale_lab.utils.logging import get_eval_logger, should_log_eval
from martingale_lab.core.constraints import enforce_schedule_shape_fixed
from martingale_lab.core.penalties import compute_shape_penalties

# Use the new centralized logging system
logger = get_eval_logger()


# Penalty weight presets
PRESET_WEIGHTS = {
    "explore": {
        "w_fixed": 2.0,
        "w_second": 2.0,
        "w_gband": 1.5,
        "w_front": 2.0,
        "w_tv": 0.5,
        "w_wave": 0.5
    },
    "robust": {
        "w_fixed": 3.0,
        "w_second": 3.0,
        "w_gband": 2.5,
        "w_front": 3.0,
        "w_tv": 1.5,
        "w_wave": 1.0
    },
    "tight": {
        "w_fixed": 4.0,
        "w_second": 4.0,
        "w_gband": 3.0,
        "w_front": 4.0,
        "w_tv": 2.0,
        "w_wave": 1.5
    }
}


def batch_evaluate(candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Vectorized batch evaluation skeleton. Currently falls back to per-item eval."""
    if not candidates:
        return []
    results: List[Dict[str, Any]] = []
    for c in candidates:
        res = evaluation_function(**c)
        # stable_id can be derived by orchestrator; keep payload minimal here
        results.append(res)
    return results


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


def _interp_from_anchors(num_orders: int, anchors: int, rng: np.random.Generator,
                         v0: float, g1_max: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # t in [0,1] at order indices
    t = np.linspace(0.0, 1.0, num_orders)
    # choose anchor positions uniformly in t-space excluding first 2 indices
    anchor_idx = np.linspace(2, num_orders - 1, anchors, dtype=int)
    # sample log-volumes for anchors with mild smoothness
    logv = rng.normal(0.0, 0.5, size=anchors)
    # piecewise-linear in log-space
    v = np.zeros(num_orders, dtype=np.float64)
    v[0] = v0
    # sample v1 <= v0
    v[1] = min(v0, v0 * max(0.0, min(1.0, rng.normal(0.95, 0.03)))) if num_orders >= 2 else v0
    # set anchor values (positive)
    vals = np.exp(logv)
    # normalize anchors roughly around median scale of v1
    vals *= (v[1] + 1e-6) / (np.median(vals) + 1e-6)
    # linear interpolate anchors for i>=2
    for i in range(2, num_orders):
        # find surrounding anchors
        right = np.searchsorted(anchor_idx, i)
        left = max(0, right - 1)
        if right >= anchors:
            left = anchors - 1
            right = anchors - 1
        i0 = anchor_idx[left]
        i1 = anchor_idx[right]
        if i1 == i0:
            vi = vals[left]
        else:
            w = (i - i0) / max(1, (i1 - i0))
            vi = (1 - w) * vals[left] + w * vals[right]
        v[i] = max(1e-6, vi)
    return v, anchor_idx.astype(np.int64), logv


def _wave_blocks(num_orders: int, blocks: int, rng: np.random.Generator,
                 v0: float, amp_min: float, amp_max: float) -> np.ndarray:
    v = np.zeros(num_orders, dtype=np.float64)
    v[0] = v0
    if num_orders >= 2:
        v[1] = min(v0, v0 * max(0.0, min(1.0, rng.normal(0.95, 0.03))))
    if num_orders <= 2:
        return v
    # block-wise base growth and amplitude
    length = num_orders - 2
    blk_len = max(1, length // max(1, blocks))
    i = 2
    prev = v[1]
    while i < num_orders:
        this_len = min(blk_len, num_orders - i)
        g_base = max(1.01, min(1.20, rng.normal(1.06, 0.03)))
        amp = max(0.0, min(1.0, rng.uniform(amp_min, amp_max)))
        phase = rng.uniform(0.0, np.pi)
        for k in range(this_len):
            t = k / max(1, this_len - 1)
            wave = np.sin(2 * np.pi * t + phase)
            g = g_base * (1.0 + amp * wave)
            g = max(1.01, min(1.30, g))
            prev = prev * g
            v[i] = prev
            i += 1
            if i >= num_orders:
                break
    return v


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
    # New shape-enforcement parameters
    first_volume_target: float = 0.01,
    first_indent_target: float = 0.0,
    k_front: int = 3,
    front_cap: float = 5.0,
    g_min: float = 1.01,
    g_max: float = 1.20,
    # New penalty weights
    w_fixed: float = 3.0,
    w_band: float = 2.0,
    w_front: float = 3.0,
    w_tv: float = 1.0,
    w_sec: float = 3.0,
    w_wave: float = 1.0,
    # Generation mode knobs
    wave_mode: str = "anchors",  # [anchors, blocks]
    anchors: int = 6,
    blocks: int = 3,
    wave_amp_min: float = 0.05,
    wave_amp_max: float = 0.30,
    # Post-band controls and isotonic smoothing
    g_min_post: float = 1.01,
    g_max_post: float = 1.30,
    isotonic_tail: bool = False,
    # Penalty weight preset
    penalty_preset: Optional[str] = None,
    # New hard constraints
    m2_min: float = 0.10,
    m2_max: float = 1.00,
    m_min: float = 0.05,
    m_max: float = 1.00,
    firstK_min: float = 1.0,
    strict_inc_eps: float = 1e-5,
    # New soft penalties
    target_std: float = 0.10,
    w_varm: float = 2.0,
    w_blocks: float = 1.0,
    use_entropy: bool = False,
    entropy_target: float = 1.0,
    **kwargs
) -> Dict[str, Any]:
    """
    DCA/Martingale evaluation function exactly as defined in README.
    
    Returns complete dict with all required outputs, always JSON-serializable.
    Never throws exceptions - returns error state in dict if needed.
    """
    start_time = time.time()
    
    # Log evaluation call only if sampling allows it
    if should_log_eval():
        logger.debug(
            "Starting evaluation",
            extra={
                "event": "EVAL_CALL",
                "overlap": overlap_pct,
                "orders": num_orders,
                "wave_pattern": wave_pattern,
                "alpha": alpha,
                "beta": beta,
                "gamma": gamma,
                "lambda_penalty": lambda_penalty
            }
        )
    
    try:
        # Generate random parameters
        rng = np.random.default_rng(seed)
        raw_indents = rng.normal(0.0, 1.0, size=num_orders)
        raw_volumes = rng.normal(0.0, 1.0, size=num_orders)

        # 1. Build indent_pct with monotonic steps
        steps = _softplus(raw_indents)
        steps = np.maximum(steps, min_indent_step / 100.0)
        steps_sum = np.sum(steps)
        if steps_sum <= 1e-12:
            steps = np.full(num_orders, overlap_pct / (100.0 * num_orders))
        else:
            steps = steps * (overlap_pct / 100.0) / steps_sum
        indent_cumulative = np.concatenate([[0.0], np.cumsum(steps)]) * 100.0
        indent_pct_initial = indent_cumulative[1:].tolist()

        # 2. Volume generation (anchors/blocks) with v0 fixed and v1<=v0
        vol_sample = np.zeros(num_orders, dtype=np.float64)
        anchor_points_norm = None
        anchor_logv = None
        if wave_mode == "blocks":
            vol_sample = _wave_blocks(num_orders, blocks, rng, first_volume_target, wave_amp_min, wave_amp_max)
        else:
            vol_sample, anchor_idx, logv = _interp_from_anchors(num_orders, max(4, min(anchors, 8)), rng, first_volume_target, 1.0)
            # Normalize anchor indices to t-space [0,1]
            if num_orders > 1:
                anchor_points_norm = anchor_idx.astype(np.float64) / float(num_orders - 1)
            else:
                anchor_points_norm = anchor_idx.astype(np.float64)
            anchor_logv = logv

        # Ensure minimal pre-band (clip ratios for i>=2) and build monotone-biased raw chain
        for i in range(2, num_orders):
            denom = max(vol_sample[i-1], 1e-12)
            gi = vol_sample[i] / denom
            gi = min(g_max, max(g_min, gi))
            vol_sample[i] = vol_sample[i-1] * gi

        # Normalize keeping v0 fixed, tail scaled with single factor
        if num_orders > 1:
            rest_sum = float(np.sum(vol_sample[1:]))
            if rest_sum > 1e-12:
                scale_rest = (100.0 - vol_sample[0]) / rest_sum
                vol_sample[1:] *= scale_rest
            else:
                vol_sample[1:] = (100.0 - vol_sample[0]) / (num_orders - 1)
        else:
            vol_sample[0] = 100.0
        volume_pct_initial = vol_sample.copy()

        # Optional preset overrides for penalty weights
        if penalty_preset:
            pp = (penalty_preset or "").lower()
            if pp == "explore":
                w_fixed_local = 2.0; w_sec_local = 2.0; w_band_local = 1.5; w_front_local = 2.0; w_tv_local = 0.5; w_wave_local = 0.5
            elif pp == "robust":
                w_fixed_local = 3.0; w_sec_local = 3.0; w_band_local = 2.5; w_front_local = 3.0; w_tv_local = 1.5; w_wave_local = 1.0
            elif pp == "tight":
                w_fixed_local = 4.0; w_sec_local = 4.0; w_band_local = 3.0; w_front_local = 4.0; w_tv_local = 2.0; w_wave_local = 1.5
            else:
                w_fixed_local = w_fixed; w_sec_local = w_sec; w_band_local = w_band; w_front_local = w_front; w_tv_local = w_tv; w_wave_local = w_wave
        else:
            w_fixed_local = w_fixed; w_sec_local = w_sec; w_band_local = w_band; w_front_local = w_front; w_tv_local = w_tv; w_wave_local = w_wave
        (indent_pct,
         volume_pct,
         martingale_pct,
         needpct,
         order_prices,
         price_step_pct,
         repair_diag) = enforce_schedule_shape_fixed(
            indent_pct_initial,
            volume_pct_initial.tolist(),
            base_price,
            first_volume_target,
            first_indent_target,
            k_front,
            front_cap,
            g_min,
            g_max,
            g_min_post,
            g_max_post,
            isotonic_tail,
            # New parameters
            m2_min,
            m2_max,
            m_min,
            m_max,
            firstK_min,
            strict_inc_eps,
        )
        
        # Calculate core metrics from repaired arrays
        max_need = float(np.max(needpct)) if len(needpct) > 0 else 0.0
        var_need = float(np.var(needpct)) if len(needpct) > 0 else 0.0
        
        # Tail calculation (last 20% of orders) from repaired volumes
        tail_start = max(0, int(0.8 * num_orders))
        tail = float(np.sum(np.asarray(volume_pct)[tail_start:]) / 100.0)
        
        # Build schedule dict with all required fields (repaired)
        schedule = {
            "indent_pct": _ensure_json_serializable(indent_pct),
            "volume_pct": _ensure_json_serializable(volume_pct),
            "martingale_pct": _ensure_json_serializable(martingale_pct),
            "needpct": _ensure_json_serializable(needpct),
            "order_prices": _ensure_json_serializable(order_prices),
            "price_step_pct": _ensure_json_serializable(price_step_pct),
        }
        
        # Calculate sanity checks on repaired arrays
        calculated_max_need = float(np.max(needpct)) if len(needpct) > 0 else 0.0
        collapse_indents = False
        if len(indent_pct) > 1:
            diff_ind = np.diff(np.asarray([0.0] + indent_pct))
            collapse_indents = bool(np.any(diff_ind < min_indent_step))
        sanity = {
            "max_need_mismatch": bool(abs(max_need - calculated_max_need) > 1e-6),
            "collapse_indents": collapse_indents,
            "tail_overflow": bool((len(volume_pct) > 0) and (volume_pct[-1] > tail_cap * 100.0)),
        }
        
        # Diagnostics (repaired)
        volume_pct_np = np.asarray(volume_pct, dtype=np.float64)
        diagnostics = {
            "wci": float(_weight_center_index(volume_pct_np)),
            "sign_flips": int(_count_sign_flips(np.asarray(needpct, dtype=np.float64))),
            "gini": float(_gini_coefficient(volume_pct_np / 100.0)),
            "entropy": float(_entropy_normalized(volume_pct_np)),
            # Repair diagnostics for batch aggregation
            "repair_clipped_frac": float(repair_diag.get("clipped_frac", 0.0)),
            "repair_front_excess_before": float(repair_diag.get("front_excess_before", 0.0)),
            "repair_front_excess_after": float(repair_diag.get("front_excess_after", 0.0)),
            "repair_tv_before": float(repair_diag.get("tv_before", 0.0)),
            "repair_tv_after": float(repair_diag.get("tv_after", 0.0)),
            # New diagnostics
            "first3_sum": float(repair_diag.get("first3_sum", 0.0)),
            "m2": float(repair_diag.get("m2", 0.0)),
            "std_m": float(repair_diag.get("std_m", 0.0)),
            "sign_changes": float(repair_diag.get("sign_changes", 0.0)),
            "clips_hi_count": int(repair_diag.get("clip_hi_post", 0)),
            "clips_lo_count": int(repair_diag.get("clip_lo_post", 0)),
            "iter_fix_loops": int(repair_diag.get("iter_fix_loops", 0)),
            # Generation/repair flags
            "wave_mode": wave_mode,
            "anchors": int(anchors) if wave_mode == "anchors" else None,
            "blocks": int(blocks) if wave_mode == "blocks" else None,
            "isotonic_applied": bool(isotonic_tail),
        }
        if anchor_points_norm is not None:
            diagnostics["anchor_points"] = anchor_points_norm.tolist()
        if anchor_logv is not None:
            diagnostics["anchor_logv"] = _ensure_json_serializable(anchor_logv)
        
        # Penalties
        penalties: Dict[str, float] = {}
        penalties["P_gini"] = float(_gini_coefficient(volume_pct_np / 100.0))
        penalties["P_entropy"] = float(max(0.0, 1.0 - _entropy_normalized(volume_pct_np)))
        # Indent monotonicity penalty (using repaired indent cumulative)
        indent_cumulative_repaired = np.asarray([0.0] + indent_pct, dtype=np.float64)
        monotone_violations = float(np.sum(np.maximum(0, -np.diff(indent_cumulative_repaired[1:])))) if len(indent_pct) > 1 else 0.0
        penalties["P_monotone"] = float(monotone_violations)
        # Price step smoothness
        if len(price_step_pct) > 1:
            penalties["P_smooth"] = float(np.var(np.asarray(price_step_pct)) / 100.0)
        else:
            penalties["P_smooth"] = 0.0
        # Tailcap penalty
        if len(volume_pct) > 0 and volume_pct[-1] > tail_cap * 100.0:
            penalties["P_tailcap"] = float((volume_pct[-1] - tail_cap * 100.0) / (tail_cap * 100.0))
        else:
            penalties["P_tailcap"] = 0.0
        penalties["P_need_mismatch"] = float(1.0 if sanity["max_need_mismatch"] else 0.0)
        
        # Wave pattern penalty/reward (on repaired martingale_pct)
        wave_score = 0.0
        if wave_pattern and len(martingale_pct) > 2:
            for i in range(2, len(martingale_pct)):
                prev_mart = martingale_pct[i-1]
                curr_mart = martingale_pct[i]
                if prev_mart >= wave_strong_threshold and curr_mart <= wave_weak_threshold:
                    wave_score += 0.1
                elif prev_mart <= wave_weak_threshold and curr_mart >= wave_strong_threshold:
                    wave_score += 0.1
                if prev_mart >= wave_strong_threshold and curr_mart >= wave_strong_threshold:
                    wave_score -= 0.2
                elif prev_mart <= wave_weak_threshold and curr_mart <= wave_weak_threshold:
                    wave_score -= 0.2
        penalties["P_wave"] = float(max(0.0, -wave_score))
        
        # New shape penalties after repair
        shape_pens = compute_shape_penalties(
            np.asarray(volume_pct, dtype=np.float64),
            np.asarray(indent_pct, dtype=np.float64),
            first_volume_target,
            first_indent_target,
            g_min,
            g_max,
            k_front,
            front_cap,
            np.asarray(martingale_pct, dtype=np.float64),
            target_std,
            use_entropy,
            entropy_target,
        )
        penalties.update(shape_pens)
        
        # Weighted sum of shape penalties
        shape_penalty_sum = (
            w_fixed_local * shape_pens["penalty_first_fixed"] +
            w_sec_local * shape_pens.get("penalty_second_leq", 0.0) +
            w_band_local * shape_pens["penalty_g_band"] +
            w_front_local * shape_pens["penalty_frontload"] +
            w_tv_local * shape_pens["penalty_tv_vol"] +
            w_wave_local * shape_pens.get("penalty_wave", 0.0) +
            w_varm * shape_pens.get("penalty_uniform", 0.0) +
            w_blocks * shape_pens.get("penalty_flat_blocks", 0.0)
        )
        
        # Add entropy penalty if enabled
        if use_entropy and "penalty_low_entropy" in shape_pens:
            shape_penalty_sum += w_varm * shape_pens["penalty_low_entropy"]
        
        # Log repair diagnostics and penalties (INFO)
        logger.info(
            f"REPAIR v0={first_volume_target:.3f} v1={volume_pct[1] if len(volume_pct) > 1 else 0:.3f} "
            f"m2={repair_diag.get('m2', 0):+.1%} first3={repair_diag.get('first3_sum', 0):.2%} "
            f"clips_hi={repair_diag.get('clip_hi_post', 0)} clips_lo={repair_diag.get('clip_lo_post', 0)} "
            f"iters={repair_diag.get('iter_fix_loops', 0)}",
            extra={
                "event": "REPAIR",
                "clipped_frac": repair_diag.get("clipped_frac", 0.0),
                "front_excess_before": repair_diag.get("front_excess_before", 0.0),
                "front_excess_after": repair_diag.get("front_excess_after", 0.0),
                "tv_before": repair_diag.get("tv_before", 0.0),
                "tv_after": repair_diag.get("tv_after", 0.0),
                "first3_sum": repair_diag.get("first3_sum", 0.0),
                "g2": repair_diag.get("g2", 0.0),
                "clip_hi_post": repair_diag.get("clip_hi_post", 0),
                "clip_lo_post": repair_diag.get("clip_lo_post", 0),
                "m2": repair_diag.get("m2", 0.0),
                "std_m": repair_diag.get("std_m", 0.0),
                "iter_fix_loops": repair_diag.get("iter_fix_loops", 0),
            },
        )
        logger.info(
            f"PEN m_uniform={shape_pens.get('penalty_uniform', 0.0):.2f} "
            f"m_blocks={shape_pens.get('penalty_flat_blocks', 0.0):.2f} "
            f"total={shape_penalty_sum:.4f}",
            extra={
                "event": "PENALTIES",
                "fixed": shape_pens["penalty_first_fixed"],
                "second_leq": shape_pens.get("penalty_second_leq", 0.0),
                "gband": shape_pens["penalty_g_band"],
                "front": shape_pens["penalty_frontload"],
                "tv": shape_pens["penalty_tv_vol"],
                "wave": shape_pens.get("penalty_wave", 0.0),
                "uniform": shape_pens.get("penalty_uniform", 0.0),
                "flat_blocks": shape_pens.get("penalty_flat_blocks", 0.0),
                "low_entropy": shape_pens.get("penalty_low_entropy", 0.0),
                "sum": shape_penalty_sum,
            },
        )
        
        # Final score
        penalty_sum = sum(penalties.values()) + shape_penalty_sum
        score = alpha * max_need + beta * var_need + gamma * tail + lambda_penalty * penalty_sum
        
        # Log successful evaluation only if sampling allows it
        duration_ms = (time.time() - start_time) * 1000
        if should_log_eval():
            logger.debug(
                "Evaluation completed successfully",
                extra={
                    "event": "EVAL_RETURN",
                    "score": float(score),
                    "max_need": float(max_need),
                    "var_need": float(var_need),
                    "tail": float(tail),
                    "duration_ms": duration_ms,
                    "penalty_sum": penalty_sum,
                    "sanity_violations": sum(1 for v in sanity.values() if v),
                    "overlap": overlap_pct,
                    "orders": num_orders,
                },
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
            "penalties": penalties,
        }
        
    except Exception as e:
        # Log evaluation error only if sampling allows it
        duration_ms = (time.time() - start_time) * 1000
        if should_log_eval():
            logger.debug(
                f"Evaluation failed: {str(e)}",
                extra={
                    "event": "EVAL_ERROR",
                    "error": str(e),
                    "duration_ms": duration_ms,
                    "overlap": overlap_pct,
                    "orders": num_orders
                }
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