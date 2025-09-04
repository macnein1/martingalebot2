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
import traceback

from martingale_lab.utils.logging import get_eval_logger, should_log_eval
from martingale_lab.core.constraints import enforce_schedule_shape_fixed
from martingale_lab.core.penalties import compute_shape_penalties
from martingale_lab.core.parameter_forwarding import filter_kwargs_for_function
from martingale_lab.core.schedule_normalizer import normalize_schedule_to_2dp, is_schedule_normalized

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
    softmax_temp: float = 1.4,
    # New shape-enforcement parameters
    first_volume_target: float = 1.0,  # Changed from 0.01 to 1.0 for reasonable m2
    first_indent_target: float = 0.0,
    # Legacy parameters (deprecated, kept for compatibility)
    k_front: int = 3,  # DEPRECATED: Use Q1/Q4 mass control instead
    front_cap: float = 5.0,  # DEPRECATED: Use q1_cap instead
    g_min: float = 1.01,
    g_max: float = 1.20,
    # New penalty weights
    w_fixed: float = 3.0,
    w_band: float = 2.0,
    w_front: float = 3.0,
    w_tv: float = 3.5,
    w_sec: float = 3.0,
    w_wave: float = 1.0,
    # Diversity/sensitivity/template penalties
    w_sens: float = 1.0,
    sens_min: float = 0.25,
    w_template: float = 0.8,
    template_close: float = 0.6,
    template_mode: str = "doubling",  # [doubling, linear, custom]
    template_custom: Optional[List[float]] = None,
    # Generation mode knobs
    wave_mode: str = "anchors",  # [anchors, blocks]
    anchors: int = 9,
    blocks: int = 3,
    wave_amp_min: float = 0.05,
    wave_amp_max: float = 0.30,
    # Penalty weight preset
    penalty_preset: Optional[str] = None,
    # New hard constraints
    m2_min: float = 0.10,
    m2_max: float = 1.00,
    m_min: float = 0.05,
    m_max: float = 1.00,
    firstK_min: float = 1.0,
    strict_inc_eps: float = 1e-6,
    # New HC parameters
    second_upper_c2: float = 2.0,
    m_head: float = 0.40,
    m_tail: float = 0.20,
    tau_scale: float = 1/3,
    slope_cap: float = 0.25,
    q1_cap: float = 22.0,
    tail_floor: float = 32.0,
    # Head budget parameters
    head_budget_pct: float = 2.0,
    use_head_budget: bool = False,
    use_hc0_bootstrap: bool = True,
    # New soft penalties
    target_std: float = 0.10,
    w_varm: float = 2.0,
    w_blocks: float = 1.0,
    # New SP penalty weights
    w_second: float = 3.0,
    w_plateau: float = 2.0,
    w_front_share: float = 2.0,
    w_tailweak: float = 2.0,
    w_slope: float = 1.0,
    w_wave_shape: float = 1.2,
    # Adaptive parameters
    use_adaptive: bool = False,
    strategy_type: str = "balanced",
    # Smart initial generation
    use_smart_init: bool = False,
    history_db: Optional[str] = None,
    # Schedule normalization parameters
    post_round_2dp: bool = False,
    post_round_strategy: str = "tail-first",
    post_round_m2_tolerance: float = 0.05,
    post_round_keep_v1_band: bool = True,
    # Post-normalization optional smoothing
    post_norm_smoothing: bool = False,
    smoothing_alpha: float = 0.15,
    **kwargs
) -> Dict[str, Any]:
    """
    DCA/Martingale evaluation function exactly as defined in README.
    
    Returns complete dict with all required outputs, always JSON-serializable.
    Never throws exceptions - returns error state in dict if needed.
    """
    start_time = time.time()
    
    # Import adaptive module
    if use_adaptive:
        from martingale_lab.core.adaptive import get_adaptive_parameters
        
        # Get adaptive parameters
        adaptive_params = get_adaptive_parameters(
            num_orders, 
            overlap_pct,
            strategy_type
        )
        
        # Override with adaptive values
        m2_min = adaptive_params.get('m2_min', m2_min)
        m2_max = adaptive_params.get('m2_max', m2_max)
        m_min = adaptive_params.get('m_min', m_min)
        m_max = adaptive_params.get('m_max', m_max)
        m_head = adaptive_params.get('m_head', m_head)
        m_tail = adaptive_params.get('m_tail', m_tail)
        tau_scale = adaptive_params.get('tau_scale', tau_scale)
        slope_cap = adaptive_params.get('slope_cap', slope_cap)
        q1_cap = adaptive_params.get('q1_cap', q1_cap)
        tail_floor = adaptive_params.get('tail_floor', tail_floor)
        
        # Adaptive weights
        w_front = adaptive_params.get('w_front', w_front)
        w_tailweak = adaptive_params.get('w_tailweak', w_tailweak)
        w_slope = adaptive_params.get('w_slope', w_slope)
        w_plateau = adaptive_params.get('w_plateau', w_plateau)
        
        # Exit-ease weight adjustment
        exit_ease_weight = adaptive_params.get('exit_ease_weight', 100.0)
    else:
        exit_ease_weight = 100.0
    
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
        # Generate random parameters for indents
        rng = np.random.default_rng(seed)
        raw_indents = rng.normal(0.0, 1.0, size=num_orders)
        
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
        
        # 2. Generate initial volumes
        anchor_points_norm = None
        anchor_logv = None
        
        if use_smart_init:
            # Try smart initial generation
            try:
                from martingale_lab.core.smart_init import get_smart_initial_strategy
                
                logger.debug(f"Using smart initial generation (history_db={history_db})")
                volume_pct_array = get_smart_initial_strategy(
                    num_orders,
                    overlap_pct,
                    seed=seed,
                    history_db=history_db
                )
                volume_pct_initial = volume_pct_array.tolist()
                
                logger.debug(f"Smart init generated: v[:5]={volume_pct_initial[:5] if len(volume_pct_initial) >= 5 else volume_pct_initial}")
                
            except Exception as e:
                logger.warning(f"Smart init failed, falling back to standard: {e}")
                use_smart_init = False
        
        if not use_smart_init:
            # Standard initial generation (current method)
            if wave_mode == "blocks":
                vol_sample = _wave_blocks(num_orders, blocks, rng, first_volume_target, wave_amp_min, wave_amp_max)
            else:
                vol_sample, anchor_idx, logv = _interp_from_anchors(num_orders, max(4, min(anchors, 10)), rng, first_volume_target, 1.0)
                # Normalize anchor indices to t-space [0,1]
                if num_orders > 1:
                    anchor_points_norm = anchor_idx.astype(np.float64) / float(num_orders - 1)
                else:
                    anchor_points_norm = anchor_idx.astype(np.float64)
                anchor_logv = logv

            # Ensure minimal pre-band (clip ratios for i>=2) and build monotone-biased raw chain
            for i in range(2, num_orders):
                if vol_sample[i] < vol_sample[i-1] * 1.01:
                    vol_sample[i] = vol_sample[i-1] * 1.01

            # Softmax normalization to get volume_pct
            vol_sample = _softplus(vol_sample)
            vol_sample = vol_sample / np.sum(vol_sample) * 100.0
            volume_pct_initial = vol_sample.tolist()
            
            # Check if initial volumes are reasonable (fallback mechanism)
            if len(volume_pct_initial) > 2:
                first_two_sum = volume_pct_initial[0] + volume_pct_initial[1]
                min_vol = min(volume_pct_initial)
                max_vol = max(volume_pct_initial)
                
                if first_two_sum > 30.0 or first_two_sum < 0.5 or max_vol / (min_vol + 1e-10) > 100:
                    logger.debug(f"Initial volumes unreasonable, using geometric fallback")
                    volume_pct_initial = []
                    
                    v0 = 1.0
                    growth = 1.115
                    
                    for i in range(num_orders):
                        if i == 0:
                            volume_pct_initial.append(v0)
                        elif i == 1:
                            volume_pct_initial.append(v0 * 1.10)
                        else:
                            base_growth = growth
                            if i < 5:
                                base_growth = 1.10 + (i - 2) * 0.02
                            elif i > num_orders - 5:
                                base_growth = growth * 1.05
                            volume_pct_initial.append(volume_pct_initial[-1] * base_growth)
                    
                    total = sum(volume_pct_initial)
                    volume_pct_initial = [v / total * 100.0 for v in volume_pct_initial]

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
        # Prepare all constraint parameters
        constraint_params = {
            "indent_pct": indent_pct_initial,
            "volume_pct": volume_pct_initial,
            "base_price": base_price,
            "first_volume_target": first_volume_target,
            "first_indent_target": first_indent_target,
            "k_front": k_front,
            "front_cap": front_cap,
            "g_min": g_min,
            "g_max": g_max,
            "g_min_post": 1.01,  # Default value
            "g_max_post": 1.30,  # Default value
            "isotonic_tail": True,  # Default value
            "m2_min": m2_min,
            "m2_max": m2_max,
            "m_min": m_min,
            "m_max": m_max,
            "firstK_min": firstK_min,
            "eps_inc": strict_inc_eps,
            "slope_cap": slope_cap,
        }
        
        # Filter parameters to only those accepted by enforce_schedule_shape_fixed
        safe_params = filter_kwargs_for_function(enforce_schedule_shape_fixed, constraint_params)
        
        (indent_pct,
         volume_pct,
         martingale_pct,
         needpct,
         order_prices,
         price_step_pct,
         repair_diag) = enforce_schedule_shape_fixed(**safe_params)
        
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
        
        # Normalization-always-on: rescale to sum exactly 100.0 for scoring
        # Preserve raw arrays as debug fields
        volume_pct_raw = list(volume_pct)
        indent_pct_raw = list(indent_pct)
        vol_sum = float(np.sum(np.asarray(volume_pct, dtype=np.float64)))
        if vol_sum > 1e-12:
            scale = 100.0 / vol_sum
            volume_pct = [float(v * scale) for v in volume_pct]
        else:
            volume_pct = [float(100.0 / max(1, len(volume_pct)))] * len(volume_pct)

        # Assert sum ~ 100 and epsilon-fix the last element if needed
        s = float(np.sum(np.asarray(volume_pct, dtype=np.float64)))
        diff = 100.0 - s
        if abs(diff) > 1e-9 and len(volume_pct) > 0:
            volume_pct[-1] = float(volume_pct[-1] + diff)
            # Guard against negative due to numerical issues
            if volume_pct[-1] < 0:
                volume_pct[-1] = 0.0
                # Rebalance tiny deficit to previous element if exists
                if len(volume_pct) > 1:
                    volume_pct[-2] = float(max(0.0, volume_pct[-2] + diff))

        # Optional light smoothing (then re-project to 100 and enforce monotonic)
        if post_norm_smoothing and len(volume_pct) >= 3:
            v = np.asarray(volume_pct, dtype=np.float64)
            alpha = max(0.0, min(1.0, smoothing_alpha))
            # 3-point smoothing kernel [0.25, 0.5, 0.25] blended by alpha
            vs = v.copy()
            vs[1:-1] = (1 - alpha) * v[1:-1] + alpha * (0.25 * v[:-2] + 0.5 * v[1:-1] + 0.25 * v[2:])
            # Re-project to sum 100
            total = float(np.sum(vs))
            if total > 1e-12:
                vs = vs * (100.0 / total)
            # Enforce non-decreasing very softly
            for i in range(1, len(vs)):
                if vs[i] < vs[i-1]:
                    vs[i] = vs[i-1]
            # Re-project again to sum 100
            total = float(np.sum(vs))
            if total > 1e-12:
                vs = vs * (100.0 / total)
            volume_pct = [float(x) for x in vs.tolist()]

        # Recompute derived fields AFTER smoothing
        martingale_pct = []
        needpct = []
        cumsum = 0.0
        base_price_local = order_prices[0] if len(order_prices) > 0 else 1.0
        for i, v in enumerate(volume_pct):
            if i == 0:
                martingale_pct.append(0.0)
            else:
                cumsum += volume_pct[i-1]
                martingale_pct.append(v / cumsum * 100.0 if cumsum > 0 else 0.0)
            if i < len(order_prices):
                price_ratio = (order_prices[i] / base_price_local) if base_price_local else 1.0
                needpct.append(v * price_ratio)
            else:
                needpct.append(v)

        # Recompute derived fields after rescale
        martingale_pct = []
        needpct = []
        cumsum = 0.0
        base_price_local = order_prices[0] if len(order_prices) > 0 else 1.0
        for i, v in enumerate(volume_pct):
            if i == 0:
                martingale_pct.append(0.0)
            else:
                cumsum += volume_pct[i-1]
                martingale_pct.append(v / cumsum * 100.0 if cumsum > 0 else 0.0)
            if i < len(order_prices):
                price_ratio = (order_prices[i] / base_price_local) if base_price_local else 1.0
                needpct.append(v * price_ratio)
            else:
                needpct.append(v)

        # Attach normalized and raw arrays to schedule for reporting
        schedule["volume_pct_norm"] = [float(v) for v in volume_pct]
        schedule["indent_pct_norm"] = [float(i) for i in indent_pct]
        schedule["volume_pct_raw"] = [float(v) for v in volume_pct_raw]
        schedule["indent_pct_raw"] = [float(i) for i in indent_pct_raw]
        schedule["volume_pct_norm_2dp"] = [round(float(v), 2) for v in schedule["volume_pct_norm"]]
        schedule["volume_pct_raw_2dp"] = [round(float(v), 2) for v in schedule["volume_pct_raw"]]
        schedule["indent_pct_2dp"] = [round(float(i), 2) for i in schedule.get("indent_pct", [])]

        # Optionally produce 2dp rounded schedule for consumers
        if post_round_2dp:
            rounded = normalize_schedule_to_2dp(
                {
                    "indent_pct": schedule.get("indent_pct", []),
                    "volume_pct": schedule["volume_pct_norm"],
                    "martingale_pct": martingale_pct,
                    "needpct": needpct,
                    "order_prices": order_prices,
                    "price_step_pct": price_step_pct,
                },
                post_round_strategy=post_round_strategy,
                post_round_m2_tolerance=post_round_m2_tolerance,
                post_round_keep_v1_band=post_round_keep_v1_band,
                strict_monotonicity=True,
                preserve_quartiles=True,
                verbose=should_log_eval()
            )
            volume_pct = rounded["volume_pct"]
            indent_pct = rounded.get("indent_pct", indent_pct)
            martingale_pct = rounded.get("martingale_pct", martingale_pct)
            needpct = rounded.get("needpct", needpct)
            schedule.update(rounded)
        else:
            schedule["volume_pct"] = [float(v) for v in volume_pct]
            schedule["indent_pct"] = [float(i) for i in indent_pct]
        
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
        
        # Calculate repair diagnostics
        volume_pct_np = np.asarray(volume_pct, dtype=np.float64)
        
        # Import at the beginning to avoid UnboundLocalError
        from martingale_lab.core.repair import compute_m_from_v
        
        m = compute_m_from_v(volume_pct_np)
        
        # Import exit-ease metrics
        from martingale_lab.core.metrics import compute_exit_ease_metrics
        ee_metrics = compute_exit_ease_metrics(needpct, volume_pct)
        
        # Add micro pattern detection
        from martingale_lab.core.pattern_detection import (
            compute_pattern_penalties,
            analyze_micro_patterns
        )
        
        # Compute pattern penalties
        pattern_penalty = compute_pattern_penalties(
            volume_pct,
            martingale_pct,
            w_plateau=w_plateau,
            w_zigzag=1.0,  # Lower weight for zigzag
            w_cliff=15.0,   # High penalty for cliffs
            w_stagnation=2.0,
            w_acceleration=3.0
        )
        
        # Analyze patterns for diagnostics
        pattern_analysis = analyze_micro_patterns(volume_pct, martingale_pct)
        
        # Add portfolio metrics
        from martingale_lab.core.portfolio_metrics import calculate_portfolio_metrics
        portfolio_metrics = calculate_portfolio_metrics(volume_pct)
        
        # Add to diagnostics
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
            # New diagnostics from HC pipeline
            "first3_sum": float(repair_diag.get("first3_sum", 0.0)),
            "v0": float(repair_diag.get("v0", 0.0)),
            "v1": float(repair_diag.get("v1", 0.0)),
            "m2": float(repair_diag.get("m2", 0.0)),
            "std_m": float(repair_diag.get("std_m", 0.0)),
            "v1_band_applied": bool(repair_diag.get("v1_band_applied", False)),
            "m2_clip_applied": bool(repair_diag.get("m2_clip_applied", False)),
            "decaying_clips": int(repair_diag.get("decaying_clips_count", 0)),
            "slope_clips": int(repair_diag.get("slope_clips_count", 0)),
            "q1_share": float(repair_diag.get("q1_share", 0.0)),
            "q4_share": float(repair_diag.get("q4_share", 0.0)),
            "plateau_max_run": int(repair_diag.get("plateau_max_run", 0)),
            "turn_count": int(repair_diag.get("turn_count", 0)),
            # Exit-ease metrics
            "ee_harmonic": float(ee_metrics.get("ee_harmonic", 0.0)),
            "ee_tail_weighted": float(ee_metrics.get("ee_tail_weighted", 0.0)),
            "ee_front_tail_ratio": float(ee_metrics.get("ee_front_tail_ratio", 0.0)),
            "ee_balance_penalty": float(ee_metrics.get("ee_balance_penalty", 0.0)),
            # Pattern analysis
            "pattern_quality_score": float(pattern_analysis.get("pattern_quality_score", 0.0)),
            "pattern_plateaus": int(pattern_analysis.get("plateau_count", 0)),
            "pattern_zigzag": int(pattern_analysis.get("zigzag_count", 0)),
            "pattern_cliffs": int(pattern_analysis.get("cliff_count", 0)),
            "pattern_stagnation": int(pattern_analysis.get("stagnation_zones", 0)),
            "pattern_max_acceleration": float(pattern_analysis.get("max_acceleration", 0.0)),
            # Portfolio metrics
            "sortino_ratio": float(portfolio_metrics.get("sortino_ratio", 0.0)),
            "calmar_ratio": float(portfolio_metrics.get("calmar_ratio", 0.0)),
            "omega_ratio": float(portfolio_metrics.get("omega_ratio", 0.0)),
            "recovery_efficiency": float(portfolio_metrics.get("recovery_efficiency", 0.0)),
            "tail_risk_ratio": float(portfolio_metrics.get("tail_risk_ratio", 0.0)),
            "var_95": float(portfolio_metrics.get("var_95", 0.0)),
            "cvar_95": float(portfolio_metrics.get("cvar_95", 0.0)),
            "hc0_applied": bool(repair_diag.get("hc0_applied", False)),
            "head_budget_applied": bool(repair_diag.get("head_budget_applied", False)),
            # Generation/repair flags
            "wave_mode": wave_mode,
            "anchors": int(anchors) if wave_mode == "anchors" else None,
            "blocks": int(blocks) if wave_mode == "blocks" else None,
        }
        if anchor_points_norm is not None:
            diagnostics["anchor_points"] = anchor_points_norm.tolist()
        if anchor_logv is not None:
            diagnostics["anchor_logv"] = _ensure_json_serializable(anchor_logv)
        
        # Add exit-ease metrics to diagnostics
        diagnostics.update(ee_metrics)
        
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
        # (compute_m_from_v already imported above)
        
        # Compute shape-related penalties
        shape_pens = compute_shape_penalties(
            volume_pct_np,
            np.asarray(indent_pct, dtype=np.float64),
            k_front,
            front_cap,
            np.asarray(martingale_pct, dtype=np.float64),
            target_std,
            # New SP parameters
            1.10,  # v1_min_mult
            second_upper_c2,  # v1_max_mult
            0.02,  # plateau_tol
            3,  # plateau_max_len
            target_std,  # target_std_varm
            g_min,  # g_min (still used for wave calculation)
            m_head,  # wave_m_head
            m_tail,  # wave_m_tail
            tau_scale,  # wave_tau_scale
            0.0,  # wave_phase
            q1_cap,  # q1_cap
            tail_floor,  # tail_floor
            slope_cap * 0.8,  # slope_delta_soft
            slope_cap,  # slope_delta_cap
        )
        penalties.update(shape_pens)

        # Sensitivity penalty
        try:
            vol_arr = np.asarray(schedule.get("volume_pct_norm", volume_pct), dtype=np.float64)
            ind_arr = np.asarray(schedule.get("indent_pct", indent_pct), dtype=np.float64)
            if len(vol_arr) >= 2 and len(ind_arr) >= 2:
                dv = np.abs(np.diff(vol_arr))
                di = np.abs(np.diff(ind_arr))
                mean_di = float(np.mean(di)) if di.size > 0 else 0.0
                eps = 1e-6 * mean_di + 1e-9
                mask = di >= eps  # ignore pairs with tiny indent differences
                if np.any(mask):
                    S = float(np.mean(dv[mask] / (eps + di[mask])))
                else:
                    S = 0.0
            else:
                S = 0.0
        except Exception:
            S = 0.0
        penalty_sens = max(0.0, sens_min - S) * w_sens
        penalties["P_sensitivity"] = float(penalty_sens)

        # Template closeness penalty (geometric doubling template)
        try:
            n = len(volume_pct)
            if n > 0:
                if (template_mode or "").lower() == "linear":
                    g = np.linspace(1.0, float(n), n)
                elif (template_mode or "").lower() == "custom" and template_custom and len(template_custom) == n:
                    g = np.asarray(template_custom, dtype=np.float64)
                else:
                    g = np.array([2.0 ** i for i in range(n)], dtype=np.float64)
                g = g / np.sum(g) * 100.0
                v_norm = np.asarray(schedule.get("volume_pct_norm", volume_pct), dtype=np.float64)
                v_norm = v_norm / max(1e-12, np.sum(v_norm)) * 100.0
                D = float(np.sum(np.abs(v_norm - g)) / 100.0)
            else:
                D = 0.0
        except Exception:
            D = 0.0
        penalty_template = max(0.0, template_close - D) * w_template
        penalties["P_template"] = float(penalty_template)
        # Diagnostics for analysis
        diagnostics["sensitivity_S"] = float(S)
        diagnostics["template_D"] = float(D)
        
        # Weighted sum of shape penalties (including new SP penalties)
        shape_penalty_sum = (
            w_fixed_local * shape_pens["penalty_first_fixed"] +
            w_sec_local * shape_pens.get("penalty_second_leq", 0.0) +
            w_band_local * shape_pens["penalty_g_band"] +
            w_front_local * shape_pens["penalty_frontload"] +
            w_tv_local * shape_pens["penalty_tv_vol"] +
            w_wave_local * shape_pens.get("penalty_wave", 0.0) +
            w_varm * shape_pens.get("penalty_uniform", 0.0) +
            w_blocks * shape_pens.get("penalty_flat_blocks", 0.0) +
            # New SP penalties
            w_second * shape_pens.get("penalty_second_band", 0.0) +
            w_plateau * shape_pens.get("penalty_plateau", 0.0) +
            w_varm * shape_pens.get("penalty_varm", 0.0) +
            w_wave_shape * shape_pens.get("penalty_wave_shape", 0.0) +
            w_front_share * shape_pens.get("penalty_front_share", 0.0) +
            w_tailweak * shape_pens.get("penalty_tailweak", 0.0) +
            w_slope * shape_pens.get("penalty_slope", 0.0) +
            penalty_sens + penalty_template
        )
        
        # Add entropy penalty if enabled
        if "penalty_low_entropy" in shape_pens:
            shape_penalty_sum += w_varm * shape_pens["penalty_low_entropy"]
        
        # Log repair diagnostics and penalties (INFO) - updated format
        logger.info(
            f"REPAIR: v0={repair_diag.get('v0', 0):.5f} v1={repair_diag.get('v1', 0):.5f} "
            f"m2={repair_diag.get('m2', 0):.2%} first3={repair_diag.get('first3_sum', 0):.2%} "
            f"clips(decay={repair_diag.get('decaying_clips_count', 0)}/slope={repair_diag.get('slope_clips_count', 0)}) "
            f"q1={repair_diag.get('q1_share', 0):.1f}% q4={repair_diag.get('q4_share', 0):.1f}% "
            f"plateau_max={repair_diag.get('plateau_max_run', 0)} std_m={repair_diag.get('std_m', 0):.3f} "
            f"turns={repair_diag.get('turn_count', 0)}",
            extra={
                "event": "REPAIR",
                "v0": float(repair_diag.get("v0", 0.0)),
                "v1": float(repair_diag.get("v1", 0.0)),
                "m2": float(repair_diag.get("m2", 0.0)),
                "first3_sum": float(repair_diag.get("first3_sum", 0.0)),
                "decaying_clips": int(repair_diag.get("decaying_clips_count", 0)),
                "slope_clips": int(repair_diag.get("slope_clips_count", 0)),
                "q1_share": float(repair_diag.get("q1_share", 0.0)),
                "q4_share": float(repair_diag.get("q4_share", 0.0)),
                "plateau_max_run": int(repair_diag.get("plateau_max_run", 0)),
                "std_m": float(repair_diag.get("std_m", 0.0)),
                "turn_count": int(repair_diag.get("turn_count", 0)),
                "hc0_applied": bool(repair_diag.get("hc0_applied", False)),
                "head_budget_applied": bool(repair_diag.get("head_budget_applied", False)),
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
        
        # Add pattern penalty
        total_pattern_penalty = pattern_penalty
        
        # Calculate final score with exit-ease bonus
        # Higher exit-ease (easier exits) should reduce (improve) the score
        exit_ease_bonus = max(0.0, ee_metrics['ee_tail_weighted'] - 5.0) * exit_ease_weight  # Adaptive weight
        exit_ease_penalty = ee_metrics['ee_balance_penalty'] * 10.0  # Penalty for unbalanced blocks
        
        score = (
            alpha * max_need 
            + beta * var_need 
            + gamma * (1.0 - tail)
            + lambda_penalty * penalty_sum
            + shape_penalty_sum
            + total_pattern_penalty  # Add pattern penalty
            - exit_ease_bonus  # Subtract bonus (lower score is better)
            + exit_ease_penalty  # Add penalty for imbalance
        )
        
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
        
        # Always log exceptions for debugging
        logger.error(
            f"Evaluation failed with exception: {str(e)}",
            extra={
                "event": "EVAL_ERROR",
                "error": str(e),
                "error_type": type(e).__name__,
                "traceback": traceback.format_exc(),
                "duration_ms": duration_ms,
                "overlap": overlap_pct,
                "orders": num_orders,
                "params": {
                    "base_price": base_price,
                    "first_volume_target": first_volume_target,
                    "first_indent_target": first_indent_target,
                    "m2_min": m2_min,
                    "m2_max": m2_max,
                    "m_min": m_min
                }
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
                "entropy": 0.0,
                "error_type": type(e).__name__,
                "error_msg": str(e)
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