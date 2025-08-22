"""
Evaluation contract helpers for martingale optimization.
Implements schedule projection, risk metrics, penalties, and scoring.
"""
from __future__ import annotations

import json
import math
import hashlib
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# -----------------------------
# Config dataclass (weights and knobs)
# -----------------------------
@dataclass
class EvalKnobs:
    alpha: float = 0.4
    beta: float = 0.3
    gamma: float = 0.3

    use_softplus_indents: bool = True
    use_softmax_volumes: bool = True

    penalty_gini: float = 0.0
    penalty_entropy: float = 0.0
    penalty_monotone: float = 0.0
    penalty_step_smooth: float = 0.0

    limit_tail: Optional[float] = None
    last_order_cap: Optional[float] = None  # max allowed last order volume pct (0-100)

    random_seed: Optional[int] = None


# -----------------------------
# Math utilities
# -----------------------------

def _softplus(x: np.ndarray) -> np.ndarray:
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)


def _softmax(x: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    x_max = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - x_max)
    s = np.sum(e, axis=axis, keepdims=True) + eps
    return e / s


def _safe_var(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    m = float(np.mean(x))
    return float(max(0.0, np.mean((x - m) ** 2)))


def _gini(weights: np.ndarray) -> float:
    if weights.size == 0:
        return 0.0
    w = np.sort(weights)
    cum = np.cumsum(w)
    acc = np.sum(cum)
    m = float(weights.size)
    denom = m * cum[-1] + 1e-12
    g = (m + 1.0 - 2.0 * acc / denom) / m
    return float(max(0.0, g))


def _entropy(weights: np.ndarray, eps: float = 1e-12) -> float:
    if weights.size == 0:
        return 0.0
    p = weights / (np.sum(weights) + eps)
    h = -np.sum(p * (np.log(p + eps)))
    # normalize by log(m) so 0..1
    m = float(weights.size)
    if m <= 1:
        return 0.0
    return float(h / math.log(m))


def _weight_center_index(weights: np.ndarray) -> float:
    m = weights.size
    if m <= 1:
        return 0.0
    idx = np.arange(m, dtype=np.float64)
    w_sum = float(np.sum(weights))
    if w_sum <= 0:
        return 0.0
    center = float(np.sum(weights * idx) / (w_sum * (m - 1)))
    return float(min(1.0, max(0.0, center)))


# -----------------------------
# Core schedule constructors
# -----------------------------

def build_schedule(
    overlap_pct: float,
    num_orders: int,
    knobs: EvalKnobs,
) -> Dict[str, Any]:
    if not (0.0 <= overlap_pct <= 100.0):
        raise ValueError("overlap_pct must be in [0,100]")
    if num_orders < 2:
        raise ValueError("num_orders must be >= 2")

    rng = np.random.default_rng(knobs.random_seed)

    # logits for steps and volumes
    ind_logits = rng.normal(0.0, 1.0, size=num_orders).astype(np.float64)
    vol_logits = rng.normal(0.0, 1.0, size=num_orders).astype(np.float64)

    # step sizes -> indent_pct cumulative
    if knobs.use_softplus_indents:
        steps = _softplus(ind_logits)
    else:
        steps = np.maximum(ind_logits, 0.0)

    steps_sum = float(np.sum(steps))
    if not np.isfinite(steps_sum) or steps_sum <= 1e-12:
        steps = np.ones_like(steps)
        steps_sum = float(num_orders)

    steps = steps * ((overlap_pct / 100.0) / steps_sum)
    indent = np.cumsum(steps) * 100.0  # percent scale [0,overlap_pct]

    # volumes -> percent that sums ~100
    if knobs.use_softmax_volumes:
        vol = _softmax(vol_logits.reshape(1, -1), axis=1).reshape(-1)
    else:
        vol = np.maximum(vol_logits, 0.0)
        s = float(np.sum(vol))
        if s <= 1e-12:
            vol = np.ones_like(vol) / float(num_orders)
        else:
            vol = vol / s

    volume_pct = vol * 100.0

    # cap last order volume if requested
    if knobs.last_order_cap is not None:
        cap = float(knobs.last_order_cap)
        if cap >= 0.0:
            last = float(volume_pct[-1])
            if last > cap:
                # reduce last to cap and renormalize the others proportionally
                diff = last - cap
                volume_pct[-1] = cap
                other_sum = float(np.sum(volume_pct[:-1]))
                if other_sum > 1e-9:
                    volume_pct[:-1] = volume_pct[:-1] * ((other_sum + diff) / other_sum)

    # price steps and martingale pct
    price_step_pct = np.empty(num_orders, dtype=np.float64)
    price_step_pct[0] = indent[0]
    if num_orders > 1:
        price_step_pct[1:] = indent[1:] - indent[:-1]

    martingale_pct = np.zeros(num_orders, dtype=np.float64)
    for i in range(1, num_orders):
        prev = volume_pct[i - 1]
        cur = volume_pct[i]
        martingale_pct[i] = 0.0 if prev <= 1e-12 else ((cur / prev) - 1.0) * 100.0

    # prices on base_price=1.0
    order_prices = 1.0 - (indent / 100.0)
    order_prices = np.maximum(order_prices, 1e-6)

    # NeedPct sequence per order k
    needpct = np.empty(num_orders, dtype=np.float64)
    vol_acc = 0.0
    val_acc = 0.0
    for k in range(num_orders):
        vol_acc += volume_pct[k]
        val_acc += volume_pct[k] * order_prices[k]
        avg_price = val_acc / max(vol_acc, 1e-12)
        needpct[k] = (avg_price / max(order_prices[k], 1e-12) - 1.0) * 100.0

    schedule = {
        "indent_pct": indent.tolist(),
        "price_step_pct": price_step_pct.tolist(),
        "martingale_pct": martingale_pct.tolist(),
        "volume_pct": volume_pct.tolist(),
        "order_prices": order_prices.tolist(),
        "needpct": needpct.tolist(),
    }
    return schedule


# -----------------------------
# Risk, penalties, scoring
# -----------------------------

def compute_risk_and_penalties(
    schedule: Dict[str, Any],
    knobs: EvalKnobs,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    volume_pct = np.asarray(schedule["volume_pct"], dtype=np.float64)
    needpct = np.asarray(schedule["needpct"], dtype=np.float64)

    max_need = float(np.max(needpct)) if needpct.size else 0.0
    var_need = _safe_var(needpct)

    # Tail: combine center index and gini for last-k concentration
    vol_weights = volume_pct / (float(np.sum(volume_pct)) + 1e-12)
    tail_center = _weight_center_index(vol_weights)
    tail_gini = _gini(vol_weights)
    tail = 0.6 * tail_center + 0.4 * tail_gini

    # Apply tail cap
    if knobs.limit_tail is not None and tail <= knobs.limit_tail:
        tail_eff = 0.0
    else:
        tail_eff = tail

    # Penalties
    gini = _gini(vol_weights)
    entropy = _entropy(vol_weights)

    # Monotone violations for indent
    indent = np.asarray(schedule["indent_pct"], dtype=np.float64)
    monotone_viol = float(np.sum(np.clip(-(np.diff(indent)), a_min=0.0, a_max=None)))

    # Smoothness violations for price steps (encourage small change between steps)
    steps = np.asarray(schedule["price_step_pct"], dtype=np.float64)
    if steps.size >= 3:
        d_steps = np.diff(steps)
        smooth_viol = float(np.mean(d_steps * d_steps))
    else:
        smooth_viol = 0.0

    risk = {
        "max_need": max_need,
        "var_need": var_need,
        "tail": float(tail_eff),
    }

    penalties = {
        "gini": float(gini),
        "entropy": float(entropy),
        "monotone_viol": float(monotone_viol),
        "smooth_viol": float(smooth_viol),
    }

    return risk, penalties


def compute_score(risk: Dict[str, float], penalties: Dict[str, float], knobs: EvalKnobs) -> float:
    base = knobs.alpha * risk["max_need"] + knobs.beta * risk["var_need"] + knobs.gamma * risk["tail"]
    pen = (
        knobs.penalty_gini * penalties["gini"]
        + knobs.penalty_entropy * (1.0 - penalties["entropy"])  # higher entropy -> lower penalty
        + knobs.penalty_monotone * penalties["monotone_viol"]
        + knobs.penalty_step_smooth * penalties["smooth_viol"]
    )
    return float(base + pen)


# -----------------------------
# Public API
# -----------------------------

def evaluate_configuration(
    overlap_pct: float,
    num_orders: int,
    **kwargs: Any,
) -> Tuple[float, Dict[str, Any]]:
    knobs = EvalKnobs(
        alpha=float(kwargs.get("alpha", 0.4)),
        beta=float(kwargs.get("beta", 0.3)),
        gamma=float(kwargs.get("gamma", 0.3)),
        use_softplus_indents=bool(kwargs.get("use_softplus_indents", True)),
        use_softmax_volumes=bool(kwargs.get("use_softmax_volumes", True)),
        penalty_gini=float(kwargs.get("penalty_gini", 0.0)),
        penalty_entropy=float(kwargs.get("penalty_entropy", 0.0)),
        penalty_monotone=float(kwargs.get("penalty_monotone", 0.0)),
        penalty_step_smooth=float(kwargs.get("penalty_step_smooth", 0.0)),
        limit_tail=kwargs.get("limit_tail", None),
        last_order_cap=kwargs.get("last_order_cap", None),
        random_seed=kwargs.get("random_seed", None),
    )

    try:
        schedule = build_schedule(overlap_pct=overlap_pct, num_orders=num_orders, knobs=knobs)
        risk, penalties = compute_risk_and_penalties(schedule, knobs)
        J = compute_score(risk, penalties, knobs)

        params = {
            "overlap_pct": float(overlap_pct),
            "num_orders": int(num_orders),
            "alpha": knobs.alpha,
            "beta": knobs.beta,
            "gamma": knobs.gamma,
            "use_softplus_indents": knobs.use_softplus_indents,
            "use_softmax_volumes": knobs.use_softmax_volumes,
            "penalty_gini": knobs.penalty_gini,
            "penalty_entropy": knobs.penalty_entropy,
            "penalty_monotone": knobs.penalty_monotone,
            "penalty_step_smooth": knobs.penalty_step_smooth,
            "limit_tail": knobs.limit_tail,
            "last_order_cap": knobs.last_order_cap,
        }
        stable_id = hashlib.sha1(json.dumps(params, sort_keys=True).encode()).hexdigest()[:16]

        metrics: Dict[str, Any] = {
            "score": float(J),
            "schedule": schedule,
            "risk": risk,
            "penalties": penalties,
            "params": params,
            "stable_id": stable_id,
        }

        # Invariants
        indent = schedule["indent_pct"]
        volume = schedule["volume_pct"]
        mart = schedule["martingale_pct"]
        need = schedule["needpct"]
        if not (
            len(indent) == len(volume) == len(mart) == len(need) == num_orders
            and abs(sum(volume) - 100.0) <= 1e-6
            and mart[0] == 0.0
        ):
            raise ValueError("Invariant violation in schedule dimensions or totals")

        return float(J), metrics

    except Exception as e:
        # Error handling path per spec
        metrics = {
            "error": str(e),
            "schedule": {
                "indent_pct": [],
                "price_step_pct": [],
                "martingale_pct": [],
                "volume_pct": [],
                "order_prices": [],
                "needpct": [],
            },
            "risk": {"max_need": float("inf"), "var_need": float("inf"), "tail": float("inf")},
            "penalties": {"gini": float("inf"), "entropy": 0.0, "monotone_viol": float("inf"), "smooth_viol": float("inf")},
            "params": {
                "overlap_pct": float(overlap_pct),
                "num_orders": int(num_orders),
            },
            "stable_id": None,
        }
        return float("inf"), metrics
