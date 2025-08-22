"""
Numba-based optimization engine for martingale strategies.
"""
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Tuple
import numpy as np
from numba import njit, prange

# -----------------------
# Datamodels
# -----------------------
@dataclass
class NumbaEvalConfig:
    base_price: float             = 1.0
    alpha: float                  = 0.4   # weight for max_need
    beta: float                   = 0.3   # weight for var_need
    gamma: float                  = 0.3   # weight for tail_penalty
    eps: float                    = 1e-9

# -----------------------
# Utilities (NumPy)
# -----------------------
def _softplus(x: np.ndarray) -> np.ndarray:
    # stable softplus
    out = np.empty_like(x)
    np.copyto(out, np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0))
    return out

def _row_softmax(x: np.ndarray, eps: float) -> np.ndarray:
    # x: (n, m)
    x_max = np.max(x, axis=1, keepdims=True)
    e = np.exp(x - x_max)
    s = np.sum(e, axis=1, keepdims=True) + eps
    return e / s

# -----------------------
# Core kernels (Numba)
# -----------------------
@njit(cache=True, fastmath=True)
def _gini(weights: np.ndarray, eps: float) -> float:
    # weights: (m,)
    m = weights.size
    if m == 0:
        return 0.0
    # sort ascending
    w = np.sort(weights)
    cum = 0.0
    acc = 0.0
    for i in range(m):
        cum += w[i]
        acc += cum
    denom = m * cum + eps
    g = (m + 1.0 - 2.0 * acc / (cum + eps)) / m
    if g < 0.0:
        g = 0.0
    return g

@njit(cache=True, fastmath=True)
def _weight_center_index(weights: np.ndarray) -> float:
    """
    0 -> tüm ağırlık başta, 1 -> tüm ağırlık sonda.
    """
    m = weights.size
    if m <= 1:
        return 0.0
    idx_sum = 0.0
    w_sum = 0.0
    for i in range(m):
        idx_sum += weights[i] * i
        w_sum += weights[i]
    if w_sum <= 0.0:
        return 0.0
    center = idx_sum / (w_sum * (m - 1))
    if center < 0.0:
        center = 0.0
    if center > 1.0:
        center = 1.0
    return center

@njit(cache=True, fastmath=True)
def _eval_single(base_price: float,
                 overlap_pct: float,
                 ind_logits: np.ndarray,
                 vol_logits: np.ndarray,
                 eps: float) -> Tuple[float, float, float, float]:
    """
    Tek aday skoru.
    Returns: (J, max_need, var_need, tail_penalty)
    """
    m = ind_logits.size
    # --- normalize indents to positive step sizes summing to overlap_pct
    # softplus -> positive steps; then normalize to sum=overlap_pct
    steps_raw = ind_logits.copy()
    # softplus
    for i in range(m):
        x = steps_raw[i]
        steps_raw[i] = math.log1p(math.exp(-abs(x))) + (x if x > 0 else 0.0)
    total = 0.0
    for i in range(m):
        total += steps_raw[i]
    if total <= eps:
        # uniform minimal steps
        for i in range(m):
            steps_raw[i] = 1.0
        total = float(m)
    scale = (overlap_pct / 100.0) / total
    for i in range(m):
        steps_raw[i] *= scale

    # cumulative indent percentages from entry (0..overlap_pct)
    cum_indent = np.empty(m)
    acc = 0.0
    for i in range(m):
        acc += steps_raw[i]
        cum_indent[i] = acc
    # price per order
    prices = np.empty(m)
    for i in range(m):
        dec = cum_indent[i]
        if dec >= 0.95:  # safety clamp
            dec = 0.95
        prices[i] = base_price * (1.0 - dec)

    # --- volumes: softmax -> sum=1
    # stabilize logits
    vmax = -1e30
    for i in range(m):
        if vol_logits[i] > vmax:
            vmax = vol_logits[i]
    exps = np.empty(m)
    ssum = 0.0
    for i in range(m):
        e = math.exp(vol_logits[i] - vmax)
        exps[i] = e
        ssum += e
    if ssum <= eps:
        for i in range(m):
            exps[i] = 1.0
        ssum = float(m)
    volumes = np.empty(m)
    for i in range(m):
        volumes[i] = exps[i] / (ssum + eps)

    # --- NeedPct sequence
    max_need = 0.0
    mean_need = 0.0
    mean2_need = 0.0
    vol_acc = 0.0
    val_acc = 0.0
    for k in range(m):
        vol_acc += volumes[k]
        val_acc += volumes[k] * prices[k]
        avg_price = val_acc / (vol_acc + eps)
        need = (avg_price / (prices[k] + eps) - 1.0) * 100.0
        # track stats
        if need > max_need:
            max_need = need
        mean_need += need
        mean2_need += need * need
    mean_need /= m
    var_need = max(0.0, (mean2_need / m) - (mean_need * mean_need))

    # --- tail penalty (yığılma): weight-center + gini
    tail = 0.6 * _weight_center_index(volumes) + 0.4 * _gini(volumes, eps)

    # --- objective
    J = 0.4 * max_need + 0.3 * var_need + 0.3 * tail
    return J, max_need, var_need, tail

@njit(parallel=True, cache=True, fastmath=True)
def evaluate_batch(base_price: float,
                   overlaps: np.ndarray,       # (n,)
                   ind_logits: np.ndarray,     # (n, m)
                   vol_logits: np.ndarray,     # (n, m)
                   eps: float = 1e-9
                   ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Tüm adayları değerlendir.
    Returns arrays of shape (n,): J, max_need, var_need, tail
    """
    n = overlaps.shape[0]
    J = np.empty(n)
    max_need = np.empty(n)
    var_need = np.empty(n)
    tail = np.empty(n)
    for i in prange(n):
        Ji, mx, vr, tl = _eval_single(base_price,
                                      overlaps[i],
                                      ind_logits[i],
                                      vol_logits[i],
                                      eps)
        J[i] = Ji
        max_need[i] = mx
        var_need[i] = vr
        tail[i] = tl
    return J, max_need, var_need, tail