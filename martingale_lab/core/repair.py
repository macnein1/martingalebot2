from __future__ import annotations

import numpy as np
from typing import Tuple, Dict


def hard_clip_local_growth(volumes: np.ndarray, g_min: float, g_max: float) -> Tuple[np.ndarray, Dict[str, int]]:
    """
    Apply hard clipping to local growth ratios for i>=2 and rebuild the chain.

    Keeps v0 as-is and ensures v1 <= v0. For i>=2, enforces g_i in [g_min, g_max].

    Args:
        volumes: Volume percentages array (sum can be arbitrary)
        g_min: Minimum allowed local growth ratio (>=1.0 for monotone)
        g_max: Maximum allowed local growth ratio

    Returns:
        Tuple of (new_volumes, stats) where new_volumes preserves v0 and clamps ratios,
        and stats contains counts of hi/lo clips.
    """
    if volumes.size == 0:
        return volumes.copy(), {"hi": 0, "lo": 0}

    v = volumes.astype(np.float64).copy()
    M = v.size

    out = np.zeros_like(v)
    out[0] = v[0]
    if M >= 2:
        out[1] = min(v[1], out[0])
    hi = 0
    lo = 0
    for i in range(2, M):
        prev = max(out[i-1], 1e-12)
        g = v[i] / max(v[i-1], 1e-12)
        if g > g_max:
            g = g_max
            hi += 1
        elif g < g_min:
            g = g_min
            lo += 1
        out[i] = prev * g

    return out, {"hi": int(hi), "lo": int(lo)}


def isotonic_non_decreasing(y: np.ndarray) -> np.ndarray:
    """
    Pool Adjacent Violators Algorithm (PAVA) for isotonic regression (non-decreasing).
    Equal weights are assumed.

    Args:
        y: 1D array to be made non-decreasing

    Returns:
        Isotonic-regressed array of same shape
    """
    n = int(y.size)
    if n <= 1:
        return y.astype(np.float64).copy()

    # Initialize blocks
    levels = []  # block means
    counts = []  # block sizes (ints)

    for i in range(n):
        levels.append(float(y[i]))
        counts.append(1)
        # Merge while decreasing
        while len(levels) >= 2 and levels[-2] > levels[-1]:
            total_count = counts[-2] + counts[-1]
            new_level = (levels[-2] * counts[-2] + levels[-1] * counts[-1]) / total_count
            levels[-2] = new_level
            counts[-2] = total_count
            del levels[-1]
            del counts[-1]

    # Expand blocks back to full array
    out = np.empty(n, dtype=np.float64)
    idx = 0
    for lvl, cnt in zip(levels, counts):
        out[idx: idx + cnt] = lvl
        idx += cnt
    return out


def tail_only_rescale_keep_first_two(v: np.ndarray) -> float:
    """
    v[0], v[1] sabit kalsın; v[2:] tek bir faktörle ölçeklensin ki sum(v)=100 olsun.
    f = (100 - (v0+v1)) / sum(v[2:])
    v[2:] *= f
    return f
    # S_tail=0 durumunda: v[2:]'yi küçük artan bir geometrik rampa ile doldur,
    # sonra aynı formülle ölçekle. Bölüm sıfır guard'ları ekle.
    """
    if v.size < 2:
        return 1.0
    
    v0, v1 = v[0], v[1]
    tail_sum = np.sum(v[2:])
    
    if tail_sum <= 1e-12:
        # S_tail=0 durumunda: küçük artan geometrik rampa ile doldur
        n_tail = v.size - 2
        if n_tail > 0:
            # v[2] = v1 * 1.01, v[3] = v[2] * 1.01, ...
            v[2] = v1 * 1.01
            for i in range(3, v.size):
                v[i] = v[i-1] * 1.01
            tail_sum = np.sum(v[2:])
    
    if tail_sum <= 1e-12:
        return 1.0
    
    target_tail = 100.0 - v0 - v1
    if target_tail <= 0:
        # Eğer v0+v1 >= 100, tail'i çok küçük yap
        f = 1e-6 / tail_sum
        v[2:] *= f
        return f
    
    f = target_tail / tail_sum
    v[2:] *= f
    return f


def compute_m_from_v(v: np.ndarray) -> np.ndarray:
    """ m[0]=0; m[i] = v[i]/max(v[i-1],1e-12) - 1 """
    m = np.zeros_like(v)
    for i in range(1, v.size):
        m[i] = v[i] / max(v[i-1], 1e-12) - 1.0
    return m


def rechain_v_from_m(v0: float, v1: float, m: np.ndarray) -> np.ndarray:
    """ v[0]=v0; v[1]=v1; i>=2: v[i] = v[i-1]*(1+m[i]) """
    v = np.zeros_like(m)
    v[0] = v0
    if m.size > 1:
        v[1] = v1
        for i in range(2, m.size):
            v[i] = v[i-1] * (1.0 + m[i])
    return v


def longest_plateau_run(m: np.ndarray, center=1.0, tol=0.02, start_idx=2):
    """ |m-center|<tol koşullu run'ları ve max uzunluğu döndür (HC6 için) """
    if m.size <= start_idx:
        return 0
    
    max_run = 0
    current_run = 0
    
    for i in range(start_idx, m.size):
        if abs(m[i] - center) < tol:
            current_run += 1
            max_run = max(max_run, current_run)
        else:
            current_run = 0
    
    return max_run