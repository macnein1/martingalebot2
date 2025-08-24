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