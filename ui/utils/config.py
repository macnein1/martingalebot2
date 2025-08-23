import streamlit as st
import os
from typing import Dict, Optional

try:
    import psutil  # type: ignore
except Exception:  # optional at runtime
    psutil = None

from martingale_lab.orchestrator.dca_orchestrator import DCAConfig


def get_system_info() -> Dict[str, float]:
    """Detect system CPU and RAM information.

    Returns: {"cpu_count": int, "ram_gb": float}
    """
    cpu_count = os.cpu_count() or 1
    if psutil is not None:
        try:
            total_bytes = psutil.virtual_memory().total  # type: ignore[attr-defined]
            ram_gb = float(total_bytes) / (1024 ** 3)
        except Exception:
            ram_gb = 8.0
    else:
        ram_gb = 8.0

    return {"cpu_count": int(cpu_count), "ram_gb": float(ram_gb)}


def _clip(value: float, lower: float, upper: float) -> float:
    if value < lower:
        return lower
    if value > upper:
        return upper
    return value


def make_auto_config(search_space: Dict[str, float], sys_info: Optional[Dict[str, float]] = None) -> DCAConfig:
    """Create an automatic DCAConfig from search_space and system info.

    search_space: {overlap_min, overlap_max, orders_min, orders_max}
    sys_info: {cpu_count, ram_gb} (optional; auto-detected if None)
    """
    if sys_info is None:
        sys_info = get_system_info()

    overlap_min = float(search_space.get("overlap_min", 10.0))
    overlap_max = float(search_space.get("overlap_max", 30.0))
    orders_min = int(search_space.get("orders_min", 5))
    orders_max = int(search_space.get("orders_max", 15))

    span = max(0.0, overlap_max - overlap_min)
    mavg = (orders_min + orders_max) / 2.0

    # Scoring weights
    alpha = 0.55
    beta = 0.25
    gamma = 0.20
    lambda_penalty = 0.10

    # Constraints
    tail_cap = _clip(0.30 + 0.01 * (mavg - 10.0), 0.25, 0.45)
    min_indent_step = max(0.03, span / max(10.0, mavg * 2.0))
    softmax_temp = 0.8 if span <= 5.0 else 1.0

    # Parallel and batch sizing
    cpu_count = max(1, int(sys_info.get("cpu_count", 4)))
    base_candidates = int(round(1500 * (cpu_count / 4.0)))
    n_candidates_per_batch = int(_clip(float(base_candidates), 1000.0, 8000.0))
    n_workers = min(cpu_count, 8)

    # Wave pattern
    wave_pattern = mavg >= 10.0
    wave_strong_threshold = 50.0
    wave_weak_threshold = 10.0

    return DCAConfig(
        overlap_min=overlap_min,
        overlap_max=overlap_max,
        orders_min=orders_min,
        orders_max=orders_max,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        lambda_penalty=lambda_penalty,
        wave_pattern=wave_pattern,
        wave_strong_threshold=wave_strong_threshold,
        wave_weak_threshold=wave_weak_threshold,
        tail_cap=tail_cap,
        min_indent_step=min_indent_step,
        softmax_temp=softmax_temp,
        n_candidates_per_batch=n_candidates_per_batch,
        n_workers=n_workers,
        early_stop_patience=10,
        top_k_keep=10000,
    )
