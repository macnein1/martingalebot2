import streamlit as st
import os
from typing import Dict, Optional, List, Any

try:
    import psutil  # type: ignore
except Exception:  # optional at runtime
    psutil = None

from martingale_lab.orchestrator.dca_orchestrator import DCAConfig


def setup_page_config():
    """Setup page configuration for Streamlit pages"""
    st.set_page_config(
        page_title="DCA Optimizer",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )


def get_icon_html(icon_name: str) -> str:
    """Get HTML for various icons used in the UI"""
    icons = {
        "clock": "⏱️",
        "chart": "📊",
        "gear": "⚙️",
        "check": "✅",
        "warning": "⚠️",
        "error": "❌",
        "info": "ℹ️",
        "success": "✅",
        "loading": "🔄",
        "play": "▶️",
        "stop": "⏹️",
        "clear": "🗑️",
        "download": "📥",
        "upload": "📤",
        "settings": "⚙️",
        "results": "📋",
        "home": "🏠"
    }
    return icons.get(icon_name, "📄")


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


def compute_exit_speed(needpct: List[float]) -> float:
    """Compute exit speed metric: 1 / (1 + mean(needpct/100))"""
    if not needpct:
        return 0.0
    mean_need = sum(needpct) / len(needpct)
    return 1.0 / (1.0 + mean_need / 100.0)


def compute_wave_score(martingale_pct: List[float], strong: float = 50.0, weak: float = 10.0) -> float:
    """Compute wave pattern score based on strong-weak alternation."""
    if len(martingale_pct) < 2:
        return 0.0
    
    score = 0.0
    transitions = 0
    
    for i in range(1, len(martingale_pct)):
        prev_strong = martingale_pct[i-1] >= strong
        curr_strong = martingale_pct[i] >= strong
        prev_weak = martingale_pct[i-1] <= weak
        curr_weak = martingale_pct[i] <= weak
        
        # Reward strong-weak alternation
        if (prev_strong and curr_weak) or (prev_weak and curr_strong):
            score += 1.0
        # Penalize strong-strong consecutive
        elif prev_strong and curr_strong:
            score -= 0.5
        
        transitions += 1
    
    return max(0.0, min(1.0, score / max(1, transitions)))


def build_bullets(indent_pct: List[float], volume_pct: List[float], 
                 martingale_pct: List[float], needpct: List[float]) -> List[str]:
    """Build bullet points for order details."""
    bullets = []
    n = len(volume_pct)
    
    for i in range(n):
        indent = indent_pct[i] if i < len(indent_pct) else 0.0
        volume = volume_pct[i] if i < len(volume_pct) else 0.0
        martingale = martingale_pct[i] if i < len(martingale_pct) else 0.0
        need = needpct[i] if i < len(needpct) else 0.0
        
        if i == 0:
            mtxt = "(no martingale, first order)"
        else:
            mtxt = f"(Martingale %{martingale:.2f})"
        
        bullet = f"{i+1}. Emir: Indent %{indent:.2f}  Volume %{volume:.2f}  {mtxt} — NeedPct %{need:.2f}"
        bullets.append(bullet)
    
    return bullets


def sanity_badges(sanity: Dict[str, Any]) -> List[str]:
    """Create sanity check badges."""
    badges = []
    if sanity.get("max_need_mismatch"):
        badges.append("max≠")
    if sanity.get("collapse_indents"):
        badges.append("indent↧")
    if sanity.get("tail_overflow"):
        badges.append("tail↑")
    return badges


def create_needpct_sparkline(needpct: List[float], width: int = 20) -> str:
    """Create ASCII sparkline for NeedPct values."""
    if not needpct:
        return "─" * width
    
    min_val = min(needpct)
    max_val = max(needpct)
    if max_val == min_val:
        return "─" * width
    
    # Normalize to 0-7 range for spark characters
    spark_chars = "▁▂▃▄▅▆▇█"
    normalized = [(val - min_val) / (max_val - min_val) * 7 for val in needpct]
    
    # Sample to fit width
    if len(normalized) > width:
        step = len(normalized) / width
        sampled = [normalized[int(i * step)] for i in range(width)]
    else:
        sampled = normalized
    
    sparkline = "".join([spark_chars[min(7, int(val))] for val in sampled])
    return sparkline
