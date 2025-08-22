"""
Evaluation engine for batch processing of martingale optimization.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple
import numpy as np

from martingale_lab.adapters.numba_adapter import NumbaAdapter, NumbaAdapterConfig
from .numba_optimizer import evaluate_batch
from .eval_contract import evaluate_configuration


def evaluation_function(overlap_pct: float, num_orders: int, **kwargs) -> Tuple[float, Dict[str, Any]]:
    """
    Contract: returns (J, metrics) with complete schedule/risk/penalties/params/stable_id.
    """
    return evaluate_configuration(overlap_pct=overlap_pct, num_orders=num_orders, **kwargs)


@dataclass
class ExperimentConfig:
    base_price: float = 1.0
    overlap_min: float = 10.0
    overlap_max: float = 20.0
    orders_min: int = 5
    orders_max: int = 20
    n_candidates_per_M: int = 50_000
    seed: int = 123
    top_k_global: int = 100

def _format_bullets(indent_pct: np.ndarray, volume_pct: np.ndarray, martingale_pct: np.ndarray) -> List[str]:
    """
    İstenen metin çıktısı:
    1. Emir: Indent %0.00 Volume %0.10 (no martingale because its first order)
    2. Emir: Indent %0.10 Volume %0.20 (Martingale %100.00)
    """
    lines: List[str] = []
    m = len(indent_pct)
    for i in range(m):
        ind = indent_pct[i]
        vol = volume_pct[i]
        if i == 0:
            lines.append(f"{i+1}. Emir: Indent %{ind:.2f} Volume %{vol:.2f} (no martingale, first order)")
        else:
            lines.append(f"{i+1}. Emir: Indent %{ind:.2f} Volume %{vol:.2f} (Martingale %{martingale_pct[i]:.2f})")
    return lines

def run_experiment(cfg: ExperimentConfig) -> Dict[str, Any]:
    """
    orders_min..orders_max arasında her M için NumbaAdapter çalıştırır,
    tüm adayları birleştirip global top-K döner.
    """
    all_rows: List[Dict[str, Any]] = []
    rng = np.random.default_rng(cfg.seed)
    for M in range(cfg.orders_min, cfg.orders_max + 1):
        nac = NumbaAdapterConfig(
            base_price=cfg.base_price,
            overlap_min=cfg.overlap_min,
            overlap_max=cfg.overlap_max,
            orders=M,
            n_candidates=cfg.n_candidates_per_M,
            seed=int(rng.integers(0, 2**31 - 1)),
        )
        rows = NumbaAdapter(nac).run(top_k=min(cfg.top_k_global, 100), include_schedule=True)
        for r in rows:
            r["M"] = M
        all_rows.extend(rows)

    # global top-K (küçük J en iyi)
    if not all_rows:
        return {"top": [], "best": None}

    order = np.argsort([r["score"] for r in all_rows])[: cfg.top_k_global]
    top = [all_rows[i] for i in order]
    best = top[0]

    # okunur metin
    bullets = _format_bullets(
        indent_pct=np.asarray(best["indent_pct"]),
        volume_pct=np.asarray(best["volume_pct"]),
        martingale_pct=np.asarray(best["martingale_pct"]),
    )

    return {
        "top": top,
        "best": {
            "score": best["score"],
            "overlap_pct": best["overlap_pct"],
            "orders": best["orders"],
            "max_need": best["max_need"],
            "var_need": best["var_need"],
            "tail": best["tail"],
            "bullets": bullets,
            "indent_pct": best["indent_pct"],
            "volume_pct": best["volume_pct"],
            "martingale_pct": best["martingale_pct"],
            "order_prices": best["order_prices"],
        },
    }