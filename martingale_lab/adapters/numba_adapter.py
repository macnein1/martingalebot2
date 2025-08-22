"""
Numba-based optimization engine for martingale strategies.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, List
import numpy as np

from ..optimizer.numba_optimizer import evaluate_batch

@dataclass
class NumbaAdapterConfig:
    base_price: float = 1.0
    overlap_min: float = 5.0
    overlap_max: float = 30.0
    orders: int = 10
    n_candidates: int = 50_000
    seed: int = 42

class NumbaAdapter:
    """
    Numba tabanlı geniş-tarama adaptörü.
    - Adayları logit uzayında üretir
    - JIT kernel ile skorlar
    - Top sonuçlar için schedule (indent %, volume %, martingale %) decode eder
    """
    def __init__(self, cfg: NumbaAdapterConfig):
        self.cfg = cfg
        self._rng = np.random.default_rng(cfg.seed)

    # ---------- candidate sampling ----------
    def _sample_candidates(self) -> Dict[str, np.ndarray]:
        n, m = self.cfg.n_candidates, self.cfg.orders
        overlaps = self._rng.uniform(self.cfg.overlap_min, self.cfg.overlap_max, size=n).astype(np.float64)
        ind_logits = self._rng.normal(loc=0.0, scale=1.0, size=(n, m)).astype(np.float64)
        vol_logits = self._rng.normal(loc=0.0, scale=1.0, size=(n, m)).astype(np.float64)

        # entry'ye yakın başlat: ilk emrin adımı küçük olsun
        ind_logits[:, 0] -= 2.0
        ind_logits += np.linspace(0.0, 1.0, m, dtype=np.float64)  # sonlara doğru hafif artış

        # vol için hafif dalga → yığılma kırıcı
        wave = 0.3 * np.sin(np.linspace(0, np.pi, m, dtype=np.float64))
        vol_logits += wave

        return {"overlaps": overlaps, "ind_logits": ind_logits, "vol_logits": vol_logits}

    # ---------- decode helpers (Python; UI için) ----------
    @staticmethod
    def _softplus(x: np.ndarray) -> np.ndarray:
        return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)

    @staticmethod
    def _softmax(x: np.ndarray, axis: int = -1, eps: float = 1e-9) -> np.ndarray:
        x = x - np.max(x, axis=axis, keepdims=True)
        e = np.exp(x)
        s = np.sum(e, axis=axis, keepdims=True) + eps
        return e / s

    def _decode_schedule(
        self,
        overlap_pct: float,
        ind_logits_row: np.ndarray,
        vol_logits_row: np.ndarray,
        base_price: float,
        eps: float = 1e-9,
    ) -> Dict[str, np.ndarray]:
        """
        UI gösterimi için schedule üretir:
        - indent_pct (kümülatif, %)
        - volume_pct (%, toplam 100)
        - martingale_pct (iki fiyat arası % düşüş)
        - order_prices
        """
        m = ind_logits_row.size

        # indents: pozitif step → toplamı overlap_pct
        steps = self._softplus(ind_logits_row.astype(np.float64))
        if steps.sum() <= eps:
            steps = np.ones_like(steps)
        steps = steps * ((overlap_pct / 100.0) / steps.sum())

        cum_indent = np.cumsum(steps)  # 0..overlap_pct (oran)
        prices = base_price * (1.0 - np.clip(cum_indent, 0, 0.95))

        # volumes: softmax → toplam 1
        vols = self._softmax(vol_logits_row.astype(np.float64))
        volume_pct = 100.0 * vols

        # martingale %: ardışık fiyat düşüşü
        mart = np.zeros(m, dtype=np.float64)
        for i in range(1, m):
            prev_p, cur_p = prices[i - 1], prices[i]
            mart[i] = ((prev_p - cur_p) / prev_p) * 100.0

        # indent % (kümülatif)
        indent_pct = 100.0 * cum_indent

        return {
            "indent_pct": indent_pct,
            "volume_pct": volume_pct,
            "martingale_pct": mart,
            "order_prices": prices,
        }

    # ---------- run ----------
    def run(self, top_k: int = 100, include_schedule: bool = True) -> List[Dict[str, Any]]:
        cands = self._sample_candidates()
        J, mx, vr, tl = evaluate_batch(
            base_price=self.cfg.base_price,
            overlaps=cands["overlaps"],
            ind_logits=cands["ind_logits"],
            vol_logits=cands["vol_logits"],
        )

        n = J.shape[0]
        idx = np.argsort(J)[: min(top_k, n)]  # küçük J daha iyi

        results: List[Dict[str, Any]] = []
        for i in idx:
            row: Dict[str, Any] = {
                "score": float(J[i]),
                "overlap_pct": float(cands["overlaps"][i]),
                "orders": int(self.cfg.orders),
                "max_need": float(mx[i]),
                "var_need": float(vr[i]),
                "tail": float(tl[i]),
            }
            if include_schedule:
                sch = self._decode_schedule(
                    overlap_pct=cands["overlaps"][i],
                    ind_logits_row=cands["ind_logits"][i],
                    vol_logits_row=cands["vol_logits"][i],
                    base_price=self.cfg.base_price,
                )
                row.update(sch)
            results.append(row)
        return results