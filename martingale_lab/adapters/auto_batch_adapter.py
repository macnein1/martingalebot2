"""
AutoBatch adapter for martingale optimization.
"""
from __future__ import annotations
import time
import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import logging
import numpy as np

from ..interfaces import OptimizerAdapter, SearchConfig, Candidate, ResultTrace
from ..optimizer.evaluation_engine import evaluation_function

logger = logging.getLogger(__name__)


# -----------------------------
# Config
# -----------------------------
@dataclass
class AutoBatchConfig:
    """Basit batch tabanlı arama ayarları (adapter içi, orchestrator BAĞIMSIZ)"""
    target_evals_per_batch: int = 100_000
    max_batches: int = 20
    batch_seconds: Optional[float] = None     # örn. 5*60; None ise eval sayısı bazlı
    plateau_tol: float = 1e-4                 # iyileşme eşiği
    plateau_window: int = 3                   # son k batch’e bak
    expand_factor: float = 1.25               # plato varsa aralığı genişlet
    topk_keep: int = 50                       # her batch’ten sonra tutulacak aday sayısı
    random_seed: Optional[int] = None


# -----------------------------
# Adapter
# -----------------------------
@dataclass
class AutoBatchAdapter(OptimizerAdapter):
    """AutoBatch mantığını taşıyan adapter. Orchestrator çağırır."""
    name: str = "auto_batch"
    config: AutoBatchConfig = field(default_factory=AutoBatchConfig)

    def search(
        self,
        cfg: SearchConfig,
        time_budget_s: Optional[float] = None,
    ) -> Tuple[List[Candidate], List[ResultTrace]]:
        """
        Basit batch’li arama:
          - her batch’te rastgele/Latin örnekler
          - evaluation_engine ile skorla
          - en iyileri tut, plato tespit et, gerekiyorsa aralığı genişlet
        """
        rng = np.random.default_rng(self.config.random_seed)
        start = time.time()

        best_J = math.inf
        best_candidates: List[Candidate] = []
        traces: List[ResultTrace] = []

        # aktif aralıklar (mutasyon için kopya)
        o_min, o_max = cfg.overlap_min, cfg.overlap_max
        m_min, m_max = cfg.orders_min, cfg.orders_max

        # plato takibi
        recent_best: List[float] = []

        for b in range(self.config.max_batches):
            if time_budget_s is not None and time.time() - start >= time_budget_s:
                logger.info("[auto_batch] time budget reached at batch %d", b)
                break

            # Batch büyüklüğü: süre verildiyse tahmini throughput yoksa sabit eval
            eval_budget = self.config.target_evals_per_batch

            # Basit örnekleme (orders ve overlap):
            overlaps = rng.uniform(o_min, o_max, size=eval_budget).astype(np.float32)
            orders = rng.integers(m_min, m_max + 1, size=eval_budget, dtype=np.int32)

            batch_best_J = math.inf
            batch_best_items: List[Tuple[float, Candidate, ResultTrace]] = []

            for ov, M in zip(overlaps, orders):
                # evaluation_engine, ihtiyacınız olan schedule alanlarını üretir
                J, metrics = evaluation_function(
                    overlap_pct=float(ov),
                    num_orders=int(M),
                    # buraya gerekiyorsa weight/penalty/alpha-beta-gamma gibi
                    # ek parametreler (cfg’den) geçirilebilir
                )

                cand = Candidate(
                    params={
                        "overlap_pct": float(ov),
                        "num_orders": int(M),
                    },
                    schedule=metrics.get("schedule", {}),  # indent_pct[], volume_pct[] vb.
                    score=float(J),
                )

                trace = ResultTrace(
                    score=float(J),
                    metrics=metrics,
                    elapsed_s=None,
                    evals=1,
                    adapter=self.name,
                )

                batch_best_items.append((cand.score, cand, trace))
                if J < batch_best_J:
                    batch_best_J = J

            # Batch sonrası en iyileri sırala ve kıs
            batch_best_items.sort(key=lambda x: x[0])
            keep = batch_best_items[: self.config.topk_keep]

            # global set’e ekle
            for Jx, cx, tx in keep:
                traces.append(tx)
                best_candidates.append(cx)

            # global sıralama ve kısma (hafızayı kontrol)
            best_candidates.sort(key=lambda c: c.score)
            best_candidates = best_candidates[: self.config.topk_keep]

            # plato tespiti
            recent_best.append(float(batch_best_items[0][0]))
            if len(recent_best) > self.config.plateau_window:
                recent_best.pop(0)

            plateau = False
            if len(recent_best) == self.config.plateau_window:
                gain = recent_best[0] - recent_best[-1]
                if abs(gain) <= self.config.plateau_tol:
                    plateau = True

            if plateau:
                # aralığı biraz genişlet (örnek; alt sınırlar sabit tutulabilir)
                span_o = (o_max - o_min) * (self.config.expand_factor - 1.0)
                span_m = max(1, int((m_max - m_min) * (self.config.expand_factor - 1.0)))

                o_max = min(cfg.overlap_max_global or 100.0, o_max + span_o)
                m_max = min(cfg.orders_max_global or 1000, m_max + span_m)
                logger.info(
                    "[auto_batch] plateau detected → expand ranges: overlap_max=%.4f, orders_max=%d",
                    o_max, m_max
                )

            # global en iyi güncelle
            if batch_best_J < best_J:
                best_J = float(batch_best_J)

            # zaman kontrolü (süre bazlı batch’te de çıkış emniyeti)
            if time_budget_s is not None and time.time() - start >= time_budget_s:
                break

        # Son sıralama
        best_candidates.sort(key=lambda c: c.score)
        return best_candidates, traces