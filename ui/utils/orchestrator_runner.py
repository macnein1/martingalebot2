from __future__ import annotations

import logging
import threading
import time
import hashlib
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Any, Optional, List

from .constants import DB_PATH
from .logging_buffer import ensure_ring_handler
from martingale_lab.adapters.numba_adapter import NumbaAdapter, NumbaAdapterConfig
from martingale_lab.storage.experiments_store import ExperimentsStore


logger = logging.getLogger("mlab")
ensure_ring_handler("mlab")


def _stable_id_from_params(params: Dict[str, Any]) -> str:
    raw = "|".join(f"{k}={params[k]}" for k in sorted(params.keys()))
    return hashlib.md5(raw.encode("utf-8")).hexdigest()[:16]


@dataclass
class RunnerConfig:
    min_overlap: float
    max_overlap: float
    min_order: int
    max_order: int
    db_path: str = DB_PATH
    batches: int = 5
    top_k_per_batch: int = 20
    n_candidates_per_batch: int = 20000
    sleep_between_batches: float = 0.5


class BackgroundOrchestrator:
    def __init__(self, cfg: RunnerConfig, stop_event: Optional[threading.Event] = None):
        self.cfg = cfg
        self.stop_event = stop_event or threading.Event()
        self.store = ExperimentsStore(cfg.db_path)
        self.experiment_id: Optional[int] = None

    def run(self, run_id: str) -> None:
        try:
            logger.info("BUILD_CONFIG run_id=%s overlap=[%.2f, %.2f] orders=[%d, %d]",
                        run_id, self.cfg.min_overlap, self.cfg.max_overlap,
                        self.cfg.min_order, self.cfg.max_order)

            # Create experiment row
            self.experiment_id = self.store.create_experiment(
                adapter="adaptive",
                cfg={
                    "overlap_min": self.cfg.min_overlap,
                    "overlap_max": self.cfg.max_overlap,
                    "orders_min": self.cfg.min_order,
                    "orders_max": self.cfg.max_order,
                },
            )

            logger.info("ORCH.START run_id=%s experiment_id=%s", run_id, self.experiment_id)

            # Use numba adapter; vary orders crudely by choosing max_order
            adapter_cfg = NumbaAdapterConfig(
                overlap_min=float(self.cfg.min_overlap),
                overlap_max=float(self.cfg.max_overlap),
                orders=int(self.cfg.max_order),
                n_candidates=int(self.cfg.n_candidates_per_batch),
                seed=int(time.time()) % 2**31,
            )
            adapter = NumbaAdapter(adapter_cfg)

            global_best = float("inf")
            total_evals = 0
            for b in range(1, self.cfg.batches + 1):
                if self.stop_event.is_set():
                    logger.info("ORCH.STOPPED run_id=%s at batch=%d", run_id, b)
                    break

                batch_results: List[Dict[str, Any]] = adapter.run(
                    top_k=self.cfg.top_k_per_batch, include_schedule=True
                )

                items = []
                for r in batch_results:
                    params = {
                        "overlap_pct": r.get("overlap_pct"),
                        "orders": r.get("orders"),
                    }
                    indent = r.get("indent_pct", [])
                    volume = r.get("volume_pct", [])
                    mart = r.get("martingale_pct", [])
                    prices = r.get("order_prices", [])
                    sched = {
                        "indent_pct": indent,
                        "volume_pct": volume,
                        "martingale_pct": mart,
                        "order_prices": prices,
                        "price_step_pct": mart,
                        "needpct": [0.0 for _ in range(len(indent))],
                    }
                    risk = {
                        "max_need": r.get("max_need", 0.0),
                        "var_need": r.get("var_need", 0.0),
                        "tail": r.get("tail", 0.0),
                    }
                    sid = _stable_id_from_params(params)
                    items.append({
                        "stable_id": sid,
                        "score": r.get("score", 0.0),
                        "params": params,
                        "schedule": sched,
                        "risk": risk,
                        "penalties": {},
                    })
                    if r.get("score", float("inf")) < global_best:
                        global_best = float(r.get("score"))
                inserted = self.store.upsert_results(self.experiment_id, items)
                total_evals += len(batch_results)
                logger.info(
                    "BATCH %d/%d inserted=%d best=%.6f run_id=%s",
                    b, self.cfg.batches, inserted, global_best, run_id,
                )
                logger.info("SAVE_OK run_id=%s rows=%d", run_id, inserted)

                # Update summary each batch
                self.store.update_experiment_summary(
                    self.experiment_id,
                    best_score=global_best,
                    total_evals=total_evals,
                    elapsed_s=float(b),
                )

                time.sleep(self.cfg.sleep_between_batches)

            logger.info("ORCH.DONE run_id=%s best=%.6f", run_id, global_best)
        except Exception:
            logger.exception("ORCH.ERROR run_id=%s", run_id)
            raise


