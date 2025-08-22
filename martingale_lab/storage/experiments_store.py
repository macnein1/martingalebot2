"""
ExperimentsStore: SQLite persistence for experiments and results with upsert semantics.
Schema:
- experiments: summary rows per experiment run
- results: candidate rows per unique stable_id
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class ExperimentRow:
    id: Optional[int]
    created_at: str
    adapter: str
    best_score: float
    total_evals: int
    elapsed_s: float
    overlap_min: Optional[float] = None
    overlap_max: Optional[float] = None
    orders_min: Optional[int] = None
    orders_max: Optional[int] = None
    alpha: Optional[float] = None
    beta: Optional[float] = None
    gamma: Optional[float] = None
    notes: Optional[str] = None
    deleted: int = 0


class ExperimentsStore:
    def __init__(self, db_path: str = "experiments.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.cursor()
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS experiments (
                  id INTEGER PRIMARY KEY,
                  created_at TEXT NOT NULL,
                  adapter TEXT NOT NULL,
                  best_score REAL NOT NULL,
                  total_evals INTEGER NOT NULL,
                  elapsed_s REAL NOT NULL,
                  overlap_min REAL, overlap_max REAL,
                  orders_min INTEGER, orders_max INTEGER,
                  alpha REAL, beta REAL, gamma REAL,
                  notes TEXT,
                  deleted INTEGER DEFAULT 0
                );
                """
            )
            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_experiments_created ON experiments(created_at DESC);
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS results (
                  id INTEGER PRIMARY KEY,
                  experiment_id INTEGER NOT NULL REFERENCES experiments(id),
                  stable_id TEXT NOT NULL,
                  score REAL NOT NULL,
                  params_json TEXT NOT NULL,
                  schedule_json TEXT NOT NULL,
                  risk_json TEXT NOT NULL,
                  penalties_json TEXT,
                  created_at TEXT NOT NULL
                );
                """
            )
            cur.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS idx_results_stable ON results(stable_id);
                """
            )
            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_results_exp ON results(experiment_id, score);
                """
            )
            conn.commit()

    def create_experiment(self, adapter: str, cfg: Dict[str, Any]) -> int:
        created_at = datetime.now().isoformat()
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO experiments (
                  created_at, adapter, best_score, total_evals, elapsed_s,
                  overlap_min, overlap_max, orders_min, orders_max, alpha, beta, gamma, notes, deleted
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    created_at,
                    adapter,
                    float("inf"),  # placeholder
                    0,
                    0.0,
                    cfg.get("overlap_min"),
                    cfg.get("overlap_max"),
                    cfg.get("orders_min"),
                    cfg.get("orders_max"),
                    cfg.get("alpha"),
                    cfg.get("beta"),
                    cfg.get("gamma"),
                    cfg.get("notes"),
                    0,
                ),
            )
            exp_id = cur.lastrowid
            conn.commit()
            return int(exp_id)

    def update_experiment_summary(self, experiment_id: int, best_score: float, total_evals: int, elapsed_s: float) -> None:
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.cursor()
            cur.execute(
                """
                UPDATE experiments
                SET best_score = MIN(best_score, ?),
                    total_evals = ?,
                    elapsed_s = ?
                WHERE id = ?
                """,
                (best_score, total_evals, elapsed_s, experiment_id),
            )
            conn.commit()

    def upsert_results(self, experiment_id: int, items: List[Dict[str, Any]]) -> int:
        if not items:
            return 0
        now = datetime.now().isoformat()
        inserted = 0
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.cursor()
            for row in items:
                cur.execute(
                    """
                    INSERT INTO results (experiment_id, stable_id, score, params_json, schedule_json, risk_json, penalties_json, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(stable_id) DO UPDATE SET
                      experiment_id=excluded.experiment_id,
                      score=excluded.score,
                      params_json=excluded.params_json,
                      schedule_json=excluded.schedule_json,
                      risk_json=excluded.risk_json,
                      penalties_json=excluded.penalties_json,
                      created_at=excluded.created_at
                    """,
                    (
                        experiment_id,
                        row["stable_id"],
                        row["score"],
                        json.dumps(row["params"], separators=(",", ":")),
                        json.dumps(row["schedule"], separators=(",", ":")),
                        json.dumps(row["risk"], separators=(",", ":")),
                        json.dumps(row.get("penalties", {}), separators=(",", ":")),
                        now,
                    ),
                )
                inserted += 1
            conn.commit()
        return inserted

    def soft_delete_experiment(self, experiment_id: int) -> None:
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.cursor()
            cur.execute("UPDATE experiments SET deleted = 1 WHERE id = ?", (experiment_id,))
            conn.commit()