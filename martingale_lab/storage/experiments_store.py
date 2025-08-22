"""
ExperimentsStore: SQLite persistence for experiments and results with upsert semantics.
Schema:
- experiments: summary rows per experiment run
- results: candidate rows per unique stable_id
"""
from __future__ import annotations

import json
import numpy as np
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple


def _jsonify(o: Any):
    # numpy array -> list
    if isinstance(o, np.ndarray):
        return o.tolist()
    # numpy scalars -> python scalars
    if isinstance(o, np.generic):
        return o.item()
    # recursive containers
    if isinstance(o, (list, tuple)):
        return [ _jsonify(x) for x in o ]
    if isinstance(o, dict):
        return { k: _jsonify(v) for k, v in o.items() }
    return o

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
    lambda_penalty: Optional[float] = None
    wave_pattern: Optional[bool] = None
    tail_cap: Optional[float] = None
    notes: Optional[str] = None
    deleted: int = 0


class ExperimentsStore:
    def __init__(self, db_path: str = "db_results/experiments.db"):
        import os
        from pathlib import Path
        self.db_path = db_path
        # Ensure directory exists
        Path(os.path.dirname(self.db_path) or ".").mkdir(parents=True, exist_ok=True)
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
                  lambda_penalty REAL,
                  wave_pattern INTEGER,
                  tail_cap REAL,
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
                  max_need REAL NOT NULL,
                  var_need REAL NOT NULL,
                  tail REAL NOT NULL,
                  shape_reward REAL,
                  cvar80 REAL,
                  params_json TEXT NOT NULL,
                  schedule_json TEXT NOT NULL,
                  sanity_json TEXT NOT NULL,
                  diagnostics_json TEXT NOT NULL,
                  penalties_json TEXT NOT NULL,
                  knobs_json TEXT NOT NULL,
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
            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_results_metrics ON results(max_need, var_need, tail);
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
                  overlap_min, overlap_max, orders_min, orders_max, 
                  alpha, beta, gamma, lambda_penalty, wave_pattern, tail_cap, 
                  notes, deleted
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                    cfg.get("alpha", 0.5),
                    cfg.get("beta", 0.3),
                    cfg.get("gamma", 0.2),
                    cfg.get("lambda_penalty", 0.1),
                    1 if cfg.get("wave_pattern", False) else 0,
                    cfg.get("tail_cap", 0.40),
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
        """
        Upsert results with complete DCA evaluation contract structure.
        Each item should contain: score, max_need, var_need, tail, schedule, sanity, diagnostics, penalties, knobs.
        """
        if not items:
            return 0
        now = datetime.now().isoformat()
        inserted = 0
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.cursor()
            for item in items:
                # Generate stable_id from parameters
                params = item.get("params", {})
                stable_id = item.get("stable_id")
                if not stable_id:
                    # Generate from core parameters
                    import hashlib
                    params_str = json.dumps(params, sort_keys=True)
                    stable_id = hashlib.sha1(params_str.encode()).hexdigest()[:16]
                
                cur.execute(
                    """
                    INSERT INTO results (
                        experiment_id, stable_id, score, max_need, var_need, tail, shape_reward, cvar80,
                        params_json, schedule_json, sanity_json, diagnostics_json, 
                        penalties_json, knobs_json, created_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(stable_id) DO UPDATE SET
                      experiment_id=excluded.experiment_id,
                      score=excluded.score,
                      max_need=excluded.max_need,
                      var_need=excluded.var_need,
                      tail=excluded.tail,
                      shape_reward=excluded.shape_reward,
                      cvar80=excluded.cvar80,
                      params_json=excluded.params_json,
                      schedule_json=excluded.schedule_json,
                      sanity_json=excluded.sanity_json,
                      diagnostics_json=excluded.diagnostics_json,
                      penalties_json=excluded.penalties_json,
                      knobs_json=excluded.knobs_json,
                      created_at=excluded.created_at
                    """,
                    (
                        experiment_id,
                        stable_id,
                        float(item.get("score", float("inf"))),
                        float(item.get("max_need", 0.0)),
                        float(item.get("var_need", 0.0)),
                        float(item.get("tail", 0.0)),
                        float(item.get("shape_reward", 0.0)),
                        float(item.get("cvar_need", 0.0)),  # Store as cvar80
                        json.dumps(_jsonify(params), separators=(",", ":"), ensure_ascii=False),
                        json.dumps(_jsonify(item.get("schedule", {})), separators=(",", ":"), ensure_ascii=False),
                        json.dumps(_jsonify(item.get("sanity", {})), separators=(",", ":"), ensure_ascii=False),
                        json.dumps(_jsonify(item.get("diagnostics", {})), separators=(",", ":"), ensure_ascii=False),
                        json.dumps(_jsonify(item.get("penalties", {})), separators=(",", ":"), ensure_ascii=False),
                        json.dumps(_jsonify(item.get("knobs", {})), separators=(",", ":"), ensure_ascii=False),
                        now,
                    ),
                )
                inserted += 1
            conn.commit()
        return inserted

    def get_top_results(self, 
                       experiment_id: Optional[int] = None,
                       limit: int = 100,
                       filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Get top results with optional filtering."""
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.cursor()
            
            query = """
                SELECT r.*, e.adapter, e.created_at as exp_created_at
                FROM results r
                JOIN experiments e ON r.experiment_id = e.id
                WHERE e.deleted = 0
            """
            params = []
            
            if experiment_id is not None:
                query += " AND r.experiment_id = ?"
                params.append(experiment_id)
                
            if filters:
                if "max_score" in filters:
                    query += " AND r.score <= ?"
                    params.append(filters["max_score"])
                if "min_max_need" in filters:
                    query += " AND r.max_need >= ?"
                    params.append(filters["min_max_need"])
                if "max_max_need" in filters:
                    query += " AND r.max_need <= ?"
                    params.append(filters["max_max_need"])
                if "wave_pattern_only" in filters and filters["wave_pattern_only"]:
                    query += " AND e.wave_pattern = 1"
                    
            query += " ORDER BY r.score ASC LIMIT ?"
            params.append(limit)
            
            cur.execute(query, params)
            rows = cur.fetchall()
            
            results = []
            for row in rows:
                # Parse JSON fields
                try:
                    result = {
                        "id": row[0],
                        "experiment_id": row[1],
                        "stable_id": row[2],
                        "score": row[3],
                        "max_need": row[4],
                        "var_need": row[5],
                        "tail": row[6],
                        "params": json.loads(row[7]),
                        "schedule": json.loads(row[8]),
                        "sanity": json.loads(row[9]),
                        "diagnostics": json.loads(row[10]),
                        "penalties": json.loads(row[11]),
                        "knobs": json.loads(row[12]),
                        "created_at": row[13],
                        "adapter": row[14],
                        "exp_created_at": row[15],
                    }
                    results.append(result)
                except json.JSONDecodeError as e:
                    # Skip malformed records
                    continue
                    
            return results

    def get_experiment_summary(self, experiment_id: int) -> Optional[Dict[str, Any]]:
        """Get experiment summary with statistics."""
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.cursor()
            
            # Get experiment details
            cur.execute(
                "SELECT * FROM experiments WHERE id = ? AND deleted = 0",
                (experiment_id,)
            )
            exp_row = cur.fetchone()
            if not exp_row:
                return None
                
            # Get result statistics
            cur.execute(
                """
                SELECT 
                    COUNT(*) as total_results,
                    MIN(score) as best_score,
                    AVG(max_need) as avg_max_need,
                    MIN(max_need) as min_max_need,
                    MAX(max_need) as max_max_need,
                    AVG(var_need) as avg_var_need,
                    AVG(tail) as avg_tail
                FROM results 
                WHERE experiment_id = ?
                """,
                (experiment_id,)
            )
            stats_row = cur.fetchone()
            
            return {
                "id": exp_row[0],
                "created_at": exp_row[1],
                "adapter": exp_row[2],
                "best_score": exp_row[3],
                "total_evals": exp_row[4],
                "elapsed_s": exp_row[5],
                "config": {
                    "overlap_min": exp_row[6],
                    "overlap_max": exp_row[7],
                    "orders_min": exp_row[8],
                    "orders_max": exp_row[9],
                    "alpha": exp_row[10],
                    "beta": exp_row[11],
                    "gamma": exp_row[12],
                    "lambda_penalty": exp_row[13],
                    "wave_pattern": bool(exp_row[14]) if exp_row[14] is not None else False,
                    "tail_cap": exp_row[15],
                },
                "statistics": {
                    "total_results": stats_row[0] if stats_row else 0,
                    "best_score": stats_row[1] if stats_row else float("inf"),
                    "avg_max_need": stats_row[2] if stats_row else 0.0,
                    "min_max_need": stats_row[3] if stats_row else 0.0,
                    "max_max_need": stats_row[4] if stats_row else 0.0,
                    "avg_var_need": stats_row[5] if stats_row else 0.0,
                    "avg_tail": stats_row[6] if stats_row else 0.0,
                }
            }

    def soft_delete_experiment(self, experiment_id: int) -> None:
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.cursor()
            cur.execute("UPDATE experiments SET deleted = 1 WHERE id = ?", (experiment_id,))
            conn.commit()
