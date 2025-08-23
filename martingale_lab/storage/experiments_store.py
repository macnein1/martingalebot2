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
import os
from pathlib import Path

from martingale_lab.utils.structured_logging import (
    get_structured_logger, EventNames, Timer, ensure_json_serializable
)
from ui.utils.constants import Status, DB_PATH

# Initialize structured logger for database operations
logger = get_structured_logger("mlab.db")


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
        self.db_path = db_path
        # Ensure directory exists
        Path(os.path.dirname(self.db_path) or ".").mkdir(parents=True, exist_ok=True)
        
        # Log database initialization
        logger.info(EventNames.DB_INIT, f"Initializing database at {db_path}", db_path=db_path)
        
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.cursor()
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS experiments (
                  id INTEGER PRIMARY KEY,
                  run_id TEXT NOT NULL,
                  adapter TEXT NOT NULL,
                  config_json TEXT NOT NULL,
                  started_at TEXT NOT NULL,
                  finished_at TEXT,
                  status TEXT NOT NULL DEFAULT 'PENDING',
                  best_score REAL NOT NULL,
                  eval_count INTEGER NOT NULL DEFAULT 0,
                  notes TEXT,
                  created_at TEXT NOT NULL,
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
                  score REAL NOT NULL,
                  payload_json TEXT NOT NULL,
                  sanity_json TEXT NOT NULL,
                  diagnostics_json TEXT NOT NULL,
                  penalties_json TEXT NOT NULL,
                  created_at TEXT NOT NULL
                );
                """
            )
            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_results_exp ON results(experiment_id, score);
                """
            )
            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_experiments_run ON experiments(run_id);
                """
            )
            conn.commit()

    def create_experiment(self, adapter: str, cfg: Dict[str, Any], run_id: str) -> int:
        created_at = datetime.now().isoformat()
        
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO experiments (
                  run_id, adapter, config_json, started_at, status, best_score, eval_count, notes, created_at, deleted
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    adapter,
                    json.dumps(_jsonify(cfg), separators=(",", ":"), ensure_ascii=False),
                    created_at,
                    Status.RUNNING,
                    float("inf"),  # placeholder
                    0,
                    cfg.get("notes"),
                    created_at,
                    0,
                ),
            )
            exp_id = cur.lastrowid
            conn.commit()
            
            # Log experiment creation
            logger.info(
                EventNames.DB_UPSERT_EXP,
                f"Created experiment {exp_id} for {adapter}",
                exp_id=int(exp_id),
                adapter=adapter,
                run_id=run_id
            )
            
            return int(exp_id)

    def update_experiment_summary(self, experiment_id: int, best_score: float, total_evals: int, elapsed_s: float) -> None:
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.cursor()
            cur.execute(
                """
                UPDATE experiments
                SET best_score = MIN(best_score, ?),
                    eval_count = ?,
                    finished_at = ?,
                    status = ?
                WHERE id = ?
                """,
                (best_score, total_evals, datetime.now().isoformat(), Status.COMPLETED, experiment_id),
            )
            conn.commit()

    def upsert_results(self, experiment_id: int, items: List[Dict[str, Any]]) -> int:
        """
        Upsert results with complete DCA evaluation contract structure.
        Each item should contain: score, schedule, sanity, diagnostics, penalties.
        """
        if not items:
            return 0
        now = datetime.now().isoformat()
        inserted = 0
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cur = conn.cursor()
                for item in items:
                    cur.execute(
                        """
                        INSERT INTO results (
                            experiment_id, score, payload_json, sanity_json, 
                            diagnostics_json, penalties_json, created_at
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            experiment_id,
                            float(item.get("score", float("inf"))),
                            json.dumps(_jsonify(item), separators=(",", ":"), ensure_ascii=False),
                            json.dumps(_jsonify(item.get("sanity", {})), separators=(",", ":"), ensure_ascii=False),
                            json.dumps(_jsonify(item.get("diagnostics", {})), separators=(",", ":"), ensure_ascii=False),
                            json.dumps(_jsonify(item.get("penalties", {})), separators=(",", ":"), ensure_ascii=False),
                            now,
                        ),
                    )
                    inserted += 1
                conn.commit()
                
                # Verify insertion
                cur.execute("SELECT COUNT(*) FROM results WHERE experiment_id = ?", (experiment_id,))
                count = cur.fetchone()[0]
                
                logger.info(
                    EventNames.DB_UPSERT_RES,
                    f"Inserted {inserted} results for experiment {experiment_id}",
                    experiment_id=experiment_id,
                    rows=inserted,
                    total_rows=count
                )
                
                logger.info(
                    EventNames.DB_VERIFY,
                    f"Verification successful: {count} total rows",
                    ok=True,
                    expected=inserted,
                    actual=count
                )
                
        except Exception as e:
            logger.error(
                EventNames.DB_ERROR,
                f"Database error in upsert_results: {str(e)}",
                error=str(e),
                operation="upsert_results",
                experiment_id=experiment_id
            )
            raise
            
        return inserted

    def get_top_results(self, 
                       experiment_id: Optional[int] = None,
                       limit: int = 100,
                       filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Get top results with optional filtering."""
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.cursor()
            
            query = """
                SELECT r.id, r.experiment_id, r.score, r.payload_json, r.sanity_json, 
                       r.diagnostics_json, r.penalties_json, r.created_at,
                       e.adapter, e.created_at as exp_created_at
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
                    
            query += " ORDER BY r.score ASC LIMIT ?"
            params.append(limit)
            
            cur.execute(query, params)
            rows = cur.fetchall()
            
            results = []
            for row in rows:
                # Parse JSON fields
                try:
                    payload = json.loads(row[3]) if row[3] else {}
                    sanity = json.loads(row[4]) if row[4] else {}
                    diagnostics = json.loads(row[5]) if row[5] else {}
                    penalties = json.loads(row[6]) if row[6] else {}
                    
                    result = {
                        "id": row[0],
                        "experiment_id": row[1],
                        "score": row[2],
                        "payload": payload,
                        "sanity": sanity,
                        "diagnostics": diagnostics,
                        "penalties": penalties,
                        "created_at": row[7],
                        "adapter": row[8],
                        "exp_created_at": row[9],
                    }
                    results.append(result)
                except Exception as e:
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
