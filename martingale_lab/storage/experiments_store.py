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
from typing import Any, Dict, List, Optional
import os
from pathlib import Path
import hashlib
from typing import Iterable, Sequence, Tuple

from martingale_lab.utils.logging import db_logger as logger

# Status constants
class Status:
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


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
        logger.info(
            f"Initializing database at {db_path}",
            extra={"event": "mlab.db.init", "db_path": db_path}
        )
        
        self._init_db()
        self.migrate_if_needed()

    def _get_user_version(self, cur) -> int:
        cur.execute("PRAGMA user_version;")
        row = cur.fetchone()
        try:
            return int(row[0]) if row and row[0] is not None else 0
        except Exception:
            return 0

    def _set_user_version(self, cur, v: int) -> None:
        cur.execute(f"PRAGMA user_version={int(v)};")

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

    def migrate_if_needed(self) -> None:
        """Run PRAGMA user_version-based idempotent migrations."""
        with sqlite3.connect(self.db_path) as con:
            cur = con.cursor()
            # Enable WAL for better concurrency
            try:
                cur.execute("PRAGMA journal_mode=WAL;")
            except Exception:
                pass

            v = self._get_user_version(cur)

            # v0 -> v1 : results.stable_id + UNIQUE(experiment_id, stable_id)
            if v < 1:
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS results(
                      id INTEGER PRIMARY KEY AUTOINCREMENT,
                      experiment_id INTEGER NOT NULL REFERENCES experiments(id),
                      score REAL NOT NULL,
                      payload_json TEXT NOT NULL,
                      sanity_json TEXT NOT NULL,
                      diagnostics_json TEXT NOT NULL,
                      penalties_json TEXT NOT NULL,
                      created_at TEXT NOT NULL DEFAULT (datetime('now'))
                    );
                    """
                )
                # add column if missing
                cur.execute("PRAGMA table_info(results);")
                cols = {r[1] for r in cur.fetchall()}
                if "stable_id" not in cols:
                    cur.execute("ALTER TABLE results ADD COLUMN stable_id TEXT;")
                # unique index
                cur.execute(
                    """
                    CREATE UNIQUE INDEX IF NOT EXISTS
                    ux_results_exp_stable ON results(experiment_id, stable_id);
                    """
                )
                self._set_user_version(cur, 1)

            # v1 -> v2 : experiments.run_id UNIQUE
            v = self._get_user_version(cur)
            if v < 2:
                cur.execute(
                    """
                    CREATE UNIQUE INDEX IF NOT EXISTS
                    ux_experiments_run_id ON experiments(run_id);
                    """
                )
                self._set_user_version(cur, 2)

            con.commit()

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
                f"Created experiment {exp_id} for {adapter}",
                extra={"event": "mlab.db.upsert_exp", "exp_id": int(exp_id), "adapter": adapter, "run_id": run_id}
            )
            
            return int(exp_id)

    def update_experiment_summary(self, experiment_id: int, best_score: float, total_evals: int, elapsed_s: float) -> None:
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.cursor()
            
            # Check which columns exist
            cur.execute("PRAGMA table_info(experiments)")
            cols = {row[1] for row in cur.fetchall()}
            
            # Build update fragments based on existing columns
            set_fragments = []
            params: List[Any] = []
            
            if "best_score" in cols:
                set_fragments.append("best_score = MIN(best_score, ?)")
                params.append(best_score)
            
            if "eval_count" in cols:
                set_fragments.append("eval_count = ?")
                params.append(total_evals)
            
            if "total_evals" in cols:
                set_fragments.append("total_evals = ?")
                params.append(total_evals)
            
            if "finished_at" in cols:
                set_fragments.append("finished_at = ?")
                params.append(datetime.now().isoformat())
            
            if "status" in cols:
                set_fragments.append("status = ?")
                params.append(Status.COMPLETED)
            
            if "updated_at" in cols:
                set_fragments.append("updated_at = ?")
                params.append(datetime.now().isoformat())

            if set_fragments:
                query = f"UPDATE experiments SET {', '.join(set_fragments)} WHERE id = ?"
                params.append(experiment_id)
                cur.execute(query, tuple(params))
                conn.commit()

    def set_experiment_error(self, experiment_id: int, error_payload: Dict[str, Any]) -> None:
        """Persist error details to experiments.error_json."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cur = conn.cursor()
                cur.execute(
                    "UPDATE experiments SET error_json = ?, updated_at = ?, status = ? WHERE id = ?",
                    (
                        json.dumps(_jsonify(error_payload), separators=(",", ":"), ensure_ascii=False),
                        datetime.now().isoformat(),
                        Status.FAILED,
                        experiment_id,
                    ),
                )
                conn.commit()
        except Exception as e:
            # Best-effort; do not raise
            pass

    def _stable_id_from_payload(self, payload: Dict[str, Any]) -> str:
        # Düzenli sıralı json + SHA1
        try:
            blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        except Exception:
            # Fallback to generic serialization
            blob = json.dumps(_jsonify(payload), sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha1(blob).hexdigest()

    def upsert_results(
        self,
        experiment_id: int,
        items: Iterable[Dict[str, Any]],
    ) -> int:
        rows: Sequence[Tuple] = []
        for it in items:
            # Prefer provided stable_id, else derive from payload or full item
            stable_id = it.get("stable_id") or self._stable_id_from_payload(it.get("payload", it))
            payload_obj = it.get("payload", it)
            rows.append((
                experiment_id,
                stable_id,
                float(it.get("score", float("inf"))),
                json.dumps(_jsonify(payload_obj), separators=(",", ":")),
                json.dumps(_jsonify(it.get("sanity", {})), separators=(",", ":")),
                json.dumps(_jsonify(it.get("diagnostics", {})), separators=(",", ":")),
                json.dumps(_jsonify(it.get("penalties", {})), separators=(",", ":")),
            ))

        if not rows:
            return 0

        with sqlite3.connect(self.db_path) as con:
            cur = con.cursor()
            cur.executemany(
                """
                INSERT INTO results(
                    experiment_id, stable_id, score,
                    payload_json, sanity_json, diagnostics_json, penalties_json
                )
                VALUES(?,?,?,?,?,?,?)
                ON CONFLICT(experiment_id, stable_id) DO UPDATE SET
                   score=excluded.score,
                   payload_json=excluded.payload_json,
                   sanity_json=excluded.sanity_json,
                   diagnostics_json=excluded.diagnostics_json,
                   penalties_json=excluded.penalties_json;
                """,
                rows,
            )
            con.commit()
            return cur.rowcount

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

    def get_results(self, experiment_id: int, limit: int = 1000, order_by: str = 'score') -> List[Dict[str, Any]]:
        """
        Get results for an experiment with enhanced fields for results page.
        
        Returns:
            List of result dictionaries with normalized fields
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cur = conn.cursor()
                
                # Get results with all required fields
                query = """
                SELECT 
                    r.id, r.score, r.payload_json, r.sanity_json, r.diagnostics_json, r.penalties_json,
                    r.created_at
                FROM results r 
                WHERE r.experiment_id = ? 
                ORDER BY r.score ASC 
                LIMIT ?
                """
                
                cur.execute(query, (experiment_id, limit))
                rows = cur.fetchall()
                
                results = []
                for row in rows:
                    try:
                        # Parse JSON fields
                        payload = json.loads(row[2]) if row[2] else {}
                        sanity = json.loads(row[3]) if row[3] else {}
                        diagnostics = json.loads(row[4]) if row[4] else {}
                        penalties = json.loads(row[5]) if row[5] else {}
                        
                        # Extract schedule data
                        schedule = payload.get('schedule', {})
                        needpct = schedule.get('needpct', [])
                        volume_pct = schedule.get('volume_pct', [])
                        indent_pct = schedule.get('indent_pct', [])
                        martingale_pct = schedule.get('martingale_pct', [])
                        
                        # Calculate derived metrics
                        max_need = max(needpct) if needpct else 0.0
                        var_need = np.var(needpct) if len(needpct) > 1 else 0.0
                        tail = volume_pct[-1] if volume_pct else 0.0
                        
                        # Extract overlap and orders from schedule
                        overlap_pct = schedule.get('overlap_pct', 0.0)
                        orders = len(volume_pct) if volume_pct else 0
                        
                        result = {
                            "id": row[0],
                            "score": float(row[1]),
                            "max_need": float(max_need),
                            "var_need": float(var_need),
                            "tail": float(tail),
                            "overlap_pct": float(overlap_pct),
                            "orders": int(orders),
                            "needpct": needpct,
                            "volume_pct": volume_pct,
                            "indent_pct": indent_pct,
                            "martingale_pct": martingale_pct,
                            "sanity": sanity,
                            "diagnostics": diagnostics,
                            "penalties": penalties,
                            "created_at": row[6],
                            "_raw": {
                                "id": row[0],
                                "score": float(row[1]),
                                "payload": payload,
                                "sanity": sanity,
                                "diagnostics": diagnostics,
                                "penalties": penalties,
                                "created_at": row[6]
                            }
                        }
                        results.append(result)
                        
                    except Exception as e:
                        logger.warning(
                            f"Failed to parse result row {row[0]}: {str(e)}",
                            extra={"event": "mlab.db.verify", "exp_id": experiment_id, "result_id": row[0], "error": str(e)}
                        )
                        continue
                
                logger.info(
                    f"Retrieved {len(results)} results for experiment {experiment_id}",
                    extra={"event": "mlab.db.verify", "exp_id": experiment_id, "result_count": len(results)}
                )
                
                return results
                
        except Exception as e:
            logger.error(
                f"Failed to get results for experiment {experiment_id}: {str(e)}",
                extra={"event": "mlab.db.error", "exp_id": experiment_id, "error": str(e)}
            )
            return []

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
                    MIN(score) as best_score
                FROM results 
                WHERE experiment_id = ?
                """,
                (experiment_id,)
            )
            stats_row = cur.fetchone()
            
            # Map fields defensively given evolving schema
            # Use column names instead of indices when possible
            cur.execute("PRAGMA table_info(experiments)")
            cols = [r[1] for r in cur.fetchall()]
            col_index = {name: idx for idx, name in enumerate(cols)}

            return {
                "id": exp_row[0],
                "created_at": exp_row[col_index.get("created_at", 1)],
                "adapter": exp_row[col_index.get("adapter", 2)],
                "best_score": exp_row[col_index.get("best_score", 3)],
                "total_evals": exp_row[col_index.get("total_evals", col_index.get("eval_count", 4))],
                "status": exp_row[col_index.get("status", 6)] if len(exp_row) > 6 else "PENDING",
                "config_json": exp_row[col_index.get("config_json", 3)] if len(exp_row) > 3 else None,
                "statistics": {
                    "total_results": stats_row[0] if stats_row else 0,
                    "best_score": stats_row[1] if stats_row else float("inf"),
                }
            }

    def soft_delete_experiment(self, experiment_id: int) -> None:
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.cursor()
            cur.execute("UPDATE experiments SET deleted = 1 WHERE id = ?", (experiment_id,))
            conn.commit()

    def get_experiments(self) -> List[Dict[str, Any]]:
        """Get all experiments with enhanced fields."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cur = conn.cursor()
                
                # Get experiments with all required fields, handling both total_evals and eval_count
                query = """
                SELECT 
                    e.id, e.created_at, e.adapter, e.best_score, 
                    COALESCE(e.total_evals, e.eval_count, 0) as total_evals,
                    e.status, e.config_json, e.notes, e.run_id
                FROM experiments e 
                WHERE e.deleted = 0 
                ORDER BY e.created_at DESC
                """
                
                cur.execute(query)
                rows = cur.fetchall()
                
                experiments = []
                for row in rows:
                    try:
                        experiment = {
                            "id": row[0],
                            "created_at": row[1],
                            "adapter": row[2],
                            "best_score": float(row[3]) if row[3] is not None else float('inf'),
                            "total_evals": int(row[4]) if row[4] is not None else 0,
                            "status": row[5] if row[5] else "UNKNOWN",
                            "config_json": row[6],
                            "notes": row[7],
                            "run_id": row[8]
                        }
                        experiments.append(experiment)
                        
                    except Exception as e:
                        logger.warning(
                            f"Failed to parse experiment row {row[0]}: {str(e)}",
                            extra={"event": "mlab.db.verify", "error": str(e)}
                        )
                        continue
                
                logger.info(
                    f"Retrieved {len(experiments)} experiments",
                    extra={"event": "mlab.db.verify", "experiment_count": len(experiments)}
                )
                
                return experiments
                
        except Exception as e:
            logger.error(
                f"Failed to get experiments: {str(e)}",
                extra={"event": "mlab.db.error", "error": str(e)}
            )
            return []
