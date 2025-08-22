"""
Checkpoint and resume functionality for optimization runs.
Provides database schema and operations for persistent state management.
"""
import sqlite3
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime

from ..utils.runctx import RunCtx
from ..utils.logging import LogContext


@dataclass
class RunRecord:
    """Database record for optimization runs."""
    id: str
    started_at: str
    finished_at: Optional[str]
    seed: int
    code_version: str
    status: str  # 'running', 'completed', 'failed', 'cancelled'
    params_json: str
    last_batch_idx: int = 0
    total_evaluations: int = 0
    best_score: Optional[float] = None


@dataclass
class BatchRecord:
    """Database record for optimization batches."""
    run_id: str
    batch_idx: int
    started_at: str
    finished_at: Optional[str]
    eval_count: int
    best_J: Optional[float]
    space_json: str
    success_rate: float = 0.0
    avg_duration_ms: float = 0.0


@dataclass
class CandidateRecord:
    """Database record for individual candidates."""
    id: str
    run_id: str
    batch_idx: int
    overlap: float
    orders: int
    params_json: str
    schedule_json: str
    J: Optional[float]
    max_need: Optional[float]
    var_need: Optional[float]
    tail: Optional[float]
    gini: Optional[float]
    entropy: Optional[float]
    penalties_json: str
    evaluation_time_ms: float = 0.0
    fallback_used: bool = False


class CheckpointStore:
    """Database store for checkpointing optimization state."""
    
    def __init__(self, db_path: str = "experiments.db"):
        self.db_path = db_path
        self.log_ctx: Optional[LogContext] = None
        self._ensure_database()
    
    def set_log_context(self, log_ctx: LogContext):
        """Set logging context for database operations."""
        self.log_ctx = log_ctx
    
    def _ensure_database(self):
        """Create database tables if they don't exist."""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            # Runs table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS runs (
                    id TEXT PRIMARY KEY,
                    started_at TEXT NOT NULL,
                    finished_at TEXT,
                    seed INTEGER NOT NULL,
                    code_version TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'running',
                    params_json TEXT NOT NULL,
                    last_batch_idx INTEGER DEFAULT 0,
                    total_evaluations INTEGER DEFAULT 0,
                    best_score REAL
                )
            """)
            
            # Batches table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS batches (
                    run_id TEXT NOT NULL,
                    batch_idx INTEGER NOT NULL,
                    started_at TEXT NOT NULL,
                    finished_at TEXT,
                    eval_count INTEGER NOT NULL,
                    best_J REAL,
                    space_json TEXT NOT NULL,
                    success_rate REAL DEFAULT 0.0,
                    avg_duration_ms REAL DEFAULT 0.0,
                    PRIMARY KEY (run_id, batch_idx),
                    FOREIGN KEY (run_id) REFERENCES runs(id)
                )
            """)
            
            # Candidates table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS candidates (
                    id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    batch_idx INTEGER NOT NULL,
                    overlap REAL NOT NULL,
                    orders INTEGER NOT NULL,
                    params_json TEXT NOT NULL,
                    schedule_json TEXT NOT NULL,
                    J REAL,
                    max_need REAL,
                    var_need REAL,
                    tail REAL,
                    gini REAL,
                    entropy REAL,
                    penalties_json TEXT NOT NULL,
                    evaluation_time_ms REAL DEFAULT 0.0,
                    fallback_used INTEGER DEFAULT 0,
                    FOREIGN KEY (run_id) REFERENCES runs(id)
                )
            """)
            
            # Metrics table for time series data
            conn.execute("""
                CREATE TABLE IF NOT EXISTS metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    batch_idx INTEGER NOT NULL,
                    timestamp TEXT NOT NULL,
                    metric_name TEXT NOT NULL,
                    metric_value REAL NOT NULL,
                    FOREIGN KEY (run_id) REFERENCES runs(id)
                )
            """)
            
            # Logs table for structured log storage
            conn.execute("""
                CREATE TABLE IF NOT EXISTS logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    event TEXT NOT NULL,
                    level TEXT NOT NULL DEFAULT 'INFO',
                    data_json TEXT,
                    FOREIGN KEY (run_id) REFERENCES runs(id)
                )
            """)
            
            # Create indexes for performance
            conn.execute("CREATE INDEX IF NOT EXISTS idx_candidates_run ON candidates(run_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_candidates_batch ON candidates(run_id, batch_idx)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_candidates_score ON candidates(J DESC)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_metrics_run_batch ON metrics(run_id, batch_idx)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_logs_run ON logs(run_id)")
            
            conn.commit()
    
    def start_run(self, run_ctx: RunCtx, params: Dict[str, Any]) -> RunRecord:
        """Start a new optimization run."""
        run_record = RunRecord(
            id=run_ctx.run_id,
            started_at=datetime.utcnow().isoformat(),
            finished_at=None,
            seed=run_ctx.seed,
            code_version=run_ctx.code_version,
            status='running',
            params_json=json.dumps(params, default=str)
        )
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO runs (id, started_at, seed, code_version, status, params_json)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (run_record.id, run_record.started_at, run_record.seed,
                  run_record.code_version, run_record.status, run_record.params_json))
            conn.commit()
        
        if self.log_ctx:
            self.log_ctx.log('run_started', run_id=run_ctx.run_id, 
                           seed=run_ctx.seed, code_version=run_ctx.code_version)
        
        return run_record
    
    def start_batch(self, run_id: str, batch_idx: int, 
                   search_space: Dict[str, Any]) -> BatchRecord:
        """Start a new batch within a run."""
        batch_record = BatchRecord(
            run_id=run_id,
            batch_idx=batch_idx,
            started_at=datetime.utcnow().isoformat(),
            finished_at=None,
            eval_count=0,
            best_J=None,
            space_json=json.dumps(search_space, default=str)
        )
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO batches (run_id, batch_idx, started_at, eval_count, space_json)
                VALUES (?, ?, ?, ?, ?)
            """, (batch_record.run_id, batch_record.batch_idx, 
                  batch_record.started_at, batch_record.eval_count, 
                  batch_record.space_json))
            conn.commit()
        
        if self.log_ctx:
            self.log_ctx.log('batch_started', batch_idx=batch_idx)
        
        return batch_record
    
    def save_candidate(self, candidate_record: CandidateRecord):
        """Save a candidate result to database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO candidates (
                    id, run_id, batch_idx, overlap, orders, params_json, 
                    schedule_json, J, max_need, var_need, tail, gini, entropy,
                    penalties_json, evaluation_time_ms, fallback_used
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                candidate_record.id, candidate_record.run_id, candidate_record.batch_idx,
                candidate_record.overlap, candidate_record.orders, candidate_record.params_json,
                candidate_record.schedule_json, candidate_record.J, candidate_record.max_need,
                candidate_record.var_need, candidate_record.tail, candidate_record.gini,
                candidate_record.entropy, candidate_record.penalties_json,
                candidate_record.evaluation_time_ms, int(candidate_record.fallback_used)
            ))
            conn.commit()
    
    def finish_batch(self, run_id: str, batch_idx: int, 
                    eval_count: int, best_score: Optional[float],
                    success_rate: float, avg_duration_ms: float):
        """Mark a batch as finished with summary statistics."""
        finished_at = datetime.utcnow().isoformat()
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE batches 
                SET finished_at = ?, eval_count = ?, best_J = ?, 
                    success_rate = ?, avg_duration_ms = ?
                WHERE run_id = ? AND batch_idx = ?
            """, (finished_at, eval_count, best_score, success_rate, 
                  avg_duration_ms, run_id, batch_idx))
            
            # Update run's last batch index and best score
            conn.execute("""
                UPDATE runs 
                SET last_batch_idx = ?, total_evaluations = total_evaluations + ?,
                    best_score = CASE 
                        WHEN best_score IS NULL OR ? > best_score 
                        THEN ? ELSE best_score 
                    END
                WHERE id = ?
            """, (batch_idx, eval_count, best_score, best_score, run_id))
            
            conn.commit()
        
        if self.log_ctx:
            self.log_ctx.log('batch_finished', batch_idx=batch_idx, 
                           eval_count=eval_count, best_score=best_score,
                           success_rate=success_rate)
    
    def finish_run(self, run_id: str, status: str = 'completed'):
        """Mark a run as finished."""
        finished_at = datetime.utcnow().isoformat()
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE runs SET finished_at = ?, status = ? WHERE id = ?
            """, (finished_at, status, run_id))
            conn.commit()
        
        if self.log_ctx:
            self.log_ctx.log('run_finished', status=status)
    
    def get_resumable_runs(self) -> List[RunRecord]:
        """Get runs that can be resumed (status='running')."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT * FROM runs WHERE status = 'running' ORDER BY started_at DESC
            """)
            
            runs = []
            for row in cursor.fetchall():
                runs.append(RunRecord(**dict(row)))
            
            return runs
    
    def get_run_progress(self, run_id: str) -> Dict[str, Any]:
        """Get progress information for a run."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            
            # Get run info
            run_row = conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
            if not run_row:
                return {}
            
            # Get batch count and evaluations
            batch_stats = conn.execute("""
                SELECT COUNT(*) as batch_count, 
                       COALESCE(SUM(eval_count), 0) as total_evals,
                       MAX(best_J) as best_score
                FROM batches WHERE run_id = ?
            """, (run_id,)).fetchone()
            
            # Get recent candidates
            recent_candidates = conn.execute("""
                SELECT id, J, overlap, orders, evaluation_time_ms, fallback_used
                FROM candidates 
                WHERE run_id = ? 
                ORDER BY ROWID DESC 
                LIMIT 10
            """, (run_id,)).fetchall()
            
            return {
                'run_info': dict(run_row),
                'batch_count': batch_stats['batch_count'] if batch_stats else 0,
                'total_evaluations': batch_stats['total_evals'] if batch_stats else 0,
                'best_score': batch_stats['best_score'] if batch_stats else None,
                'recent_candidates': [dict(row) for row in recent_candidates]
            }
    
    def get_best_candidates(self, run_id: str, limit: int = 10) -> List[CandidateRecord]:
        """Get best candidates from a run."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT * FROM candidates 
                WHERE run_id = ? AND J IS NOT NULL
                ORDER BY J ASC 
                LIMIT ?
            """, (run_id, limit))
            
            candidates = []
            for row in cursor.fetchall():
                candidates.append(CandidateRecord(**dict(row)))
            
            return candidates
    
    def log_metric(self, run_id: str, batch_idx: int, 
                  metric_name: str, metric_value: float):
        """Log a time series metric."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO metrics (run_id, batch_idx, timestamp, metric_name, metric_value)
                VALUES (?, ?, ?, ?, ?)
            """, (run_id, batch_idx, datetime.utcnow().isoformat(), 
                  metric_name, metric_value))
            conn.commit()
    
    def get_metrics(self, run_id: str, metric_name: str) -> List[Tuple[str, float]]:
        """Get time series data for a metric."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT timestamp, metric_value 
                FROM metrics 
                WHERE run_id = ? AND metric_name = ?
                ORDER BY timestamp
            """, (run_id, metric_name))
            
            return cursor.fetchall()
    
    def cleanup_old_runs(self, keep_days: int = 30):
        """Clean up old completed runs."""
        cutoff_date = datetime.utcnow().replace(day=datetime.utcnow().day - keep_days).isoformat()
        
        with sqlite3.connect(self.db_path) as conn:
            # Get run IDs to delete
            cursor = conn.execute("""
                SELECT id FROM runs 
                WHERE status IN ('completed', 'failed', 'cancelled') 
                AND finished_at < ?
            """, (cutoff_date,))
            
            run_ids = [row[0] for row in cursor.fetchall()]
            
            if run_ids:
                placeholders = ','.join('?' * len(run_ids))
                
                # Delete related records
                conn.execute(f"DELETE FROM logs WHERE run_id IN ({placeholders})", run_ids)
                conn.execute(f"DELETE FROM metrics WHERE run_id IN ({placeholders})", run_ids)
                conn.execute(f"DELETE FROM candidates WHERE run_id IN ({placeholders})", run_ids)
                conn.execute(f"DELETE FROM batches WHERE run_id IN ({placeholders})", run_ids)
                conn.execute(f"DELETE FROM runs WHERE id IN ({placeholders})", run_ids)
                
                conn.commit()
                
                if self.log_ctx:
                    self.log_ctx.log('cleanup_completed', deleted_runs=len(run_ids))
        
        return len(run_ids) if 'run_ids' in locals() else 0
