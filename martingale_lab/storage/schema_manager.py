"""
Centralized Database Schema Manager with Migration Support
Handles all table creation, versioning, and migrations for the entire system.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Optional, List, Dict, Any, Callable
from datetime import datetime
import logging
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class SchemaManager:
    """
    Centralized schema management with versioned migrations.
    
    This replaces scattered CREATE TABLE statements across:
    - experiments_store.py
    - checkpoint_store.py  
    - sqlite_store.py
    """
    
    CURRENT_VERSION = 3  # Current schema version
    
    def __init__(self, db_path: str = "db_results/experiments.db"):
        self.db_path = db_path
        self._ensure_directory()
        self._init_schema()
    
    def _ensure_directory(self):
        """Ensure database directory exists."""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
    
    @contextmanager
    def get_connection(self):
        """Context manager for database connections with proper cleanup."""
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row  # Enable column access by name
            yield conn
            conn.commit()
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            if conn:
                conn.close()
    
    def _init_schema(self):
        """Initialize schema and run migrations."""
        with self.get_connection() as conn:
            # Enable WAL mode for better concurrency
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            
            # Create schema version table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS schema_version (
                    version INTEGER PRIMARY KEY,
                    applied_at TEXT NOT NULL,
                    description TEXT
                )
            """)
            
            # Get current version
            current = self._get_current_version(conn)
            
            # Run migrations
            self._run_migrations(conn, current)
    
    def _get_current_version(self, conn: sqlite3.Connection) -> int:
        """Get current schema version."""
        cursor = conn.execute(
            "SELECT MAX(version) FROM schema_version"
        )
        result = cursor.fetchone()
        return result[0] if result[0] is not None else 0
    
    def _run_migrations(self, conn: sqlite3.Connection, from_version: int):
        """Run all migrations from current version to latest."""
        migrations = self._get_migrations()
        
        for version, (description, migration_func) in migrations.items():
            if version > from_version:
                logger.info(f"Running migration {version}: {description}")
                try:
                    migration_func(conn)
                    conn.execute(
                        "INSERT INTO schema_version (version, applied_at, description) VALUES (?, ?, ?)",
                        (version, datetime.utcnow().isoformat(), description)
                    )
                    conn.commit()
                    logger.info(f"Migration {version} completed successfully")
                except Exception as e:
                    logger.error(f"Migration {version} failed: {e}")
                    conn.rollback()
                    raise
    
    def _get_migrations(self) -> Dict[int, tuple[str, Callable]]:
        """Get all migration definitions."""
        return {
            1: ("Create core tables", self._migration_v1),
            2: ("Add checkpoint tables", self._migration_v2),
            3: ("Add indexes and constraints", self._migration_v3),
        }
    
    def _migration_v1(self, conn: sqlite3.Connection):
        """Version 1: Core tables for experiments and results."""
        
        # Experiments table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS experiments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT UNIQUE NOT NULL,
                orchestrator TEXT NOT NULL,
                config_json TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'PENDING',
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                started_at TEXT,
                finished_at TEXT,
                error_json TEXT,
                best_score REAL,
                total_evaluations INTEGER DEFAULT 0,
                notes TEXT
            )
        """)
        
        # Results table (unified from all stores)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id INTEGER NOT NULL,
                stable_id TEXT NOT NULL,
                score REAL NOT NULL,
                max_need REAL,
                var_need REAL,
                tail REAL,
                payload_json TEXT NOT NULL,
                schedule_json TEXT NOT NULL,
                sanity_json TEXT,
                diagnostics_json TEXT,
                penalties_json TEXT,
                params_json TEXT,
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                FOREIGN KEY (experiment_id) REFERENCES experiments(id) ON DELETE CASCADE,
                UNIQUE(experiment_id, stable_id)
            )
        """)
        
        # Create indexes for v1
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_results_exp_score 
            ON results(experiment_id, score)
        """)
        
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_experiments_run_id 
            ON experiments(run_id)
        """)
    
    def _migration_v2(self, conn: sqlite3.Connection):
        """Version 2: Add checkpoint and monitoring tables."""
        
        # Runs table (for checkpoint recovery)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS runs (
                id TEXT PRIMARY KEY,
                experiment_id INTEGER,
                started_at TEXT NOT NULL,
                finished_at TEXT,
                last_checkpoint_at TEXT,
                seed INTEGER NOT NULL,
                code_version TEXT,
                status TEXT NOT NULL DEFAULT 'running',
                params_json TEXT NOT NULL,
                last_batch_idx INTEGER DEFAULT 0,
                total_evaluations INTEGER DEFAULT 0,
                best_score REAL,
                checkpoint_data BLOB,
                FOREIGN KEY (experiment_id) REFERENCES experiments(id)
            )
        """)
        
        # Batches table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS batches (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                batch_idx INTEGER NOT NULL,
                started_at TEXT NOT NULL,
                finished_at TEXT,
                eval_count INTEGER NOT NULL,
                best_score REAL,
                avg_score REAL,
                space_json TEXT,
                success_rate REAL DEFAULT 0.0,
                avg_duration_ms REAL DEFAULT 0.0,
                FOREIGN KEY (run_id) REFERENCES runs(id) ON DELETE CASCADE,
                UNIQUE(run_id, batch_idx)
            )
        """)
        
        # Metrics table for time series
        conn.execute("""
            CREATE TABLE IF NOT EXISTS metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                batch_idx INTEGER,
                timestamp TEXT NOT NULL,
                metric_name TEXT NOT NULL,
                metric_value REAL NOT NULL,
                metric_json TEXT,
                FOREIGN KEY (run_id) REFERENCES runs(id) ON DELETE CASCADE
            )
        """)
        
        # Logs table for structured logging
        conn.execute("""
            CREATE TABLE IF NOT EXISTS logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT,
                experiment_id INTEGER,
                timestamp TEXT NOT NULL DEFAULT (datetime('now')),
                level TEXT NOT NULL,
                event TEXT NOT NULL,
                message TEXT,
                data_json TEXT,
                FOREIGN KEY (run_id) REFERENCES runs(id) ON DELETE CASCADE,
                FOREIGN KEY (experiment_id) REFERENCES experiments(id) ON DELETE CASCADE
            )
        """)
        
        # Create indexes for v2
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_batches_run_batch 
            ON batches(run_id, batch_idx)
        """)
        
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_metrics_run_name 
            ON metrics(run_id, metric_name)
        """)
        
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_logs_timestamp 
            ON logs(timestamp)
        """)
    
    def _migration_v3(self, conn: sqlite3.Connection):
        """Version 3: Add performance indexes and constraints."""
        
        # Add composite indexes for common queries
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_results_score_filtered 
            ON results(experiment_id, score) 
            WHERE score > 0 AND score < 1000000
        """)
        
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_experiments_status_time 
            ON experiments(status, created_at)
        """)
        
        # Add checkpoint recovery index
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_runs_checkpoint 
            ON runs(status, last_checkpoint_at) 
            WHERE status = 'running'
        """)
        
        # Add trigger to update experiment status
        conn.execute("""
            CREATE TRIGGER IF NOT EXISTS update_experiment_finished
            AFTER UPDATE OF status ON runs
            WHEN NEW.status IN ('completed', 'failed')
            BEGIN
                UPDATE experiments 
                SET status = NEW.status,
                    finished_at = datetime('now')
                WHERE id = NEW.experiment_id;
            END
        """)
        
        # Add trigger to update best score
        conn.execute("""
            CREATE TRIGGER IF NOT EXISTS update_best_score
            AFTER INSERT ON results
            BEGIN
                UPDATE experiments 
                SET best_score = (
                    SELECT MIN(score) FROM results 
                    WHERE experiment_id = NEW.experiment_id 
                    AND score > 0
                )
                WHERE id = NEW.experiment_id;
                
                UPDATE runs
                SET best_score = (
                    SELECT MIN(score) FROM results 
                    WHERE experiment_id = NEW.experiment_id 
                    AND score > 0
                )
                WHERE experiment_id = NEW.experiment_id;
            END
        """)
    
    def drop_legacy_tables(self, conn: Optional[sqlite3.Connection] = None):
        """Drop legacy tables from old schema (use with caution!)."""
        legacy_tables = [
            'optimization_results',  # from sqlite_store
            'optimization_traces',   # from sqlite_store
            'candidates',            # from checkpoint_store (replaced by results)
        ]
        
        if conn is None:
            with self.get_connection() as conn:
                for table in legacy_tables:
                    try:
                        conn.execute(f"DROP TABLE IF EXISTS {table}")
                        logger.info(f"Dropped legacy table: {table}")
                    except Exception as e:
                        logger.warning(f"Could not drop table {table}: {e}")
        else:
            for table in legacy_tables:
                try:
                    conn.execute(f"DROP TABLE IF EXISTS {table}")
                    logger.info(f"Dropped legacy table: {table}")
                except Exception as e:
                    logger.warning(f"Could not drop table {table}: {e}")
    
    def get_schema_info(self) -> Dict[str, Any]:
        """Get information about current schema."""
        with self.get_connection() as conn:
            version = self._get_current_version(conn)
            
            # Get table list
            cursor = conn.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' 
                ORDER BY name
            """)
            tables = [row[0] for row in cursor.fetchall()]
            
            # Get migration history
            cursor = conn.execute("""
                SELECT version, applied_at, description 
                FROM schema_version 
                ORDER BY version
            """)
            migrations = [
                {"version": row[0], "applied_at": row[1], "description": row[2]}
                for row in cursor.fetchall()
            ]
            
            return {
                "current_version": version,
                "target_version": self.CURRENT_VERSION,
                "tables": tables,
                "migrations": migrations,
                "db_path": self.db_path
            }
    
    def validate_schema(self) -> tuple[bool, List[str]]:
        """Validate that schema is correct and complete."""
        issues = []
        
        with self.get_connection() as conn:
            version = self._get_current_version(conn)
            
            if version < self.CURRENT_VERSION:
                issues.append(f"Schema version {version} is behind current {self.CURRENT_VERSION}")
            
            # Check required tables exist
            required_tables = [
                'experiments', 'results', 'runs', 'batches', 
                'metrics', 'logs', 'schema_version'
            ]
            
            cursor = conn.execute("""
                SELECT name FROM sqlite_master WHERE type='table'
            """)
            existing_tables = {row[0] for row in cursor.fetchall()}
            
            for table in required_tables:
                if table not in existing_tables:
                    issues.append(f"Required table '{table}' is missing")
            
            # Check for legacy tables that should be removed
            legacy_tables = ['optimization_results', 'optimization_traces', 'candidates']
            for table in legacy_tables:
                if table in existing_tables:
                    issues.append(f"Legacy table '{table}' still exists")
        
        return len(issues) == 0, issues


# Singleton instance
_schema_manager: Optional[SchemaManager] = None


def get_schema_manager(db_path: str = "db_results/experiments.db") -> SchemaManager:
    """Get or create singleton schema manager."""
    global _schema_manager
    if _schema_manager is None or _schema_manager.db_path != db_path:
        _schema_manager = SchemaManager(db_path)
    return _schema_manager