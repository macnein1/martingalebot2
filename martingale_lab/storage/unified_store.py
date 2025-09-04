"""
Unified Store with Proper Error Recovery and Resource Management
Replaces experiments_store, checkpoint_store, and sqlite_store with a single, robust implementation.
"""
from __future__ import annotations

import json
import sqlite3
import pickle
from contextlib import contextmanager
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Iterator
import logging
import hashlib
from threading import Lock

from .schema_manager import get_schema_manager
from .memory_manager import BoundedBestCandidates, BatchAccumulator
from .config_store import ConfigStore
from .write_queue_manager import WriteQueueManager, get_write_queue, WriteOperation

logger = logging.getLogger(__name__)


class UnifiedStore:
    """
    Unified database store with proper resource management and error recovery.
    
    Features:
    - Context managers for all database operations
    - Automatic cleanup on errors
    - Transaction management
    - Connection pooling
    - Thread-safe operations
    """
    
    def __init__(self, db_path: str = "db_results/experiments.db", 
                 max_candidates_memory: int = 500,
                 use_write_queue: bool = True):
        """
        Initialize unified store.
        
        Args:
            db_path: Path to SQLite database
            max_candidates_memory: Maximum candidates to keep in memory
            use_write_queue: Whether to use write queue for better concurrency
        """
        self.db_path = db_path
        self._lock = Lock()  # Thread safety
        self._schema_manager = get_schema_manager(db_path)
        
        # Memory management
        self._candidates_cache = BoundedBestCandidates(max_size=max_candidates_memory)
        self._batch_accumulator = BatchAccumulator(batch_size=100)
        
        # Config store
        self.config_store = ConfigStore(db_path)
        
        # Write queue for concurrent safety
        self.use_write_queue = use_write_queue
        if use_write_queue:
            self._write_queue = get_write_queue(
                db_path,
                batch_size=500,
                flush_interval_ms=100
            )
        else:
            self._write_queue = None
        
        # Connection pool (simple implementation)
        self._connection_pool: List[sqlite3.Connection] = []
        self._max_connections = 5
        
        logger.info(f"UnifiedStore initialized: {db_path}")
    
    @contextmanager
    def transaction(self, isolation_level: str = "DEFERRED"):
        """
        Context manager for database transactions with automatic rollback.
        
        Args:
            isolation_level: DEFERRED, IMMEDIATE, or EXCLUSIVE
            
        Example:
            with store.transaction():
                store.insert_result(...)
                store.update_experiment(...)
                # Automatically commits on success, rolls back on error
        """
        conn = None
        cursor = None
        
        try:
            conn = self._get_connection()
            conn.isolation_level = None  # Manual transaction control
            cursor = conn.cursor()
            cursor.execute(f"BEGIN {isolation_level}")
            
            yield cursor
            
            conn.commit()
            logger.debug("Transaction committed successfully")
            
        except Exception as e:
            if conn:
                try:
                    conn.rollback()
                    logger.warning(f"Transaction rolled back due to: {e}")
                except Exception as rollback_error:
                    logger.error(f"Rollback failed: {rollback_error}")
            raise
            
        finally:
            if cursor:
                cursor.close()
            if conn:
                self._return_connection(conn)
    
    @contextmanager
    def read_connection(self):
        """
        Context manager for read-only database access.
        
        Example:
            with store.read_connection() as cursor:
                cursor.execute("SELECT * FROM experiments")
                results = cursor.fetchall()
        """
        conn = None
        cursor = None
        
        try:
            conn = self._get_connection()
            conn.row_factory = sqlite3.Row  # Enable column name access
            cursor = conn.cursor()
            
            yield cursor
            
        except Exception as e:
            logger.error(f"Read operation failed: {e}")
            raise
            
        finally:
            if cursor:
                cursor.close()
            if conn:
                self._return_connection(conn)
    
    def _get_connection(self) -> sqlite3.Connection:
        """Get a connection from the pool or create new one."""
        with self._lock:
            if self._connection_pool:
                conn = self._connection_pool.pop()
                # Test if connection is still alive
                try:
                    conn.execute("SELECT 1")
                    return conn
                except:
                    # Connection is dead, create new one
                    pass
            
            # Create new connection
            conn = sqlite3.connect(self.db_path)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            return conn
    
    def _return_connection(self, conn: sqlite3.Connection):
        """Return connection to pool or close if pool is full."""
        with self._lock:
            if len(self._connection_pool) < self._max_connections:
                self._connection_pool.append(conn)
            else:
                conn.close()
    
    def close_all_connections(self):
        """Close all pooled connections (call on shutdown)."""
        with self._lock:
            while self._connection_pool:
                conn = self._connection_pool.pop()
                try:
                    conn.close()
                except:
                    pass
    
    # === Experiment Management ===
    
    def create_experiment(self, 
                         run_id: str,
                         orchestrator: str,
                         config: Dict[str, Any],
                         notes: Optional[str] = None) -> int:
        """
        Create a new experiment with proper error handling.
        
        Returns:
            Experiment ID
        """
        with self.transaction() as cursor:
            cursor.execute("""
                INSERT INTO experiments (run_id, orchestrator, config_json, notes, status, started_at)
                VALUES (?, ?, ?, ?, 'RUNNING', datetime('now'))
            """, (run_id, orchestrator, json.dumps(config), notes))
            
            exp_id = cursor.lastrowid
            
            # Also create run entry for checkpointing
            cursor.execute("""
                INSERT INTO runs (id, experiment_id, started_at, seed, params_json, status)
                VALUES (?, ?, ?, ?, ?, 'running')
            """, (run_id, exp_id, datetime.utcnow().isoformat(), 
                  config.get('random_seed', 0), json.dumps(config)))
            
            logger.info(f"Created experiment {exp_id} with run_id {run_id}")
            return exp_id
    
    def update_experiment_status(self, 
                                exp_id: int, 
                                status: str,
                                error: Optional[str] = None):
        """Update experiment status with error handling."""
        with self.transaction() as cursor:
            if error:
                cursor.execute("""
                    UPDATE experiments 
                    SET status = ?, finished_at = datetime('now'), error_json = ?
                    WHERE id = ?
                """, (status, json.dumps({"error": error}), exp_id))
            else:
                cursor.execute("""
                    UPDATE experiments 
                    SET status = ?, finished_at = datetime('now')
                    WHERE id = ?
                """, (status, exp_id))
            
            logger.info(f"Updated experiment {exp_id} status to {status}")
    
    # === Result Management with Memory Control ===
    
    def insert_results_batch(self, 
                            exp_id: int,
                            results: List[Dict[str, Any]]) -> int:
        """
        Insert batch of results with memory management.
        
        Returns:
            Number of results inserted
        """
        if not results:
            return 0
        
        # Add to memory cache (bounded)
        kept_in_memory = self._candidates_cache.add_batch(results)
        
        # Use write queue if available
        if self.use_write_queue and self._write_queue:
            # Prepare batch data
            batch_data = []
            for result in results:
                stable_id = result.get('stable_id') or self._generate_stable_id(result)
                batch_data.append({
                    'experiment_id': exp_id,
                    'stable_id': stable_id,
                    'score': result.get('score', float('inf')),
                    'max_need': result.get('max_need'),
                    'var_need': result.get('var_need'),
                    'tail': result.get('tail'),
                    'payload_json': json.dumps(result),
                    # Schedule JSON now includes raw and normalized arrays
                    'schedule_json': json.dumps(result.get('schedule', {})),
                    'sanity_json': json.dumps(result.get('sanity', {})),
                    'diagnostics_json': json.dumps(result.get('diagnostics', {})),
                    'penalties_json': json.dumps(result.get('penalties', {})),
                    'params_json': json.dumps(result.get('params', {}))
                })
            
            # Queue batch insert
            success = self._write_queue.insert_many('results', batch_data)
            
            # Log memory stats periodically
            if len(results) > 0 and len(results) % 1000 == 0:
                stats = self._candidates_cache.get_memory_stats()
                logger.info(f"Memory stats: {stats}")
            
            return len(results) if success else 0
        
        else:
            # Fallback to direct insert
            inserted = 0
            with self.transaction() as cursor:
                for result in results:
                    try:
                        # Generate stable ID if not present
                        stable_id = result.get('stable_id') or self._generate_stable_id(result)
                        
                        cursor.execute("""
                            INSERT OR REPLACE INTO results (
                                experiment_id, stable_id, score, max_need, var_need, tail,
                                payload_json, schedule_json, sanity_json, 
                                diagnostics_json, penalties_json, params_json
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            exp_id,
                            stable_id,
                            result.get('score', float('inf')),
                            result.get('max_need'),
                            result.get('var_need'),
                            result.get('tail'),
                            json.dumps(result),
                            json.dumps(result.get('schedule', {})),
                            json.dumps(result.get('sanity', {})),
                            json.dumps(result.get('diagnostics', {})),
                            json.dumps(result.get('penalties', {})),
                            json.dumps(result.get('params', {}))
                        ))
                        inserted += 1
                        
                    except sqlite3.IntegrityError as e:
                        # Duplicate stable_id, skip
                        logger.debug(f"Skipping duplicate result: {stable_id}")
                    except sqlite3.Error as e:
                        logger.error(f"SQLite error inserting result: {e}", exc_info=True)
                    except Exception as e:
                        logger.error(f"Unexpected error inserting result: {e}", exc_info=True)
            
            # Log memory stats periodically
            if inserted > 0 and inserted % 1000 == 0:
                stats = self._candidates_cache.get_memory_stats()
                logger.info(f"Memory stats: {stats}")
            
            return inserted
    
    def get_best_results(self, 
                         exp_id: int, 
                         limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get best results from database (not from memory cache).
        
        Args:
            exp_id: Experiment ID
            limit: Maximum number of results
            
        Returns:
            List of best results
        """
        with self.read_connection() as cursor:
            cursor.execute("""
                SELECT payload_json 
                FROM results 
                WHERE experiment_id = ? AND score > 0
                ORDER BY score ASC 
                LIMIT ?
            """, (exp_id, limit))
            
            results = []
            for row in cursor.fetchall():
                try:
                    results.append(json.loads(row['payload_json']))
                except:
                    pass
            
            return results
    
    def get_best_from_cache(self, n: int = 10) -> List[Dict[str, Any]]:
        """Get best candidates from memory cache (fast)."""
        return self._candidates_cache.get_best(n)
    
    # === Checkpoint Management ===
    
    def save_checkpoint(self, 
                       run_id: str,
                       batch_idx: int,
                       checkpoint_data: Any) -> bool:
        """
        Save checkpoint with proper error handling.
        
        Args:
            run_id: Run identifier
            batch_idx: Current batch index
            checkpoint_data: Data to checkpoint (will be pickled)
            
        Returns:
            Success status
        """
        try:
            serialized = pickle.dumps(checkpoint_data)
            
            with self.transaction() as cursor:
                cursor.execute("""
                    UPDATE runs 
                    SET last_batch_idx = ?,
                        last_checkpoint_at = datetime('now'),
                        checkpoint_data = ?
                    WHERE id = ?
                """, (batch_idx, serialized, run_id))
                
                logger.info(f"Saved checkpoint for run {run_id} at batch {batch_idx}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            return False
    
    def load_checkpoint(self, run_id: str) -> Optional[Tuple[int, Any]]:
        """
        Load checkpoint with error recovery.
        
        Returns:
            Tuple of (batch_idx, checkpoint_data) or None
        """
        try:
            with self.read_connection() as cursor:
                cursor.execute("""
                    SELECT last_batch_idx, checkpoint_data 
                    FROM runs 
                    WHERE id = ? AND checkpoint_data IS NOT NULL
                """, (run_id,))
                
                row = cursor.fetchone()
                if row:
                    batch_idx = row['last_batch_idx']
                    checkpoint_data = pickle.loads(row['checkpoint_data'])
                    logger.info(f"Loaded checkpoint for run {run_id} at batch {batch_idx}")
                    return batch_idx, checkpoint_data
                    
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
        
        return None
    
    # === Metrics and Monitoring ===
    
    def log_metric(self, 
                  run_id: str,
                  metric_name: str,
                  metric_value: float,
                  batch_idx: Optional[int] = None,
                  metadata: Optional[Dict] = None):
        """Log a metric with proper error handling."""
        try:
            with self.transaction() as cursor:
                cursor.execute("""
                    INSERT INTO metrics (run_id, batch_idx, metric_name, metric_value, metric_json, timestamp)
                    VALUES (?, ?, ?, ?, ?, datetime('now'))
                """, (run_id, batch_idx, metric_name, metric_value, 
                      json.dumps(metadata) if metadata else None))
                
        except Exception as e:
            logger.error(f"Failed to log metric: {e}")
    
    def log_event(self,
                 level: str,
                 event: str,
                 message: str,
                 run_id: Optional[str] = None,
                 exp_id: Optional[int] = None,
                 data: Optional[Dict] = None):
        """Log an event with proper error handling."""
        try:
            with self.transaction() as cursor:
                cursor.execute("""
                    INSERT INTO logs (run_id, experiment_id, level, event, message, data_json)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (run_id, exp_id, level, event, message, 
                      json.dumps(data) if data else None))
                
        except Exception as e:
            # Don't fail on logging errors
            logger.debug(f"Failed to log event: {e}")
    
    # === Utility Methods ===
    
    def _generate_stable_id(self, result: Dict[str, Any]) -> str:
        """Generate stable ID for result deduplication."""
        # Use key parameters for stable ID
        key_parts = [
            str(result.get('params', {}).get('overlap_pct', '')),
            str(result.get('params', {}).get('num_orders', '')),
            str(result.get('params', {}).get('seed', ''))
        ]
        key_string = '|'.join(key_parts)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def vacuum(self):
        """Vacuum database to reclaim space (call during maintenance)."""
        try:
            with self._lock:
                conn = sqlite3.connect(self.db_path)
                conn.execute("VACUUM")
                conn.close()
                logger.info("Database vacuumed successfully")
        except Exception as e:
            logger.error(f"Vacuum failed: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get database and memory statistics."""
        stats = {}
        
        # Database stats
        with self.read_connection() as cursor:
            cursor.execute("SELECT COUNT(*) FROM experiments")
            stats['total_experiments'] = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM results")
            stats['total_results'] = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM runs WHERE status = 'running'")
            stats['active_runs'] = cursor.fetchone()[0]
        
        # Memory stats
        stats['memory'] = self._candidates_cache.get_memory_stats()
        
        # Database file size
        db_size = Path(self.db_path).stat().st_size if Path(self.db_path).exists() else 0
        stats['db_size_mb'] = db_size / (1024 * 1024)
        
        return stats
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.close_all_connections()
        if exc_type:
            logger.error(f"Store exiting with error: {exc_val}")
        return False  # Don't suppress exceptions