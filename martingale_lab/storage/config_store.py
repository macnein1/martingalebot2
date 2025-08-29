"""
Configuration storage and versioning for experiments.
Provides structured config storage with version tracking.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import hashlib

from martingale_lab.core.config_classes import EvaluationConfig
from martingale_lab.core.config_adapter import config_to_flat_dict


class ConfigStore:
    """
    Manages configuration storage with versioning and tracking.
    """
    
    def __init__(self, db_path: str):
        """Initialize config store with database connection."""
        self.db_path = db_path
        self._ensure_tables()
    
    def _ensure_tables(self):
        """Create config tables if they don't exist."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Config versions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS config_versions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    config_hash TEXT UNIQUE NOT NULL,
                    config_json TEXT NOT NULL,
                    flat_json TEXT NOT NULL,
                    version TEXT,
                    created_at TEXT NOT NULL DEFAULT (datetime('now')),
                    first_used_at TEXT,
                    last_used_at TEXT,
                    use_count INTEGER DEFAULT 0,
                    avg_score REAL,
                    best_score REAL,
                    notes TEXT
                )
            """)
            
            # Config lineage table (tracks config evolution)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS config_lineage (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    parent_hash TEXT,
                    child_hash TEXT NOT NULL,
                    change_type TEXT,  -- 'manual', 'auto_tune', 'preset', 'merge'
                    change_description TEXT,
                    created_at TEXT NOT NULL DEFAULT (datetime('now')),
                    FOREIGN KEY (parent_hash) REFERENCES config_versions(config_hash),
                    FOREIGN KEY (child_hash) REFERENCES config_versions(config_hash)
                )
            """)
            
            # Config performance table (tracks config effectiveness)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS config_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    config_hash TEXT NOT NULL,
                    experiment_id INTEGER NOT NULL,
                    run_id TEXT NOT NULL,
                    best_score REAL,
                    avg_score REAL,
                    total_evaluations INTEGER,
                    convergence_batch INTEGER,  -- Which batch reached best score
                    time_to_best REAL,  -- Seconds to reach best score
                    created_at TEXT NOT NULL DEFAULT (datetime('now')),
                    FOREIGN KEY (config_hash) REFERENCES config_versions(config_hash)
                )
            """)
            
            # Add indexes
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_config_hash 
                ON config_versions(config_hash)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_config_performance 
                ON config_performance(config_hash, best_score)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_config_lineage 
                ON config_lineage(parent_hash, child_hash)
            """)
            
            conn.commit()
    
    def _compute_hash(self, config: EvaluationConfig) -> str:
        """Compute stable hash for configuration."""
        # Convert to flat dict for consistent ordering
        flat = config_to_flat_dict(config)
        
        # Sort keys for stable hash
        sorted_items = sorted(flat.items())
        
        # Create hash
        hash_str = json.dumps(sorted_items, sort_keys=True)
        return hashlib.sha256(hash_str.encode()).hexdigest()[:16]
    
    def store_config(
        self, 
        config: EvaluationConfig,
        parent_hash: Optional[str] = None,
        change_type: Optional[str] = None,
        change_description: Optional[str] = None,
        notes: Optional[str] = None
    ) -> str:
        """
        Store configuration and return its hash.
        
        Args:
            config: Configuration to store
            parent_hash: Parent config hash if this is derived
            change_type: Type of change from parent
            change_description: Description of changes
            notes: Additional notes
            
        Returns:
            Config hash
        """
        config_hash = self._compute_hash(config)
        config_json = config.to_json()
        flat_json = json.dumps(config_to_flat_dict(config), indent=2)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Check if config already exists
            cursor.execute(
                "SELECT id, use_count FROM config_versions WHERE config_hash = ?",
                (config_hash,)
            )
            existing = cursor.fetchone()
            
            if existing:
                # Update use count and last used
                cursor.execute("""
                    UPDATE config_versions 
                    SET use_count = use_count + 1,
                        last_used_at = datetime('now')
                    WHERE config_hash = ?
                """, (config_hash,))
            else:
                # Insert new config
                cursor.execute("""
                    INSERT INTO config_versions 
                    (config_hash, config_json, flat_json, version, notes, first_used_at)
                    VALUES (?, ?, ?, ?, ?, datetime('now'))
                """, (config_hash, config_json, flat_json, "1.0", notes))
            
            # Record lineage if parent provided
            if parent_hash:
                cursor.execute("""
                    INSERT INTO config_lineage 
                    (parent_hash, child_hash, change_type, change_description)
                    VALUES (?, ?, ?, ?)
                """, (parent_hash, config_hash, change_type, change_description))
            
            conn.commit()
        
        return config_hash
    
    def load_config(self, config_hash: str) -> Optional[EvaluationConfig]:
        """Load configuration by hash."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT config_json FROM config_versions WHERE config_hash = ?",
                (config_hash,)
            )
            row = cursor.fetchone()
            
            if row:
                return EvaluationConfig.from_json(row[0])
            return None
    
    def record_performance(
        self,
        config_hash: str,
        experiment_id: int,
        run_id: str,
        best_score: float,
        avg_score: float,
        total_evaluations: int,
        convergence_batch: Optional[int] = None,
        time_to_best: Optional[float] = None
    ):
        """Record performance metrics for a configuration."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO config_performance 
                (config_hash, experiment_id, run_id, best_score, avg_score,
                 total_evaluations, convergence_batch, time_to_best)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (config_hash, experiment_id, run_id, best_score, avg_score,
                  total_evaluations, convergence_batch, time_to_best))
            
            # Update aggregate stats in config_versions
            cursor.execute("""
                UPDATE config_versions
                SET avg_score = (
                    SELECT AVG(best_score) FROM config_performance 
                    WHERE config_hash = ?
                ),
                best_score = (
                    SELECT MIN(best_score) FROM config_performance 
                    WHERE config_hash = ?
                )
                WHERE config_hash = ?
            """, (config_hash, config_hash, config_hash))
            
            conn.commit()
    
    def get_config_stats(self, config_hash: str) -> Dict[str, Any]:
        """Get performance statistics for a configuration."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Get config info
            cursor.execute("""
                SELECT use_count, avg_score, best_score, created_at
                FROM config_versions WHERE config_hash = ?
            """, (config_hash,))
            config_row = cursor.fetchone()
            
            if not config_row:
                return {}
            
            # Get performance history
            cursor.execute("""
                SELECT best_score, avg_score, total_evaluations, 
                       convergence_batch, time_to_best
                FROM config_performance 
                WHERE config_hash = ?
                ORDER BY best_score ASC
            """, (config_hash,))
            perf_rows = cursor.fetchall()
            
            return {
                'config_hash': config_hash,
                'use_count': config_row[0],
                'avg_score': config_row[1],
                'best_score': config_row[2],
                'created_at': config_row[3],
                'performance_history': [
                    {
                        'best_score': row[0],
                        'avg_score': row[1],
                        'total_evaluations': row[2],
                        'convergence_batch': row[3],
                        'time_to_best': row[4]
                    }
                    for row in perf_rows
                ]
            }
    
    def get_best_configs(self, limit: int = 10) -> List[Tuple[str, float, int]]:
        """
        Get best performing configurations.
        
        Returns:
            List of (config_hash, best_score, use_count)
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT config_hash, best_score, use_count
                FROM config_versions
                WHERE best_score IS NOT NULL
                ORDER BY best_score ASC
                LIMIT ?
            """, (limit,))
            return cursor.fetchall()
    
    def get_config_lineage(self, config_hash: str) -> Dict[str, Any]:
        """Get configuration lineage (parents and children)."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Get parents
            cursor.execute("""
                SELECT parent_hash, change_type, change_description, created_at
                FROM config_lineage
                WHERE child_hash = ?
            """, (config_hash,))
            parents = cursor.fetchall()
            
            # Get children
            cursor.execute("""
                SELECT child_hash, change_type, change_description, created_at
                FROM config_lineage
                WHERE parent_hash = ?
            """, (config_hash,))
            children = cursor.fetchall()
            
            return {
                'config_hash': config_hash,
                'parents': [
                    {
                        'hash': row[0],
                        'change_type': row[1],
                        'change_description': row[2],
                        'created_at': row[3]
                    }
                    for row in parents
                ],
                'children': [
                    {
                        'hash': row[0],
                        'change_type': row[1],
                        'change_description': row[2],
                        'created_at': row[3]
                    }
                    for row in children
                ]
            }
    
    def find_similar_configs(
        self, 
        config: EvaluationConfig, 
        threshold: float = 0.9
    ) -> List[Tuple[str, float]]:
        """
        Find similar configurations based on parameter similarity.
        
        Returns:
            List of (config_hash, similarity_score)
        """
        target_flat = config_to_flat_dict(config)
        similar = []
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT config_hash, flat_json FROM config_versions")
            
            for row in cursor.fetchall():
                config_hash, flat_json = row
                stored_flat = json.loads(flat_json)
                
                # Calculate similarity (simple parameter matching)
                matches = 0
                total = 0
                for key in target_flat:
                    if key in stored_flat:
                        total += 1
                        if target_flat[key] == stored_flat[key]:
                            matches += 1
                
                if total > 0:
                    similarity = matches / total
                    if similarity >= threshold:
                        similar.append((config_hash, similarity))
            
            # Sort by similarity
            similar.sort(key=lambda x: x[1], reverse=True)
            return similar
    
    def export_config_history(self, output_path: str):
        """Export configuration history to JSON file."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Get all configs with stats
            cursor.execute("""
                SELECT config_hash, config_json, use_count, 
                       avg_score, best_score, created_at
                FROM config_versions
                ORDER BY best_score ASC
            """)
            
            configs = []
            for row in cursor.fetchall():
                configs.append({
                    'hash': row[0],
                    'config': json.loads(row[1]),
                    'use_count': row[2],
                    'avg_score': row[3],
                    'best_score': row[4],
                    'created_at': row[5]
                })
            
            # Write to file
            output = Path(output_path)
            output.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output, 'w') as f:
                json.dump({
                    'export_date': datetime.now().isoformat(),
                    'total_configs': len(configs),
                    'configs': configs
                }, f, indent=2)
            
            return len(configs)