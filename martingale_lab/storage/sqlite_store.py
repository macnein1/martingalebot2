"""
SQLite storage for optimization results and traces.
"""
import sqlite3
import json
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

from ..core.types import Params, Schedule, ScoreBreakdown


class SQLiteStore:
    """SQLite-based storage for optimization results."""
    
    def __init__(self, db_path: str = "martingale_optimization.db"):
        """Initialize SQLite store."""
        self.db_path = db_path
        self.init_database()
        self.conn = None  # For potential persistent connection
    
    def init_database(self):
        """Initialize database tables."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Results table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS optimization_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    params_json TEXT NOT NULL,
                    score REAL NOT NULL,
                    max_score REAL NOT NULL,
                    variance_score REAL NOT NULL,
                    tail_score REAL NOT NULL,
                    gini_penalty REAL NOT NULL,
                    entropy_penalty REAL NOT NULL,
                    monotone_penalty REAL NOT NULL,
                    smoothness_penalty REAL NOT NULL,
                    schedule_json TEXT,
                    metadata TEXT
                )
            """)
            
            # Traces table for optimization progress
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS optimization_traces (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    iteration INTEGER NOT NULL,
                    best_score REAL NOT NULL,
                    current_score REAL NOT NULL,
                    params_json TEXT NOT NULL,
                    metadata TEXT
                )
            """)
            
            # Top-N results table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS top_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    rank INTEGER NOT NULL,
                    timestamp TEXT NOT NULL,
                    params_json TEXT NOT NULL,
                    score REAL NOT NULL,
                    breakdown_json TEXT NOT NULL,
                    schedule_json TEXT,
                    metadata TEXT
                )
            """)
            
            # Create indexes for better performance
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_results_score ON optimization_results(score)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_results_timestamp ON optimization_results(timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_traces_session ON optimization_traces(session_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_traces_iteration ON optimization_traces(iteration)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_top_rank ON top_results(rank)")
            
            conn.commit()
    
    def get_results(self, run_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get results for a specific run.
        
        Args:
            run_id: Run identifier
            limit: Maximum number of results
            
        Returns:
            List of result dictionaries
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Try to get from metadata
            cursor.execute("""
                SELECT params_json, score, schedule_json, metadata
                FROM optimization_results
                WHERE metadata LIKE ?
                ORDER BY score ASC
                LIMIT ?
            """, (f'%"run_id": "{run_id}"%', limit))
            
            results = []
            for row in cursor.fetchall():
                result = {
                    'params': json.loads(row[0]),
                    'score': row[1],
                    'schedule': json.loads(row[2]) if row[2] else None,
                    'metadata': json.loads(row[3]) if row[3] else {}
                }
                results.append(result)
            
            return results
    
    def store_result(self, run_id: str, result_data: Dict[str, Any]):
        """
        Store result with simple interface (alias for compatibility).
        
        Args:
            run_id: Run identifier
            result_data: Dictionary with result data
        """
        # Extract required fields from result_data
        score = result_data.get('score', float('inf'))
        
        # Create minimal params object
        from martingale_lab.core.types import Params, ScoreBreakdown
        
        overlap = result_data.get('overlap_pct', 10.0)
        num_orders = result_data.get('num_orders', 20)
        
        params = Params(
            min_overlap=overlap,
            max_overlap=overlap,
            min_order=num_orders,
            max_order=num_orders
        )
        
        breakdown = ScoreBreakdown(
            total_score=score,  # Add required field
            max_score=score,
            variance_score=0.0,
            tail_score=0.0,
            gini_penalty=0.0,
            entropy_penalty=0.0,
            monotone_penalty=0.0,
            smoothness_penalty=0.0
        )
        
        # Add run_id to metadata
        metadata = result_data.copy()
        metadata['run_id'] = run_id
        
        self.save_result(params, score, breakdown, metadata=metadata)
    
    def save_result(self, params: Params, score: float, breakdown: ScoreBreakdown, 
                   schedule: Optional[Schedule] = None, metadata: Optional[Dict] = None):
        """Save optimization result."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO optimization_results 
                (timestamp, params_json, score, max_score, variance_score, tail_score,
                 gini_penalty, entropy_penalty, monotone_penalty, smoothness_penalty,
                 schedule_json, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                datetime.now().isoformat(),
                json.dumps(params.to_dict()),
                score,
                breakdown.max_score,
                breakdown.variance_score,
                breakdown.tail_score,
                breakdown.gini_penalty,
                breakdown.entropy_penalty,
                breakdown.monotone_penalty,
                breakdown.smoothness_penalty,
                json.dumps(schedule.to_dict()) if schedule else None,
                json.dumps(metadata) if metadata else None
            ))
            
            conn.commit()
    
    def save_trace(self, session_id: str, iteration: int, best_score: float, 
                  current_score: float, params: Params, metadata: Optional[Dict] = None):
        """Save optimization trace."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO optimization_traces 
                (session_id, timestamp, iteration, best_score, current_score, params_json, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                session_id,
                datetime.now().isoformat(),
                iteration,
                best_score,
                current_score,
                json.dumps(params.to_dict()),
                json.dumps(metadata) if metadata else None
            ))
            
            conn.commit()
    
    def save_top_results(self, results: List[Tuple[float, Params, ScoreBreakdown]], 
                        schedules: Optional[List[Schedule]] = None):
        """Save top-N results."""
        # Clear existing top results
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM top_results")
            
            # Insert new top results
            for i, (score, params, breakdown) in enumerate(results):
                schedule = schedules[i] if schedules and i < len(schedules) else None
                
                cursor.execute("""
                    INSERT INTO top_results 
                    (rank, timestamp, params_json, score, breakdown_json, schedule_json)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    i + 1,
                    datetime.now().isoformat(),
                    json.dumps(params.to_dict()),
                    score,
                    json.dumps(breakdown.to_dict()),
                    json.dumps(schedule.to_dict()) if schedule else None
                ))
            
            conn.commit()
    
    def get_top_results(self, limit: int = 10) -> List[Tuple[int, float, Params, ScoreBreakdown]]:
        """Get top-N results."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT rank, score, params_json, breakdown_json 
                FROM top_results 
                ORDER BY rank 
                LIMIT ?
            """, (limit,))
            
            results = []
            for row in cursor.fetchall():
                rank, score, params_json, breakdown_json = row
                params = Params.from_dict(json.loads(params_json))
                breakdown = ScoreBreakdown.from_dict(json.loads(breakdown_json))
                results.append((rank, score, params, breakdown))
            
            return results
    
    def get_results_by_score_range(self, min_score: float, max_score: float) -> List[Tuple[float, Params, ScoreBreakdown]]:
        """Get results within score range."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT score, params_json, breakdown_json 
                FROM optimization_results 
                WHERE score BETWEEN ? AND ?
                ORDER BY score DESC
            """, (min_score, max_score))
            
            results = []
            for row in cursor.fetchall():
                score, params_json, breakdown_json = row
                params = Params.from_dict(json.loads(params_json))
                breakdown = ScoreBreakdown.from_dict(json.loads(breakdown_json))
                results.append((score, params, breakdown))
            
            return results
    
    def get_trace_by_session(self, session_id: str) -> List[Tuple[int, float, float, Params]]:
        """Get optimization trace for a session."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT iteration, best_score, current_score, params_json 
                FROM optimization_traces 
                WHERE session_id = ?
                ORDER BY iteration
            """, (session_id,))
            
            results = []
            for row in cursor.fetchall():
                iteration, best_score, current_score, params_json = row
                params = Params.from_dict(json.loads(params_json))
                results.append((iteration, best_score, current_score, params))
            
            return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Count results
            cursor.execute("SELECT COUNT(*) FROM optimization_results")
            total_results = cursor.fetchone()[0]
            
            # Best score
            cursor.execute("SELECT MAX(score) FROM optimization_results")
            best_score = cursor.fetchone()[0] or 0.0
            
            # Average score
            cursor.execute("SELECT AVG(score) FROM optimization_results")
            avg_score = cursor.fetchone()[0] or 0.0
            
            # Recent results (last 24 hours)
            cursor.execute("""
                SELECT COUNT(*) FROM optimization_results 
                WHERE timestamp > datetime('now', '-1 day')
            """)
            recent_results = cursor.fetchone()[0]
            
            # Session count
            cursor.execute("SELECT COUNT(DISTINCT session_id) FROM optimization_traces")
            session_count = cursor.fetchone()[0]
            
            return {
                'total_results': total_results,
                'best_score': best_score,
                'average_score': avg_score,
                'recent_results_24h': recent_results,
                'session_count': session_count
            }
    
    def clear_old_results(self, days: int = 30):
        """Clear results older than specified days."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                DELETE FROM optimization_results 
                WHERE timestamp < datetime('now', '-{} days')
            """.format(days))
            
            cursor.execute("""
                DELETE FROM optimization_traces 
                WHERE timestamp < datetime('now', '-{} days')
            """.format(days))
            
            conn.commit()
    
    def export_results(self, filepath: str, format: str = 'json'):
        """Export results to file."""
        if format == 'json':
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM optimization_results")
                
                results = []
                for row in cursor.fetchall():
                    results.append({
                        'id': row[0],
                        'timestamp': row[1],
                        'params': json.loads(row[2]),
                        'score': row[3],
                        'max_score': row[4],
                        'variance_score': row[5],
                        'tail_score': row[6],
                        'gini_penalty': row[7],
                        'entropy_penalty': row[8],
                        'monotone_penalty': row[9],
                        'smoothness_penalty': row[10],
                        'schedule': json.loads(row[11]) if row[11] else None,
                        'metadata': json.loads(row[12]) if row[12] else None
                    })
                
                with open(filepath, 'w') as f:
                    json.dump(results, f, indent=2)
        
        elif format == 'csv':
            import pandas as pd
            
            with sqlite3.connect(self.db_path) as conn:
                df = pd.read_sql_query("SELECT * FROM optimization_results", conn)
                df.to_csv(filepath, index=False)
        
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def backup_database(self, backup_path: str):
        """Create database backup."""
        import shutil
        shutil.copy2(self.db_path, backup_path)

    def close(self):
        """
        Close database connection if open.
        """
        if self.conn:
            try:
                self.conn.close()
                self.conn = None
            except:
                pass  # Ignore errors on close
