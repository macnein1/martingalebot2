"""
Adaptive Orchestrator for DCA/Martingale Optimization
Implements comprehensive logging, identity management, and error handling
"""
import json
import time
import traceback
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional, Callable
import numpy as np
import os

from martingale_lab.utils.structured_logging import (
    get_structured_logger, EventNames, Timer, generate_run_id, generate_span_id,
    ensure_json_serializable
)
from martingale_lab.optimizer.evaluation_engine import evaluation_function
from martingale_lab.storage.experiments_store import ExperimentsStore
from ui.utils.constants import DB_PATH, Status, CRASH_SNAPSHOTS_DIR

# Initialize structured logger for orchestrator
logger = get_structured_logger("mlab.orch")


@dataclass
class OrchConfig:
    """Configuration for orchestrator run"""
    run_id: str
    overlap_min: float = 10.0
    overlap_max: float = 30.0
    orders_min: int = 5
    orders_max: int = 15
    alpha: float = 0.5
    beta: float = 0.3
    gamma: float = 0.2
    lambda_penalty: float = 0.1
    wave_pattern: bool = True
    tail_cap: float = 0.40
    min_indent_step: float = 0.05
    softmax_temp: float = 1.0
    
    # Orchestrator settings
    batch_size: int = 100
    max_batches: int = 50
    patience: int = 5
    prune_factor: float = 2.0
    top_k: int = 50
    base_price: float = 100.0


class AdaptiveOrchestrator:
    """
    Orchestrates DCA optimization with comprehensive logging and error handling
    """
    
    def __init__(self, config: OrchConfig, db_path: str = DB_PATH):
        self.config = config
        self.db_path = db_path
        self.store = ExperimentsStore(db_path)
        self.exp_id: Optional[int] = None
        self.eval_count = 0
        self.best_score = float('inf')
        self.best_payload: Optional[Dict[str, Any]] = None
        self.top_candidates: List[Dict[str, Any]] = []
        self.patience_counter = 0
        self.should_stop = False
        
        # Initialize with structured logging
        logger.info(
            EventNames.ORCH_START,
            f"Initializing orchestrator with run_id {config.run_id}",
            run_id=config.run_id,
            config_json=json.dumps(asdict(config), default=str)
        )
    
    def create_experiment(self) -> int:
        """Create experiment record and return exp_id"""
        config_dict = asdict(self.config)
        config_dict.pop('run_id', None)  # Don't duplicate in config_json
        
        self.exp_id = self.store.create_experiment(
            adapter="adaptive_orchestrator",
            cfg=config_dict,
            run_id=self.config.run_id
        )
        
        logger.info(
            EventNames.BUILD_CONFIG,
            f"Created experiment {self.exp_id}",
            run_id=self.config.run_id,
            exp_id=self.exp_id,
            config_snapshot=config_dict
        )
        
        return self.exp_id
    
    def generate_candidate(self, rng: np.random.Generator) -> Dict[str, Any]:
        """Generate a single candidate with random parameters"""
        return {
            'base_price': self.config.base_price,
            'overlap_pct': rng.uniform(self.config.overlap_min, self.config.overlap_max),
            'num_orders': int(rng.integers(self.config.orders_min, self.config.orders_max + 1)),
            'alpha': self.config.alpha,
            'beta': self.config.beta,
            'gamma': self.config.gamma,
            'lambda_penalty': self.config.lambda_penalty,
            'wave_pattern': self.config.wave_pattern,
            'tail_cap': self.config.tail_cap,
            'min_indent_step': self.config.min_indent_step,
            'softmax_temp': self.config.softmax_temp,
            'seed': int(rng.integers(0, 2**31))
        }
    
    def evaluate_candidate(self, candidate: Dict[str, Any], 
                          run_id: str, exp_id: int, span_id: str) -> Optional[Dict[str, Any]]:
        """Evaluate a single candidate with timing and logging"""
        with Timer() as timer:
            # Log evaluation call
            logger.info(
                EventNames.EVAL_CALL,
                "Calling evaluation function",
                run_id=run_id,
                exp_id=exp_id,
                span_id=span_id,
                eval_count=self.eval_count + 1,
                overlap=candidate['overlap_pct'],
                orders=candidate['num_orders']
            )
            
            # Call evaluation function
            result = evaluation_function(**candidate)
            
            # Ensure JSON serializable
            result = ensure_json_serializable(result)
            
            # Log evaluation return
            logger.info(
                EventNames.EVAL_RETURN,
                f"Evaluation completed with score {result['score']:.4f}",
                run_id=run_id,
                exp_id=exp_id,
                span_id=span_id,
                eval_count=self.eval_count + 1,
                score=result['score'],
                duration_ms=timer.duration_ms,
                sanity_violations=sum(1 for v in result.get('sanity', {}).values() if v)
            )
            
            return result
    
    def should_prune(self, score: float) -> bool:
        """Determine if candidate should be pruned"""
        if self.best_score == float('inf'):
            return False
        return score > self.best_score * self.config.prune_factor
    
    def update_best(self, result: Dict[str, Any]) -> bool:
        """Update best score and payload, return True if improved"""
        score = result['score']
        improved = score < self.best_score
        
        if improved:
            self.best_score = score
            self.best_payload = result.copy()
            self.patience_counter = 0
        
        # Maintain top-K candidates
        self.top_candidates.append(result)
        self.top_candidates.sort(key=lambda x: x['score'])
        if len(self.top_candidates) > self.config.top_k:
            self.top_candidates = self.top_candidates[:self.config.top_k]
        
        return improved
    
    def run_batch(self, batch_idx: int, rng: np.random.Generator) -> Dict[str, Any]:
        """Run a single batch of evaluations"""
        span_id = generate_span_id(batch_idx)
        batch_start = time.time()
        
        logger.info(
            EventNames.ORCH_BATCH,
            f"Starting batch {batch_idx}",
            run_id=self.config.run_id,
            exp_id=self.exp_id,
            span_id=span_id,
            batch_idx=batch_idx,
            batch_size=self.config.batch_size
        )
        
        batch_results = []
        pruned_count = 0
        
        for i in range(self.config.batch_size):
            if self.should_stop:
                break
                
            # Generate and evaluate candidate
            candidate = self.generate_candidate(rng)
            result = self.evaluate_candidate(candidate, self.config.run_id, self.exp_id, span_id)
            
            if result is None:
                continue
                
            self.eval_count += 1
            score = result['score']
            
            # Check for pruning
            if self.should_prune(score):
                pruned_count += 1
                logger.debug(
                    EventNames.ORCH_PRUNE,
                    f"Pruned candidate with score {score:.4f}",
                    run_id=self.config.run_id,
                    exp_id=self.exp_id,
                    span_id=span_id,
                    score=score,
                    best_score=self.best_score
                )
                continue
            
            # Update best and add to batch results
            improved = self.update_best(result)
            batch_results.append(result)
            
            if improved:
                logger.info(
                    EventNames.ORCH_BATCH,
                    f"New best score: {score:.4f}",
                    run_id=self.config.run_id,
                    exp_id=self.exp_id,
                    span_id=span_id,
                    score=score,
                    improvement=self.best_score - score if self.best_score != float('inf') else 0
                )
        
        # Save batch results to database
        if batch_results:
            rows_saved = self.store.upsert_results(self.exp_id, batch_results)
            logger.info(
                EventNames.ORCH_SAVE_OK,
                f"Saved {rows_saved} results from batch {batch_idx}",
                run_id=self.config.run_id,
                exp_id=self.exp_id,
                span_id=span_id,
                rows=rows_saved
            )
        
        # Log pruning statistics
        if pruned_count > 0:
            logger.info(
                EventNames.ORCH_PRUNE,
                f"Pruned {pruned_count} candidates in batch {batch_idx}",
                run_id=self.config.run_id,
                exp_id=self.exp_id,
                span_id=span_id,
                pruned_count=pruned_count
            )
        
        batch_duration = (time.time() - batch_start) * 1000
        return {
            'batch_idx': batch_idx,
            'evaluated': len(batch_results),
            'pruned': pruned_count,
            'best_score': self.best_score,
            'duration_ms': batch_duration
        }
    
    def check_early_stopping(self) -> bool:
        """Check if we should stop early due to lack of improvement"""
        self.patience_counter += 1
        if self.patience_counter >= self.config.patience:
            logger.info(
                EventNames.ORCH_EARLY_STOP,
                f"Early stopping after {self.patience_counter} batches without improvement",
                run_id=self.config.run_id,
                exp_id=self.exp_id,
                patience=self.config.patience,
                best_score=self.best_score
            )
            return True
        return False
    
    def save_crash_snapshot(self, error: Exception, context: Dict[str, Any]) -> str:
        """Save crash snapshot for debugging"""
        snapshot = {
            'timestamp': datetime.now().isoformat(),
            'run_id': self.config.run_id,
            'exp_id': self.exp_id,
            'error': str(error),
            'traceback': traceback.format_exc(),
            'context': context,
            'config': asdict(self.config),
            'stats': {
                'eval_count': self.eval_count,
                'best_score': self.best_score,
                'top_candidates_count': len(self.top_candidates)
            }
        }
        
        filename = f"{self.config.run_id}_{int(time.time())}.json"
        filepath = CRASH_SNAPSHOTS_DIR / filename
        
        with open(filepath, 'w') as f:
            json.dump(snapshot, f, indent=2, default=str)
        
        return str(filepath)
    
    def run_optimization(self, progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Main optimization loop with comprehensive error handling and logging
        """
        start_time = time.time()
        
        try:
            # Create experiment record
            if self.exp_id is None:
                self.create_experiment()
            
            # Initialize random generator
            rng = np.random.default_rng()
            
            # Main batch loop
            for batch_idx in range(self.config.max_batches):
                if self.should_stop:
                    break
                
                # Run batch
                batch_stats = self.run_batch(batch_idx, rng)
                
                # Update progress callback
                if progress_callback:
                    progress_callback({
                        'batch_idx': batch_idx,
                        'eval_count': self.eval_count,
                        'best_score': self.best_score,
                        'batch_stats': batch_stats
                    })
                
                # Check early stopping
                if self.check_early_stopping():
                    break
            
            # Update experiment summary
            elapsed_s = time.time() - start_time
            self.store.update_experiment_summary(
                self.exp_id, self.best_score, self.eval_count, elapsed_s
            )
            
            # Log completion
            logger.info(
                EventNames.ORCH_DONE,
                f"Optimization completed successfully",
                run_id=self.config.run_id,
                exp_id=self.exp_id,
                best_score=self.best_score,
                total_evals=self.eval_count,
                elapsed_ms=(elapsed_s * 1000)
            )
            
            return {
                'status': 'completed',
                'run_id': self.config.run_id,
                'exp_id': self.exp_id,
                'best_score': self.best_score,
                'total_evals': self.eval_count,
                'elapsed_s': elapsed_s,
                'best_payload': self.best_payload
            }
            
        except Exception as e:
            # Log error with full context
            elapsed_s = time.time() - start_time
            
            # Save crash snapshot
            context = {
                'batch_idx': getattr(self, 'current_batch_idx', -1),
                'eval_count': self.eval_count,
                'best_score': self.best_score
            }
            snapshot_path = self.save_crash_snapshot(e, context)
            
            logger.error(
                EventNames.ORCH_ERROR,
                f"Optimization failed: {str(e)}",
                run_id=self.config.run_id,
                exp_id=self.exp_id,
                error=str(e),
                traceback=traceback.format_exc(),
                snapshot_path=snapshot_path,
                stats={
                    'evals_total': self.eval_count,
                    'evals_ok': self.eval_count,  # Approximate
                    'evals_failed': 0,  # Would need to track separately
                    'elapsed_s': elapsed_s
                }
            )
            
            # Update experiment status to FAILED
            if self.exp_id:
                 try:
                     import sqlite3
                     with sqlite3.connect(self.db_path) as conn:
                         cur = conn.cursor()
                         cur.execute(
                             "UPDATE experiments SET status = ?, finished_at = ? WHERE id = ?",
                             (Status.FAILED, datetime.now().isoformat(), self.exp_id)
                         )
                         conn.commit()
                 except:
                     pass  # Don't let secondary error mask primary error
            
            # Re-raise the original exception
            raise
    
    def stop(self):
        """Signal orchestrator to stop gracefully"""
        self.should_stop = True
        logger.info(
            EventNames.ORCH_EARLY_STOP,
            "Stop signal received",
            run_id=self.config.run_id,
            exp_id=self.exp_id
        )