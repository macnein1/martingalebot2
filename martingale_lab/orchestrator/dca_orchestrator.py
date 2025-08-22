"""
DCA Orchestrator - "İşlemden En Hızlı Çıkış" Optimization Engine
Integrates new evaluation contract with adaptive search, early pruning, and batch processing.
"""
from __future__ import annotations

import time
import logging
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Tuple, Callable
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

from martingale_lab.optimizer.dca_evaluation_contract import evaluation_function
from martingale_lab.storage.experiments_store import ExperimentsStore


@dataclass
class DCAConfig:
    """Configuration for DCA optimization."""
    # Search space
    base_price: float = 1.0
    overlap_min: float = 10.0
    overlap_max: float = 30.0
    orders_min: int = 5
    orders_max: int = 15
    
    # Optimization parameters
    n_candidates_per_batch: int = 1000
    max_batches: int = 100
    top_k_keep: int = 10000  # Keep best K candidates between batches
    
    # Evaluation weights
    alpha: float = 0.5
    beta: float = 0.3
    gamma: float = 0.2
    lambda_penalty: float = 0.1
    
    # Wave pattern settings
    wave_pattern: bool = False
    wave_strong_threshold: float = 50.0
    wave_weak_threshold: float = 10.0
    
    # Constraints
    tail_cap: float = 0.40
    min_indent_step: float = 0.05
    softmax_temp: float = 1.0
    
    # Early stopping
    early_stop_threshold: float = 1e-6  # Stop if improvement < threshold
    early_stop_patience: int = 10  # Number of batches without improvement
    
    # Parallelization
    n_workers: int = 4
    
    # Random seed
    random_seed: Optional[int] = None


class DCAOrchestrator:
    """Main orchestrator for DCA optimization with new evaluation contract."""
    
    def __init__(self, config: DCAConfig, store: Optional[ExperimentsStore] = None):
        self.config = config
        self.store = store or ExperimentsStore()
        self.logger = logging.getLogger(__name__)
        
        # State tracking
        self.current_experiment_id: Optional[int] = None
        self.best_candidates: List[Dict[str, Any]] = []
        self.batch_count = 0
        self.total_evaluations = 0
        self.start_time = 0.0
        self.best_score = float("inf")
        self.patience_counter = 0
        
        # Statistics
        self.stats = {
            "total_time": 0.0,
            "evaluations_per_second": 0.0,
            "batches_completed": 0,
            "early_stopped": False,
            "sanity_violations": 0,
            "wave_pattern_rewards": 0
        }
    
    def create_experiment(self, notes: Optional[str] = None) -> int:
        """Create a new experiment in the database."""
        config_dict = {
            "overlap_min": self.config.overlap_min,
            "overlap_max": self.config.overlap_max,
            "orders_min": self.config.orders_min,
            "orders_max": self.config.orders_max,
            "alpha": self.config.alpha,
            "beta": self.config.beta,
            "gamma": self.config.gamma,
            "lambda_penalty": self.config.lambda_penalty,
            "wave_pattern": self.config.wave_pattern,
            "tail_cap": self.config.tail_cap,
            "notes": notes or "DCA optimization with new evaluation contract"
        }
        
        self.current_experiment_id = self.store.create_experiment("DCAOrchestrator", config_dict)
        return self.current_experiment_id
    
    def generate_random_parameters(self, n_samples: int) -> List[Dict[str, Any]]:
        """Generate random parameter combinations for evaluation."""
        rng = np.random.default_rng(self.config.random_seed)
        
        parameters = []
        for _ in range(n_samples):
            # Random overlap and orders
            overlap_pct = float(rng.uniform(self.config.overlap_min, self.config.overlap_max))
            num_orders = int(rng.integers(self.config.orders_min, self.config.orders_max + 1))
            
            # Random seed for evaluation
            eval_seed = int(rng.integers(0, 2**31 - 1))
            
            params = {
                "base_price": self.config.base_price,
                "overlap_pct": overlap_pct,
                "num_orders": num_orders,
                "seed": eval_seed,
                "wave_pattern": self.config.wave_pattern,
                "alpha": self.config.alpha,
                "beta": self.config.beta,
                "gamma": self.config.gamma,
                "lambda_penalty": self.config.lambda_penalty,
                "wave_strong_threshold": self.config.wave_strong_threshold,
                "wave_weak_threshold": self.config.wave_weak_threshold,
                "tail_cap": self.config.tail_cap,
                "min_indent_step": self.config.min_indent_step,
                "softmax_temp": self.config.softmax_temp
            }
            parameters.append(params)
        
        return parameters
    
    def evaluate_candidate(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate a single candidate using the new evaluation contract."""
        try:
            result = evaluation_function(**params)
            
            # Add parameter info and stable_id
            import hashlib
            import json
            
            param_subset = {
                "base_price": params["base_price"],
                "overlap_pct": params["overlap_pct"],
                "num_orders": params["num_orders"],
                "alpha": params["alpha"],
                "beta": params["beta"],
                "gamma": params["gamma"],
                "lambda_penalty": params["lambda_penalty"],
                "wave_pattern": params["wave_pattern"],
                "tail_cap": params["tail_cap"]
            }
            
            stable_id = hashlib.sha1(
                json.dumps(param_subset, sort_keys=True).encode()
            ).hexdigest()[:16]
            
            result["params"] = param_subset
            result["stable_id"] = stable_id
            
            return result
            
        except Exception as e:
            self.logger.error(f"Evaluation failed for params {params}: {e}")
            return {
                "score": float("inf"),
                "max_need": float("inf"),
                "var_need": float("inf"),
                "tail": float("inf"),
                "schedule": {},
                "sanity": {"max_need_mismatch": True, "collapse_indents": True, "tail_overflow": True},
                "diagnostics": {"wci": 0.0, "sign_flips": 0, "gini": 1.0, "entropy": 0.0},
                "penalties": {},
                "params": params,
                "stable_id": None,
                "error": str(e)
            }
    
    def evaluate_batch_parallel(self, param_batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Evaluate a batch of candidates in parallel."""
        results = []
        
        if self.config.n_workers <= 1:
            # Sequential evaluation
            for params in param_batch:
                result = self.evaluate_candidate(params)
                results.append(result)
        else:
            # Parallel evaluation
            with ThreadPoolExecutor(max_workers=self.config.n_workers) as executor:
                future_to_params = {
                    executor.submit(self.evaluate_candidate, params): params
                    for params in param_batch
                }
                
                for future in as_completed(future_to_params):
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        params = future_to_params[future]
                        self.logger.error(f"Parallel evaluation failed for {params}: {e}")
                        results.append(self.evaluate_candidate(params))  # Fallback
        
        return results
    
    def early_pruning(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply early pruning based on partial evaluation metrics."""
        if not results:
            return results
        
        # Filter out infinite scores and sanity violations
        valid_results = []
        for result in results:
            score = result.get("score", float("inf"))
            sanity = result.get("sanity", {})
            
            # Skip infinite scores
            if not np.isfinite(score):
                continue
            
            # Skip severe sanity violations (optional - can be made configurable)
            severe_violations = (
                sanity.get("max_need_mismatch", False) and
                sanity.get("collapse_indents", False)
            )
            
            if not severe_violations:
                valid_results.append(result)
        
        # Early pruning: keep only candidates with score < current_best * 1.5
        if self.best_score < float("inf"):
            threshold = self.best_score * 1.5
            pruned_results = [r for r in valid_results if r.get("score", float("inf")) < threshold]
            
            if len(pruned_results) < len(valid_results):
                self.logger.info(f"Early pruning: kept {len(pruned_results)}/{len(valid_results)} candidates")
            
            return pruned_results
        
        return valid_results
    
    def update_best_candidates(self, new_results: List[Dict[str, Any]]):
        """Update the list of best candidates."""
        # Combine with existing candidates
        all_candidates = self.best_candidates + new_results
        
        # Sort by score (ascending - lower is better)
        all_candidates.sort(key=lambda x: x.get("score", float("inf")))
        
        # Keep top K
        self.best_candidates = all_candidates[:self.config.top_k_keep]
        
        # Update best score
        if self.best_candidates:
            new_best = self.best_candidates[0].get("score", float("inf"))
            if new_best < self.best_score - self.config.early_stop_threshold:
                self.best_score = new_best
                self.patience_counter = 0  # Reset patience
            else:
                self.patience_counter += 1
    
    def should_stop_early(self) -> bool:
        """Check if early stopping criteria are met."""
        return self.patience_counter >= self.config.early_stop_patience
    
    def run_optimization(self, 
                        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
                        notes: Optional[str] = None) -> Dict[str, Any]:
        """Run the complete DCA optimization process."""
        self.logger.info("Starting DCA optimization with new evaluation contract")
        
        # Initialize experiment
        exp_id = self.create_experiment(notes)
        self.start_time = time.time()
        
        try:
            for batch_idx in range(self.config.max_batches):
                batch_start = time.time()
                
                # Generate parameters for this batch
                param_batch = self.generate_random_parameters(self.config.n_candidates_per_batch)
                
                # Evaluate batch
                batch_results = self.evaluate_batch_parallel(param_batch)
                self.total_evaluations += len(batch_results)
                
                # Apply early pruning
                pruned_results = self.early_pruning(batch_results)
                
                # Update best candidates
                self.update_best_candidates(pruned_results)
                
                # Update statistics
                batch_time = time.time() - batch_start
                self.stats["batches_completed"] = batch_idx + 1
                self.stats["total_time"] = time.time() - self.start_time
                self.stats["evaluations_per_second"] = self.total_evaluations / self.stats["total_time"]
                
                # Count sanity violations and wave patterns
                for result in batch_results:
                    sanity = result.get("sanity", {})
                    if any(sanity.values()):
                        self.stats["sanity_violations"] += 1
                    
                    if self.config.wave_pattern and result.get("penalties", {}).get("P_wave", 0) < 0:
                        self.stats["wave_pattern_rewards"] += 1
                
                # Progress callback
                if progress_callback:
                    progress_info = {
                        "batch": batch_idx + 1,
                        "total_batches": self.config.max_batches,
                        "best_score": self.best_score,
                        "total_evaluations": self.total_evaluations,
                        "evaluations_per_second": self.stats["evaluations_per_second"],
                        "batch_time": batch_time,
                        "candidates_kept": len(self.best_candidates)
                    }
                    progress_callback(progress_info)
                
                # Log progress
                self.logger.info(
                    f"Batch {batch_idx + 1}/{self.config.max_batches}: "
                    f"Best={self.best_score:.6f}, "
                    f"Evaluated={len(batch_results)}, "
                    f"Kept={len(pruned_results)}, "
                    f"Total_candidates={len(self.best_candidates)}, "
                    f"Time={batch_time:.2f}s"
                )
                
                # Check early stopping
                if self.should_stop_early():
                    self.logger.info(f"Early stopping after {batch_idx + 1} batches")
                    self.stats["early_stopped"] = True
                    break
                
                # Save intermediate results to database (every 10 batches)
                if (batch_idx + 1) % 10 == 0:
                    self.save_results_to_db()
            
            # Final save
            self.save_results_to_db()
            
            # Update experiment summary
            self.store.update_experiment_summary(
                exp_id,
                self.best_score,
                self.total_evaluations,
                self.stats["total_time"]
            )
            
            return self.get_optimization_results()
            
        except Exception as e:
            self.logger.error(f"Optimization failed: {e}")
            raise
    
    def save_results_to_db(self):
        """Save current best results to database."""
        if not self.current_experiment_id or not self.best_candidates:
            return
        
        # Save top candidates
        top_candidates = self.best_candidates[:100]  # Save top 100
        inserted = self.store.upsert_results(self.current_experiment_id, top_candidates)
        
        self.logger.info(f"Saved {inserted} results to database")
    
    def get_optimization_results(self) -> Dict[str, Any]:
        """Get final optimization results."""
        return {
            "experiment_id": self.current_experiment_id,
            "best_candidates": self.best_candidates[:50],  # Return top 50
            "statistics": {
                "total_evaluations": self.total_evaluations,
                "total_time": self.stats["total_time"],
                "evaluations_per_second": self.stats["evaluations_per_second"],
                "batches_completed": self.stats["batches_completed"],
                "early_stopped": self.stats["early_stopped"],
                "sanity_violations": self.stats["sanity_violations"],
                "wave_pattern_rewards": self.stats["wave_pattern_rewards"],
                "best_score": self.best_score,
                "candidates_found": len(self.best_candidates)
            },
            "config": {
                "overlap_range": f"{self.config.overlap_min}-{self.config.overlap_max}%",
                "orders_range": f"{self.config.orders_min}-{self.config.orders_max}",
                "wave_pattern": self.config.wave_pattern,
                "tail_cap": self.config.tail_cap,
                "scoring_weights": f"α={self.config.alpha}, β={self.config.beta}, γ={self.config.gamma}, λ={self.config.lambda_penalty}"
            }
        }


# Factory function for easy configuration
def create_dca_orchestrator(
    overlap_range: Tuple[float, float] = (10.0, 30.0),
    orders_range: Tuple[int, int] = (5, 15),
    wave_pattern: bool = False,
    n_candidates: int = 1000,
    max_batches: int = 100,
    **kwargs
) -> DCAOrchestrator:
    """Create a DCA orchestrator with common configurations."""
    
    config = DCAConfig(
        overlap_min=overlap_range[0],
        overlap_max=overlap_range[1],
        orders_min=orders_range[0],
        orders_max=orders_range[1],
        wave_pattern=wave_pattern,
        n_candidates_per_batch=n_candidates,
        max_batches=max_batches,
        **kwargs
    )
    
    return DCAOrchestrator(config)