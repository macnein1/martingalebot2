"""
DCA Orchestrator - "İşlemden En Hızlı Çıkış" Optimization Engine
Integrates new evaluation contract with adaptive search, early pruning, and batch processing.
"""
from __future__ import annotations

import time
import logging
import traceback
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Tuple, Callable
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

from martingale_lab.optimizer.evaluation_engine import evaluation_function
from martingale_lab.storage.experiments_store import ExperimentsStore
from ui.utils.structured_logging import Events, orch_logger, LogContext, generate_run_id, create_crash_snapshot
from ui.utils.constants import Status


@dataclass
class OrchestratorConfig:
    """Configuration for orchestrator behavior (pruning, early stopping, etc.)."""
    # Pruning configuration
    prune_enabled: bool = True
    prune_mode: str = "quantile"    # ["quantile", "multiplier", "none"]
    prune_quantile: float = 0.5     # en kötü %50'yi kes (ör: 0.5)
    prune_multiplier: float = 1.20  # cand_score <= best_score * 1.20 ise tut
    prune_min_keep: int = 50        # en az bu kadar adayı KESME
    prune_grace_batches: int = 3    # ilk N batch'te kırpma YAPMA

    # Early stop configuration
    early_stop_enabled: bool = True
    early_stop_patience: int = 10   # kaç batch iyileşme olmazsa dur
    early_stop_delta: float = 1e-6  # iyileşme eşiği (daha iyi = daha küçük)

    # Full grid/exhaustive mode
    exhaustive_mode: bool = False   # True => hiç pruning yok, hiç early stop yok
    exhaustive_progress_total: Optional[int] = None  # toplam kombinasyon (UI progress)


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
    
    def __init__(self, config: DCAConfig, store: Optional[ExperimentsStore] = None, 
                 run_id: Optional[str] = None, orch_config: Optional[OrchestratorConfig] = None):
        self.config = config
        self.orch_config = orch_config or OrchestratorConfig()
        self.store = store or ExperimentsStore()
        self.logger = orch_logger
        self.run_id = run_id or generate_run_id()
        
        # Set context for logging
        LogContext.set_run_id(self.run_id)
        
        # State tracking
        self.current_experiment_id: Optional[int] = None
        self.best_candidates: List[Dict[str, Any]] = []
        self.batch_count = 0
        self.total_evaluations = 0
        self.start_time = 0.0
        self.best_score = float("inf")
        self.best_score_so_far = float("inf")
        self.stalls = 0
        self.early_stop_reason: Optional[str] = None
        
        # Statistics
        self.stats = {
            "total_time": 0.0,
            "evaluations_per_second": 0.0,
            "batches_completed": 0,
            "early_stopped": False,
            "sanity_violations": 0,
            "wave_pattern_rewards": 0,
            "evals_total": 0,
            "evals_ok": 0,
            "evals_failed": 0,
            "pruned": 0,
            "saved_rows": 0
        }
    
    def create_experiment(self, notes: Optional[str] = None) -> int:
        """Create a new experiment in the database."""
        config_dict = {
            "run_id": self.run_id,
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
        
        self.current_experiment_id = self.store.create_experiment("DCAOrchestrator", config_dict, self.run_id)
        LogContext.set_exp_id(self.current_experiment_id)
        
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
        self.stats["evals_total"] += 1
        
        try:
            result = evaluation_function(**params)
            
            # Check if evaluation was successful
            if result.get("score", float("inf")) == float("inf"):
                self.stats["evals_failed"] += 1
            else:
                self.stats["evals_ok"] += 1
            
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
            self.stats["evals_failed"] += 1
            
            # Log evaluation error with crash snapshot
            error_msg = str(e)
            crash_file = create_crash_snapshot(self.run_id, params, error_msg)
            
            self.logger.event(
                Events.EVAL_ERROR,
                error=error_msg,
                crash_file=crash_file,
                overlap=params.get("overlap_pct"),
                orders=params.get("num_orders")
            )
            
            return {
                "score": float("inf"),
                "max_need": float("inf"),
                "var_need": float("inf"),
                "tail": float("inf"),
                "schedule": {},
                "sanity": {"max_need_mismatch": True, "collapse_indents": True, "tail_overflow": True, "error": True, "reason": error_msg},
                "diagnostics": {"wci": 0.0, "sign_flips": 0, "gini": 1.0, "entropy": 0.0},
                "penalties": {},
                "params": params,
                "stable_id": None,
                "error": error_msg
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
    
    def apply_pruning(self, results: List[Dict[str, Any]], batch_idx: int) -> List[Dict[str, Any]]:
        """Apply pruning based on orchestrator configuration."""
        if not results:
            return results
        
        # Extract scores
        scores = np.array([r.get("score", float("inf")) for r in results])
        
        # Skip pruning if disabled or in grace period
        if not self.orch_config.prune_enabled or batch_idx < self.orch_config.prune_grace_batches:
            return results
        
        # Skip pruning in exhaustive mode
        if self.orch_config.exhaustive_mode:
            return results
        
        best = np.min(scores)
        keep_mask = np.ones_like(scores, dtype=bool)
        thresh = None
        
        # Apply pruning based on mode
        if self.orch_config.prune_mode == "quantile":
            thresh = np.quantile(scores, self.orch_config.prune_quantile)
            keep_mask = scores <= thresh + 1e-12
        elif self.orch_config.prune_mode == "multiplier":
            thresh = best * self.orch_config.prune_multiplier
            keep_mask = scores <= thresh + 1e-12
        
        # Apply min_keep safety
        if keep_mask.sum() < self.orch_config.prune_min_keep:
            # Keep the best prune_min_keep candidates
            order = np.argsort(scores)
            keep_mask[:] = False
            keep_mask[order[:self.orch_config.prune_min_keep]] = True
        
        kept = int(keep_mask.sum())
        pruned = int(len(scores) - kept)
        
        # Log pruning event
        self.logger.info(
            "ORCH.PRUNE",
            extra={
                "event": "ORCH.PRUNE",
                "run_id": self.run_id,
                "exp_id": self.current_experiment_id,
                "batch_idx": batch_idx,
                "mode": self.orch_config.prune_mode,
                "best": float(best),
                "threshold": float(thresh) if thresh is not None else np.nan,
                "kept": kept,
                "pruned": pruned
            }
        )
        
        # Return kept results
        return [r for i, r in enumerate(results) if keep_mask[i]]
    
    def update_best_candidates(self, new_results: List[Dict[str, Any]]):
        """Update the list of best candidates."""
        # Combine with existing candidates
        all_candidates = self.best_candidates + new_results
        
        # Sort by score (ascending - lower is better)
        all_candidates.sort(key=lambda x: x.get("score", float("inf")))
        
        # Keep top K
        self.best_candidates = all_candidates[:self.config.top_k_keep]
        
        # Update best score and early stopping logic
        if self.best_candidates:
            new_best = self.best_candidates[0].get("score", float("inf"))
            
            # Check for improvement
            improved = (new_best < self.best_score_so_far - self.orch_config.early_stop_delta)
            
            if improved:
                self.best_score_so_far = new_best
                self.stalls = 0
            else:
                self.stalls += 1
            
            self.best_score = new_best
    
    def should_stop_early(self) -> Tuple[bool, Optional[str]]:
        """Check if early stopping criteria are met."""
        should_stop = False
        reason = None
        
        if self.orch_config.early_stop_enabled and self.stalls >= self.orch_config.early_stop_patience:
            should_stop = True
            reason = f"no_improve_{self.stalls}_batches_delta_{self.orch_config.early_stop_delta}"
        
        # Never stop early in exhaustive mode
        if self.orch_config.exhaustive_mode:
            should_stop = False
            reason = None
        
        if should_stop:
            self.early_stop_reason = reason
            self.logger.info(
                "ORCH.EARLY_STOP",
                extra={
                    "event": "ORCH.EARLY_STOP",
                    "run_id": self.run_id,
                    "exp_id": self.current_experiment_id,
                    "batch_idx": self.batch_count,
                    "stalls": self.stalls,
                    "delta": self.orch_config.early_stop_delta,
                    "best": float(self.best_score_so_far)
                }
            )
        
        return should_stop, reason
    
    def run_optimization(self, 
                        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
                        notes: Optional[str] = None) -> Dict[str, Any]:
        """Run the complete DCA optimization process."""
        # Initialize experiment
        exp_id = self.create_experiment(notes)
        self.start_time = time.time()
        
        # Calculate total combinations for exhaustive mode
        if self.orch_config.exhaustive_mode:
            overlap_steps = int((self.config.overlap_max - self.config.overlap_min) / 0.5) + 1
            orders_range = self.config.orders_max - self.config.orders_min + 1
            total_combinations = overlap_steps * orders_range * self.config.n_candidates_per_batch
            self.orch_config.exhaustive_progress_total = total_combinations
        
        # Log orchestrator start with config snapshot
        self.logger.info(
            "ORCH.START",
            extra={
                "event": "ORCH.START",
                "run_id": self.run_id,
                "exp_id": self.current_experiment_id,
                "adapter": "DCAOrchestrator",
                "overlap_range": f"{self.config.overlap_min}-{self.config.overlap_max}",
                "orders_range": f"{self.config.orders_min}-{self.config.orders_max}",
                "alpha": self.config.alpha,
                "beta": self.config.beta,
                "gamma": self.config.gamma,
                "lambda_penalty": self.config.lambda_penalty,
                "wave_pattern": self.config.wave_pattern,
                "tail_cap": self.config.tail_cap,
                "n_candidates_per_batch": self.config.n_candidates_per_batch,
                "max_batches": self.config.max_batches,
                "prune_enabled": self.orch_config.prune_enabled,
                "prune_mode": self.orch_config.prune_mode,
                "early_stop_enabled": self.orch_config.early_stop_enabled,
                "exhaustive_mode": self.orch_config.exhaustive_mode,
                "exhaustive_progress_total": self.orch_config.exhaustive_progress_total
            }
        )
        
        try:
            for batch_idx in range(self.config.max_batches):
                batch_start = time.time()
                LogContext.set_batch_idx(batch_idx)
                self.batch_count = batch_idx
                
                # Generate parameters for this batch
                param_batch = self.generate_random_parameters(self.config.n_candidates_per_batch)
                
                # Evaluate batch
                batch_results = self.evaluate_batch_parallel(param_batch)
                self.total_evaluations += len(batch_results)
                
                # Apply pruning
                pre_prune_count = len(batch_results)
                pruned_results = self.apply_pruning(batch_results, batch_idx)
                post_prune_count = len(pruned_results)
                
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
                
                # Log batch summary
                mode = "exhaustive" if self.orch_config.exhaustive_mode else "adaptive"
                self.logger.info(
                    "BATCH_END",
                    extra={
                        "event": "BATCH_END",
                        "run_id": self.run_id,
                        "exp_id": self.current_experiment_id,
                        "batch_idx": batch_idx,
                        "best": float(self.best_score),
                        "evaluated": len(batch_results),
                        "kept": post_prune_count,
                        "pruned": pre_prune_count - post_prune_count,
                        "mode": mode,
                        "time_s": batch_time
                    }
                )
                
                # Call progress callback
                if progress_callback:
                    progress_data = {
                        "batch": batch_idx + 1,
                        "total_batches": self.config.max_batches,
                        "best_score": self.best_score,
                        "total_evaluations": self.total_evaluations,
                        "evaluations_per_second": self.stats["evaluations_per_second"],
                        "candidates_kept": len(self.best_candidates),
                        "batch_time": batch_time,
                        "mode": mode
                    }
                    progress_callback(progress_data)
                
                # Check early stopping
                should_stop, reason = self.should_stop_early()
                if should_stop:
                    self.logger.info(f"Early stopping after {batch_idx + 1} batches due to {reason}")
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
    # Orchestrator config parameters
    prune_enabled: bool = True,
    prune_mode: str = "quantile",
    prune_quantile: float = 0.5,
    prune_multiplier: float = 1.20,
    prune_min_keep: int = 50,
    prune_grace_batches: int = 3,
    early_stop_enabled: bool = True,
    early_stop_patience: int = 10,
    early_stop_delta: float = 1e-6,
    exhaustive_mode: bool = False,
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
    
    orch_config = OrchestratorConfig(
        prune_enabled=prune_enabled,
        prune_mode=prune_mode,
        prune_quantile=prune_quantile,
        prune_multiplier=prune_multiplier,
        prune_min_keep=prune_min_keep,
        prune_grace_batches=prune_grace_batches,
        early_stop_enabled=early_stop_enabled,
        early_stop_patience=early_stop_patience,
        early_stop_delta=early_stop_delta,
        exhaustive_mode=exhaustive_mode
    )
    
    return DCAOrchestrator(config, orch_config=orch_config)
