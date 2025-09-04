"""
DCA Orchestrator - "İşlemden En Hızlı Çıkış" Optimization Engine
Integrates new evaluation contract with adaptive search, early pruning, and batch processing.
"""
from __future__ import annotations

import time
import pickle
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Tuple, Callable
import numpy as np
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from datetime import datetime
import random

from martingale_lab.optimizer.evaluation_engine import evaluation_function
from martingale_lab.storage.unified_store import UnifiedStore
from martingale_lab.storage.memory_manager import BoundedBestCandidates, log_memory_stats
from martingale_lab.core.config_classes import EvaluationConfig
from martingale_lab.core.config_adapter import config_to_flat_dict
from martingale_lab.utils.stable_id import make_stable_id
from martingale_lab.utils.logging import (
    get_orchestrator_logger, BatchAggregator, log_with_context
)

# Use the new centralized logging system
orch_logger = get_orchestrator_logger()

def _process_eval_worker(params: Dict[str, Any]) -> Dict[str, Any]:
    """Process-safe evaluation worker (top-level for pickling)."""
    try:
        result = evaluation_function(**params)
        # Attach parameter subset for downstream logging/DB payload
        param_subset = {
            "base_price": params["base_price"],
            "overlap_pct": params["overlap_pct"],
            "num_orders": params["num_orders"],
            "alpha": params["alpha"],
            "beta": params["beta"],
            "gamma": params["gamma"],
            "lambda_penalty": params["lambda_penalty"],
            "wave_pattern": params["wave_pattern"],
            "tail_cap": params["tail_cap"],
            "min_indent_step": params.get("min_indent_step"),
            "softmax_temp": params.get("softmax_temp"),
            "seed": params.get("seed"),
            # Generation controls
            "wave_mode": params.get("wave_mode"),
            "anchors": params.get("anchors"),
            "blocks": params.get("blocks"),
            "wave_amp_min": params.get("wave_amp_min"),
            "wave_amp_max": params.get("wave_amp_max"),
            # Hard constraints and bands
            "first_volume_target": params.get("first_volume_target"),
            "first_indent_target": params.get("first_indent_target"),
            "k_front": params.get("k_front"),
            "front_cap": params.get("front_cap"),
            "g_min": params.get("g_min"),
            "g_max": params.get("g_max"),
            "g_min_post": params.get("g_min_post"),
            "g_max_post": params.get("g_max_post"),
            "isotonic_tail": params.get("isotonic_tail"),
            # Penalty weights/preset
            "penalty_preset": params.get("penalty_preset"),
            "w_fixed": params.get("w_fixed"),
            "w_sec": params.get("w_sec"),
            "w_band": params.get("w_band"),
            "w_front": params.get("w_front"),
            "w_tv": params.get("w_tv"),
            "w_wave": params.get("w_wave"),
            "w_sens": params.get("w_sens"),
            "sens_min": params.get("sens_min"),
            "w_template": params.get("w_template"),
            "template_close": params.get("template_close"),
        }
        result["params"] = param_subset
        return result
    except Exception as e:
        return {
            "score": float("inf"),
            "max_need": float("inf"),
            "var_need": float("inf"),
            "tail": float("inf"),
            "schedule": {},
            "sanity": {"max_need_mismatch": True, "collapse_indents": True, "tail_overflow": True, "error": True, "reason": str(e)},
            "diagnostics": {"wci": 0.0, "sign_flips": 0, "gini": 1.0, "entropy": 0.0},
            "penalties": {},
            "params": params,
            "stable_id": None,
            "error": str(e),
        }

def generate_run_id() -> str:
    """Generate a unique run ID"""
    from datetime import datetime
    import secrets
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    random_suffix = secrets.token_hex(4).upper()
    return f"{timestamp}-{random_suffix}"

def create_crash_snapshot(run_id: str, params: Dict[str, Any], error_msg: str) -> str:
    """Create a crash snapshot file"""
    from pathlib import Path
    import json
    from datetime import datetime
    
    crash_dir = Path("db_results/crash_snapshots")
    crash_dir.mkdir(parents=True, exist_ok=True)
    
    snapshot = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "run_id": run_id,
        "params": params,
        "error": error_msg
    }
    
    filename = f"{run_id}_{int(time.time())}.json"
    filepath = crash_dir / filename
    
    with open(filepath, 'w') as f:
        json.dump(snapshot, f, indent=2, default=str)
    
    return str(filepath)

class LogContext:
    """Simple context manager for run/exp/batch IDs"""
    _run_id = None
    _exp_id = None
    _batch_idx = None
    
    @classmethod
    def set_run_id(cls, run_id: str):
        cls._run_id = run_id
    
    @classmethod  
    def set_exp_id(cls, exp_id: int):
        cls._exp_id = exp_id
        
    @classmethod
    def set_batch_idx(cls, batch_idx: int):
        cls._batch_idx = batch_idx


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
    
    # Optional: Use EvaluationConfig for structured parameters
    evaluation_config: Optional['EvaluationConfig'] = None
    
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
    wave_pattern: bool = True
    wave_strong_threshold: float = 50.0
    wave_weak_threshold: float = 10.0
    
    # Constraints
    tail_cap: float = 0.40
    min_indent_step: float = 0.05
    softmax_temp: float = 1.4

    # Generation controls
    wave_mode: str = "anchors"  # [anchors, blocks]
    anchors: int = 9
    blocks: int = 3
    wave_amp_min: float = 0.05
    wave_amp_max: float = 0.30

    # Hard constraint parameters
    first_volume: float = 0.01
    first_indent: float = 0.0
    g_pre_min: float = 1.01
    g_pre_max: float = 1.20
    g_post_min: float = 1.01
    g_post_max: float = 1.30
    front_cap: float = 5.0
    k_front: int = 3
    isotonic_tail: bool = False
    
    # New hard constraints
    m2_min: float = 0.10
    m2_max: float = 1.00
    m_min: float = 0.05
    m_max: float = 1.00
    firstK_min: float = 1.0
    strict_inc_eps: float = 1e-5
    
    # New HC parameters (HC0-HC7)
    second_upper_c2: float = 2.0
    m_head: float = 0.40
    m_tail: float = 0.20
    tau_scale: float = 1/3
    slope_cap: float = 0.25
    q1_cap: float = 22.0
    tail_floor: float = 32.0
    head_budget_pct: float = 2.0
    use_head_budget: bool = False
    use_hc0_bootstrap: bool = True
    
    # New soft penalties
    target_std: float = 0.10
    w_varm: float = 2.0
    w_blocks: float = 1.0
    use_entropy: bool = False
    entropy_target: float = 1.0

    # Penalty weight preset and overrides
    penalty_preset: Optional[str] = None
    w_fixed: float = 3.0
    w_second: float = 3.0
    w_gband: float = 2.0
    w_front: float = 3.0
    w_tv: float = 3.5
    w_wave: float = 1.0
    # New penalties (sensitivity/template)
    w_sens: float = 1.0
    sens_min: float = 0.25
    w_template: float = 0.8
    template_close: float = 0.6
    
    # Early stopping
    early_stop_threshold: float = 1e-6  # Stop if improvement < threshold
    early_stop_patience: int = 10  # Number of batches without improvement
    
    # Parallelization
    n_workers: int = 4
    
    # Random seed
    random_seed: Optional[int] = None
    
    # Logging configuration
    log_eval_sample: float = 0.0  # Per-evaluation log sampling rate (0.0-1.0)
    log_every_batch: int = 1      # Log batch summary every N batches
    max_time_sec: Optional[int] = None  # Maximum runtime in seconds
    
    # Schedule normalization parameters
    post_round_2dp: bool = False
    post_round_strategy: str = "tail-first"
    post_round_m2_tolerance: float = 0.05
    post_round_keep_v1_band: bool = True
    # Post-normalization smoothing (optional)
    post_norm_smoothing: bool = False
    smoothing_alpha: float = 0.15

    # Novelty filter parameters
    diversity_min_l1: float = 0.8
    novelty_k: int = 500
    diversity_metric: str = "l1"  # [l1, emd]


class DCAOrchestrator:
    """Main orchestrator for DCA optimization with new evaluation contract."""
    
    def __init__(self, config: DCAConfig, store: Optional[UnifiedStore] = None, 
                 run_id: Optional[str] = None, orch_config: Optional[OrchestratorConfig] = None,
                 workers_mode: str = "thread"):
        self.config = config
        self.orch_config = orch_config or OrchestratorConfig()
        self.store = store or UnifiedStore(max_candidates_memory=500)
        self.logger = orch_logger
        self.run_id = run_id or generate_run_id()
        self.workers_mode = workers_mode if workers_mode in ("thread", "process") else "thread"
        
        # Set context for logging
        LogContext.set_run_id(self.run_id)
        
        # State tracking
        self.current_experiment_id: Optional[int] = None
        # Use bounded memory-safe candidate storage
        self.best_candidates = BoundedBestCandidates(
            max_size=self.orch_config.prune_min_keep * 10,  # Keep 10x minimum
            score_key="score"
        )
        self.batch_count = 0
        self.total_evaluations = 0
        self.start_time = 0.0
        self.best_score = float("inf")
        self.best_score_so_far = float("inf")
        self.stalls = 0
        self.early_stop_reason: Optional[str] = None
        
        # Checkpoint state
        self.kept_total = 0  # Total candidates kept across all batches
        self.rng_state = None  # Random number generator state for resuming
        # Persistent RNG for deterministic continuity across batches
        self.rng = np.random.default_rng(self.config.random_seed)
        
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
        # Novelty filter pool of normalized volume shapes (list of numpy arrays)
        self._novelty_pool: List[np.ndarray] = []
    
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
        
        self.current_experiment_id = self.store.create_experiment(
            run_id=self.run_id,
            orchestrator="DCAOrchestrator", 
            config=config_dict,
            notes=notes
        )
        LogContext.set_exp_id(self.current_experiment_id)
        
        return self.current_experiment_id
    
    def generate_random_parameters(self, n_samples: int) -> List[Dict[str, Any]]:
        """Generate random parameter combinations for evaluation."""
        parameters = []
        for _ in range(n_samples):
            # Random overlap and orders
            overlap_pct = float(self.rng.uniform(self.config.overlap_min, self.config.overlap_max))
            num_orders = int(self.rng.integers(self.config.orders_min, self.config.orders_max + 1))
            
            # Random seed for evaluation
            eval_seed = int(self.rng.integers(0, 2**31 - 1))
            
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
                "softmax_temp": self.config.softmax_temp,
                # Generation controls
                "wave_mode": self.config.wave_mode,
                "anchors": self.config.anchors,
                "blocks": self.config.blocks,
                "wave_amp_min": self.config.wave_amp_min,
                "wave_amp_max": self.config.wave_amp_max,
                # Hard constraints and bands
                "first_volume_target": self.config.first_volume,
                "first_indent_target": self.config.first_indent,
                "k_front": self.config.k_front,
                "front_cap": self.config.front_cap,
                "g_min": self.config.g_pre_min,
                "g_max": self.config.g_pre_max,
                "g_min_post": self.config.g_post_min,
                "g_max_post": self.config.g_post_max,
                "isotonic_tail": self.config.isotonic_tail,
                # New hard constraints
                "m2_min": self.config.m2_min,
                "m2_max": self.config.m2_max,
                "m_min": self.config.m_min,
                "m_max": self.config.m_max,
                "firstK_min": self.config.firstK_min,
                "strict_inc_eps": self.config.strict_inc_eps,
                # New HC parameters (HC0-HC7)
                "second_upper_c2": self.config.second_upper_c2,
                "m_head": self.config.m_head,
                "m_tail": self.config.m_tail,
                "tau_scale": self.config.tau_scale,
                "slope_cap": self.config.slope_cap,
                "q1_cap": self.config.q1_cap,
                "tail_floor": self.config.tail_floor,
                "head_budget_pct": self.config.head_budget_pct,
                "use_head_budget": self.config.use_head_budget,
                "use_hc0_bootstrap": self.config.use_hc0_bootstrap,
                # Penalty weights/preset
                "penalty_preset": self.config.penalty_preset,
                "w_fixed": self.config.w_fixed,
                "w_sec": self.config.w_second,
                "w_band": self.config.w_gband,
                "w_front": self.config.w_front,
                "w_tv": self.config.w_tv,
                "w_wave": self.config.w_wave,
                # Schedule normalization
                "post_round_2dp": self.config.post_round_2dp,
                "post_round_strategy": self.config.post_round_strategy,
                "post_round_m2_tolerance": self.config.post_round_m2_tolerance,
                "post_round_keep_v1_band": self.config.post_round_keep_v1_band,
                # Post-normalization smoothing
                "post_norm_smoothing": self.config.post_norm_smoothing,
                "smoothing_alpha": self.config.smoothing_alpha,
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
            
            # Attach parameter subset for downstream logging/DB payload
            param_subset = {
                "base_price": params["base_price"],
                "overlap_pct": params["overlap_pct"],
                "num_orders": params["num_orders"],
                "alpha": params["alpha"],
                "beta": params["beta"],
                "gamma": params["gamma"],
                "lambda_penalty": params["lambda_penalty"],
                "wave_pattern": params["wave_pattern"],
                "tail_cap": params["tail_cap"],
                "min_indent_step": params.get("min_indent_step"),
                "softmax_temp": params.get("softmax_temp"),
                "seed": params.get("seed"),
                # Generation controls
                "wave_mode": params.get("wave_mode"),
                "anchors": params.get("anchors"),
                "blocks": params.get("blocks"),
                "wave_amp_min": params.get("wave_amp_min"),
                "wave_amp_max": params.get("wave_amp_max"),
                # Hard constraints and bands
                "first_volume_target": params.get("first_volume_target"),
                "first_indent_target": params.get("first_indent_target"),
                "k_front": params.get("k_front"),
                "front_cap": params.get("front_cap"),
                "g_min": params.get("g_min"),
                "g_max": params.get("g_max"),
                "g_min_post": params.get("g_min_post"),
                "g_max_post": params.get("g_max_post"),
                "isotonic_tail": params.get("isotonic_tail"),
                # Penalty weights/preset
                "penalty_preset": params.get("penalty_preset"),
                "w_fixed": params.get("w_fixed"),
                "w_sec": params.get("w_sec"),
                "w_band": params.get("w_band"),
                "w_front": params.get("w_front"),
                "w_tv": params.get("w_tv"),
                "w_wave": params.get("w_wave"),
                # New penalties
                "w_sens": params.get("w_sens"),
                "sens_min": params.get("sens_min"),
                "w_template": params.get("w_template"),
                "template_close": params.get("template_close"),
            }
            result["params"] = param_subset
            
            return result
            
        except Exception as e:
            self.stats["evals_failed"] += 1
            
            # Log evaluation error with crash snapshot
            error_msg = str(e)
            crash_file = create_crash_snapshot(self.run_id, params, error_msg)
            
            self.logger.error(
                f"Evaluation error: {error_msg}",
                extra={
                    "event": "EVAL_ERROR",
                    "run_id": self.run_id,
                    "exp_id": self.current_experiment_id,
                    "error": error_msg,
                    "crash_file": crash_file,
                    "overlap": params.get("overlap_pct"),
                    "orders": params.get("num_orders")
                }
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
            Executor = ProcessPoolExecutor if self.workers_mode == "process" else ThreadPoolExecutor
            with Executor(max_workers=self.config.n_workers) as executor:
                if self.workers_mode == "process":
                    future_to_params = {executor.submit(_process_eval_worker, params): params for params in param_batch}
                else:
                    future_to_params = {executor.submit(self.evaluate_candidate, params): params for params in param_batch}
                
                for future in as_completed(future_to_params):
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        params = future_to_params[future]
                        self.logger.error(f"Parallel evaluation failed for {params}: {e}")
                        # Fallback to sequential evaluation for this item
                        results.append(self.evaluate_candidate(params))
        
        # Apply novelty filter within-batch: build pool incrementally to penalize clones immediately
        if results:
            filtered: List[Dict[str, Any]] = []
            l1_vals: List[float] = []
            for res in results:
                try:
                    sch = res.get("schedule", {})
                    v_norm = sch.get("volume_pct_norm") or sch.get("volume_pct")
                    if not v_norm:
                        filtered.append(res)
                        continue
                    # Canonicalize with 2dp rounding and ensure sum 100
                    v_arr = np.asarray([round(float(x), 2) for x in v_norm], dtype=np.float64)
                    s = float(np.sum(v_arr))
                    if s > 1e-12:
                        v_arr = v_arr * (100.0 / s)
                    if np.sum(v_arr) > 0:
                        v_arr = v_arr / np.sum(v_arr) * 100.0
                    # Compute min distance versus current pool
                    min_l1 = 1e9
                    for p in self._novelty_pool:
                        if len(p) != len(v_arr):
                            continue
                        if self.config.diversity_metric == "emd":
                            # Wasserstein-1 via CDF difference (O(n))
                            c1 = np.cumsum(v_arr) / 100.0
                            c2 = np.cumsum(p) / 100.0
                            d = float(np.sum(np.abs(c1 - c2)) / len(v_arr))
                        else:
                            # Default L1 on percentages normalized by 100
                            d = float(np.sum(np.abs(v_arr - p)) / 100.0)
                        if d < min_l1:
                            min_l1 = d
                            if min_l1 < self.config.diversity_min_l1:
                                break
                    l1_vals.append(min_l1 if np.isfinite(min_l1) else float('nan'))
                    if min_l1 < self.config.diversity_min_l1:
                        # Strong downweight: inflate score to deprioritize clones but still retain for stats
                        res["score"] = float(res.get("score", float("inf")) + 1000.0 * (self.config.diversity_min_l1 - min_l1))
                        filtered.append(res)
                        # Do not add shape to pool
                    else:
                        filtered.append(res)
                        # Add to novelty pool
                        self._novelty_pool.append(v_arr)
                        if len(self._novelty_pool) > max(1, self.config.novelty_k):
                            # Simple LRU: keep most recent K
                            self._novelty_pool = self._novelty_pool[-self.config.novelty_k:]
                except Exception:
                    filtered.append(res)
            results = filtered
            # Log novelty stats
            try:
                reject_rate = float(sum(1 for x in l1_vals if x < self.config.diversity_min_l1) / max(1, len(l1_vals)))
                avg_l1 = float(np.nanmean(np.asarray(l1_vals, dtype=np.float64))) if l1_vals else float('nan')
            except Exception:
                reject_rate = 0.0
                avg_l1 = float('nan')
            self.logger.info(
                f"NOVELTY: reject_rate={reject_rate:.2%} avg_L1={avg_l1:.3f} pool={len(self._novelty_pool)}",
                extra={
                    "event": "NOVELTY.STATS",
                    "run_id": self.run_id,
                    "exp_id": self.current_experiment_id,
                    "batch_idx": self.batch_count,
                    "reject_rate": reject_rate,
                    "avg_l1": avg_l1,
                    "pool_size": len(self._novelty_pool),
                }
            )
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
            f"ORCH.PRUNE: mode={self.orch_config.prune_mode}, kept={kept}, pruned={pruned}",
            extra={
                "event": "ORCH.PRUNE",
                "run_id": self.run_id,
                "exp_id": self.current_experiment_id,
                "batch_idx": batch_idx,
                "mode": self.orch_config.prune_mode,
                "best": float(best),
                "threshold": float(thresh) if thresh is not None else float('nan'),
                "kept": kept,
                "pruned": pruned
            }
        )
        
        # Return kept results
        return [r for i, r in enumerate(results) if keep_mask[i]]
    
    def update_best_candidates(self, new_results: List[Dict[str, Any]]):
        """Update the list of best candidates with memory-safe storage."""
        # Add new results to bounded storage (automatic pruning)
        kept = self.best_candidates.add_batch(new_results)
        
        # Get current best for tracking
        best_list = self.best_candidates.get_best(1)
        
        # Update best score and early stopping logic
        if best_list:
            new_best = best_list[0].get("score", float("inf"))
            
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
                f"ORCH.EARLY_STOP: {reason}, stalls={self.stalls}",
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
    
    def save_checkpoint(self, batch_idx: int):
        """Save checkpoint after batch completion."""
        try:
            # Save RNG state
            self.rng_state = self.rng.bit_generator.state
            
            checkpoint_data = {
                "run_id": self.run_id,
                "exp_id": self.current_experiment_id,
                "batch_idx": batch_idx,
                "kept_total": self.kept_total,
                "best_score": self.best_score,
                "rng_state": self.rng_state,
                "total_evaluations": self.total_evaluations,
                "best_candidates_count": len(self.best_candidates),
                "stalls": self.stalls,
                "best_score_so_far": self.best_score_so_far
            }
            
            # Save to checkpoint store
            self.store.save_checkpoint(
                run_id=self.run_id,
                batch_idx=batch_idx,
                checkpoint_data=checkpoint_data
            )
            
            self.logger.debug(
                f"Checkpoint saved: batch={batch_idx}, kept_total={self.kept_total}, best={self.best_score:.6f}",
                extra={
                    "event": "CHECKPOINT_SAVED",
                    "run_id": self.run_id,
                    "exp_id": self.current_experiment_id,
                    "batch_idx": batch_idx,
                    "kept_total": self.kept_total,
                    "best_score": self.best_score
                }
            )
            
        except Exception as e:
            self.logger.warning(
                f"Failed to save checkpoint: {e}",
                extra={
                    "event": "CHECKPOINT_ERROR",
                    "run_id": self.run_id,
                    "exp_id": self.current_experiment_id,
                    "batch_idx": batch_idx,
                    "error": str(e)
                }
            )
    
    def load_checkpoint(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Load checkpoint for resuming optimization."""
        try:
            checkpoint_result = self.store.load_checkpoint(run_id)
            if checkpoint_result:
                batch_idx, checkpoint_data = checkpoint_result
            else:
                checkpoint_data = None
            if checkpoint_data is None:
                return None
            
            # Restore state
            self.run_id = checkpoint_data["run_id"]  # Use the checkpoint run_id
            self.current_experiment_id = checkpoint_data["exp_id"]
            self.batch_count = checkpoint_data["batch_idx"]
            self.kept_total = checkpoint_data["kept_total"]
            self.best_score = checkpoint_data["best_score"]
            self.rng_state = checkpoint_data["rng_state"]
            self.total_evaluations = checkpoint_data["total_evaluations"]
            self.stalls = checkpoint_data["stalls"]
            self.best_score_so_far = checkpoint_data["best_score_so_far"]
            # Restore RNG state for deterministic continuation
            try:
                if self.rng_state is not None:
                    self.rng.bit_generator.state = self.rng_state
            except Exception:
                # Recreate RNG from seed as fallback
                self.rng = np.random.default_rng(self.config.random_seed)
            
            # Set logging context
            LogContext.set_run_id(self.run_id)
            LogContext.set_exp_id(self.current_experiment_id)
            
            self.logger.info(
                f"Resuming from checkpoint: batch={self.batch_count}, kept_total={self.kept_total}, best={self.best_score:.6f}",
                extra={
                    "event": "CHECKPOINT_LOADED",
                    "run_id": self.run_id,
                    "exp_id": self.current_experiment_id,
                    "batch_idx": self.batch_count,
                    "kept_total": self.kept_total,
                    "best_score": self.best_score,
                    "total_evaluations": self.total_evaluations
                }
            )
            
            return checkpoint_data
            
        except Exception as e:
            self.logger.error(
                f"Failed to load checkpoint: {e}",
                extra={
                    "event": "CHECKPOINT_LOAD_ERROR",
                    "run_id": run_id,
                    "error": str(e)
                }
            )
            return None
    
    def get_resumable_runs(self) -> List[Dict[str, Any]]:
        """Get list of runs that can be resumed."""
        # TODO: Implement in UnifiedStore
        return []
    
    def run_optimization(self, 
                        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
                        notes: Optional[str] = None,
                        max_time_sec: Optional[int] = None,
                        resume_from: Optional[str] = None) -> Dict[str, Any]:
        """Run the complete DCA optimization process with timeout support and resume capability."""
        # Handle resume if requested
        if resume_from:
            checkpoint_data = self.load_checkpoint(resume_from)
            if checkpoint_data is None:
                raise ValueError(f"No checkpoint found for run_id: {resume_from}")
            
            # Start from the next batch after the checkpoint
            start_batch = checkpoint_data["batch_idx"] + 1
            self.logger.info(f"Resuming optimization from batch {start_batch}")
        else:
            # Initialize experiment for new run
            exp_id = self.create_experiment(notes)
            start_batch = 0
        
        self.start_time = time.time()
        
        # Use config max_time_sec if not provided in call
        if max_time_sec is None:
            max_time_sec = self.config.max_time_sec
        
        # Calculate total combinations for exhaustive mode
        if self.orch_config.exhaustive_mode:
            overlap_steps = int((self.config.overlap_max - self.config.overlap_min) / 0.5) + 1
            orders_range = self.config.orders_max - self.config.orders_min + 1
            total_combinations = overlap_steps * orders_range * self.config.n_candidates_per_batch
            self.orch_config.exhaustive_progress_total = total_combinations
        
        # Log orchestrator start with config snapshot
        log_with_context(
            self.logger, "info",
            f"DCA Orchestration starting: run_id={self.run_id} exp_id={self.current_experiment_id}",
            run_id=self.run_id,
            exp_id=self.current_experiment_id,
            event="ORCH.START",
            adapter="DCAOrchestrator",
            overlap_range=f"{self.config.overlap_min}-{self.config.overlap_max}",
            orders_range=f"{self.config.orders_min}-{self.config.orders_max}",
            max_time_sec=max_time_sec,
            batch_size=self.config.n_candidates_per_batch,
            max_batches=self.config.max_batches,
            resume_from=resume_from
        )
        
        timeout_reached = False
        
        try:
            for batch_idx in range(start_batch, self.config.max_batches):
                batch_start = time.time()
                LogContext.set_batch_idx(batch_idx)
                self.batch_count = batch_idx
                
                # Check timeout before starting batch
                if max_time_sec and (time.time() - self.start_time) >= max_time_sec:
                    timeout_reached = True
                    log_with_context(
                        self.logger, "info",
                        f"Timeout reached ({max_time_sec}s), stopping optimization",
                        run_id=self.run_id,
                        exp_id=self.current_experiment_id,
                        event="ORCH.TIMEOUT",
                        batch_idx=batch_idx,
                        elapsed_sec=time.time() - self.start_time
                    )
                    break
                
                # Generate parameters for this batch
                param_batch = self.generate_random_parameters(self.config.n_candidates_per_batch)
                
                # Evaluate batch
                batch_results = self.evaluate_batch_parallel(param_batch)
                self.total_evaluations += len(batch_results)
                
                # Apply pruning
                pre_prune_count = len(batch_results)
                pruned_results = self.apply_pruning(batch_results, batch_idx)
                post_prune_count = len(pruned_results)
                # Pruning diagnostics for visibility
                self.logger.info(
                    f"PRUNE.COUNTS pre={pre_prune_count} post={post_prune_count}",
                    extra={
                        "event": "PRUNE.COUNTS",
                        "run_id": self.run_id,
                        "exp_id": self.current_experiment_id,
                        "batch_idx": batch_idx,
                        "pre_prune": pre_prune_count,
                        "post_prune": post_prune_count,
                    }
                )
                
                # Update best candidates
                self.update_best_candidates(pruned_results)
                
                # Update kept_total counter
                kept_in_this_batch = len(pruned_results)
                self.kept_total += kept_in_this_batch

                # Persist pruned results to database with deterministic stable_id
                db_items: List[Dict[str, Any]] = []
                for cand_idx, res in enumerate(pruned_results):
                    params_info = res.get("params", {})
                    payload = {
                        "schedule": res.get("schedule", {}),
                        "overlap": params_info.get("overlap_pct"),
                        "orders": params_info.get("num_orders"),
                        "alpha": params_info.get("alpha"),
                        "beta": params_info.get("beta"),
                        "gamma": params_info.get("gamma"),
                        "lambda_penalty": params_info.get("lambda_penalty"),
                        "wave_pattern": params_info.get("wave_pattern"),
                        "tail_cap": params_info.get("tail_cap"),
                        "min_indent_step": params_info.get("min_indent_step"),
                        "softmax_temp": params_info.get("softmax_temp"),
                        "seed": params_info.get("seed"),
                        "candidate_uid": f"{self.run_id}:{batch_idx}:{cand_idx}",
                        "payload_schema": 2,
                        "param_repr": {
                            "mode": params_info.get("wave_mode", res.get("diagnostics", {}).get("wave_mode", "anchors")),
                            "orders": params_info.get("num_orders"),
                            "anchors": params_info.get("anchors", res.get("diagnostics", {}).get("anchors", None)),
                            "blocks": params_info.get("blocks", res.get("diagnostics", {}).get("blocks", None)),
                            "anchor_points": res.get("diagnostics", {}).get("anchor_points"),
                            "anchor_logv": res.get("diagnostics", {}).get("anchor_logv"),
                            "g_pre_band": [params_info.get("g_min", 1.01), params_info.get("g_max", 1.20)],
                            "g_post_band": [params_info.get("g_min_post", 1.01), params_info.get("g_max_post", 1.30)],
                            "front_cap": params_info.get("front_cap", 5.0),
                            "k_front": params_info.get("k_front", 3),
                            "isotonic_tail": bool(params_info.get("isotonic_tail", False))
                        },
                        "run_id": self.run_id,
                        "seed": params_info.get("seed"),
                    }
                    sid = make_stable_id(payload, run_id=self.run_id, batch_idx=batch_idx, cand_idx=cand_idx)
                    db_items.append({
                        "stable_id": sid,
                        "score": float(res.get("score", float("inf"))),
                        "schedule": res.get("schedule", {}),
                        "payload": payload,
                        "sanity": res.get("sanity", {}),
                        "diagnostics": res.get("diagnostics", {}),
                        "penalties": res.get("penalties", {}),
                        "created_at": datetime.now().isoformat(),
                    })

                rows_written = 0
                if db_items:
                    try:
                        rows_written = self.store.insert_results_batch(self.current_experiment_id, db_items)
                    except Exception as e:
                        self.logger.error(
                            f"DB upsert failed for batch {batch_idx}: {e}",
                            extra={
                                "event": "DB.UPSERT_ERROR",
                                "run_id": self.run_id,
                                "exp_id": self.current_experiment_id,
                                "batch_idx": batch_idx,
                                "error": str(e),
                                "items": len(db_items),
                            }
                        )
                        rows_written = 0

                # Sampled debug for first 2 items
                for sample_item in db_items[: min(2, len(db_items))]:
                    if random.random() < 0.001:
                        self.logger.debug(
                            f"DB row sample sid={sample_item['stable_id']} score={sample_item['score']:.6f} overlap={sample_item['payload'].get('overlap')} orders={sample_item['payload'].get('orders')}",
                            extra={
                                "event": "DB.ROW_SAMPLE",
                                "run_id": self.run_id,
                                "exp_id": self.current_experiment_id,
                                "batch_idx": batch_idx,
                                "stable_id": sample_item["stable_id"],
                                "score": sample_item["score"],
                                "overlap": sample_item["payload"].get("overlap"),
                                "orders": sample_item["payload"].get("orders"),
                            }
                        )

                # Log batch DB write summary
                self.logger.info(
                    f"DB wrote rows={rows_written} kept_batch={kept_in_this_batch} kept_total={self.kept_total}",
                    extra={
                        "event": "DB.BATCH_WRITE",
                        "run_id": self.run_id,
                        "exp_id": self.current_experiment_id,
                        "batch_idx": batch_idx,
                        "db_rows_written": rows_written,
                        "kept_batch": kept_in_this_batch,
                        "kept_total": self.kept_total,
                    }
                )
                
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
                
                # Use the new single-line batch logging format
                prune_mode = "exhaustive" if self.orch_config.exhaustive_mode else self.orch_config.prune_mode
                BatchAggregator.log_batch_summary(
                    batch_idx=batch_idx,
                    total_batches=self.config.max_batches,
                    best_score=self.best_score,
                    evaluations=len(batch_results),
                    candidates_kept=post_prune_count,  # Legacy parameter
                    prune_mode=prune_mode,
                    evaluations_per_second=self.stats["evaluations_per_second"],
                    log_every_batch=self.config.log_every_batch,
                    logger=self.logger,
                    kept_in_this_batch=kept_in_this_batch,
                    kept_total=self.kept_total
                )
                
                # Save checkpoint after batch completion
                self.save_checkpoint(batch_idx)
                
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
                        "batch_evaluations": len(batch_results),
                        "prune_mode": prune_mode,
                        "elapsed_time": self.stats["total_time"],
                        "timeout_remaining": max_time_sec - (time.time() - self.start_time) if max_time_sec else None,
                        "kept_in_this_batch": kept_in_this_batch,
                        "kept_total": self.kept_total
                    }
                    progress_callback(progress_data)
                
                # Check early stopping
                should_stop, stop_reason = self.should_stop_early()
                if should_stop:
                    self.stats["early_stopped"] = True
                    break
                
                # Check timeout after batch completion
                if max_time_sec and (time.time() - self.start_time) >= max_time_sec:
                    timeout_reached = True
                    log_with_context(
                        self.logger, "info",
                        f"Timeout reached after batch completion ({max_time_sec}s)",
                        run_id=self.run_id,
                        exp_id=self.current_experiment_id,
                        event="ORCH.TIMEOUT_POST_BATCH",
                        batch_idx=batch_idx,
                        elapsed_sec=time.time() - self.start_time
                    )
                    break
            
            # Calculate final statistics
            self.stats["total_time"] = time.time() - self.start_time
            self.stats["evaluations_per_second"] = self.total_evaluations / max(self.stats["total_time"], 1e-6)
            self.stats["candidates_found"] = len(self.best_candidates)
            self.stats["timeout_reached"] = timeout_reached
            
            # Log completion
            log_with_context(
                self.logger, "info",
                f"DCA Orchestration completed: {self.stats['batches_completed']} batches, {self.total_evaluations} evaluations",
                run_id=self.run_id,
                exp_id=self.current_experiment_id,
                event="ORCH.COMPLETE",
                best_score=self.best_score,
                total_evaluations=self.total_evaluations,
                total_time=self.stats["total_time"],
                timeout_reached=timeout_reached,
                early_stopped=self.stats["early_stopped"],
                kept_total=self.kept_total
            )
            
            return self.get_optimization_results()
            
        except Exception as e:
            # Create crash snapshot
            error_context = {
                "run_id": self.run_id,
                "experiment_id": self.current_experiment_id,
                "batch_count": self.batch_count,
                "total_evaluations": self.total_evaluations,
                "config": self.config.__dict__,
                "orch_config": self.orch_config.__dict__
            }
            
            snapshot_path = create_crash_snapshot(self.run_id, error_context, str(e))
            
            log_with_context(
                self.logger, "error",
                f"DCA Orchestration failed: {str(e)}",
                run_id=self.run_id,
                exp_id=self.current_experiment_id,
                event="ORCH.ERROR",
                error=str(e),
                snapshot_path=snapshot_path
            )
            
            # Re-raise the exception
            raise
    
    def save_results_to_db(self):
        """Save current best results to database."""
        if not self.current_experiment_id or not self.best_candidates:
            return
        
        # Save top candidates
        top_candidates = self.best_candidates.get_best(100)  # Save top 100
        inserted = self.store.insert_results_batch(self.current_experiment_id, top_candidates)
        
        self.logger.info(f"Saved {inserted} results to database")
    
    def get_optimization_results(self) -> Dict[str, Any]:
        """Get final optimization results."""
        return {
            "experiment_id": self.current_experiment_id,
            "best_candidates": self.best_candidates.get_best(50),  # Return top 50
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
