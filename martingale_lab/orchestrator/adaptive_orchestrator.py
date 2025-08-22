"""
Adaptive Orchestrator for martingale optimization.
"""
import time
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import logging

from ..interfaces import (
    SearchConfig, Candidate, ResultTrace, OptimizerAdapter,
    SearchResult, TraceResult, OptimizationStats
)
from ..core.reduction import TopKHeap, ConvergenceChecker, ResultAggregator
from ..storage.sqlite_store import SQLiteStore
from ..storage.experiments_store import ExperimentsStore


@dataclass
class OrchestrationConfig:
    """Configuration for adaptive orchestration."""
    # Phase budgets (as percentages of total)
    warmup_budget: float = 0.3  # 30% for coarse scan
    focus_budget: float = 0.4   # 40% for focused search
    fine_tune_budget: float = 0.3  # 30% for fine-tuning
    
    # Convergence parameters
    plateau_threshold: float = 1e-6  # ε for plateau detection
    plateau_window: int = 5  # Number of batches to check for plateau
    plateau_expansion: float = 0.2  # %X to expand range when plateau detected
    
    # Early stopping
    early_stop_patience: int = 10  # Batches without improvement
    min_improvement: float = 1e-4  # Minimum improvement threshold
    
    # Batch sizing
    target_eval_rate: float = 1000.0  # Evaluations per second target
    min_batch_size: int = 50
    max_batch_size: int = 1000
    
    # Top-N results
    top_n_results: int = 10
    
    # Database
    save_to_db: bool = True
    db_path: str = "martingale_optimization.db"


class AdaptiveOrchestrator:
    """Adaptive orchestration for martingale optimization."""
    
    def __init__(self, 
                 numba_adapter: OptimizerAdapter,
                 auto_batch_adapter: OptimizerAdapter,
                 config: Optional[OrchestrationConfig] = None):
        """Initialize adaptive orchestrator."""
        self.numba_adapter = numba_adapter
        self.auto_batch_adapter = auto_batch_adapter
        self.config = config or OrchestrationConfig()
        
        # Internal state
        self.top_k_heap = TopKHeap(self.config.top_n_results)
        self.convergence_checker = ConvergenceChecker(
            window_size=self.config.plateau_window,
            tolerance=self.config.plateau_threshold
        )
        self.result_aggregator = ResultAggregator()
        
        # Storage
        self.storage = SQLiteStore(self.config.db_path) if self.config.save_to_db else None
        self.exp_store = ExperimentsStore("db_results/experiments.db") if self.config.save_to_db else None
        self.experiment_id: Optional[int] = None
        
        # Session tracking
        self.session_id = f"session_{int(time.time())}"
        self.start_time = None
        self.phase_results: List[Tuple[str, SearchResult, TraceResult]] = []
        
        # Logging
        self.logger = logging.getLogger(__name__)
    
    def run(self, search_config: SearchConfig, 
            total_time_budget: Optional[float] = None,
            total_eval_budget: Optional[int] = None) -> Tuple[SearchResult, TraceResult, OptimizationStats]:
        """
        Run adaptive optimization orchestration.
        
        Args:
            search_config: Search configuration
            total_time_budget: Total time budget in seconds
            total_eval_budget: Total evaluation budget
            
        Returns:
            Tuple of (candidates, traces, statistics)
        """
        self.start_time = time.time()
        self.logger.info(f"Starting adaptive orchestration with session {self.session_id}")
        
        # Validate configuration
        if not search_config.validate():
            raise ValueError("Invalid search configuration")
        
        # Calculate budgets
        time_budgets = self._calculate_phase_budgets(total_time_budget)
        eval_budgets = self._calculate_eval_budgets(total_eval_budget)

        # Create experiment row (summary)
        if self.exp_store and self.experiment_id is None:
            cfg_payload = {
                "overlap_min": getattr(search_config, "overlap_min", None),
                "overlap_max": getattr(search_config, "overlap_max", None),
                "orders_min": getattr(search_config, "min_order", None),
                "orders_max": getattr(search_config, "max_order", None),
                "alpha": getattr(search_config, "alpha", None),
                "beta": getattr(search_config, "beta", None),
                "gamma": getattr(search_config, "gamma", None),
            }
            self.experiment_id = self.exp_store.create_experiment(adapter="adaptive", cfg=cfg_payload)
        
        try:
            # Phase 1: Warmup (Coarse Scan)
            self.logger.info("Phase 1: Warmup (Coarse Scan)")
            warmup_candidates, warmup_traces = self._run_warmup_phase(
                search_config, time_budgets['warmup'], eval_budgets['warmup']
            )
            self.phase_results.append(('warmup', warmup_candidates, warmup_traces))
            # Upsert batch results
            self._upsert_phase_results(warmup_traces)
            
            # Phase 2: Focus (Narrowed Ranges)
            self.logger.info("Phase 2: Focus (Narrowed Ranges)")
            focus_candidates, focus_traces = self._run_focus_phase(
                search_config, time_budgets['focus'], eval_budgets['focus']
            )
            self.phase_results.append(('focus', focus_candidates, focus_traces))
            self._upsert_phase_results(focus_traces)
            
            # Phase 3: Fine-tuning (Deep Exploration)
            self.logger.info("Phase 3: Fine-tuning (Deep Exploration)")
            fine_tune_candidates, fine_tune_traces = self._run_fine_tune_phase(
                search_config, time_budgets['fine_tune'], eval_budgets['fine_tune']
            )
            self.phase_results.append(('fine_tune', fine_tune_candidates, fine_tune_traces))
            self._upsert_phase_results(fine_tune_traces)
            
            # Phase 4: Output (Results and Storage)
            self.logger.info("Phase 4: Output (Results and Storage)")
            final_candidates, final_traces, stats = self._run_output_phase()
            
            return final_candidates, final_traces, stats
            
        except Exception as e:
            self.logger.error(f"Orchestration failed: {e}")
            raise
    
    def _calculate_phase_budgets(self, total_time: Optional[float]) -> Dict[str, float]:
        """Calculate time budgets for each phase."""
        if total_time is None:
            return {
                'warmup': None,
                'focus': None,
                'fine_tune': None
            }
        
        return {
            'warmup': total_time * self.config.warmup_budget,
            'focus': total_time * self.config.focus_budget,
            'fine_tune': total_time * self.config.fine_tune_budget
        }
    
    def _calculate_eval_budgets(self, total_eval: Optional[int]) -> Dict[str, int]:
        """Calculate evaluation budgets for each phase."""
        if total_eval is None:
            return {
                'warmup': None,
                'focus': None,
                'fine_tune': None
            }
        
        return {
            'warmup': int(total_eval * self.config.warmup_budget),
            'focus': int(total_eval * self.config.focus_budget),
            'fine_tune': int(total_eval * self.config.fine_tune_budget)
        }
    
    def _run_warmup_phase(self, search_config: SearchConfig, 
                         time_budget: Optional[float], 
                         eval_budget: Optional[int]) -> Tuple[SearchResult, TraceResult]:
        """Run warmup phase with coarse scan."""
        # Use numba adapter for fast JIT evaluation
        candidates = self.numba_adapter.search(search_config, time_budget)
        traces = self.numba_adapter.get_trace()
        
        # Update top-k heap
        for candidate in candidates:
            # Find corresponding trace
            trace = next((t for t in traces if t.stable_id == candidate.stable_id), None)
            if trace:
                self.top_k_heap.push(trace.J, candidate, trace)
                self.convergence_checker.add_score(trace.J)
        
        self.logger.info(f"Warmup completed: {len(candidates)} candidates, {len(traces)} traces")
        return candidates, traces
    
    def _run_focus_phase(self, search_config: SearchConfig,
                        time_budget: Optional[float],
                        eval_budget: Optional[int]) -> Tuple[SearchResult, TraceResult]:
        """Run focus phase with narrowed ranges."""
        # Get best candidates from warmup
        best_results = self.top_k_heap.get_top_k()
        if not best_results:
            return [], []
        
        # Calculate focused search ranges
        focused_config = self._create_focused_config(search_config, best_results)
        
        # Use numba adapter for focused search
        candidates = self.numba_adapter.search(focused_config, time_budget)
        traces = self.numba_adapter.get_trace()
        
        # Update top-k heap
        for candidate in candidates:
            trace = next((t for t in traces if t.stable_id == candidate.stable_id), None)
            if trace:
                self.top_k_heap.push(trace.J, candidate, trace)
                self.convergence_checker.add_score(trace.J)
        
        self.logger.info(f"Focus completed: {len(candidates)} candidates, {len(traces)} traces")
        return candidates, traces
    
    def _run_fine_tune_phase(self, search_config: SearchConfig,
                           time_budget: Optional[float],
                           eval_budget: Optional[int]) -> Tuple[SearchResult, TraceResult]:
        """Run fine-tuning phase with deep exploration."""
        # Get current best results
        best_results = self.top_k_heap.get_top_k()
        if not best_results:
            return [], []
        
        # Check for plateau and expand if necessary
        if self.convergence_checker.is_converged():
            self.logger.info("Plateau detected, expanding search range")
            search_config = self._expand_search_range(search_config, best_results)
        
        # Use auto-batch adapter for fine-tuning
        candidates = self.auto_batch_adapter.search(search_config, time_budget)
        traces = self.auto_batch_adapter.get_trace()
        
        # Update top-k heap
        for candidate in candidates:
            trace = next((t for t in traces if t.stable_id == candidate.stable_id), None)
            if trace:
                self.top_k_heap.push(trace.J, candidate, trace)
                self.convergence_checker.add_score(trace.J)
        
        self.logger.info(f"Fine-tuning completed: {len(candidates)} candidates, {len(traces)} traces")
        return candidates, traces
    
    def _run_output_phase(self) -> Tuple[SearchResult, TraceResult, OptimizationStats]:
        """Run output phase with results and storage."""
        # Get final top-N results
        final_results = self.top_k_heap.get_top_k()
        candidates = [result[1] for result in final_results]
        traces = [result[2] for result in final_results]
        
        # Calculate statistics
        stats = self._calculate_final_statistics()
        
        # Save to database if enabled
        if self.storage and self.config.save_to_db:
            self._save_results_to_db(candidates, traces, stats)
        # Update experiment summary
        if self.exp_store and self.experiment_id is not None:
            best = self.top_k_heap.get_best()[0] if self.top_k_heap.heap else float("inf")
            total_evals = sum(t.eval_count for t in traces) if traces else 0
            elapsed_s = stats.get('total_time', 0.0)
            self.exp_store.update_experiment_summary(self.experiment_id, best_score=best, total_evals=total_evals, elapsed_s=elapsed_s)
        
        self.logger.info(f"Output completed: {len(candidates)} final candidates")
        return candidates, traces, stats
    
    def _create_focused_config(self, base_config: SearchConfig, 
                             best_results: List[Tuple[float, Candidate, ResultTrace]]) -> SearchConfig:
        """Create focused search configuration based on best results."""
        # Extract parameter ranges from best results
        min_overlaps = [r[1].min_overlap for r in best_results]
        max_overlaps = [r[1].max_overlap for r in best_results]
        min_orders = [r[1].min_order for r in best_results]
        max_orders = [r[1].max_order for r in best_results]
        
        # Calculate focused ranges with margin
        margin = 0.1  # 10% margin
        
        focused_config = SearchConfig(
            min_overlap=max(0, min(min_overlaps) * (1 - margin)),
            max_overlap=min(100, max(max_overlaps) * (1 + margin)),
            min_order=max(1, min(min_orders) - 1),
            max_order=min(50, max(max_orders) + 1),
            seed=base_config.seed,
            time_budget=base_config.time_budget,
            eval_budget=base_config.eval_budget,
            iteration_budget=base_config.iteration_budget,
            risk_factor=base_config.risk_factor,
            smoothing_factor=base_config.smoothing_factor,
            tail_weight=base_config.tail_weight,
            alpha=base_config.alpha,
            beta=base_config.beta,
            gamma=base_config.gamma
        )
        
        return focused_config
    
    def _expand_search_range(self, base_config: SearchConfig,
                           best_results: List[Tuple[float, Candidate, ResultTrace]]) -> SearchConfig:
        """Expand search range when plateau is detected."""
        expansion_factor = self.config.plateau_expansion
        
        # Expand parameter ranges
        expanded_config = SearchConfig(
            min_overlap=max(0, base_config.min_overlap * (1 - expansion_factor)),
            max_overlap=min(100, base_config.max_overlap * (1 + expansion_factor)),
            min_order=max(1, base_config.min_order - 2),
            max_order=min(50, base_config.max_order + 2),
            seed=base_config.seed,
            time_budget=base_config.time_budget,
            eval_budget=base_config.eval_budget,
            iteration_budget=base_config.iteration_budget,
            risk_factor=base_config.risk_factor,
            smoothing_factor=base_config.smoothing_factor,
            tail_weight=base_config.tail_weight,
            alpha=base_config.alpha,
            beta=base_config.beta,
            gamma=base_config.gamma
        )
        
        return expanded_config
    
    def _calculate_final_statistics(self) -> OptimizationStats:
        """Calculate final optimization statistics."""
        total_time = time.time() - self.start_time if self.start_time else 0
        
        # Get all traces from all phases
        all_traces = []
        for phase_name, candidates, traces in self.phase_results:
            all_traces.extend(traces)
        
        # Calculate statistics
        stats = {
            'session_id': self.session_id,
            'total_time': total_time,
            'total_evaluations': sum(t.eval_count for t in all_traces),
            'total_candidates': sum(len(candidates) for _, candidates, _ in self.phase_results),
            'phases_completed': len(self.phase_results),
            'best_score': self.top_k_heap.get_best()[0] if self.top_k_heap.heap else None,
            'convergence_metrics': self.convergence_checker.get_convergence_metrics(),
            'phase_results': [
                {
                    'phase': phase_name,
                    'candidates': len(candidates),
                    'traces': len(traces),
                    'best_score': max((t.J for t in traces), default=None)
                }
                for phase_name, candidates, traces in self.phase_results
            ]
        }
        
        return stats
    
    def _save_results_to_db(self, candidates: SearchResult, 
                          traces: TraceResult, 
                          stats: OptimizationStats):
        """Save results to database."""
        try:
            # Save candidates and traces
            for candidate, trace in zip(candidates, traces):
                self.storage.save_result(
                    params=candidate,
                    score=trace.J,
                    breakdown=trace,
                    metadata={'session_id': self.session_id}
                )
            
            # Save top results
            top_results = [(trace.J, candidate, trace) for candidate, trace in zip(candidates, traces)]
            self.storage.save_top_results(top_results)
            
            self.logger.info(f"Results saved to database: {len(candidates)} candidates")
            
        except Exception as e:
            self.logger.error(f"Failed to save results to database: {e}")
    
    def _upsert_phase_results(self, traces: TraceResult) -> None:
        if not self.exp_store or self.experiment_id is None:
            return
        items: List[Dict[str, Any]] = []
        for t in traces:
            m = t.to_dict() if hasattr(t, 'to_dict') else {}
            if not m:
                # Build from metrics fallback
                m = {
                    "stable_id": getattr(t, 'stable_id', None),
                    "score": getattr(t, 'J', None),
                    "params": {},
                    "schedule": {},
                    "risk": {},
                    "penalties": {},
                }
            else:
                items.append({
                    "stable_id": m.get("stable_id"),
                    "score": m.get("J"),
                    "params": m.get("candidate", {}) or m.get("params", {}),
                    "schedule": getattr(getattr(t, 'candidate', None), 'schedule', None) or m.get("schedule", {}),
                    "risk": {
                        "max_need": m.get("max_score"),
                        "var_need": m.get("variance_score"),
                        "tail": m.get("tail_score"),
                    },
                    "penalties": {
                        "gini": m.get("gini_penalty"),
                        "entropy": m.get("entropy_penalty"),
                        "monotone_viol": m.get("monotone_penalty"),
                        "smooth_viol": m.get("smoothness_penalty"),
                    }
                })
        if items:
            self.exp_store.upsert_results(self.experiment_id, items)
    
    def get_best_candidate(self) -> Optional[Candidate]:
        """Get the best candidate found."""
        if not self.top_k_heap.heap:
            return None
        return self.top_k_heap.get_best()[1]
    
    def get_top_candidates(self, n: int = None) -> List[Candidate]:
        """Get top-N candidates."""
        if n is None:
            n = self.config.top_n_results
        
        results = self.top_k_heap.get_top_k()[:n]
        return [result[1] for result in results]
    
    def get_optimization_trace(self) -> TraceResult:
        """Get complete optimization trace."""
        all_traces = []
        for _, _, traces in self.phase_results:
            all_traces.extend(traces)
        return all_traces