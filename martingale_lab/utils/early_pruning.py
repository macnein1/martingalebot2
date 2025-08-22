"""
Early pruning and elimination strategies for efficient optimization.
Implements ASHA-style successive halving and budget control mechanisms.
"""
import numpy as np
import time
from typing import List, Dict, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict

from .logging import LogContext


class PruningStrategy(Enum):
    """Available pruning strategies."""
    ASHA = "asha"                    # Asynchronous Successive Halving
    MEDIAN_STOPPING = "median"       # Stop if below median performance
    PERCENTILE_STOPPING = "percentile"  # Stop if below percentile
    BUDGET_CONTROL = "budget"        # Time/evaluation budget control
    EARLY_CONVERGENCE = "convergence"  # Stop if no improvement


@dataclass
class CandidateState:
    """State tracking for a candidate during evaluation."""
    candidate_id: str
    start_time: float
    evaluations_completed: int = 0
    partial_scores: List[float] = field(default_factory=list)
    current_score: Optional[float] = None
    is_eliminated: bool = False
    elimination_reason: Optional[str] = None
    budget_used: float = 0.0  # Time or evaluation budget used
    
    def add_partial_score(self, score: float):
        """Add a partial evaluation score."""
        self.partial_scores.append(score)
        self.current_score = score  # Most recent score
        self.evaluations_completed += 1
    
    def get_average_score(self) -> Optional[float]:
        """Get average score across partial evaluations."""
        return np.mean(self.partial_scores) if self.partial_scores else None


@dataclass
class PruningConfig:
    """Configuration for pruning strategies."""
    strategy: PruningStrategy = PruningStrategy.ASHA
    
    # ASHA parameters
    reduction_factor: int = 3          # Eliminate 2/3 of candidates each round
    min_budget: int = 3               # Minimum evaluations before pruning
    max_budget: int = 27              # Maximum evaluations per candidate
    
    # Percentile stopping parameters
    percentile_threshold: float = 25.0  # Stop if below 25th percentile
    min_candidates_for_percentile: int = 10
    
    # Budget control parameters
    max_time_per_candidate: float = 30.0  # Max seconds per candidate
    max_batch_time: float = 300.0         # Max seconds per batch
    target_evaluations: int = 1000        # Target total evaluations
    
    # Early convergence parameters
    convergence_patience: int = 5         # Rounds without improvement
    convergence_threshold: float = 0.01   # Minimum improvement threshold
    
    # Grace period (don't eliminate too early)
    grace_period_evaluations: int = 3
    grace_period_seconds: float = 5.0


class ASHAPruner:
    """Asynchronous Successive Halving Algorithm (ASHA) implementation."""
    
    def __init__(self, config: PruningConfig, log_ctx: Optional[LogContext] = None):
        self.config = config
        self.log_ctx = log_ctx
        
        # Track candidates by rung (evaluation level)
        self.rungs: Dict[int, List[CandidateState]] = defaultdict(list)
        self.active_candidates: Dict[str, CandidateState] = {}
        self.eliminated_candidates: Dict[str, CandidateState] = {}
        
        # Statistics
        self.total_candidates_seen = 0
        self.total_eliminations = 0
        self.elimination_reasons = defaultdict(int)
    
    def register_candidate(self, candidate_id: str) -> CandidateState:
        """Register a new candidate for evaluation."""
        state = CandidateState(
            candidate_id=candidate_id,
            start_time=time.perf_counter()
        )
        
        self.active_candidates[candidate_id] = state
        self.total_candidates_seen += 1
        
        if self.log_ctx:
            self.log_ctx.log('candidate_registered', candidate_id=candidate_id)
        
        return state
    
    def should_continue_evaluation(self, candidate_id: str, 
                                  partial_score: Optional[float] = None) -> bool:
        """
        Determine if candidate evaluation should continue.
        
        Args:
            candidate_id: ID of candidate being evaluated
            partial_score: Latest partial score (optional)
            
        Returns:
            True if evaluation should continue, False if should be eliminated
        """
        if candidate_id not in self.active_candidates:
            return False
        
        state = self.active_candidates[candidate_id]
        
        # Add partial score if provided
        if partial_score is not None:
            state.add_partial_score(partial_score)
        
        # Check if already eliminated
        if state.is_eliminated:
            return False
        
        # Grace period - don't eliminate too early
        if self._in_grace_period(state):
            return True
        
        # Apply pruning strategy
        if self.config.strategy == PruningStrategy.ASHA:
            return self._asha_should_continue(state)
        elif self.config.strategy == PruningStrategy.MEDIAN_STOPPING:
            return self._median_stopping_should_continue(state)
        elif self.config.strategy == PruningStrategy.PERCENTILE_STOPPING:
            return self._percentile_stopping_should_continue(state)
        elif self.config.strategy == PruningStrategy.BUDGET_CONTROL:
            return self._budget_control_should_continue(state)
        else:
            return True  # No pruning
    
    def _in_grace_period(self, state: CandidateState) -> bool:
        """Check if candidate is in grace period (shouldn't be eliminated yet)."""
        time_elapsed = time.perf_counter() - state.start_time
        
        return (state.evaluations_completed < self.config.grace_period_evaluations or
                time_elapsed < self.config.grace_period_seconds)
    
    def _asha_should_continue(self, state: CandidateState) -> bool:
        """ASHA elimination logic."""
        evals = state.evaluations_completed
        
        # Don't eliminate before minimum budget
        if evals < self.config.min_budget:
            return True
        
        # Stop at maximum budget
        if evals >= self.config.max_budget:
            self._eliminate_candidate(state, "max_budget_reached")
            return False
        
        # Check if we should eliminate based on current rung
        current_rung = self._get_rung_for_evaluations(evals)
        
        if current_rung > 0:  # Only prune after first rung
            rung_candidates = self.rungs[current_rung]
            
            if len(rung_candidates) >= self.config.reduction_factor:
                # Sort by score (lower is better)
                sorted_candidates = sorted(rung_candidates, 
                                         key=lambda c: c.get_average_score() or float('inf'))
                
                # Eliminate bottom candidates
                keep_count = len(sorted_candidates) // self.config.reduction_factor
                if keep_count == 0:
                    keep_count = 1  # Always keep at least one
                
                if state in sorted_candidates[keep_count:]:
                    self._eliminate_candidate(state, f"asha_rung_{current_rung}")
                    return False
        
        # Move to appropriate rung
        self.rungs[current_rung].append(state)
        return True
    
    def _median_stopping_should_continue(self, state: CandidateState) -> bool:
        """Median stopping rule."""
        if state.evaluations_completed < self.config.min_budget:
            return True
        
        current_score = state.get_average_score()
        if current_score is None:
            return True
        
        # Get scores of all active candidates with similar evaluation counts
        similar_scores = []
        for other_state in self.active_candidates.values():
            if (not other_state.is_eliminated and 
                abs(other_state.evaluations_completed - state.evaluations_completed) <= 2):
                other_score = other_state.get_average_score()
                if other_score is not None:
                    similar_scores.append(other_score)
        
        if len(similar_scores) >= 5:  # Need enough samples
            median_score = np.median(similar_scores)
            if current_score > median_score * 1.2:  # 20% worse than median
                self._eliminate_candidate(state, "below_median")
                return False
        
        return True
    
    def _percentile_stopping_should_continue(self, state: CandidateState) -> bool:
        """Percentile-based stopping rule."""
        if state.evaluations_completed < self.config.min_budget:
            return True
        
        current_score = state.get_average_score()
        if current_score is None:
            return True
        
        # Get all current scores
        all_scores = []
        for other_state in self.active_candidates.values():
            if not other_state.is_eliminated:
                other_score = other_state.get_average_score()
                if other_score is not None:
                    all_scores.append(other_score)
        
        if len(all_scores) >= self.config.min_candidates_for_percentile:
            percentile_score = np.percentile(all_scores, self.config.percentile_threshold)
            if current_score > percentile_score:
                self._eliminate_candidate(state, f"below_{self.config.percentile_threshold}th_percentile")
                return False
        
        return True
    
    def _budget_control_should_continue(self, state: CandidateState) -> bool:
        """Budget-based elimination."""
        time_elapsed = time.perf_counter() - state.start_time
        
        # Individual candidate time limit
        if time_elapsed > self.config.max_time_per_candidate:
            self._eliminate_candidate(state, "time_budget_exceeded")
            return False
        
        # Check evaluation budget
        if state.evaluations_completed >= self.config.max_budget:
            self._eliminate_candidate(state, "evaluation_budget_exceeded")
            return False
        
        return True
    
    def _get_rung_for_evaluations(self, evaluations: int) -> int:
        """Get ASHA rung for given number of evaluations."""
        if evaluations < self.config.min_budget:
            return 0
        
        # Exponential rungs: 3, 9, 27, ...
        rung = 0
        budget = self.config.min_budget
        
        while budget <= evaluations and budget < self.config.max_budget:
            rung += 1
            budget *= self.config.reduction_factor
        
        return rung - 1 if rung > 0 else 0
    
    def _eliminate_candidate(self, state: CandidateState, reason: str):
        """Mark candidate as eliminated."""
        state.is_eliminated = True
        state.elimination_reason = reason
        
        # Move from active to eliminated
        if state.candidate_id in self.active_candidates:
            self.eliminated_candidates[state.candidate_id] = self.active_candidates.pop(state.candidate_id)
        
        self.total_eliminations += 1
        self.elimination_reasons[reason] += 1
        
        if self.log_ctx:
            self.log_ctx.log('candidate_eliminated', 
                           candidate_id=state.candidate_id,
                           reason=reason,
                           evaluations_completed=state.evaluations_completed,
                           final_score=state.get_average_score())
    
    def get_active_candidates(self) -> List[CandidateState]:
        """Get list of currently active (non-eliminated) candidates."""
        return [state for state in self.active_candidates.values() if not state.is_eliminated]
    
    def get_elimination_stats(self) -> Dict[str, Any]:
        """Get statistics about eliminations."""
        return {
            'total_candidates': self.total_candidates_seen,
            'active_candidates': len(self.active_candidates),
            'eliminated_candidates': self.total_eliminations,
            'elimination_rate': self.total_eliminations / self.total_candidates_seen if self.total_candidates_seen > 0 else 0,
            'elimination_reasons': dict(self.elimination_reasons),
            'rungs_populated': {rung: len(candidates) for rung, candidates in self.rungs.items()}
        }


class BudgetController:
    """Controls evaluation budget and graceful stopping."""
    
    def __init__(self, config: PruningConfig, log_ctx: Optional[LogContext] = None):
        self.config = config
        self.log_ctx = log_ctx
        
        self.batch_start_time = time.perf_counter()
        self.total_evaluations = 0
        self.budget_exhausted = False
        self.stop_reason: Optional[str] = None
    
    def start_batch(self):
        """Start a new batch."""
        self.batch_start_time = time.perf_counter()
        self.total_evaluations = 0
        self.budget_exhausted = False
        self.stop_reason = None
        
        if self.log_ctx:
            self.log_ctx.log('budget_batch_start', 
                           target_evaluations=self.config.target_evaluations,
                           max_batch_time=self.config.max_batch_time)
    
    def should_continue_batch(self) -> bool:
        """Check if batch should continue or stop gracefully."""
        if self.budget_exhausted:
            return False
        
        current_time = time.perf_counter()
        batch_duration = current_time - self.batch_start_time
        
        # Check time budget
        if batch_duration >= self.config.max_batch_time:
            self.budget_exhausted = True
            self.stop_reason = "time_budget_exhausted"
            
            if self.log_ctx:
                self.log_ctx.log('batch_stopped_time_budget',
                               duration=batch_duration,
                               evaluations_completed=self.total_evaluations)
            return False
        
        # Check evaluation budget
        if self.total_evaluations >= self.config.target_evaluations:
            self.budget_exhausted = True
            self.stop_reason = "evaluation_budget_reached"
            
            if self.log_ctx:
                self.log_ctx.log('batch_stopped_eval_budget',
                               evaluations_completed=self.total_evaluations,
                               duration=batch_duration)
            return False
        
        return True
    
    def record_evaluation(self):
        """Record completion of an evaluation."""
        self.total_evaluations += 1
    
    def get_progress(self) -> Dict[str, Any]:
        """Get current progress information."""
        current_time = time.perf_counter()
        batch_duration = current_time - self.batch_start_time
        
        time_progress = min(batch_duration / self.config.max_batch_time, 1.0)
        eval_progress = min(self.total_evaluations / self.config.target_evaluations, 1.0)
        
        return {
            'batch_duration': batch_duration,
            'total_evaluations': self.total_evaluations,
            'time_progress_pct': time_progress * 100,
            'eval_progress_pct': eval_progress * 100,
            'estimated_evals_per_second': self.total_evaluations / batch_duration if batch_duration > 0 else 0,
            'budget_exhausted': self.budget_exhausted,
            'stop_reason': self.stop_reason
        }


class EarlyStoppingManager:
    """Manages multiple pruning strategies and budget control."""
    
    def __init__(self, config: PruningConfig, log_ctx: Optional[LogContext] = None):
        self.config = config
        self.log_ctx = log_ctx
        
        self.pruner = ASHAPruner(config, log_ctx)
        self.budget_controller = BudgetController(config, log_ctx)
        
        # Convergence tracking
        self.best_score_history: List[float] = []
        self.rounds_without_improvement = 0
    
    def start_batch(self):
        """Start a new batch with fresh state."""
        self.budget_controller.start_batch()
        self.best_score_history.clear()
        self.rounds_without_improvement = 0
        
        if self.log_ctx:
            self.log_ctx.log('early_stopping_batch_start')
    
    def should_evaluate_candidate(self, candidate_id: str, 
                                partial_score: Optional[float] = None) -> bool:
        """
        Determine if a candidate should continue evaluation.
        
        Args:
            candidate_id: Candidate identifier
            partial_score: Latest partial evaluation score
            
        Returns:
            True if should continue, False if should stop
        """
        # Check budget constraints first
        if not self.budget_controller.should_continue_batch():
            return False
        
        # Check candidate-specific pruning
        return self.pruner.should_continue_evaluation(candidate_id, partial_score)
    
    def register_candidate(self, candidate_id: str) -> CandidateState:
        """Register a new candidate."""
        return self.pruner.register_candidate(candidate_id)
    
    def record_evaluation_complete(self, score: Optional[float] = None):
        """Record completion of an evaluation."""
        self.budget_controller.record_evaluation()
        
        # Track best score for convergence detection
        if score is not None:
            if not self.best_score_history or score < min(self.best_score_history):
                self.rounds_without_improvement = 0
                if self.log_ctx:
                    self.log_ctx.log('new_best_score', score=score)
            else:
                self.rounds_without_improvement += 1
            
            self.best_score_history.append(score)
    
    def check_early_convergence(self) -> bool:
        """Check if optimization has converged early."""
        if len(self.best_score_history) < self.config.convergence_patience:
            return False
        
        if self.rounds_without_improvement >= self.config.convergence_patience:
            if self.log_ctx:
                self.log_ctx.log('early_convergence_detected',
                               rounds_without_improvement=self.rounds_without_improvement,
                               patience=self.config.convergence_patience)
            return True
        
        return False
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics from all components."""
        elimination_stats = self.pruner.get_elimination_stats()
        budget_progress = self.budget_controller.get_progress()
        
        return {
            'elimination_stats': elimination_stats,
            'budget_progress': budget_progress,
            'convergence_info': {
                'best_score_history': self.best_score_history[-10:],  # Last 10 scores
                'rounds_without_improvement': self.rounds_without_improvement,
                'converged': self.check_early_convergence()
            },
            'active_candidates': len(self.pruner.get_active_candidates())
        }
