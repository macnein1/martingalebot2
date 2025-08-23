"""
Reduction and Analysis for DCA/Martingale Optimization
Implements batch summaries, Pareto-meta analysis, and trace systems.
"""
from __future__ import annotations

import numpy as np
from numba import njit
from typing import Dict, List, Any, Tuple
import math


@njit(cache=True, fastmath=True)
def pareto_dominance_check(scores_a: np.ndarray, scores_b: np.ndarray) -> int:
    """
    Check Pareto dominance between two multi-objective score vectors.
    
    Args:
        scores_a: Score vector for candidate A [max_need, var_need, tail, ...]
        scores_b: Score vector for candidate B [max_need, var_need, tail, ...]
        
    Returns:
        1 if A dominates B, -1 if B dominates A, 0 if non-dominated
    """
    if scores_a.size != scores_b.size:
        return 0
    
    a_better_count = 0
    b_better_count = 0
    
    for i in range(scores_a.size):
        if scores_a[i] < scores_b[i]:  # Lower is better for all objectives
            a_better_count += 1
        elif scores_b[i] < scores_a[i]:
            b_better_count += 1
    
    if a_better_count > 0 and b_better_count == 0:
        return 1  # A dominates B
    elif b_better_count > 0 and a_better_count == 0:
        return -1  # B dominates A
    else:
        return 0  # Non-dominated


@njit(cache=True, fastmath=True)
def calculate_crowding_distance(objective_matrix: np.ndarray) -> np.ndarray:
    """
    Calculate crowding distance for diversity preservation in Pareto front.
    
    Args:
        objective_matrix: Matrix of objectives (n_candidates, n_objectives)
        
    Returns:
        Crowding distances for each candidate
    """
    n_candidates, n_objectives = objective_matrix.shape
    distances = np.zeros(n_candidates)
    
    for obj_idx in range(n_objectives):
        # Sort by this objective
        obj_values = objective_matrix[:, obj_idx]
        sorted_indices = np.argsort(obj_values)
        
        # Set boundary points to infinity
        distances[sorted_indices[0]] = float('inf')
        distances[sorted_indices[-1]] = float('inf')
        
        # Calculate range for normalization
        obj_range = obj_values[sorted_indices[-1]] - obj_values[sorted_indices[0]]
        if obj_range <= 1e-12:
            continue
        
        # Calculate crowding distance for intermediate points
        for i in range(1, n_candidates - 1):
            current_idx = sorted_indices[i]
            prev_idx = sorted_indices[i - 1]
            next_idx = sorted_indices[i + 1]
            
            distance_contribution = (obj_values[next_idx] - obj_values[prev_idx]) / obj_range
            distances[current_idx] += distance_contribution
    
    return distances


@njit(cache=True, fastmath=True)
def fast_non_dominated_sort(objective_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fast non-dominated sorting algorithm for multi-objective optimization.
    
    Args:
        objective_matrix: Matrix of objectives (n_candidates, n_objectives)
        
    Returns:
        Tuple of (fronts_assignment, dominance_counts)
    """
    n_candidates = objective_matrix.shape[0]
    
    # Initialize arrays
    dominance_counts = np.zeros(n_candidates, dtype=np.int32)  # Number of solutions that dominate this one
    dominated_solutions = [[] for _ in range(n_candidates)]    # Solutions dominated by this one
    fronts = [[] for _ in range(n_candidates)]                 # Fronts
    front_assignment = np.zeros(n_candidates, dtype=np.int32)
    
    # Find domination relationships
    for i in range(n_candidates):
        for j in range(n_candidates):
            if i != j:
                dominance = pareto_dominance_check(objective_matrix[i], objective_matrix[j])
                if dominance == 1:  # i dominates j
                    dominated_solutions[i].append(j)
                elif dominance == -1:  # j dominates i
                    dominance_counts[i] += 1
        
        # If no one dominates this solution, it's in the first front
        if dominance_counts[i] == 0:
            fronts[0].append(i)
            front_assignment[i] = 0
    
    # Build subsequent fronts
    current_front = 0
    while len(fronts[current_front]) > 0:
        next_front = []
        
        for solution_idx in fronts[current_front]:
            for dominated_idx in dominated_solutions[solution_idx]:
                dominance_counts[dominated_idx] -= 1
                if dominance_counts[dominated_idx] == 0:
                    next_front.append(dominated_idx)
                    front_assignment[dominated_idx] = current_front + 1
        
        current_front += 1
        if current_front < n_candidates:
            fronts[current_front] = next_front
        else:
            break
    
    return front_assignment, dominance_counts


def extract_pareto_front(candidates: List[Dict[str, Any]], 
                        objectives: List[str] = None) -> List[Dict[str, Any]]:
    """
    Extract Pareto-optimal candidates from a list of candidates.
    
    Args:
        candidates: List of candidate dictionaries
        objectives: List of objective names to consider (default: ['max_need', 'var_need', 'tail'])
        
    Returns:
        List of Pareto-optimal candidates
    """
    if not candidates:
        return []
    
    if objectives is None:
        objectives = ['max_need', 'var_need', 'tail']
    
    # Extract objective matrix
    n_candidates = len(candidates)
    n_objectives = len(objectives)
    objective_matrix = np.zeros((n_candidates, n_objectives))
    
    for i, candidate in enumerate(candidates):
        for j, obj_name in enumerate(objectives):
            objective_matrix[i, j] = candidate.get(obj_name, float('inf'))
    
    # Perform non-dominated sorting
    front_assignment, _ = fast_non_dominated_sort(objective_matrix)
    
    # Extract first front (Pareto-optimal solutions)
    pareto_candidates = []
    for i, candidate in enumerate(candidates):
        if front_assignment[i] == 0:
            pareto_candidates.append(candidate)
    
    return pareto_candidates


@njit(cache=True, fastmath=True)
def batch_statistics(scores: np.ndarray) -> Tuple[float, float, float, float, float]:
    """
    Calculate batch statistics for scores.
    
    Args:
        scores: Array of scores
        
    Returns:
        Tuple of (min, max, mean, std, median)
    """
    if scores.size == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    
    min_score = np.min(scores)
    max_score = np.max(scores)
    mean_score = np.mean(scores)
    std_score = np.std(scores)
    
    # Calculate median
    sorted_scores = np.sort(scores)
    n = len(sorted_scores)
    if n % 2 == 0:
        median_score = (sorted_scores[n//2 - 1] + sorted_scores[n//2]) / 2.0
    else:
        median_score = sorted_scores[n//2]
    
    return min_score, max_score, mean_score, std_score, median_score


@njit(cache=True, fastmath=True)
def top_k_indices(scores: np.ndarray, k: int) -> np.ndarray:
    """
    Get indices of top-k best (lowest) scores.
    
    Args:
        scores: Array of scores (lower is better)
        k: Number of top candidates to return
        
    Returns:
        Indices of top-k candidates
    """
    if scores.size == 0:
        return np.array([], dtype=np.int32)
    
    k = min(k, scores.size)
    
    # Use partial sort for efficiency
    indices = np.argsort(scores)[:k]
    return indices


@njit(cache=True, fastmath=True)
def diversity_filter(objective_matrix: np.ndarray, diversity_threshold: float = 0.01) -> np.ndarray:
    """
    Filter candidates based on diversity threshold.
    
    Args:
        objective_matrix: Matrix of objectives (n_candidates, n_objectives)
        diversity_threshold: Minimum distance threshold
        
    Returns:
        Indices of diverse candidates
    """
    n_candidates = objective_matrix.shape[0]
    if n_candidates <= 1:
        return np.arange(n_candidates, dtype=np.int32)
    
    # Normalize objectives to [0, 1] for distance calculation
    normalized_matrix = np.empty_like(objective_matrix)
    for j in range(objective_matrix.shape[1]):
        col = objective_matrix[:, j]
        min_val = np.min(col)
        max_val = np.max(col)
        if max_val - min_val > 1e-12:
            normalized_matrix[:, j] = (col - min_val) / (max_val - min_val)
        else:
            normalized_matrix[:, j] = 0.5  # All values are the same
    
    # Select diverse candidates
    selected = [0]  # Always include the first candidate
    
    for i in range(1, n_candidates):
        min_distance = float('inf')
        
        # Calculate minimum distance to already selected candidates
        for j in selected:
            distance = 0.0
            for k in range(objective_matrix.shape[1]):
                diff = normalized_matrix[i, k] - normalized_matrix[j, k]
                distance += diff * diff
            distance = math.sqrt(distance)
            
            if distance < min_distance:
                min_distance = distance
        
        # Add candidate if it's diverse enough
        if min_distance >= diversity_threshold:
            selected.append(i)
    
    return np.array(selected, dtype=np.int32)


class BatchSummary:
    """Summary statistics for a batch of optimization results."""
    
    def __init__(self, candidates: List[Dict[str, Any]]):
        """Initialize batch summary from candidates."""
        self.n_candidates = len(candidates)
        self.candidates = candidates
        
        if not candidates:
            self.stats = {}
            return
        
        # Extract key metrics
        scores = np.array([c.get('score', float('inf')) for c in candidates])
        max_needs = np.array([c.get('max_need', 0.0) for c in candidates])
        var_needs = np.array([c.get('var_need', 0.0) for c in candidates])
        tails = np.array([c.get('tail', 0.0) for c in candidates])
        
        # Calculate statistics
        self.stats = {
            'score': self._calc_stats(scores),
            'max_need': self._calc_stats(max_needs),
            'var_need': self._calc_stats(var_needs),
            'tail': self._calc_stats(tails),
        }
        
        # Additional metrics if available
        if 'cvar_need' in candidates[0]:
            cvar_needs = np.array([c.get('cvar_need', 0.0) for c in candidates])
            self.stats['cvar_need'] = self._calc_stats(cvar_needs)
        
        if 'shape_reward' in candidates[0]:
            shape_rewards = np.array([c.get('shape_reward', 0.0) for c in candidates])
            self.stats['shape_reward'] = self._calc_stats(shape_rewards)
    
    def _calc_stats(self, values: np.ndarray) -> Dict[str, float]:
        """Calculate statistics for a value array."""
        if values.size == 0:
            return {'min': 0.0, 'max': 0.0, 'mean': 0.0, 'std': 0.0, 'median': 0.0}
        
        min_val, max_val, mean_val, std_val, median_val = batch_statistics(values)
        return {
            'min': float(min_val),
            'max': float(max_val),
            'mean': float(mean_val),
            'std': float(std_val),
            'median': float(median_val)
        }
    
    def get_top_k(self, k: int = 10) -> List[Dict[str, Any]]:
        """Get top-k candidates by score."""
        if not self.candidates:
            return []
        
        scores = np.array([c.get('score', float('inf')) for c in self.candidates])
        indices = top_k_indices(scores, k)
        
        return [self.candidates[i] for i in indices]
    
    def get_pareto_front(self, objectives: List[str] = None) -> List[Dict[str, Any]]:
        """Get Pareto-optimal candidates."""
        return extract_pareto_front(self.candidates, objectives)
    
    def get_diverse_subset(self, max_candidates: int = 100, 
                          diversity_threshold: float = 0.01) -> List[Dict[str, Any]]:
        """Get diverse subset of candidates."""
        if len(self.candidates) <= max_candidates:
            return self.candidates
        
        # First, get top candidates by score
        top_candidates = self.get_top_k(max_candidates * 2)  # Get more for diversity filtering
        
        if not top_candidates:
            return []
        
        # Extract objective matrix
        objectives = ['max_need', 'var_need', 'tail']
        objective_matrix = np.zeros((len(top_candidates), len(objectives)))
        
        for i, candidate in enumerate(top_candidates):
            for j, obj_name in enumerate(objectives):
                objective_matrix[i, j] = candidate.get(obj_name, 0.0)
        
        # Apply diversity filter
        diverse_indices = diversity_filter(objective_matrix, diversity_threshold)
        
        # Return diverse subset, limited to max_candidates
        diverse_candidates = [top_candidates[i] for i in diverse_indices[:max_candidates]]
        
        return diverse_candidates


class TraceRecord:
    """Individual trace record for a candidate."""
    
    def __init__(self, candidate: Dict[str, Any]):
        """Initialize trace record from candidate."""
        self.score = candidate.get('score', float('inf'))
        self.max_need = candidate.get('max_need', 0.0)
        self.var_need = candidate.get('var_need', 0.0)
        self.tail = candidate.get('tail', 0.0)
        self.cvar_need = candidate.get('cvar_need', 0.0)
        self.shape_reward = candidate.get('shape_reward', 0.0)
        
        # Sanity flags
        sanity = candidate.get('sanity', {})
        self.max_need_mismatch = sanity.get('max_need_mismatch', False)
        self.collapse_indents = sanity.get('collapse_indents', False)
        self.tail_overflow = sanity.get('tail_overflow', False)
        
        # Diagnostics
        diagnostics = candidate.get('diagnostics', {})
        self.wci = diagnostics.get('wci', 0.0)
        self.sign_flips = diagnostics.get('sign_flips', 0)
        self.gini = diagnostics.get('gini', 0.0)
        self.entropy = diagnostics.get('entropy', 0.0)
        
        # Parameters
        params = candidate.get('params', {})
        self.overlap_pct = params.get('overlap_pct', 0.0)
        self.num_orders = params.get('num_orders', 0)
    
    def has_violations(self) -> bool:
        """Check if candidate has any sanity violations."""
        return (self.max_need_mismatch or 
                self.collapse_indents or 
                self.tail_overflow)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert trace record to dictionary."""
        return {
            'score': self.score,
            'max_need': self.max_need,
            'var_need': self.var_need,
            'tail': self.tail,
            'cvar_need': self.cvar_need,
            'shape_reward': self.shape_reward,
            'wci': self.wci,
            'sign_flips': self.sign_flips,
            'gini': self.gini,
            'entropy': self.entropy,
            'overlap_pct': self.overlap_pct,
            'num_orders': self.num_orders,
            'has_violations': self.has_violations()
        }


class OptimizationTrace:
    """Trace system for optimization progress."""
    
    def __init__(self):
        """Initialize empty trace."""
        self.records: List[TraceRecord] = []
        self.batch_summaries: List[BatchSummary] = []
        self.best_score_history: List[float] = []
        self.improvement_history: List[float] = []
    
    def add_batch(self, candidates: List[Dict[str, Any]]):
        """Add a batch of candidates to the trace."""
        # Create trace records
        batch_records = [TraceRecord(c) for c in candidates]
        self.records.extend(batch_records)
        
        # Create batch summary
        summary = BatchSummary(candidates)
        self.batch_summaries.append(summary)
        
        # Update best score history
        if candidates:
            batch_best = min(c.get('score', float('inf')) for c in candidates)
            
            if not self.best_score_history:
                self.best_score_history.append(batch_best)
                self.improvement_history.append(0.0)
            else:
                prev_best = self.best_score_history[-1]
                new_best = min(prev_best, batch_best)
                self.best_score_history.append(new_best)
                
                # Calculate improvement
                if prev_best > 0:
                    improvement = (prev_best - new_best) / prev_best
                else:
                    improvement = 0.0
                self.improvement_history.append(improvement)
    
    def get_convergence_info(self) -> Dict[str, Any]:
        """Get convergence information."""
        if not self.best_score_history:
            return {}
        
        return {
            'total_batches': len(self.batch_summaries),
            'total_candidates': len(self.records),
            'best_score': self.best_score_history[-1],
            'initial_score': self.best_score_history[0],
            'total_improvement': self.best_score_history[0] - self.best_score_history[-1],
            'recent_improvement': np.mean(self.improvement_history[-5:]) if len(self.improvement_history) >= 5 else 0.0,
            'stagnation_count': self._count_stagnation()
        }
    
    def _count_stagnation(self) -> int:
        """Count consecutive batches without improvement."""
        if len(self.improvement_history) < 2:
            return 0
        
        stagnation = 0
        threshold = 1e-6
        
        for i in range(len(self.improvement_history) - 1, -1, -1):
            if self.improvement_history[i] < threshold:
                stagnation += 1
            else:
                break
        
        return stagnation
    
    def should_early_stop(self, patience: int = 10, 
                         min_improvement: float = 1e-6) -> bool:
        """Check if optimization should stop early."""
        if len(self.improvement_history) < patience:
            return False
        
        recent_improvements = self.improvement_history[-patience:]
        return all(imp < min_improvement for imp in recent_improvements)
    
    def get_summary_report(self) -> str:
        """Generate summary report."""
        if not self.batch_summaries:
            return "No optimization data available."
        
        convergence = self.get_convergence_info()
        latest_summary = self.batch_summaries[-1]
        
        lines = [
            "=== DCA Optimization Summary ===",
            f"Total Batches: {convergence['total_batches']}",
            f"Total Candidates: {convergence['total_candidates']}",
            f"Best Score: {convergence['best_score']:.6f}",
            f"Total Improvement: {convergence.get('total_improvement', 0):.6f}",
            "",
            "=== Latest Batch Statistics ===",
            f"Candidates: {latest_summary.n_candidates}",
        ]
        
        if 'score' in latest_summary.stats:
            score_stats = latest_summary.stats['score']
            lines.extend([
                f"Score - Min: {score_stats['min']:.6f}, Mean: {score_stats['mean']:.6f}, Max: {score_stats['max']:.6f}",
                f"Max Need - Min: {latest_summary.stats['max_need']['min']:.3f}%, Mean: {latest_summary.stats['max_need']['mean']:.3f}%",
                f"Var Need - Mean: {latest_summary.stats['var_need']['mean']:.6f}",
                f"Tail - Mean: {latest_summary.stats['tail']['mean']:.3f}"
            ])
        
        # Violation statistics
        violation_count = sum(1 for r in self.records[-latest_summary.n_candidates:] if r.has_violations())
        lines.append(f"Sanity Violations: {violation_count}/{latest_summary.n_candidates} ({100*violation_count/latest_summary.n_candidates:.1f}%)")
        
        return "\n".join(lines)
