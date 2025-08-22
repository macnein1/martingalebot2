"""
Reduction utilities for optimization results processing.
"""
import numpy as np
from typing import List, Tuple, Dict, Any
from heapq import heappush, heappop, nlargest
from .types import Params, ScoreBreakdown


class TopKHeap:
    """Maintain top-k results using a min-heap."""
    
    def __init__(self, k: int):
        """Initialize with maximum size k."""
        self.k = k
        self.heap = []
        self.size = 0
    
    def push(self, score: float, params: Params, breakdown: ScoreBreakdown):
        """Add a result to the heap."""
        if self.size < self.k:
            # Heap not full, just add
            heappush(self.heap, (score, params, breakdown))
            self.size += 1
        elif score > self.heap[0][0]:
            # Score is better than worst in heap, replace
            heappop(self.heap)
            heappush(self.heap, (score, params, breakdown))
    
    def get_top_k(self) -> List[Tuple[float, Params, ScoreBreakdown]]:
        """Get top-k results sorted by score (descending)."""
        return sorted(self.heap, key=lambda x: x[0], reverse=True)
    
    def get_best(self) -> Tuple[float, Params, ScoreBreakdown]:
        """Get the best result."""
        if not self.heap:
            raise ValueError("Heap is empty")
        return max(self.heap, key=lambda x: x[0])


class ParetoFilter:
    """Pareto front filtering for multi-objective optimization."""
    
    @staticmethod
    def is_dominated(point1: np.ndarray, point2: np.ndarray) -> bool:
        """Check if point1 is dominated by point2."""
        # point2 dominates point1 if it's better in all objectives
        return np.all(point2 >= point1) and np.any(point2 > point1)
    
    @staticmethod
    def find_pareto_front(points: np.ndarray) -> List[int]:
        """Find Pareto front indices."""
        n_points = len(points)
        pareto_indices = []
        
        for i in range(n_points):
            is_pareto = True
            for j in range(n_points):
                if i != j and ParetoFilter.is_dominated(points[i], points[j]):
                    is_pareto = False
                    break
            if is_pareto:
                pareto_indices.append(i)
        
        return pareto_indices
    
    @staticmethod
    def filter_pareto_results(results: List[Tuple[float, Params, ScoreBreakdown]]) -> List[int]:
        """Filter results to Pareto front based on score components."""
        if not results:
            return []
        
        # Extract score components
        points = []
        for score, params, breakdown in results:
            point = np.array([
                breakdown.max_score,
                -breakdown.variance_score,  # Minimize variance
                breakdown.tail_score,
                -breakdown.penalty_total  # Minimize penalties
            ])
            points.append(point)
        
        points = np.array(points)
        return ParetoFilter.find_pareto_front(points)


class ZoomBoxCalculator:
    """Calculate zoom boxes for adaptive refinement."""
    
    @staticmethod
    def calculate_zoom_box(best_params: Params, radius: float = 0.1) -> Dict[str, Tuple[float, float]]:
        """Calculate zoom box around best parameters."""
        return {
            'min_overlap': (
                max(0, best_params.min_overlap - radius * 10),
                min(100, best_params.min_overlap + radius * 10)
            ),
            'max_overlap': (
                max(0, best_params.max_overlap - radius * 20),
                min(100, best_params.max_overlap + radius * 20)
            ),
            'min_order': (
                max(1, best_params.min_order - int(radius * 5)),
                min(50, best_params.min_order + int(radius * 5))
            ),
            'max_order': (
                max(1, best_params.max_order - int(radius * 10)),
                min(50, best_params.max_order + int(radius * 10))
            )
        }
    
    @staticmethod
    def adaptive_zoom_box(results: List[Tuple[float, Params, ScoreBreakdown]], 
                         top_k: int = 10) -> Dict[str, Tuple[float, float]]:
        """Calculate adaptive zoom box based on top-k results."""
        if not results:
            return {}
        
        # Sort by score and take top-k
        sorted_results = sorted(results, key=lambda x: x[0], reverse=True)[:top_k]
        
        # Extract parameter ranges
        min_overlaps = [r[1].min_overlap for r in sorted_results]
        max_overlaps = [r[1].max_overlap for r in sorted_results]
        min_orders = [r[1].min_order for r in sorted_results]
        max_orders = [r[1].max_order for r in sorted_results]
        
        # Calculate zoom box with some margin
        margin = 0.1  # 10% margin
        
        return {
            'min_overlap': (
                max(0, min(min_overlaps) * (1 - margin)),
                min(100, max(min_overlaps) * (1 + margin))
            ),
            'max_overlap': (
                max(0, min(max_overlaps) * (1 - margin)),
                min(100, max(max_overlaps) * (1 + margin))
            ),
            'min_order': (
                max(1, min(min_orders) - 1),
                min(50, max(min_orders) + 1)
            ),
            'max_order': (
                max(1, min(max_orders) - 1),
                min(50, max(max_orders) + 1)
            )
        }


class ResultAggregator:
    """Aggregate and summarize optimization results."""
    
    @staticmethod
    def aggregate_statistics(results: List[Tuple[float, Params, ScoreBreakdown]]) -> Dict[str, Any]:
        """Calculate aggregate statistics from results."""
        if not results:
            return {}
        
        scores = [r[0] for r in results]
        max_scores = [r[2].max_score for r in results]
        variance_scores = [r[2].variance_score for r in results]
        tail_scores = [r[2].tail_score for r in results]
        penalty_totals = [r[2].penalty_total for r in results]
        
        return {
            'num_results': len(results),
            'score_stats': {
                'mean': np.mean(scores),
                'std': np.std(scores),
                'min': np.min(scores),
                'max': np.max(scores),
                'median': np.median(scores)
            },
            'component_stats': {
                'max_score_mean': np.mean(max_scores),
                'variance_score_mean': np.mean(variance_scores),
                'tail_score_mean': np.mean(tail_scores),
                'penalty_total_mean': np.mean(penalty_totals)
            },
            'best_score': max(scores),
            'worst_score': min(scores)
        }
    
    @staticmethod
    def get_parameter_distributions(results: List[Tuple[float, Params, ScoreBreakdown]]) -> Dict[str, Dict[str, float]]:
        """Get parameter distributions from results."""
        if not results:
            return {}
        
        params_list = [r[1] for r in results]
        
        return {
            'min_overlap': {
                'mean': np.mean([p.min_overlap for p in params_list]),
                'std': np.std([p.min_overlap for p in params_list]),
                'min': np.min([p.min_overlap for p in params_list]),
                'max': np.max([p.min_overlap for p in params_list])
            },
            'max_overlap': {
                'mean': np.mean([p.max_overlap for p in params_list]),
                'std': np.std([p.max_overlap for p in params_list]),
                'min': np.min([p.max_overlap for p in params_list]),
                'max': np.max([p.max_overlap for p in params_list])
            },
            'min_order': {
                'mean': np.mean([p.min_order for p in params_list]),
                'std': np.std([p.min_order for p in params_list]),
                'min': np.min([p.min_order for p in params_list]),
                'max': np.max([p.min_order for p in params_list])
            },
            'max_order': {
                'mean': np.mean([p.max_order for p in params_list]),
                'std': np.std([p.max_order for p in params_list]),
                'min': np.min([p.max_order for p in params_list]),
                'max': np.max([p.max_order for p in params_list])
            }
        }


class ConvergenceChecker:
    """Check convergence of optimization process."""
    
    def __init__(self, window_size: int = 10, tolerance: float = 1e-6):
        """Initialize convergence checker."""
        self.window_size = window_size
        self.tolerance = tolerance
        self.score_history = []
    
    def add_score(self, score: float):
        """Add a score to the history."""
        self.score_history.append(score)
        if len(self.score_history) > self.window_size:
            self.score_history.pop(0)
    
    def is_converged(self) -> bool:
        """Check if optimization has converged."""
        if len(self.score_history) < self.window_size:
            return False
        
        # Check if the improvement in the last window is below tolerance
        recent_scores = self.score_history[-self.window_size:]
        improvement = max(recent_scores) - min(recent_scores)
        
        return improvement < self.tolerance
    
    def get_convergence_metrics(self) -> Dict[str, float]:
        """Get convergence metrics."""
        if len(self.score_history) < 2:
            return {}
        
        return {
            'current_score': self.score_history[-1],
            'best_score': max(self.score_history),
            'improvement': max(self.score_history) - min(self.score_history),
            'recent_improvement': max(self.score_history[-self.window_size:]) - min(self.score_history[-self.window_size:])
        }
