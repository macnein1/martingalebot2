"""
Objective functions for martingale optimization.
"""
import numpy as np
from typing import List, Dict, Any, Callable
from ..core.types import ScoreBreakdown
from ..core.penalties import CompositePenalty


class ObjectiveFunction:
    """Base class for objective functions."""
    
    def __init__(self, alpha: float = 0.4, beta: float = 0.3, gamma: float = 0.3):
        """Initialize objective function weights."""
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
    
    def __call__(self, breakdown: ScoreBreakdown) -> float:
        """Calculate objective value."""
        raise NotImplementedError


class LinearObjective(ObjectiveFunction):
    """Linear objective function: J = α·max + β·var + γ·tail - penalties."""
    
    def __init__(self, alpha: float = 0.4, beta: float = 0.3, gamma: float = 0.3,
                 penalty_weights: Dict[str, float] = None):
        """Initialize linear objective."""
        super().__init__(alpha, beta, gamma)
        self.penalty_weights = penalty_weights or {
            'gini': 1.0,
            'entropy': 0.5,
            'tail': 1.0,
            'monotone': 2.0,
            'smoothness': 1.0
        }
    
    def __call__(self, breakdown: ScoreBreakdown) -> float:
        """Calculate linear objective value."""
        # Base components
        max_component = self.alpha * breakdown.max_score
        var_component = self.beta * breakdown.variance_score
        tail_component = self.gamma * breakdown.tail_score
        
        # Penalties
        penalty_component = (
            self.penalty_weights.get('gini', 0) * breakdown.gini_penalty +
            self.penalty_weights.get('entropy', 0) * breakdown.entropy_penalty +
            self.penalty_weights.get('monotone', 0) * breakdown.monotone_penalty +
            self.penalty_weights.get('smoothness', 0) * breakdown.smoothness_penalty
        )
        
        return max_component + var_component + tail_component - penalty_component


class ExponentialObjective(ObjectiveFunction):
    """Exponential objective function for non-linear optimization."""
    
    def __init__(self, alpha: float = 0.4, beta: float = 0.3, gamma: float = 0.3,
                 base: float = 2.0):
        """Initialize exponential objective."""
        super().__init__(alpha, beta, gamma)
        self.base = base
    
    def __call__(self, breakdown: ScoreBreakdown) -> float:
        """Calculate exponential objective value."""
        # Exponential components
        max_component = self.alpha * (self.base ** breakdown.max_score)
        var_component = self.beta * (self.base ** breakdown.variance_score)
        tail_component = self.gamma * (self.base ** breakdown.tail_score)
        
        # Penalties (linear)
        penalty_component = breakdown.penalty_total
        
        return max_component + var_component + tail_component - penalty_component


class MultiObjective:
    """Multi-objective optimization function."""
    
    def __init__(self, objectives: List[Callable], weights: List[float] = None):
        """Initialize multi-objective function."""
        self.objectives = objectives
        self.weights = weights or [1.0] * len(objectives)
        
        if len(self.weights) != len(self.objectives):
            raise ValueError("Number of weights must match number of objectives")
    
    def __call__(self, breakdown: ScoreBreakdown) -> List[float]:
        """Calculate multiple objective values."""
        return [obj(breakdown) for obj in self.objectives]
    
    def weighted_sum(self, breakdown: ScoreBreakdown) -> float:
        """Calculate weighted sum of objectives."""
        values = self(breakdown)
        return sum(w * v for w, v in zip(self.weights, values))


class AdaptiveObjective:
    """Adaptive objective function that adjusts weights based on performance."""
    
    def __init__(self, base_alpha: float = 0.4, base_beta: float = 0.3, base_gamma: float = 0.3,
                 adaptation_rate: float = 0.1):
        """Initialize adaptive objective."""
        self.base_alpha = base_alpha
        self.base_beta = base_beta
        self.base_gamma = base_gamma
        self.adaptation_rate = adaptation_rate
        self.performance_history = []
    
    def add_performance(self, breakdown: ScoreBreakdown, target_score: float):
        """Add performance data for adaptation."""
        self.performance_history.append((breakdown, target_score))
        
        # Keep only recent history
        if len(self.performance_history) > 100:
            self.performance_history.pop(0)
    
    def adapt_weights(self):
        """Adapt weights based on performance history."""
        if len(self.performance_history) < 10:
            return
        
        # Calculate correlation between components and target
        max_scores = [h[0].max_score for h in self.performance_history]
        var_scores = [h[0].variance_score for h in self.performance_history]
        tail_scores = [h[0].tail_score for h in self.performance_history]
        targets = [h[1] for h in self.performance_history]
        
        # Calculate correlations
        max_corr = np.corrcoef(max_scores, targets)[0, 1] if len(max_scores) > 1 else 0
        var_corr = np.corrcoef(var_scores, targets)[0, 1] if len(var_scores) > 1 else 0
        tail_corr = np.corrcoef(tail_scores, targets)[0, 1] if len(tail_scores) > 1 else 0
        
        # Adapt weights based on correlations
        self.base_alpha += self.adaptation_rate * max_corr
        self.base_beta += self.adaptation_rate * var_corr
        self.base_gamma += self.adaptation_rate * tail_corr
        
        # Normalize weights
        total = self.base_alpha + self.base_beta + self.base_gamma
        if total > 0:
            self.base_alpha /= total
            self.base_beta /= total
            self.base_gamma /= total
    
    def __call__(self, breakdown: ScoreBreakdown) -> float:
        """Calculate adaptive objective value."""
        return (self.base_alpha * breakdown.max_score +
                self.base_beta * breakdown.variance_score +
                self.base_gamma * breakdown.tail_score -
                breakdown.penalty_total)


class RobustObjective:
    """Robust objective function that considers worst-case scenarios."""
    
    def __init__(self, alpha: float = 0.4, beta: float = 0.3, gamma: float = 0.3,
                 robustness_factor: float = 0.1):
        """Initialize robust objective."""
        super().__init__(alpha, beta, gamma)
        self.robustness_factor = robustness_factor
    
    def __call__(self, breakdown: ScoreBreakdown) -> float:
        """Calculate robust objective value."""
        # Base objective
        base_objective = (self.alpha * breakdown.max_score +
                         self.beta * breakdown.variance_score +
                         self.gamma * breakdown.tail_score)
        
        # Robustness penalty based on variance
        robustness_penalty = self.robustness_factor * breakdown.variance_score
        
        # Total penalty
        total_penalty = breakdown.penalty_total + robustness_penalty
        
        return base_objective - total_penalty


# Factory functions for creating objective functions
def create_objective(objective_type: str = 'linear', **kwargs) -> ObjectiveFunction:
    """Create objective function by type."""
    if objective_type == 'linear':
        return LinearObjective(**kwargs)
    elif objective_type == 'exponential':
        return ExponentialObjective(**kwargs)
    elif objective_type == 'adaptive':
        return AdaptiveObjective(**kwargs)
    elif objective_type == 'robust':
        return RobustObjective(**kwargs)
    else:
        raise ValueError(f"Unknown objective type: {objective_type}")


def create_multi_objective(objective_types: List[str], weights: List[float] = None) -> MultiObjective:
    """Create multi-objective function."""
    objectives = [create_objective(obj_type) for obj_type in objective_types]
    return MultiObjective(objectives, weights)


# Predefined objective configurations
PREDEFINED_OBJECTIVES = {
    'balanced': {
        'type': 'linear',
        'alpha': 0.4,
        'beta': 0.3,
        'gamma': 0.3,
        'penalty_weights': {
            'gini': 1.0,
            'entropy': 0.5,
            'monotone': 2.0,
            'smoothness': 1.0
        }
    },
    'aggressive': {
        'type': 'linear',
        'alpha': 0.6,
        'beta': 0.2,
        'gamma': 0.2,
        'penalty_weights': {
            'gini': 0.5,
            'entropy': 0.2,
            'monotone': 1.0,
            'smoothness': 0.5
        }
    },
    'conservative': {
        'type': 'linear',
        'alpha': 0.2,
        'beta': 0.4,
        'gamma': 0.4,
        'penalty_weights': {
            'gini': 2.0,
            'entropy': 1.0,
            'monotone': 4.0,
            'smoothness': 2.0
        }
    },
    'robust': {
        'type': 'robust',
        'alpha': 0.4,
        'beta': 0.3,
        'gamma': 0.3,
        'robustness_factor': 0.2
    }
}


def get_predefined_objective(name: str) -> ObjectiveFunction:
    """Get predefined objective function by name."""
    if name not in PREDEFINED_OBJECTIVES:
        raise ValueError(f"Unknown predefined objective: {name}")
    
    config = PREDEFINED_OBJECTIVES[name]
    return create_objective(**config)
