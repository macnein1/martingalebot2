"""
Penalty functions for optimization constraints.
Enhanced with practical defaults and comprehensive penalty system.
"""
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from .types import Schedule


# Practical default penalty weights
DEFAULT_PENALTY_WEIGHTS = {
    'penalty_gini': 0.5,        # Prevent concentration in single order
    'penalty_entropy': 0.2,     # Encourage diversity
    'penalty_monotone': 1.0,    # Enforce indent monotonicity (critical)
    'penalty_step_smooth': 0.1, # Smooth NeedPct transitions
    'penalty_tail_cap': 2.0,    # Limit last order volume
    'penalty_extreme_vol': 1.5, # Prevent extreme volume values
    'penalty_need_variance': 0.3 # Penalize high NeedPct variance
}


def monotone_violation(sequence: np.ndarray) -> float:
    """
    Calculate monotonicity violation penalty.
    Returns 0.0 if sequence is monotonically non-decreasing.
    
    Args:
        sequence: Array to check for monotonicity
        
    Returns:
        Violation penalty (0.0 if monotonic)
    """
    if len(sequence) <= 1:
        return 0.0
    
    violations = 0.0
    for i in range(len(sequence) - 1):
        if sequence[i] > sequence[i + 1]:
            violations += (sequence[i] - sequence[i + 1])
    
    return violations


def tail_cap_penalty(volumes: np.ndarray, max_last_order_pct: float = 0.25) -> float:
    """
    Penalize if last order's volume exceeds maximum percentage.
    
    Args:
        volumes: Volume array
        max_last_order_pct: Maximum allowed percentage for last order
        
    Returns:
        Penalty value
    """
    if len(volumes) == 0:
        return 0.0
    
    last_volume_pct = volumes[-1]
    if last_volume_pct > max_last_order_pct:
        return (last_volume_pct - max_last_order_pct) * 10  # Strong penalty
    
    return 0.0


def extreme_volume_penalty(volumes: np.ndarray, min_vol: float = 0.01, 
                          max_vol: float = 0.5) -> float:
    """
    Penalize volumes outside reasonable bounds.
    
    Args:
        volumes: Volume array
        min_vol: Minimum reasonable volume
        max_vol: Maximum reasonable volume
        
    Returns:
        Penalty for extreme volumes
    """
    penalty = 0.0
    
    for vol in volumes:
        if vol < min_vol:
            penalty += (min_vol - vol) * 5  # Penalty for too small
        elif vol > max_vol:
            penalty += (vol - max_vol) * 5  # Penalty for too large
    
    return penalty


def need_pct_smoothness_penalty(need_pct_values: np.ndarray, 
                               max_jump_pct: float = 5.0) -> float:
    """
    Penalize large jumps in NeedPct values between consecutive orders.
    
    Args:
        need_pct_values: Array of NeedPct values
        max_jump_pct: Maximum allowed jump between consecutive values
        
    Returns:
        Smoothness penalty
    """
    if len(need_pct_values) <= 1:
        return 0.0
    
    penalty = 0.0
    for i in range(len(need_pct_values) - 1):
        jump = abs(need_pct_values[i + 1] - need_pct_values[i])
        if jump > max_jump_pct:
            penalty += (jump - max_jump_pct) * 0.5
    
    return penalty


def need_pct_variance_penalty(need_pct_values: np.ndarray, 
                             max_variance: float = 25.0) -> float:
    """
    Penalize high variance in NeedPct values.
    
    Args:
        need_pct_values: Array of NeedPct values
        max_variance: Maximum allowed variance
        
    Returns:
        Variance penalty
    """
    if len(need_pct_values) <= 1:
        return 0.0
    
    variance = np.var(need_pct_values)
    if variance > max_variance:
        return (variance - max_variance) * 0.1
    
    return 0.0


class GiniPenalty:
    """Gini coefficient penalty for volume distribution inequality."""
    
    @staticmethod
    def calculate_gini(volumes: np.ndarray) -> float:
        """Calculate Gini coefficient for volume distribution."""
        if len(volumes) <= 1:
            return 0.0
        
        sorted_volumes = np.sort(volumes)
        n = len(sorted_volumes)
        cumsum = np.cumsum(sorted_volumes)
        
        # Gini coefficient calculation
        return (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n
    
    @staticmethod
    def penalty(volumes: np.ndarray, target_gini: float = 0.3, 
                weight: float = 1.0) -> float:
        """Calculate Gini penalty."""
        gini = GiniPenalty.calculate_gini(volumes)
        return weight * abs(gini - target_gini)


class EntropyPenalty:
    """Entropy penalty for volume distribution diversity."""
    
    @staticmethod
    def calculate_entropy(volumes: np.ndarray) -> float:
        """Calculate entropy of volume distribution."""
        if len(volumes) <= 1:
            return 0.0
        
        # Normalize volumes to probabilities
        total = np.sum(volumes)
        if total == 0:
            return 0.0
        
        probabilities = volumes / total
        # Remove zero probabilities for log calculation
        probabilities = probabilities[probabilities > 0]
        
        return -np.sum(probabilities * np.log(probabilities))
    
    @staticmethod
    def max_entropy(num_levels: int) -> float:
        """Calculate maximum possible entropy for given number of levels."""
        return np.log(num_levels)
    
    @staticmethod
    def penalty(volumes: np.ndarray, weight: float = 1.0) -> float:
        """Calculate entropy penalty (higher entropy is better)."""
        entropy = EntropyPenalty.calculate_entropy(volumes)
        max_entropy = EntropyPenalty.max_entropy(len(volumes))
        
        # Penalize low entropy (lack of diversity)
        return weight * (max_entropy - entropy)


class TailPenalty:
    """Tail risk penalty for extreme volume values."""
    
    @staticmethod
    def calculate_tail_risk(volumes: np.ndarray, percentile: float = 0.95) -> float:
        """Calculate tail risk at given percentile."""
        if len(volumes) == 0:
            return 0.0
        
        threshold = np.percentile(volumes, percentile * 100)
        tail_volumes = volumes[volumes > threshold]
        
        if len(tail_volumes) == 0:
            return 0.0
        
        return np.mean(tail_volumes)
    
    @staticmethod
    def penalty(volumes: np.ndarray, max_tail_risk: float = 1000.0, 
                weight: float = 1.0) -> float:
        """Calculate tail risk penalty."""
        tail_risk = TailPenalty.calculate_tail_risk(volumes)
        return weight * max(0, tail_risk - max_tail_risk)


class MonotonePenalty:
    """Monotonicity penalty for non-increasing sequences."""
    
    @staticmethod
    def penalty(sequence: np.ndarray, weight: float = 1.0) -> float:
        """Calculate monotonicity penalty."""
        return weight * monotone_violation(sequence)


class SmoothnessPenalty:
    """Smoothness penalty for abrupt changes in sequence."""
    
    @staticmethod
    def penalty(sequence: np.ndarray, max_change: float = 0.5, 
                weight: float = 1.0) -> float:
        """Calculate smoothness penalty."""
        if len(sequence) <= 1:
            return 0.0
        
        # Calculate relative changes
        diffs = np.abs(np.diff(sequence))
        relative_changes = diffs / (sequence[:-1] + 1e-8)  # Avoid division by zero
        
        # Penalize changes larger than max_change
        violations = np.sum(relative_changes > max_change)
        
        return weight * violations


class ComprehensivePenaltySystem:
    """
    Comprehensive penalty system with practical defaults and detailed breakdown.
    
    Implements the scoring formula:
    J = α·max_need + β·var_need + γ·tail
      + λ1·Gini(volume) + λ2·(H_max - Entropy(volume))
      + λ3·MonotoneViol(indent) + λ4·SmoothnessPenalty(needpct)
      + λ5·TailCapPenalty(volume_last)
    """
    
    def __init__(self, weights: Optional[Dict[str, float]] = None,
                 objective_weights: Optional[Dict[str, float]] = None):
        """
        Initialize penalty system with weights.
        
        Args:
            weights: Penalty weights (uses defaults if None)
            objective_weights: Objective function weights (α, β, γ)
        """
        self.penalty_weights = {**DEFAULT_PENALTY_WEIGHTS, **(weights or {})}
        self.objective_weights = objective_weights or {
            'alpha': 0.4,   # max_need weight
            'beta': 0.3,    # var_need weight  
            'gamma': 0.3    # tail weight
        }
    
    def calculate_comprehensive_score(self, 
                                    max_need: float,
                                    var_need: float, 
                                    tail: float,
                                    volumes: np.ndarray,
                                    indent_pct: np.ndarray,
                                    need_pct_values: np.ndarray) -> Dict[str, float]:
        """
        Calculate comprehensive score with detailed breakdown.
        
        Args:
            max_need: Maximum NeedPct value
            var_need: Variance of NeedPct values
            tail: Tail risk measure
            volumes: Volume distribution
            indent_pct: Indent percentages
            need_pct_values: NeedPct values for smoothness
            
        Returns:
            Dictionary with score breakdown
        """
        # Primary objectives
        objective_score = (
            self.objective_weights['alpha'] * max_need +
            self.objective_weights['beta'] * var_need +
            self.objective_weights['gamma'] * tail
        )
        
        # Penalty calculations
        penalties = {}
        
        # Gini penalty
        gini_coeff = GiniPenalty.calculate_gini(volumes)
        penalties['gini'] = self.penalty_weights['penalty_gini'] * gini_coeff
        
        # Entropy penalty (encourage diversity)
        entropy = EntropyPenalty.calculate_entropy(volumes)
        max_entropy = EntropyPenalty.max_entropy(len(volumes))
        penalties['entropy'] = self.penalty_weights['penalty_entropy'] * (max_entropy - entropy)
        
        # Monotone penalty (critical - indent must be non-decreasing)
        penalties['monotone'] = (
            self.penalty_weights['penalty_monotone'] * 
            monotone_violation(indent_pct)
        )
        
        # Smoothness penalty for NeedPct transitions
        penalties['step_smooth'] = (
            self.penalty_weights['penalty_step_smooth'] * 
            need_pct_smoothness_penalty(need_pct_values)
        )
        
        # Tail cap penalty (limit last order volume)
        penalties['tail_cap'] = (
            self.penalty_weights['penalty_tail_cap'] * 
            tail_cap_penalty(volumes)
        )
        
        # Extreme volume penalty
        penalties['extreme_vol'] = (
            self.penalty_weights['penalty_extreme_vol'] * 
            extreme_volume_penalty(volumes)
        )
        
        # NeedPct variance penalty
        penalties['need_variance'] = (
            self.penalty_weights['penalty_need_variance'] * 
            need_pct_variance_penalty(need_pct_values)
        )
        
        # Total penalty
        total_penalty = sum(penalties.values())
        
        # Final score
        final_score = objective_score + total_penalty
        
        return {
            'final_score': final_score,
            'objective_score': objective_score,
            'total_penalty': total_penalty,
            'objective_breakdown': {
                'max_need_component': self.objective_weights['alpha'] * max_need,
                'var_need_component': self.objective_weights['beta'] * var_need,
                'tail_component': self.objective_weights['gamma'] * tail
            },
            'penalty_breakdown': penalties,
            'raw_metrics': {
                'max_need': max_need,
                'var_need': var_need,
                'tail': tail,
                'gini_coefficient': gini_coeff,
                'entropy': entropy,
                'max_entropy': max_entropy
            }
        }
    
    def get_penalty_summary(self, score_breakdown: Dict[str, Any]) -> str:
        """
        Generate human-readable penalty summary.
        
        Args:
            score_breakdown: Result from calculate_comprehensive_score
            
        Returns:
            Formatted summary string
        """
        penalties = score_breakdown['penalty_breakdown']
        
        summary_lines = [
            f"Final Score: {score_breakdown['final_score']:.4f}",
            f"Objective: {score_breakdown['objective_score']:.4f}, Penalties: {score_breakdown['total_penalty']:.4f}",
            "",
            "Penalty Breakdown:"
        ]
        
        for penalty_name, value in penalties.items():
            if value > 0.001:  # Only show significant penalties
                summary_lines.append(f"  {penalty_name}: {value:.4f}")
        
        if score_breakdown['penalty_breakdown']['monotone'] > 0:
            summary_lines.append("  ⚠️  CRITICAL: Monotonicity violation detected!")
        
        return "\n".join(summary_lines)


class CompositePenalty:
    """Composite penalty combining multiple penalty functions (legacy compatibility)."""
    
    def __init__(self, weights: dict = None):
        """Initialize with penalty weights."""
        self.weights = weights or {
            'gini': 1.0,
            'entropy': 0.5,
            'tail': 1.0,
            'monotone': 2.0,
            'smoothness': 1.0
        }
    
    def calculate_total_penalty(self, schedule: Schedule) -> float:
        """Calculate total penalty for a schedule."""
        total_penalty = 0.0
        
        # Gini penalty
        if 'gini' in self.weights:
            total_penalty += GiniPenalty.penalty(
                schedule.volumes, weight=self.weights['gini']
            )
        
        # Entropy penalty
        if 'entropy' in self.weights:
            total_penalty += EntropyPenalty.penalty(
                schedule.volumes, weight=self.weights['entropy']
            )
        
        # Tail penalty
        if 'tail' in self.weights:
            total_penalty += TailPenalty.penalty(
                schedule.volumes, weight=self.weights['tail']
            )
        
        # Monotone penalty for orders
        if 'monotone' in self.weights:
            total_penalty += MonotonePenalty.penalty(
                schedule.orders, weight=self.weights['monotone']
            )
        
        # Smoothness penalty for volumes
        if 'smoothness' in self.weights:
            total_penalty += SmoothnessPenalty.penalty(
                schedule.volumes, weight=self.weights['smoothness']
            )
        
        return total_penalty
    
    def get_penalty_breakdown(self, schedule: Schedule) -> dict:
        """Get detailed breakdown of penalties."""
        return {
            'gini': GiniPenalty.penalty(schedule.volumes, weight=self.weights.get('gini', 0)),
            'entropy': EntropyPenalty.penalty(schedule.volumes, weight=self.weights.get('entropy', 0)),
            'tail': TailPenalty.penalty(schedule.volumes, weight=self.weights.get('tail', 0)),
            'monotone': MonotonePenalty.penalty(schedule.orders, weight=self.weights.get('monotone', 0)),
            'smoothness': SmoothnessPenalty.penalty(schedule.volumes, weight=self.weights.get('smoothness', 0))
        }
