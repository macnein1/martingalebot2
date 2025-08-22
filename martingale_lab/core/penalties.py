"""
Penalty functions for optimization constraints.
"""
import numpy as np
from typing import List, Tuple
from .types import Schedule


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
    def penalty(volumes: np.ndarray, min_entropy: float = 1.0, 
                weight: float = 1.0) -> float:
        """Calculate entropy penalty."""
        entropy = EntropyPenalty.calculate_entropy(volumes)
        return weight * max(0, min_entropy - entropy)


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
        if len(sequence) <= 1:
            return 0.0
        
        # Check if sequence is non-decreasing
        diffs = np.diff(sequence)
        violations = np.sum(diffs < 0)
        
        return weight * violations


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


class CompositePenalty:
    """Composite penalty combining multiple penalty functions."""
    
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
