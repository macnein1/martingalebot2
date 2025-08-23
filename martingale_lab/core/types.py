"""
Core data types for martingale optimization.
"""
from dataclasses import dataclass
from typing import Dict, Any
import numpy as np


@dataclass
class Params:
    """Optimization parameters for martingale strategy."""
    min_overlap: float
    max_overlap: float
    min_order: int
    max_order: int
    
    # Optional advanced parameters
    risk_factor: float = 1.0
    smoothing_factor: float = 0.1
    tail_weight: float = 0.2
    
    def validate(self) -> bool:
        """Validate parameter constraints."""
        return (
            0 <= self.min_overlap <= self.max_overlap <= 100 and
            1 <= self.min_order <= self.max_order <= 50 and
            self.risk_factor > 0 and
            0 <= self.smoothing_factor <= 1 and
            0 <= self.tail_weight <= 1
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'min_overlap': self.min_overlap,
            'max_overlap': self.max_overlap,
            'min_order': self.min_order,
            'max_order': self.max_order,
            'risk_factor': self.risk_factor,
            'smoothing_factor': self.smoothing_factor,
            'tail_weight': self.tail_weight
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Params':
        """Create from dictionary."""
        return cls(**data)


@dataclass
class Schedule:
    """Martingale schedule with order levels and volumes."""
    orders: np.ndarray  # Order sizes
    volumes: np.ndarray  # Volume levels
    overlaps: np.ndarray  # Overlap percentages
    
    def __post_init__(self):
        """Validate schedule after initialization."""
        assert len(self.orders) == len(self.volumes) == len(self.overlaps)
        assert len(self.orders) > 0
    
    @property
    def num_levels(self) -> int:
        """Number of order levels."""
        return len(self.orders)
    
    def total_volume(self) -> float:
        """Calculate total volume."""
        return np.sum(self.volumes)
    
    def max_exposure(self) -> float:
        """Calculate maximum exposure."""
        return np.max(self.volumes)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'orders': self.orders.tolist(),
            'volumes': self.volumes.tolist(),
            'overlaps': self.overlaps.tolist()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Schedule':
        """Create from dictionary."""
        return cls(
            orders=np.array(data['orders']),
            volumes=np.array(data['volumes']),
            overlaps=np.array(data['overlaps'])
        )


@dataclass
class ScoreBreakdown:
    """Detailed breakdown of optimization score components."""
    total_score: float
    max_score: float
    variance_score: float
    tail_score: float
    gini_penalty: float = 0.0
    entropy_penalty: float = 0.0
    monotone_penalty: float = 0.0
    smoothness_penalty: float = 0.0
    
    @property
    def penalty_total(self) -> float:
        """Total penalty score."""
        return (self.gini_penalty + self.entropy_penalty + 
                self.monotone_penalty + self.smoothness_penalty)
    
    @property
    def base_score(self) -> float:
        """Score before penalties."""
        return self.max_score + self.variance_score + self.tail_score
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'total_score': self.total_score,
            'max_score': self.max_score,
            'variance_score': self.variance_score,
            'tail_score': self.tail_score,
            'gini_penalty': self.gini_penalty,
            'entropy_penalty': self.entropy_penalty,
            'monotone_penalty': self.monotone_penalty,
            'smoothness_penalty': self.smoothness_penalty
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ScoreBreakdown':
        """Create from dictionary."""
        return cls(**data)
