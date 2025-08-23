"""
Common interfaces for martingale optimization.
"""
from typing import Protocol, List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime


@dataclass
class SearchConfig:
    """Configuration for optimization search."""
    # Parameter bounds
    min_overlap: float
    max_overlap: float
    min_order: int
    max_order: int
    
    # M range (martingale multiplier)
    min_m: float = 1.5
    max_m: float = 3.0
    
    # Overlap range
    overlap_min: float = 0.1
    overlap_max: float = 50.0
    
    # Seed for reproducibility
    seed: Optional[int] = None
    
    # Budget constraints
    time_budget: Optional[float] = None  # seconds
    eval_budget: Optional[int] = None    # max evaluations
    iteration_budget: Optional[int] = None  # max iterations
    
    # Advanced parameters
    risk_factor: float = 1.0
    smoothing_factor: float = 0.1
    tail_weight: float = 0.2
    
    # Objective weights
    alpha: float = 0.4  # max score weight
    beta: float = 0.3   # variance weight
    gamma: float = 0.3  # tail weight
    
    # Penalty weights
    gini_penalty_weight: float = 1.0
    entropy_penalty_weight: float = 0.5
    monotone_penalty_weight: float = 2.0
    smoothness_penalty_weight: float = 1.0
    
    def validate(self) -> bool:
        """Validate configuration parameters."""
        return (
            0 <= self.min_overlap <= self.max_overlap <= 100 and
            1 <= self.min_order <= self.max_order <= 50 and
            self.min_m > 1.0 and self.max_m > self.min_m and
            self.overlap_min >= 0 and self.overlap_max <= 100 and
            self.overlap_min <= self.overlap_max and
            self.risk_factor > 0 and
            0 <= self.smoothing_factor <= 1 and
            0 <= self.tail_weight <= 1 and
            self.alpha >= 0 and self.beta >= 0 and self.gamma >= 0 and
            (self.alpha + self.beta + self.gamma) > 0
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'min_overlap': self.min_overlap,
            'max_overlap': self.max_overlap,
            'min_order': self.min_order,
            'max_order': self.max_order,
            'min_m': self.min_m,
            'max_m': self.max_m,
            'overlap_min': self.overlap_min,
            'overlap_max': self.overlap_max,
            'seed': self.seed,
            'time_budget': self.time_budget,
            'eval_budget': self.eval_budget,
            'iteration_budget': self.iteration_budget,
            'risk_factor': self.risk_factor,
            'smoothing_factor': self.smoothing_factor,
            'tail_weight': self.tail_weight,
            'alpha': self.alpha,
            'beta': self.beta,
            'gamma': self.gamma,
            'gini_penalty_weight': self.gini_penalty_weight,
            'entropy_penalty_weight': self.entropy_penalty_weight,
            'monotone_penalty_weight': self.monotone_penalty_weight,
            'smoothness_penalty_weight': self.smoothness_penalty_weight
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SearchConfig':
        """Create from dictionary."""
        return cls(**data)


@dataclass
class Candidate:
    """Optimization candidate with parameters and schedule."""
    # Parameters
    min_overlap: float
    max_overlap: float
    min_order: int
    max_order: int
    risk_factor: float
    smoothing_factor: float
    tail_weight: float
    
    # Schedule fields
    indent_pct: List[float]  # Indent percentages
    volume_pct: List[float]  # Volume percentages
    martingale_pct: List[float]  # Martingale percentages
    needpct: List[float]  # Need percentages
    order_prices: List[float]  # Order prices
    
    # Metadata
    stable_id: Optional[str] = None
    timestamp: Optional[datetime] = None
    
    def __post_init__(self):
        """Validate candidate after initialization."""
        if not self.stable_id:
            self.stable_id = self._generate_stable_id()
        if not self.timestamp:
            self.timestamp = datetime.now()
    
    def _generate_stable_id(self) -> str:
        """Generate stable ID from parameters."""
        import hashlib
        param_str = f"{self.min_overlap:.2f}_{self.max_overlap:.2f}_{self.min_order}_{self.max_order}_{self.risk_factor:.2f}_{self.smoothing_factor:.2f}_{self.tail_weight:.2f}"
        return hashlib.md5(param_str.encode()).hexdigest()[:16]
    
    def validate(self) -> bool:
        """Validate candidate data."""
        if len(self.indent_pct) != len(self.volume_pct) != len(self.martingale_pct) != len(self.needpct) != len(self.order_prices):
            return False
        
        if not all(0 <= pct <= 100 for pct in self.indent_pct):
            return False
        
        if not all(0 <= pct <= 100 for pct in self.volume_pct):
            return False
        
        if not all(0 <= pct <= 100 for pct in self.martingale_pct):
            return False
        
        if not all(0 <= pct <= 100 for pct in self.needpct):
            return False
        
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'min_overlap': self.min_overlap,
            'max_overlap': self.max_overlap,
            'min_order': self.min_order,
            'max_order': self.max_order,
            'risk_factor': self.risk_factor,
            'smoothing_factor': self.smoothing_factor,
            'tail_weight': self.tail_weight,
            'indent_pct': self.indent_pct,
            'volume_pct': self.volume_pct,
            'martingale_pct': self.martingale_pct,
            'needpct': self.needpct,
            'order_prices': self.order_prices,
            'stable_id': self.stable_id,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Candidate':
        """Create from dictionary."""
        # Handle timestamp conversion
        if 'timestamp' in data and data['timestamp']:
            data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        
        return cls(**data)


@dataclass
class ResultTrace:
    """Optimization result trace with metrics and metadata."""
    # Main score
    J: float  # Total objective score
    
    # Sub-metrics
    max_score: float
    variance_score: float
    tail_score: float
    
    # Penalty scores
    gini_penalty: float = 0.0
    entropy_penalty: float = 0.0
    monotone_penalty: float = 0.0
    smoothness_penalty: float = 0.0
    
    # Performance metrics
    duration: float = 0.0  # seconds
    eval_count: int = 0
    
    # Identification
    stable_id: Optional[str] = None
    candidate: Optional[Candidate] = None
    
    # Metadata
    timestamp: Optional[datetime] = None
    session_id: Optional[str] = None
    
    def __post_init__(self):
        """Initialize metadata after creation."""
        if not self.timestamp:
            self.timestamp = datetime.now()
    
    @property
    def penalty_total(self) -> float:
        """Calculate total penalty score."""
        return (self.gini_penalty + self.entropy_penalty + 
                self.monotone_penalty + self.smoothness_penalty)
    
    @property
    def base_score(self) -> float:
        """Calculate base score before penalties."""
        return self.max_score + self.variance_score + self.tail_score
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'J': self.J,
            'max_score': self.max_score,
            'variance_score': self.variance_score,
            'tail_score': self.tail_score,
            'gini_penalty': self.gini_penalty,
            'entropy_penalty': self.entropy_penalty,
            'monotone_penalty': self.monotone_penalty,
            'smoothness_penalty': self.smoothness_penalty,
            'duration': self.duration,
            'eval_count': self.eval_count,
            'stable_id': self.stable_id,
            'candidate': self.candidate.to_dict() if self.candidate else None,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'session_id': self.session_id
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ResultTrace':
        """Create from dictionary."""
        # Handle timestamp conversion
        if 'timestamp' in data and data['timestamp']:
            data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        
        # Handle candidate conversion
        if 'candidate' in data and data['candidate']:
            data['candidate'] = Candidate.from_dict(data['candidate'])
        
        return cls(**data)


class OptimizerAdapter(Protocol):
    """Protocol for optimizer adapters."""
    
    def search(self, config: SearchConfig, time_budget: Optional[float] = None) -> List[Candidate]:
        """
        Perform optimization search.
        
        Args:
            config: Search configuration
            time_budget: Optional time budget override
            
        Returns:
            List of optimization candidates
        """
        ...
    
    def get_trace(self) -> List[ResultTrace]:
        """
        Get optimization trace.
        
        Returns:
            List of result traces
        """
        ...
    
    def get_best_candidate(self) -> Optional[Candidate]:
        """
        Get best candidate found.
        
        Returns:
            Best candidate or None if no candidates
        """
        ...
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get optimization statistics.
        
        Returns:
            Dictionary with statistics
        """
        ...


# Type aliases for convenience
SearchResult = List[Candidate]
TraceResult = List[ResultTrace]
OptimizationStats = Dict[str, Any]
