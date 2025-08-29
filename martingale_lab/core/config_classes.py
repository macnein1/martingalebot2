"""
Configuration classes for evaluation engine parameters.
Groups related parameters into logical, maintainable dataclasses.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List
import json
from enum import Enum


class WaveMode(str, Enum):
    """Wave generation modes."""
    ANCHORS = "anchors"
    BLOCKS = "blocks"


class StrategyType(str, Enum):
    """Adaptive strategy types."""
    BALANCED = "balanced"
    AGGRESSIVE = "aggressive"
    CONSERVATIVE = "conservative"


class RoundStrategy(str, Enum):
    """Rounding strategies for normalization."""
    TAIL_FIRST = "tail-first"
    LARGEST_REMAINDER = "largest-remainder"
    BALANCED = "balanced"
    MINIMAL = "minimal"


@dataclass
class CoreConfig:
    """Core optimization parameters (required)."""
    base_price: float = 1.0
    overlap_pct: float = 20.0
    num_orders: int = 10
    seed: Optional[int] = None
    
    def validate(self) -> List[str]:
        """Validate core parameters."""
        errors = []
        if self.base_price <= 0:
            errors.append("base_price must be positive")
        if not (0 < self.overlap_pct <= 100):
            errors.append("overlap_pct must be between 0 and 100")
        if self.num_orders < 2:
            errors.append("num_orders must be at least 2")
        return errors


@dataclass
class GenerationConfig:
    """Parameters for initial volume/indent generation."""
    # Wave pattern generation
    wave_pattern: bool = True
    wave_mode: str = WaveMode.ANCHORS
    anchors: int = 6
    blocks: int = 3
    wave_amp_min: float = 0.05
    wave_amp_max: float = 0.30
    
    # Wave thresholds
    wave_strong_threshold: float = 50.0
    wave_weak_threshold: float = 10.0
    
    # Smart initialization
    use_smart_init: bool = False
    history_db: Optional[str] = None
    
    # Basic generation
    min_indent_step: float = 0.05
    softmax_temp: float = 1.0
    first_volume_target: float = 1.0
    first_indent_target: float = 0.0
    
    def validate(self) -> List[str]:
        """Validate generation parameters."""
        errors = []
        if self.wave_mode not in [WaveMode.ANCHORS, WaveMode.BLOCKS]:
            errors.append(f"Invalid wave_mode: {self.wave_mode}")
        if not (0 <= self.wave_amp_min <= self.wave_amp_max <= 1):
            errors.append("Wave amplitudes must be in [0, 1] with min <= max")
        if self.anchors < 2:
            errors.append("anchors must be at least 2")
        if self.blocks < 1:
            errors.append("blocks must be at least 1")
        return errors


@dataclass
class HardConstraintConfig:
    """Hard constraints for schedule shape."""
    # Martingale constraints
    m2_min: float = 0.10
    m2_max: float = 1.00
    m_min: float = 0.05
    m_max: float = 1.00
    
    # Growth bands
    g_min: float = 1.01
    g_max: float = 1.20
    
    # Slope and shape
    slope_cap: float = 0.25
    strict_inc_eps: float = 1e-6
    second_upper_c2: float = 2.0
    
    # Mass distribution
    q1_cap: float = 22.0
    tail_floor: float = 32.0
    tail_cap: float = 0.40
    
    # Head/tail parameters
    m_head: float = 0.40
    m_tail: float = 0.20
    tau_scale: float = 1/3
    
    # Bootstrap and budget
    use_hc0_bootstrap: bool = True
    use_head_budget: bool = False
    head_budget_pct: float = 2.0
    
    # Minimum constraints
    firstK_min: float = 1.0
    
    # Legacy (deprecated)
    k_front: int = 3  # DEPRECATED
    front_cap: float = 5.0  # DEPRECATED
    
    def validate(self) -> List[str]:
        """Validate hard constraints."""
        errors = []
        if not (0 <= self.m2_min <= self.m2_max <= 10):
            errors.append("Invalid m2 bounds")
        if not (0 <= self.m_min <= self.m_max <= 10):
            errors.append("Invalid m bounds")
        if not (1 < self.g_min <= self.g_max < 2):
            errors.append("Invalid growth bounds")
        if not (0 < self.slope_cap < 1):
            errors.append("slope_cap must be in (0, 1)")
        if not (0 <= self.q1_cap <= 100):
            errors.append("q1_cap must be in [0, 100]")
        if not (0 <= self.tail_floor <= 100):
            errors.append("tail_floor must be in [0, 100]")
        if not (0 < self.tail_cap <= 1):
            errors.append("tail_cap must be in (0, 1]")
        return errors


@dataclass
class PenaltyWeightConfig:
    """Penalty weights for soft constraints."""
    # Basic penalties
    w_fixed: float = 3.0
    w_band: float = 2.0
    w_front: float = 3.0
    w_tv: float = 1.0
    w_sec: float = 3.0
    w_wave: float = 1.0
    
    # Shape penalties
    w_second: float = 3.0
    w_plateau: float = 2.0
    w_front_share: float = 2.0
    w_tailweak: float = 2.0
    w_slope: float = 1.0
    w_wave_shape: float = 1.0
    
    # Variance penalties
    w_varm: float = 2.0
    w_blocks: float = 1.0
    
    # Penalty preset
    penalty_preset: Optional[str] = None
    
    # Target values
    target_std: float = 0.10
    
    def apply_preset(self, preset: str):
        """Apply a penalty weight preset."""
        presets = {
            "explore": {
                "w_fixed": 2.0, "w_second": 2.0, "w_band": 1.5,
                "w_front": 2.0, "w_tv": 0.5, "w_wave": 0.5
            },
            "robust": {
                "w_fixed": 3.0, "w_second": 3.0, "w_band": 2.5,
                "w_front": 3.0, "w_tv": 1.5, "w_wave": 1.0
            },
            "tight": {
                "w_fixed": 4.0, "w_second": 4.0, "w_band": 3.0,
                "w_front": 4.0, "w_tv": 2.0, "w_wave": 1.5
            }
        }
        
        if preset in presets:
            for key, value in presets[preset].items():
                setattr(self, key, value)
            self.penalty_preset = preset
    
    def validate(self) -> List[str]:
        """Validate penalty weights."""
        errors = []
        # All weights should be non-negative
        for attr_name in dir(self):
            if attr_name.startswith('w_'):
                value = getattr(self, attr_name)
                if value < 0:
                    errors.append(f"{attr_name} must be non-negative")
        if not (0 <= self.target_std <= 1):
            errors.append("target_std must be in [0, 1]")
        return errors


@dataclass
class ScoringConfig:
    """Objective function scoring weights."""
    alpha: float = 0.5  # Weight for max_need
    beta: float = 0.3   # Weight for var_need
    gamma: float = 0.2  # Weight for tail
    lambda_penalty: float = 0.1  # Weight for penalties
    
    def validate(self) -> List[str]:
        """Validate scoring weights."""
        errors = []
        weights = [self.alpha, self.beta, self.gamma]
        if any(w < 0 for w in weights):
            errors.append("Scoring weights must be non-negative")
        if sum(weights) == 0:
            errors.append("At least one scoring weight must be positive")
        if self.lambda_penalty < 0:
            errors.append("lambda_penalty must be non-negative")
        return errors


@dataclass
class AdaptiveConfig:
    """Adaptive optimization parameters."""
    use_adaptive: bool = False
    strategy_type: str = StrategyType.BALANCED
    
    def validate(self) -> List[str]:
        """Validate adaptive parameters."""
        errors = []
        valid_strategies = [StrategyType.BALANCED, StrategyType.AGGRESSIVE, StrategyType.CONSERVATIVE]
        if self.use_adaptive and self.strategy_type not in valid_strategies:
            errors.append(f"Invalid strategy_type: {self.strategy_type}")
        return errors


@dataclass
class NormalizationConfig:
    """Output normalization settings."""
    post_round_2dp: bool = True
    post_round_strategy: str = RoundStrategy.TAIL_FIRST
    post_round_m2_tolerance: float = 0.05
    post_round_keep_v1_band: bool = True
    
    def validate(self) -> List[str]:
        """Validate normalization parameters."""
        errors = []
        valid_strategies = [RoundStrategy.TAIL_FIRST, RoundStrategy.LARGEST_REMAINDER, 
                          RoundStrategy.BALANCED, RoundStrategy.MINIMAL]
        if self.post_round_strategy not in valid_strategies:
            errors.append(f"Invalid post_round_strategy: {self.post_round_strategy}")
        if not (0 <= self.post_round_m2_tolerance <= 1):
            errors.append("post_round_m2_tolerance must be in [0, 1]")
        return errors


@dataclass
class EvaluationConfig:
    """Complete configuration for evaluation function."""
    core: CoreConfig = field(default_factory=CoreConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    constraints: HardConstraintConfig = field(default_factory=HardConstraintConfig)
    penalties: PenaltyWeightConfig = field(default_factory=PenaltyWeightConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    adaptive: AdaptiveConfig = field(default_factory=AdaptiveConfig)
    normalization: NormalizationConfig = field(default_factory=NormalizationConfig)
    
    def validate(self) -> Dict[str, List[str]]:
        """Validate all configurations."""
        errors = {}
        
        for config_name in ['core', 'generation', 'constraints', 'penalties', 
                           'scoring', 'adaptive', 'normalization']:
            config = getattr(self, config_name)
            config_errors = config.validate()
            if config_errors:
                errors[config_name] = config_errors
        
        return errors
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> EvaluationConfig:
        """Create from dictionary."""
        config = cls()
        
        if 'core' in data:
            config.core = CoreConfig(**data['core'])
        if 'generation' in data:
            config.generation = GenerationConfig(**data['generation'])
        if 'constraints' in data:
            config.constraints = HardConstraintConfig(**data['constraints'])
        if 'penalties' in data:
            config.penalties = PenaltyWeightConfig(**data['penalties'])
        if 'scoring' in data:
            config.scoring = ScoringConfig(**data['scoring'])
        if 'adaptive' in data:
            config.adaptive = AdaptiveConfig(**data['adaptive'])
        if 'normalization' in data:
            config.normalization = NormalizationConfig(**data['normalization'])
        
        return config
    
    @classmethod
    def from_json(cls, json_str: str) -> EvaluationConfig:
        """Create from JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)
    
    @classmethod
    def from_legacy_kwargs(cls, **kwargs) -> EvaluationConfig:
        """Create from legacy keyword arguments (backward compatibility)."""
        config = cls()
        
        # Map legacy parameters to new config structure
        # Core parameters
        if 'base_price' in kwargs:
            config.core.base_price = kwargs['base_price']
        if 'overlap_pct' in kwargs:
            config.core.overlap_pct = kwargs['overlap_pct']
        if 'num_orders' in kwargs:
            config.core.num_orders = kwargs['num_orders']
        if 'seed' in kwargs:
            config.core.seed = kwargs['seed']
        
        # Generation parameters
        for param in ['wave_pattern', 'wave_mode', 'anchors', 'blocks', 
                     'wave_amp_min', 'wave_amp_max', 'wave_strong_threshold',
                     'wave_weak_threshold', 'use_smart_init', 'history_db',
                     'min_indent_step', 'softmax_temp', 'first_volume_target',
                     'first_indent_target']:
            if param in kwargs:
                setattr(config.generation, param, kwargs[param])
        
        # Hard constraints
        for param in ['m2_min', 'm2_max', 'm_min', 'm_max', 'g_min', 'g_max',
                     'slope_cap', 'strict_inc_eps', 'second_upper_c2', 'q1_cap',
                     'tail_floor', 'tail_cap', 'm_head', 'm_tail', 'tau_scale',
                     'use_hc0_bootstrap', 'use_head_budget', 'head_budget_pct',
                     'firstK_min', 'k_front', 'front_cap']:
            if param in kwargs:
                setattr(config.constraints, param, kwargs[param])
        
        # Penalty weights
        for param in ['w_fixed', 'w_band', 'w_front', 'w_tv', 'w_sec', 'w_wave',
                     'w_second', 'w_plateau', 'w_front_share', 'w_tailweak',
                     'w_slope', 'w_wave_shape', 'w_varm', 'w_blocks',
                     'penalty_preset', 'target_std']:
            if param in kwargs:
                setattr(config.penalties, param, kwargs[param])
        
        # Apply preset if specified
        if config.penalties.penalty_preset:
            config.penalties.apply_preset(config.penalties.penalty_preset)
        
        # Scoring weights
        for param in ['alpha', 'beta', 'gamma', 'lambda_penalty']:
            if param in kwargs:
                setattr(config.scoring, param, kwargs[param])
        
        # Adaptive parameters
        for param in ['use_adaptive', 'strategy_type']:
            if param in kwargs:
                setattr(config.adaptive, param, kwargs[param])
        
        # Normalization parameters
        for param in ['post_round_2dp', 'post_round_strategy', 
                     'post_round_m2_tolerance', 'post_round_keep_v1_band']:
            if param in kwargs:
                setattr(config.normalization, param, kwargs[param])
        
        return config


# Preset configurations for common use cases
class ConfigPresets:
    """Predefined configurations for common optimization scenarios."""
    
    @staticmethod
    def exploration() -> EvaluationConfig:
        """Configuration for exploration phase - looser constraints."""
        config = EvaluationConfig()
        config.penalties.apply_preset("explore")
        config.constraints.slope_cap = 0.30
        config.constraints.m2_max = 1.20
        config.scoring.lambda_penalty = 0.05
        return config
    
    @staticmethod
    def production() -> EvaluationConfig:
        """Configuration for production - balanced constraints."""
        config = EvaluationConfig()
        config.penalties.apply_preset("robust")
        config.constraints.slope_cap = 0.25
        config.normalization.post_round_2dp = True
        return config
    
    @staticmethod
    def strict() -> EvaluationConfig:
        """Configuration for strict optimization - tight constraints."""
        config = EvaluationConfig()
        config.penalties.apply_preset("tight")
        config.constraints.slope_cap = 0.20
        config.constraints.m2_max = 0.80
        config.scoring.lambda_penalty = 0.15
        return config
    
    @staticmethod
    def fast_exit() -> EvaluationConfig:
        """Configuration optimized for fast exit strategies."""
        config = EvaluationConfig()
        config.constraints.tail_floor = 35.0
        config.constraints.q1_cap = 20.0
        config.penalties.w_tailweak = 3.0
        config.scoring.gamma = 0.3  # Higher weight on tail
        return config