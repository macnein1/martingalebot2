"""
Adapter for backward compatibility between old and new evaluation function signatures.
"""
from __future__ import annotations

import warnings
from typing import Dict, Any, Optional
from functools import wraps
import logging

from .config_classes import (
    EvaluationConfig, CoreConfig, GenerationConfig, HardConstraintConfig,
    PenaltyWeightConfig, ScoringConfig, AdaptiveConfig, NormalizationConfig
)

logger = logging.getLogger(__name__)


def extract_config_from_kwargs(**kwargs) -> tuple[EvaluationConfig, Dict[str, Any]]:
    """
    Extract configuration from mixed kwargs.
    
    Returns:
        Tuple of (EvaluationConfig, remaining_kwargs)
    """
    config = EvaluationConfig()
    remaining = {}
    
    # Core parameters (required, passed directly)
    core_params = ['base_price', 'overlap_pct', 'num_orders', 'seed']
    for param in core_params:
        if param in kwargs:
            setattr(config.core, param, kwargs[param])
    
    # Generation parameters
    generation_params = {
        'wave_pattern', 'wave_mode', 'anchors', 'blocks',
        'wave_amp_min', 'wave_amp_max', 'wave_strong_threshold',
        'wave_weak_threshold', 'use_smart_init', 'history_db',
        'min_indent_step', 'softmax_temp', 'first_volume_target',
        'first_indent_target'
    }
    
    for param in generation_params:
        if param in kwargs:
            setattr(config.generation, param, kwargs[param])
    
    # Hard constraints
    constraint_params = {
        'm2_min', 'm2_max', 'm_min', 'm_max', 'g_min', 'g_max',
        'slope_cap', 'strict_inc_eps', 'second_upper_c2', 'q1_cap',
        'tail_floor', 'tail_cap', 'm_head', 'm_tail', 'tau_scale',
        'use_hc0_bootstrap', 'use_head_budget', 'head_budget_pct',
        'firstK_min', 'k_front', 'front_cap'
    }
    
    for param in constraint_params:
        if param in kwargs:
            setattr(config.constraints, param, kwargs[param])
    
    # Penalty weights
    penalty_params = {
        'w_fixed', 'w_band', 'w_front', 'w_tv', 'w_sec', 'w_wave',
        'w_second', 'w_plateau', 'w_front_share', 'w_tailweak',
        'w_slope', 'w_wave_shape', 'w_varm', 'w_blocks',
        'penalty_preset', 'target_std'
    }
    
    for param in penalty_params:
        if param in kwargs:
            setattr(config.penalties, param, kwargs[param])
    
    # Apply preset if specified
    if config.penalties.penalty_preset:
        config.penalties.apply_preset(config.penalties.penalty_preset)
    
    # Scoring weights
    scoring_params = {'alpha', 'beta', 'gamma', 'lambda_penalty'}
    for param in scoring_params:
        if param in kwargs:
            setattr(config.scoring, param, kwargs[param])
    
    # Adaptive parameters
    adaptive_params = {'use_adaptive', 'strategy_type'}
    for param in adaptive_params:
        if param in kwargs:
            setattr(config.adaptive, param, kwargs[param])
    
    # Normalization parameters
    norm_params = {
        'post_round_2dp', 'post_round_strategy',
        'post_round_m2_tolerance', 'post_round_keep_v1_band'
    }
    for param in norm_params:
        if param in kwargs:
            setattr(config.normalization, param, kwargs[param])
    
    # Collect remaining unknown parameters
    known_params = (
        set(core_params) | generation_params | constraint_params |
        penalty_params | scoring_params | adaptive_params | norm_params
    )
    
    for key, value in kwargs.items():
        if key not in known_params:
            remaining[key] = value
    
    return config, remaining


def config_to_flat_dict(config: EvaluationConfig) -> Dict[str, Any]:
    """
    Convert EvaluationConfig back to flat dictionary (for backward compatibility).
    """
    flat = {}
    
    # Core
    flat['base_price'] = config.core.base_price
    flat['overlap_pct'] = config.core.overlap_pct
    flat['num_orders'] = config.core.num_orders
    flat['seed'] = config.core.seed
    
    # Generation
    for attr in ['wave_pattern', 'wave_mode', 'anchors', 'blocks',
                 'wave_amp_min', 'wave_amp_max', 'wave_strong_threshold',
                 'wave_weak_threshold', 'use_smart_init', 'history_db',
                 'min_indent_step', 'softmax_temp', 'first_volume_target',
                 'first_indent_target']:
        flat[attr] = getattr(config.generation, attr)
    
    # Constraints
    for attr in ['m2_min', 'm2_max', 'm_min', 'm_max', 'g_min', 'g_max',
                 'slope_cap', 'strict_inc_eps', 'second_upper_c2', 'q1_cap',
                 'tail_floor', 'tail_cap', 'm_head', 'm_tail', 'tau_scale',
                 'use_hc0_bootstrap', 'use_head_budget', 'head_budget_pct',
                 'firstK_min', 'k_front', 'front_cap']:
        flat[attr] = getattr(config.constraints, attr)
    
    # Penalties
    for attr in ['w_fixed', 'w_band', 'w_front', 'w_tv', 'w_sec', 'w_wave',
                 'w_second', 'w_plateau', 'w_front_share', 'w_tailweak',
                 'w_slope', 'w_wave_shape', 'w_varm', 'w_blocks',
                 'penalty_preset', 'target_std']:
        flat[attr] = getattr(config.penalties, attr)
    
    # Scoring
    for attr in ['alpha', 'beta', 'gamma', 'lambda_penalty']:
        flat[attr] = getattr(config.scoring, attr)
    
    # Adaptive
    flat['use_adaptive'] = config.adaptive.use_adaptive
    flat['strategy_type'] = config.adaptive.strategy_type
    
    # Normalization
    for attr in ['post_round_2dp', 'post_round_strategy',
                 'post_round_m2_tolerance', 'post_round_keep_v1_band']:
        flat[attr] = getattr(config.normalization, attr)
    
    return flat


def deprecated_parameter_warning(param_name: str, suggestion: str = None):
    """Issue a deprecation warning for a parameter."""
    msg = f"Parameter '{param_name}' is deprecated and will be removed in v2.0"
    if suggestion:
        msg += f". {suggestion}"
    warnings.warn(msg, DeprecationWarning, stacklevel=3)


def validate_and_fix_config(config: EvaluationConfig) -> EvaluationConfig:
    """
    Validate configuration and fix common issues.
    """
    errors = config.validate()
    
    if errors:
        # Log validation errors
        for category, error_list in errors.items():
            for error in error_list:
                logger.warning(f"Config validation ({category}): {error}")
        
        # Try to fix common issues
        # Fix m2 bounds if they conflict
        if config.constraints.m2_min > config.constraints.m2_max:
            config.constraints.m2_min, config.constraints.m2_max = \
                config.constraints.m2_max, config.constraints.m2_min
            logger.info("Fixed m2_min/m2_max order")
        
        # Fix growth bounds
        if config.constraints.g_min > config.constraints.g_max:
            config.constraints.g_min, config.constraints.g_max = \
                config.constraints.g_max, config.constraints.g_min
            logger.info("Fixed g_min/g_max order")
        
        # Fix wave amplitudes
        if config.generation.wave_amp_min > config.generation.wave_amp_max:
            config.generation.wave_amp_min, config.generation.wave_amp_max = \
                config.generation.wave_amp_max, config.generation.wave_amp_min
            logger.info("Fixed wave_amp_min/wave_amp_max order")
    
    # Check for deprecated parameters
    if config.constraints.k_front != 3 or config.constraints.front_cap != 5.0:
        deprecated_parameter_warning(
            "k_front/front_cap",
            "Use q1_cap for Q1 mass control instead"
        )
    
    return config


def backward_compatible_wrapper(new_function):
    """
    Decorator to make new config-based function backward compatible.
    """
    @wraps(new_function)
    def wrapper(*args, **kwargs):
        # Check if called with new style (config objects)
        if len(args) >= 4 and isinstance(args[3], EvaluationConfig):
            # New style call
            return new_function(*args, **kwargs)
        
        # Old style call - convert kwargs to config
        # Handle positional arguments
        if len(args) >= 1:
            kwargs['base_price'] = args[0]
        if len(args) >= 2:
            kwargs['overlap_pct'] = args[1]
        if len(args) >= 3:
            kwargs['num_orders'] = args[2]
        if len(args) >= 4:
            kwargs['seed'] = args[3]
        
        # Extract config from kwargs
        config, remaining_kwargs = extract_config_from_kwargs(**kwargs)
        
        # Validate and fix config
        config = validate_and_fix_config(config)
        
        # Call new function with config
        return new_function(
            config.core.base_price,
            config.core.overlap_pct,
            config.core.num_orders,
            config,
            **remaining_kwargs
        )
    
    return wrapper


class ConfigBuilder:
    """
    Builder pattern for creating configurations programmatically.
    """
    def __init__(self):
        self.config = EvaluationConfig()
    
    def with_core(self, base_price: float = None, overlap_pct: float = None,
                  num_orders: int = None, seed: int = None) -> ConfigBuilder:
        """Set core parameters."""
        if base_price is not None:
            self.config.core.base_price = base_price
        if overlap_pct is not None:
            self.config.core.overlap_pct = overlap_pct
        if num_orders is not None:
            self.config.core.num_orders = num_orders
        if seed is not None:
            self.config.core.seed = seed
        return self
    
    def with_wave_generation(self, mode: str = None, anchors: int = None,
                            blocks: int = None) -> ConfigBuilder:
        """Configure wave generation."""
        if mode is not None:
            self.config.generation.wave_mode = mode
        if anchors is not None:
            self.config.generation.anchors = anchors
        if blocks is not None:
            self.config.generation.blocks = blocks
        return self
    
    def with_constraints(self, m2_min: float = None, m2_max: float = None,
                        slope_cap: float = None) -> ConfigBuilder:
        """Set constraint parameters."""
        if m2_min is not None:
            self.config.constraints.m2_min = m2_min
        if m2_max is not None:
            self.config.constraints.m2_max = m2_max
        if slope_cap is not None:
            self.config.constraints.slope_cap = slope_cap
        return self
    
    def with_scoring(self, alpha: float = None, beta: float = None,
                     gamma: float = None) -> ConfigBuilder:
        """Set scoring weights."""
        if alpha is not None:
            self.config.scoring.alpha = alpha
        if beta is not None:
            self.config.scoring.beta = beta
        if gamma is not None:
            self.config.scoring.gamma = gamma
        return self
    
    def with_preset(self, preset: str) -> ConfigBuilder:
        """Apply a preset configuration."""
        from .config_classes import ConfigPresets
        
        if preset == "exploration":
            self.config = ConfigPresets.exploration()
        elif preset == "production":
            self.config = ConfigPresets.production()
        elif preset == "strict":
            self.config = ConfigPresets.strict()
        elif preset == "fast_exit":
            self.config = ConfigPresets.fast_exit()
        else:
            raise ValueError(f"Unknown preset: {preset}")
        
        return self
    
    def build(self) -> EvaluationConfig:
        """Build and validate the configuration."""
        return validate_and_fix_config(self.config)