"""
Evaluation Engine V2 - Refactored with configuration classes.
Clean implementation using structured configuration objects.
"""
from __future__ import annotations

import json
import numpy as np
import time
from typing import Dict, Any, Optional
import traceback
import logging

from martingale_lab.core.config_classes import EvaluationConfig
from martingale_lab.core.config_adapter import (
    extract_config_from_kwargs, 
    config_to_flat_dict,
    backward_compatible_wrapper
)
from martingale_lab.utils.logging import get_eval_logger, should_log_eval

# Import existing evaluation function to delegate to
from martingale_lab.optimizer.evaluation_engine import (
    evaluation_function as evaluation_function_legacy,
    _ensure_json_serializable,
    create_bullets_format
)

logger = get_eval_logger()


def evaluation_function_v2(
    base_price: float,
    overlap_pct: float, 
    num_orders: int,
    config: Optional[EvaluationConfig] = None,
    **legacy_kwargs
) -> Dict[str, Any]:
    """
    Evaluation function with clean configuration-based interface.
    
    Args:
        base_price: Base price for calculations
        overlap_pct: Overlap percentage
        num_orders: Number of orders
        config: EvaluationConfig object containing all parameters
        **legacy_kwargs: Additional kwargs for backward compatibility
        
    Returns:
        Dict with evaluation results
    """
    start_time = time.time()
    
    # If no config provided, extract from legacy kwargs
    if config is None:
        config, remaining_kwargs = extract_config_from_kwargs(
            base_price=base_price,
            overlap_pct=overlap_pct,
            num_orders=num_orders,
            **legacy_kwargs
        )
        legacy_kwargs = remaining_kwargs
    else:
        # Update config with core parameters if different
        config.core.base_price = base_price
        config.core.overlap_pct = overlap_pct
        config.core.num_orders = num_orders
    
    # Validate configuration
    validation_errors = config.validate()
    if validation_errors:
        logger.warning(f"Configuration validation issues: {validation_errors}")
    
    # Log if sampling allows
    if should_log_eval():
        logger.debug(
            "Starting evaluation_v2",
            extra={
                "event": "EVAL_V2_CALL",
                "overlap": overlap_pct,
                "orders": num_orders,
                "config_preset": config.penalties.penalty_preset,
                "adaptive": config.adaptive.use_adaptive
            }
        )
    
    try:
        # Convert config to flat dict for legacy function
        flat_params = config_to_flat_dict(config)
        
        # Merge with any remaining legacy kwargs
        flat_params.update(legacy_kwargs)
        
        # Call legacy evaluation function
        result = evaluation_function_legacy(**flat_params)
        
        # Add config metadata to result
        result['_config'] = {
            'version': 'v2',
            'preset': config.penalties.penalty_preset,
            'adaptive': config.adaptive.use_adaptive,
            'normalized': config.normalization.post_round_2dp
        }
        
        # Log successful evaluation
        duration_ms = (time.time() - start_time) * 1000
        if should_log_eval():
            logger.debug(
                "Evaluation_v2 completed",
                extra={
                    "event": "EVAL_V2_RETURN",
                    "score": result.get('score'),
                    "duration_ms": duration_ms
                }
            )
        
        return result
        
    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        
        logger.error(
            f"Evaluation_v2 failed: {str(e)}",
            extra={
                "event": "EVAL_V2_ERROR",
                "error": str(e),
                "error_type": type(e).__name__,
                "duration_ms": duration_ms
            }
        )
        
        # Return error state
        return {
            "score": float("inf"),
            "max_need": float("inf"),
            "var_need": float("inf"),
            "tail": float("inf"),
            "schedule": {
                "indent_pct": [],
                "volume_pct": [],
                "martingale_pct": [],
                "needpct": [],
                "order_prices": [],
                "price_step_pct": []
            },
            "sanity": {
                "max_need_mismatch": True,
                "collapse_indents": True,
                "tail_overflow": True
            },
            "diagnostics": {
                "error_type": type(e).__name__,
                "error_msg": str(e)
            },
            "penalties": {},
            "_error": str(e),
            "_config": {
                'version': 'v2',
                'error': True
            }
        }


def batch_evaluate_v2(
    candidates: list[Dict[str, Any]],
    config: Optional[EvaluationConfig] = None
) -> list[Dict[str, Any]]:
    """
    Batch evaluation with configuration.
    
    Args:
        candidates: List of candidate parameters
        config: Shared configuration for all evaluations
        
    Returns:
        List of evaluation results
    """
    if not candidates:
        return []
    
    results = []
    for candidate in candidates:
        # Extract core parameters
        base_price = candidate.get('base_price', 1.0)
        overlap_pct = candidate.get('overlap_pct', 20.0)
        num_orders = candidate.get('num_orders', 10)
        
        # Use shared config or create from candidate
        if config:
            eval_config = config
        else:
            eval_config, _ = extract_config_from_kwargs(**candidate)
        
        # Evaluate
        result = evaluation_function_v2(
            base_price, overlap_pct, num_orders, eval_config
        )
        
        results.append(result)
    
    return results


def create_config_from_cli_args(args) -> EvaluationConfig:
    """
    Create EvaluationConfig from CLI arguments.
    
    Args:
        args: Parsed argparse arguments
        
    Returns:
        EvaluationConfig instance
    """
    config = EvaluationConfig()
    
    # Core parameters
    config.core.base_price = getattr(args, 'base_price', 1.0)
    
    # Generation parameters
    config.generation.wave_pattern = getattr(args, 'wave_pattern', True)
    config.generation.wave_mode = getattr(args, 'wave_mode', 'anchors')
    config.generation.anchors = getattr(args, 'anchors', 6)
    config.generation.blocks = getattr(args, 'blocks', 3)
    config.generation.wave_amp_min = getattr(args, 'wave_amp_min', 0.05)
    config.generation.wave_amp_max = getattr(args, 'wave_amp_max', 0.30)
    config.generation.use_smart_init = getattr(args, 'use_smart_init', False)
    config.generation.min_indent_step = getattr(args, 'min_indent_step', 0.05)
    config.generation.softmax_temp = getattr(args, 'softmax_temp', 1.0)
    config.generation.first_volume_target = getattr(args, 'first_volume', 0.01)
    config.generation.first_indent_target = getattr(args, 'first_indent', 0.0)
    
    # Hard constraints
    config.constraints.m2_min = getattr(args, 'm2_min', 0.10)
    config.constraints.m2_max = getattr(args, 'm2_max', 1.00)
    config.constraints.m_min = getattr(args, 'm_min', 0.05)
    config.constraints.m_max = getattr(args, 'm_max', 1.00)
    config.constraints.g_min = getattr(args, 'g_pre_min', 1.01)
    config.constraints.g_max = getattr(args, 'g_pre_max', 1.20)
    config.constraints.slope_cap = getattr(args, 'slope_cap', 0.25)
    config.constraints.q1_cap = getattr(args, 'q1_cap', 22.0)
    config.constraints.tail_floor = getattr(args, 'tail_floor', 32.0)
    config.constraints.tail_cap = getattr(args, 'tail_cap', 0.40)
    config.constraints.m_head = getattr(args, 'm_head', 0.40)
    config.constraints.m_tail = getattr(args, 'm_tail', 0.20)
    config.constraints.tau_scale = getattr(args, 'tau_scale', 1/3)
    config.constraints.use_hc0_bootstrap = getattr(args, 'use_hc0_bootstrap', True)
    config.constraints.use_head_budget = getattr(args, 'use_head_budget', False)
    config.constraints.head_budget_pct = getattr(args, 'head_budget_pct', 2.0)
    
    # Penalty weights
    penalty_preset = getattr(args, 'penalty_preset', None)
    if penalty_preset:
        config.penalties.apply_preset(penalty_preset)
    else:
        config.penalties.w_fixed = getattr(args, 'w_fixed', 3.0)
        config.penalties.w_second = getattr(args, 'w_second', 3.0)
        config.penalties.w_band = getattr(args, 'w_gband', 2.0)
        config.penalties.w_front = getattr(args, 'w_front', 3.0)
        config.penalties.w_tv = getattr(args, 'w_tv', 1.0)
        config.penalties.w_wave = getattr(args, 'w_wave', 1.0)
        config.penalties.w_varm = getattr(args, 'w_varm', 2.0)
        config.penalties.w_blocks = getattr(args, 'w_blocks', 1.0)
    
    config.penalties.target_std = getattr(args, 'target_std', 0.10)
    
    # Scoring weights
    config.scoring.alpha = getattr(args, 'alpha', 0.5)
    config.scoring.beta = getattr(args, 'beta', 0.3)
    config.scoring.gamma = getattr(args, 'gamma', 0.2)
    config.scoring.lambda_penalty = getattr(args, 'penalty', 0.1)
    
    # Adaptive parameters
    config.adaptive.use_adaptive = getattr(args, 'use_adaptive', False)
    config.adaptive.strategy_type = getattr(args, 'strategy_type', 'balanced')
    
    # Normalization
    config.normalization.post_round_2dp = getattr(args, 'post_round_2dp', True)
    config.normalization.post_round_strategy = getattr(args, 'post_round_strategy', 'tail-first')
    config.normalization.post_round_m2_tolerance = getattr(args, 'post_round_m2_tolerance', 0.05)
    config.normalization.post_round_keep_v1_band = getattr(args, 'post_round_keep_v1_band', True)
    
    return config


# Backward compatible wrapper
evaluation_function_compat = backward_compatible_wrapper(evaluation_function_v2)


# Export the new interface
__all__ = [
    'evaluation_function_v2',
    'batch_evaluate_v2',
    'create_config_from_cli_args',
    'evaluation_function_compat'
]