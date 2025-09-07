"""
Configuration-based CLI utilities.
Provides config file loading and CLI argument mapping.
"""
from __future__ import annotations

import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import argparse
import sys

from martingale_lab.core.config_classes import (
    EvaluationConfig, ConfigPresets,
    CoreConfig, GenerationConfig, HardConstraintConfig,
    PenaltyWeightConfig, ScoringConfig, AdaptiveConfig, NormalizationConfig
)
from martingale_lab.core.config_adapter import ConfigBuilder


def add_config_arguments(parser: argparse.ArgumentParser):
    """
    Add configuration-related arguments to parser.
    """
    config_group = parser.add_argument_group('Configuration')
    
    # Config file loading
    config_group.add_argument(
        '--config', '--config-file', 
        type=str, 
        help='Load configuration from JSON/YAML file'
    )
    
    config_group.add_argument(
        '--config-preset',
        choices=['exploration', 'production', 'strict', 'fast_exit'],
        help='Use a predefined configuration preset'
    )
    
    config_group.add_argument(
        '--save-config',
        type=str,
        help='Save current configuration to file (JSON/YAML)'
    )
    
    config_group.add_argument(
        '--print-config',
        action='store_true',
        help='Print current configuration and exit'
    )
    
    return parser


def load_config_file(filepath: str) -> EvaluationConfig:
    """
    Load configuration from JSON or YAML file.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {filepath}")
    
    with open(path, 'r') as f:
        if path.suffix.lower() in ['.yaml', '.yml']:
            # Try to load YAML
            try:
                data = yaml.safe_load(f)
            except ImportError:
                print("Warning: PyYAML not installed. Install with: pip install pyyaml")
                print("Falling back to JSON parsing...")
                f.seek(0)
                data = json.load(f)
        else:
            # Assume JSON
            data = json.load(f)
    
    return EvaluationConfig.from_dict(data)


def save_config_file(config: EvaluationConfig, filepath: str):
    """
    Save configuration to JSON or YAML file.
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    data = config.to_dict()
    
    with open(path, 'w') as f:
        if path.suffix.lower() in ['.yaml', '.yml']:
            # Try to save as YAML
            try:
                yaml.safe_dump(data, f, default_flow_style=False, indent=2)
                print(f"Configuration saved to {filepath} (YAML)")
            except ImportError:
                # Fallback to JSON
                json.dump(data, f, indent=2)
                print(f"Configuration saved to {filepath} (JSON, PyYAML not installed)")
        else:
            # Save as JSON
            json.dump(data, f, indent=2)
            print(f"Configuration saved to {filepath} (JSON)")


def args_to_config(args: argparse.Namespace) -> EvaluationConfig:
    """
    Convert CLI arguments to EvaluationConfig.
    Maps both old-style and new-style arguments.
    """
    # Start with base config or preset
    if hasattr(args, 'config_preset') and args.config_preset:
        if args.config_preset == 'exploration':
            config = ConfigPresets.exploration()
        elif args.config_preset == 'production':
            config = ConfigPresets.production()
        elif args.config_preset == 'strict':
            config = ConfigPresets.strict()
        elif args.config_preset == 'fast_exit':
            config = ConfigPresets.fast_exit()
        else:
            config = EvaluationConfig()
    else:
        config = EvaluationConfig()
    
    # Load from config file if specified
    if hasattr(args, 'config') and args.config:
        config = load_config_file(args.config)
    
    # Override with CLI arguments (only if explicitly provided)
    
    # Core parameters
    if hasattr(args, 'base_price'):
        config.core.base_price = args.base_price
    if hasattr(args, 'seed') and args.seed is not None:
        config.core.seed = args.seed
    
    # Note: overlap and orders are handled by orchestrator search space
    
    # Generation parameters
    # Wave pattern default is True; allow explicit disable to override preset
    if hasattr(args, 'no_wave_pattern') and args.no_wave_pattern:
        config.generation.wave_pattern = False
    elif hasattr(args, 'wave_pattern'):
        config.generation.wave_pattern = True if args.wave_pattern else config.generation.wave_pattern
    if hasattr(args, 'wave_mode'):
        config.generation.wave_mode = args.wave_mode
    if hasattr(args, 'anchors'):
        config.generation.anchors = args.anchors
    if hasattr(args, 'blocks'):
        config.generation.blocks = args.blocks
    if hasattr(args, 'wave_amp_min'):
        config.generation.wave_amp_min = args.wave_amp_min
    if hasattr(args, 'wave_amp_max'):
        config.generation.wave_amp_max = args.wave_amp_max
    if hasattr(args, 'wave_strong_threshold'):
        config.generation.wave_strong_threshold = args.wave_strong_threshold
    if hasattr(args, 'wave_weak_threshold'):
        config.generation.wave_weak_threshold = args.wave_weak_threshold
    if hasattr(args, 'use_smart_init'):
        config.generation.use_smart_init = args.use_smart_init
    if hasattr(args, 'min_indent_step'):
        config.generation.min_indent_step = args.min_indent_step
    if hasattr(args, 'softmax_temp'):
        config.generation.softmax_temp = args.softmax_temp
    if hasattr(args, 'first_volume'):
        config.generation.first_volume_target = args.first_volume
    if hasattr(args, 'first_indent'):
        config.generation.first_indent_target = args.first_indent
    
    # Hard constraints
    if hasattr(args, 'm2_min'):
        config.constraints.m2_min = args.m2_min
    if hasattr(args, 'm2_max'):
        config.constraints.m2_max = args.m2_max
    if hasattr(args, 'm_min'):
        config.constraints.m_min = args.m_min
    if hasattr(args, 'm_max'):
        config.constraints.m_max = args.m_max
    
    # Parse g_pre_band and g_post_band
    if hasattr(args, 'g_pre_band') and args.g_pre_band:
        parts = args.g_pre_band.split(',')
        if len(parts) == 2:
            config.constraints.g_min = float(parts[0])
            config.constraints.g_max = float(parts[1])
    
    if hasattr(args, 'slope_cap'):
        config.constraints.slope_cap = args.slope_cap
    if hasattr(args, 'strict_inc_eps'):
        config.constraints.strict_inc_eps = args.strict_inc_eps
    if hasattr(args, 'second_upper_c2'):
        config.constraints.second_upper_c2 = args.second_upper_c2
    if hasattr(args, 'q1_cap'):
        config.constraints.q1_cap = args.q1_cap
    if hasattr(args, 'tail_floor'):
        config.constraints.tail_floor = args.tail_floor
    if hasattr(args, 'tail_cap'):
        config.constraints.tail_cap = args.tail_cap
    if hasattr(args, 'm_head'):
        config.constraints.m_head = args.m_head
    if hasattr(args, 'm_tail'):
        config.constraints.m_tail = args.m_tail
    if hasattr(args, 'tau_scale'):
        config.constraints.tau_scale = args.tau_scale
    if hasattr(args, 'use_hc0_bootstrap'):
        config.constraints.use_hc0_bootstrap = args.use_hc0_bootstrap
    if hasattr(args, 'use_head_budget'):
        config.constraints.use_head_budget = args.use_head_budget
    if hasattr(args, 'head_budget_pct'):
        config.constraints.head_budget_pct = args.head_budget_pct
    if hasattr(args, 'firstK_min'):
        config.constraints.firstK_min = args.firstK_min
    if hasattr(args, 'k_front'):
        config.constraints.k_front = args.k_front
    if hasattr(args, 'front_cap'):
        config.constraints.front_cap = args.front_cap
    
    # Penalty weights
    if hasattr(args, 'penalty_preset') and args.penalty_preset:
        config.penalties.apply_preset(args.penalty_preset)
    else:
        if hasattr(args, 'w_fixed'):
            config.penalties.w_fixed = args.w_fixed
        if hasattr(args, 'w_second'):
            config.penalties.w_second = args.w_second
        if hasattr(args, 'w_gband'):
            config.penalties.w_band = args.w_gband
        if hasattr(args, 'w_front'):
            config.penalties.w_front = args.w_front
        if hasattr(args, 'w_tv'):
            config.penalties.w_tv = args.w_tv
        if hasattr(args, 'w_wave'):
            config.penalties.w_wave = args.w_wave
        if hasattr(args, 'w_varm'):
            config.penalties.w_varm = args.w_varm
        if hasattr(args, 'w_blocks'):
            config.penalties.w_blocks = args.w_blocks
        if hasattr(args, 'w_plateau'):
            config.penalties.w_plateau = args.w_plateau
        if hasattr(args, 'w_front_share'):
            config.penalties.w_front_share = args.w_front_share
        if hasattr(args, 'w_tailweak'):
            config.penalties.w_tailweak = args.w_tailweak
        if hasattr(args, 'w_slope'):
            config.penalties.w_slope = args.w_slope
        if hasattr(args, 'w_wave_shape'):
            config.penalties.w_wave_shape = args.w_wave_shape
    
    if hasattr(args, 'target_std'):
        config.penalties.target_std = args.target_std
    
    # Scoring weights
    if hasattr(args, 'alpha'):
        config.scoring.alpha = args.alpha
    if hasattr(args, 'beta'):
        config.scoring.beta = args.beta
    if hasattr(args, 'gamma'):
        config.scoring.gamma = args.gamma
    if hasattr(args, 'penalty'):
        config.scoring.lambda_penalty = args.penalty
    
    # Adaptive parameters
    if hasattr(args, 'use_adaptive'):
        config.adaptive.use_adaptive = args.use_adaptive
    if hasattr(args, 'strategy_type'):
        config.adaptive.strategy_type = args.strategy_type
    
    # Normalization parameters
    if hasattr(args, 'post_round_2dp'):
        config.normalization.post_round_2dp = args.post_round_2dp
    if hasattr(args, 'post_round_strategy'):
        config.normalization.post_round_strategy = args.post_round_strategy
    if hasattr(args, 'post_round_m2_tolerance'):
        config.normalization.post_round_m2_tolerance = args.post_round_m2_tolerance
    if hasattr(args, 'post_round_keep_v1_band'):
        config.normalization.post_round_keep_v1_band = args.post_round_keep_v1_band
    
    return config


def print_config(config: EvaluationConfig):
    """
    Pretty print configuration.
    """
    print("\n" + "=" * 60)
    print("EVALUATION CONFIGURATION")
    print("=" * 60)
    
    print("\n📦 Core:")
    print(f"  Base Price: {config.core.base_price}")
    print(f"  Overlap: {config.core.overlap_pct}%")
    print(f"  Orders: {config.core.num_orders}")
    print(f"  Seed: {config.core.seed}")
    
    print("\n🎯 Generation:")
    print(f"  Wave Pattern: {config.generation.wave_pattern}")
    print(f"  Wave Mode: {config.generation.wave_mode}")
    print(f"  Anchors: {config.generation.anchors}")
    print(f"  Blocks: {config.generation.blocks}")
    print(f"  Wave Amplitude: [{config.generation.wave_amp_min:.2f}, {config.generation.wave_amp_max:.2f}]")
    print(f"  Smart Init: {config.generation.use_smart_init}")
    
    print("\n🔒 Constraints:")
    print(f"  M2 Range: [{config.constraints.m2_min:.2f}, {config.constraints.m2_max:.2f}]")
    print(f"  M Range: [{config.constraints.m_min:.2f}, {config.constraints.m_max:.2f}]")
    print(f"  Growth Band: [{config.constraints.g_min:.2f}, {config.constraints.g_max:.2f}]")
    print(f"  Slope Cap: {config.constraints.slope_cap:.2f}")
    print(f"  Q1 Cap: {config.constraints.q1_cap:.1f}%")
    print(f"  Tail Floor: {config.constraints.tail_floor:.1f}%")
    print(f"  HC0 Bootstrap: {config.constraints.use_hc0_bootstrap}")
    
    print("\n⚖️ Penalties:")
    if config.penalties.penalty_preset:
        print(f"  Preset: {config.penalties.penalty_preset}")
    print(f"  w_fixed: {config.penalties.w_fixed:.1f}")
    print(f"  w_second: {config.penalties.w_second:.1f}")
    print(f"  w_band: {config.penalties.w_band:.1f}")
    print(f"  w_front: {config.penalties.w_front:.1f}")
    print(f"  Target STD: {config.penalties.target_std:.2f}")
    
    print("\n📊 Scoring:")
    print(f"  Alpha (max_need): {config.scoring.alpha:.2f}")
    print(f"  Beta (var_need): {config.scoring.beta:.2f}")
    print(f"  Gamma (tail): {config.scoring.gamma:.2f}")
    print(f"  Lambda (penalty): {config.scoring.lambda_penalty:.2f}")
    
    if config.adaptive.use_adaptive:
        print("\n🔄 Adaptive:")
        print(f"  Strategy: {config.adaptive.strategy_type}")
    
    print("\n✨ Normalization:")
    print(f"  Round to 2dp: {config.normalization.post_round_2dp}")
    if config.normalization.post_round_2dp:
        print(f"  Strategy: {config.normalization.post_round_strategy}")
        print(f"  M2 Tolerance: {config.normalization.post_round_m2_tolerance:.3f}")
        print(f"  Keep V1 Band: {config.normalization.post_round_keep_v1_band}")
    
    print("\n" + "=" * 60)


def create_example_configs():
    """
    Create example configuration files.
    """
    examples_dir = Path("config_examples")
    examples_dir.mkdir(exist_ok=True)
    
    # Exploration config
    exploration = ConfigPresets.exploration()
    save_config_file(exploration, str(examples_dir / "exploration.json"))
    
    # Production config
    production = ConfigPresets.production()
    save_config_file(production, str(examples_dir / "production.json"))
    
    # Strict config
    strict = ConfigPresets.strict()
    save_config_file(strict, str(examples_dir / "strict.json"))
    
    # Custom aggressive config
    aggressive = EvaluationConfig()
    aggressive.constraints.m2_min = 0.20
    aggressive.constraints.m2_max = 0.60
    aggressive.constraints.slope_cap = 0.15
    aggressive.constraints.tail_floor = 40.0
    aggressive.scoring.gamma = 0.4  # Focus on tail
    aggressive.penalties.w_tailweak = 4.0
    save_config_file(aggressive, str(examples_dir / "aggressive.json"))
    
    print(f"Example configurations created in {examples_dir}/")
    print("  - exploration.json: For initial exploration")
    print("  - production.json: Balanced for production")
    print("  - strict.json: Tight constraints")
    print("  - aggressive.json: Fast exit focus")


if __name__ == "__main__":
    # Create example configs when run directly
    create_example_configs()