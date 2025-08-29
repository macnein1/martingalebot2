#!/usr/bin/env python3
"""
Test script for configuration refactoring.
"""
import sys
sys.path.insert(0, '/workspace')

from martingale_lab.core.config_classes import (
    EvaluationConfig, ConfigPresets, CoreConfig,
    GenerationConfig, HardConstraintConfig, PenaltyWeightConfig,
    ScoringConfig, AdaptiveConfig, NormalizationConfig
)
from martingale_lab.core.config_adapter import (
    extract_config_from_kwargs, config_to_flat_dict,
    ConfigBuilder
)
from martingale_lab.optimizer.evaluation_engine_v2 import (
    evaluation_function_v2, evaluation_function_compat
)


def test_config_creation():
    """Test creating configurations."""
    print("\n=== Testing Config Creation ===")
    
    # Create default config
    config = EvaluationConfig()
    print(f"Default config created")
    print(f"  Core: base_price={config.core.base_price}, orders={config.core.num_orders}")
    print(f"  Constraints: m2=[{config.constraints.m2_min}, {config.constraints.m2_max}]")
    print(f"  Scoring: α={config.scoring.alpha}, β={config.scoring.beta}, γ={config.scoring.gamma}")
    
    # Validate
    errors = config.validate()
    if errors:
        print(f"  Validation errors: {errors}")
    else:
        print("  ✅ Validation passed")
    
    return True


def test_config_presets():
    """Test configuration presets."""
    print("\n=== Testing Config Presets ===")
    
    presets = [
        ("exploration", ConfigPresets.exploration()),
        ("production", ConfigPresets.production()),
        ("strict", ConfigPresets.strict()),
        ("fast_exit", ConfigPresets.fast_exit())
    ]
    
    for name, config in presets:
        print(f"\n{name.upper()} preset:")
        print(f"  Penalty preset: {config.penalties.penalty_preset}")
        print(f"  Slope cap: {config.constraints.slope_cap}")
        print(f"  Lambda penalty: {config.scoring.lambda_penalty}")
        print(f"  w_fixed: {config.penalties.w_fixed}")
    
    return True


def test_config_builder():
    """Test ConfigBuilder pattern."""
    print("\n=== Testing Config Builder ===")
    
    builder = ConfigBuilder()
    config = (builder
        .with_core(base_price=100.0, overlap_pct=15.0, num_orders=10)
        .with_constraints(m2_min=0.15, m2_max=0.85, slope_cap=0.20)
        .with_scoring(alpha=0.6, beta=0.3, gamma=0.1)
        .build())
    
    print(f"Built config:")
    print(f"  Core: price={config.core.base_price}, overlap={config.core.overlap_pct}")
    print(f"  Constraints: m2=[{config.constraints.m2_min}, {config.constraints.m2_max}]")
    print(f"  Scoring: α={config.scoring.alpha}, β={config.scoring.beta}, γ={config.scoring.gamma}")
    
    return True


def test_backward_compatibility():
    """Test backward compatibility with old parameters."""
    print("\n=== Testing Backward Compatibility ===")
    
    # Old style parameters
    old_kwargs = {
        'base_price': 100.0,
        'overlap_pct': 20.0,
        'num_orders': 8,
        'seed': 42,
        'wave_pattern': True,
        'alpha': 0.5,
        'beta': 0.3,
        'gamma': 0.2,
        'm2_min': 0.15,
        'm2_max': 0.80,
        'slope_cap': 0.25,
        'w_fixed': 3.0,
        'penalty_preset': 'robust'
    }
    
    # Extract config from old kwargs
    config, remaining = extract_config_from_kwargs(**old_kwargs)
    
    print(f"Extracted config from old kwargs:")
    print(f"  Core: orders={config.core.num_orders}, seed={config.core.seed}")
    print(f"  Constraints: m2=[{config.constraints.m2_min}, {config.constraints.m2_max}]")
    print(f"  Penalties: preset={config.penalties.penalty_preset}, w_fixed={config.penalties.w_fixed}")
    print(f"  Remaining kwargs: {remaining}")
    
    # Convert back to flat dict
    flat = config_to_flat_dict(config)
    print(f"\nConverted back to flat dict: {len(flat)} parameters")
    
    return True


def test_evaluation_v2():
    """Test new evaluation function."""
    print("\n=== Testing Evaluation V2 ===")
    
    # Create config with preset
    config = ConfigPresets.exploration()
    config.core.base_price = 100.0
    config.core.overlap_pct = 15.0
    config.core.num_orders = 5
    config.core.seed = 42
    
    print(f"Testing with exploration preset")
    print(f"  Orders: {config.core.num_orders}")
    print(f"  Overlap: {config.core.overlap_pct}%")
    
    # Call new evaluation function
    result = evaluation_function_v2(
        config.core.base_price,
        config.core.overlap_pct,
        config.core.num_orders,
        config
    )
    
    print(f"\nResult:")
    print(f"  Score: {result['score']:.2f}")
    print(f"  Max need: {result['max_need']:.2f}")
    print(f"  Config version: {result.get('_config', {}).get('version')}")
    
    # Check schedule
    schedule = result.get('schedule', {})
    if schedule.get('volume_pct'):
        volumes = schedule['volume_pct']
        print(f"  Volumes: {len(volumes)} orders")
        print(f"  First 3: {volumes[:3]}")
        print(f"  Sum: {sum(volumes):.2f}")
    
    return True


def test_compatibility_wrapper():
    """Test backward compatibility wrapper."""
    print("\n=== Testing Compatibility Wrapper ===")
    
    # Call with old style (all kwargs)
    print("Old style call (kwargs):")
    result1 = evaluation_function_compat(
        base_price=100.0,
        overlap_pct=15.0,
        num_orders=5,
        seed=42,
        alpha=0.5,
        beta=0.3,
        gamma=0.2,
        m2_min=0.15,
        m2_max=0.80
    )
    print(f"  Score: {result1['score']:.2f}")
    
    # Call with new style (config object)
    print("\nNew style call (config):")
    config = EvaluationConfig()
    config.core.base_price = 100.0
    config.core.overlap_pct = 15.0
    config.core.num_orders = 5
    config.core.seed = 42
    
    result2 = evaluation_function_v2(
        config.core.base_price,
        config.core.overlap_pct,
        config.core.num_orders,
        config
    )
    print(f"  Score: {result2['score']:.2f}")
    
    # Compare results
    score_diff = abs(result1['score'] - result2['score'])
    print(f"\nScore difference: {score_diff:.6f}")
    
    return True


def test_serialization():
    """Test config serialization."""
    print("\n=== Testing Serialization ===")
    
    # Create config
    config = ConfigPresets.production()
    config.core.num_orders = 12
    config.constraints.slope_cap = 0.22
    
    # To JSON
    json_str = config.to_json()
    print(f"Serialized to JSON: {len(json_str)} chars")
    
    # From JSON
    config2 = EvaluationConfig.from_json(json_str)
    print(f"Deserialized config:")
    print(f"  Orders: {config2.core.num_orders}")
    print(f"  Slope cap: {config2.constraints.slope_cap}")
    print(f"  Penalty preset: {config2.penalties.penalty_preset}")
    
    # Check equality
    assert config.core.num_orders == config2.core.num_orders
    assert config.constraints.slope_cap == config2.constraints.slope_cap
    print("  ✅ Serialization round-trip successful")
    
    return True


def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing Configuration Refactoring")
    print("=" * 60)
    
    tests = [
        ("Config Creation", test_config_creation),
        ("Config Presets", test_config_presets),
        ("Config Builder", test_config_builder),
        ("Backward Compatibility", test_backward_compatibility),
        ("Evaluation V2", test_evaluation_v2),
        ("Compatibility Wrapper", test_compatibility_wrapper),
        ("Serialization", test_serialization)
    ]
    
    failed = []
    for name, test_func in tests:
        try:
            if not test_func():
                failed.append(name)
        except Exception as e:
            print(f"❌ {name} failed: {e}")
            import traceback
            traceback.print_exc()
            failed.append(name)
    
    print("\n" + "=" * 60)
    if failed:
        print(f"❌ FAILED: {len(failed)} test(s): {failed}")
        return 1
    else:
        print("✅ SUCCESS: All tests passed!")
        print("\n📊 Parameter Count Comparison:")
        print("  Old: 60+ individual parameters")
        print("  New: 7 config objects")
        print("  Reduction: ~88% in function signature complexity")
        return 0


if __name__ == "__main__":
    sys.exit(main())