"""
Test script for DCA v2 System - Multi-Objective "İşlemden En Hızlı Çıkış"
Tests the complete evaluation engine with penalties, constraints, and scoring.
"""
import sys
from pathlib import Path
import logging

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from martingale_lab.optimizer.dca_evaluation_engine import evaluate_dca_candidate, validate_evaluation_result, create_bullets_text
from martingale_lab.core.penalties import compute_all_penalties, compute_composite_score
from martingale_lab.core.constraints import apply_soft_constraints
from martingale_lab.core.jit_kernels import need_curve_calculation
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_multi_objective_evaluation():
    """Test the multi-objective evaluation system."""
    logger.info("Testing multi-objective evaluation...")
    
    result = evaluate_dca_candidate(
        base_price=1.0,
        overlap_pct=20.0,
        num_orders=5,
        seed=42,
        wave_pattern=True,
        shape_template='late_surge',
        alpha=0.45,
        beta=0.20,
        gamma=0.20,
        delta=0.10,
        rho=0.05
    )
    
    # Validate structure
    is_valid, msg = validate_evaluation_result(result)
    assert is_valid, f"Evaluation result invalid: {msg}"
    
    # Check new metrics
    assert 'shape_reward' in result
    assert 'cvar_need' in result
    assert 0.0 <= result['shape_reward'] <= 1.0
    assert result['cvar_need'] >= 0.0
    
    logger.info(f"✅ Multi-objective test passed")
    logger.info(f"Score: {result['score']:.6f}, Shape reward: {result['shape_reward']:.3f}, CVaR: {result['cvar_need']:.3f}")
    
    return True


def test_wave_pattern_rewards():
    """Test wave pattern reward system."""
    logger.info("Testing wave pattern rewards...")
    
    # Test with wave pattern enabled
    result_wave = evaluate_dca_candidate(
        base_price=1.0,
        overlap_pct=25.0,
        num_orders=6,
        seed=123,
        wave_pattern=True,
        wave_strong_threshold=50.0,
        wave_weak_threshold=10.0
    )
    
    # Test without wave pattern
    result_no_wave = evaluate_dca_candidate(
        base_price=1.0,
        overlap_pct=25.0,
        num_orders=6,
        seed=123,
        wave_pattern=False
    )
    
    # Both should have valid structures
    assert validate_evaluation_result(result_wave)[0]
    assert validate_evaluation_result(result_no_wave)[0]
    
    # Wave pattern should have wave_reward in penalties
    penalties_wave = result_wave.get('penalties', {})
    assert 'wave_reward' in penalties_wave
    
    logger.info("✅ Wave pattern test passed")
    return True


def test_shape_rewards():
    """Test different shape reward templates."""
    logger.info("Testing shape reward templates...")
    
    templates = ['late_surge', 'double_hump', 'flat']
    results = {}
    
    for template in templates:
        result = evaluate_dca_candidate(
            base_price=1.0,
            overlap_pct=20.0,
            num_orders=8,
            seed=456,
            shape_template=template
        )
        
        assert validate_evaluation_result(result)[0]
        results[template] = result['shape_reward']
        
        logger.info(f"Template {template}: shape_reward = {result['shape_reward']:.3f}")
    
    # All templates should produce valid rewards
    for template, reward in results.items():
        assert 0.0 <= reward <= 1.0, f"Invalid shape reward for {template}: {reward}"
    
    logger.info("✅ Shape rewards test passed")
    return True


def test_constraint_penalties():
    """Test constraint penalty system."""
    logger.info("Testing constraint penalties...")
    
    # Test with extreme parameters to trigger constraints
    result_extreme = evaluate_dca_candidate(
        base_price=1.0,
        overlap_pct=50.0,  # High overlap
        num_orders=3,      # Few orders
        seed=789,
        tail_cap=0.1       # Very restrictive tail cap
    )
    
    assert validate_evaluation_result(result_extreme)[0]
    
    # Should have constraint penalties
    penalties = result_extreme.get('penalties', {})
    constraint_penalties = {k: v for k, v in penalties.items() if k.startswith('constraint_')}
    
    logger.info(f"Constraint penalties found: {len(constraint_penalties)}")
    for k, v in constraint_penalties.items():
        logger.info(f"  {k}: {v:.6f}")
    
    logger.info("✅ Constraint penalties test passed")
    return True


def test_bullets_format_v2():
    """Test the bullets format with new system."""
    logger.info("Testing bullets format v2...")
    
    result = evaluate_dca_candidate(
        base_price=1.0,
        overlap_pct=15.0,
        num_orders=4,
        seed=999
    )
    
    bullets = create_bullets_text(result['schedule'])
    
    assert len(bullets) == 4
    assert "1. Emir:" in bullets[0]
    assert "no martingale, first order" in bullets[0]
    assert "NeedPct %" in bullets[0]
    
    if len(bullets) > 1:
        assert "2. Emir:" in bullets[1]
        assert "Martingale %" in bullets[1]
        assert "NeedPct %" in bullets[1]
    
    logger.info("✅ Bullets format v2 test passed")
    for bullet in bullets:
        logger.info(f"  {bullet}")
    
    return True


def test_cvar_calculation():
    """Test CVaR calculation."""
    logger.info("Testing CVaR calculation...")
    
    from martingale_lab.core.penalties import cvar_calculation
    
    # Test with known values
    need_values = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 15.0, 20.0])
    cvar_80 = cvar_calculation(need_values, q=0.8)
    
    # CVaR should be the mean of top 20% (last 2 values)
    expected_cvar = np.mean([15.0, 20.0])
    assert abs(cvar_80 - expected_cvar) < 1e-6, f"CVaR mismatch: {cvar_80} != {expected_cvar}"
    
    logger.info(f"✅ CVaR test passed. CVaR@80% = {cvar_80:.3f}")
    return True


def test_performance_benchmark():
    """Test performance with multiple evaluations."""
    logger.info("Testing performance benchmark...")
    
    import time
    
    start_time = time.time()
    n_evaluations = 100
    
    for i in range(n_evaluations):
        result = evaluate_dca_candidate(
            base_price=1.0,
            overlap_pct=20.0,
            num_orders=5,
            seed=i,
            wave_pattern=i % 2 == 0,  # Alternate wave pattern
            shape_template='late_surge' if i % 3 == 0 else 'double_hump'
        )
        
        assert validate_evaluation_result(result)[0]
    
    elapsed = time.time() - start_time
    evals_per_second = n_evaluations / elapsed
    
    logger.info(f"✅ Performance test passed")
    logger.info(f"Evaluated {n_evaluations} candidates in {elapsed:.3f}s")
    logger.info(f"Performance: {evals_per_second:.1f} evaluations/second")
    
    # Should be reasonably fast
    assert evals_per_second > 10, f"Performance too slow: {evals_per_second:.1f} eval/s"
    
    return True


def main():
    """Run all DCA v2 tests."""
    logger.info("🎯 Starting DCA v2 System Tests...")
    
    tests = [
        ("Multi-Objective Evaluation", test_multi_objective_evaluation),
        ("Wave Pattern Rewards", test_wave_pattern_rewards),
        ("Shape Rewards", test_shape_rewards),
        ("Constraint Penalties", test_constraint_penalties),
        ("Bullets Format v2", test_bullets_format_v2),
        ("CVaR Calculation", test_cvar_calculation),
        ("Performance Benchmark", test_performance_benchmark),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            logger.info(f"\n--- Running {test_name} Test ---")
            test_func()
            passed += 1
            logger.info(f"✅ {test_name} test PASSED")
        except Exception as e:
            failed += 1
            logger.error(f"❌ {test_name} test FAILED: {e}")
            import traceback
            traceback.print_exc()
    
    logger.info(f"\n🎯 DCA v2 Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        logger.info("🎉 All DCA v2 tests passed! System is ready.")
        
        # Show comprehensive example
        logger.info("\n📋 Comprehensive DCA v2 Example:")
        result = evaluate_dca_candidate(
            base_price=1.0,
            overlap_pct=22.5,
            num_orders=6,
            seed=12345,
            wave_pattern=True,
            shape_template='late_surge',
            alpha=0.45,
            beta=0.20,
            gamma=0.20,
            delta=0.10,
            rho=0.05,
            tail_cap=0.35
        )
        
        logger.info(f"Final Score (J): {result['score']:.6f}")
        logger.info(f"Max Need: {result['max_need']:.2f}%")
        logger.info(f"Var Need: {result['var_need']:.6f}")
        logger.info(f"Tail: {result['tail']:.3f}")
        logger.info(f"Shape Reward: {result['shape_reward']:.3f}")
        logger.info(f"CVaR@80%: {result['cvar_need']:.3f}")
        logger.info(f"WCI: {result['diagnostics']['wci']:.3f}")
        
        # Show bullets
        bullets = create_bullets_text(result['schedule'])
        logger.info("\nOrder Bullets:")
        for bullet in bullets:
            logger.info(f"  {bullet}")
        
        # Show sanity status
        sanity_flags = [k for k, v in result['sanity'].items() if v]
        if sanity_flags:
            logger.info(f"\nSanity Flags: {sanity_flags}")
        else:
            logger.info("\nSanity: All checks passed ✅")
        
        return True
    else:
        logger.error(f"💥 {failed} test(s) failed.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
