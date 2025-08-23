"""
Test for Simple DCA v2 Engine
"""
import sys
from pathlib import Path
import logging

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from martingale_lab.optimizer.simple_dca_engine import evaluate_simple_dca, create_bullets_format, validate_simple_result
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_simple_evaluation():
    """Test simple evaluation."""
    logger.info("Testing simple evaluation...")
    
    result = evaluate_simple_dca(
        base_price=1.0,
        overlap_pct=20.0,
        num_orders=5,
        seed=42,
        alpha=0.45,
        beta=0.20,
        gamma=0.20,
        delta=0.10,
        rho=0.05
    )
    
    is_valid, msg = validate_simple_result(result)
    assert is_valid, f"Result invalid: {msg}"
    
    logger.info(f"Score: {result['score']:.6f}")
    logger.info(f"Max Need: {result['max_need']:.2f}%")
    logger.info(f"Shape Reward: {result['shape_reward']:.3f}")
    
    # Test bullets
    bullets = create_bullets_format(result['schedule'])
    logger.info(f"Bullets count: {len(bullets)}")
    for bullet in bullets:
        logger.info(f"  {bullet}")
    
    return True


def test_wave_patterns():
    """Test wave patterns."""
    logger.info("Testing wave patterns...")
    
    result_wave = evaluate_simple_dca(
        base_price=1.0,
        overlap_pct=25.0,
        num_orders=6,
        seed=123,
        wave_pattern=True
    )
    
    result_no_wave = evaluate_simple_dca(
        base_price=1.0,
        overlap_pct=25.0,
        num_orders=6,
        seed=123,
        wave_pattern=False
    )
    
    assert validate_simple_result(result_wave)[0]
    assert validate_simple_result(result_no_wave)[0]
    
    logger.info(f"Wave enabled score: {result_wave['score']:.6f}")
    logger.info(f"Wave disabled score: {result_no_wave['score']:.6f}")
    
    return True


def main():
    """Run simple tests."""
    logger.info("🎯 Testing Simple DCA v2...")
    
    tests = [
        ("Simple Evaluation", test_simple_evaluation),
        ("Wave Patterns", test_wave_patterns),
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
    
    logger.info(f"\n🎯 Simple DCA v2 Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        logger.info("🎉 Simple DCA v2 system working!")
        
        # Show example
        result = evaluate_simple_dca(
            base_price=1.0,
            overlap_pct=18.5,
            num_orders=4,
            seed=999,
            wave_pattern=True,
            shape_template='late_surge'
        )
        
        logger.info(f"\n📋 Example Result:")
        logger.info(f"Score (J): {result['score']:.6f}")
        logger.info(f"Max Need: {result['max_need']:.2f}%")
        logger.info(f"Var Need: {result['var_need']:.6f}")
        logger.info(f"Tail: {result['tail']:.3f}")
        logger.info(f"Shape Reward: {result['shape_reward']:.3f}")
        logger.info(f"CVaR: {result['cvar_need']:.3f}")
        
        bullets = create_bullets_format(result['schedule'])
        logger.info("\nBullets:")
        for bullet in bullets:
            logger.info(f"  {bullet}")
        
        return True
    else:
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
