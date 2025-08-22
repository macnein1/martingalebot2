"""
Simple test for DCA evaluation contract without UI dependencies
"""
import sys
from pathlib import Path
import logging

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from martingale_lab.optimizer.dca_evaluation_contract import evaluation_function
from martingale_lab.storage.experiments_store import ExperimentsStore
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_basic_evaluation():
    """Test basic evaluation functionality."""
    logger.info("Testing basic evaluation...")
    
    result = evaluation_function(
        base_price=1.0,
        overlap_pct=20.0,
        num_orders=5,
        seed=42,
        wave_pattern=False
    )
    
    # Check structure
    assert 'score' in result
    assert 'max_need' in result
    assert 'var_need' in result
    assert 'tail' in result
    assert 'schedule' in result
    assert 'sanity' in result
    assert 'diagnostics' in result
    
    schedule = result['schedule']
    assert len(schedule['volume_pct']) == 5
    assert abs(sum(schedule['volume_pct']) - 100.0) < 1e-6
    assert schedule['martingale_pct'][0] == 0.0
    
    logger.info(f"✅ Basic test passed. Score: {result['score']:.6f}")
    return True


def format_bullets_simple(schedule):
    """Simple bullets formatter without UI dependencies."""
    indent_pct = schedule.get("indent_pct", [])
    volume_pct = schedule.get("volume_pct", [])
    martingale_pct = schedule.get("martingale_pct", [])
    needpct = schedule.get("needpct", [])
    
    bullets = []
    n = len(volume_pct)
    
    for i in range(n):
        indent = indent_pct[i] if i < len(indent_pct) else 0.0
        volume = volume_pct[i] if i < len(volume_pct) else 0.0
        martingale = martingale_pct[i] if i < len(martingale_pct) else 0.0
        need = needpct[i] if i < len(needpct) else 0.0
        
        if i == 0:
            bullet = f"{i+1}. Emir: Indent %{indent:.2f}  Volume %{volume:.2f}  (no martingale, first order) — NeedPct %{need:.2f}"
        else:
            bullet = f"{i+1}. Emir: Indent %{indent:.2f}  Volume %{volume:.2f}  (Martingale %{martingale:.2f}) — NeedPct %{need:.2f}"
        
        bullets.append(bullet)
    
    return bullets


def test_bullets_format():
    """Test bullets format."""
    logger.info("Testing bullets format...")
    
    result = evaluation_function(
        base_price=1.0,
        overlap_pct=15.0,
        num_orders=3,
        seed=123
    )
    
    bullets = format_bullets_simple(result['schedule'])
    assert len(bullets) == 3
    assert "1. Emir:" in bullets[0]
    assert "no martingale, first order" in bullets[0]
    
    logger.info("✅ Bullets format test passed")
    for bullet in bullets:
        logger.info(f"  {bullet}")
    
    return True


def test_wave_pattern():
    """Test wave pattern functionality."""
    logger.info("Testing wave pattern...")
    
    result = evaluation_function(
        base_price=1.0,
        overlap_pct=20.0,
        num_orders=6,
        seed=42,
        wave_pattern=True
    )
    
    assert 'penalties' in result
    assert 'P_wave' in result['penalties']
    
    logger.info("✅ Wave pattern test passed")
    return True


def test_database():
    """Test database functionality."""
    logger.info("Testing database...")
    
    store = ExperimentsStore("simple_test.db")
    
    # Create experiment
    config = {
        "overlap_min": 10.0,
        "overlap_max": 20.0,
        "orders_min": 3,
        "orders_max": 5,
        "notes": "Simple test"
    }
    
    exp_id = store.create_experiment("SimpleTest", config)
    
    # Create test result
    result = evaluation_function(
        base_price=1.0,
        overlap_pct=15.0,
        num_orders=4,
        seed=100
    )
    
    result['params'] = {"test": "value"}
    result['stable_id'] = "test_001"
    
    inserted = store.upsert_results(exp_id, [result])
    assert inserted == 1
    
    retrieved = store.get_top_results(experiment_id=exp_id, limit=5)
    assert len(retrieved) == 1
    
    logger.info("✅ Database test passed")
    
    # Cleanup
    import os
    if os.path.exists("simple_test.db"):
        os.remove("simple_test.db")
    
    return True


def main():
    """Run all simple tests."""
    logger.info("🎯 Starting simple DCA tests...")
    
    tests = [
        ("Basic Evaluation", test_basic_evaluation),
        ("Bullets Format", test_bullets_format),
        ("Wave Pattern", test_wave_pattern),
        ("Database", test_database),
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
    
    logger.info(f"\n🎯 Simple Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        logger.info("🎉 All simple tests passed!")
        
        # Show a sample result
        logger.info("\n📋 Sample DCA Result:")
        result = evaluation_function(
            base_price=1.0,
            overlap_pct=18.5,
            num_orders=4,
            seed=999,
            wave_pattern=True
        )
        
        logger.info(f"Score (J): {result['score']:.6f}")
        logger.info(f"Max Need: {result['max_need']:.2f}%")
        logger.info(f"Var Need: {result['var_need']:.6f}")
        logger.info(f"Tail: {result['tail']:.3f}")
        logger.info(f"WCI: {result['diagnostics']['wci']:.3f}")
        logger.info(f"Sign Flips: {result['diagnostics']['sign_flips']}")
        
        logger.info("\nBullets:")
        bullets = format_bullets_simple(result['schedule'])
        for bullet in bullets:
            logger.info(f"  {bullet}")
        
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