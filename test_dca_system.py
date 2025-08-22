"""
Test script for DCA/Martingale "İşlemden En Hızlı Çıkış" system
Tests the complete evaluation contract, orchestrator, and UI components.
"""
import sys
from pathlib import Path
import logging

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from martingale_lab.optimizer.dca_evaluation_contract import evaluation_function
from martingale_lab.orchestrator.dca_orchestrator import create_dca_orchestrator, DCAConfig
from martingale_lab.storage.experiments_store import ExperimentsStore
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_evaluation_contract():
    """Test the core evaluation contract functionality."""
    logger.info("Testing DCA evaluation contract...")
    
    # Test basic evaluation
    result = evaluation_function(
        base_price=1.0,
        overlap_pct=20.0,
        num_orders=5,
        seed=42,
        wave_pattern=False
    )
    
    # Verify contract structure
    required_keys = ['score', 'max_need', 'var_need', 'tail', 'schedule', 'sanity', 'diagnostics']
    for key in required_keys:
        assert key in result, f"Missing required key: {key}"
    
    # Verify schedule structure
    schedule = result['schedule']
    schedule_keys = ['indent_pct', 'volume_pct', 'martingale_pct', 'needpct', 'order_prices', 'price_step_pct']
    for key in schedule_keys:
        assert key in schedule, f"Missing schedule key: {key}"
    
    # Verify dimensions
    n_orders = 5
    assert len(schedule['volume_pct']) == n_orders
    assert len(schedule['martingale_pct']) == n_orders
    assert len(schedule['needpct']) == n_orders
    assert len(schedule['indent_pct']) == n_orders
    
    # Verify constraints
    volume_sum = sum(schedule['volume_pct'])
    assert abs(volume_sum - 100.0) < 1e-6, f"Volume sum should be 100, got {volume_sum}"
    
    # Verify first martingale is 0
    assert schedule['martingale_pct'][0] == 0.0, "First martingale should be 0"
    
    # Verify sanity checks
    sanity = result['sanity']
    assert isinstance(sanity['max_need_mismatch'], bool)
    assert isinstance(sanity['collapse_indents'], bool)
    assert isinstance(sanity['tail_overflow'], bool)
    
    logger.info("✅ Basic evaluation contract test passed")
    
    # Test wave pattern
    result_wave = evaluation_function(
        base_price=1.0,
        overlap_pct=20.0,
        num_orders=6,
        seed=42,
        wave_pattern=True,
        wave_strong_threshold=50.0,
        wave_weak_threshold=10.0
    )
    
    assert 'P_wave' in result_wave.get('penalties', {}), "Wave pattern penalty should be present"
    logger.info("✅ Wave pattern test passed")
    
    return True


def test_bullets_format():
    """Test the bullets format generation."""
    logger.info("Testing bullets format...")
    
    result = evaluation_function(
        base_price=1.0,
        overlap_pct=15.0,
        num_orders=3,
        seed=123
    )
    
    schedule = result['schedule']
    
    # Import the format function
    from ui.components.results_section import format_bullets
    bullets = format_bullets(schedule)
    
    assert len(bullets) == 3, f"Should have 3 bullets, got {len(bullets)}"
    
    # Check first bullet format (no martingale)
    first_bullet = bullets[0]
    assert "1. Emir:" in first_bullet
    assert "no martingale, first order" in first_bullet
    assert "NeedPct %" in first_bullet
    
    # Check second bullet format (has martingale)
    if len(bullets) > 1:
        second_bullet = bullets[1]
        assert "2. Emir:" in second_bullet
        assert "Martingale %" in second_bullet
        assert "NeedPct %" in second_bullet
    
    logger.info("✅ Bullets format test passed")
    logger.info(f"Sample bullets:\n" + "\n".join(bullets))
    
    return True


def test_sanity_checks():
    """Test sanity check detection."""
    logger.info("Testing sanity checks...")
    
    # Test with various parameters to trigger different sanity flags
    test_cases = [
        {"overlap_pct": 5.0, "num_orders": 10, "tail_cap": 0.1},  # May trigger tail overflow
        {"overlap_pct": 50.0, "num_orders": 2, "min_indent_step": 10.0},  # May trigger collapse
    ]
    
    sanity_flags_found = set()
    
    for i, params in enumerate(test_cases):
        result = evaluation_function(
            base_price=1.0,
            seed=42 + i,
            **params
        )
        
        sanity = result['sanity']
        for flag, value in sanity.items():
            if value:
                sanity_flags_found.add(flag)
                logger.info(f"Triggered sanity flag: {flag}")
    
    logger.info(f"✅ Sanity checks test passed. Flags found: {sanity_flags_found}")
    return True


def test_database_integration():
    """Test database storage and retrieval."""
    logger.info("Testing database integration...")
    
    # Create temporary store
    store = ExperimentsStore("test_experiments.db")
    
    # Create experiment
    config = {
        "overlap_min": 10.0,
        "overlap_max": 20.0,
        "orders_min": 3,
        "orders_max": 5,
        "alpha": 0.5,
        "beta": 0.3,
        "gamma": 0.2,
        "lambda_penalty": 0.1,
        "wave_pattern": False,
        "tail_cap": 0.4,
        "notes": "Test experiment"
    }
    
    exp_id = store.create_experiment("TestAdapter", config)
    assert exp_id is not None, "Experiment creation should return an ID"
    
    # Generate some test results
    test_results = []
    for i in range(5):
        result = evaluation_function(
            base_price=1.0,
            overlap_pct=15.0,
            num_orders=4,
            seed=100 + i
        )
        
        # Add required fields for database storage
        result['params'] = {
            "overlap_pct": 15.0,
            "num_orders": 4,
            "alpha": 0.5,
            "beta": 0.3,
            "gamma": 0.2
        }
        result['stable_id'] = f"test_{i}"
        
        test_results.append(result)
    
    # Store results
    inserted = store.upsert_results(exp_id, test_results)
    assert inserted == 5, f"Should insert 5 results, got {inserted}"
    
    # Retrieve results
    retrieved = store.get_top_results(experiment_id=exp_id, limit=10)
    assert len(retrieved) == 5, f"Should retrieve 5 results, got {len(retrieved)}"
    
    # Check result structure
    first_result = retrieved[0]
    required_fields = ['score', 'max_need', 'var_need', 'tail', 'schedule', 'sanity', 'diagnostics']
    for field in required_fields:
        assert field in first_result, f"Missing field in retrieved result: {field}"
    
    # Get experiment summary
    summary = store.get_experiment_summary(exp_id)
    assert summary is not None, "Should retrieve experiment summary"
    assert summary['statistics']['total_results'] == 5
    
    logger.info("✅ Database integration test passed")
    
    # Cleanup
    import os
    if os.path.exists("test_experiments.db"):
        os.remove("test_experiments.db")
    
    return True


def test_orchestrator():
    """Test the DCA orchestrator."""
    logger.info("Testing DCA orchestrator...")
    
    # Create minimal config for testing
    config = DCAConfig(
        overlap_min=10.0,
        overlap_max=20.0,
        orders_min=3,
        orders_max=5,
        n_candidates_per_batch=10,  # Small batch for testing
        max_batches=2,  # Only 2 batches for testing
        n_workers=1,  # Single threaded for testing
        random_seed=42
    )
    
    # Create orchestrator with test database
    store = ExperimentsStore("test_orchestrator.db")
    orchestrator = create_dca_orchestrator(
        overlap_range=(10.0, 20.0),
        orders_range=(3, 5),
        n_candidates=10,
        max_batches=2,
        random_seed=42
    )
    orchestrator.store = store
    
    # Run optimization
    results = orchestrator.run_optimization(notes="Test optimization")
    
    # Verify results structure
    assert 'experiment_id' in results
    assert 'best_candidates' in results
    assert 'statistics' in results
    
    # Check statistics
    stats = results['statistics']
    assert stats['total_evaluations'] > 0
    assert stats['batches_completed'] > 0
    assert stats['best_score'] < float('inf')
    
    # Check candidates
    candidates = results['best_candidates']
    assert len(candidates) > 0, "Should have at least one candidate"
    
    # Verify candidate structure
    first_candidate = candidates[0]
    required_keys = ['score', 'max_need', 'var_need', 'tail', 'schedule', 'sanity', 'diagnostics']
    for key in required_keys:
        assert key in first_candidate, f"Missing key in candidate: {key}"
    
    logger.info(f"✅ Orchestrator test passed. Found {len(candidates)} candidates, best score: {stats['best_score']:.6f}")
    
    # Cleanup
    import os
    if os.path.exists("test_orchestrator.db"):
        os.remove("test_orchestrator.db")
    
    return True


def run_all_tests():
    """Run all tests and report results."""
    logger.info("🎯 Starting DCA system tests...")
    
    tests = [
        ("Evaluation Contract", test_evaluation_contract),
        ("Bullets Format", test_bullets_format),
        ("Sanity Checks", test_sanity_checks),
        ("Database Integration", test_database_integration),
        ("Orchestrator", test_orchestrator),
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
    
    logger.info(f"\n🎯 Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        logger.info("🎉 All tests passed! DCA system is ready.")
        return True
    else:
        logger.error(f"💥 {failed} test(s) failed. Please fix issues before using the system.")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)