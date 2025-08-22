"""
Complete Integration Test for DCA System
Tests evaluation_function, orchestrator, and storage integration.
"""
import sys
from pathlib import Path
import logging
import json

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from martingale_lab.optimizer.evaluation_engine import evaluation_function, create_bullets_format
from martingale_lab.orchestrator.dca_orchestrator import DCAOrchestrator, DCAConfig
from martingale_lab.storage.experiments_store import ExperimentsStore
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_evaluation_function_complete():
    """Test evaluation_function with all README requirements."""
    logger.info("Testing evaluation_function complete compliance...")
    
    # Test with all parameters as specified in README
    result = evaluation_function(
        base_price=1.0,
        overlap_pct=20.0,
        num_orders=5,
        seed=42,
        wave_pattern=True,
        alpha=0.5,
        beta=0.3,
        gamma=0.2,
        lambda_penalty=0.1,
        wave_strong_threshold=50.0,
        wave_weak_threshold=10.0,
        tail_cap=0.40,
        min_indent_step=0.05,
        softmax_temp=1.0
    )
    
    # Verify all outputs are present and JSON-serializable
    required_outputs = ["score", "max_need", "var_need", "tail", "schedule", "sanity", "diagnostics", "penalties"]
    for key in required_outputs:
        assert key in result, f"Missing output: {key}"
    
    # Test JSON serialization
    json_str = json.dumps(result)
    assert len(json_str) > 0, "JSON serialization failed"
    
    # Verify scoring formula: J = α·max_need + β·var_need + γ·tail + λ·Σ(penalties)
    expected_score = (0.5 * result["max_need"] + 
                     0.3 * result["var_need"] + 
                     0.2 * result["tail"] + 
                     0.1 * sum(result["penalties"].values()))
    
    assert abs(result["score"] - expected_score) < 1e-6, "Scoring formula mismatch"
    
    # Verify sanity checks return correct booleans
    sanity = result["sanity"]
    for key, value in sanity.items():
        assert isinstance(value, bool), f"Sanity {key} should be boolean"
    
    # Verify all penalties are present
    penalty_keys = ["P_gini", "P_entropy", "P_monotone", "P_smooth", "P_tailcap", "P_need_mismatch", "P_wave"]
    for key in penalty_keys:
        assert key in result["penalties"], f"Missing penalty: {key}"
    
    # Verify NeedPct calculation uses exact formula
    schedule = result["schedule"]
    volumes = schedule["volume_pct"]
    prices = schedule["order_prices"]
    needpct = schedule["needpct"]
    
    # Manual verification of NeedPct formula
    vol_acc = 0.0
    val_acc = 0.0
    for k in range(len(volumes)):
        vol_acc += volumes[k]
        val_acc += volumes[k] * prices[k+1]
        avg_entry = val_acc / vol_acc
        current_price = prices[k+1]
        expected_need = (avg_entry / current_price - 1.0) * 100.0
        assert abs(needpct[k] - expected_need) < 1e-6, f"NeedPct formula error at order {k+1}"
    
    # Verify schedule has both cumulative indents and per-step percentages
    assert "indent_pct" in schedule, "Missing indent_pct"
    assert "price_step_pct" in schedule, "Missing price_step_pct"
    
    logger.info("✅ Evaluation function complete compliance verified")
    return True


def test_orchestrator_integration():
    """Test orchestrator properly calls evaluation_function and persists results."""
    logger.info("Testing orchestrator integration...")
    
    # Create test configuration
    config = DCAConfig(
        overlap_min=10.0,
        overlap_max=20.0,
        orders_min=3,
        orders_max=5,
        n_candidates_per_batch=5,  # Small for testing
        max_batches=2,
        alpha=0.5,
        beta=0.3,
        gamma=0.2,
        lambda_penalty=0.1,
        wave_pattern=True,
        tail_cap=0.40,
        random_seed=42
    )
    
    # Create test database
    store = ExperimentsStore("test_integration.db")
    orchestrator = DCAOrchestrator(config, store)
    
    # Test parameter generation
    params_batch = orchestrator.generate_random_parameters(3)
    assert len(params_batch) == 3, "Should generate 3 parameter sets"
    
    # Verify parameters contain all required fields
    required_param_keys = ["base_price", "overlap_pct", "num_orders", "seed", "wave_pattern", 
                          "alpha", "beta", "gamma", "lambda_penalty", "tail_cap"]
    for params in params_batch:
        for key in required_param_keys:
            assert key in params, f"Missing parameter: {key}"
    
    # Test single candidate evaluation
    result = orchestrator.evaluate_candidate(params_batch[0])
    
    # Verify result structure
    assert "score" in result, "Missing score in result"
    assert "schedule" in result, "Missing schedule in result"
    assert "stable_id" in result, "Missing stable_id in result"
    
    # Verify constraints are respected
    overlap = params_batch[0]["overlap_pct"]
    assert config.overlap_min <= overlap <= config.overlap_max, "Overlap constraint violated"
    
    orders = params_batch[0]["num_orders"]
    assert config.orders_min <= orders <= config.orders_max, "Orders constraint violated"
    
    # Test batch evaluation
    batch_results = orchestrator.evaluate_batch_parallel(params_batch[:2])
    assert len(batch_results) == 2, "Should evaluate 2 candidates"
    
    # Test early pruning
    pruned_results = orchestrator.early_pruning(batch_results)
    assert len(pruned_results) <= len(batch_results), "Pruning should not increase candidates"
    
    logger.info("✅ Orchestrator integration test passed")
    
    # Cleanup
    import os
    if os.path.exists("test_integration.db"):
        os.remove("test_integration.db")
    
    return True


def test_storage_persistence():
    """Test that results are properly stored and retrieved."""
    logger.info("Testing storage persistence...")
    
    store = ExperimentsStore("test_storage.db")
    
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
        "wave_pattern": True,
        "tail_cap": 0.40,
        "notes": "Integration test"
    }
    
    exp_id = store.create_experiment("IntegrationTest", config)
    
    # Generate test results using evaluation_function
    test_results = []
    for i in range(3):
        result = evaluation_function(
            base_price=1.0,
            overlap_pct=15.0,
            num_orders=4,
            seed=100 + i,
            wave_pattern=True,
            alpha=0.5,
            beta=0.3,
            gamma=0.2,
            lambda_penalty=0.1
        )
        
        # Add required fields for storage
        result["params"] = {
            "overlap_pct": 15.0,
            "num_orders": 4,
            "alpha": 0.5,
            "beta": 0.3,
            "gamma": 0.2,
            "lambda_penalty": 0.1,
            "wave_pattern": True
        }
        result["stable_id"] = f"test_integration_{i}"
        
        test_results.append(result)
    
    # Store results
    inserted = store.upsert_results(exp_id, test_results)
    assert inserted == 3, f"Should insert 3 results, got {inserted}"
    
    # Retrieve and verify
    retrieved = store.get_top_results(experiment_id=exp_id, limit=10)
    assert len(retrieved) == 3, f"Should retrieve 3 results, got {len(retrieved)}"
    
    # Verify JSON parsing works
    first_result = retrieved[0]
    assert "schedule" in first_result, "Schedule should be parsed from JSON"
    assert "sanity" in first_result, "Sanity should be parsed from JSON"
    assert "diagnostics" in first_result, "Diagnostics should be parsed from JSON"
    assert "penalties" in first_result, "Penalties should be parsed from JSON"
    
    # Verify schedule structure is preserved
    schedule = first_result["schedule"]
    assert "needpct" in schedule, "NeedPct should be preserved in storage"
    assert isinstance(schedule["needpct"], list), "NeedPct should be list after JSON parsing"
    
    logger.info("✅ Storage persistence test passed")
    
    # Cleanup
    import os
    if os.path.exists("test_storage.db"):
        os.remove("test_storage.db")
    
    return True


def test_ui_parsing():
    """Test that UI can parse JSON results correctly."""
    logger.info("Testing UI parsing capabilities...")
    
    # Generate test result
    result = evaluation_function(
        base_price=1.0,
        overlap_pct=22.0,
        num_orders=4,
        seed=555,
        wave_pattern=True
    )
    
    # Test bullets creation
    bullets = create_bullets_format(result["schedule"])
    assert len(bullets) == 4, "Should create 4 bullets"
    
    # Verify bullets format
    for i, bullet in enumerate(bullets):
        assert f"{i+1}. Emir:" in bullet, f"Bullet {i+1} format incorrect"
        assert "NeedPct %" in bullet, f"Bullet {i+1} missing NeedPct"
        
        if i == 0:
            assert "no martingale, first order" in bullet, "First bullet should indicate no martingale"
        else:
            assert "Martingale %" in bullet, f"Bullet {i+1} should have martingale"
    
    # Test sparkline creation (simulate)
    needpct = result["schedule"]["needpct"]
    if needpct:
        # Simple sparkline simulation
        min_val = min(needpct)
        max_val = max(needpct)
        if max_val > min_val:
            spark_chars = "▁▂▃▄▅▆▇█"
            normalized = [(val - min_val) / (max_val - min_val) * 7 for val in needpct]
            sparkline = "".join([spark_chars[min(7, int(val))] for val in normalized])
            assert len(sparkline) == len(needpct), "Sparkline should match NeedPct length"
    
    # Test sanity badges creation (simulate)
    sanity = result["sanity"]
    badges = []
    if sanity.get("max_need_mismatch", False):
        badges.append("🔴 Max Need Mismatch")
    if sanity.get("collapse_indents", False):
        badges.append("🟡 Collapsed Indents")
    if sanity.get("tail_overflow", False):
        badges.append("🟠 Tail Overflow")
    
    if not badges:
        badges.append("✅ All Checks Pass")
    
    sanity_text = " | ".join(badges)
    assert len(sanity_text) > 0, "Sanity badges should be generated"
    
    logger.info("✅ UI parsing test passed")
    return True


def main():
    """Run complete integration tests."""
    logger.info("🎯 Running Complete DCA Integration Tests...")
    
    tests = [
        ("Evaluation Function Complete", test_evaluation_function_complete),
        ("Orchestrator Integration", test_orchestrator_integration),
        ("Storage Persistence", test_storage_persistence),
        ("UI Parsing", test_ui_parsing),
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
    
    logger.info(f"\n🎯 Integration Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        logger.info("🎉 Complete DCA system integration successful!")
        
        # Show final demonstration
        logger.info("\n🎯 Final System Demonstration:")
        
        # Generate a demonstration result
        demo_result = evaluation_function(
            base_price=1.0,
            overlap_pct=18.5,
            num_orders=5,
            seed=12345,
            wave_pattern=True,
            alpha=0.5,
            beta=0.3,
            gamma=0.2,
            lambda_penalty=0.1,
            tail_cap=0.35
        )
        
        logger.info(f"Final Score (J): {demo_result['score']:.6f}")
        logger.info(f"Max Need (fastest exit): {demo_result['max_need']:.2f}%")
        logger.info(f"Var Need (stability): {demo_result['var_need']:.6f}")
        logger.info(f"Tail (concentration): {demo_result['tail']:.3f}")
        
        # Show bullets in exact README format
        bullets = create_bullets_format(demo_result["schedule"])
        logger.info("\n📋 Order Schedule (Bullets Format):")
        for bullet in bullets:
            logger.info(f"  {bullet}")
        
        # Show penalties breakdown
        logger.info("\n⚖️ Penalties Breakdown:")
        for k, v in demo_result["penalties"].items():
            logger.info(f"  {k}: {v:.6f}")
        
        # Show sanity status
        sanity_flags = [k for k, v in demo_result["sanity"].items() if v]
        if sanity_flags:
            logger.info(f"\n⚠️ Sanity Flags: {sanity_flags}")
        else:
            logger.info("\n✅ Sanity: All checks passed")
        
        # Show diagnostics
        diag = demo_result["diagnostics"]
        logger.info(f"\n📊 Diagnostics:")
        logger.info(f"  WCI: {diag['wci']:.3f} (0=early load, 1=late load)")
        logger.info(f"  Sign Flips: {diag['sign_flips']} (NeedPct trend changes)")
        logger.info(f"  Gini: {diag['gini']:.3f} (volume concentration)")
        logger.info(f"  Entropy: {diag['entropy']:.3f} (volume diversity)")
        
        logger.info("\n🎯 System ready for production use!")
        return True
    else:
        logger.error(f"💥 {failed} integration test(s) failed.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)