"""
Test for updated evaluation_engine.py compliance with README specification
"""
import sys
from pathlib import Path
import logging
import json

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from martingale_lab.optimizer.evaluation_engine import evaluation_function, create_bullets_format
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_evaluation_function_contract():
    """Test that evaluation_function matches README specification exactly."""
    logger.info("Testing evaluation_function contract compliance...")
    
    # Test with all required parameters
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
    
    # Check all required top-level keys
    required_keys = ["score", "max_need", "var_need", "tail", "schedule", "sanity", "diagnostics", "penalties"]
    for key in required_keys:
        assert key in result, f"Missing required key: {key}"
    
    # Check schedule structure
    schedule = result["schedule"]
    schedule_keys = ["indent_pct", "volume_pct", "martingale_pct", "needpct", "order_prices", "price_step_pct"]
    for key in schedule_keys:
        assert key in schedule, f"Missing schedule key: {key}"
    
    # Check sanity structure
    sanity = result["sanity"]
    sanity_keys = ["max_need_mismatch", "collapse_indents", "tail_overflow"]
    for key in sanity_keys:
        assert key in sanity, f"Missing sanity key: {key}"
        assert isinstance(sanity[key], bool), f"Sanity {key} should be boolean, got {type(sanity[key])}"
    
    # Check diagnostics structure
    diagnostics = result["diagnostics"]
    diagnostics_keys = ["wci", "sign_flips", "gini", "entropy"]
    for key in diagnostics_keys:
        assert key in diagnostics, f"Missing diagnostics key: {key}"
    
    # Check penalties structure (all must be present)
    penalties = result["penalties"]
    penalty_keys = ["P_gini", "P_entropy", "P_monotone", "P_smooth", "P_tailcap", "P_need_mismatch", "P_wave"]
    for key in penalty_keys:
        assert key in penalties, f"Missing penalty key: {key}"
        assert isinstance(penalties[key], (int, float)), f"Penalty {key} should be numeric, got {type(penalties[key])}"
    
    # Check JSON serializability
    try:
        json_str = json.dumps(result)
        parsed_back = json.loads(json_str)
        assert parsed_back is not None
    except Exception as e:
        assert False, f"Result not JSON serializable: {e}"
    
    # Check scoring formula: J = α·max_need + β·var_need + γ·tail + λ·Σ(penalties)
    expected_score = (0.5 * result["max_need"] + 
                     0.3 * result["var_need"] + 
                     0.2 * result["tail"] + 
                     0.1 * sum(penalties.values()))
    
    actual_score = result["score"]
    assert abs(actual_score - expected_score) < 1e-6, f"Score formula mismatch: {actual_score} != {expected_score}"
    
    # Check dimensions consistency
    n_orders = 5
    assert len(schedule["volume_pct"]) == n_orders
    assert len(schedule["martingale_pct"]) == n_orders
    assert len(schedule["needpct"]) == n_orders
    assert len(schedule["indent_pct"]) == n_orders
    assert len(schedule["order_prices"]) == n_orders + 1  # Includes base price
    
    # Check volume sum constraint
    volume_sum = sum(schedule["volume_pct"])
    assert abs(volume_sum - 100.0) < 1e-6, f"Volume sum should be 100, got {volume_sum}"
    
    # Check martingale first element
    assert schedule["martingale_pct"][0] == 0.0, "First martingale should be 0"
    
    logger.info("✅ Evaluation function contract test passed")
    return True


def test_needpct_calculation():
    """Test that NeedPct uses exact formula from README."""
    logger.info("Testing NeedPct calculation formula...")
    
    result = evaluation_function(
        base_price=1.0,
        overlap_pct=15.0,
        num_orders=3,
        seed=123
    )
    
    schedule = result["schedule"]
    volumes = schedule["volume_pct"]
    prices = schedule["order_prices"]
    needpct = schedule["needpct"]
    
    # Manually calculate NeedPct using README formula
    expected_needpct = []
    vol_acc = 0.0
    val_acc = 0.0
    
    for k in range(len(volumes)):
        vol_acc += volumes[k]
        val_acc += volumes[k] * prices[k+1]  # prices[k+1] is k-th order price
        
        avg_entry_price = val_acc / vol_acc
        current_price = prices[k+1]
        need_k = (avg_entry_price / current_price - 1.0) * 100.0
        expected_needpct.append(need_k)
    
    # Compare with calculated values
    for i, (actual, expected) in enumerate(zip(needpct, expected_needpct)):
        assert abs(actual - expected) < 1e-6, f"NeedPct mismatch at order {i+1}: {actual} != {expected}"
    
    logger.info("✅ NeedPct calculation test passed")
    return True


def test_wave_pattern_logic():
    """Test wave pattern logic consistency with README."""
    logger.info("Testing wave pattern logic...")
    
    # Test with wave pattern enabled
    result = evaluation_function(
        base_price=1.0,
        overlap_pct=25.0,
        num_orders=6,
        seed=456,
        wave_pattern=True,
        wave_strong_threshold=50.0,
        wave_weak_threshold=10.0
    )
    
    penalties = result["penalties"]
    assert "P_wave" in penalties, "P_wave penalty must be present when wave_pattern=True"
    
    # Test with wave pattern disabled
    result_no_wave = evaluation_function(
        base_price=1.0,
        overlap_pct=25.0,
        num_orders=6,
        seed=456,
        wave_pattern=False
    )
    
    penalties_no_wave = result_no_wave["penalties"]
    assert "P_wave" in penalties_no_wave, "P_wave penalty must always be present"
    assert penalties_no_wave["P_wave"] == 0.0, "P_wave should be 0 when wave_pattern=False"
    
    logger.info("✅ Wave pattern logic test passed")
    return True


def test_bullets_format():
    """Test bullets format matches README specification."""
    logger.info("Testing bullets format...")
    
    result = evaluation_function(
        base_price=1.0,
        overlap_pct=18.0,
        num_orders=4,
        seed=789
    )
    
    bullets = create_bullets_format(result["schedule"])
    
    # Check format compliance
    assert len(bullets) == 4, f"Should have 4 bullets, got {len(bullets)}"
    
    # Check first bullet (no martingale)
    first_bullet = bullets[0]
    assert "1. Emir:" in first_bullet
    assert "no martingale, first order" in first_bullet
    assert "NeedPct %" in first_bullet
    
    # Check second bullet (has martingale)
    if len(bullets) > 1:
        second_bullet = bullets[1]
        assert "2. Emir:" in second_bullet
        assert "Martingale %" in second_bullet
        assert "NeedPct %" in second_bullet
    
    logger.info("✅ Bullets format test passed")
    for bullet in bullets:
        logger.info(f"  {bullet}")
    
    return True


def test_error_handling():
    """Test that evaluation_function never throws."""
    logger.info("Testing error handling...")
    
    # Test with invalid parameters
    test_cases = [
        {"base_price": -1.0, "overlap_pct": 20.0, "num_orders": 5},
        {"base_price": 1.0, "overlap_pct": -10.0, "num_orders": 5},
        {"base_price": 1.0, "overlap_pct": 20.0, "num_orders": 0},
        {"base_price": 1.0, "overlap_pct": 20.0, "num_orders": 1000},
    ]
    
    for i, params in enumerate(test_cases):
        try:
            result = evaluation_function(**params)
            
            # Should always return a complete dict
            assert isinstance(result, dict), f"Result should be dict, got {type(result)}"
            assert "score" in result, "Score should always be present"
            assert "schedule" in result, "Schedule should always be present"
            assert "sanity" in result, "Sanity should always be present"
            assert "diagnostics" in result, "Diagnostics should always be present"
            assert "penalties" in result, "Penalties should always be present"
            
            logger.info(f"Test case {i+1}: Score = {result['score']}")
            
        except Exception as e:
            assert False, f"evaluation_function threw exception for test case {i+1}: {e}"
    
    logger.info("✅ Error handling test passed")
    return True


def test_json_serialization():
    """Test complete JSON serializability."""
    logger.info("Testing JSON serialization...")
    
    result = evaluation_function(
        base_price=1.0,
        overlap_pct=22.0,
        num_orders=6,
        seed=999,
        wave_pattern=True
    )
    
    # Test full serialization
    try:
        json_str = json.dumps(result, indent=2)
        parsed = json.loads(json_str)
        
        # Verify structure is preserved
        assert parsed["score"] == result["score"]
        assert len(parsed["schedule"]["volume_pct"]) == len(result["schedule"]["volume_pct"])
        assert parsed["sanity"]["max_need_mismatch"] == result["sanity"]["max_need_mismatch"]
        
        logger.info(f"JSON size: {len(json_str)} characters")
        
    except Exception as e:
        assert False, f"JSON serialization failed: {e}"
    
    logger.info("✅ JSON serialization test passed")
    return True


def main():
    """Run all evaluation engine tests."""
    logger.info("🎯 Testing Evaluation Engine README Compliance...")
    
    tests = [
        ("Contract Compliance", test_evaluation_function_contract),
        ("NeedPct Formula", test_needpct_calculation),
        ("Wave Pattern Logic", test_wave_pattern_logic),
        ("Bullets Format", test_bullets_format),
        ("Error Handling", test_error_handling),
        ("JSON Serialization", test_json_serialization),
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
    
    logger.info(f"\n🎯 Evaluation Engine Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        logger.info("🎉 Evaluation engine fully compliant with README!")
        
        # Show example output
        logger.info("\n📋 Example Evaluation Output:")
        result = evaluation_function(
            base_price=1.0,
            overlap_pct=20.0,
            num_orders=5,
            seed=12345,
            wave_pattern=True,
            alpha=0.5,
            beta=0.3,
            gamma=0.2,
            lambda_penalty=0.1
        )
        
        logger.info(f"Score (J): {result['score']:.6f}")
        logger.info(f"Max Need: {result['max_need']:.2f}%")
        logger.info(f"Var Need: {result['var_need']:.6f}")
        logger.info(f"Tail: {result['tail']:.3f}")
        
        # Show all penalties
        logger.info("\nPenalties:")
        for k, v in result["penalties"].items():
            logger.info(f"  {k}: {v:.6f}")
        
        # Show bullets
        bullets = create_bullets_format(result["schedule"])
        logger.info("\nBullets:")
        for bullet in bullets:
            logger.info(f"  {bullet}")
        
        # Show sanity
        logger.info(f"\nSanity checks:")
        for k, v in result["sanity"].items():
            logger.info(f"  {k}: {v}")
        
        return True
    else:
        logger.error(f"💥 {failed} test(s) failed.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
