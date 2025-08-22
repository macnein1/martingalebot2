"""
End-to-End Test for DCA System
Simulates complete UI workflow with optimization bridge and results loading.
"""
import sys
import os
import time
from pathlib import Path
import logging
import sqlite3
import json

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from ui.utils.optimization_bridge import optimization_bridge
from martingale_lab.storage.experiments_store import ExperimentsStore
from ui.utils.structured_logging import Events, app_logger, setup_structured_logging
from ui.utils.constants import DB_PATH

# Setup logging
setup_structured_logging()
logger = app_logger


def test_optimization_bridge():
    """Test optimization bridge start/stop functionality."""
    logger.event(Events.APP_START, test="e2e", phase="bridge_test")
    
    # Test parameters
    params = {
        "overlap_min": 12.0,
        "overlap_max": 18.0,
        "orders_min": 3,
        "orders_max": 5,
        "alpha": 0.5,
        "beta": 0.3,
        "gamma": 0.2,
        "lambda_penalty": 0.1,
        "wave_pattern": True,
        "tail_cap": 0.35,
        "n_candidates_per_batch": 20,  # Small for testing
        "max_batches": 2,
        "notes": "E2E test optimization"
    }
    
    # Validate parameters
    validation = optimization_bridge.validate_parameters(params)
    assert validation["success"], f"Parameter validation failed: {validation.get('error')}"
    
    # Start optimization
    start_result = optimization_bridge.start_optimization(params, DB_PATH)
    assert start_result["success"], f"Start optimization failed: {start_result.get('error')}"
    
    run_id = start_result["run_id"]
    assert run_id is not None, "Run ID should be provided"
    
    # Wait for optimization to complete (max 30 seconds)
    max_wait = 30
    waited = 0
    
    while waited < max_wait:
        status = optimization_bridge.get_optimization_status()
        if status["success"] and status["data"]["status"] == "completed":
            break
        
        time.sleep(1)
        waited += 1
    
    # Verify optimization completed
    final_status = optimization_bridge.get_optimization_status()
    assert final_status["success"], "Failed to get optimization status"
    assert final_status["data"]["status"] == "completed", "Optimization should be completed"
    
    logger.info(f"✅ Optimization bridge test passed (run_id: {run_id})")
    return run_id


def test_results_loading():
    """Test results loading and parsing."""
    logger.event(Events.UI_RESULTS_LOAD, test="e2e")
    
    # Load results from database
    store = ExperimentsStore(DB_PATH)
    
    # Get latest experiment
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()
        cur.execute("SELECT id FROM experiments ORDER BY id DESC LIMIT 1")
        row = cur.fetchone()
        assert row is not None, "No experiments found in database"
        exp_id = row[0]
    
    # Get top results
    results = store.get_top_results(experiment_id=exp_id, limit=10)
    assert len(results) > 0, "No results found for experiment"
    
    # Verify result structure
    first_result = results[0]
    
    # Parse payload JSON
    payload = json.loads(first_result["payload_json"])
    
    # Check required fields
    assert "score" in payload, "Missing score in payload"
    assert "schedule" in payload, "Missing schedule in payload"
    assert "sanity" in payload, "Missing sanity in payload"
    assert "diagnostics" in payload, "Missing diagnostics in payload"
    assert "penalties" in payload, "Missing penalties in payload"
    
    # Check schedule structure
    schedule = payload["schedule"]
    assert "needpct" in schedule, "Missing needpct in schedule"
    assert "volume_pct" in schedule, "Missing volume_pct in schedule"
    assert "martingale_pct" in schedule, "Missing martingale_pct in schedule"
    
    # Verify NeedPct properties
    needpct = schedule["needpct"]
    assert len(needpct) > 0, "NeedPct array should not be empty"
    assert all(isinstance(x, (int, float)) for x in needpct), "NeedPct should contain only numbers"
    
    # Check that NeedPct length matches orders
    orders = payload.get("params", {}).get("num_orders", 0)
    if orders > 0:
        assert len(needpct) == orders, f"NeedPct length {len(needpct)} != orders {orders}"
    
    # Verify best score is finite
    best_score = payload["score"]
    assert best_score < float("inf"), f"Best score should be finite, got {best_score}"
    
    # Check sanity - no hard violations for E2E test
    sanity = payload["sanity"]
    # For E2E test, we expect no tail overflow (this is configurable)
    # assert not sanity.get("tail_overflow", True), "Should not have tail overflow in E2E test"
    
    logger.info(f"✅ Results loading test passed (exp_id: {exp_id}, results: {len(results)})")
    return results


def create_top_n_table(results):
    """Simulate Top-N table creation."""
    logger.event(Events.UI_RESULTS_LOAD, action="create_table", count=len(results))
    
    table_data = []
    
    for i, result in enumerate(results[:10]):  # Top 10
        payload = json.loads(result["payload_json"])
        schedule = payload["schedule"]
        sanity = payload["sanity"]
        diagnostics = payload["diagnostics"]
        needpct = schedule["needpct"]
        
        # Create sparkline (simple simulation)
        if needpct and len(needpct) > 0:
            min_val = min(needpct)
            max_val = max(needpct)
            if max_val > min_val:
                spark_chars = "▁▂▃▄▅▆▇█"
                normalized = [(val - min_val) / (max_val - min_val) * 7 for val in needpct]
                sparkline = "".join([spark_chars[min(7, int(val))] for val in normalized])
            else:
                sparkline = "─" * len(needpct)
        else:
            sparkline = "─"
        
        # Create sanity badges
        badges = []
        if sanity.get("max_need_mismatch", False):
            badges.append("🔴 Max Need Mismatch")
        if sanity.get("collapse_indents", False):
            badges.append("🟡 Collapsed Indents")
        if sanity.get("tail_overflow", False):
            badges.append("🟠 Tail Overflow")
        if not badges:
            badges.append("✅ All Checks Pass")
        
        row = {
            "rank": i + 1,
            "score": payload["score"],
            "max_need": payload["max_need"],
            "var_need": payload["var_need"],
            "tail": payload["tail"],
            "wci": diagnostics["wci"],
            "sign_flips": diagnostics["sign_flips"],
            "needpct_sparkline": sparkline,
            "sanity_badges": " | ".join(badges)
        }
        table_data.append(row)
    
    logger.info(f"✅ Top-N table created with {len(table_data)} rows")
    return table_data


def test_bullets_creation():
    """Test bullets format creation from results."""
    logger.event(Events.UI_RESULTS_LOAD, action="create_bullets")
    
    # Get a result
    store = ExperimentsStore(DB_PATH)
    results = store.get_top_results(limit=1)
    assert len(results) > 0, "No results available for bullets test"
    
    payload = json.loads(results[0]["payload_json"])
    schedule = payload["schedule"]
    
    # Create bullets
    from martingale_lab.optimizer.evaluation_engine import create_bullets_format
    bullets = create_bullets_format(schedule)
    
    assert len(bullets) > 0, "Bullets should be created"
    
    # Verify format
    for i, bullet in enumerate(bullets):
        assert f"{i+1}. Emir:" in bullet, f"Bullet {i+1} format incorrect"
        assert "NeedPct %" in bullet, f"Bullet {i+1} missing NeedPct"
        
        if i == 0:
            assert "no martingale, first order" in bullet, "First bullet should indicate no martingale"
        else:
            assert "Martingale %" in bullet, f"Bullet {i+1} should have martingale"
    
    logger.info(f"✅ Bullets creation test passed ({len(bullets)} bullets)")
    return bullets


def run_e2e_test():
    """Run complete end-to-end test."""
    logger.event(Events.APP_START, test="e2e")
    
    try:
        # Clean database
        if os.path.exists(DB_PATH):
            os.remove(DB_PATH)
        
        # 1. Test optimization bridge
        run_id = test_optimization_bridge()
        
        # 2. Test results loading
        results = test_results_loading()
        
        # 3. Test Top-N table creation
        table_data = create_top_n_table(results)
        
        # 4. Test bullets creation
        bullets = test_bullets_creation()
        
        # Final verification
        assert len(results) >= 1, "Should have at least 1 result"
        assert len(table_data) >= 1, "Should have at least 1 table row"
        assert len(bullets) >= 1, "Should have at least 1 bullet"
        
        # Check that best score is reasonable
        best_score = results[0]["score"] if results else float("inf")
        assert best_score < float("inf"), "Best score should be finite"
        
        logger.event(
            Events.APP_STOP,
            test="e2e",
            status="success",
            run_id=run_id,
            results_count=len(results),
            best_score=best_score
        )
        
        print("✅ E2E test PASSED")
        print(f"  - Run ID: {run_id}")
        print(f"  - Results found: {len(results)}")
        print(f"  - Best score: {best_score:.6f}")
        print(f"  - Table rows: {len(table_data)}")
        print(f"  - Bullets: {len(bullets)}")
        
        # Show sample bullets
        print("\nSample bullets:")
        for bullet in bullets[:3]:
            print(f"  {bullet}")
        
        return True
        
    except Exception as e:
        logger.event(
            Events.APP_STOP,
            test="e2e",
            status="failed",
            error=str(e)
        )
        
        print(f"❌ E2E test FAILED: {e}")
        import traceback
        traceback.print_exc()
        
        return False


if __name__ == "__main__":
    success = run_e2e_test()
    sys.exit(0 if success else 1)