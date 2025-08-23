"""
End-to-End Test for Martingale Lab
UI simulation with optimization bridge and log flow verification
"""
import os
import sys
import time
import threading
import json
import traceback
from typing import Dict, Any, List, Optional

# Add parent directories to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

from martingale_lab.utils.structured_logging import (
    setup_structured_logging, get_structured_logger, EventNames, generate_run_id
)
from ui.utils.optimization_bridge import OptimizationBridge
from ui.utils.logging_buffer import get_live_trace, clear_logs
from ui.utils.constants import DB_PATH, Status
from martingale_lab.storage.experiments_store import ExperimentsStore
from pages.results import load_experiments_data, load_results_data


def setup_test_environment():
    """Setup test environment with logging"""
    # Setup structured logging
    logger = setup_structured_logging("mlab", level=10)  # DEBUG level
    
    # Clear existing logs
    clear_logs("mlab")
    
    return logger


def cleanup_database():
    """Clean up test database"""
    import sqlite3
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM results")
            cursor.execute("DELETE FROM experiments")
            conn.commit()
        print(f"✅ Database cleaned: {DB_PATH}")
    except Exception as e:
        print(f"⚠️  Database cleanup failed: {e}")


def create_test_parameters() -> Dict[str, Any]:
    """Create test optimization parameters"""
    return {
        'overlap_min': 5.0,
        'overlap_max': 15.0,
        'orders_min': 3,
        'orders_max': 5,
        'alpha': 0.5,
        'beta': 0.3,
        'gamma': 0.2,
        'lambda_penalty': 0.1,
        'wave_pattern': True,
        'tail_cap': 0.40,
        'min_indent_step': 0.05,
        'softmax_temp': 1.0,
        'batch_size': 20,
        'max_batches': 3,
        'patience': 3,
        'prune_factor': 2.0,
        'top_k': 10,
        'base_price': 100.0
    }


def test_optimization_bridge():
    """Test optimization bridge functionality"""
    print("🌉 Testing optimization bridge...")
    
    bridge = OptimizationBridge(DB_PATH)
    params = create_test_parameters()
    
    # Progress tracking
    progress_updates = []
    
    def progress_callback(progress_data: Dict[str, Any]):
        progress_updates.append(progress_data)
        print(f"   Progress: Batch {progress_data.get('batch_idx', 0)}, "
              f"Evals: {progress_data.get('eval_count', 0)}, "
              f"Best: {progress_data.get('best_score', float('inf')):.4f}")
    
    try:
        # Start optimization
        run_id = bridge.start_optimization(params, progress_callback)
        print(f"✅ Optimization started with run_id: {run_id}")
        
        # Check status
        status = bridge.get_status()
        if not status['is_running']:
            print("❌ Optimization should be running")
            return False
        
        # Wait for completion (with timeout)
        timeout = 60  # 60 seconds timeout
        start_time = time.time()
        
        while bridge.get_status()['is_running']:
            if time.time() - start_time > timeout:
                print("❌ Optimization timed out")
                bridge.stop_optimization()
                return False
            time.sleep(1)
        
        print(f"✅ Optimization completed")
        print(f"   Progress updates received: {len(progress_updates)}")
        
        if not progress_updates:
            print("❌ No progress updates received")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Bridge test failed: {e}")
        traceback.print_exc()
        return False


def verify_log_flow() -> bool:
    """Verify complete log flow from start to finish"""
    print("📋 Verifying log flow...")
    
    try:
        # Debug: Check if ring buffer handler is attached
        import logging
        from ui.utils.logging_buffer import _handlers_registry
        
        logger = logging.getLogger("mlab")
        print(f"   Logger handlers: {len(logger.handlers)}")
        for i, handler in enumerate(logger.handlers):
            print(f"   Handler {i}: {type(handler).__name__}")
        
        print(f"   Ring buffer registry: {list(_handlers_registry.keys())}")
        if "mlab" in _handlers_registry:
            ring_handler = _handlers_registry["mlab"]
            print(f"   Ring buffer size: {len(ring_handler._buffer)}")
            print(f"   Ring buffer JSON size: {len(ring_handler._json_buffer)}")
        
        # Get all logs
        logs = get_live_trace("mlab", last_n=2000)
        
        print(f"   get_live_trace returned: {type(logs)} with {len(logs) if logs else 0} items")
        if logs and len(logs) > 0:
            print(f"   First log item: {logs[0]}")
        
        if not logs:
            print("❌ No logs found")
            return False
        
        # Required log sequence
        required_sequence = [
            EventNames.UI_CLICK_START,
            EventNames.ORCH_START,
            EventNames.BUILD_CONFIG,
            EventNames.ORCH_BATCH,
            EventNames.EVAL_CALL,
            EventNames.EVAL_RETURN,
            EventNames.DB_UPSERT_RES,
            EventNames.ORCH_SAVE_OK,
            EventNames.ORCH_DONE
        ]
        
        # Find events in sequence
        event_sequence = []
        for log in logs:
            event = log.get('event', '')
            if event in required_sequence:
                event_sequence.append(event)
        
        # Check if we have the minimum required sequence
        found_events = set(event_sequence)
        missing_events = set(required_sequence) - found_events
        
        if missing_events:
            print(f"❌ Missing events in log flow: {missing_events}")
            return False
        
        # Count specific events
        eval_calls = event_sequence.count(EventNames.EVAL_CALL)
        eval_returns = event_sequence.count(EventNames.EVAL_RETURN)
        db_upserts = event_sequence.count(EventNames.DB_UPSERT_RES)
        
        print(f"✅ Log flow verification passed:")
        print(f"   Total logs: {len(logs)}")
        print(f"   Eval calls: {eval_calls}")
        print(f"   Eval returns: {eval_returns}")
        print(f"   DB upserts: {db_upserts}")
        
        # Basic sanity checks
        if eval_calls != eval_returns:
            print(f"⚠️  Warning: Eval calls ({eval_calls}) != returns ({eval_returns})")
        
        if db_upserts == 0:
            print("❌ No database upserts found")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Log flow verification failed: {e}")
        return False


def test_ui_data_loading():
    """Test UI data loading functions"""
    print("📊 Testing UI data loading...")
    
    try:
        # Load experiments
        experiments = load_experiments_data(DB_PATH)
        
        if not experiments:
            print("❌ No experiments loaded")
            return False
        
        print(f"✅ Loaded {len(experiments)} experiments")
        
        # Get first experiment
        exp = experiments[0]
        exp_id = exp['id']
        
        # Load results for experiment
        results = load_results_data(DB_PATH, exp_id, limit=10)
        
        if not results:
            print("❌ No results loaded")
            return False
        
        print(f"✅ Loaded {len(results)} results for experiment {exp_id}")
        
        # Verify result structure
        result = results[0]
        payload = result.get('payload', {})
        
        required_fields = ['score', 'max_need', 'var_need', 'tail', 'schedule']
        missing_fields = [field for field in required_fields if field not in payload]
        
        if missing_fields:
            print(f"❌ Missing fields in result payload: {missing_fields}")
            return False
        
        # Verify schedule structure
        schedule = payload.get('schedule', {})
        needpct = schedule.get('needpct', [])
        
        if not needpct:
            print("❌ No NeedPct data in schedule")
            return False
        
        print(f"✅ UI data loading verification passed:")
        print(f"   Best score: {result['score']:.4f}")
        print(f"   NeedPct length: {len(needpct)}")
        print(f"   Max need: {payload['max_need']:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ UI data loading test failed: {e}")
        traceback.print_exc()
        return False


def verify_data_consistency():
    """Verify data consistency across system components"""
    print("🔍 Verifying data consistency...")
    
    try:
        # Load from storage
        store = ExperimentsStore(DB_PATH)
        
        # Get latest experiment
        import sqlite3
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id FROM experiments ORDER BY created_at DESC LIMIT 1")
            row = cursor.fetchone()
            if not row:
                print("❌ No experiments found")
                return False
            exp_id = row[0]
        
        # Get results from storage
        storage_results = store.get_top_results(exp_id, limit=5)
        
        # Get results from UI loader
        ui_results = load_results_data(DB_PATH, exp_id, limit=5)
        
        if len(storage_results) != len(ui_results):
            print(f"❌ Result count mismatch: storage={len(storage_results)}, ui={len(ui_results)}")
            return False
        
        # Compare first result
        if storage_results and ui_results:
            storage_score = storage_results[0]['score']
            ui_score = ui_results[0]['score']
            
            if abs(storage_score - ui_score) > 1e-6:
                print(f"❌ Score mismatch: storage={storage_score}, ui={ui_score}")
                return False
        
        print(f"✅ Data consistency verification passed:")
        print(f"   Results count: {len(storage_results)}")
        print(f"   Scores match: {storage_results[0]['score']:.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Data consistency verification failed: {e}")
        return False


def test_error_handling():
    """Test error handling and recovery"""
    print("⚠️  Testing error handling...")
    
    try:
        bridge = OptimizationBridge(DB_PATH)
        
        # Try to start optimization when one is already running
        params = create_test_parameters()
        params['batch_size'] = 5  # Small batch for quick test
        params['max_batches'] = 1
        
        run_id1 = bridge.start_optimization(params)
        
        try:
            run_id2 = bridge.start_optimization(params)
            print("❌ Should have failed to start second optimization")
            return False
        except RuntimeError:
            print("✅ Correctly prevented second optimization start")
        
        # Wait for completion
        timeout = 30
        start_time = time.time()
        while bridge.get_status()['is_running']:
            if time.time() - start_time > timeout:
                bridge.stop_optimization()
                break
            time.sleep(0.5)
        
        return True
        
    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
        return False


def print_test_summary(logs: List[Dict[str, Any]]):
    """Print test summary with key metrics"""
    print("\n📈 Test Summary")
    print("-" * 50)
    
    # Count events by type
    event_counts = {}
    for log in logs:
        event = log.get('event', 'UNKNOWN')
        event_counts[event] = event_counts.get(event, 0) + 1
    
    print("Event Counts:")
    for event, count in sorted(event_counts.items()):
        print(f"  {event:20} {count:4d}")
    
    # Find timing information
    start_time = None
    end_time = None
    
    for log in logs:
        if log.get('event') == EventNames.ORCH_START:
            start_time = log.get('ts', 0)
        elif log.get('event') == EventNames.ORCH_DONE:
            end_time = log.get('ts', 0)
    
    if start_time and end_time:
        duration = end_time - start_time
        print(f"\nTiming:")
        print(f"  Total duration: {duration:.2f}s")
        
        eval_count = event_counts.get(EventNames.EVAL_CALL, 0)
        if eval_count > 0:
            print(f"  Evals per second: {eval_count / duration:.1f}")


def main():
    """Main E2E test function"""
    print("🎯 Martingale Lab E2E Test")
    print("=" * 50)
    
    exit_code = 0
    
    try:
        # Setup test environment
        logger = setup_test_environment()
        print("✅ Test environment setup complete")
        
        # Clean database
        cleanup_database()
        
        # Test optimization bridge
        if not test_optimization_bridge():
            exit_code = 1
        
        # Verify log flow
        if not verify_log_flow():
            exit_code = 1
        
        # Test UI data loading
        if not test_ui_data_loading():
            exit_code = 1
        
        # Verify data consistency
        if not verify_data_consistency():
            exit_code = 1
        
        # Test error handling
        if not test_error_handling():
            exit_code = 1
        
        # Get final logs for summary
        logs = get_live_trace("mlab", last_n=2000)
        
        if exit_code == 0:
            print("\n🎉 E2E TEST PASSED")
            print("   All components working correctly")
        else:
            print("\n❌ E2E TEST FAILED")
            print("   One or more tests failed")
        
        print_test_summary(logs)
        
    except Exception as e:
        print(f"\n💥 E2E TEST CRASHED: {e}")
        traceback.print_exc()
        exit_code = 1
    
    print("=" * 50)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
