"""
Smoke Test for Martingale Lab
Headless optimization verification with comprehensive logging checks
"""
import os
import sys
import sqlite3
import json
import time
import traceback
from typing import List, Dict, Any

# Add parent directories to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

from martingale_lab.orchestrator.adaptive_orchestrator import AdaptiveOrchestrator, OrchConfig
from martingale_lab.storage.experiments_store import ExperimentsStore
from martingale_lab.utils.structured_logging import (
    setup_structured_logging, get_structured_logger, EventNames, generate_run_id
)
from ui.utils.logging_buffer import get_live_trace, clear_logs
from ui.utils.constants import DB_PATH, Status


def setup_test_environment():
    """Setup test environment with logging"""
    # Setup structured logging
    logger = setup_structured_logging("mlab", level=10)  # DEBUG level
    
    # Clear existing logs
    clear_logs("mlab")
    
    return logger


def cleanup_database(db_path: str):
    """Clean up test database"""
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM results")
            cursor.execute("DELETE FROM experiments")
            conn.commit()
        print(f"✅ Database cleaned: {db_path}")
    except Exception as e:
        print(f"⚠️  Database cleanup failed: {e}")


def run_small_optimization(logger) -> Dict[str, Any]:
    """Run small optimization for smoke test"""
    print("🔥 Starting smoke test optimization...")
    
    # Generate run ID
    run_id = generate_run_id()
    
    # Create minimal configuration
    config = OrchConfig(
        run_id=run_id,
        overlap_min=1.0,
        overlap_max=3.0,
        orders_min=2,
        orders_max=3,
        alpha=0.5,
        beta=0.3,
        gamma=0.2,
        lambda_penalty=0.1,
        wave_pattern=True,
        tail_cap=0.40,
        min_indent_step=0.05,
        softmax_temp=1.0,
        batch_size=10,  # Small batch size
        max_batches=2,  # Only 2 batches
        patience=2,
        prune_factor=2.0,
        top_k=5,
        base_price=100.0
    )
    
    # Create orchestrator
    orchestrator = AdaptiveOrchestrator(config, DB_PATH)
    
    # Run optimization
    start_time = time.time()
    result = orchestrator.run_optimization()
    elapsed_time = time.time() - start_time
    
    print(f"✅ Optimization completed in {elapsed_time:.2f}s")
    print(f"   Run ID: {result['run_id']}")
    print(f"   Exp ID: {result['exp_id']}")
    print(f"   Best Score: {result['best_score']:.4f}")
    print(f"   Total Evals: {result['total_evals']}")
    
    return result


def verify_database_content(exp_id: int) -> bool:
    """Verify database contains expected content"""
    print("🔍 Verifying database content...")
    
    try:
        store = ExperimentsStore(DB_PATH)
        
        # Check experiment record
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            
            # Verify experiment exists
            cursor.execute("SELECT COUNT(*) FROM experiments WHERE id = ?", (exp_id,))
            exp_count = cursor.fetchone()[0]
            
            if exp_count != 1:
                print(f"❌ Expected 1 experiment, found {exp_count}")
                return False
            
            # Verify results exist
            cursor.execute("SELECT COUNT(*) FROM results WHERE experiment_id = ?", (exp_id,))
            results_count = cursor.fetchone()[0]
            
            if results_count < 1:
                print(f"❌ Expected ≥1 results, found {results_count}")
                return False
            
            print(f"✅ Database verification passed:")
            print(f"   Experiments: {exp_count}")
            print(f"   Results: {results_count}")
            
            return True
            
    except Exception as e:
        print(f"❌ Database verification failed: {e}")
        return False


def verify_log_events() -> bool:
    """Verify required log events are present"""
    print("📋 Verifying log events...")
    
    try:
        # Get all logs from the ring buffer
        from ui.utils.logging_buffer import ensure_ring_handler
        handler = ensure_ring_handler("mlab", use_json=True)
        logs = handler.tail_json(1000)
        
        if not logs:
            print("❌ No logs found")
            return False
        
        # Required events
        required_events = [
            EventNames.ORCH_START,
            EventNames.ORCH_BATCH,
            EventNames.EVAL_CALL,
            EventNames.EVAL_RETURN,
            EventNames.DB_UPSERT_RES,
            EventNames.ORCH_DONE
        ]
        
        found_events = set()
        for log in logs:
            event = log.get('event', '')
            if event in required_events:
                found_events.add(event)
        
        missing_events = set(required_events) - found_events
        
        if missing_events:
            print(f"❌ Missing required events: {missing_events}")
            return False
        
        print(f"✅ Log verification passed:")
        print(f"   Total logs: {len(logs)}")
        print(f"   Required events found: {len(found_events)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Log verification failed: {e}")
        return False


def verify_payload_structure(exp_id: int) -> bool:
    """Verify payload structure in results"""
    print("📊 Verifying payload structure...")
    
    try:
        store = ExperimentsStore(DB_PATH)
        results = store.get_top_results(exp_id, limit=1)
        
        if not results:
            print("❌ No results found")
            return False
        
        result = results[0]
        # Parse payload JSON
        payload_json = result.get('payload_json', '{}')
        try:
            payload = json.loads(payload_json)
        except:
            payload = {}
        
        # Required payload fields
        required_fields = ['score', 'max_need', 'var_need', 'tail', 'schedule', 'sanity', 'diagnostics', 'penalties']
        
        missing_fields = []
        for field in required_fields:
            if field not in payload:
                missing_fields.append(field)
        
        if missing_fields:
            print(f"❌ Missing payload fields: {missing_fields}")
            return False
        
        # Verify schedule structure
        schedule = payload.get('schedule', {})
        schedule_fields = ['indent_pct', 'volume_pct', 'martingale_pct', 'needpct', 'order_prices', 'price_step_pct']
        
        for field in schedule_fields:
            if field not in schedule:
                print(f"❌ Missing schedule field: {field}")
                return False
        
        # Verify needpct length matches orders
        needpct = schedule.get('needpct', [])
        orders_count = len(schedule.get('indent_pct', []))
        
        if len(needpct) != orders_count:
            print(f"❌ NeedPct length mismatch: {len(needpct)} != {orders_count}")
            return False
        
        # Verify all needpct values are non-negative
        if any(need < 0 for need in needpct):
            print(f"❌ Negative NeedPct values found: {needpct}")
            return False
        
        print(f"✅ Payload verification passed:")
        print(f"   Orders count: {orders_count}")
        print(f"   NeedPct values: {needpct}")
        print(f"   Score: {payload['score']:.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Payload verification failed: {e}")
        traceback.print_exc()
        return False


def print_recent_logs(count: int = 200):
    """Print recent logs for debugging"""
    print(f"\n📋 Last {count} log entries:")
    print("-" * 80)
    
    try:
        logs = get_live_trace("mlab", last_n=count)
        
        for log in logs[-count:]:
            event = log.get('event', 'UNKNOWN')
            msg = log.get('msg', '')
            ts = log.get('ts', 0)
            
            # Format timestamp
            try:
                import datetime
                dt = datetime.datetime.fromtimestamp(ts)
                time_str = dt.strftime("%H:%M:%S.%f")[:-3]
            except:
                time_str = "??:??:??.???"
            
            print(f"[{time_str}] {event:15} {msg}")
    
    except Exception as e:
        print(f"Failed to print logs: {e}")


def main():
    """Main smoke test function"""
    print("🔥 Martingale Lab Smoke Test")
    print("=" * 50)
    
    exit_code = 0
    
    try:
        # Setup test environment
        logger = setup_test_environment()
        print("✅ Test environment setup complete")
        
        # Clean database
        cleanup_database(DB_PATH)
        
        # Run small optimization
        result = run_small_optimization(logger)
        exp_id = result['exp_id']
        
        # Verify database content
        if not verify_database_content(exp_id):
            exit_code = 1
        
        # Verify log events
        if not verify_log_events():
            exit_code = 1
        
        # Verify payload structure
        if not verify_payload_structure(exp_id):
            exit_code = 1
        
        if exit_code == 0:
            print("\n🎉 SMOKE TEST PASSED")
            print("   All verifications successful")
        else:
            print("\n❌ SMOKE TEST FAILED")
            print("   One or more verifications failed")
            print_recent_logs(200)
    
    except Exception as e:
        print(f"\n💥 SMOKE TEST CRASHED: {e}")
        traceback.print_exc()
        print_recent_logs(200)
        exit_code = 1
    
    print("=" * 50)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()