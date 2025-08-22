"""
Smoke Test for DCA System
Headless test that verifies end-to-end functionality with structured logging.
"""
import sys
import os
from pathlib import Path
import logging
import sqlite3
import json

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from martingale_lab.orchestrator.dca_orchestrator import DCAOrchestrator, DCAConfig
from martingale_lab.storage.experiments_store import ExperimentsStore
from ui.utils.structured_logging import Events, app_logger, setup_structured_logging, LogContext
from ui.utils.constants import DB_PATH, Status

# Setup logging
setup_structured_logging()
logger = app_logger


def clean_database():
    """Clean database for fresh test."""
    logger.event(Events.APP_START, test="smoke", action="clean_db")
    
    if os.path.exists(DB_PATH):
        with sqlite3.connect(DB_PATH) as conn:
            cur = conn.cursor()
            cur.execute("DELETE FROM results")
            cur.execute("DELETE FROM experiments")
            conn.commit()
    
    logger.info("Database cleaned for smoke test")


def run_smoke_test():
    """Run smoke test with small parameters."""
    logger.event(Events.APP_START, test="smoke")
    
    try:
        # Clean database
        clean_database()
        
        # Create small test configuration
        config = DCAConfig(
            overlap_min=10.0,
            overlap_max=15.0,
            orders_min=3,
            orders_max=4,
            n_candidates_per_batch=25,  # Small batch
            max_batches=2,
            alpha=0.5,
            beta=0.3,
            gamma=0.2,
            lambda_penalty=0.1,
            wave_pattern=True,
            tail_cap=0.40,
            random_seed=42
        )
        
        logger.event(
            Events.BUILD_CONFIG,
            overlap_range=f"{config.overlap_min}-{config.overlap_max}",
            orders_range=f"{config.orders_min}-{config.orders_max}",
            n_candidates=config.n_candidates_per_batch,
            max_batches=config.max_batches
        )
        
        # Create orchestrator
        store = ExperimentsStore(DB_PATH)
        orchestrator = DCAOrchestrator(config, store)
        
        # Run optimization
        results = orchestrator.run_optimization(notes="Smoke test run")
        
        # Verify results
        assert "experiment_id" in results, "Missing experiment_id in results"
        assert "best_candidates" in results, "Missing best_candidates in results"
        assert "statistics" in results, "Missing statistics in results"
        
        exp_id = results["experiment_id"]
        stats = results["statistics"]
        
        # Check database state
        with sqlite3.connect(DB_PATH) as conn:
            cur = conn.cursor()
            
            # Check experiments table
            cur.execute("SELECT COUNT(*) FROM experiments WHERE id = ?", (exp_id,))
            exp_count = cur.fetchone()[0]
            assert exp_count >= 1, f"Expected >=1 experiment record, got {exp_count}"
            
            # Check results table
            cur.execute("SELECT COUNT(*) FROM results WHERE experiment_id = ?", (exp_id,))
            result_count = cur.fetchone()[0]
            assert result_count >= 1, f"Expected >=1 result record, got {result_count}"
            
            # Check that we have enough evaluations
            total_evals = stats["total_evaluations"]
            assert total_evals >= 50, f"Expected >=50 evaluations, got {total_evals}"
            
            # Verify payload structure
            cur.execute("SELECT payload_json FROM results WHERE experiment_id = ? LIMIT 1", (exp_id,))
            payload_row = cur.fetchone()
            if payload_row:
                payload = json.loads(payload_row[0])
                assert "schedule" in payload, "Missing schedule in payload"
                assert "needpct" in payload["schedule"], "Missing needpct in schedule"
                
                needpct = payload["schedule"]["needpct"]
                orders = payload.get("params", {}).get("num_orders", 0)
                if orders > 0:
                    assert len(needpct) == orders, f"NeedPct length {len(needpct)} != orders {orders}"
        
        # Check for ORCH.DONE log (would be in the logs if we were capturing them)
        logger.event(Events.APP_STOP, test="smoke", status="success", total_evals=total_evals)
        
        print(f"✅ Smoke test PASSED")
        print(f"  - Experiment ID: {exp_id}")
        print(f"  - Total evaluations: {total_evals}")
        print(f"  - Results in DB: {result_count}")
        print(f"  - Best score: {stats['best_score']:.6f}")
        
        return True
        
    except Exception as e:
        logger.event(Events.APP_STOP, test="smoke", status="failed", error=str(e))
        
        print(f"❌ Smoke test FAILED: {e}")
        
        # Print last 200 logs (simulated - in real implementation would use ring buffer)
        import traceback
        print("\nError traceback:")
        traceback.print_exc()
        
        return False


if __name__ == "__main__":
    success = run_smoke_test()
    sys.exit(0 if success else 1)