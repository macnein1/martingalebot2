"""
Debug script for DCA evaluation engine
"""
import sys
from pathlib import Path
import logging

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from martingale_lab.optimizer.dca_evaluation_engine import evaluate_dca_candidate, validate_evaluation_result
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def debug_evaluation():
    """Debug the evaluation function."""
    logger.info("Debugging DCA evaluation...")
    
    try:
        result = evaluate_dca_candidate(
            base_price=1.0,
            overlap_pct=20.0,
            num_orders=5,
            seed=42
        )
        
        logger.info(f"Score: {result.get('score', 'MISSING')}")
        logger.info(f"Max Need: {result.get('max_need', 'MISSING')}")
        logger.info(f"Schedule keys: {list(result.get('schedule', {}).keys())}")
        logger.info(f"Sanity keys: {list(result.get('sanity', {}).keys())}")
        
        # Check validation
        is_valid, msg = validate_evaluation_result(result)
        logger.info(f"Validation: {is_valid}, Message: {msg}")
        
        if 'error' in result:
            logger.error(f"Error in result: {result['error']}")
        
    except Exception as e:
        logger.error(f"Exception during evaluation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_evaluation()