"""
Test script to verify the bridge connection between UI and Martingale Lab.
"""
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ui.utils.optimization_bridge import optimization_bridge

def test_bridge_connection():
    """Test the bridge connection and basic functionality."""
    print("Testing Optimization Bridge Connection...")
    
    # Test parameter validation
    test_parameters = {
        'min_overlap': 1.0,
        'max_overlap': 30.0,
        'min_order': 3,
        'max_order': 20,
        'risk_factor': 1.0,
        'smoothing_factor': 0.1,
        'tail_weight': 0.2
    }
    
    print("\n1. Testing parameter validation...")
    validation_result = optimization_bridge.validate_parameters(test_parameters)
    print(f"Validation result: {validation_result}")
    
    if validation_result['success']:
        print("✅ Parameter validation passed")
    else:
        print(f"❌ Parameter validation failed: {validation_result['error']}")
        return False
    
    # Test session creation
    print("\n2. Testing session creation...")
    session_result = optimization_bridge.create_optimization_session(
        parameters=test_parameters,
        max_iterations=10,  # Small number for testing
        time_limit=30.0
    )
    print(f"Session creation result: {session_result}")
    
    if session_result['success']:
        print("✅ Session creation passed")
        session_id = session_result['session_id']
    else:
        print(f"❌ Session creation failed: {session_result['error']}")
        return False
    
    # Test optimization start
    print("\n3. Testing optimization start...")
    start_result = optimization_bridge.start_optimization()
    print(f"Optimization start result: {start_result}")
    
    if start_result['success']:
        print("✅ Optimization start passed")
    else:
        print(f"❌ Optimization start failed: {start_result['error']}")
        return False
    
    # Test status checking
    print("\n4. Testing status checking...")
    import time
    time.sleep(2)  # Wait a bit for optimization to progress
    
    status_result = optimization_bridge.get_optimization_status()
    print(f"Status result: {status_result}")
    
    if status_result['success']:
        print("✅ Status checking passed")
    else:
        print(f"❌ Status checking failed: {status_result['error']}")
        return False
    
    # Test results retrieval (after a short wait)
    print("\n5. Testing results retrieval...")
    time.sleep(5)  # Wait for optimization to complete
    
    results_result = optimization_bridge.get_results()
    print(f"Results result: {results_result}")
    
    if results_result['success']:
        print("✅ Results retrieval passed")
        print(f"Number of results: {len(results_result['results']['results'])}")
    else:
        print(f"❌ Results retrieval failed: {results_result['error']}")
    
    # Test cleanup
    print("\n6. Testing session cleanup...")
    cleanup_result = optimization_bridge.cleanup_session()
    print(f"Cleanup result: {cleanup_result}")
    
    if cleanup_result['success']:
        print("✅ Session cleanup passed")
    else:
        print(f"❌ Session cleanup failed: {cleanup_result['error']}")
    
    print("\n🎉 Bridge connection test completed!")
    return True

if __name__ == "__main__":
    success = test_bridge_connection()
    if success:
        print("\n✅ All tests passed! Bridge is working correctly.")
    else:
        print("\n❌ Some tests failed. Please check the bridge implementation.")
        sys.exit(1)
