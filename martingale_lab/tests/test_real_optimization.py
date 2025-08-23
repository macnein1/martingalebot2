"""
Real optimization test script for the Streamlit UI.
"""
import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from ui.utils.optimization_bridge import optimization_bridge

def test_real_optimization():
    """Test real optimization with actual parameters."""
    print("🚀 Starting Real Optimization Test...")
    
    # Real optimization parameters
    test_parameters = {
        'min_overlap': 5.0,
        'max_overlap': 25.0,
        'min_order': 3,
        'max_order': 15,
        'risk_factor': 1.2,
        'smoothing_factor': 0.15,
        'tail_weight': 0.25
    }
    
    print(f"📊 Parameters: {test_parameters}")
    
    # Validate parameters
    print("\n1️⃣ Validating parameters...")
    validation_result = optimization_bridge.validate_parameters(test_parameters)
    if not validation_result['success']:
        print(f"❌ Validation failed: {validation_result['error']}")
        return False
    print("✅ Parameters validated")
    
    # Create session
    print("\n2️⃣ Creating optimization session...")
    session_result = optimization_bridge.create_optimization_session(
        parameters=test_parameters,
        max_iterations=50,  # Reasonable number for real test
        time_limit=120.0   # 2 minutes max
    )
    
    if not session_result['success']:
        print(f"❌ Session creation failed: {session_result['error']}")
        return False
    
    session_id = session_result['session_id']
    print(f"✅ Session created: {session_id}")
    
    # Start optimization
    print("\n3️⃣ Starting optimization...")
    start_result = optimization_bridge.start_optimization()
    if not start_result['success']:
        print(f"❌ Start failed: {start_result['error']}")
        return False
    print("✅ Optimization started")
    
    # Monitor progress
    print("\n4️⃣ Monitoring progress...")
    max_wait_time = 130  # seconds
    start_time = time.time()
    
    while time.time() - start_time < max_wait_time:
        status_result = optimization_bridge.get_optimization_status()
        
        if status_result['success']:
            status_data = status_result['data']
            current_status = status_data.get('status', 'unknown')
            elapsed = status_data.get('elapsed_time', 0)
            
            print(f"⏱️  Status: {current_status}, Elapsed: {elapsed:.1f}s")
            
            if current_status == 'completed':
                print("🎉 Optimization completed!")
                break
            elif current_status == 'error':
                error_info = status_data.get('error', {})
                print(f"❌ Optimization failed: {error_info.get('error_message', 'Unknown error')}")
                return False
        
        time.sleep(2)  # Check every 2 seconds
    
    # Get results
    print("\n5️⃣ Retrieving results...")
    results_result = optimization_bridge.get_results()
    
    if results_result['success']:
        results = results_result['results']
        statistics = results_result['statistics']
        
        print("✅ Results retrieved successfully!")
        print(f"📈 Results count: {len(results['results'])}")
        print(f"⚡ Total evaluations: {statistics.get('total_evaluations', 0)}")
        print(f"⏱️  Total time: {statistics.get('total_time', 0):.2f}s")
        print(f"🚀 Evaluations/sec: {statistics.get('evaluations_per_second', 0):.1f}")
        
        if results['results']:
            best_result = results['results'][0]
            print(f"🏆 Best score: {best_result['score']:.4f}")
            print(f"🎯 Best params: {best_result['params']}")
        
        return True
    else:
        print(f"❌ Failed to get results: {results_result['error']}")
        return False

if __name__ == "__main__":
    success = test_real_optimization()
    if success:
        print("\n🎉 Real optimization test PASSED!")
    else:
        print("\n❌ Real optimization test FAILED!")
        sys.exit(1)
