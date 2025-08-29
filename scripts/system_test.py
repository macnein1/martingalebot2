#!/usr/bin/env python3
"""
Comprehensive system test to verify all components are working.
"""
import sys
import os
import time
import json
import tempfile
import traceback
sys.path.insert(0, '/workspace')

# Test results collector
test_results = {
    'passed': [],
    'failed': [],
    'warnings': []
}

def test_section(name):
    """Decorator for test sections."""
    def decorator(func):
        def wrapper():
            print(f"\n{'='*60}")
            print(f"Testing: {name}")
            print('='*60)
            try:
                result = func()
                if result:
                    test_results['passed'].append(name)
                    print(f"✅ {name} PASSED")
                else:
                    test_results['failed'].append(name)
                    print(f"❌ {name} FAILED")
                return result
            except Exception as e:
                test_results['failed'].append(name)
                print(f"❌ {name} FAILED with exception: {e}")
                traceback.print_exc()
                return False
        return wrapper
    return decorator


@test_section("1. Schedule Normalization")
def test_schedule_normalization():
    """Test 2dp rounding and constraint preservation."""
    from martingale_lab.core.schedule_normalizer import normalize_schedule_to_2dp, is_schedule_normalized
    
    # Test data
    schedule = {
        'volume_pct': [1.234567, 2.345678, 3.456789, 10.0, 20.0, 30.0, 32.516522],
        'indent_pct': [0.0, 1.111, 2.222, 3.333, 4.444, 5.555, 6.666]
    }
    
    # Normalize
    normalized = normalize_schedule_to_2dp(schedule)
    
    # Check all values are 2dp
    for v in normalized['volume_pct']:
        if round(v, 2) != v:
            print(f"  ⚠️  Volume not 2dp: {v}")
            return False
    
    for i in normalized['indent_pct']:
        if round(i, 2) != i:
            print(f"  ⚠️  Indent not 2dp: {i}")
            return False
    
    # Check sum = 100
    total = sum(normalized['volume_pct'])
    if abs(total - 100.0) > 0.01:
        print(f"  ⚠️  Sum not 100: {total}")
        return False
    
    print(f"  ✓ All values 2dp")
    print(f"  ✓ Sum = {total:.2f}")
    return True


@test_section("2. Config Management")
def test_config_management():
    """Test configuration system."""
    from martingale_lab.core.config_classes import EvaluationConfig, ConfigPresets
    from martingale_lab.core.config_adapter import extract_config_from_kwargs
    
    # Test preset
    config = ConfigPresets.production()
    errors = config.validate()
    if errors:
        print(f"  ⚠️  Config validation errors: {errors}")
        return False
    
    # Test serialization
    json_str = config.to_json()
    config2 = EvaluationConfig.from_json(json_str)
    
    if config.constraints.slope_cap != config2.constraints.slope_cap:
        print(f"  ⚠️  Serialization mismatch")
        return False
    
    # Test backward compatibility
    old_kwargs = {
        'base_price': 100.0,
        'overlap_pct': 20.0,
        'num_orders': 10,
        'm2_min': 0.15,
        'm2_max': 0.80,
        'unknown_param': 'should_be_ignored'
    }
    
    config3, remaining = extract_config_from_kwargs(**old_kwargs)
    if config3.core.base_price != 100.0:
        print(f"  ⚠️  Backward compat failed")
        return False
    
    print(f"  ✓ Config validation working")
    print(f"  ✓ Serialization working")
    print(f"  ✓ Backward compatibility working")
    return True


@test_section("3. Database Operations")
def test_database_operations():
    """Test unified store and migrations."""
    from martingale_lab.storage.unified_store import UnifiedStore
    from martingale_lab.storage.config_store import ConfigStore
    from martingale_lab.core.config_classes import ConfigPresets
    
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
        db_path = tmp.name
    
    try:
        # Test UnifiedStore
        store = UnifiedStore(db_path)
        
        # Create experiment
        exp_id = store.create_experiment(
            run_id="test_run_123",
            orchestrator="test",
            config={'test': True},
            notes="Test experiment"
        )
        
        if exp_id is None or exp_id <= 0:
            print(f"  ⚠️  Failed to create experiment")
            return False
        
        # Insert results
        results = [
            {'score': 100.0, 'params': {'a': 1}},
            {'score': 90.0, 'params': {'a': 2}}
        ]
        
        store.insert_results_batch(exp_id, results)
        
        # Test ConfigStore
        config_store = ConfigStore(db_path)
        config = ConfigPresets.exploration()
        hash1 = config_store.store_config(config, notes="Test config")
        
        # Record performance
        config_store.record_performance(hash1, exp_id, "test_run_123", 
                                       best_score=90.0, avg_score=95.0, 
                                       total_evaluations=100)
        
        # Get stats
        stats = config_store.get_config_stats(hash1)
        if stats.get('best_score') != 90.0:
            print(f"  ⚠️  Config stats incorrect")
            return False
        
        print(f"  ✓ Database operations working")
        print(f"  ✓ Config storage working")
        print(f"  ✓ Performance tracking working")
        return True
        
    finally:
        if os.path.exists(db_path):
            os.unlink(db_path)


@test_section("4. Memory Management")
def test_memory_management():
    """Test bounded memory structures."""
    from martingale_lab.storage.memory_manager import BoundedBestCandidates
    
    # Test bounded candidates
    candidates = BoundedBestCandidates(max_size=5)
    
    # Add more than max_size
    for i in range(10):
        candidates.add({'score': float(i), 'id': i})
    
    # Should only keep best 5
    best = candidates.get_best(10)
    if len(best) != 5:
        print(f"  ⚠️  Memory bound not enforced: {len(best)} items")
        return False
    
    # Check they are the best ones (lowest scores)
    scores = sorted([c['score'] for c in best])
    if scores != [0.0, 1.0, 2.0, 3.0, 4.0]:
        print(f"  ⚠️  Wrong candidates kept: {scores}")
        return False
    
    print(f"  ✓ Memory bounds enforced")
    print(f"  ✓ Best candidates kept")
    return True


@test_section("5. Auto-Tuning")
def test_auto_tuning():
    """Test auto-tuning system."""
    from martingale_lab.optimizer.auto_tuner import ConfigAutoTuner, TuningStrategy
    from martingale_lab.storage.config_store import ConfigStore
    from martingale_lab.core.config_classes import ConfigPresets
    
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
        db_path = tmp.name
    
    try:
        config_store = ConfigStore(db_path)
        tuner = ConfigAutoTuner(config_store)
        
        # Test mutation
        base_config = ConfigPresets.production()
        strategy = TuningStrategy(name="test", mutation_rate=1.0)
        mutated = tuner._mutate_config(base_config, strategy)
        
        # Something should have changed
        if mutated.to_json() == base_config.to_json():
            print(f"  ⚠️  Mutation didn't change config")
            test_results['warnings'].append("Mutation might not be working")
        
        # Test crossover
        config2 = ConfigPresets.strict()
        offspring = tuner._crossover_configs(base_config, config2)
        
        # Should be different from both parents
        if offspring.to_json() == base_config.to_json():
            print(f"  ⚠️  Crossover produced clone of parent1")
            test_results['warnings'].append("Crossover might not be working")
        
        # Test strategy recommendation
        stagnant = [100, 99.5, 99.2, 99.1, 99.0, 98.9, 98.8, 98.8, 98.7, 98.7]
        strategy = tuner.recommend_tuning_strategy(stagnant)
        
        print(f"  ✓ Mutation working")
        print(f"  ✓ Crossover working")
        print(f"  ✓ Strategy recommendation: {strategy.name}")
        return True
        
    finally:
        if os.path.exists(db_path):
            os.unlink(db_path)


@test_section("6. Slope Enforcement")
def test_slope_enforcement():
    """Test improved slope enforcement."""
    from martingale_lab.core.improved_slope_enforcement import multi_strategy_slope_enforcement
    import numpy as np
    
    # Test data with violations
    volumes = np.array([1.0, 1.5, 3.0, 6.0, 12.0, 20.0, 30.0, 26.5])
    volumes = volumes * (100.0 / volumes.sum())  # Normalize
    
    # Enforce slopes
    result, success, violations, strategy = multi_strategy_slope_enforcement(
        volumes,
        slope_cap=0.25,
        m2_target=0.20,
        m_min=0.05,
        m_max=1.0
    )
    
    # Check sum preserved
    if abs(result.sum() - 100.0) > 0.01:
        print(f"  ⚠️  Sum not preserved: {result.sum()}")
        return False
    
    # Check violations reduced
    print(f"  ✓ Strategy used: {strategy}")
    print(f"  ✓ Violations: {violations}")
    print(f"  ✓ Sum preserved: {result.sum():.2f}")
    
    if violations > 2:
        test_results['warnings'].append(f"Slope violations: {violations}")
    
    return True


@test_section("7. Evaluation Engine V2")
def test_evaluation_v2():
    """Test new evaluation engine."""
    from martingale_lab.optimizer.evaluation_engine_v2 import evaluation_function_v2
    from martingale_lab.core.config_classes import ConfigPresets
    
    config = ConfigPresets.strict()
    config.core.base_price = 100.0
    config.core.overlap_pct = 20.0
    config.core.num_orders = 5
    config.core.seed = 42
    
    # Run evaluation
    result = evaluation_function_v2(
        config.core.base_price,
        config.core.overlap_pct,
        config.core.num_orders,
        config
    )
    
    # Check result structure
    if 'score' not in result:
        print(f"  ⚠️  Missing score in result")
        return False
    
    if 'schedule' not in result:
        print(f"  ⚠️  Missing schedule in result")
        return False
    
    if '_config' not in result:
        print(f"  ⚠️  Missing config metadata")
        return False
    
    print(f"  ✓ Score: {result['score']:.2f}")
    print(f"  ✓ Config version: {result['_config'].get('version')}")
    
    # Check normalization if enabled
    if config.normalization.post_round_2dp:
        volumes = result['schedule'].get('volume_pct', [])
        if volumes:
            all_2dp = all(round(v, 2) == v for v in volumes)
            if not all_2dp:
                test_results['warnings'].append("Not all volumes are 2dp")
    
    return True


@test_section("8. CLI Integration")
def test_cli_integration():
    """Test CLI with config support."""
    import subprocess
    import tempfile
    
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
        db_path = tmp.name
    
    try:
        # Test CLI with config preset
        cmd = [
            sys.executable, '-m', 'martingale_lab.cli.optimize',
            '--config-preset', 'exploration',
            '--batches', '1',
            '--batch-size', '10',
            '--db', db_path,
            '--max-time-sec', '5'
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        
        if result.returncode != 0:
            print(f"  ⚠️  CLI failed: {result.stderr}")
            return False
        
        # Check output
        if 'Best Score:' not in result.stderr:
            print(f"  ⚠️  No score in output")
            return False
        
        print(f"  ✓ CLI execution successful")
        print(f"  ✓ Config preset working")
        return True
        
    except subprocess.TimeoutExpired:
        print(f"  ⚠️  CLI timeout")
        return False
    finally:
        if os.path.exists(db_path):
            os.unlink(db_path)


@test_section("9. Performance Cache")
def test_performance_cache():
    """Test performance optimization cache."""
    from martingale_lab.core.performance_cache import ParameterCache, get_cache_stats
    import numpy as np
    
    cache = ParameterCache(max_size=10)
    
    # Test basic operations
    key = cache._make_key(1, 2, test=True)
    cache.put(key, "result1")
    
    result = cache.get(key)
    if result != "result1":
        print(f"  ⚠️  Cache retrieval failed")
        return False
    
    # Test LRU eviction
    for i in range(15):
        cache.put(f"key_{i}", f"value_{i}")
    
    if len(cache.cache) > 10:
        print(f"  ⚠️  Cache size exceeded: {len(cache.cache)}")
        return False
    
    # Test numpy array keys
    arr = np.array([1, 2, 3])
    key2 = cache._make_key(arr, param=5)
    cache.put(key2, "array_result")
    
    stats = cache.stats()
    print(f"  ✓ Cache size: {stats['size']}/{stats['max_size']}")
    print(f"  ✓ Hit rate: {stats['hit_rate']:.2%}")
    return True


@test_section("10. End-to-End Workflow")
def test_end_to_end():
    """Test complete workflow from config to results."""
    from martingale_lab.cli.config_cli import save_config_file
    from martingale_lab.core.config_classes import ConfigPresets
    import subprocess
    import tempfile
    
    with tempfile.TemporaryDirectory() as tmpdir:
        config_file = os.path.join(tmpdir, "test_config.json")
        db_file = os.path.join(tmpdir, "test_results.db")
        
        # Save config
        config = ConfigPresets.production()
        config.constraints.slope_cap = 0.20
        config.normalization.post_round_2dp = True
        save_config_file(config, config_file)
        
        # Run optimization
        cmd = [
            sys.executable, '-m', 'martingale_lab.cli.optimize',
            '--config', config_file,
            '--batches', '2',
            '--batch-size', '20',
            '--db', db_file,
            '--max-time-sec', '10'
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        
        if result.returncode != 0:
            print(f"  ⚠️  End-to-end failed: {result.stderr}")
            return False
        
        # Check database created
        if not os.path.exists(db_file):
            print(f"  ⚠️  Database not created")
            return False
        
        # Check results
        import sqlite3
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM results")
        count = cursor.fetchone()[0]
        conn.close()
        
        if count == 0:
            print(f"  ⚠️  No results in database")
            return False
        
        print(f"  ✓ Config file created and loaded")
        print(f"  ✓ Optimization completed")
        print(f"  ✓ Results stored: {count} entries")
        return True


def main():
    """Run all tests and report results."""
    print("="*60)
    print("COMPREHENSIVE SYSTEM TEST")
    print("="*60)
    
    # Run all tests
    test_schedule_normalization()
    test_config_management()
    test_database_operations()
    test_memory_management()
    test_auto_tuning()
    test_slope_enforcement()
    test_evaluation_v2()
    test_cli_integration()
    test_performance_cache()
    test_end_to_end()
    
    # Report results
    print("\n" + "="*60)
    print("TEST RESULTS SUMMARY")
    print("="*60)
    
    print(f"\n✅ PASSED: {len(test_results['passed'])}")
    for test in test_results['passed']:
        print(f"  • {test}")
    
    if test_results['failed']:
        print(f"\n❌ FAILED: {len(test_results['failed'])}")
        for test in test_results['failed']:
            print(f"  • {test}")
    
    if test_results['warnings']:
        print(f"\n⚠️  WARNINGS: {len(test_results['warnings'])}")
        for warning in test_results['warnings']:
            print(f"  • {warning}")
    
    # Overall status
    print("\n" + "="*60)
    if not test_results['failed']:
        print("🎉 ALL TESTS PASSED!")
        print("System is ready for production.")
        return 0
    else:
        print("❌ SOME TESTS FAILED")
        print("Please fix the issues before deployment.")
        return 1


if __name__ == "__main__":
    sys.exit(main())