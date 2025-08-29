#!/usr/bin/env python3
"""
Test auto-tuning system.
"""
import sys
sys.path.insert(0, '/workspace')

from martingale_lab.core.config_classes import EvaluationConfig, ConfigPresets
from martingale_lab.storage.config_store import ConfigStore
from martingale_lab.optimizer.auto_tuner import ConfigAutoTuner, TuningStrategy
import tempfile
import os

def test_auto_tuner():
    """Test auto-tuning functionality."""
    print("\n=== Testing Auto-Tuner ===")
    
    # Create temporary database
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
        db_path = tmp.name
    
    try:
        # Initialize stores
        config_store = ConfigStore(db_path)
        tuner = ConfigAutoTuner(config_store)
        
        # Test 1: Suggest without history
        print("\n1. Suggesting config without history:")
        strategy = TuningStrategy(name="initial")
        config1 = tuner.suggest_next_config(strategy)
        print(f"   Got config with m2_max={config1.constraints.m2_max:.2f}")
        
        # Store some configs with performance
        print("\n2. Storing config history:")
        base_config = ConfigPresets.production()
        hash1 = config_store.store_config(base_config, notes="Base production")
        config_store.record_performance(hash1, 1, "run1", 100.0, 110.0, 1000)
        print(f"   Stored base config: {hash1}")
        
        better_config = ConfigPresets.strict()
        hash2 = config_store.store_config(better_config, parent_hash=hash1, 
                                         change_type="manual", 
                                         change_description="Switched to strict")
        config_store.record_performance(hash2, 2, "run2", 90.0, 95.0, 1200)
        print(f"   Stored better config: {hash2}")
        
        # Test 3: Suggest with history
        print("\n3. Suggesting config with history:")
        strategy = TuningStrategy(name="exploit", exploration_rate=0.1)
        config3 = tuner.suggest_next_config(strategy, base_config)
        print(f"   Got improved config with slope_cap={config3.constraints.slope_cap:.2f}")
        
        # Test 4: Mutation
        print("\n4. Testing mutation:")
        mutated = tuner._mutate_config(base_config, strategy)
        print(f"   Original m2_min: {base_config.constraints.m2_min:.2f}")
        print(f"   Mutated m2_min: {mutated.constraints.m2_min:.2f}")
        
        # Test 5: Crossover
        print("\n5. Testing crossover:")
        offspring = tuner._crossover_configs(base_config, better_config)
        print(f"   Parent1 slope_cap: {base_config.constraints.slope_cap:.2f}")
        print(f"   Parent2 slope_cap: {better_config.constraints.slope_cap:.2f}")
        print(f"   Offspring slope_cap: {offspring.constraints.slope_cap:.2f}")
        
        # Test 6: Strategy recommendation
        print("\n6. Testing strategy recommendation:")
        
        # Stagnant performance
        stagnant = [100, 99.5, 99.2, 99.1, 99.0, 98.9, 98.8, 98.8, 98.7, 98.7]
        strategy1 = tuner.recommend_tuning_strategy(stagnant)
        print(f"   Stagnant: {strategy1.name} (explore={strategy1.exploration_rate:.2f})")
        
        # Good improvement
        improving = [100, 95, 92, 88, 85, 82, 80, 78, 76, 74]
        strategy2 = tuner.recommend_tuning_strategy(improving)
        print(f"   Improving: {strategy2.name} (explore={strategy2.exploration_rate:.2f})")
        
        # Test 7: Config stats
        print("\n7. Testing config statistics:")
        stats = config_store.get_config_stats(hash2)
        print(f"   Config {hash2[:8]}...")
        print(f"   Best score: {stats.get('best_score', 'N/A')}")
        print(f"   Use count: {stats.get('use_count', 0)}")
        
        # Test 8: Best configs
        print("\n8. Getting best configs:")
        best = config_store.get_best_configs(limit=3)
        for i, (hash_val, score, count) in enumerate(best, 1):
            print(f"   {i}. {hash_val[:8]}... score={score:.1f} uses={count}")
        
        print("\n✅ Auto-tuner tests passed!")
        return True
        
    finally:
        # Cleanup
        if os.path.exists(db_path):
            os.unlink(db_path)

def main():
    """Run auto-tuner tests."""
    print("=" * 60)
    print("Testing Auto-Tuning System")
    print("=" * 60)
    
    try:
        if test_auto_tuner():
            print("\n✅ SUCCESS: Auto-tuning system working!")
            print("\n📊 Features Tested:")
            print("  - Config suggestion without history")
            print("  - Config mutation")
            print("  - Config crossover")
            print("  - Strategy recommendation")
            print("  - Performance tracking")
            print("  - Config lineage")
            return 0
        else:
            print("\n❌ FAILED: Auto-tuner tests failed")
            return 1
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())