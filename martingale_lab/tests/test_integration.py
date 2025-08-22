"""
Integration tests for the complete martingale optimization system.
Tests the full pipeline with small search spaces for deterministic results.
"""
import pytest
import numpy as np
import tempfile
import os
from pathlib import Path

from ..utils.runctx import make_runctx
from ..utils.logging import setup_logging, LogContext
from ..utils.error_boundaries import SafeEvaluator, BatchProcessor
from ..utils.metrics_monitor import PerformanceTracker
from ..utils.early_pruning import PruningConfig, EarlyStoppingManager
from ..storage.checkpoint_store import CheckpointStore, CandidateRecord
from ..core.constraints import validate_search_space, ConstraintValidator
from ..core.penalties import ComprehensivePenaltySystem, DEFAULT_PENALTY_WEIGHTS


class MockCandidate:
    """Mock candidate for testing."""
    
    def __init__(self, candidate_id: str, overlap: float, orders: int, 
                 score: float = None):
        self.id = candidate_id
        self.overlap = overlap
        self.orders = orders
        self.score = score or np.random.uniform(10, 100)
    
    def __str__(self):
        return f"Candidate({self.id}, overlap={self.overlap}, orders={self.orders})"


class MockSchedule:
    """Mock schedule for testing."""
    
    def __init__(self, num_levels: int = 5):
        self.num_levels = num_levels
        self.volumes = np.random.dirichlet(np.ones(num_levels))  # Sum to 1
        self.orders = np.sort(np.random.randint(1, 20, num_levels))  # Monotonic
        self.overlaps = np.sort(np.random.uniform(0, 100, num_levels))  # Monotonic
        self.indent_pct = np.sort(np.random.uniform(0, 10, num_levels))  # Monotonic


class TestEndToEndPipeline:
    """Test complete optimization pipeline."""
    
    def test_small_optimization_run(self):
        """Test a complete small optimization run."""
        # Setup
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, "test.db")
            
            # Create run context
            run_ctx = make_runctx(seed=42)
            
            # Setup logging
            logger = setup_logging(run_ctx.run_id, log_dir=temp_dir)
            log_ctx = LogContext(logger, run_ctx.run_id)
            
            # Setup checkpoint store
            checkpoint_store = CheckpointStore(db_path)
            checkpoint_store.set_log_context(log_ctx)
            
            # Start run
            params = {
                'overlap_min': 1.0,
                'overlap_max': 5.0,
                'orders_min': 3,
                'orders_max': 6,
                'alpha': 0.4,
                'beta': 0.3,
                'gamma': 0.3
            }
            
            # Validate configuration
            validate_search_space(params)
            
            run_record = checkpoint_store.start_run(run_ctx, params)
            assert run_record.id == run_ctx.run_id
            
            # Setup performance tracking
            performance_tracker = PerformanceTracker(log_ctx)
            performance_tracker.start_batch()
            
            # Create small batch of candidates
            candidates = []
            for i in range(10):
                candidate = MockCandidate(
                    f"candidate-{i:03d}",
                    overlap=1.0 + i * 0.4,  # 1.0 to 4.6
                    orders=3 + i % 4,       # 3 to 6
                    score=20.0 + i * 2.0    # Deterministic scores
                )
                candidates.append(candidate)
            
            # Process batch
            def mock_evaluator(candidate):
                # Simulate evaluation with some computation
                performance_tracker.record_evaluation(True, candidate.score)
                return candidate.score
            
            batch_processor = BatchProcessor(log_ctx)
            batch_results = batch_processor.process_batch_safe(
                candidates, mock_evaluator, max_failures_pct=20.0
            )
            
            # Verify results
            assert batch_results['success_count'] == len(candidates)
            assert batch_results['failure_count'] == 0
            assert batch_results['success_rate_pct'] == 100.0
            
            # Save candidates to database
            for i, result in enumerate(batch_results['successful_results']):
                candidate = result['candidate']
                candidate_record = CandidateRecord(
                    id=candidate.id,
                    run_id=run_ctx.run_id,
                    batch_idx=0,
                    overlap=candidate.overlap,
                    orders=candidate.orders,
                    params_json='{"test": true}',
                    schedule_json='{"mock": true}',
                    J=result['result'],
                    max_need=result['result'] * 0.8,
                    var_need=result['result'] * 0.1,
                    tail=result['result'] * 0.1,
                    gini=0.5,
                    entropy=1.0,
                    penalties_json='{}',
                    evaluation_time_ms=result['duration_ms'],
                    fallback_used=result['fallback_used']
                )
                checkpoint_store.save_candidate(candidate_record)
            
            # Finish batch
            best_score = min(r['result'] for r in batch_results['successful_results'])
            checkpoint_store.finish_batch(
                run_ctx.run_id, 0, len(candidates), best_score, 100.0, 10.0
            )
            
            # Finish run
            checkpoint_store.finish_run(run_ctx.run_id, 'completed')
            
            # Verify database state
            progress = checkpoint_store.get_run_progress(run_ctx.run_id)
            assert progress['batch_count'] == 1
            assert progress['total_evaluations'] == len(candidates)
            assert progress['best_score'] == best_score
            
            best_candidates = checkpoint_store.get_best_candidates(run_ctx.run_id, limit=3)
            assert len(best_candidates) == 3
            assert best_candidates[0].J <= best_candidates[1].J <= best_candidates[2].J
    
    def test_error_recovery_pipeline(self):
        """Test pipeline with errors and recovery."""
        with tempfile.TemporaryDirectory() as temp_dir:
            run_ctx = make_runctx(seed=123)
            logger = setup_logging(run_ctx.run_id, log_dir=temp_dir)
            log_ctx = LogContext(logger, run_ctx.run_id)
            
            # Create evaluator that fails sometimes
            def unreliable_evaluator(candidate):
                if candidate.id.endswith('5'):  # Fail on candidates ending with 5
                    raise ValueError("Simulated failure")
                return candidate.score
            
            def fallback_evaluator(candidate):
                return candidate.score * 1.1  # Slightly different result
            
            # Create batch with some failing candidates
            candidates = [MockCandidate(f"test-{i}", 1.0, 3, 10.0 + i) for i in range(10)]
            
            batch_processor = BatchProcessor(log_ctx)
            results = batch_processor.process_batch_safe(
                candidates, unreliable_evaluator, fallback_evaluator, max_failures_pct=50.0
            )
            
            # Should have successes and fallback usage
            assert results['success_count'] > 0
            assert results['success_rate_pct'] < 100.0  # Some used fallback
            
            # Check fallback usage
            fallback_used_count = sum(1 for r in results['successful_results'] if r['fallback_used'])
            assert fallback_used_count > 0  # At least one fallback was used
    
    def test_early_stopping_integration(self):
        """Test early stopping with the complete pipeline."""
        with tempfile.TemporaryDirectory() as temp_dir:
            run_ctx = make_runctx(seed=456)
            logger = setup_logging(run_ctx.run_id, log_dir=temp_dir)
            log_ctx = LogContext(logger, run_ctx.run_id)
            
            # Configure aggressive early stopping
            pruning_config = PruningConfig(
                max_batch_time=5.0,  # 5 second limit
                target_evaluations=50,
                min_budget=2,
                max_budget=10
            )
            
            early_stopping = EarlyStoppingManager(pruning_config, log_ctx)
            early_stopping.start_batch()
            
            # Create many candidates
            candidates = [MockCandidate(f"early-{i}", 1.0, 3) for i in range(100)]
            
            def slow_evaluator(candidate):
                import time
                time.sleep(0.1)  # Slow evaluation
                return candidate.score
            
            # Process with early stopping
            evaluated_count = 0
            for candidate in candidates:
                if not early_stopping.should_evaluate_candidate(candidate.id):
                    break
                
                early_stopping.register_candidate(candidate.id)
                score = slow_evaluator(candidate)
                early_stopping.record_evaluation_complete(score)
                evaluated_count += 1
            
            # Should stop before evaluating all candidates
            assert evaluated_count < len(candidates)
            
            # Get stopping statistics
            stats = early_stopping.get_comprehensive_stats()
            assert stats['budget_progress']['budget_exhausted'] == True
            assert stats['budget_progress']['stop_reason'] in ['time_budget_exhausted', 'evaluation_budget_reached']


class TestConstraintValidationIntegration:
    """Test constraint validation in realistic scenarios."""
    
    def test_schedule_validation_pipeline(self):
        """Test schedule validation with realistic data."""
        # Create valid schedule
        schedule = MockSchedule(num_levels=5)
        
        # Should pass validation
        validator = ConstraintValidator()
        assert validator.validate_schedule(schedule)
        
        # Test with invalid schedule (non-monotonic orders)
        invalid_schedule = MockSchedule(num_levels=4)
        invalid_schedule.orders = np.array([10, 5, 15, 8])  # Non-monotonic
        
        assert not validator.validate_schedule(invalid_schedule)
    
    def test_penalty_system_integration(self):
        """Test comprehensive penalty system with realistic data."""
        penalty_system = ComprehensivePenaltySystem()
        
        # Create test data
        max_need = 25.0
        var_need = 5.0
        tail = 3.0
        volumes = np.array([0.1, 0.2, 0.3, 0.25, 0.15])  # Sum to 1
        indent_pct = np.array([0, 1, 2, 3, 4])  # Monotonic
        need_pct_values = np.array([20, 22, 25, 24, 23])  # Reasonable values
        
        # Calculate comprehensive score
        score_breakdown = penalty_system.calculate_comprehensive_score(
            max_need, var_need, tail, volumes, indent_pct, need_pct_values
        )
        
        # Verify structure
        assert 'final_score' in score_breakdown
        assert 'objective_score' in score_breakdown
        assert 'total_penalty' in score_breakdown
        assert 'penalty_breakdown' in score_breakdown
        
        # Final score should be sum of objective and penalties
        expected_final = score_breakdown['objective_score'] + score_breakdown['total_penalty']
        assert abs(score_breakdown['final_score'] - expected_final) < 1e-6
        
        # Test penalty summary
        summary = penalty_system.get_penalty_summary(score_breakdown)
        assert isinstance(summary, str)
        assert 'Final Score:' in summary


class TestDatabaseIntegration:
    """Test database operations in realistic scenarios."""
    
    def test_checkpoint_resume_scenario(self):
        """Test checkpoint and resume functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, "resume_test.db")
            
            # First run - simulate interruption
            run_ctx1 = make_runctx(seed=789)
            store = CheckpointStore(db_path)
            
            params = {'test': 'params'}
            run_record = store.start_run(run_ctx1, params)
            
            # Add some batches and candidates
            store.start_batch(run_ctx1.run_id, 0, {'space': 'small'})
            store.start_batch(run_ctx1.run_id, 1, {'space': 'medium'})
            
            # Add candidates to first batch
            for i in range(5):
                candidate = CandidateRecord(
                    id=f"resume-{i}",
                    run_id=run_ctx1.run_id,
                    batch_idx=0,
                    overlap=1.0 + i,
                    orders=3 + i,
                    params_json='{}',
                    schedule_json='{}',
                    J=10.0 + i,
                    max_need=8.0 + i,
                    var_need=1.0,
                    tail=1.0,
                    gini=0.5,
                    entropy=1.0,
                    penalties_json='{}'
                )
                store.save_candidate(candidate)
            
            store.finish_batch(run_ctx1.run_id, 0, 5, 10.0, 100.0, 50.0)
            
            # Simulate run interruption (don't finish run)
            
            # Resume scenario
            resumable_runs = store.get_resumable_runs()
            assert len(resumable_runs) == 1
            assert resumable_runs[0].id == run_ctx1.run_id
            assert resumable_runs[0].last_batch_idx == 0
            
            # Get progress
            progress = store.get_run_progress(run_ctx1.run_id)
            assert progress['batch_count'] == 2  # 2 batches started
            assert progress['total_evaluations'] == 5  # Only first batch finished
            assert progress['best_score'] == 10.0
            
            # Continue from where left off
            store.finish_batch(run_ctx1.run_id, 1, 3, 8.0, 100.0, 40.0)  # Finish second batch
            store.finish_run(run_ctx1.run_id, 'completed')
            
            # Verify final state
            final_progress = store.get_run_progress(run_ctx1.run_id)
            assert final_progress['total_evaluations'] == 8  # 5 + 3
            assert final_progress['best_score'] == 8.0  # Better score from second batch
            
            # No more resumable runs
            resumable_runs = store.get_resumable_runs()
            assert len(resumable_runs) == 0
    
    def test_metrics_storage_integration(self):
        """Test metrics storage and retrieval."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, "metrics_test.db")
            
            run_ctx = make_runctx(seed=999)
            store = CheckpointStore(db_path)
            
            # Start run
            store.start_run(run_ctx, {'test': True})
            
            # Log various metrics
            metrics_data = [
                ('eval_rate', 10.5),
                ('eval_rate', 12.3),
                ('eval_rate', 11.8),
                ('accept_rate', 0.85),
                ('accept_rate', 0.90),
                ('memory_usage', 256.7),
                ('memory_usage', 267.2)
            ]
            
            for metric_name, value in metrics_data:
                store.log_metric(run_ctx.run_id, 0, metric_name, value)
            
            # Retrieve metrics
            eval_rates = store.get_metrics(run_ctx.run_id, 'eval_rate')
            assert len(eval_rates) == 3
            assert all(isinstance(timestamp, str) and isinstance(value, float) 
                      for timestamp, value in eval_rates)
            
            accept_rates = store.get_metrics(run_ctx.run_id, 'accept_rate')
            assert len(accept_rates) == 2
            
            memory_metrics = store.get_metrics(run_ctx.run_id, 'memory_usage')
            assert len(memory_metrics) == 2


@pytest.fixture
def temp_workspace():
    """Provide a temporary workspace for tests."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


class TestSystemRobustness:
    """Test system robustness under various conditions."""
    
    def test_high_failure_rate_handling(self, temp_workspace):
        """Test system behavior with high failure rates."""
        run_ctx = make_runctx(seed=111)
        logger = setup_logging(run_ctx.run_id, log_dir=temp_workspace)
        log_ctx = LogContext(logger, run_ctx.run_id)
        
        def high_failure_evaluator(candidate):
            # 70% failure rate
            if np.random.random() < 0.7:
                raise RuntimeError("High failure rate test")
            return candidate.score
        
        def reliable_fallback(candidate):
            return candidate.score * 1.2
        
        candidates = [MockCandidate(f"robust-{i}", 1.0, 3) for i in range(20)]
        
        batch_processor = BatchProcessor(log_ctx)
        results = batch_processor.process_batch_safe(
            candidates, high_failure_evaluator, reliable_fallback, max_failures_pct=80.0
        )
        
        # Should complete despite high failure rate
        assert results['total_processed'] == len(candidates)
        assert results['success_count'] > 0  # Some should succeed via fallback
        
        # Most should use fallback
        fallback_count = sum(1 for r in results['successful_results'] if r['fallback_used'])
        assert fallback_count > results['success_count'] * 0.5  # Most used fallback
    
    def test_memory_constraint_handling(self, temp_workspace):
        """Test handling of memory-intensive operations."""
        run_ctx = make_runctx(seed=222)
        logger = setup_logging(run_ctx.run_id, log_dir=temp_workspace)
        log_ctx = LogContext(logger, run_ctx.run_id)
        
        performance_tracker = PerformanceTracker(log_ctx)
        performance_tracker.start_batch()
        
        # Simulate memory-intensive evaluations
        for i in range(10):
            # Create some temporary arrays to use memory
            temp_data = np.random.random((1000, 1000))  # ~8MB array
            score = np.mean(temp_data)
            
            performance_tracker.record_evaluation(True, score)
            del temp_data  # Clean up
        
        metrics = performance_tracker.get_current_metrics()
        
        # Should track memory usage
        assert metrics.memory_usage_mb > 0
        assert metrics.evaluations_per_second > 0
        assert metrics.accept_ratio == 1.0  # All succeeded
