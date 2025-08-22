"""
Property-based tests for martingale optimization system.
Uses hypothesis to verify mathematical invariants and system properties.
"""
import pytest
import numpy as np
from hypothesis import given, strategies as st, assume, settings
from hypothesis.extra.numpy import arrays

from ..core.constraints import (
    validate_search_space, assert_schedule_invariants, 
    validate_martingale_bounds, validate_need_pct_sanity
)
from ..core.penalties import (
    monotone_violation, tail_cap_penalty, extreme_volume_penalty,
    need_pct_smoothness_penalty, GiniPenalty, EntropyPenalty
)
from ..utils.runctx import make_runctx
from ..utils.error_boundaries import SafeEvaluator, BatchProcessor
from ..utils.logging import LogContext, setup_logging


class TestMonotonicity:
    """Test monotonicity properties."""
    
    @given(st.lists(st.floats(min_value=0, max_value=10), min_size=3, max_size=10))
    def test_monotone_sequence_has_zero_violation(self, values):
        """Monotonically increasing sequences should have zero violation."""
        # Create monotonic sequence by cumulative sum
        monotonic_sequence = np.cumsum(np.abs(values))
        
        violation = monotone_violation(monotonic_sequence)
        assert violation == 0.0, f"Monotonic sequence {monotonic_sequence} should have zero violation"
    
    @given(arrays(np.float64, (5,), elements=st.floats(0, 100)))
    def test_sorted_array_is_monotonic(self, arr):
        """Sorted arrays should always be monotonic."""
        sorted_arr = np.sort(arr)
        violation = monotone_violation(sorted_arr)
        assert violation == 0.0
    
    @given(st.lists(st.floats(min_value=1, max_value=100), min_size=2, max_size=8))
    def test_decreasing_sequence_has_positive_violation(self, values):
        """Decreasing sequences should have positive violation."""
        # Create decreasing sequence
        decreasing_sequence = np.array(sorted(values, reverse=True))
        
        # Skip if all values are the same (no violation)
        assume(len(set(values)) > 1)
        
        violation = monotone_violation(decreasing_sequence)
        assert violation > 0, f"Decreasing sequence should have positive violation"


class TestVolumeConstraints:
    """Test volume distribution properties."""
    
    @given(arrays(np.float64, (5,), elements=st.floats(0.01, 0.5)))
    def test_volume_normalization(self, volumes):
        """Normalized volumes should sum to approximately 1."""
        # Normalize to sum to 1
        normalized_volumes = volumes / np.sum(volumes)
        
        assert abs(np.sum(normalized_volumes) - 1.0) < 1e-10
        assert np.all(normalized_volumes >= 0)
    
    @given(st.integers(3, 10))
    def test_uniform_distribution_has_maximum_entropy(self, num_levels):
        """Uniform distribution should have maximum entropy."""
        uniform_volumes = np.ones(num_levels) / num_levels
        
        entropy = EntropyPenalty.calculate_entropy(uniform_volumes)
        max_entropy = EntropyPenalty.max_entropy(num_levels)
        
        assert abs(entropy - max_entropy) < 1e-10
    
    @given(arrays(np.float64, (4,), elements=st.floats(0.01, 1.0)))
    def test_gini_coefficient_bounds(self, volumes):
        """Gini coefficient should be between 0 and 1."""
        # Normalize volumes
        volumes = volumes / np.sum(volumes)
        
        gini = GiniPenalty.calculate_gini(volumes)
        assert 0 <= gini <= 1, f"Gini coefficient {gini} out of bounds [0,1]"
    
    @given(st.floats(0.1, 0.8))
    def test_single_volume_gini_is_maximum(self, total_volume):
        """Single non-zero volume should give maximum Gini coefficient."""
        # Create array with one large volume, rest tiny
        volumes = np.array([total_volume, 0.01, 0.01, 0.01])
        volumes = volumes / np.sum(volumes)
        
        gini = GiniPenalty.calculate_gini(volumes)
        # Should be high (close to maximum inequality)
        assert gini > 0.5


class TestNeedPctProperties:
    """Test NeedPct calculation properties."""
    
    @given(arrays(np.float64, (5,), elements=st.floats(0, 30)))
    def test_need_pct_sanity_bounds(self, need_pct_values):
        """NeedPct values should pass sanity checks when in bounds."""
        # Ensure all values are within reasonable bounds
        bounded_values = np.clip(need_pct_values, 0, 30)
        
        # Should not raise assertion error
        validate_need_pct_sanity(bounded_values)
    
    @given(st.lists(st.floats(0, 50), min_size=3, max_size=8))
    def test_smooth_need_pct_has_low_penalty(self, values):
        """Smoothly changing NeedPct should have low penalty."""
        # Create smooth sequence
        smooth_values = np.linspace(min(values), max(values), len(values))
        
        penalty = need_pct_smoothness_penalty(smooth_values, max_jump_pct=5.0)
        # Should be zero or very low for smooth transitions
        assert penalty <= 1.0


class TestSearchSpaceValidation:
    """Test search space validation properties."""
    
    @given(
        st.floats(0.1, 50.0),  # overlap_min
        st.floats(50.1, 100.0),  # overlap_max
        st.integers(1, 20),     # orders_min
        st.integers(21, 100),   # orders_max
        st.floats(0, 1),        # alpha
        st.floats(0, 1),        # beta
        st.floats(0, 1)         # gamma
    )
    def test_valid_search_space_passes_validation(self, overlap_min, overlap_max, 
                                                 orders_min, orders_max, 
                                                 alpha, beta, gamma):
        """Valid search space configuration should pass validation."""
        # Ensure at least one objective is active
        if alpha + beta + gamma == 0:
            alpha = 0.1  # Make at least one active
        
        config = {
            'overlap_min': overlap_min,
            'overlap_max': overlap_max,
            'orders_min': orders_min,
            'orders_max': orders_max,
            'alpha': alpha,
            'beta': beta,
            'gamma': gamma
        }
        
        # Should not raise assertion error
        validate_search_space(config)
    
    @given(st.floats(-10, 110))
    def test_martingale_bounds_validation(self, martingale_pct):
        """Test martingale percentage bounds validation."""
        if 0 <= martingale_pct <= 100:
            # Should pass validation
            validate_martingale_bounds(martingale_pct)
        else:
            # Should raise assertion error
            with pytest.raises(AssertionError):
                validate_martingale_bounds(martingale_pct)


class TestPenaltyFunctions:
    """Test penalty function properties."""
    
    @given(arrays(np.float64, (4,), elements=st.floats(0.05, 0.4)))
    def test_tail_cap_penalty_threshold(self, volumes):
        """Tail cap penalty should activate above threshold."""
        # Normalize volumes
        volumes = volumes / np.sum(volumes)
        
        # Test with different thresholds
        penalty_strict = tail_cap_penalty(volumes, max_last_order_pct=0.1)
        penalty_lenient = tail_cap_penalty(volumes, max_last_order_pct=0.8)
        
        # Strict threshold should give higher or equal penalty
        assert penalty_strict >= penalty_lenient
    
    @given(arrays(np.float64, (5,), elements=st.floats(0.001, 0.999)))
    def test_extreme_volume_penalty_symmetry(self, volumes):
        """Extreme volume penalty should penalize both too small and too large."""
        # Test very small volumes
        small_volumes = volumes * 0.001  # Make very small
        small_penalty = extreme_volume_penalty(small_volumes, min_vol=0.01, max_vol=0.5)
        
        # Test very large volumes  
        large_volumes = volumes * 2.0  # Make large
        large_penalty = extreme_volume_penalty(large_volumes, min_vol=0.01, max_vol=0.5)
        
        # Both should have some penalty if volumes are extreme
        if np.any(small_volumes < 0.01):
            assert small_penalty > 0
        if np.any(large_volumes > 0.5):
            assert large_penalty > 0


class TestRunContextProperties:
    """Test run context properties."""
    
    @given(st.integers(1, 2**30))
    def test_run_context_reproducibility(self, seed):
        """Same seed should produce reproducible run context."""
        ctx1 = make_runctx(seed)
        ctx2 = make_runctx(seed)
        
        # Seeds should be the same
        assert ctx1.seed == ctx2.seed == seed
        
        # Run IDs should be different (UUIDs)
        assert ctx1.run_id != ctx2.run_id
        
        # Code versions should be the same
        assert ctx1.code_version == ctx2.code_version
    
    def test_run_context_without_seed_is_random(self):
        """Run context without seed should generate different seeds."""
        ctx1 = make_runctx()
        ctx2 = make_runctx()
        
        # Should have different seeds and run IDs
        assert ctx1.seed != ctx2.seed
        assert ctx1.run_id != ctx2.run_id


class TestErrorBoundaries:
    """Test error handling properties."""
    
    def test_safe_evaluator_handles_exceptions(self):
        """Safe evaluator should handle exceptions gracefully."""
        logger = setup_logging("test-run")
        log_ctx = LogContext(logger, "test-run")
        evaluator = SafeEvaluator(log_ctx)
        
        def failing_evaluator(candidate):
            raise ValueError("Test error")
        
        def fallback_evaluator(candidate):
            return 1.0
        
        # Should not raise exception, should use fallback
        result = evaluator.safe_eval_candidate(
            "test-candidate", 
            failing_evaluator, 
            fallback_evaluator
        )
        
        assert result.success == True
        assert result.fallback_used == True
        assert result.result == 1.0
    
    def test_batch_processor_isolation(self):
        """Batch processor should isolate candidate failures."""
        logger = setup_logging("test-batch")
        log_ctx = LogContext(logger, "test-batch")
        processor = BatchProcessor(log_ctx)
        
        def mixed_evaluator(candidate):
            # Fail on even indices, succeed on odd
            if int(candidate.split('-')[1]) % 2 == 0:
                raise ValueError("Even candidate fails")
            return float(candidate.split('-')[1])
        
        candidates = [f"candidate-{i}" for i in range(10)]
        
        result = processor.process_batch_safe(
            candidates, 
            mixed_evaluator,
            max_failures_pct=80.0  # Allow high failure rate for test
        )
        
        # Should have some successes and some failures
        assert result['success_count'] > 0
        assert result['failure_count'] > 0
        assert result['success_count'] + result['failure_count'] == len(candidates)


# Configuration for hypothesis
settings.register_profile("ci", max_examples=50, deadline=1000)
settings.register_profile("dev", max_examples=10, deadline=500)
settings.load_profile("dev")  # Use dev profile by default
