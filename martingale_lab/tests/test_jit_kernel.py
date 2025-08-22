"""
Tests for JIT kernels module.
"""
import unittest
import numpy as np
from ..core.types import Params, Schedule, ScoreBreakdown
from ..core.jit_kernels import (
    calculate_martingale_volumes, calculate_overlaps, calculate_orders,
    calculate_max_score, calculate_variance_score, calculate_tail_score,
    calculate_gini_penalty, calculate_monotone_penalty, calculate_smoothness_penalty,
    evaluate_kernel, evaluate_params, evaluate_params_batch, create_schedule_from_params
)


class TestJITKernels(unittest.TestCase):
    """Test JIT kernel functions."""
    
    def test_calculate_martingale_volumes(self):
        """Test martingale volume calculation."""
        volumes = calculate_martingale_volumes(base_volume=100.0, multiplier=2.0, num_levels=4)
        expected = np.array([100.0, 200.0, 400.0, 800.0])
        np.testing.assert_array_almost_equal(volumes, expected)
    
    def test_calculate_overlaps(self):
        """Test overlap calculation."""
        overlaps = calculate_overlaps(min_overlap=1.0, max_overlap=30.0, num_levels=4)
        expected = np.array([1.0, 10.67, 20.33, 30.0])
        np.testing.assert_array_almost_equal(overlaps, expected, decimal=2)
    
    def test_calculate_orders(self):
        """Test order calculation."""
        orders = calculate_orders(min_order=3, max_order=10, num_levels=4)
        expected = np.array([3, 5, 7, 10])
        np.testing.assert_array_equal(orders, expected)
    
    def test_calculate_max_score(self):
        """Test max score calculation."""
        volumes = np.array([100.0, 200.0, 400.0, 800.0])
        max_score = calculate_max_score(volumes)
        self.assertEqual(max_score, 800.0)
    
    def test_calculate_variance_score(self):
        """Test variance score calculation."""
        volumes = np.array([100.0, 200.0, 400.0, 800.0])
        variance_score = calculate_variance_score(volumes)
        expected_variance = np.var(volumes)
        self.assertAlmostEqual(variance_score, expected_variance)
    
    def test_calculate_variance_score_single_element(self):
        """Test variance score calculation with single element."""
        volumes = np.array([100.0])
        variance_score = calculate_variance_score(volumes)
        self.assertEqual(variance_score, 0.0)
    
    def test_calculate_tail_score(self):
        """Test tail score calculation."""
        volumes = np.array([100.0, 200.0, 400.0, 800.0])
        tail_score = calculate_tail_score(volumes, percentile=0.75)
        # Should return mean of top 25% (last element)
        self.assertEqual(tail_score, 800.0)
    
    def test_calculate_gini_penalty(self):
        """Test Gini penalty calculation."""
        volumes = np.array([100.0, 200.0, 400.0, 800.0])
        gini_penalty = calculate_gini_penalty(volumes, target_gini=0.3)
        # Should be absolute difference between calculated and target Gini
        self.assertGreaterEqual(gini_penalty, 0.0)
    
    def test_calculate_monotone_penalty(self):
        """Test monotone penalty calculation."""
        # Monotonic sequence
        orders = np.array([3, 5, 7, 10])
        penalty = calculate_monotone_penalty(orders)
        self.assertEqual(penalty, 0.0)
        
        # Non-monotonic sequence
        orders = np.array([3, 5, 2, 10])
        penalty = calculate_monotone_penalty(orders)
        self.assertGreater(penalty, 0.0)
    
    def test_calculate_smoothness_penalty(self):
        """Test smoothness penalty calculation."""
        # Smooth sequence
        volumes = np.array([100.0, 200.0, 400.0, 800.0])
        penalty = calculate_smoothness_penalty(volumes, max_change=0.5)
        # Should have some violations due to large relative changes
        self.assertGreaterEqual(penalty, 0.0)
        
        # Very smooth sequence
        volumes = np.array([100.0, 105.0, 110.0, 115.0])
        penalty = calculate_smoothness_penalty(volumes, max_change=0.5)
        self.assertEqual(penalty, 0.0)


class TestEvaluationKernels(unittest.TestCase):
    """Test evaluation kernel functions."""
    
    def test_evaluate_kernel_valid_params(self):
        """Test evaluation kernel with valid parameters."""
        params_array = np.array([0.01, 0.30, 0.1, 0.4, 0.2, 0.1, 0.2])  # Normalized params
        score, breakdown = evaluate_kernel(params_array)
        
        self.assertIsInstance(score, float)
        self.assertIsInstance(breakdown, np.ndarray)
        self.assertEqual(len(breakdown), 8)
    
    def test_evaluate_kernel_invalid_params(self):
        """Test evaluation kernel with invalid parameters."""
        # Invalid params where min_order > max_order
        params_array = np.array([0.01, 0.30, 0.8, 0.4, 0.2, 0.1, 0.2])
        score, breakdown = evaluate_kernel(params_array)
        
        self.assertEqual(score, -np.inf)
        np.testing.assert_array_equal(breakdown, np.zeros(8))
    
    def test_evaluate_params(self):
        """Test evaluate_params function."""
        params = Params(
            min_overlap=1.0,
            max_overlap=30.0,
            min_order=3,
            max_order=20,
            risk_factor=2.0,
            smoothing_factor=0.5,
            tail_weight=0.3
        )
        
        score, breakdown = evaluate_params(params)
        
        self.assertIsInstance(score, float)
        self.assertIsInstance(breakdown, ScoreBreakdown)
        self.assertEqual(breakdown.total_score, score)
    
    def test_create_schedule_from_params(self):
        """Test schedule creation from parameters."""
        params = Params(
            min_overlap=1.0,
            max_overlap=30.0,
            min_order=3,
            max_order=6,
            risk_factor=2.0,
            smoothing_factor=0.5,
            tail_weight=0.3
        )
        
        schedule = create_schedule_from_params(params)
        
        self.assertIsInstance(schedule, Schedule)
        self.assertEqual(schedule.num_levels, 4)  # max_order - min_order + 1
        self.assertEqual(len(schedule.orders), 4)
        self.assertEqual(len(schedule.volumes), 4)
        self.assertEqual(len(schedule.overlaps), 4)


class TestBatchEvaluation(unittest.TestCase):
    """Test batch evaluation functions."""
    
    def test_evaluate_params_batch(self):
        """Test batch parameter evaluation."""
        params_list = [
            Params(min_overlap=1.0, max_overlap=30.0, min_order=3, max_order=6),
            Params(min_overlap=5.0, max_overlap=25.0, min_order=4, max_order=8),
            Params(min_overlap=10.0, max_overlap=20.0, min_order=5, max_order=10)
        ]
        
        results = evaluate_params_batch(params_list)
        
        self.assertEqual(len(results), 3)
        for score, breakdown in results:
            self.assertIsInstance(score, float)
            self.assertIsInstance(breakdown, ScoreBreakdown)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions."""
    
    def test_empty_volumes(self):
        """Test with empty volumes array."""
        volumes = np.array([])
        max_score = calculate_max_score(volumes)
        variance_score = calculate_variance_score(volumes)
        tail_score = calculate_tail_score(volumes)
        
        self.assertEqual(max_score, 0.0)
        self.assertEqual(variance_score, 0.0)
        self.assertEqual(tail_score, 0.0)
    
    def test_single_element_volumes(self):
        """Test with single element volumes array."""
        volumes = np.array([100.0])
        max_score = calculate_max_score(volumes)
        variance_score = calculate_variance_score(volumes)
        tail_score = calculate_tail_score(volumes)
        
        self.assertEqual(max_score, 100.0)
        self.assertEqual(variance_score, 0.0)
        self.assertEqual(tail_score, 100.0)
    
    def test_zero_volumes(self):
        """Test with zero volumes."""
        volumes = np.array([0.0, 0.0, 0.0])
        max_score = calculate_max_score(volumes)
        variance_score = calculate_variance_score(volumes)
        
        self.assertEqual(max_score, 0.0)
        self.assertEqual(variance_score, 0.0)


if __name__ == '__main__':
    unittest.main()
