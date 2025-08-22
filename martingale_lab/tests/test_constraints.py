"""
Tests for constraints module.
"""
import unittest
import numpy as np
from ..core.types import Params, Schedule
from ..core.constraints import (
    ConstraintValidator, VolumeConstraints, OverlapConstraints, 
    OrderConstraints, Normalizer
)


class TestConstraintValidator(unittest.TestCase):
    """Test constraint validation."""
    
    def test_validate_params_valid(self):
        """Test valid parameter validation."""
        params = Params(
            min_overlap=1.0,
            max_overlap=30.0,
            min_order=3,
            max_order=20
        )
        self.assertTrue(ConstraintValidator.validate_params(params))
    
    def test_validate_params_invalid_overlap(self):
        """Test invalid overlap validation."""
        params = Params(
            min_overlap=50.0,  # Greater than max_overlap
            max_overlap=30.0,
            min_order=3,
            max_order=20
        )
        self.assertFalse(ConstraintValidator.validate_params(params))
    
    def test_validate_params_invalid_order(self):
        """Test invalid order validation."""
        params = Params(
            min_overlap=1.0,
            max_overlap=30.0,
            min_order=25,  # Greater than max_order
            max_order=20
        )
        self.assertFalse(ConstraintValidator.validate_params(params))
    
    def test_validate_schedule_valid(self):
        """Test valid schedule validation."""
        schedule = Schedule(
            orders=np.array([3, 5, 7, 10]),
            volumes=np.array([100, 200, 400, 800]),
            overlaps=np.array([1.0, 10.0, 20.0, 30.0])
        )
        self.assertTrue(ConstraintValidator.validate_schedule(schedule))
    
    def test_validate_schedule_invalid_volumes(self):
        """Test invalid volume validation."""
        schedule = Schedule(
            orders=np.array([3, 5, 7, 10]),
            volumes=np.array([100, -200, 400, 800]),  # Negative volume
            overlaps=np.array([1.0, 10.0, 20.0, 30.0])
        )
        self.assertFalse(ConstraintValidator.validate_schedule(schedule))


class TestVolumeConstraints(unittest.TestCase):
    """Test volume constraints."""
    
    def test_calculate_martingale_volumes(self):
        """Test martingale volume calculation."""
        volumes = VolumeConstraints.calculate_martingale_volumes(
            base_volume=100.0, multiplier=2.0, num_levels=4
        )
        expected = np.array([100.0, 200.0, 400.0, 800.0])
        np.testing.assert_array_almost_equal(volumes, expected)
    
    def test_normalize_volumes(self):
        """Test volume normalization."""
        volumes = np.array([100.0, 200.0, 400.0, 800.0])
        normalized = VolumeConstraints.normalize_volumes(volumes, target_total=1000.0)
        self.assertAlmostEqual(np.sum(normalized), 1000.0)
    
    def test_apply_volume_limits(self):
        """Test volume limit application."""
        volumes = np.array([50.0, 200.0, 500.0, 1000.0])
        limited = VolumeConstraints.apply_volume_limits(
            volumes, min_volume=100.0, max_volume=800.0
        )
        self.assertTrue(np.all(limited >= 100.0))
        self.assertTrue(np.all(limited <= 800.0))


class TestOverlapConstraints(unittest.TestCase):
    """Test overlap constraints."""
    
    def test_generate_overlaps_linear(self):
        """Test linear overlap generation."""
        overlaps = OverlapConstraints.generate_overlaps(
            min_overlap=1.0, max_overlap=30.0, num_levels=4, distribution='linear'
        )
        expected = np.array([1.0, 10.67, 20.33, 30.0])
        np.testing.assert_array_almost_equal(overlaps, expected, decimal=2)
    
    def test_generate_overlaps_exponential(self):
        """Test exponential overlap generation."""
        overlaps = OverlapConstraints.generate_overlaps(
            min_overlap=1.0, max_overlap=30.0, num_levels=4, distribution='exponential'
        )
        self.assertEqual(len(overlaps), 4)
        self.assertAlmostEqual(overlaps[0], 1.0)
        self.assertAlmostEqual(overlaps[-1], 30.0)
    
    def test_validate_overlap_sequence_valid(self):
        """Test valid overlap sequence validation."""
        overlaps = np.array([1.0, 10.0, 20.0, 30.0])
        self.assertTrue(OverlapConstraints.validate_overlap_sequence(overlaps))
    
    def test_validate_overlap_sequence_invalid(self):
        """Test invalid overlap sequence validation."""
        overlaps = np.array([1.0, 110.0, 20.0, 30.0])  # > 100%
        self.assertFalse(OverlapConstraints.validate_overlap_sequence(overlaps))


class TestOrderConstraints(unittest.TestCase):
    """Test order constraints."""
    
    def test_generate_orders_linear(self):
        """Test linear order generation."""
        orders = OrderConstraints.generate_orders(
            min_order=3, max_order=10, num_levels=4, distribution='linear'
        )
        expected = np.array([3, 5, 7, 10])
        np.testing.assert_array_equal(orders, expected)
    
    def test_generate_orders_exponential(self):
        """Test exponential order generation."""
        orders = OrderConstraints.generate_orders(
            min_order=3, max_order=10, num_levels=4, distribution='exponential'
        )
        self.assertEqual(len(orders), 4)
        self.assertEqual(orders[0], 3)
        self.assertEqual(orders[-1], 10)
    
    def test_validate_order_sequence_valid(self):
        """Test valid order sequence validation."""
        orders = np.array([3, 5, 7, 10])
        self.assertTrue(OrderConstraints.validate_order_sequence(orders))
    
    def test_validate_order_sequence_invalid(self):
        """Test invalid order sequence validation."""
        orders = np.array([3, 5, 2, 10])  # Non-monotonic
        self.assertFalse(OrderConstraints.validate_order_sequence(orders))


class TestNormalizer(unittest.TestCase):
    """Test parameter normalization."""
    
    def test_normalize_params(self):
        """Test parameter normalization."""
        params = Params(
            min_overlap=1.0,
            max_overlap=30.0,
            min_order=3,
            max_order=20,
            risk_factor=2.0,
            smoothing_factor=0.5,
            tail_weight=0.3
        )
        normalized = Normalizer.normalize_params(params)
        self.assertEqual(len(normalized), 7)
        self.assertTrue(np.all(normalized >= 0.0))
        self.assertTrue(np.all(normalized <= 1.0))
    
    def test_denormalize_params(self):
        """Test parameter denormalization."""
        original_params = Params(
            min_overlap=1.0,
            max_overlap=30.0,
            min_order=3,
            max_order=20,
            risk_factor=2.0,
            smoothing_factor=0.5,
            tail_weight=0.3
        )
        normalized = Normalizer.normalize_params(original_params)
        denormalized = Normalizer.denormalize_params(normalized)
        
        self.assertAlmostEqual(denormalized.min_overlap, original_params.min_overlap, places=1)
        self.assertAlmostEqual(denormalized.max_overlap, original_params.max_overlap, places=1)
        self.assertEqual(denormalized.min_order, original_params.min_order)
        self.assertEqual(denormalized.max_order, original_params.max_order)


if __name__ == '__main__':
    unittest.main()
