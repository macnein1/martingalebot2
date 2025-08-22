"""
Tests for objective functions module.
"""
import unittest
import numpy as np
from ..core.types import ScoreBreakdown
from ..optimizer.objective_functions import (
    LinearObjective, ExponentialObjective, MultiObjective, 
    AdaptiveObjective, RobustObjective, create_objective, 
    get_predefined_objective, PREDEFINED_OBJECTIVES
)


class TestLinearObjective(unittest.TestCase):
    """Test linear objective function."""
    
    def setUp(self):
        """Set up test data."""
        self.breakdown = ScoreBreakdown(
            total_score=1.5,
            max_score=2.0,
            variance_score=1.0,
            tail_score=0.5,
            gini_penalty=0.1,
            entropy_penalty=0.05,
            monotone_penalty=0.2,
            smoothness_penalty=0.15
        )
    
    def test_linear_objective_default_weights(self):
        """Test linear objective with default weights."""
        objective = LinearObjective()
        score = objective(self.breakdown)
        
        # Expected: 0.4*2.0 + 0.3*1.0 + 0.3*0.5 - (0.1 + 0.05 + 0.2 + 0.15)
        # = 0.8 + 0.3 + 0.15 - 0.5 = 0.75
        expected = 0.4 * 2.0 + 0.3 * 1.0 + 0.3 * 0.5 - 0.5
        self.assertAlmostEqual(score, expected)
    
    def test_linear_objective_custom_weights(self):
        """Test linear objective with custom weights."""
        objective = LinearObjective(alpha=0.5, beta=0.3, gamma=0.2)
        score = objective(self.breakdown)
        
        expected = 0.5 * 2.0 + 0.3 * 1.0 + 0.2 * 0.5 - 0.5
        self.assertAlmostEqual(score, expected)
    
    def test_linear_objective_custom_penalty_weights(self):
        """Test linear objective with custom penalty weights."""
        penalty_weights = {
            'gini': 2.0,
            'entropy': 1.0,
            'monotone': 3.0,
            'smoothness': 2.0
        }
        objective = LinearObjective(penalty_weights=penalty_weights)
        score = objective(self.breakdown)
        
        expected_penalty = (2.0 * 0.1 + 1.0 * 0.05 + 3.0 * 0.2 + 2.0 * 0.15)
        expected = 0.4 * 2.0 + 0.3 * 1.0 + 0.3 * 0.5 - expected_penalty
        self.assertAlmostEqual(score, expected)


class TestExponentialObjective(unittest.TestCase):
    """Test exponential objective function."""
    
    def setUp(self):
        """Set up test data."""
        self.breakdown = ScoreBreakdown(
            total_score=1.5,
            max_score=2.0,
            variance_score=1.0,
            tail_score=0.5,
            gini_penalty=0.1,
            entropy_penalty=0.05,
            monotone_penalty=0.2,
            smoothness_penalty=0.15
        )
    
    def test_exponential_objective(self):
        """Test exponential objective function."""
        objective = ExponentialObjective(base=2.0)
        score = objective(self.breakdown)
        
        # Expected: 0.4*2^2.0 + 0.3*2^1.0 + 0.3*2^0.5 - 0.5
        # = 0.4*4 + 0.3*2 + 0.3*1.414 - 0.5
        expected = 0.4 * 4 + 0.3 * 2 + 0.3 * 1.414 - 0.5
        self.assertAlmostEqual(score, expected, places=3)


class TestMultiObjective(unittest.TestCase):
    """Test multi-objective function."""
    
    def setUp(self):
        """Set up test data."""
        self.breakdown = ScoreBreakdown(
            total_score=1.5,
            max_score=2.0,
            variance_score=1.0,
            tail_score=0.5,
            gini_penalty=0.1,
            entropy_penalty=0.05,
            monotone_penalty=0.2,
            smoothness_penalty=0.15
        )
    
    def test_multi_objective(self):
        """Test multi-objective function."""
        obj1 = LinearObjective(alpha=0.5, beta=0.3, gamma=0.2)
        obj2 = ExponentialObjective(base=2.0)
        
        multi_obj = MultiObjective([obj1, obj2])
        values = multi_obj(self.breakdown)
        
        self.assertEqual(len(values), 2)
        self.assertIsInstance(values[0], float)
        self.assertIsInstance(values[1], float)
    
    def test_multi_objective_weighted_sum(self):
        """Test multi-objective weighted sum."""
        obj1 = LinearObjective(alpha=0.5, beta=0.3, gamma=0.2)
        obj2 = ExponentialObjective(base=2.0)
        
        multi_obj = MultiObjective([obj1, obj2], weights=[0.6, 0.4])
        weighted_sum = multi_obj.weighted_sum(self.breakdown)
        
        individual_values = multi_obj(self.breakdown)
        expected = 0.6 * individual_values[0] + 0.4 * individual_values[1]
        self.assertAlmostEqual(weighted_sum, expected)


class TestAdaptiveObjective(unittest.TestCase):
    """Test adaptive objective function."""
    
    def setUp(self):
        """Set up test data."""
        self.breakdown = ScoreBreakdown(
            total_score=1.5,
            max_score=2.0,
            variance_score=1.0,
            tail_score=0.5,
            gini_penalty=0.1,
            entropy_penalty=0.05,
            monotone_penalty=0.2,
            smoothness_penalty=0.15
        )
    
    def test_adaptive_objective_initial(self):
        """Test adaptive objective initial state."""
        objective = AdaptiveObjective()
        score = objective(self.breakdown)
        
        # Should use base weights initially
        expected = 0.4 * 2.0 + 0.3 * 1.0 + 0.3 * 0.5 - 0.5
        self.assertAlmostEqual(score, expected)
    
    def test_adaptive_objective_adaptation(self):
        """Test adaptive objective weight adaptation."""
        objective = AdaptiveObjective()
        
        # Add some performance data
        for i in range(20):
            breakdown = ScoreBreakdown(
                total_score=1.0 + i * 0.1,
                max_score=2.0 + i * 0.05,
                variance_score=1.0 + i * 0.02,
                tail_score=0.5 + i * 0.01,
                gini_penalty=0.1,
                entropy_penalty=0.05,
                monotone_penalty=0.2,
                smoothness_penalty=0.15
            )
            objective.add_performance(breakdown, target_score=1.0 + i * 0.1)
        
        # Adapt weights
        objective.adapt_weights()
        
        # Weights should have changed
        score = objective(self.breakdown)
        self.assertIsInstance(score, float)


class TestRobustObjective(unittest.TestCase):
    """Test robust objective function."""
    
    def setUp(self):
        """Set up test data."""
        self.breakdown = ScoreBreakdown(
            total_score=1.5,
            max_score=2.0,
            variance_score=1.0,
            tail_score=0.5,
            gini_penalty=0.1,
            entropy_penalty=0.05,
            monotone_penalty=0.2,
            smoothness_penalty=0.15
        )
    
    def test_robust_objective(self):
        """Test robust objective function."""
        objective = RobustObjective(robustness_factor=0.2)
        score = objective(self.breakdown)
        
        # Expected: base_objective - (penalty_total + robustness_penalty)
        base_objective = 0.4 * 2.0 + 0.3 * 1.0 + 0.3 * 0.5
        penalty_total = 0.1 + 0.05 + 0.2 + 0.15
        robustness_penalty = 0.2 * 1.0  # 0.2 * variance_score
        expected = base_objective - (penalty_total + robustness_penalty)
        
        self.assertAlmostEqual(score, expected)


class TestFactoryFunctions(unittest.TestCase):
    """Test factory functions."""
    
    def test_create_objective_linear(self):
        """Test creating linear objective."""
        objective = create_objective('linear', alpha=0.5, beta=0.3, gamma=0.2)
        self.assertIsInstance(objective, LinearObjective)
        self.assertEqual(objective.alpha, 0.5)
        self.assertEqual(objective.beta, 0.3)
        self.assertEqual(objective.gamma, 0.2)
    
    def test_create_objective_exponential(self):
        """Test creating exponential objective."""
        objective = create_objective('exponential', base=3.0)
        self.assertIsInstance(objective, ExponentialObjective)
        self.assertEqual(objective.base, 3.0)
    
    def test_create_objective_adaptive(self):
        """Test creating adaptive objective."""
        objective = create_objective('adaptive', adaptation_rate=0.2)
        self.assertIsInstance(objective, AdaptiveObjective)
        self.assertEqual(objective.adaptation_rate, 0.2)
    
    def test_create_objective_robust(self):
        """Test creating robust objective."""
        objective = create_objective('robust', robustness_factor=0.3)
        self.assertIsInstance(objective, RobustObjective)
        self.assertEqual(objective.robustness_factor, 0.3)
    
    def test_create_objective_invalid_type(self):
        """Test creating objective with invalid type."""
        with self.assertRaises(ValueError):
            create_objective('invalid_type')
    
    def test_get_predefined_objective_balanced(self):
        """Test getting predefined balanced objective."""
        objective = get_predefined_objective('balanced')
        self.assertIsInstance(objective, LinearObjective)
        self.assertEqual(objective.alpha, 0.4)
        self.assertEqual(objective.beta, 0.3)
        self.assertEqual(objective.gamma, 0.3)
    
    def test_get_predefined_objective_aggressive(self):
        """Test getting predefined aggressive objective."""
        objective = get_predefined_objective('aggressive')
        self.assertIsInstance(objective, LinearObjective)
        self.assertEqual(objective.alpha, 0.6)
        self.assertEqual(objective.beta, 0.2)
        self.assertEqual(objective.gamma, 0.2)
    
    def test_get_predefined_objective_conservative(self):
        """Test getting predefined conservative objective."""
        objective = get_predefined_objective('conservative')
        self.assertIsInstance(objective, LinearObjective)
        self.assertEqual(objective.alpha, 0.2)
        self.assertEqual(objective.beta, 0.4)
        self.assertEqual(objective.gamma, 0.4)
    
    def test_get_predefined_objective_robust(self):
        """Test getting predefined robust objective."""
        objective = get_predefined_objective('robust')
        self.assertIsInstance(objective, RobustObjective)
        self.assertEqual(objective.robustness_factor, 0.2)
    
    def test_get_predefined_objective_invalid(self):
        """Test getting predefined objective with invalid name."""
        with self.assertRaises(ValueError):
            get_predefined_objective('invalid_name')


class TestPredefinedObjectives(unittest.TestCase):
    """Test predefined objective configurations."""
    
    def test_predefined_objectives_structure(self):
        """Test structure of predefined objectives."""
        expected_types = ['balanced', 'aggressive', 'conservative', 'robust']
        
        for obj_type in expected_types:
            self.assertIn(obj_type, PREDEFINED_OBJECTIVES)
            config = PREDEFINED_OBJECTIVES[obj_type]
            self.assertIn('type', config)
    
    def test_predefined_objectives_creation(self):
        """Test creation of all predefined objectives."""
        for name in PREDEFINED_OBJECTIVES:
            objective = get_predefined_objective(name)
            self.assertIsNotNone(objective)
            
            # Test that it can be called
            breakdown = ScoreBreakdown(
                total_score=1.0,
                max_score=2.0,
                variance_score=1.0,
                tail_score=0.5
            )
            score = objective(breakdown)
            self.assertIsInstance(score, float)


if __name__ == '__main__':
    unittest.main()
