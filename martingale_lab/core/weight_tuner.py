"""
Automatic weight tuning for scoring function optimization.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import json


class WeightTuner:
    """
    Automatic tuning of scoring weights based on reference strategies.
    """
    
    def __init__(self):
        self.reference_strategies = []
        self.current_weights = {
            'alpha': 10.0,  # max_need weight
            'beta': 5.0,    # var_need weight
            'gamma': 100.0,  # tail weight
            'lambda_penalty': 1.0,  # general penalty weight
            'exit_ease_weight': 100.0,  # exit-ease bonus weight
            'pattern_penalty_weight': 1.0,  # pattern penalty weight
            'w_front': 3.0,
            'w_tailweak': 3.0,
            'w_slope': 2.0,
            'w_plateau': 2.0,
            'w_zigzag': 1.0,
            'w_cliff': 15.0,
            'w_stagnation': 2.0,
            'w_acceleration': 3.0,
        }
        self.tuning_history = []
    
    def add_reference_strategy(
        self,
        volumes: List[float],
        sharpe_ratio: float,
        label: str = "reference"
    ):
        """
        Add a reference strategy with known performance.
        
        Args:
            volumes: Volume percentages
            sharpe_ratio: Known Sharpe ratio
            label: Strategy label
        """
        self.reference_strategies.append({
            'volumes': np.array(volumes),
            'sharpe_ratio': sharpe_ratio,
            'label': label
        })
    
    def compute_strategy_features(self, volumes: np.ndarray) -> Dict[str, float]:
        """
        Extract features from a strategy.
        
        Args:
            volumes: Volume percentages
            
        Returns:
            Feature dictionary
        """
        from martingale_lab.core.repair import compute_m_from_v
        
        features = {}
        
        # Basic statistics
        features['mean'] = np.mean(volumes)
        features['std'] = np.std(volumes)
        features['max'] = np.max(volumes)
        features['min'] = np.min(volumes)
        
        # Quartile distribution
        Q1 = len(volumes) // 4
        Q4 = 3 * len(volumes) // 4
        features['q1_sum'] = np.sum(volumes[:Q1+1])
        features['q4_sum'] = np.sum(volumes[Q4:])
        features['q1_q4_ratio'] = features['q1_sum'] / (features['q4_sum'] + 1e-10)
        
        # Martingale metrics
        m = compute_m_from_v(volumes)
        if len(m) > 2:
            features['m2'] = m[2]
            features['m_mean'] = np.mean(m[2:])
            features['m_std'] = np.std(m[2:])
            features['m_max'] = np.max(m[2:])
        
        # Growth pattern
        growth_rates = []
        for i in range(1, len(volumes)):
            if volumes[i-1] > 0:
                growth_rates.append((volumes[i] - volumes[i-1]) / volumes[i-1])
        
        if growth_rates:
            features['avg_growth'] = np.mean(growth_rates)
            features['growth_consistency'] = 1.0 / (np.std(growth_rates) + 0.01)
        
        # Tail strength
        tail_idx = int(0.75 * len(volumes))
        features['tail_concentration'] = np.sum(volumes[tail_idx:]) / 100.0
        
        return features
    
    def similarity_score(
        self,
        features1: Dict[str, float],
        features2: Dict[str, float]
    ) -> float:
        """
        Compute similarity between two strategies.
        
        Args:
            features1: First strategy features
            features2: Second strategy features
            
        Returns:
            Similarity score (0-1, higher is more similar)
        """
        score = 0.0
        weight_sum = 0.0
        
        # Feature weights for similarity
        feature_weights = {
            'q1_sum': 2.0,
            'q4_sum': 2.0,
            'm2': 3.0,
            'm_mean': 2.0,
            'avg_growth': 2.0,
            'growth_consistency': 1.5,
            'tail_concentration': 2.5,
            'q1_q4_ratio': 1.5,
        }
        
        for feature, weight in feature_weights.items():
            if feature in features1 and feature in features2:
                val1 = features1[feature]
                val2 = features2[feature]
                
                # Normalized difference
                if abs(val1) + abs(val2) > 0:
                    diff = abs(val1 - val2) / (abs(val1) + abs(val2))
                    score += weight * (1.0 - diff)
                    weight_sum += weight
        
        if weight_sum > 0:
            return score / weight_sum
        return 0.0
    
    def auto_tune_weights(
        self,
        candidate_strategies: List[np.ndarray],
        candidate_scores: List[float],
        iterations: int = 10
    ) -> Dict[str, float]:
        """
        Automatically tune weights based on reference strategies.
        
        Args:
            candidate_strategies: List of candidate volume arrays
            candidate_scores: List of corresponding scores
            iterations: Number of tuning iterations
            
        Returns:
            Optimized weights
        """
        if not self.reference_strategies:
            return self.current_weights
        
        best_weights = self.current_weights.copy()
        best_correlation = -1.0
        
        for iteration in range(iterations):
            # Compute features for all strategies
            ref_features = [
                self.compute_strategy_features(ref['volumes'])
                for ref in self.reference_strategies
            ]
            
            cand_features = [
                self.compute_strategy_features(volumes)
                for volumes in candidate_strategies
            ]
            
            # Compute similarity scores
            similarities = []
            for cand_feat in cand_features:
                max_sim = 0.0
                for ref_feat in ref_features:
                    sim = self.similarity_score(cand_feat, ref_feat)
                    max_sim = max(max_sim, sim)
                similarities.append(max_sim)
            
            # Compute correlation between similarity and scores
            # We want high similarity to correlate with low scores
            correlation = np.corrcoef(similarities, candidate_scores)[0, 1]
            
            if correlation < best_correlation or best_correlation < 0:
                best_correlation = correlation
                best_weights = self.current_weights.copy()
            
            # Adjust weights based on correlation
            if correlation > 0:  # Bad: similar strategies have high scores
                # Increase exit-ease weight
                self.current_weights['exit_ease_weight'] *= 1.1
                # Decrease pattern penalties
                self.current_weights['w_plateau'] *= 0.9
                self.current_weights['w_zigzag'] *= 0.9
            else:  # Good: similar strategies have low scores
                # Fine-tune based on feature differences
                self._fine_tune_weights(ref_features, cand_features, candidate_scores)
            
            self.tuning_history.append({
                'iteration': iteration,
                'correlation': correlation,
                'weights': self.current_weights.copy()
            })
        
        return best_weights
    
    def _fine_tune_weights(
        self,
        ref_features: List[Dict],
        cand_features: List[Dict],
        scores: List[float]
    ):
        """
        Fine-tune weights based on feature analysis.
        """
        # Find best candidate (lowest score)
        best_idx = np.argmin(scores)
        best_features = cand_features[best_idx]
        
        # Compare with reference features
        for ref_feat in ref_features:
            # If Q1 is higher in candidate, increase front penalty
            if best_features.get('q1_sum', 0) > ref_feat.get('q1_sum', 0):
                self.current_weights['w_front'] *= 1.05
            
            # If tail is weaker, increase tail penalty
            if best_features.get('q4_sum', 0) < ref_feat.get('q4_sum', 0):
                self.current_weights['w_tailweak'] *= 1.05
            
            # If growth is less consistent, increase slope penalty
            if best_features.get('growth_consistency', 0) < ref_feat.get('growth_consistency', 0):
                self.current_weights['w_slope'] *= 1.05
    
    def suggest_weights_for_market(
        self,
        market_volatility: float,
        market_trend: float
    ) -> Dict[str, float]:
        """
        Suggest weights based on market conditions.
        
        Args:
            market_volatility: Volatility level (0-1)
            market_trend: Trend direction (-1 to 1)
            
        Returns:
            Suggested weights
        """
        weights = self.current_weights.copy()
        
        # High volatility: emphasize exit-ease and tail strength
        if market_volatility > 0.7:
            weights['exit_ease_weight'] *= 1.5
            weights['w_tailweak'] *= 1.3
            weights['w_cliff'] *= 1.5  # Avoid sudden jumps
        
        # Strong downtrend: need strong recovery
        if market_trend < -0.5:
            weights['w_tailweak'] *= 1.4
            weights['gamma'] *= 1.2  # Emphasize tail in scoring
            weights['w_stagnation'] *= 1.3  # Avoid stagnation
        
        # Uptrend: can be more conservative
        if market_trend > 0.5:
            weights['w_front'] *= 0.8  # Less penalty on front
            weights['w_plateau'] *= 0.8  # Plateaus more acceptable
        
        return weights
    
    def export_weights(self, filepath: str):
        """
        Export current weights to file.
        
        Args:
            filepath: Path to save weights
        """
        with open(filepath, 'w') as f:
            json.dump({
                'weights': self.current_weights,
                'tuning_history': self.tuning_history,
                'reference_count': len(self.reference_strategies)
            }, f, indent=2)
    
    def import_weights(self, filepath: str):
        """
        Import weights from file.
        
        Args:
            filepath: Path to load weights from
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
            self.current_weights = data['weights']
            self.tuning_history = data.get('tuning_history', [])
