"""
Auto-tuning system for configuration optimization.
Uses historical performance data to suggest better configurations.
"""
from __future__ import annotations

import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
import json
import logging

from martingale_lab.core.config_classes import (
    EvaluationConfig, ConfigPresets,
    HardConstraintConfig, PenaltyWeightConfig, ScoringConfig
)
from martingale_lab.core.config_adapter import config_to_flat_dict
from martingale_lab.storage.config_store import ConfigStore

logger = logging.getLogger(__name__)


@dataclass
class TuningStrategy:
    """Strategy for auto-tuning."""
    name: str
    exploration_rate: float = 0.2  # Probability of exploration vs exploitation
    mutation_rate: float = 0.1  # Probability of parameter mutation
    crossover_rate: float = 0.3  # Probability of combining configs
    elite_size: int = 5  # Number of best configs to preserve
    population_size: int = 20  # Size of config population
    
    # Parameter ranges for mutations
    param_ranges: Dict[str, Tuple[float, float]] = field(default_factory=dict)


class ConfigAutoTuner:
    """
    Automatic configuration tuner using evolutionary strategies.
    """
    
    def __init__(self, config_store: ConfigStore):
        """Initialize auto-tuner with config store."""
        self.config_store = config_store
        self.rng = np.random.default_rng()
        
        # Default parameter ranges for mutations
        self.param_ranges = {
            # Constraints
            'm2_min': (0.05, 0.30),
            'm2_max': (0.30, 1.20),
            'm_min': (0.03, 0.20),
            'm_max': (0.20, 1.50),
            'slope_cap': (0.10, 0.40),
            'q1_cap': (15.0, 30.0),
            'tail_floor': (25.0, 40.0),
            
            # Penalties
            'w_fixed': (1.0, 5.0),
            'w_second': (1.0, 5.0),
            'w_band': (0.5, 3.0),
            'w_front': (1.0, 5.0),
            'w_tailweak': (1.0, 4.0),
            
            # Scoring
            'alpha': (0.3, 0.7),
            'beta': (0.2, 0.5),
            'gamma': (0.1, 0.4),
            'lambda_penalty': (0.05, 0.20),
            
            # Generation
            'wave_amp_min': (0.02, 0.10),
            'wave_amp_max': (0.20, 0.50),
            'anchors': (3, 10),
            'blocks': (2, 6)
        }
    
    def suggest_next_config(
        self,
        strategy: TuningStrategy,
        base_config: Optional[EvaluationConfig] = None,
        performance_target: Optional[float] = None
    ) -> EvaluationConfig:
        """
        Suggest next configuration to try.
        
        Args:
            strategy: Tuning strategy to use
            base_config: Base configuration to improve upon
            performance_target: Target score to beat
            
        Returns:
            Suggested configuration
        """
        # Get best historical configs
        best_configs = self.config_store.get_best_configs(limit=strategy.elite_size)
        
        if not best_configs and not base_config:
            # No history, start with a preset
            logger.info("No config history, starting with exploration preset")
            return ConfigPresets.exploration()
        
        # Decide action based on exploration rate
        if self.rng.random() < strategy.exploration_rate:
            # Exploration: try something new
            if base_config:
                return self._mutate_config(base_config, strategy)
            else:
                return self._random_config()
        else:
            # Exploitation: improve on best known
            if best_configs:
                best_hash = best_configs[0][0]
                best_config = self.config_store.load_config(best_hash)
                
                if best_config:
                    # Decide between mutation and crossover
                    if len(best_configs) > 1 and self.rng.random() < strategy.crossover_rate:
                        # Crossover two good configs
                        second_hash = best_configs[1][0]
                        second_config = self.config_store.load_config(second_hash)
                        if second_config:
                            return self._crossover_configs(best_config, second_config)
                    
                    # Mutate best config
                    return self._mutate_config(best_config, strategy)
            
            # Fallback to base config mutation
            if base_config:
                return self._mutate_config(base_config, strategy)
            
            return ConfigPresets.production()
    
    def _mutate_config(
        self, 
        config: EvaluationConfig, 
        strategy: TuningStrategy
    ) -> EvaluationConfig:
        """
        Mutate a configuration by randomly changing parameters.
        """
        import copy
        mutated = copy.deepcopy(config)
        
        # Determine number of mutations
        n_mutations = self.rng.poisson(2) + 1  # At least 1 mutation
        
        for _ in range(n_mutations):
            if self.rng.random() > strategy.mutation_rate:
                continue
            
            # Choose parameter category
            category = self.rng.choice(['constraints', 'penalties', 'scoring', 'generation'])
            
            if category == 'constraints':
                self._mutate_constraints(mutated.constraints)
            elif category == 'penalties':
                self._mutate_penalties(mutated.penalties)
            elif category == 'scoring':
                self._mutate_scoring(mutated.scoring)
            elif category == 'generation':
                self._mutate_generation(mutated.generation)
        
        return mutated
    
    def _mutate_constraints(self, constraints: HardConstraintConfig):
        """Mutate constraint parameters."""
        param = self.rng.choice(['m2_min', 'm2_max', 'slope_cap', 'q1_cap', 'tail_floor'])
        
        if param in self.param_ranges:
            min_val, max_val = self.param_ranges[param]
            current = getattr(constraints, param)
            
            # Gaussian mutation around current value
            std = (max_val - min_val) * 0.1
            new_val = self.rng.normal(current, std)
            new_val = np.clip(new_val, min_val, max_val)
            
            setattr(constraints, param, float(new_val))
            
            # Ensure m2_min < m2_max
            if param == 'm2_min' and constraints.m2_min >= constraints.m2_max:
                constraints.m2_min = constraints.m2_max * 0.8
            elif param == 'm2_max' and constraints.m2_max <= constraints.m2_min:
                constraints.m2_max = constraints.m2_min * 1.2
    
    def _mutate_penalties(self, penalties: PenaltyWeightConfig):
        """Mutate penalty weights."""
        param = self.rng.choice(['w_fixed', 'w_second', 'w_band', 'w_front', 'w_tailweak'])
        
        if param in self.param_ranges:
            min_val, max_val = self.param_ranges[param]
            current = getattr(penalties, param)
            
            # Log-normal mutation for weights
            log_current = np.log(current + 0.1)
            log_new = self.rng.normal(log_current, 0.2)
            new_val = np.exp(log_new) - 0.1
            new_val = np.clip(new_val, min_val, max_val)
            
            setattr(penalties, param, float(new_val))
    
    def _mutate_scoring(self, scoring: ScoringConfig):
        """Mutate scoring weights."""
        # Mutate one weight
        param = self.rng.choice(['alpha', 'beta', 'gamma', 'lambda_penalty'])
        
        if param in self.param_ranges:
            min_val, max_val = self.param_ranges[param]
            current = getattr(scoring, param)
            
            # Small perturbation
            delta = self.rng.normal(0, 0.05)
            new_val = current + delta
            new_val = np.clip(new_val, min_val, max_val)
            
            setattr(scoring, param, float(new_val))
            
            # Renormalize alpha, beta, gamma if needed
            if param in ['alpha', 'beta', 'gamma']:
                total = scoring.alpha + scoring.beta + scoring.gamma
                if total > 0:
                    scoring.alpha /= total
                    scoring.beta /= total
                    scoring.gamma /= total
    
    def _mutate_generation(self, generation):
        """Mutate generation parameters."""
        param = self.rng.choice(['wave_amp_min', 'wave_amp_max', 'anchors', 'blocks'])
        
        if param in self.param_ranges:
            min_val, max_val = self.param_ranges[param]
            current = getattr(generation, param)
            
            if param in ['anchors', 'blocks']:
                # Integer mutation
                delta = self.rng.choice([-1, 0, 1])
                new_val = current + delta
                new_val = int(np.clip(new_val, min_val, max_val))
            else:
                # Float mutation
                std = (max_val - min_val) * 0.1
                new_val = self.rng.normal(current, std)
                new_val = float(np.clip(new_val, min_val, max_val))
            
            setattr(generation, param, new_val)
            
            # Ensure wave_amp_min < wave_amp_max
            if param == 'wave_amp_min' and generation.wave_amp_min >= generation.wave_amp_max:
                generation.wave_amp_min = generation.wave_amp_max * 0.8
            elif param == 'wave_amp_max' and generation.wave_amp_max <= generation.wave_amp_min:
                generation.wave_amp_max = generation.wave_amp_min * 1.2
    
    def _crossover_configs(
        self,
        config1: EvaluationConfig,
        config2: EvaluationConfig
    ) -> EvaluationConfig:
        """
        Create offspring by combining two configurations.
        """
        import copy
        offspring = copy.deepcopy(config1)
        
        # Randomly inherit components from each parent
        if self.rng.random() < 0.5:
            offspring.constraints = copy.deepcopy(config2.constraints)
        
        if self.rng.random() < 0.5:
            offspring.penalties = copy.deepcopy(config2.penalties)
        
        if self.rng.random() < 0.5:
            offspring.scoring = copy.deepcopy(config2.scoring)
        
        if self.rng.random() < 0.5:
            offspring.generation = copy.deepcopy(config2.generation)
        
        # Small chance of additional mutation
        if self.rng.random() < 0.1:
            strategy = TuningStrategy(name="crossover_mutation", mutation_rate=0.3)
            offspring = self._mutate_config(offspring, strategy)
        
        return offspring
    
    def _random_config(self) -> EvaluationConfig:
        """Generate a random configuration within reasonable bounds."""
        config = EvaluationConfig()
        
        # Random constraints
        config.constraints.m2_min = float(self.rng.uniform(0.05, 0.25))
        config.constraints.m2_max = float(self.rng.uniform(0.40, 1.00))
        config.constraints.slope_cap = float(self.rng.uniform(0.15, 0.35))
        config.constraints.q1_cap = float(self.rng.uniform(18.0, 28.0))
        config.constraints.tail_floor = float(self.rng.uniform(28.0, 38.0))
        
        # Random penalties
        config.penalties.w_fixed = float(self.rng.uniform(1.5, 4.0))
        config.penalties.w_second = float(self.rng.uniform(1.5, 4.0))
        config.penalties.w_band = float(self.rng.uniform(1.0, 3.0))
        config.penalties.w_front = float(self.rng.uniform(1.5, 4.0))
        
        # Random scoring (normalized)
        weights = self.rng.dirichlet([2, 2, 2])  # Dirichlet ensures sum=1
        config.scoring.alpha = float(weights[0])
        config.scoring.beta = float(weights[1])
        config.scoring.gamma = float(weights[2])
        config.scoring.lambda_penalty = float(self.rng.uniform(0.05, 0.15))
        
        return config
    
    def analyze_sensitivity(
        self,
        base_config: EvaluationConfig,
        n_samples: int = 100
    ) -> Dict[str, float]:
        """
        Analyze parameter sensitivity around a configuration.
        
        Returns:
            Dictionary of parameter -> sensitivity score
        """
        sensitivities = {}
        
        # Sample variations and check similar configs
        for param_name in self.param_ranges:
            variations = []
            
            for _ in range(n_samples // len(self.param_ranges)):
                # Create small variation
                varied = self._mutate_single_param(base_config, param_name)
                
                # Find similar configs in history
                similar = self.config_store.find_similar_configs(varied, threshold=0.95)
                
                if similar:
                    # Get performance of similar configs
                    for config_hash, similarity in similar[:3]:
                        stats = self.config_store.get_config_stats(config_hash)
                        if stats.get('best_score'):
                            variations.append(stats['best_score'])
            
            if variations:
                # Calculate variance as sensitivity measure
                sensitivities[param_name] = float(np.std(variations))
        
        return sensitivities
    
    def _mutate_single_param(
        self,
        config: EvaluationConfig,
        param_name: str
    ) -> EvaluationConfig:
        """Mutate a single parameter."""
        import copy
        mutated = copy.deepcopy(config)
        
        if param_name in self.param_ranges:
            min_val, max_val = self.param_ranges[param_name]
            
            # Find parameter in config
            for component in [mutated.constraints, mutated.penalties, 
                            mutated.scoring, mutated.generation]:
                if hasattr(component, param_name):
                    current = getattr(component, param_name)
                    
                    # Small perturbation
                    if isinstance(current, int):
                        delta = self.rng.choice([-1, 1])
                        new_val = int(np.clip(current + delta, min_val, max_val))
                    else:
                        std = (max_val - min_val) * 0.05
                        new_val = float(np.clip(
                            self.rng.normal(current, std), 
                            min_val, max_val
                        ))
                    
                    setattr(component, param_name, new_val)
                    break
        
        return mutated
    
    def recommend_tuning_strategy(
        self,
        performance_history: List[float]
    ) -> TuningStrategy:
        """
        Recommend tuning strategy based on performance history.
        """
        if not performance_history:
            # No history, explore aggressively
            return TuningStrategy(
                name="initial_exploration",
                exploration_rate=0.5,
                mutation_rate=0.2,
                crossover_rate=0.1
            )
        
        # Check convergence
        if len(performance_history) > 10:
            recent = performance_history[-10:]
            improvement = (recent[0] - recent[-1]) / recent[0] if recent[0] > 0 else 0
            
            if improvement < 0.01:  # Less than 1% improvement
                # Stagnant, increase exploration
                return TuningStrategy(
                    name="break_stagnation",
                    exploration_rate=0.4,
                    mutation_rate=0.3,
                    crossover_rate=0.2
                )
            elif improvement > 0.1:  # More than 10% improvement
                # Good progress, exploit more
                return TuningStrategy(
                    name="exploit_momentum",
                    exploration_rate=0.1,
                    mutation_rate=0.1,
                    crossover_rate=0.4
                )
        
        # Default balanced strategy
        return TuningStrategy(
            name="balanced",
            exploration_rate=0.2,
            mutation_rate=0.15,
            crossover_rate=0.3
        )