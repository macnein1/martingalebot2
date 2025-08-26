"""
Smart Initial Generation for Martingale Strategies

Uses historical successful strategies, MCMC sampling, and genetic algorithms
to generate better initial populations.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
import sqlite3
import json
from numba import njit


class SmartInitializer:
    """
    Intelligent initial strategy generation using multiple techniques.
    """
    
    def __init__(self, history_db: Optional[str] = None):
        """
        Initialize with optional historical database.
        
        Args:
            history_db: Path to database with historical successful strategies
        """
        self.history_db = history_db
        self.successful_strategies = []
        self.strategy_features = []
        
        if history_db:
            self._load_historical_strategies()
    
    def _load_historical_strategies(self, score_threshold: float = 2000.0):
        """
        Load successful strategies from historical database.
        
        Args:
            score_threshold: Maximum score to consider as successful
        """
        try:
            conn = sqlite3.connect(self.history_db)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT payload_json, diagnostics_json, score
                FROM results
                WHERE score < ? AND score > 0
                ORDER BY score ASC
                LIMIT 100
            """, (score_threshold,))
            
            for payload_json, diag_json, score in cursor.fetchall():
                payload = json.loads(payload_json)
                diagnostics = json.loads(diag_json)
                
                if 'schedule' in payload and 'volume_pct' in payload['schedule']:
                    volumes = payload['schedule']['volume_pct']
                    self.successful_strategies.append({
                        'volumes': volumes,
                        'score': score,
                        'q1_share': diagnostics.get('q1_share', 0),
                        'q4_share': diagnostics.get('q4_share', 0),
                        'm2': diagnostics.get('m2', 0),
                        'sortino': diagnostics.get('sortino_ratio', 0)
                    })
            
            conn.close()
            
            # Extract features for similarity matching
            self._extract_features()
            
        except Exception as e:
            print(f"Warning: Could not load historical strategies: {e}")
    
    def _extract_features(self):
        """Extract statistical features from successful strategies."""
        for strategy in self.successful_strategies:
            volumes = np.array(strategy['volumes'])
            
            # Growth pattern features
            if len(volumes) > 1:
                growth_rates = []
                for i in range(1, len(volumes)):
                    if volumes[i-1] > 0:
                        growth_rates.append((volumes[i] - volumes[i-1]) / volumes[i-1])
                
                features = {
                    'mean_growth': np.mean(growth_rates) if growth_rates else 0,
                    'std_growth': np.std(growth_rates) if growth_rates else 0,
                    'q1_share': strategy['q1_share'],
                    'q4_share': strategy['q4_share'],
                    'm2': strategy['m2']
                }
                self.strategy_features.append(features)
    
    def generate_from_historical(
        self,
        num_orders: int,
        n_samples: int = 10,
        mutation_rate: float = 0.1
    ) -> List[np.ndarray]:
        """
        Generate initial strategies based on historical successful ones.
        
        Args:
            num_orders: Number of orders in strategy
            n_samples: Number of strategies to generate
            mutation_rate: Amount of random variation to add
            
        Returns:
            List of volume arrays
        """
        if not self.successful_strategies:
            return []
        
        generated = []
        
        for _ in range(n_samples):
            # Select random successful strategy as template
            template_idx = np.random.randint(0, len(self.successful_strategies))
            template = self.successful_strategies[template_idx]['volumes']
            
            # Adapt to target number of orders
            new_volumes = self._adapt_strategy_length(template, num_orders)
            
            # Add controlled mutation
            new_volumes = self._mutate_strategy(new_volumes, mutation_rate)
            
            # Ensure sum = 100
            new_volumes = new_volumes / np.sum(new_volumes) * 100.0
            
            generated.append(new_volumes)
        
        return generated
    
    def _adapt_strategy_length(
        self,
        template: List[float],
        target_length: int
    ) -> np.ndarray:
        """
        Adapt strategy from one length to another using interpolation.
        
        Args:
            template: Original strategy volumes
            target_length: Desired number of orders
            
        Returns:
            Adapted volume array
        """
        template = np.array(template)
        current_length = len(template)
        
        if current_length == target_length:
            return template.copy()
        
        # Use linear interpolation to resize
        x_old = np.linspace(0, 1, current_length)
        x_new = np.linspace(0, 1, target_length)
        
        # Interpolate volumes
        new_volumes = np.interp(x_new, x_old, template)
        
        return new_volumes
    
    def _mutate_strategy(
        self,
        volumes: np.ndarray,
        mutation_rate: float
    ) -> np.ndarray:
        """
        Add controlled random mutations to strategy.
        
        Args:
            volumes: Volume array
            mutation_rate: Strength of mutations (0-1)
            
        Returns:
            Mutated volume array
        """
        mutated = volumes.copy()
        
        for i in range(len(mutated)):
            if np.random.random() < 0.5:  # 50% chance to mutate each order
                # Add Gaussian noise proportional to current value
                noise = np.random.normal(0, mutation_rate * mutated[i])
                mutated[i] = max(0.1, mutated[i] + noise)  # Keep positive
        
        return mutated
    
    def generate_mcmc_samples(
        self,
        num_orders: int,
        n_samples: int = 10,
        n_steps: int = 100,
        target_q4: float = 55.0,
        target_m2: float = 0.12
    ) -> List[np.ndarray]:
        """
        Generate strategies using Markov Chain Monte Carlo sampling.
        
        Args:
            num_orders: Number of orders
            n_samples: Number of samples to generate
            n_steps: MCMC steps per sample
            target_q4: Target Q4 share percentage
            target_m2: Target m[2] value
            
        Returns:
            List of volume arrays
        """
        samples = []
        
        for _ in range(n_samples):
            # Start with geometric progression
            volumes = self._geometric_start(num_orders, m2=target_m2)
            
            # MCMC refinement
            for step in range(n_steps):
                # Propose new state
                proposed = self._mcmc_proposal(volumes)
                
                # Calculate acceptance probability
                current_score = self._mcmc_score(volumes, target_q4, target_m2)
                proposed_score = self._mcmc_score(proposed, target_q4, target_m2)
                
                # Metropolis-Hastings acceptance
                accept_prob = min(1.0, np.exp(-(proposed_score - current_score) / 0.1))
                
                if np.random.random() < accept_prob:
                    volumes = proposed
            
            # Ensure sum = 100
            volumes = volumes / np.sum(volumes) * 100.0
            samples.append(volumes)
        
        return samples
    
    def _geometric_start(
        self,
        num_orders: int,
        v0: float = 1.0,
        m2: float = 0.12
    ) -> np.ndarray:
        """
        Create geometric progression starting point.
        
        Args:
            num_orders: Number of orders
            v0: First volume
            m2: Target m[2] ratio
            
        Returns:
            Volume array
        """
        volumes = np.zeros(num_orders)
        volumes[0] = v0
        
        if num_orders > 1:
            volumes[1] = v0 * 1.1  # v1 slightly higher
        
        if num_orders > 2:
            volumes[2] = volumes[1] * (1 + m2)
            
            # Continue with similar growth
            for i in range(3, num_orders):
                growth = 1.0 + m2 * (1 + 0.02 * (i - 2))  # Slightly increasing growth
                volumes[i] = volumes[i-1] * growth
        
        return volumes
    
    def _mcmc_proposal(self, volumes: np.ndarray) -> np.ndarray:
        """
        Generate MCMC proposal by perturbing current state.
        
        Args:
            volumes: Current volume array
            
        Returns:
            Proposed volume array
        """
        proposed = volumes.copy()
        
        # Random walk in log space for better exploration
        log_volumes = np.log(proposed + 0.01)
        
        # Perturb 2-3 random positions
        n_changes = np.random.randint(2, 4)
        positions = np.random.choice(len(proposed), n_changes, replace=False)
        
        for pos in positions:
            log_volumes[pos] += np.random.normal(0, 0.05)
        
        # Convert back from log space
        proposed = np.exp(log_volumes)
        
        return proposed
    
    def _mcmc_score(
        self,
        volumes: np.ndarray,
        target_q4: float,
        target_m2: float
    ) -> float:
        """
        Score function for MCMC (lower is better).
        
        Args:
            volumes: Volume array
            target_q4: Target Q4 share
            target_m2: Target m[2]
            
        Returns:
            Score (lower is better)
        """
        # Calculate metrics
        q4_start = 3 * len(volumes) // 4
        q4_actual = np.sum(volumes[q4_start:]) / np.sum(volumes) * 100
        
        # Calculate m[2]
        if len(volumes) > 2 and volumes[1] > 0:
            m2_actual = (volumes[2] - volumes[1]) / volumes[1]
        else:
            m2_actual = 0
        
        # Score based on distance from targets
        score = abs(q4_actual - target_q4) * 2.0
        score += abs(m2_actual - target_m2) * 100.0
        
        # Penalty for non-monotonic
        for i in range(1, len(volumes)):
            if volumes[i] <= volumes[i-1]:
                score += 10.0
        
        return score
    
    def generate_genetic_population(
        self,
        num_orders: int,
        population_size: int = 20,
        n_generations: int = 10,
        elite_ratio: float = 0.2
    ) -> List[np.ndarray]:
        """
        Generate initial population using genetic algorithm.
        
        Args:
            num_orders: Number of orders
            population_size: Size of population
            n_generations: Number of evolution generations
            elite_ratio: Fraction of population to keep as elite
            
        Returns:
            List of volume arrays
        """
        # Initialize random population
        population = []
        for _ in range(population_size):
            volumes = self._geometric_start(
                num_orders,
                v0=np.random.uniform(0.5, 1.5),
                m2=np.random.uniform(0.08, 0.15)
            )
            # Add variation
            volumes = self._mutate_strategy(volumes, 0.1)
            volumes = volumes / np.sum(volumes) * 100.0
            population.append(volumes)
        
        # Evolve population
        for generation in range(n_generations):
            # Evaluate fitness
            fitness_scores = []
            for individual in population:
                fitness = self._genetic_fitness(individual)
                fitness_scores.append(fitness)
            
            # Sort by fitness (higher is better)
            sorted_indices = np.argsort(fitness_scores)[::-1]
            population = [population[i] for i in sorted_indices]
            fitness_scores = [fitness_scores[i] for i in sorted_indices]
            
            # Keep elite
            n_elite = int(population_size * elite_ratio)
            new_population = population[:n_elite]
            
            # Generate offspring through crossover and mutation
            while len(new_population) < population_size:
                # Select parents (tournament selection)
                parent1 = self._tournament_select(population, fitness_scores)
                parent2 = self._tournament_select(population, fitness_scores)
                
                # Crossover
                child = self._crossover(parent1, parent2)
                
                # Mutation
                if np.random.random() < 0.3:
                    child = self._mutate_strategy(child, 0.1)
                
                # Ensure valid
                child = child / np.sum(child) * 100.0
                new_population.append(child)
            
            population = new_population[:population_size]
        
        return population
    
    def _genetic_fitness(self, volumes: np.ndarray) -> float:
        """
        Fitness function for genetic algorithm (higher is better).
        
        Args:
            volumes: Volume array
            
        Returns:
            Fitness score
        """
        fitness = 100.0
        
        # Reward good Q4 share (50-60%)
        q4_start = 3 * len(volumes) // 4
        q4_share = np.sum(volumes[q4_start:]) / np.sum(volumes) * 100
        if 50 <= q4_share <= 60:
            fitness += 20.0
        else:
            fitness -= abs(q4_share - 55.0)
        
        # Reward reasonable m[2] (0.10-0.15)
        if len(volumes) > 2 and volumes[1] > 0:
            m2 = (volumes[2] - volumes[1]) / volumes[1]
            if 0.10 <= m2 <= 0.15:
                fitness += 20.0
            else:
                fitness -= abs(m2 - 0.125) * 50
        
        # Reward monotonicity
        is_monotonic = True
        for i in range(1, len(volumes)):
            if volumes[i] <= volumes[i-1]:
                is_monotonic = False
                fitness -= 5.0
        if is_monotonic:
            fitness += 10.0
        
        # Reward smooth growth
        if len(volumes) > 3:
            growth_rates = []
            for i in range(1, len(volumes)):
                if volumes[i-1] > 0:
                    growth_rates.append((volumes[i] - volumes[i-1]) / volumes[i-1])
            
            if growth_rates:
                std_growth = np.std(growth_rates)
                if std_growth < 0.05:
                    fitness += 10.0
        
        return max(0, fitness)
    
    def _tournament_select(
        self,
        population: List[np.ndarray],
        fitness_scores: List[float],
        tournament_size: int = 3
    ) -> np.ndarray:
        """
        Tournament selection for genetic algorithm.
        
        Args:
            population: Current population
            fitness_scores: Fitness scores
            tournament_size: Size of tournament
            
        Returns:
            Selected individual
        """
        indices = np.random.choice(len(population), tournament_size, replace=False)
        tournament_fitness = [fitness_scores[i] for i in indices]
        winner_idx = indices[np.argmax(tournament_fitness)]
        return population[winner_idx]
    
    def _crossover(
        self,
        parent1: np.ndarray,
        parent2: np.ndarray
    ) -> np.ndarray:
        """
        Crossover two parent strategies.
        
        Args:
            parent1: First parent
            parent2: Second parent
            
        Returns:
            Child strategy
        """
        # Uniform crossover with bias towards better features
        child = np.zeros_like(parent1)
        
        for i in range(len(child)):
            if np.random.random() < 0.5:
                child[i] = parent1[i]
            else:
                child[i] = parent2[i]
        
        # Smooth the transition points
        for i in range(1, len(child) - 1):
            if abs(child[i] - child[i-1]) > abs(child[i+1] - child[i]) * 2:
                # Smooth sharp transitions
                child[i] = (child[i-1] + child[i+1]) / 2
        
        return child
    
    def generate_smart_initial(
        self,
        num_orders: int,
        n_total: int = 30,
        use_historical: bool = True,
        use_mcmc: bool = True,
        use_genetic: bool = True
    ) -> List[np.ndarray]:
        """
        Generate smart initial population using all available methods.
        
        Args:
            num_orders: Number of orders
            n_total: Total number of strategies to generate
            use_historical: Whether to use historical strategies
            use_mcmc: Whether to use MCMC sampling
            use_genetic: Whether to use genetic algorithm
            
        Returns:
            List of volume arrays
        """
        all_strategies = []
        
        # Determine how many from each method
        methods_active = sum([use_historical, use_mcmc, use_genetic])
        if methods_active == 0:
            # Fallback to basic geometric
            for _ in range(n_total):
                volumes = self._geometric_start(
                    num_orders,
                    v0=np.random.uniform(0.8, 1.2),
                    m2=np.random.uniform(0.10, 0.13)
                )
                volumes = volumes / np.sum(volumes) * 100.0
                all_strategies.append(volumes)
            return all_strategies
        
        n_per_method = n_total // methods_active
        
        # Generate from each method
        if use_historical and self.successful_strategies:
            historical = self.generate_from_historical(num_orders, n_per_method)
            all_strategies.extend(historical)
        
        if use_mcmc:
            mcmc = self.generate_mcmc_samples(num_orders, n_per_method)
            all_strategies.extend(mcmc)
        
        if use_genetic:
            genetic = self.generate_genetic_population(
                num_orders,
                population_size=n_per_method,
                n_generations=5
            )
            all_strategies.extend(genetic)
        
        # Fill remainder with best method's output
        while len(all_strategies) < n_total:
            volumes = self._geometric_start(
                num_orders,
                v0=1.0,
                m2=np.random.uniform(0.11, 0.13)
            )
            volumes = self._mutate_strategy(volumes, 0.05)
            volumes = volumes / np.sum(volumes) * 100.0
            all_strategies.append(volumes)
        
        return all_strategies[:n_total]


def get_smart_initial_strategy(
    num_orders: int,
    overlap_pct: float,
    seed: Optional[int] = None,
    history_db: Optional[str] = None
) -> np.ndarray:
    """
    Get a single smart initial strategy.
    
    Args:
        num_orders: Number of orders
        overlap_pct: Overlap percentage (affects growth rate)
        seed: Random seed
        history_db: Optional historical database
        
    Returns:
        Volume array
    """
    if seed is not None:
        np.random.seed(seed)
    
    initializer = SmartInitializer(history_db)
    
    # Adjust target parameters based on overlap
    if overlap_pct < 8:
        target_m2 = 0.10
        target_q4 = 45.0
    elif overlap_pct < 15:
        target_m2 = 0.12
        target_q4 = 55.0
    else:
        target_m2 = 0.14
        target_q4 = 60.0
    
    # Try MCMC first (usually best)
    strategies = initializer.generate_mcmc_samples(
        num_orders,
        n_samples=1,
        target_q4=target_q4,
        target_m2=target_m2
    )
    
    if strategies:
        return strategies[0]
    
    # Fallback to geometric
    volumes = initializer._geometric_start(num_orders, m2=target_m2)
    return volumes / np.sum(volumes) * 100.0
