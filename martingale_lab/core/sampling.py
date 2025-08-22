"""
Sampling utilities for parameter space exploration.
"""
import numpy as np
from typing import List, Tuple, Optional
from .types import Params


class LatinHypercubeSampler:
    """Latin Hypercube Sampling for parameter space exploration."""
    
    def __init__(self, seed: Optional[int] = None):
        """Initialize sampler with optional seed."""
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)
    
    def sample(self, n_samples: int, param_bounds: List[Tuple[float, float]]) -> np.ndarray:
        """Generate LHS samples."""
        n_params = len(param_bounds)
        samples = np.zeros((n_samples, n_params))
        
        for i in range(n_params):
            # Generate LHS samples for each parameter
            lhs_samples = np.random.permutation(n_samples)
            # Scale to parameter bounds
            min_val, max_val = param_bounds[i]
            samples[:, i] = min_val + (max_val - min_val) * lhs_samples / (n_samples - 1)
        
        return samples
    
    def sample_params(self, n_samples: int, 
                     min_overlap_range: Tuple[float, float] = (0.0, 50.0),
                     max_overlap_range: Tuple[float, float] = (10.0, 100.0),
                     min_order_range: Tuple[int, int] = (1, 20),
                     max_order_range: Tuple[int, int] = (5, 50)) -> List[Params]:
        """Generate LHS samples for martingale parameters."""
        param_bounds = [
            min_overlap_range,
            max_overlap_range,
            (float(min_order_range[0]), float(min_order_range[1])),
            (float(max_order_range[0]), float(max_order_range[1]))
        ]
        
        samples = self.sample(n_samples, param_bounds)
        params_list = []
        
        for sample in samples:
            params = Params(
                min_overlap=sample[0],
                max_overlap=sample[1],
                min_order=int(sample[2]),
                max_order=int(sample[3])
            )
            params_list.append(params)
        
        return params_list


class UniformSampler:
    """Uniform sampling for parameter space exploration."""
    
    def __init__(self, seed: Optional[int] = None):
        """Initialize sampler with optional seed."""
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)
    
    def sample(self, n_samples: int, param_bounds: List[Tuple[float, float]]) -> np.ndarray:
        """Generate uniform samples."""
        n_params = len(param_bounds)
        samples = np.zeros((n_samples, n_params))
        
        for i in range(n_params):
            min_val, max_val = param_bounds[i]
            samples[:, i] = np.random.uniform(min_val, max_val, n_samples)
        
        return samples
    
    def sample_params(self, n_samples: int,
                     min_overlap_range: Tuple[float, float] = (0.0, 50.0),
                     max_overlap_range: Tuple[float, float] = (10.0, 100.0),
                     min_order_range: Tuple[int, int] = (1, 20),
                     max_order_range: Tuple[int, int] = (5, 50)) -> List[Params]:
        """Generate uniform samples for martingale parameters."""
        param_bounds = [
            min_overlap_range,
            max_overlap_range,
            (float(min_order_range[0]), float(min_order_range[1])),
            (float(max_order_range[0]), float(max_order_range[1]))
        ]
        
        samples = self.sample(n_samples, param_bounds)
        params_list = []
        
        for sample in samples:
            params = Params(
                min_overlap=sample[0],
                max_overlap=sample[1],
                min_order=int(sample[2]),
                max_order=int(sample[3])
            )
            params_list.append(params)
        
        return params_list


class AdaptiveSampler:
    """Adaptive sampling based on previous results."""
    
    def __init__(self, seed: Optional[int] = None):
        """Initialize adaptive sampler."""
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)
        self.history = []
    
    def add_result(self, params: Params, score: float):
        """Add a result to the history."""
        self.history.append((params, score))
    
    def sample_around_best(self, n_samples: int, best_params: Params, 
                          radius: float = 0.1) -> List[Params]:
        """Sample around the best parameters found so far."""
        params_list = []
        
        for _ in range(n_samples):
            # Add random perturbation to best parameters
            min_overlap = np.clip(
                best_params.min_overlap + np.random.normal(0, radius * 10),
                0, best_params.max_overlap
            )
            max_overlap = np.clip(
                best_params.max_overlap + np.random.normal(0, radius * 20),
                min_overlap, 100
            )
            min_order = np.clip(
                best_params.min_order + int(np.random.normal(0, radius * 5)),
                1, best_params.max_order
            )
            max_order = np.clip(
                best_params.max_order + int(np.random.normal(0, radius * 10)),
                min_order, 50
            )
            
            params = Params(
                min_overlap=min_overlap,
                max_overlap=max_overlap,
                min_order=min_order,
                max_order=max_order
            )
            params_list.append(params)
        
        return params_list


class SeedGenerator:
    """Generate reproducible seeds for sampling."""
    
    def __init__(self, base_seed: int = 42):
        """Initialize with base seed."""
        self.base_seed = base_seed
        self.counter = 0
    
    def get_seed(self) -> int:
        """Get next seed."""
        seed = self.base_seed + self.counter
        self.counter += 1
        return seed
    
    def reset(self):
        """Reset counter."""
        self.counter = 0
    
    def set_base_seed(self, base_seed: int):
        """Set new base seed."""
        self.base_seed = base_seed
        self.counter = 0


class SamplingStrategy:
    """Combined sampling strategy."""
    
    def __init__(self, seed: Optional[int] = None):
        """Initialize sampling strategy."""
        self.seed_generator = SeedGenerator(seed or 42)
        self.lhs_sampler = LatinHypercubeSampler()
        self.uniform_sampler = UniformSampler()
        self.adaptive_sampler = AdaptiveSampler()
    
    def initial_exploration(self, n_samples: int) -> List[Params]:
        """Initial exploration using LHS."""
        seed = self.seed_generator.get_seed()
        self.lhs_sampler.seed = seed
        return self.lhs_sampler.sample_params(n_samples)
    
    def uniform_exploration(self, n_samples: int) -> List[Params]:
        """Uniform exploration."""
        seed = self.seed_generator.get_seed()
        self.uniform_sampler.seed = seed
        return self.uniform_sampler.sample_params(n_samples)
    
    def adaptive_refinement(self, n_samples: int, best_params: Params, 
                           radius: float = 0.1) -> List[Params]:
        """Adaptive refinement around best parameters."""
        seed = self.seed_generator.get_seed()
        self.adaptive_sampler.seed = seed
        return self.adaptive_sampler.sample_around_best(n_samples, best_params, radius)
