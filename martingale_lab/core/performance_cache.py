"""
Performance optimization through caching and memoization.
"""
from __future__ import annotations

import functools
import hashlib
import pickle
from typing import Any, Callable, Dict, Optional, Tuple
import numpy as np


class ParameterCache:
    """
    Cache for expensive parameter computations.
    """
    
    def __init__(self, max_size: int = 1000):
        """Initialize cache with maximum size."""
        self.cache: Dict[str, Any] = {}
        self.max_size = max_size
        self.hits = 0
        self.misses = 0
    
    def _make_key(self, *args, **kwargs) -> str:
        """Create cache key from arguments."""
        # Convert numpy arrays to bytes for hashing
        key_parts = []
        for arg in args:
            if isinstance(arg, np.ndarray):
                key_parts.append(arg.tobytes())
            else:
                key_parts.append(str(arg))
        
        for k, v in sorted(kwargs.items()):
            key_parts.append(f"{k}={v}")
        
        key_str = "|".join(str(p) if not isinstance(p, bytes) else p.decode() for p in key_parts)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        if key in self.cache:
            self.hits += 1
            # Move to end (LRU)
            value = self.cache.pop(key)
            self.cache[key] = value
            return value
        self.misses += 1
        return None
    
    def put(self, key: str, value: Any):
        """Put value in cache."""
        # Remove oldest if at capacity
        if len(self.cache) >= self.max_size:
            # Remove first item (oldest)
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        
        self.cache[key] = value
    
    def clear(self):
        """Clear cache."""
        self.cache.clear()
        self.hits = 0
        self.misses = 0
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0
        return {
            'size': len(self.cache),
            'max_size': self.max_size,
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate
        }


# Global caches for different computation types
_constraint_cache = ParameterCache(max_size=500)
_penalty_cache = ParameterCache(max_size=500)
_normalization_cache = ParameterCache(max_size=200)


def cached_constraint(func: Callable) -> Callable:
    """
    Decorator for caching constraint computations.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Create cache key
        key = _constraint_cache._make_key(*args, **kwargs)
        
        # Check cache
        result = _constraint_cache.get(key)
        if result is not None:
            return result
        
        # Compute and cache
        result = func(*args, **kwargs)
        _constraint_cache.put(key, result)
        return result
    
    return wrapper


def cached_penalty(func: Callable) -> Callable:
    """
    Decorator for caching penalty computations.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Create cache key
        key = _penalty_cache._make_key(*args, **kwargs)
        
        # Check cache
        result = _penalty_cache.get(key)
        if result is not None:
            return result
        
        # Compute and cache
        result = func(*args, **kwargs)
        _penalty_cache.put(key, result)
        return result
    
    return wrapper


def batch_cache_key(batch_data: np.ndarray, params: Dict[str, Any]) -> str:
    """
    Create cache key for batch operations.
    """
    # Use shape and sample of data for key
    shape_str = str(batch_data.shape)
    sample_str = str(batch_data.flat[:10])  # First 10 elements
    param_str = str(sorted(params.items()))
    
    key_str = f"{shape_str}|{sample_str}|{param_str}"
    return hashlib.md5(key_str.encode()).hexdigest()


class BatchProcessor:
    """
    Optimized batch processing with parallelization.
    """
    
    def __init__(self, batch_size: int = 100):
        """Initialize batch processor."""
        self.batch_size = batch_size
        self.cache = ParameterCache(max_size=100)
    
    def process_batch(
        self,
        data: np.ndarray,
        func: Callable,
        parallel: bool = True
    ) -> np.ndarray:
        """
        Process data in optimized batches.
        """
        n = len(data)
        results = np.zeros(n)
        
        # Process in chunks
        for i in range(0, n, self.batch_size):
            end = min(i + self.batch_size, n)
            batch = data[i:end]
            
            # Check cache
            key = batch_cache_key(batch, {})
            cached = self.cache.get(key)
            
            if cached is not None:
                results[i:end] = cached
            else:
                # Process batch
                if parallel and end - i > 10:
                    # Use vectorized operation if available
                    batch_result = func(batch)
                else:
                    # Sequential processing
                    batch_result = np.array([func(x) for x in batch])
                
                results[i:end] = batch_result
                self.cache.put(key, batch_result)
        
        return results


def clear_all_caches():
    """Clear all performance caches."""
    _constraint_cache.clear()
    _penalty_cache.clear()
    _normalization_cache.clear()


def get_cache_stats() -> Dict[str, Dict[str, Any]]:
    """Get statistics for all caches."""
    return {
        'constraint': _constraint_cache.stats(),
        'penalty': _penalty_cache.stats(),
        'normalization': _normalization_cache.stats()
    }


# Precomputed constants for common operations
COMMON_SLOPES = np.array([0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35])
COMMON_M2_VALUES = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])


def precompute_common_values():
    """
    Precompute commonly used values for faster lookup.
    """
    # Precompute growth factors
    growth_factors = {}
    for slope in COMMON_SLOPES:
        for m2 in COMMON_M2_VALUES:
            key = f"growth_{slope:.2f}_{m2:.1f}"
            growth_factors[key] = 1.0 + min(m2, slope)
    
    return growth_factors


# Initialize precomputed values
PRECOMPUTED = precompute_common_values()