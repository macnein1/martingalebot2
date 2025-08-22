"""
Numba JIT kernels for fast evaluation of martingale strategies.
"""
import numpy as np
from numba import njit, prange
from typing import Tuple, List
from .types import Params, Schedule, ScoreBreakdown


@njit
def calculate_martingale_volumes(base_volume: float, multiplier: float, 
                               num_levels: int) -> np.ndarray:
    """Calculate martingale volume progression."""
    volumes = np.zeros(num_levels)
    for i in range(num_levels):
        volumes[i] = base_volume * (multiplier ** i)
    return volumes


@njit
def calculate_overlaps(min_overlap: float, max_overlap: float, 
                      num_levels: int) -> np.ndarray:
    """Calculate overlap values linearly distributed."""
    overlaps = np.zeros(num_levels)
    for i in range(num_levels):
        overlaps[i] = min_overlap + (max_overlap - min_overlap) * i / (num_levels - 1)
    return overlaps


@njit
def calculate_orders(min_order: int, max_order: int, num_levels: int) -> np.ndarray:
    """Calculate order values linearly distributed."""
    orders = np.zeros(num_levels, dtype=np.int32)
    for i in range(num_levels):
        orders[i] = min_order + int((max_order - min_order) * i / (num_levels - 1))
    return orders


@njit
def calculate_max_score(volumes: np.ndarray) -> float:
    """Calculate maximum score component."""
    return np.max(volumes)


@njit
def calculate_variance_score(volumes: np.ndarray) -> float:
    """Calculate variance score component."""
    if len(volumes) <= 1:
        return 0.0
    return np.var(volumes)


@njit
def calculate_tail_score(volumes: np.ndarray, percentile: float = 0.95) -> float:
    """Calculate tail risk score component."""
    if len(volumes) == 0:
        return 0.0
    
    # Sort volumes for percentile calculation
    sorted_volumes = np.sort(volumes)
    n = len(sorted_volumes)
    idx = int(percentile * n)
    
    if idx >= n:
        return 0.0
    
    # Calculate tail risk
    tail_volumes = sorted_volumes[idx:]
    return np.mean(tail_volumes)


@njit
def calculate_gini_penalty(volumes: np.ndarray, target_gini: float = 0.3) -> float:
    """Calculate Gini coefficient penalty."""
    if len(volumes) <= 1:
        return 0.0
    
    # Sort volumes
    sorted_volumes = np.sort(volumes)
    n = len(sorted_volumes)
    cumsum = np.cumsum(sorted_volumes)
    
    # Calculate Gini coefficient
    gini = (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n
    
    return abs(gini - target_gini)


@njit
def calculate_monotone_penalty(sequence: np.ndarray) -> float:
    """Calculate monotonicity penalty."""
    if len(sequence) <= 1:
        return 0.0
    
    violations = 0
    for i in range(1, len(sequence)):
        if sequence[i] < sequence[i-1]:
            violations += 1
    
    return violations


@njit
def calculate_smoothness_penalty(sequence: np.ndarray, max_change: float = 0.5) -> float:
    """Calculate smoothness penalty."""
    if len(sequence) <= 1:
        return 0.0
    
    violations = 0
    for i in range(1, len(sequence)):
        if sequence[i-1] > 0:
            relative_change = abs(sequence[i] - sequence[i-1]) / sequence[i-1]
            if relative_change > max_change:
                violations += 1
    
    return violations


@njit
def evaluate_kernel(params_array: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    Main evaluation kernel for martingale parameters.
    
    Args:
        params_array: Normalized parameters [min_overlap, max_overlap, min_order, max_order, ...]
    
    Returns:
        Tuple of (total_score, score_breakdown)
    """
    # Denormalize parameters
    min_overlap = params_array[0] * 100.0
    max_overlap = params_array[1] * 100.0
    min_order = int(params_array[2] * 49.0) + 1
    max_order = int(params_array[3] * 49.0) + 1
    
    # Calculate number of levels
    num_levels = max_order - min_order + 1
    if num_levels <= 0:
        return -np.inf, np.zeros(8)
    
    # Generate schedule
    base_volume = 100.0  # Base volume
    multiplier = 2.0     # Martingale multiplier
    
    volumes = calculate_martingale_volumes(base_volume, multiplier, num_levels)
    overlaps = calculate_overlaps(min_overlap, max_overlap, num_levels)
    orders = calculate_orders(min_order, max_order, num_levels)
    
    # Calculate score components
    max_score = calculate_max_score(volumes)
    variance_score = calculate_variance_score(volumes)
    tail_score = calculate_tail_score(volumes)
    
    # Calculate penalties
    gini_penalty = calculate_gini_penalty(volumes)
    monotone_penalty = calculate_monotone_penalty(orders)
    smoothness_penalty = calculate_smoothness_penalty(volumes)
    
    # Combine scores (weights can be adjusted)
    total_score = (max_score + 0.5 * variance_score + 0.2 * tail_score - 
                   gini_penalty - 2.0 * monotone_penalty - smoothness_penalty)
    
    # Score breakdown
    breakdown = np.array([
        total_score, max_score, variance_score, tail_score,
        gini_penalty, 0.0, monotone_penalty, smoothness_penalty
    ])
    
    return total_score, breakdown


@njit(parallel=True)
def evaluate_batch_kernel(params_batch: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Batch evaluation kernel for multiple parameter sets.
    
    Args:
        params_batch: Array of normalized parameters [n_samples, n_params]
    
    Returns:
        Tuple of (scores, breakdowns)
    """
    n_samples = params_batch.shape[0]
    scores = np.zeros(n_samples)
    breakdowns = np.zeros((n_samples, 8))
    
    for i in prange(n_samples):
        score, breakdown = evaluate_kernel(params_batch[i])
        scores[i] = score
        breakdowns[i] = breakdown
    
    return scores, breakdowns


def create_schedule_from_params(params: Params) -> Schedule:
    """Create a Schedule from Params object."""
    num_levels = params.max_order - params.min_order + 1
    
    # Generate schedule components
    base_volume = 100.0
    multiplier = 2.0
    
    volumes = calculate_martingale_volumes(base_volume, multiplier, num_levels)
    overlaps = calculate_overlaps(params.min_overlap, params.max_overlap, num_levels)
    orders = calculate_orders(params.min_order, params.max_order, num_levels)
    
    return Schedule(orders=orders, volumes=volumes, overlaps=overlaps)


def evaluate_params(params: Params) -> Tuple[float, ScoreBreakdown]:
    """Evaluate a Params object and return score with breakdown."""
    # Normalize parameters
    params_array = np.array([
        params.min_overlap / 100.0,
        params.max_overlap / 100.0,
        (params.min_order - 1) / 49.0,
        (params.max_order - 1) / 49.0,
        params.risk_factor / 10.0,
        params.smoothing_factor,
        params.tail_weight
    ])
    
    # Evaluate
    total_score, breakdown = evaluate_kernel(params_array)
    
    # Create ScoreBreakdown
    score_breakdown = ScoreBreakdown(
        total_score=total_score,
        max_score=breakdown[1],
        variance_score=breakdown[2],
        tail_score=breakdown[3],
        gini_penalty=breakdown[4],
        entropy_penalty=breakdown[5],
        monotone_penalty=breakdown[6],
        smoothness_penalty=breakdown[7]
    )
    
    return total_score, score_breakdown


def evaluate_params_batch(params_list: List[Params]) -> List[Tuple[float, ScoreBreakdown]]:
    """Evaluate a batch of Params objects."""
    # Convert to array
    n_samples = len(params_list)
    params_batch = np.zeros((n_samples, 7))
    
    for i, params in enumerate(params_list):
        params_batch[i] = np.array([
            params.min_overlap / 100.0,
            params.max_overlap / 100.0,
            (params.min_order - 1) / 49.0,
            (params.max_order - 1) / 49.0,
            params.risk_factor / 10.0,
            params.smoothing_factor,
            params.tail_weight
        ])
    
    # Evaluate batch
    scores, breakdowns = evaluate_batch_kernel(params_batch)
    
    # Convert results
    results = []
    for i in range(n_samples):
        score_breakdown = ScoreBreakdown(
            total_score=scores[i],
            max_score=breakdowns[i, 1],
            variance_score=breakdowns[i, 2],
            tail_score=breakdowns[i, 3],
            gini_penalty=breakdowns[i, 4],
            entropy_penalty=breakdowns[i, 5],
            monotone_penalty=breakdowns[i, 6],
            smoothness_penalty=breakdowns[i, 7]
        )
        results.append((scores[i], score_breakdown))
    
    return results
