"""
Micro pattern detection for identifying problematic patterns in martingale strategies.
"""

import numpy as np
from numba import njit
from typing import Dict, List, Tuple


@njit(cache=True)
def detect_plateaus(m: np.ndarray, tolerance: float = 0.02, min_length: int = 3) -> List[Tuple[int, int]]:
    """
    Detect plateau regions in martingale ratios.
    
    A plateau is a sequence where values are approximately constant.
    
    Args:
        m: Martingale ratios
        tolerance: Maximum deviation from mean to consider as plateau
        min_length: Minimum length to consider as plateau
        
    Returns:
        List of (start_idx, length) tuples for each plateau
    """
    plateaus = []
    M = len(m)
    
    i = 2  # Start from m[2]
    while i < M:
        # Check if this could be start of a plateau
        plateau_start = i
        plateau_values = [m[i]]
        j = i + 1
        
        while j < M:
            # Check if m[j] is close to mean of plateau
            plateau_mean = np.mean(np.array(plateau_values))
            if abs(m[j] - plateau_mean) <= tolerance:
                plateau_values.append(m[j])
                j += 1
            else:
                break
        
        plateau_length = j - plateau_start
        if plateau_length >= min_length:
            plateaus.append((plateau_start, plateau_length))
            i = j  # Skip past this plateau
        else:
            i += 1
    
    return plateaus


@njit(cache=True)
def detect_zigzag(m: np.ndarray, min_amplitude: float = 0.05) -> int:
    """
    Detect zigzag patterns (alternating up/down).
    
    Args:
        m: Martingale ratios
        min_amplitude: Minimum change to consider as zigzag
        
    Returns:
        Number of direction changes
    """
    M = len(m)
    if M < 4:
        return 0
    
    direction_changes = 0
    last_direction = 0  # 0=unknown, 1=up, -1=down
    
    for i in range(3, M):
        delta = m[i] - m[i-1]
        
        if abs(delta) >= min_amplitude:
            current_direction = 1 if delta > 0 else -1
            
            if last_direction != 0 and current_direction != last_direction:
                direction_changes += 1
            
            last_direction = current_direction
    
    return direction_changes


@njit(cache=True)
def detect_acceleration(m: np.ndarray) -> Tuple[float, float]:
    """
    Detect acceleration patterns in growth.
    
    Returns:
        (max_acceleration, avg_acceleration)
    """
    M = len(m)
    if M < 4:
        return 0.0, 0.0
    
    accelerations = np.zeros(M - 3)
    for i in range(3, M):
        # Second derivative approximation
        acc = (m[i] - m[i-1]) - (m[i-1] - m[i-2])
        accelerations[i-3] = acc
    
    if len(accelerations) > 0:
        return np.max(np.abs(accelerations)), np.mean(np.abs(accelerations))
    return 0.0, 0.0


@njit(cache=True)
def detect_cliff(v: np.ndarray, threshold: float = 2.0) -> List[int]:
    """
    Detect cliff patterns (sudden large jumps).
    
    Args:
        v: Volume percentages
        threshold: Ratio threshold to consider as cliff
        
    Returns:
        List of indices where cliffs occur
    """
    cliffs = []
    M = len(v)
    
    for i in range(1, M):
        if v[i-1] > 0:
            ratio = v[i] / v[i-1]
            if ratio > threshold:
                cliffs.append(i)
    
    return cliffs


@njit(cache=True)
def detect_stagnation(m: np.ndarray, window: int = 5, threshold: float = 0.03) -> List[int]:
    """
    Detect stagnation zones (very slow growth).
    
    Args:
        m: Martingale ratios
        window: Window size to check
        threshold: Maximum average growth to consider as stagnation
        
    Returns:
        List of stagnation zone starts
    """
    stagnation_zones = []
    M = len(m)
    
    for i in range(2, M - window + 1):
        # Calculate window average manually for numba
        window_sum = 0.0
        for j in range(window):
            window_sum += m[i + j]
        window_avg = window_sum / window
        
        if window_avg < threshold:
            stagnation_zones.append(i)
    
    return stagnation_zones


def analyze_micro_patterns(
    volumes: np.ndarray,
    martingales: np.ndarray
) -> Dict[str, any]:
    """
    Comprehensive micro pattern analysis.
    
    Args:
        volumes: Volume percentages
        martingales: Martingale ratios
        
    Returns:
        Dictionary with pattern analysis results
    """
    analysis = {}
    
    # Detect plateaus
    plateaus = detect_plateaus(martingales)
    analysis['plateau_count'] = len(plateaus)
    analysis['max_plateau_length'] = max([p[1] for p in plateaus], default=0)
    analysis['plateau_zones'] = plateaus
    
    # Detect zigzag
    zigzag_count = detect_zigzag(martingales)
    analysis['zigzag_count'] = zigzag_count
    analysis['has_excessive_zigzag'] = zigzag_count > len(martingales) // 3
    
    # Detect acceleration
    max_acc, avg_acc = detect_acceleration(martingales)
    analysis['max_acceleration'] = max_acc
    analysis['avg_acceleration'] = avg_acc
    analysis['has_sudden_acceleration'] = max_acc > 0.1
    
    # Detect cliffs
    cliffs = detect_cliff(volumes)
    analysis['cliff_count'] = len(cliffs)
    analysis['cliff_positions'] = cliffs
    analysis['has_cliffs'] = len(cliffs) > 0
    
    # Detect stagnation
    stagnation = detect_stagnation(martingales)
    analysis['stagnation_zones'] = len(stagnation)
    analysis['has_stagnation'] = len(stagnation) > 2
    
    # Overall pattern quality score
    pattern_score = 100.0
    pattern_score -= len(plateaus) * 5  # Penalize plateaus
    pattern_score -= zigzag_count * 2  # Penalize zigzag
    pattern_score -= len(cliffs) * 10  # Heavily penalize cliffs
    pattern_score -= len(stagnation) * 3  # Penalize stagnation
    pattern_score -= max_acc * 50  # Penalize sudden acceleration
    
    analysis['pattern_quality_score'] = max(0.0, pattern_score)
    
    # Recommendations
    recommendations = []
    if analysis['plateau_count'] > 2:
        recommendations.append("Too many plateaus - increase variation")
    if analysis['has_excessive_zigzag']:
        recommendations.append("Excessive zigzag - smooth the curve")
    if analysis['has_cliffs']:
        recommendations.append(f"Cliffs detected at positions {cliffs} - reduce jumps")
    if analysis['has_stagnation']:
        recommendations.append("Stagnation zones detected - increase growth rate")
    if analysis['has_sudden_acceleration']:
        recommendations.append("Sudden acceleration - make growth more gradual")
    
    analysis['recommendations'] = recommendations
    
    return analysis


@njit(cache=True)
def compute_pattern_penalties(
    volumes: np.ndarray,
    martingales: np.ndarray,
    w_plateau: float = 5.0,
    w_zigzag: float = 2.0,
    w_cliff: float = 10.0,
    w_stagnation: float = 3.0,
    w_acceleration: float = 5.0
) -> float:
    """
    Compute total penalty for micro patterns.
    
    Args:
        volumes: Volume percentages
        martingales: Martingale ratios
        w_plateau: Weight for plateau penalty
        w_zigzag: Weight for zigzag penalty
        w_cliff: Weight for cliff penalty
        w_stagnation: Weight for stagnation penalty
        w_acceleration: Weight for acceleration penalty
        
    Returns:
        Total pattern penalty
    """
    penalty = 0.0
    
    # Plateau penalty
    plateaus = detect_plateaus(martingales)
    for start, length in plateaus:
        penalty += w_plateau * (length - 2)  # Penalty grows with length
    
    # Zigzag penalty
    zigzag_count = detect_zigzag(martingales)
    penalty += w_zigzag * zigzag_count
    
    # Cliff penalty
    cliffs = detect_cliff(volumes)
    penalty += w_cliff * len(cliffs)
    
    # Stagnation penalty
    stagnation = detect_stagnation(martingales)
    penalty += w_stagnation * len(stagnation)
    
    # Acceleration penalty
    max_acc, avg_acc = detect_acceleration(martingales)
    penalty += w_acceleration * max_acc * 100
    
    return penalty


def validate_pattern_constraints(
    volumes: np.ndarray,
    martingales: np.ndarray,
    max_plateaus: int = 2,
    max_zigzag_ratio: float = 0.3,
    max_cliffs: int = 0,
    max_stagnation_zones: int = 2
) -> Tuple[bool, List[str]]:
    """
    Validate that patterns meet constraints.
    
    Args:
        volumes: Volume percentages
        martingales: Martingale ratios
        max_plateaus: Maximum allowed plateaus
        max_zigzag_ratio: Maximum zigzag to length ratio
        max_cliffs: Maximum allowed cliffs
        max_stagnation_zones: Maximum stagnation zones
        
    Returns:
        (is_valid, list_of_violations)
    """
    violations = []
    
    # Check plateaus
    plateaus = detect_plateaus(martingales)
    if len(plateaus) > max_plateaus:
        violations.append(f"Too many plateaus: {len(plateaus)} > {max_plateaus}")
    
    # Check zigzag
    zigzag_count = detect_zigzag(martingales)
    zigzag_ratio = zigzag_count / max(1, len(martingales))
    if zigzag_ratio > max_zigzag_ratio:
        violations.append(f"Excessive zigzag: ratio {zigzag_ratio:.2f} > {max_zigzag_ratio}")
    
    # Check cliffs
    cliffs = detect_cliff(volumes)
    if len(cliffs) > max_cliffs:
        violations.append(f"Cliffs detected at positions: {cliffs}")
    
    # Check stagnation
    stagnation = detect_stagnation(martingales)
    if len(stagnation) > max_stagnation_zones:
        violations.append(f"Too many stagnation zones: {len(stagnation)} > {max_stagnation_zones}")
    
    is_valid = len(violations) == 0
    return is_valid, violations