"""
Adaptive parameter calculation based on market conditions and order count.
"""

import numpy as np
from typing import Dict, Tuple


def calculate_adaptive_bands(
    num_orders: int,
    overlap_pct: float,
    base_m2_min: float = 0.10,
    base_m2_max: float = 0.20
) -> Dict[str, float]:
    """
    Calculate adaptive band parameters based on N and overlap.
    
    Principles:
    - Larger N → tighter m2 bands (more gradual growth)
    - Smaller N → wider m2 bands (need stronger recovery)
    - Higher overlap → wider bands (deeper drawdown expected)
    - Lower overlap → tighter bands (shallow drawdown)
    
    Args:
        num_orders: Number of orders (N)
        overlap_pct: Total overlap percentage
        base_m2_min: Base minimum m2 value
        base_m2_max: Base maximum m2 value
        
    Returns:
        Dictionary of adaptive parameters
    """
    
    # N-based adjustments
    # For N=10: multiplier ≈ 1.2, For N=20: multiplier ≈ 1.0, For N=30: multiplier ≈ 0.8
    n_factor = 1.0 + 0.4 * (1.0 - (num_orders - 10) / 20.0)
    n_factor = np.clip(n_factor, 0.7, 1.3)
    
    # Overlap-based adjustments
    # For overlap=5%: multiplier ≈ 0.8, For overlap=10%: multiplier ≈ 1.0, For overlap=20%: multiplier ≈ 1.2
    overlap_factor = 0.8 + 0.02 * overlap_pct
    overlap_factor = np.clip(overlap_factor, 0.7, 1.5)
    
    # Combined factor
    combined_factor = n_factor * overlap_factor
    
    # Adaptive m2 bounds
    m2_min_adaptive = base_m2_min * combined_factor
    m2_max_adaptive = base_m2_max * combined_factor
    
    # Ensure reasonable bounds
    m2_min_adaptive = np.clip(m2_min_adaptive, 0.05, 0.25)
    m2_max_adaptive = np.clip(m2_max_adaptive, 0.10, 0.50)
    
    # Ensure min < max
    if m2_min_adaptive >= m2_max_adaptive:
        m2_max_adaptive = m2_min_adaptive + 0.05
    
    # Adaptive m_min and m_max for tail
    # Smaller for large N, larger for small N
    m_min_adaptive = 0.03 + 0.02 * (20 - num_orders) / 20.0
    m_min_adaptive = np.clip(m_min_adaptive, 0.02, 0.10)
    
    m_max_adaptive = 0.20 + 0.10 * (20 - num_orders) / 20.0
    m_max_adaptive = np.clip(m_max_adaptive, 0.15, 0.40)
    
    # Adaptive decay parameters
    # Faster decay for large N (tau_scale larger)
    tau_scale_adaptive = 0.25 + 0.25 * num_orders / 30.0
    tau_scale_adaptive = np.clip(tau_scale_adaptive, 0.25, 0.5)
    
    # m_head and m_tail based on overlap
    # Higher overlap → higher initial growth allowed
    m_head_adaptive = 0.30 + 0.02 * overlap_pct
    m_head_adaptive = np.clip(m_head_adaptive, 0.25, 0.50)
    
    m_tail_adaptive = 0.15 + 0.01 * overlap_pct
    m_tail_adaptive = np.clip(m_tail_adaptive, 0.10, 0.30)
    
    # Slope cap - tighter for large N
    slope_cap_adaptive = 0.30 - 0.01 * (num_orders - 10)
    slope_cap_adaptive = np.clip(slope_cap_adaptive, 0.10, 0.30)
    
    # Q1/Q4 mass control
    # Stricter Q1 cap for large N (avoid front-loading)
    q1_cap_adaptive = 25.0 - 0.5 * num_orders
    q1_cap_adaptive = np.clip(q1_cap_adaptive, 10.0, 25.0)
    
    # Higher tail floor for small N (need strong recovery)
    tail_floor_adaptive = 30.0 + 1.0 * (20 - num_orders)
    tail_floor_adaptive = np.clip(tail_floor_adaptive, 30.0, 50.0)
    
    return {
        'm2_min': m2_min_adaptive,
        'm2_max': m2_max_adaptive,
        'm_min': m_min_adaptive,
        'm_max': m_max_adaptive,
        'm_head': m_head_adaptive,
        'm_tail': m_tail_adaptive,
        'tau_scale': tau_scale_adaptive,
        'slope_cap': slope_cap_adaptive,
        'q1_cap': q1_cap_adaptive,
        'tail_floor': tail_floor_adaptive,
    }


def calculate_adaptive_weights(
    num_orders: int,
    overlap_pct: float,
    strategy_type: str = "balanced"
) -> Dict[str, float]:
    """
    Calculate adaptive penalty weights based on conditions.
    
    Args:
        num_orders: Number of orders
        overlap_pct: Overlap percentage
        strategy_type: Type of strategy ("aggressive", "balanced", "conservative")
        
    Returns:
        Dictionary of penalty weights
    """
    
    weights = {}
    
    if strategy_type == "aggressive":
        # Focus on tail strength and exit-ease
        weights['w_front'] = 4.0  # Penalize front-loading heavily
        weights['w_tailweak'] = 5.0  # Require strong tail
        weights['w_slope'] = 1.0  # Allow more variation
        weights['w_plateau'] = 1.0  # Less concern about plateaus
        weights['exit_ease_weight'] = 150.0  # High weight on exit-ease
        
    elif strategy_type == "conservative":
        # Focus on smooth, predictable growth
        weights['w_front'] = 2.0  # Moderate front penalty
        weights['w_tailweak'] = 2.0  # Moderate tail requirement
        weights['w_slope'] = 3.0  # Penalize large changes
        weights['w_plateau'] = 3.0  # Avoid plateaus
        weights['exit_ease_weight'] = 50.0  # Lower exit-ease weight
        
    else:  # balanced
        # Balanced approach (your strategy style)
        weights['w_front'] = 3.0
        weights['w_tailweak'] = 3.0
        weights['w_slope'] = 2.0
        weights['w_plateau'] = 2.0
        weights['exit_ease_weight'] = 100.0
    
    # Adjust based on N
    if num_orders < 15:
        # Small N: focus on recovery power
        weights['w_tailweak'] *= 1.5
        weights['exit_ease_weight'] *= 1.2
    elif num_orders > 25:
        # Large N: focus on smoothness
        weights['w_slope'] *= 1.5
        weights['w_plateau'] *= 1.5
    
    # Adjust based on overlap
    if overlap_pct < 8:
        # Small overlap: need efficient recovery
        weights['w_front'] *= 0.8  # Allow more front
        weights['exit_ease_weight'] *= 1.3
    elif overlap_pct > 15:
        # Large overlap: focus on tail
        weights['w_tailweak'] *= 1.3
        weights['w_front'] *= 1.2
    
    return weights


def get_adaptive_parameters(
    num_orders: int,
    overlap_pct: float,
    strategy_type: str = "balanced",
    user_overrides: Dict[str, float] = None
) -> Dict[str, float]:
    """
    Get complete set of adaptive parameters.
    
    Args:
        num_orders: Number of orders
        overlap_pct: Overlap percentage
        strategy_type: Strategy type
        user_overrides: Optional parameter overrides
        
    Returns:
        Complete parameter dictionary
    """
    
    # Get adaptive bands
    bands = calculate_adaptive_bands(num_orders, overlap_pct)
    
    # Get adaptive weights
    weights = calculate_adaptive_weights(num_orders, overlap_pct, strategy_type)
    
    # Combine
    params = {**bands, **weights}
    
    # Apply user overrides if provided
    if user_overrides:
        params.update(user_overrides)
    
    return params


def suggest_parameters_for_market(
    volatility: float,
    trend_strength: float,
    num_orders: int
) -> Dict[str, float]:
    """
    Suggest parameters based on market conditions.
    
    Args:
        volatility: Market volatility (0-1)
        trend_strength: Trend strength (-1 to 1, negative=downtrend)
        num_orders: Number of orders
        
    Returns:
        Suggested parameters
    """
    
    suggestions = {}
    
    # High volatility → wider bands, stronger tail
    if volatility > 0.7:
        suggestions['m2_min'] = 0.12
        suggestions['m2_max'] = 0.25
        suggestions['tail_floor'] = 45.0
        suggestions['strategy_type'] = 'aggressive'
    elif volatility < 0.3:
        suggestions['m2_min'] = 0.08
        suggestions['m2_max'] = 0.15
        suggestions['tail_floor'] = 35.0
        suggestions['strategy_type'] = 'conservative'
    else:
        suggestions['strategy_type'] = 'balanced'
    
    # Strong downtrend → need more recovery power
    if trend_strength < -0.5:
        overlap = 15.0  # Deeper grid
        suggestions['overlap_pct'] = overlap
    elif trend_strength > 0.5:
        overlap = 8.0  # Shallower grid
        suggestions['overlap_pct'] = overlap
    else:
        overlap = 10.0
        suggestions['overlap_pct'] = overlap
    
    # Get adaptive parameters
    params = get_adaptive_parameters(
        num_orders,
        overlap,
        suggestions.get('strategy_type', 'balanced'),
        suggestions
    )
    
    return params