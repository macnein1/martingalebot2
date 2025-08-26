"""
Portfolio metrics for advanced risk-adjusted return analysis.
"""

import numpy as np
from numba import njit
from typing import Dict, Optional, Tuple


@njit(cache=True)
def calculate_sortino_ratio(
    returns: np.ndarray,
    target_return: float = 0.0,
    periods_per_year: float = 252.0
) -> float:
    """
    Calculate Sortino ratio (downside deviation).
    
    Better than Sharpe as it only penalizes downside volatility.
    
    Args:
        returns: Array of returns
        target_return: Minimum acceptable return
        periods_per_year: Annualization factor
        
    Returns:
        Sortino ratio
    """
    if len(returns) == 0:
        return 0.0
    
    excess_returns = returns - target_return
    mean_excess = np.mean(excess_returns)
    
    # Calculate downside deviation
    downside_returns = np.minimum(excess_returns, 0)
    downside_variance = np.mean(downside_returns ** 2)
    
    if downside_variance <= 0:
        return 0.0
    
    downside_deviation = np.sqrt(downside_variance)
    
    if downside_deviation == 0:
        return 0.0
    
    sortino = mean_excess / downside_deviation * np.sqrt(periods_per_year)
    return sortino


@njit(cache=True)
def calculate_calmar_ratio(
    returns: np.ndarray,
    periods_per_year: float = 252.0
) -> float:
    """
    Calculate Calmar ratio (return / max drawdown).
    
    Excellent for martingale strategies as it considers drawdown.
    
    Args:
        returns: Array of returns
        periods_per_year: Annualization factor
        
    Returns:
        Calmar ratio
    """
    if len(returns) == 0:
        return 0.0
    
    # Calculate cumulative returns (manual for numba)
    cumulative = np.ones(len(returns))
    cumulative[0] = 1 + returns[0]
    for i in range(1, len(returns)):
        cumulative[i] = cumulative[i-1] * (1 + returns[i])
    
    # Calculate running maximum (numba compatible)
    running_max = np.zeros_like(cumulative)
    running_max[0] = cumulative[0]
    for i in range(1, len(cumulative)):
        running_max[i] = max(running_max[i-1], cumulative[i])
    
    # Calculate drawdown
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = np.min(drawdown)
    
    if max_drawdown >= 0:
        return 0.0
    
    # Annualized return
    total_return = cumulative[-1] - 1
    n_periods = len(returns)
    annualized_return = (1 + total_return) ** (periods_per_year / n_periods) - 1
    
    calmar = annualized_return / abs(max_drawdown)
    return calmar


@njit(cache=True)
def calculate_omega_ratio(
    returns: np.ndarray,
    threshold: float = 0.0
) -> float:
    """
    Calculate Omega ratio (probability weighted ratio of gains vs losses).
    
    Superior to Sharpe as it considers all moments of distribution.
    
    Args:
        returns: Array of returns
        threshold: Minimum acceptable return threshold
        
    Returns:
        Omega ratio
    """
    if len(returns) == 0:
        return 1.0
    
    gains = np.maximum(returns - threshold, 0)
    losses = np.maximum(threshold - returns, 0)
    
    sum_gains = np.sum(gains)
    sum_losses = np.sum(losses)
    
    if sum_losses == 0:
        return np.inf if sum_gains > 0 else 1.0
    
    omega = sum_gains / sum_losses
    return omega


@njit(cache=True)
def calculate_information_ratio(
    returns: np.ndarray,
    benchmark_returns: np.ndarray,
    periods_per_year: float = 252.0
) -> float:
    """
    Calculate Information ratio (active return / tracking error).
    
    Measures skill in generating excess returns.
    
    Args:
        returns: Strategy returns
        benchmark_returns: Benchmark returns
        periods_per_year: Annualization factor
        
    Returns:
        Information ratio
    """
    if len(returns) == 0 or len(benchmark_returns) == 0:
        return 0.0
    
    # Ensure same length
    min_len = min(len(returns), len(benchmark_returns))
    returns = returns[:min_len]
    benchmark_returns = benchmark_returns[:min_len]
    
    # Calculate active returns
    active_returns = returns - benchmark_returns
    
    if len(active_returns) == 0:
        return 0.0
    
    mean_active = np.mean(active_returns)
    std_active = np.std(active_returns)
    
    if std_active == 0:
        return 0.0
    
    info_ratio = mean_active / std_active * np.sqrt(periods_per_year)
    return info_ratio


@njit(cache=True)
def calculate_kelly_fraction(
    win_probability: float,
    win_loss_ratio: float,
    kelly_multiplier: float = 0.25
) -> float:
    """
    Calculate Kelly criterion for position sizing.
    
    Gives mathematically optimal bet size for long-term growth.
    
    Args:
        win_probability: Probability of winning
        win_loss_ratio: Average win / average loss
        kelly_multiplier: Safety factor (0.25 = quarter Kelly)
        
    Returns:
        Optimal fraction of capital to risk
    """
    if win_probability <= 0 or win_probability >= 1:
        return 0.0
    
    if win_loss_ratio <= 0:
        return 0.0
    
    # Kelly formula: f = (p*b - q) / b
    # where p = win prob, q = loss prob, b = win/loss ratio
    q = 1 - win_probability
    kelly_full = (win_probability * win_loss_ratio - q) / win_loss_ratio
    
    # Apply safety multiplier (fractional Kelly)
    kelly_fraction = max(0.0, min(1.0, kelly_full * kelly_multiplier))
    
    return kelly_fraction


@njit(cache=True)
def calculate_risk_parity_weights(
    covariance_matrix: np.ndarray,
    target_risk: float = 1.0
) -> np.ndarray:
    """
    Calculate risk parity weights for portfolio.
    
    Equal risk contribution from each asset.
    
    Args:
        covariance_matrix: Asset covariance matrix
        target_risk: Target portfolio risk
        
    Returns:
        Risk parity weights
    """
    n_assets = covariance_matrix.shape[0]
    
    # Initialize with equal weights
    weights = np.ones(n_assets) / n_assets
    
    # Iterative algorithm (simplified)
    for _ in range(100):
        # Calculate marginal risk contributions
        portfolio_variance = weights @ covariance_matrix @ weights
        if portfolio_variance <= 0:
            break
        
        marginal_contrib = covariance_matrix @ weights
        risk_contrib = weights * marginal_contrib
        
        # Update weights
        target_contrib = portfolio_variance / n_assets
        weights = weights * target_contrib / (risk_contrib + 1e-10)
        
        # Normalize
        weights = weights / np.sum(weights)
    
    # Scale to target risk
    current_risk = np.sqrt(weights @ covariance_matrix @ weights)
    if current_risk > 0:
        weights = weights * target_risk / current_risk
    
    return weights


def calculate_portfolio_metrics(
    volumes: np.ndarray,
    returns: Optional[np.ndarray] = None,
    benchmark_returns: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Calculate comprehensive portfolio metrics.
    
    Args:
        volumes: Volume percentages
        returns: Historical or simulated returns
        benchmark_returns: Benchmark returns for comparison
        
    Returns:
        Dictionary of portfolio metrics
    """
    metrics = {}
    
    # Generate synthetic returns if not provided
    if returns is None:
        returns = generate_synthetic_returns(volumes)
    
    # Basic metrics
    metrics['mean_return'] = float(np.mean(returns))
    metrics['std_return'] = float(np.std(returns))
    metrics['skewness'] = float(calculate_skewness(returns))
    metrics['kurtosis'] = float(calculate_kurtosis(returns))
    
    # Risk-adjusted metrics
    metrics['sharpe_ratio'] = float(calculate_sharpe_ratio(returns))
    metrics['sortino_ratio'] = float(calculate_sortino_ratio(returns))
    metrics['calmar_ratio'] = float(calculate_calmar_ratio(returns))
    metrics['omega_ratio'] = float(calculate_omega_ratio(returns))
    
    # Benchmark comparison
    if benchmark_returns is not None:
        metrics['information_ratio'] = float(
            calculate_information_ratio(returns, benchmark_returns)
        )
        metrics['tracking_error'] = float(
            np.std(returns - benchmark_returns[:len(returns)])
        )
    
    # Risk metrics
    metrics['max_drawdown'] = float(calculate_max_drawdown(returns))
    metrics['var_95'] = float(np.percentile(returns, 5))  # Value at Risk
    
    # CVaR calculation (numba compatible)
    var_95 = np.percentile(returns, 5)
    below_var = returns[returns <= var_95]
    if len(below_var) > 0:
        metrics['cvar_95'] = float(np.mean(below_var))
    else:
        metrics['cvar_95'] = float(var_95)
    
    # Martingale-specific metrics
    metrics['recovery_efficiency'] = float(calculate_recovery_efficiency(volumes))
    metrics['tail_risk_ratio'] = float(calculate_tail_risk_ratio(volumes))
    
    return metrics


@njit(cache=True)
def generate_synthetic_returns(
    volumes: np.ndarray,
    n_scenarios: int = 1000,
    volatility: float = 0.2
) -> np.ndarray:
    """
    Generate synthetic returns for metric calculation.
    
    Args:
        volumes: Volume percentages
        n_scenarios: Number of scenarios to generate
        volatility: Market volatility
        
    Returns:
        Array of synthetic returns
    """
    n_orders = len(volumes)
    returns = np.zeros(n_scenarios)
    
    for i in range(n_scenarios):
        # Simulate price path
        price_changes = np.random.normal(0, volatility, n_orders)
        
        # Manual cumsum for price calculation
        cumsum_changes = np.zeros(n_orders)
        cumsum_changes[0] = price_changes[0] * 0.01
        for j in range(1, n_orders):
            cumsum_changes[j] = cumsum_changes[j-1] + price_changes[j] * 0.01
        
        prices = 100 * np.exp(cumsum_changes)
        
        # Calculate weighted average entry
        total_volume = 0.0
        for v in volumes:
            total_volume += v
        
        # Calculate weights manually
        weights = np.zeros(n_orders)
        for i in range(n_orders):
            weights[i] = volumes[i] / total_volume
        
        avg_entry = 0.0
        for i in range(n_orders):
            avg_entry += prices[i] * weights[i]
        
        # Random exit point
        exit_idx = np.random.randint(0, n_orders)
        exit_price = prices[exit_idx] * (1 + np.random.normal(0, 0.02))
        
        # Calculate return
        returns[i] = (exit_price - avg_entry) / avg_entry
    
    return returns


@njit(cache=True)
def calculate_sharpe_ratio(
    returns: np.ndarray,
    risk_free_rate: float = 0.0,
    periods_per_year: float = 252.0
) -> float:
    """Standard Sharpe ratio calculation."""
    if len(returns) == 0:
        return 0.0
    
    excess_returns = returns - risk_free_rate
    mean_excess = np.mean(excess_returns)
    std_returns = np.std(returns)
    
    if std_returns == 0:
        return 0.0
    
    sharpe = mean_excess / std_returns * np.sqrt(periods_per_year)
    return sharpe


@njit(cache=True)
def calculate_max_drawdown(returns: np.ndarray) -> float:
    """Calculate maximum drawdown from returns."""
    if len(returns) == 0:
        return 0.0
    
    # Manual cumprod for numba
    cumulative = np.ones(len(returns))
    cumulative[0] = 1 + returns[0]
    for i in range(1, len(returns)):
        cumulative[i] = cumulative[i-1] * (1 + returns[i])
    
    # Calculate running maximum (numba compatible)
    running_max = np.zeros_like(cumulative)
    running_max[0] = cumulative[0]
    for i in range(1, len(cumulative)):
        running_max[i] = max(running_max[i-1], cumulative[i])
    
    drawdown = (cumulative - running_max) / running_max
    
    return abs(np.min(drawdown))


@njit(cache=True)
def calculate_skewness(returns: np.ndarray) -> float:
    """Calculate skewness of returns."""
    if len(returns) < 3:
        return 0.0
    
    mean = np.mean(returns)
    std = np.std(returns)
    
    if std == 0:
        return 0.0
    
    skew = np.mean(((returns - mean) / std) ** 3)
    return skew


@njit(cache=True)
def calculate_kurtosis(returns: np.ndarray) -> float:
    """Calculate excess kurtosis of returns."""
    if len(returns) < 4:
        return 0.0
    
    mean = np.mean(returns)
    std = np.std(returns)
    
    if std == 0:
        return 0.0
    
    kurt = np.mean(((returns - mean) / std) ** 4) - 3
    return kurt


@njit(cache=True)
def calculate_recovery_efficiency(volumes: np.ndarray) -> float:
    """
    Calculate how efficiently the strategy recovers.
    
    Lower values = better (less volume needed to recover).
    """
    if len(volumes) == 0:
        return 0.0
    
    # Manual cumsum for numba compatibility
    cumsum = np.zeros(len(volumes))
    cumsum[0] = volumes[0]
    for i in range(1, len(volumes)):
        cumsum[i] = cumsum[i-1] + volumes[i]
    
    total = 0.0
    for v in volumes:
        total += v
    
    # Find point where 50% of profit potential is reached
    halfway_point = len(volumes)
    for i in range(len(volumes)):
        if cumsum[i] >= total * 0.5:
            halfway_point = i + 1
            break
    
    efficiency = halfway_point / len(volumes)
    return efficiency


@njit(cache=True)
def calculate_tail_risk_ratio(volumes: np.ndarray) -> float:
    """
    Calculate ratio of tail risk to total risk.
    
    Higher values = more risk concentrated in tail.
    """
    if len(volumes) < 4:
        return 0.0
    
    q4_start = 3 * len(volumes) // 4
    tail_sum = 0.0
    for i in range(q4_start, len(volumes)):
        tail_sum += volumes[i]
    
    total_sum = 0.0
    for v in volumes:
        total_sum += v
    
    if total_sum == 0:
        return 0.0
    
    return tail_sum / total_sum