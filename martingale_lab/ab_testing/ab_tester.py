"""
A/B Testing Framework for Martingale Strategies

Automated comparison and statistical analysis of strategy performance.
"""

import numpy as np
import sqlite3
import json
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import hashlib
from scipy import stats


@dataclass
class TestResult:
    """Result of an A/B test"""
    strategy_a_id: str
    strategy_b_id: str
    metric: str
    a_mean: float
    b_mean: float
    a_std: float
    b_std: float
    t_statistic: float
    p_value: float
    confidence_level: float
    winner: str  # 'A', 'B', or 'NONE'
    improvement_pct: float
    sample_size: int
    test_date: str


class ABTester:
    """
    Automated A/B testing for martingale strategies.
    
    Features:
    - Statistical significance testing
    - Multiple metric comparison
    - Automatic winner selection
    - Historical tracking
    - Monte Carlo simulation
    """
    
    def __init__(self, 
                 db_path: str = "ab_tests.db",
                 confidence_level: float = 0.95,
                 min_samples: int = 30):
        self.db_path = db_path
        self.confidence_level = confidence_level
        self.min_samples = min_samples
        self._init_database()
        self.results_a = []
        self.results_b = []
        self.confidence_levels = {
            'low': 0.90,
            'medium': 0.95,
            'high': 0.99
        }
    
    def _init_database(self):
        """Initialize A/B test results database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ab_tests (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                test_id TEXT UNIQUE,
                strategy_a_id TEXT,
                strategy_b_id TEXT,
                strategy_a_json TEXT,
                strategy_b_json TEXT,
                results_json TEXT,
                winner TEXT,
                confidence_level REAL,
                test_date TEXT,
                notes TEXT
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS test_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                test_id TEXT,
                metric_name TEXT,
                a_value REAL,
                b_value REAL,
                p_value REAL,
                winner TEXT,
                FOREIGN KEY (test_id) REFERENCES ab_tests(test_id)
            )
        """)
        
        conn.commit()
        conn.close()
    
    def generate_test_id(self, strategy_a: Dict, strategy_b: Dict) -> str:
        """Generate unique test ID"""
        combined = f"{json.dumps(strategy_a, sort_keys=True)}_{json.dumps(strategy_b, sort_keys=True)}"
        return hashlib.md5(combined.encode()).hexdigest()[:12]
    
    def simulate_performance(
        self,
        volumes: np.ndarray,
        market_scenarios: int = 1000,
        volatility: float = 0.2,
        trend: float = 0.0,
        seed: Optional[int] = None
    ) -> Dict[str, np.ndarray]:
        """
        Simulate strategy performance across market scenarios.
        
        Args:
            volumes: Volume percentages
            market_scenarios: Number of scenarios to simulate
            volatility: Market volatility (0-1)
            trend: Market trend (-1 to 1)
            seed: Random seed for reproducibility
            
        Returns:
            Dictionary of performance metrics
        """
        if seed is not None:
            np.random.seed(seed)
        
        results = {
            'returns': np.zeros(market_scenarios),
            'max_drawdown': np.zeros(market_scenarios),
            'recovery_time': np.zeros(market_scenarios),
            'sharpe_ratio': np.zeros(market_scenarios),
            'win_rate': np.zeros(market_scenarios),
            'avg_profit': np.zeros(market_scenarios),
            'exit_efficiency': np.zeros(market_scenarios)
        }
        
        n_orders = len(volumes)
        
        for scenario in range(market_scenarios):
            # Generate price path
            price_path = self._generate_price_path(
                n_orders * 2,  # Simulate twice the orders for recovery
                volatility,
                trend
            )
            
            # Simulate trading
            trades = self._simulate_trades(volumes, price_path)
            
            # Calculate metrics
            results['returns'][scenario] = trades['total_return']
            results['max_drawdown'][scenario] = trades['max_drawdown']
            results['recovery_time'][scenario] = trades['recovery_time']
            results['sharpe_ratio'][scenario] = trades['sharpe_ratio']
            results['win_rate'][scenario] = trades['win_rate']
            results['avg_profit'][scenario] = trades['avg_profit']
            results['exit_efficiency'][scenario] = trades['exit_efficiency']
        
        return results
    
    def _generate_price_path(
        self,
        length: int,
        volatility: float,
        trend: float
    ) -> np.ndarray:
        """Generate realistic price path"""
        dt = 1.0 / 252  # Daily steps
        drift = trend * dt
        diffusion = volatility * np.sqrt(dt)
        
        # Geometric Brownian Motion
        returns = np.random.normal(drift, diffusion, length)
        price_path = 100 * np.exp(np.cumsum(returns))
        
        return price_path
    
    def _simulate_trades(
        self,
        volumes: np.ndarray,
        price_path: np.ndarray
    ) -> Dict[str, float]:
        """Simulate trading with given volumes and prices"""
        n_orders = len(volumes)
        entry_prices = price_path[:n_orders]
        exit_prices = price_path[n_orders:]
        
        # Calculate weighted average entry
        total_volume = np.sum(volumes)
        weights = volumes / total_volume
        avg_entry = np.sum(entry_prices * weights)
        
        # Simulate different exit points
        profits = []
        for i in range(min(n_orders, len(exit_prices))):
            exit_price = exit_prices[i]
            # Calculate profit if exiting at order i
            volume_exited = np.sum(volumes[:i+1])
            profit = (exit_price - avg_entry) / avg_entry * (volume_exited / total_volume)
            profits.append(profit)
        
        if not profits:
            return {
                'total_return': 0.0,
                'max_drawdown': 0.0,
                'recovery_time': 0.0,
                'sharpe_ratio': 0.0,
                'win_rate': 0.0,
                'avg_profit': 0.0,
                'exit_efficiency': 0.0
            }
        
        # Calculate metrics
        returns = np.array(profits)
        positive_returns = returns[returns > 0]
        negative_returns = returns[returns < 0]
        
        total_return = np.mean(returns) if len(returns) > 0 else 0
        max_drawdown = np.min(returns) if len(returns) > 0 else 0
        
        # Recovery time (orders needed to profit)
        recovery_time = n_orders
        for i, r in enumerate(returns):
            if r > 0:
                recovery_time = i + 1
                break
        
        # Sharpe ratio
        if np.std(returns) > 0:
            sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)
        else:
            sharpe_ratio = 0
        
        # Win rate
        win_rate = len(positive_returns) / max(1, len(returns))
        
        # Average profit
        avg_profit = np.mean(positive_returns) if len(positive_returns) > 0 else 0
        
        # Exit efficiency (how quickly can exit profitably)
        exit_efficiency = 1.0 / (recovery_time / n_orders)
        
        return {
            'total_return': total_return,
            'max_drawdown': abs(max_drawdown),
            'recovery_time': recovery_time,
            'sharpe_ratio': sharpe_ratio,
            'win_rate': win_rate,
            'avg_profit': avg_profit,
            'exit_efficiency': exit_efficiency
        }
    
    def compare_strategies(
        self,
        strategy_a: Dict[str, Any],
        strategy_b: Dict[str, Any],
        market_scenarios: int = 1000,
        confidence: str = 'medium',
        metrics: Optional[List[str]] = None
    ) -> Dict[str, TestResult]:
        """
        Compare two strategies across multiple metrics.
        
        Args:
            strategy_a: First strategy (volumes, params)
            strategy_b: Second strategy (volumes, params)
            market_scenarios: Number of scenarios to test
            confidence: Confidence level ('low', 'medium', 'high')
            metrics: Specific metrics to test (None = all)
            
        Returns:
            Dictionary of test results per metric
        """
        if metrics is None:
            metrics = ['returns', 'sharpe_ratio', 'max_drawdown', 
                      'win_rate', 'exit_efficiency']
        
        # Extract volumes
        volumes_a = np.array(strategy_a['volumes'])
        volumes_b = np.array(strategy_b['volumes'])
        
        # Simulate performance for both strategies
        perf_a = self.simulate_performance(volumes_a, market_scenarios)
        perf_b = self.simulate_performance(volumes_b, market_scenarios)
        
        # Run statistical tests
        test_results = {}
        confidence_level = self.confidence_levels[confidence]
        
        for metric in metrics:
            if metric not in perf_a or metric not in perf_b:
                continue
            
            # Get metric values
            a_values = perf_a[metric]
            b_values = perf_b[metric]
            
            # Perform t-test
            t_stat, p_value = stats.ttest_ind(a_values, b_values)
            
            # Determine winner
            a_mean = np.mean(a_values)
            b_mean = np.mean(b_values)
            
            # For drawdown and recovery time, lower is better
            if metric in ['max_drawdown', 'recovery_time']:
                if p_value < (1 - confidence_level):
                    winner = 'A' if a_mean < b_mean else 'B'
                else:
                    winner = 'NONE'
                improvement_pct = (b_mean - a_mean) / max(abs(a_mean), 0.001) * 100
            else:
                # For other metrics, higher is better
                if p_value < (1 - confidence_level):
                    winner = 'B' if b_mean > a_mean else 'A'
                else:
                    winner = 'NONE'
                improvement_pct = (b_mean - a_mean) / max(abs(a_mean), 0.001) * 100
            
            test_results[metric] = TestResult(
                strategy_a_id=strategy_a.get('id', 'A'),
                strategy_b_id=strategy_b.get('id', 'B'),
                metric=metric,
                a_mean=a_mean,
                b_mean=b_mean,
                a_std=np.std(a_values),
                b_std=np.std(b_values),
                t_statistic=t_stat,
                p_value=p_value,
                confidence_level=confidence_level,
                winner=winner,
                improvement_pct=improvement_pct,
                sample_size=market_scenarios,
                test_date=datetime.now().isoformat()
            )
        
        return test_results
    
    def run_automated_test(
        self,
        db_path: str,
        reference_strategy: Optional[Dict] = None,
        top_n: int = 5,
        confidence: str = 'medium'
    ) -> List[Dict[str, Any]]:
        """
        Automatically test top strategies from optimization database.
        
        Args:
            db_path: Path to optimization results database
            reference_strategy: Reference strategy to compare against
            top_n: Number of top strategies to test
            confidence: Confidence level for tests
            
        Returns:
            List of test results
        """
        # Load top strategies from database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT payload_json, score, diagnostics_json
            FROM results
            WHERE score < 100000
            ORDER BY score ASC
            LIMIT ?
        """, (top_n,))
        
        strategies = []
        for payload_json, score, diag_json in cursor.fetchall():
            payload = json.loads(payload_json)
            diagnostics = json.loads(diag_json)
            
            strategies.append({
                'id': f"Score_{score:.0f}",
                'volumes': payload['schedule']['volume_pct'],
                'score': score,
                'diagnostics': diagnostics
            })
        
        conn.close()
        
        # Run tests
        all_results = []
        
        if reference_strategy:
            # Test all against reference
            for strategy in strategies:
                results = self.compare_strategies(
                    reference_strategy,
                    strategy,
                    confidence=confidence
                )
                all_results.append({
                    'comparison': f"Reference vs {strategy['id']}",
                    'results': results
                })
        
        # Test top strategies against each other
        for i in range(len(strategies) - 1):
            for j in range(i + 1, min(i + 2, len(strategies))):  # Adjacent pairs
                results = self.compare_strategies(
                    strategies[i],
                    strategies[j],
                    confidence=confidence
                )
                all_results.append({
                    'comparison': f"{strategies[i]['id']} vs {strategies[j]['id']}",
                    'results': results
                })
        
        # Save results to database
        self._save_test_results(all_results)
        
        return all_results
    
    def _save_test_results(self, results: List[Dict]):
        """Save test results to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for test in results:
            test_id = hashlib.md5(
                f"{test['comparison']}_{datetime.now().isoformat()}".encode()
            ).hexdigest()[:12]
            
            # Aggregate winner across all metrics
            winners = {}
            for metric, result in test['results'].items():
                winner = result.winner
                winners[winner] = winners.get(winner, 0) + 1
            
            overall_winner = max(winners.items(), key=lambda x: x[1])[0]
            
            cursor.execute("""
                INSERT INTO ab_tests 
                (test_id, results_json, winner, test_date, notes)
                VALUES (?, ?, ?, ?, ?)
            """, (
                test_id,
                json.dumps({k: v.__dict__ for k, v in test['results'].items()}),
                overall_winner,
                datetime.now().isoformat(),
                test['comparison']
            ))
            
            # Save individual metrics
            for metric, result in test['results'].items():
                cursor.execute("""
                    INSERT INTO test_metrics
                    (test_id, metric_name, a_value, b_value, p_value, winner)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    test_id,
                    metric,
                    result.a_mean,
                    result.b_mean,
                    result.p_value,
                    result.winner
                ))
        
        conn.commit()
        conn.close()
    
    def get_test_summary(self) -> str:
        """Get summary of all A/B tests"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT 
                notes as comparison,
                winner,
                COUNT(*) as metric_count,
                AVG(CASE WHEN m.winner != 'NONE' THEN 1 ELSE 0 END) as significance_rate,
                MIN(m.p_value) as min_p_value,
                MAX(ABS(m.b_value - m.a_value) / m.a_value) as max_improvement
            FROM ab_tests t
            JOIN test_metrics m ON t.test_id = m.test_id
            GROUP BY t.test_id
            ORDER BY t.test_date DESC
            LIMIT 10
        """)
        
        results = cursor.fetchall()
        conn.close()
        
        summary = "A/B TEST SUMMARY\n"
        summary += "=" * 60 + "\n\n"
        
        for comp, winner, metrics, sig_rate, min_p, max_imp in results:
            summary += f"Test: {comp}\n"
            summary += f"  Winner: {winner}\n"
            summary += f"  Metrics tested: {metrics}\n"
            summary += f"  Significance rate: {sig_rate:.1%}\n"
            summary += f"  Min p-value: {min_p:.4f}\n"
            summary += f"  Max improvement: {max_imp:.1%}\n\n"
        
        return summary
    
    def recommend_strategy(
        self,
        market_volatility: float,
        market_trend: float,
        risk_tolerance: str = 'medium'
    ) -> Dict[str, Any]:
        """
        Recommend best strategy based on market conditions and test results.
        
        Args:
            market_volatility: Current market volatility (0-1)
            market_trend: Current market trend (-1 to 1)
            risk_tolerance: 'low', 'medium', 'high'
            
        Returns:
            Recommended strategy with reasoning
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Weight metrics based on market conditions
        weights = {}
        
        if market_volatility > 0.7:  # High volatility
            weights['sharpe_ratio'] = 3.0
            weights['max_drawdown'] = 2.0
            weights['exit_efficiency'] = 2.5
            weights['win_rate'] = 1.5
        elif market_volatility < 0.3:  # Low volatility
            weights['returns'] = 3.0
            weights['sharpe_ratio'] = 2.0
            weights['win_rate'] = 2.0
        else:  # Medium volatility
            weights['sharpe_ratio'] = 2.5
            weights['returns'] = 2.0
            weights['exit_efficiency'] = 2.0
            weights['max_drawdown'] = 1.5
        
        # Adjust for trend
        if market_trend < -0.5:  # Strong downtrend
            weights['exit_efficiency'] *= 1.5
            weights['max_drawdown'] *= 1.3
        elif market_trend > 0.5:  # Strong uptrend
            weights['returns'] *= 1.5
            weights['win_rate'] *= 1.2
        
        # Find best performing strategies
        cursor.execute("""
            SELECT 
                t.notes,
                m.metric_name,
                m.winner,
                m.b_value - m.a_value as improvement
            FROM ab_tests t
            JOIN test_metrics m ON t.test_id = m.test_id
            WHERE m.winner != 'NONE'
        """)
        
        strategy_scores = {}
        for notes, metric, winner, improvement in cursor.fetchall():
            strategy = notes.split(' vs ')[1] if winner == 'B' else notes.split(' vs ')[0]
            
            if strategy not in strategy_scores:
                strategy_scores[strategy] = 0
            
            # Add weighted score
            weight = weights.get(metric, 1.0)
            strategy_scores[strategy] += weight * abs(improvement)
        
        conn.close()
        
        if not strategy_scores:
            return {
                'recommendation': 'No tested strategies available',
                'reasoning': 'Run A/B tests first'
            }
        
        # Get best strategy
        best_strategy = max(strategy_scores.items(), key=lambda x: x[1])
        
        reasoning = f"Based on market conditions (volatility={market_volatility:.1f}, "
        reasoning += f"trend={market_trend:+.1f}) and risk tolerance={risk_tolerance}, "
        reasoning += f"strategy '{best_strategy[0]}' is recommended with score {best_strategy[1]:.2f}"
        
        return {
            'recommendation': best_strategy[0],
            'score': best_strategy[1],
            'reasoning': reasoning,
            'strategy_scores': strategy_scores,
            'market_conditions': {
                'volatility': market_volatility,
                'trend': market_trend
            }
        }
    
    def add_result(self, strategy: str, value: float):
        """
        Add a result for simple A/B testing.
        
        Args:
            strategy: 'A' or 'B'
            value: Result value (e.g., return, score)
        """
        if strategy == 'A':
            self.results_a.append(value)
        elif strategy == 'B':
            self.results_b.append(value)
        else:
            raise ValueError(f"Strategy must be 'A' or 'B', got {strategy}")
    
    def get_results(self) -> Dict[str, Any]:
        """
        Get simple A/B test results.
        
        Returns:
            Dictionary with winner, confidence, and statistics
        """
        if len(self.results_a) < self.min_samples or len(self.results_b) < self.min_samples:
            return {
                'winner': 'INSUFFICIENT_DATA',
                'confidence': 0.0,
                'message': f'Need at least {self.min_samples} samples per strategy'
            }
        
        # Perform t-test
        t_stat, p_value = stats.ttest_ind(self.results_a, self.results_b)
        
        # Determine winner
        mean_a = np.mean(self.results_a)
        mean_b = np.mean(self.results_b)
        
        if p_value < (1 - self.confidence_level):
            winner = 'A' if mean_a > mean_b else 'B'
            confidence = 1 - p_value
        else:
            winner = 'NO_DIFFERENCE'
            confidence = p_value
        
        return {
            'winner': winner,
            'confidence': confidence,
            'p_value': p_value,
            't_statistic': t_stat,
            'mean_a': mean_a,
            'mean_b': mean_b,
            'std_a': np.std(self.results_a),
            'std_b': np.std(self.results_b),
            'samples_a': len(self.results_a),
            'samples_b': len(self.results_b)
        }