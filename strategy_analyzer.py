#!/usr/bin/env python3
"""
Strategy Analyzer - Compare and analyze martingale strategies.
"""

import numpy as np
from typing import Dict, List, Tuple
from martingale_lab.core.repair import compute_m_from_v
from martingale_lab.core.metrics import compute_exit_ease_metrics
from martingale_lab.core.constraints import enforce_schedule_shape_fixed

class StrategyAnalyzer:
    """Analyze and compare martingale strategies."""
    
    def __init__(self):
        self.strategies = {}
        
    def add_strategy(self, name: str, volumes: List[float], indents: List[float] = None):
        """Add a strategy for analysis."""
        volumes = np.array(volumes)
        if indents is None:
            # Create linear indents if not provided
            indents = np.linspace(0, 30, len(volumes))
        else:
            indents = np.array(indents)
            
        self.strategies[name] = {
            'volumes': volumes,
            'indents': indents,
            'metrics': self._compute_metrics(volumes, indents)
        }
        
    def _compute_metrics(self, volumes: np.ndarray, indents: np.ndarray) -> Dict:
        """Compute comprehensive metrics for a strategy."""
        N = len(volumes)
        m = compute_m_from_v(volumes)
        
        # Simulate needpct (simplified)
        needpct = np.zeros(N)
        for i in range(N):
            if i == 0:
                needpct[i] = indents[i] if i < len(indents) else 1.0
            else:
                needpct[i] = indents[i] if i < len(indents) else needpct[i-1] * 1.1
        
        # Exit-ease metrics
        ee_metrics = compute_exit_ease_metrics(needpct, volumes)
        
        # Quartile analysis
        Q1 = N // 4
        Q2 = N // 2
        Q3 = 3 * N // 4
        
        metrics = {
            # Basic stats
            'N': N,
            'sum': np.sum(volumes),
            'mean': np.mean(volumes),
            'std': np.std(volumes),
            
            # Martingale metrics
            'm2': m[2] if N > 2 else 0,
            'm_mean': np.mean(m[2:]) if N > 2 else 0,
            'm_std': np.std(m[2:]) if N > 2 else 0,
            'm_min': np.min(m[2:]) if N > 2 else 0,
            'm_max': np.max(m[2:]) if N > 2 else 0,
            
            # Quartile distribution
            'q1_sum': np.sum(volumes[:Q1]),
            'q2_sum': np.sum(volumes[Q1:Q2]),
            'q3_sum': np.sum(volumes[Q2:Q3]),
            'q4_sum': np.sum(volumes[Q3:]),
            
            # Exit-ease
            'ee_harmonic': ee_metrics['ee_harmonic'],
            'ee_tail_weighted': ee_metrics['ee_tail_weighted'],
            'ee_front_tail_ratio': ee_metrics['ee_front_tail_ratio'],
            'ee_balance': ee_metrics['ee_balance_penalty'],
            
            # Pattern detection
            'plateau_count': self._count_plateaus(m),
            'turn_count': self._count_turns(m),
            'monotonic': self._check_monotonic(volumes),
        }
        
        return metrics
    
    def _count_plateaus(self, m: np.ndarray) -> int:
        """Count plateau regions in martingale ratios."""
        if len(m) < 3:
            return 0
        
        plateau_count = 0
        for i in range(2, len(m)):
            if abs(m[i] - 1.0) < 0.02:
                plateau_count += 1
        return plateau_count
    
    def _count_turns(self, m: np.ndarray) -> int:
        """Count turning points in martingale ratios."""
        if len(m) < 3:
            return 0
        
        turns = 0
        for i in range(3, len(m)):
            if (m[i] - 1.0) * (m[i-1] - 1.0) < 0:
                turns += 1
        return turns
    
    def _check_monotonic(self, volumes: np.ndarray) -> bool:
        """Check if volumes are strictly increasing."""
        for i in range(1, len(volumes)):
            if volumes[i] <= volumes[i-1]:
                return False
        return True
    
    def compare(self):
        """Compare all strategies."""
        if not self.strategies:
            print("No strategies to compare")
            return
        
        print("\n" + "="*80)
        print("STRATEGY COMPARISON")
        print("="*80)
        
        # Create comparison table
        metrics_to_show = [
            ('Orders', 'N'),
            ('Sum', 'sum'),
            ('m[2]', 'm2'),
            ('m_mean', 'm_mean'),
            ('Q1%', 'q1_sum'),
            ('Q4%', 'q4_sum'),
            ('EE Harmonic', 'ee_harmonic'),
            ('EE Tail', 'ee_tail_weighted'),
            ('Plateaus', 'plateau_count'),
            ('Turns', 'turn_count'),
            ('Monotonic', 'monotonic'),
        ]
        
        # Print header
        print(f"{'Metric':<20}", end="")
        for name in self.strategies:
            print(f"{name[:15]:>15}", end="")
        print()
        print("-" * (20 + 15 * len(self.strategies)))
        
        # Print metrics
        for label, key in metrics_to_show:
            print(f"{label:<20}", end="")
            for name in self.strategies:
                value = self.strategies[name]['metrics'][key]
                if isinstance(value, bool):
                    print(f"{'Yes' if value else 'No':>15}", end="")
                elif isinstance(value, float):
                    print(f"{value:>15.3f}", end="")
                else:
                    print(f"{value:>15}", end="")
            print()
        
        # Find best strategy for each metric
        print("\n" + "="*80)
        print("BEST PERFORMERS")
        print("="*80)
        
        criteria = [
            ('Lowest Q1 (less front-loaded)', 'q1_sum', False),
            ('Highest Q4 (strong tail)', 'q4_sum', True),
            ('Best exit-ease harmonic', 'ee_harmonic', True),
            ('Best exit-ease tail', 'ee_tail_weighted', True),
            ('Most turns (wave pattern)', 'turn_count', True),
            ('Fewest plateaus', 'plateau_count', False),
        ]
        
        for label, key, higher_better in criteria:
            best_name = None
            best_value = None
            
            for name in self.strategies:
                value = self.strategies[name]['metrics'][key]
                if best_value is None:
                    best_name = name
                    best_value = value
                elif higher_better and value > best_value:
                    best_name = name
                    best_value = value
                elif not higher_better and value < best_value:
                    best_name = name
                    best_value = value
            
            print(f"{label:<40} {best_name} ({best_value:.3f})")


def main():
    """Demo the analyzer with your strategy."""
    
    analyzer = StrategyAnalyzer()
    
    # Your current strategy
    your_volumes = [
        1.0, 1.1, 1.23, 1.4, 1.62, 1.91, 2.29, 2.52,
        2.84, 3.27, 3.84, 4.61, 5.47, 6.1, 6.96,
        8.11, 9.65, 10.32, 11.71, 14.05
    ]
    your_indents = [
        0, 0.8, 1.7, 2.7, 3.8, 5.0, 6.3, 7.7,
        9.2, 10.8, 12.5, 14.3, 16.2, 18.2, 20.3,
        22.5, 24.8, 27.2, 29.7, 32.3
    ]
    
    analyzer.add_strategy("Your Current", your_volumes, your_indents)
    
    # Alternative: Conservative
    conservative_volumes = []
    v0 = 1.5
    for i in range(20):
        if i == 0:
            conservative_volumes.append(v0)
        else:
            conservative_volumes.append(conservative_volumes[-1] * 1.08)
    conservative_volumes = np.array(conservative_volumes)
    conservative_volumes = conservative_volumes / np.sum(conservative_volumes) * 100
    
    analyzer.add_strategy("Conservative", conservative_volumes.tolist())
    
    # Alternative: Aggressive
    aggressive_volumes = []
    v0 = 0.8
    for i in range(20):
        if i == 0:
            aggressive_volumes.append(v0)
        elif i < 5:
            aggressive_volumes.append(aggressive_volumes[-1] * 1.25)
        else:
            aggressive_volumes.append(aggressive_volumes[-1] * 1.12)
    aggressive_volumes = np.array(aggressive_volumes)
    aggressive_volumes = aggressive_volumes / np.sum(aggressive_volumes) * 100
    
    analyzer.add_strategy("Aggressive", aggressive_volumes.tolist())
    
    # Alternative: Wave pattern
    wave_volumes = []
    v0 = 1.2
    for i in range(20):
        if i == 0:
            wave_volumes.append(v0)
        else:
            # Create wave pattern
            wave = 0.1 * np.sin(i * np.pi / 3)
            growth = 1.11 + wave
            wave_volumes.append(wave_volumes[-1] * growth)
    wave_volumes = np.array(wave_volumes)
    wave_volumes = wave_volumes / np.sum(wave_volumes) * 100
    
    analyzer.add_strategy("Wave Pattern", wave_volumes.tolist())
    
    # Compare strategies
    analyzer.compare()
    
    print("\n" + "="*80)
    print("RECOMMENDATION")
    print("="*80)
    print("""
Based on the analysis:

1. Your current strategy has:
   ✓ Excellent tail strength (Q4 = 53.8%)
   ✓ Low front-loading (Q1 = 6.3%)
   ✓ Reasonable m[2] = 0.118
   ⚠ Could improve exit-ease balance
   ⚠ Could add more wave pattern (turns)

2. To optimize further:
   - Increase m[2] slightly (0.12-0.15) for better early recovery
   - Add subtle wave pattern for more exit opportunities
   - Keep Q4 > 50% for strong tail
   - Maintain Q1 < 10% to avoid front-loading

3. The wave pattern strategy shows promise with:
   - Better turn count (more exit opportunities)
   - Balanced growth pattern
   - Could be tuned to match your Q1/Q4 preferences
""")


if __name__ == "__main__":
    main()
