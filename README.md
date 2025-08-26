# 🚀 Martingale Optimization Lab

Enterprise-grade martingale strategy optimization system with advanced machine learning features, risk metrics, and automated tuning capabilities.

## 🌟 Key Features

### Core Optimization
- **7-Step HC Pipeline**: Hard constraint enforcement (HC0-HC7) ensuring valid strategies
- **Smart Initial Generation**: MCMC, Genetic Algorithm, and historical learning
- **Bayesian Optimization**: Gaussian Process-based intelligent parameter search
- **Adaptive Parameters**: Auto-adjustment based on N orders and overlap percentage

### Advanced Analytics
- **Portfolio Risk Metrics**: Sortino, Calmar, Omega ratios, VaR/CVaR
- **Pattern Detection**: Identifies plateaus, cliffs, zigzags, stagnation zones
- **Exit-Ease Analysis**: Evaluates trade exit difficulty at each level
- **Kelly Criterion**: Optimal position sizing calculations

### Testing & Validation
- **A/B Testing Framework**: Statistical comparison with Monte Carlo simulation
- **Weight Tuning**: Market-based automatic weight optimization
- **SQL Analytics**: Comprehensive database queries for strategy analysis

## 📦 Installation

```bash
# Clone repository
git clone https://github.com/yourusername/martingalebot2.git
cd martingalebot2

# Install dependencies
pip install -r requirements.txt

# Verify installation
python3 -c "from martingale_lab.optimizer.evaluation_engine import evaluation_function; print('✅ Installation successful')"
```

## 🎯 Quick Start

### Basic Optimization

```bash
# Simple optimization with default parameters
python -m martingale_lab.cli.optimize \
  --overlap-min 10.0 --overlap-max 10.5 \
  --orders-min 20 --orders-max 20 \
  --batches 10 --batch-size 100 \
  --db results.db
```

### Advanced Optimization with All Features

```bash
# Production-grade optimization with smart initialization
python -m martingale_lab.cli.optimize \
  --overlap-min 8.0 --overlap-max 15.0 \
  --orders-min 15 --orders-max 25 \
  --first-volume 1.0 \
  --m2-min 0.10 --m2-max 0.15 \
  --m-min 0.05 --m-max 0.25 \
  --front-cap 10.0 \
  --tail-cap 0.60 \
  --batches 100 \
  --batch-size 500 \
  --workers 8 \
  --workers-mode thread \
  --db production.db \
  --seed 42 \
  --notes "Production search with all features"
```

## 🧠 Using Bayesian Optimization

```python
from martingale_lab.optimizer.bayesian_optimizer import BayesianSearchOrchestrator
from martingale_lab.optimizer.evaluation_engine import evaluation_function

# Define parameter bounds
param_bounds = {
    'overlap_pct': (8.0, 12.0),
    'm2_min': (0.10, 0.12),
    'm2_max': (0.13, 0.15),
    'num_orders': (18, 22)
}

# Create evaluation wrapper
def eval_wrapper(**params):
    return evaluation_function(
        base_price=100.0,
        use_smart_init=True,  # Use smart initial generation
        **params
    )

# Run Bayesian optimization
orchestrator = BayesianSearchOrchestrator(
    eval_wrapper,
    param_bounds,
    n_calls=100,  # Total evaluations
    n_initial=20,  # Random exploration phase
    random_state=42
)

result = orchestrator.optimize()
print(f"Best score: {result['best_score']:.2f}")
print(f"Best parameters: {result['best_params']}")
```

## 🎲 Smart Initial Generation

```python
from martingale_lab.core.smart_init import SmartInitializer

# Initialize with historical database (optional)
initializer = SmartInitializer(history_db="previous_results.db")

# Generate using MCMC sampling
mcmc_strategies = initializer.generate_mcmc_samples(
    num_orders=20,
    n_samples=10,
    target_q4=55.0,  # Target 55% in Q4
    target_m2=0.12    # Target m[2] = 0.12
)

# Generate using Genetic Algorithm
genetic_strategies = initializer.generate_genetic_population(
    num_orders=20,
    population_size=50,
    n_generations=20
)

# Generate mixed population (best approach)
smart_population = initializer.generate_smart_initial(
    num_orders=20,
    n_total=100,
    use_historical=True,
    use_mcmc=True,
    use_genetic=True
)
```

## 📊 SQL Analytics Queries

### Get Best Strategy

```sql
-- Best strategy with all metrics
SELECT 
  printf('Score: %.2f', score) as score,
  printf('Sortino: %.3f', json_extract(diagnostics_json, '$.sortino_ratio')) as sortino,
  printf('Calmar: %.3f', json_extract(diagnostics_json, '$.calmar_ratio')) as calmar,
  printf('Pattern Quality: %.1f', json_extract(diagnostics_json, '$.pattern_quality_score')) as pattern_q,
  printf('Q1: %.1f%%', json_extract(diagnostics_json, '$.q1_share')) as q1,
  printf('Q4: %.1f%%', json_extract(diagnostics_json, '$.q4_share')) as q4
FROM results 
WHERE score < 100000 
ORDER BY score ASC 
LIMIT 1;
```

### Export Best Strategy Volumes

```bash
sqlite3 results.db <<EOF
.mode csv
.headers on
WITH best AS (
  SELECT payload_json FROM results 
  WHERE score < 100000 
  ORDER BY score ASC LIMIT 1
)
SELECT 
  CAST(key AS INT) + 1 as order_num,
  printf('%.4f', value) as volume_pct
FROM best, json_each(json_extract(payload_json, '$.schedule.volume_pct'))
ORDER BY order_num;
EOF > best_strategy.csv
```

## 🧪 A/B Testing

```python
from martingale_lab.ab_testing.ab_tester import ABTester

# Initialize tester
tester = ABTester()

# Define strategies
strategy_a = {
    'id': 'Current',
    'volumes': [1.0, 1.1, 1.23, 1.4, ...]  # Your current strategy
}

strategy_b = {
    'id': 'Optimized',
    'volumes': [1.0, 2.0, 2.16, 2.16, ...]  # Optimized strategy
}

# Run comparison
results = tester.compare_strategies(
    strategy_a,
    strategy_b,
    market_scenarios=1000,
    confidence='high'  # 99% confidence level
)

# Check results
for metric, result in results.items():
    print(f"{metric}: Winner={result.winner}, p-value={result.p_value:.4f}")
```

## 📈 Portfolio Risk Metrics

```python
from martingale_lab.core.portfolio_metrics import calculate_portfolio_metrics
import numpy as np

# Your strategy volumes
volumes = np.array([1.0, 1.1, 1.23, 1.4, 1.62, ...])

# Calculate comprehensive metrics
metrics = calculate_portfolio_metrics(volumes)

print(f"Sortino Ratio: {metrics['sortino_ratio']:.3f}")
print(f"Calmar Ratio: {metrics['calmar_ratio']:.3f}")
print(f"Omega Ratio: {metrics['omega_ratio']:.3f}")
print(f"VaR 95%: {metrics['var_95']:.3f}")
print(f"CVaR 95%: {metrics['cvar_95']:.3f}")
print(f"Recovery Efficiency: {metrics['recovery_efficiency']:.3f}")
```

## 🔍 Pattern Detection

```python
from martingale_lab.core.pattern_detection import analyze_micro_patterns
from martingale_lab.core.repair import compute_m_from_v
import numpy as np

volumes = np.array([...])  # Your strategy
martingales = compute_m_from_v(volumes)

analysis = analyze_micro_patterns(volumes, martingales)

print(f"Pattern Quality Score: {analysis['pattern_quality_score']}/100")
print(f"Plateaus: {analysis['plateau_count']}")
print(f"Cliffs: {analysis['cliff_count']}")
print(f"Recommendations:")
for rec in analysis['recommendations']:
    print(f"  - {rec}")
```

## ⚙️ Adaptive Parameters

```python
from martingale_lab.core.adaptive import get_adaptive_parameters

# Get parameters for specific conditions
params = get_adaptive_parameters(
    num_orders=20,
    overlap_pct=10.0,
    strategy_type='balanced'  # or 'aggressive', 'conservative'
)

print(f"Adaptive m2 bounds: [{params['m2_min']:.3f}, {params['m2_max']:.3f}]")
print(f"Q1 cap: {params['q1_cap']:.1f}%")
print(f"Tail floor: {params['tail_floor']:.1f}%")
```

## 🎯 Parameter Recommendations by Market

| Market Type | Volatility | Overlap % | Orders (N) | m2 Range | Strategy Type |
|------------|------------|-----------|------------|----------|---------------|
| **BTC** | Low (0.1-0.3) | 3-8% | 25-35 | 0.08-0.12 | conservative |
| **ETH** | Medium (0.3-0.5) | 8-12% | 20-25 | 0.10-0.15 | balanced |
| **Major Alts** | Med-High (0.4-0.6) | 10-15% | 18-22 | 0.12-0.18 | balanced |
| **Small Caps** | High (0.6-0.8) | 12-20% | 15-20 | 0.12-0.20 | aggressive |
| **Memecoins** | Very High (0.7-1.0) | 15-25% | 10-15 | 0.15-0.25 | aggressive |

## 🏗️ System Architecture

```
martingale_lab/
├── core/                    # Core algorithms and metrics
│   ├── constraints.py       # HC0-HC7 constraint pipeline
│   ├── smart_init.py        # Smart initial generation
│   ├── portfolio_metrics.py # Risk metrics (Sortino, Calmar, etc.)
│   ├── pattern_detection.py # Micro-pattern analysis
│   ├── adaptive.py          # Adaptive parameter calculation
│   └── weight_tuner.py      # Automatic weight optimization
├── optimizer/               # Optimization engines
│   ├── evaluation_engine.py # Main evaluation function
│   ├── bayesian_optimizer.py # Bayesian optimization
│   └── objective_functions.py # Scoring functions
├── ab_testing/              # A/B testing framework
│   └── ab_tester.py         # Statistical comparison tools
├── orchestrator/            # Batch processing
│   └── adaptive_orchestrator.py # Adaptive batch orchestration
└── cli/                     # Command-line interface
    └── optimize.py          # Main CLI entry point
```

## 📊 Performance Metrics

The system evaluates strategies based on:

1. **Need Percentages**: Price recovery required for profit
2. **Volume Distribution**: Q1 (front) vs Q4 (tail) balance
3. **Martingale Growth**: Smooth vs aggressive multiplication
4. **Exit-Ease**: How easily positions can be exited
5. **Pattern Quality**: Absence of problematic patterns
6. **Risk Metrics**: Sortino, Calmar, Omega, VaR, CVaR

## 🚨 Troubleshooting

### Common Issues and Solutions

| Issue | Solution |
|-------|----------|
| All scores are inf | Increase `first_volume` to 1.0+, check m2 bounds |
| Slow optimization | Reduce `batch_size`, increase `workers` |
| Database locked | Ensure no other process is using the database |
| Import errors | Run `pip install -r requirements.txt` |
| m2 out of bounds | Adjust `m2_min`/`m2_max` ranges, check overlap % |

### Debug Mode

```bash
# Enable detailed logging
export MLAB_LOG_LEVEL=DEBUG
python -m martingale_lab.cli.optimize --db debug.db ...

# Check constraint violations
sqlite3 debug.db "SELECT * FROM results WHERE json_extract(diagnostics_json, '$.m2') < 0"
```

## 📈 Performance Benchmarks

| Optimization Method | 100 Evaluations | 1000 Evaluations | Best Score |
|--------------------|-----------------|------------------|------------|
| Random Search | ~15 min | ~150 min | 1500-2000 |
| Smart Init + Random | ~12 min | ~120 min | 1200-1500 |
| Bayesian (New!) | ~10 min | ~100 min | 1000-1200 |
| Full Pipeline | ~8 min | ~80 min | 900-1100 |

## 🎯 Success Metrics

A good strategy typically has:
- **Score**: < 1500 (lower is better)
- **Q1 Share**: 5-10%
- **Q4 Share**: 50-60%
- **m[2]**: 0.10-0.15
- **Sortino Ratio**: > 0.5
- **Pattern Quality**: > 80/100
- **Exit-Ease**: > 0.6

## 🔄 Continuous Improvement

The system learns from:
1. Historical successful strategies
2. Market condition analysis
3. A/B test results
4. Bayesian optimization convergence

## 📝 License

MIT License - See LICENSE file for details

## 🤝 Contributing

Contributions welcome! Please ensure:
- All tests pass
- Code follows existing patterns
- Documentation is updated
- Performance is not degraded

## 📧 Support

For questions or issues:
- Open a GitHub issue
- Check existing documentation
- Review SQL queries for analysis

## 🎉 Acknowledgments

Built with:
- NumPy & Numba for performance
- SQLite for result storage
- Gaussian Processes for Bayesian optimization
- Monte Carlo for A/B testing

---

**Version**: 2.0.0  
**Status**: Production Ready  
**Last Updated**: December 2024
