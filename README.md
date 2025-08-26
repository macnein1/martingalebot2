# Martingale Optimization Lab 🚀

Advanced martingale strategy optimization system with adaptive parameters, pattern detection, and automated tuning.

## 🎯 Features

- **HC Pipeline**: 7-step hard constraint enforcement (HC0-HC7)
- **Adaptive Parameters**: Automatic adjustment based on N and overlap
- **Pattern Detection**: Identifies and penalizes problematic micro-patterns
- **Exit-Ease Metrics**: Evaluates trade exit difficulty at each level
- **Weight Tuning**: Automatic optimization of scoring weights
- **SQL Analytics**: Comprehensive queries for strategy analysis

## 📦 Installation

```bash
# Clone repository
git clone https://github.com/yourusername/martingalebot2.git
cd martingalebot2

# Install dependencies
pip install -r requirements.txt
```

## 🚀 Quick Start

### Basic Optimization

```bash
# Simple optimization with default parameters
python -m martingale_lab.cli.optimize \
  --overlap-min 10.0 --overlap-max 10.5 \
  --orders-min 20 --orders-max 20 \
  --batches 10 --batch-size 100 \
  --db results.db
```

### Advanced Optimization with Adaptive Parameters

```bash
# Balanced strategy (like your current strategy)
python -m martingale_lab.cli.optimize \
  --overlap-min 10.0 --overlap-max 12.0 \
  --orders-min 18 --orders-max 22 \
  --use-adaptive \
  --strategy-type balanced \
  --first-volume 1.0 \
  --m2-min 0.10 --m2-max 0.15 \
  --batches 20 --batch-size 200 \
  --workers 4 \
  --db results_balanced.db \
  --notes "Balanced strategy search"

# Aggressive strategy for high volatility
python -m martingale_lab.cli.optimize \
  --overlap-min 15.0 --overlap-max 20.0 \
  --orders-min 15 --orders-max 20 \
  --use-adaptive \
  --strategy-type aggressive \
  --first-volume 0.8 \
  --m2-min 0.12 --m2-max 0.25 \
  --batches 20 --batch-size 200 \
  --db results_aggressive.db \
  --notes "Aggressive for volatile markets"

# Conservative strategy for stable markets
python -m martingale_lab.cli.optimize \
  --overlap-min 5.0 --overlap-max 8.0 \
  --orders-min 25 --orders-max 30 \
  --use-adaptive \
  --strategy-type conservative \
  --first-volume 1.2 \
  --m2-min 0.08 --m2-max 0.12 \
  --batches 20 --batch-size 200 \
  --db results_conservative.db \
  --notes "Conservative for stable markets"
```

## 📊 SQL Queries - Strategy Analysis

### 1. Get Best Strategy

```bash
# Show the absolute best strategy found
sqlite3 results.db <<EOF
SELECT 
  printf('Score: %.2f', score) as score,
  printf('Q1: %.1f%%', json_extract(diagnostics_json, '$.q1_share')) as q1,
  printf('Q4: %.1f%%', json_extract(diagnostics_json, '$.q4_share')) as q4,
  printf('m[2]: %.3f', json_extract(diagnostics_json, '$.m2')) as m2,
  printf('Exit-ease: %.3f', json_extract(diagnostics_json, '$.ee_tail_weighted')) as exit_ease
FROM results 
WHERE score < 100000 
ORDER BY score ASC 
LIMIT 1;
EOF
```

### 2. Export Best Strategy Volumes

```bash
# Get volume percentages for the best strategy
sqlite3 results.db <<EOF
.mode csv
.headers off
WITH best AS (
  SELECT payload_json FROM results 
  WHERE score < 100000 
  ORDER BY score ASC LIMIT 1
)
SELECT 
  CAST(key AS INT) + 1 as order_num,
  printf('%.2f', value) as volume_pct
FROM best, json_each(json_extract(payload_json, '$.schedule.volume_pct'))
ORDER BY order_num;
EOF
```

### 3. Top 10 Strategies Comparison

```bash
# Compare top 10 strategies
sqlite3 results.db <<EOF
.mode column
.headers on
WITH top_strategies AS (
  SELECT *, ROW_NUMBER() OVER (ORDER BY score ASC) as rank
  FROM results WHERE score < 100000
)
SELECT 
  rank,
  printf('%.2f', score) as score,
  printf('%.1f', json_extract(diagnostics_json, '$.q1_share')) as q1,
  printf('%.1f', json_extract(diagnostics_json, '$.q4_share')) as q4,
  printf('%.3f', json_extract(diagnostics_json, '$.m2')) as m2
FROM top_strategies WHERE rank <= 10;
EOF
```

### 4. Find Strategies Similar to Yours

```bash
# Find strategies with characteristics similar to your reference
sqlite3 results.db <<EOF
SELECT COUNT(*) as matches,
  printf('%.2f', MIN(score)) as best_score,
  printf('%.2f', AVG(score)) as avg_score
FROM results 
WHERE score < 100000
  AND json_extract(diagnostics_json, '$.q1_share') BETWEEN 5 AND 10
  AND json_extract(diagnostics_json, '$.q4_share') BETWEEN 50 AND 60
  AND json_extract(diagnostics_json, '$.m2') BETWEEN 0.10 AND 0.15
  AND json_extract(diagnostics_json, '$.ee_tail_weighted') > 0.5;
EOF
```

### 5. Pattern Analysis of Best Strategies

```bash
# Analyze patterns in top strategies
sqlite3 results.db <<EOF
SELECT 
  printf('%.2f', score) as score,
  json_extract(diagnostics_json, '$.pattern_quality_score') as quality,
  json_extract(diagnostics_json, '$.pattern_plateaus') as plateaus,
  json_extract(diagnostics_json, '$.pattern_cliffs') as cliffs,
  json_extract(diagnostics_json, '$.pattern_zigzag') as zigzag
FROM results 
WHERE score < 100000 
ORDER BY score ASC 
LIMIT 5;
EOF
```

### 6. All SQL Queries at Once

```bash
# Run all analysis queries
sqlite3 results.db < sql_queries.sql
```

## 🎯 Strategy Comparison Tool

```python
# compare_strategies.py
from strategy_analyzer import StrategyAnalyzer

analyzer = StrategyAnalyzer()

# Your current strategy
your_strategy = [1.0, 1.1, 1.23, 1.4, 1.62, 1.91, 2.29, 2.52,
                 2.84, 3.27, 3.84, 4.61, 5.47, 6.1, 6.96,
                 8.11, 9.65, 10.32, 11.71, 14.05]

analyzer.add_strategy("Your Current", your_strategy)

# Add optimized strategies from database
import sqlite3
import json

conn = sqlite3.connect('results.db')
cursor = conn.cursor()

# Get top 3 strategies
cursor.execute("""
    SELECT payload_json, score 
    FROM results 
    WHERE score < 100000 
    ORDER BY score ASC 
    LIMIT 3
""")

for i, (payload_json, score) in enumerate(cursor.fetchall(), 1):
    payload = json.loads(payload_json)
    volumes = payload['schedule']['volume_pct']
    analyzer.add_strategy(f"Optimized #{i} (score={score:.0f})", volumes)

# Compare all strategies
analyzer.compare()
```

## 🔧 Parameter Ranges

### For Different Market Conditions

| Market Type | Overlap % | Orders (N) | m2 Range | Strategy Type |
|------------|-----------|------------|----------|---------------|
| BTC (Stable) | 3-8% | 25-35 | 0.08-0.12 | conservative |
| ETH (Medium) | 8-12% | 20-25 | 0.10-0.15 | balanced |
| Altcoins (Volatile) | 12-20% | 15-20 | 0.12-0.20 | aggressive |
| Memecoins (Extreme) | 15-25% | 10-15 | 0.15-0.25 | aggressive |

### Recommended Search Parameters

```bash
# Large-scale search for production
python -m martingale_lab.cli.optimize \
  --overlap-min 8.0 --overlap-max 15.0 --overlap-step 0.5 \
  --orders-min 15 --orders-max 25 --orders-step 1 \
  --use-adaptive \
  --strategy-type balanced \
  --first-volume 1.0 \
  --m2-min 0.10 --m2-max 0.18 \
  --m-min 0.05 --m-max 0.25 \
  --front-cap 10.0 \
  --tail-cap 0.60 \
  --batches 100 \
  --batch-size 500 \
  --workers 8 \
  --workers-mode thread \
  --db production_search.db \
  --seed 42 \
  --notes "Production search - millions of variants"
```

## 📈 Performance Metrics

The system evaluates strategies based on:

1. **Need Percentages**: How much price needs to recover for profit
2. **Volume Distribution**: Q1 (front) vs Q4 (tail) balance
3. **Martingale Growth**: Smooth vs aggressive multiplication
4. **Exit Ease**: How easy to exit at each order level
5. **Pattern Quality**: Absence of plateaus, cliffs, zigzags

## 🛠️ Advanced Features

### Weight Tuning

```python
from martingale_lab.core.weight_tuner import WeightTuner

tuner = WeightTuner()

# Add your reference strategy
tuner.add_reference_strategy(your_volumes, sharpe_ratio=1.3, label="Reference")

# Get market-specific weights
weights = tuner.suggest_weights_for_market(
    market_volatility=0.7,  # High volatility
    market_trend=-0.3       # Slight downtrend
)

# Export for later use
tuner.export_weights("tuned_weights.json")
```

### Pattern Detection

```python
from martingale_lab.core.pattern_detection import analyze_micro_patterns
import numpy as np

volumes = np.array([...])  # Your strategy
analysis = analyze_micro_patterns(volumes, martingales)

print(f"Pattern Quality: {analysis['pattern_quality_score']}/100")
for recommendation in analysis['recommendations']:
    print(f"- {recommendation}")
```

## 🚨 Troubleshooting

### Common Issues

1. **All scores are inf**: Check m2_min/m2_max bounds, ensure first_volume >= 1.0
2. **Slow optimization**: Reduce batch_size, increase workers
3. **Database locked**: Ensure no other process is using the database

### Debug Mode

```bash
# Enable debug logging
export MLAB_LOG_LEVEL=DEBUG
python -m martingale_lab.cli.optimize ...
```

## 📊 Results Interpretation

| Metric | Good Range | Excellent | Your Strategy |
|--------|------------|-----------|---------------|
| Score | < 1500 | < 1200 | ~1400 |
| Q1 Share | 5-10% | 6-8% | 6.3% |
| Q4 Share | 50-60% | 55-60% | 53.8% |
| m[2] | 0.10-0.15 | 0.11-0.13 | 0.118 |
| Exit-ease | > 0.5 | > 0.6 | ~0.6 |
| Pattern Quality | > 80 | > 90 | 88.7 |

## 🎯 Next Steps

After finding optimal strategies:

1. **Backtest** with historical data
2. **Paper trade** for validation
3. **A/B test** against current strategy
4. **Monitor** performance metrics
5. **Adjust** for market conditions

## 📝 License

MIT License - See LICENSE file for details

## 🤝 Contributing

Contributions welcome! Please read CONTRIBUTING.md first.

## 📧 Support

For questions or issues, please open a GitHub issue or contact support.
