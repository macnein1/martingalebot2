# Martingale Lab - Advanced DCA Optimization System

A production-ready optimization system for Dollar Cost Averaging (DCA) strategies with martingale position sizing. Features configuration-based workflows, auto-tuning, and robust constraint enforcement.

## 🚀 Features

### Core Capabilities
- **Advanced Constraint System**: HC0-HC7 hard constraints with slope enforcement
- **2DP Normalization**: Automatic rounding to 2 decimal places with constraint preservation
- **Multi-Strategy Optimization**: Bayesian, adaptive, and exhaustive search modes
- **Auto-Tuning**: Evolutionary strategies for configuration optimization
- **Robust Storage**: Unified database with migrations and transaction management

### Configuration System
- **Structured Configs**: 7 logical configuration groups instead of 60+ parameters
- **Presets**: Ready-to-use configurations (exploration, production, strict, fast_exit)
- **Config Files**: JSON-based configuration management
- **Backward Compatible**: Works with legacy parameter formats

### Performance & Safety
- **Memory Management**: Bounded collections prevent memory leaks
- **Performance Cache**: Smart caching reduces computation overhead
- **Error Recovery**: Automatic rollback and cleanup on failures
- **Parallel Processing**: Thread/process pool execution for evaluations

## 📦 Installation

```bash
# Clone repository
git clone https://github.com/yourusername/martingalebot2.git
cd martingalebot2

# Install dependencies
pip install -r requirements.txt
```

## 🎯 Quick Start

### Using Configuration Presets

```bash
# Exploration mode - loose constraints for initial search
python -m martingale_lab.cli.optimize \
  --config-preset exploration \
  --batches 10 \
  --batch-size 100

# Production mode - balanced constraints
python -m martingale_lab.cli.optimize \
  --config-preset production \
  --batches 50 \
  --batch-size 500 \
  --db production.db

# Strict mode - tight constraints for fine-tuning
python -m martingale_lab.cli.optimize \
  --config-preset strict \
  --post-round-2dp \
  --max-time-sec 300
```

### Using Configuration Files

```bash
# Save current configuration
python -m martingale_lab.cli.optimize \
  --config-preset production \
  --slope-cap 0.20 \
  --save-config my_config.json \
  --print-config

# Load and run with saved configuration
python -m martingale_lab.cli.optimize \
  --config my_config.json \
  --batches 100 \
  --workers 4
```

### Auto-Tuning

```python
from martingale_lab.optimizer.auto_tuner import ConfigAutoTuner, TuningStrategy
from martingale_lab.storage.config_store import ConfigStore

# Initialize
config_store = ConfigStore("results.db")
tuner = ConfigAutoTuner(config_store)

# Get next configuration to try
strategy = TuningStrategy(name="exploration", exploration_rate=0.3)
next_config = tuner.suggest_next_config(strategy)

# Run optimization with suggested config
# ... optimization code ...

# Record performance
config_store.record_performance(
    config_hash, experiment_id, run_id,
    best_score=score, avg_score=avg, total_evaluations=n
)
```

## 🔧 Configuration Structure

The system uses structured configuration objects:

```python
EvaluationConfig
├── CoreConfig           # Base parameters (price, overlap, orders)
├── GenerationConfig     # Volume/indent generation settings
├── HardConstraintConfig # Constraint boundaries (m2, slopes, quartiles)
├── PenaltyWeightConfig  # Soft constraint penalties
├── ScoringConfig        # Objective function weights
├── AdaptiveConfig       # Adaptive optimization settings
└── NormalizationConfig  # Output normalization (2dp rounding)
```

## 📊 Key Parameters

### Constraint Parameters
- `slope_cap`: Maximum martingale slope change (default: 0.25)
- `m2_min/m2_max`: Second order martingale bounds (0.10-1.00)
- `q1_cap`: Maximum Q1 mass percentage (default: 22%)
- `tail_floor`: Minimum Q4 mass percentage (default: 32%)

### Optimization Parameters
- `batches`: Number of optimization batches
- `batch_size`: Candidates per batch
- `workers`: Parallel workers (0=auto)
- `max_time_sec`: Time limit in seconds

### Normalization Parameters
- `--post-round-2dp`: Enable 2 decimal place rounding
- `--post-round-strategy`: Rounding strategy (tail-first, largest-remainder)
- `--post-round-keep-v1-band`: Preserve v1 band constraint

## 🗄️ Database Schema

The system uses SQLite with versioned migrations:

```sql
-- Core tables
experiments     -- Optimization runs
results         -- Individual evaluations
checkpoints     -- Resume capability
diagnostics     -- Detailed metrics

-- Config tables
config_versions -- Configuration history
config_performance -- Performance tracking
config_lineage  -- Parent-child relationships
```

## 🧪 Testing

```bash
# Run comprehensive system test
python comprehensive_system_test.py

# Run smoke tests
python scripts/smoke_test.py

# Check constraint violations
python scripts/check_martingales.py
```

## 📈 Performance

- **Evaluation Speed**: 200-500 evals/sec (depends on constraints)
- **Memory Usage**: Bounded to 500 candidates max
- **Database Size**: ~1MB per 10,000 evaluations
- **Convergence**: Typically 50-100 batches for good results

## 🛠️ Advanced Usage

### Custom Constraints

```python
from martingale_lab.core.config_classes import EvaluationConfig

config = EvaluationConfig()
config.constraints.slope_cap = 0.15  # Tighter slopes
config.constraints.m2_min = 0.20     # Higher minimum m2
config.constraints.tail_floor = 35.0 # More tail weight
```

### Batch Processing

```bash
# Process multiple configurations
for preset in exploration production strict; do
  python -m martingale_lab.cli.optimize \
    --config-preset $preset \
    --batches 20 \
    --db results_${preset}.db
done
```

### Resume Interrupted Runs

```bash
# Resume from checkpoint
python -m martingale_lab.cli.optimize \
  --resume \
  --db existing.db
```

## 📝 SQL Queries

Common queries for analysis:

```sql
-- Best results by score
SELECT 
  score, 
  json_extract(params_json, '$.overlap_pct') as overlap,
  json_extract(params_json, '$.num_orders') as orders
FROM results 
WHERE experiment_id = ?
ORDER BY score ASC 
LIMIT 10;

-- Configuration performance
SELECT 
  config_hash,
  best_score,
  avg_score,
  use_count
FROM config_versions
WHERE best_score IS NOT NULL
ORDER BY best_score ASC;

-- Constraint violations
SELECT 
  COUNT(*) as violations,
  json_extract(diagnostics_json, '$.slope_violations') as slope_viols
FROM results
WHERE json_extract(diagnostics_json, '$.slope_violations') > 0;
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Run tests: `python comprehensive_system_test.py`
4. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details

## 🙏 Acknowledgments

Built with:
- NumPy for numerical computing
- Numba for JIT compilation
- SQLite for data storage
- Python 3.8+ for modern features

## 📧 Contact

For questions or support, please open an issue on GitHub.