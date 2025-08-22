# 🎯 DCA/Martingale "İşlemden En Hızlı Çıkış" Optimization System

## Overview

This system implements a comprehensive DCA (Dollar Cost Averaging) / Martingale optimization engine focused on **"İşlemden En Hızlı Çıkış"** - finding the fastest exit strategies from positions. The system evaluates different martingale percentages and volume distributions to minimize the percentage needed to return to entry price after each order.

## ✨ Key Features

### 🎯 Core Evaluation Contract
- **NeedPct Calculation**: Calculates percentage needed to return to entry price after each order
- **Comprehensive Scoring**: Combines max_need, var_need, tail concentration with penalties
- **Wave Pattern Support**: Optional alternating strong-weak martingale patterns
- **Sanity Checks**: Automatic detection of collapsed indents, tail overflow, and consistency issues

### 📊 Complete Results Display
- **Bullets Format**: Exact specification format for each order
- **NeedPct Sparklines**: ASCII visualization of exit requirements
- **Sanity Badges**: Visual indicators for constraint violations
- **Interactive Filtering**: Filter by score, overlap, orders, wave patterns

### 🚀 Advanced Optimization
- **Parallel Processing**: Multi-threaded evaluation for performance
- **Early Pruning**: Eliminates poor candidates during search
- **Adaptive Search**: Maintains top-K candidates across batches
- **Early Stopping**: Stops when no improvement is found

### 💾 Persistent Storage
- **SQLite Database**: Complete experiment and result storage
- **JSON Payloads**: Full schedule, sanity, diagnostics, and penalties
- **Experiment Management**: Track multiple optimization runs
- **Result Retrieval**: Advanced filtering and sorting capabilities

## 🏗️ System Architecture

```
martingale_lab/
├── optimizer/
│   └── dca_evaluation_contract.py    # Core evaluation function
├── orchestrator/
│   └── dca_orchestrator.py          # Optimization engine
├── storage/
│   └── experiments_store.py         # Database management
ui/
└── components/
    └── results_section.py           # Results visualization
```

## 🎯 Evaluation Contract

### Input Parameters
```python
evaluation_function(
    base_price: float = 1.0,
    overlap_pct: float = 20.0,
    num_orders: int = 5,
    seed: Optional[int] = None,
    wave_pattern: bool = False,
    alpha: float = 0.5,           # max_need weight
    beta: float = 0.3,            # var_need weight  
    gamma: float = 0.2,           # tail weight
    lambda_penalty: float = 0.1,  # penalty weight
    **kwargs
)
```

### Output Contract
```python
{
    "score": float,              # Final J score (lower is better)
    "max_need": float,           # Maximum NeedPct across all orders
    "var_need": float,           # Variance of NeedPct values
    "tail": float,               # Tail concentration (last 20% orders)
    
    "schedule": {
        "indent_pct": [p1, p2, ...],      # Cumulative indent percentages
        "volume_pct": [v1, v2, ...],      # Volume distribution (sum=100)
        "martingale_pct": [0, m2, ...],   # Martingale percentages (m1=0)
        "needpct": [n1, n2, ...],         # Exit-to-entry percentages
        "order_prices": [p0, p1, ...],    # Order prices (p0=base)
        "price_step_pct": [Δp1, Δp2, ...]# Price step percentages
    },
    
    "sanity": {
        "max_need_mismatch": bool,        # max_need != max(needpct)
        "collapse_indents": bool,         # Non-monotonic/small steps
        "tail_overflow": bool             # Last order > tail_cap
    },
    
    "diagnostics": {
        "wci": float,                     # Weight Center Index (0=early, 1=late)
        "sign_flips": int,                # NeedPct trend changes
        "gini": float,                    # Volume Gini coefficient
        "entropy": float                  # Volume entropy (diversity)
    },
    
    "penalties": {
        "P_gini": float,                  # Volume concentration penalty
        "P_entropy": float,               # Low diversity penalty
        "P_monotone": float,              # Non-monotonic penalty
        "P_smooth": float,                # Price step smoothness penalty
        "P_tailcap": float,               # Tail cap violation penalty
        "P_need_mismatch": float,         # Consistency penalty
        "P_wave": float                   # Wave pattern penalty/reward
    }
}
```

## 🎯 Scoring Function

The final score J combines multiple objectives:

```
J = α·max_need + β·var_need + γ·tail + λ·Σ(penalties)
```

**Default weights:**
- α = 0.5 (max_need weight)
- β = 0.3 (var_need weight)
- γ = 0.2 (tail weight)
- λ = 0.1 (penalty weight)

## 🌊 Wave Pattern Feature

When enabled, the system encourages alternating strong-weak martingale patterns:

- **Strong threshold**: ≥50% martingale (configurable)
- **Weak threshold**: ≤10% martingale (configurable)
- **Rewards**: Alternating strong→weak or weak→strong patterns
- **Penalties**: Consecutive strong or consecutive very weak patterns

## 📋 Bullets Format

Results are displayed in the exact specified format:

```
1. Emir: Indent %0.00  Volume %25.50  (no martingale, first order) — NeedPct %0.00
2. Emir: Indent %5.25  Volume %35.20  (Martingale %38.04) — NeedPct %2.15
3. Emir: Indent %12.80 Volume %39.30  (Martingale %11.65) — NeedPct %7.89
```

## 🚀 Quick Start

### 1. Basic Evaluation
```python
from martingale_lab.optimizer.dca_evaluation_contract import evaluation_function

result = evaluation_function(
    base_price=1.0,
    overlap_pct=20.0,
    num_orders=5,
    seed=42
)

print(f"Score: {result['score']:.6f}")
print(f"Max Need: {result['max_need']:.2f}%")
```

### 2. Run Optimization
```python
from martingale_lab.orchestrator.dca_orchestrator import create_dca_orchestrator

orchestrator = create_dca_orchestrator(
    overlap_range=(10.0, 30.0),
    orders_range=(5, 15),
    wave_pattern=True,
    n_candidates=1000,
    max_batches=50
)

results = orchestrator.run_optimization(notes="Test optimization")
print(f"Found {len(results['best_candidates'])} candidates")
```

### 3. Launch UI
```bash
streamlit run main.py
```

## 🔧 Configuration Options

### DCAConfig Parameters
- **Search Space**: `overlap_min/max`, `orders_min/max`
- **Scoring Weights**: `alpha`, `beta`, `gamma`, `lambda_penalty`
- **Wave Pattern**: `wave_pattern`, `wave_strong_threshold`, `wave_weak_threshold`
- **Constraints**: `tail_cap`, `min_indent_step`, `softmax_temp`
- **Optimization**: `n_candidates_per_batch`, `max_batches`, `n_workers`
- **Early Stopping**: `early_stop_threshold`, `early_stop_patience`

### Example Configuration
```python
config = DCAConfig(
    overlap_min=10.0,
    overlap_max=30.0,
    orders_min=5,
    orders_max=15,
    alpha=0.5,
    beta=0.3,
    gamma=0.2,
    lambda_penalty=0.1,
    wave_pattern=True,
    tail_cap=0.40,
    n_candidates_per_batch=1000,
    max_batches=100,
    n_workers=4
)
```

## 🎛️ UI Features

### Configuration Page
- **Search Space**: Set overlap and orders ranges
- **Scoring Weights**: Adjust α, β, γ, λ parameters
- **Wave Pattern**: Enable/disable with thresholds
- **Constraints**: Set tail cap, indent steps, softmax temperature
- **Optimization**: Configure batch size, workers, early stopping

### Optimization Page
- **Real-time Progress**: Live batch progress and metrics
- **Performance Monitoring**: Evaluations per second, best score tracking
- **Control Panel**: Start, stop, clear results
- **Statistics**: Total time, evaluations, early stopping status

### Results Page
- **Summary Card**: Best score, max need, variance, tail metrics
- **Advanced Filtering**: Score limits, overlap/orders ranges, wave pattern only
- **Top-N Table**: Rank, score, metrics, NeedPct sparklines, sanity badges
- **Detailed View**: Expandable rows with full schedule, charts, diagnostics
- **Export Options**: CSV schedule download, full JSON export

## 📊 Results Analysis

### Key Metrics
- **Score (J)**: Lower is better, combines all objectives
- **Max Need**: Maximum percentage needed to return to entry
- **Var Need**: Variance in NeedPct values (lower = more stable)
- **Tail**: Concentration in last orders (lower = more distributed)
- **WCI**: Weight Center Index (0=early load, 1=late load)
- **Sign Flips**: Number of NeedPct trend changes

### Sanity Checks
- **Max Need Mismatch**: Consistency between calculated and reported max_need
- **Collapse Indents**: Non-monotonic or too-small indent steps
- **Tail Overflow**: Last order exceeds volume cap

### Exit Speed Score
```
Exit Speed = 1 / (1 + mean(needpct/100))
```
Higher values indicate faster exit potential.

## 🧪 Testing

Run the test suite:
```bash
python3 simple_test_dca.py
```

Expected output:
```
🎯 Simple Test Results: 4 passed, 0 failed
🎉 All simple tests passed!
```

## 📁 Database Schema

### Experiments Table
- Stores experiment metadata and configuration
- Tracks best score, total evaluations, elapsed time
- Includes wave pattern and constraint settings

### Results Table
- Complete JSON payloads for each candidate
- Indexed by stable_id for deduplication
- Includes all schedule, sanity, diagnostics, penalties data

## 🔍 Advanced Features

### Early Pruning
- Eliminates candidates with score > best_score * 1.5
- Filters out infinite scores and severe sanity violations
- Maintains performance while ensuring quality

### Backpressure Management
- Keeps top-K candidates between batches (default: 10,000)
- Prevents memory overflow during long optimizations
- Balances exploration with exploitation

### Adaptive Search
- Monitors improvement over batches
- Implements early stopping when no progress is made
- Adjusts search strategy based on sanity violation patterns

## 🎯 Use Cases

### 1. Conservative Strategy
- High β (variance weight) for stable exits
- Low tail_cap to prevent concentration
- Disabled wave patterns for simplicity

### 2. Aggressive Strategy
- High α (max_need weight) for fastest possible exits
- Higher tail_cap allowing concentration
- Enabled wave patterns for dynamic distribution

### 3. Research Mode
- Balanced weights (α=0.5, β=0.3, γ=0.2)
- Wave pattern experimentation
- Comprehensive sanity checking

## 🚀 Performance

- **Evaluation Speed**: ~1000-5000 evaluations/second (depends on hardware)
- **Parallel Processing**: Scales with CPU cores
- **Memory Efficient**: Streaming results to database
- **Early Termination**: Stops unproductive searches automatically

## 🔧 Requirements

- Python 3.7+
- NumPy
- SQLite3
- Streamlit (for UI)
- Plotly (for charts)

## 📈 Future Enhancements

- [ ] GPU acceleration for batch evaluation
- [ ] Multi-objective Pareto frontier analysis
- [ ] Advanced visualization with 3D plots
- [ ] Real-time market data integration
- [ ] Strategy backtesting framework
- [ ] Export to trading platforms

## 🎯 Summary

This DCA/Martingale optimization system provides a complete solution for finding optimal exit strategies. With its comprehensive evaluation contract, advanced optimization engine, and intuitive UI, it enables both researchers and practitioners to explore and optimize DCA strategies effectively.

The system's focus on "İşlemden En Hızlı Çıkış" ensures that all optimizations prioritize quick exits while maintaining stability and avoiding excessive risk concentration.