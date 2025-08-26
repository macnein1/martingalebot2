# HC1-HC7 Implementation Summary

## Overview
This implementation adds a robust Hard Constraint (HC) pipeline to the martingale optimization system, ensuring wave-shaped profiles with strict monotonicity and proper mass distribution.

## Key Components Added

### 1. Core Helper Functions (`martingale_lab/core/repair.py`)
- `tail_only_rescale_keep_first_two(v)`: Rescales v[2:] while keeping v[0], v[1] fixed
- `compute_m_from_v(v)`: Computes martingale ratios m[i] = v[i]/v[i-1] - 1
- `rechain_v_from_m(v0, v1, m)`: Reconstructs volumes from martingale ratios
- `longest_plateau_run(m)`: Detects plateau regions for breaking

### 2. HC Pipeline (`martingale_lab/core/constraints.py`)
Updated `enforce_schedule_shape_fixed()` with HC1-HC7:

- **HC1**: v1 band constraint [1.10*v0, 2.0*v0]
- **HC2**: Strict monotonicity v[i] >= v[i-1] + eps_inc
- **HC3**: Martingale bands with decaying ceiling
- **HC4**: Slope limits |Δm| <= 0.25 for smoothness
- **HC5**: Mass control (Q1 cap <= 22%, Q4 floor >= 32%)
- **HC6**: Plateau breaker (max run <= 3)
- **HC7**: Additional wave enforcement for diversity

### 3. New Penalty Functions (`martingale_lab/core/penalties.py`)
- **SP1** `penalty_second_band`: v1 band violation penalty
- **SP2** `penalty_plateau`: Plateau run excess penalty
- **SP3** `penalty_varm`: Low variance penalty
- **SP4** `penalty_wave_shape`: L2 distance to ideal wave
- **SP5** `penalty_front_share`: Q1 cap violation
- **SP6** `penalty_tailweak`: Q4 floor violation
- **SP7** `penalty_slope`: Slope violation penalty

### 4. Evaluation Engine Updates (`martingale_lab/optimizer/evaluation_engine.py`)
- Added new HC parameters to function signature
- Updated enforce_schedule_shape_fixed call with new parameters
- Enhanced logging with new metrics (v0, v1, m2, q1_share, q4_share, etc.)
- Added SP penalty weights to scoring

## Key Parameters

### Hard Constraint Parameters
```python
second_upper_c2: float = 2.0    # v1 upper bound multiplier
m2_min: float = 0.10            # m[2] minimum
m2_max: float = 0.80            # m[2] maximum  
m_head: float = 0.40            # Decaying ceiling head
m_tail: float = 0.20            # Decaying ceiling tail
tau_scale: float = 1/3          # Decay rate scale
slope_cap: float = 0.25         # Maximum |Δm|
q1_cap: float = 22.0            # Q1 volume cap %
tail_floor: float = 32.0        # Q4 volume floor %
eps_inc: float = 1e-6           # Minimum increment
```

### Penalty Weight Presets
- **explore**: Lower penalties for exploration
- **robust**: Balanced penalties (recommended)
- **tight**: Strict penalties for production

## Acceptance Criteria Met

### Invariants ✓
- v0 == 0.01 (fixed first volume)
- indent0 == 0.00 (fixed first indent)
- ∀i≥1: v[i] > v[i-1] (strict monotonicity)
- |sum(v) - 100| < 1e-6 (normalization)

### Early Steps ✓
- 1.10*v0 ≤ v1 ≤ 2.0*v0 (v1 band)
- m[2] ∈ [0.10, 0.80] (m2 bounds)
- first3_sum ≤ 5.0% (front concentration)

### Band/Smoothness/Mass ✓
- i≥3: m[i] ≤ m_max(i) (decaying ceiling)
- |m[i]-m[i-1]| ≤ 0.25 (slope limit)
- plateau_max_run ≤ 3 (plateau breaking)
- q1_share ≤ 22% (front cap)
- q4_share ≥ 32% (tail floor)
- std(m[2:]) ≥ 0.18 (diversity)
- turn_count ≥ 2 (wave shape)

### Determinism ✓
- Same seed → same schedule/score

## SQL Validation Queries
See `sql_validations.sh` for comprehensive validation queries.

## Testing
Run `python3 test_hc_implementation.py` to verify all constraints.

## Notes
- The implementation maintains backward compatibility with existing CLI/DB interfaces
- All constraints are deterministic (no RNG in repair/penalties)
- v0 and v1 remain fixed throughout the pipeline
- Tail-only rescaling preserves front constraints
