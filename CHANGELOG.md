# Changelog

All notable changes to Martingale Lab will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Safe Parameter Forwarding**: New `parameter_forwarding.py` module with signature-aware function calling
  - `filter_kwargs_for_function()`: Filters kwargs to only parameters accepted by target function
  - `safe_call()`: Safely calls functions with only accepted parameters
  - Prevents `TypeError` from incompatible parameter sets between versions

- **Slope Cap Enforcement**: Added `slope_cap` parameter to `enforce_schedule_shape_fixed()`
  - Limits martingale slope changes to prevent sudden jumps
  - Applies to both initial generation and iterative refinement
  - Default value: 0.25 (25% max change between consecutive martingales)

- **Comprehensive Smoke Tests**:
  - `scripts/smoke_test.py`: Full validation suite with SQL queries
  - `scripts/ci_smoke_test.sh`: CI-ready smoke test script
  - `scripts/check_martingales.py`: Diagnostic tool for martingale analysis
  - SQL validations for:
    - v1 band constraints
    - Martingale slope violations
    - Volume monotonicity
    - Total sum = 100
    - Q1/Q4 share percentages

### Changed
- **Parameter Flow Synchronization**:
  - CLI parameters now fully synchronized with constraints signature
  - `evaluation_engine.py` uses safe parameter forwarding to call `enforce_schedule_shape_fixed()`
  - All HC0-HC7 parameters properly forwarded through the chain

- **Improved v1 Band Enforcement**:
  - v0 (first volume) now properly preserved during normalization
  - v1 band constraints maintained through iterative refinement
  - Slope cap applied to m2 (v1/v0 - 1) to prevent violations

- **Better Normalization Logic**:
  - Preserves v0 = first_volume_target throughout normalization
  - Tail normalization adjusts volumes[1:] while keeping v0 fixed
  - Multiple iteration loops ensure all constraints are satisfied

### Fixed
- **v1 Band Violations**: Fixed issue where v1 exceeded allowed band after normalization
- **Martingale Slope Violations**: Reduced slope violations through proper slope cap enforcement
- **SQL Query Errors**: Fixed column naming issues in smoke test SQL queries
- **Python Path Issues**: Fixed subprocess calls to use `sys.executable` instead of hardcoded 'python'

### Technical Details

#### Parameter Forwarding Implementation
The new safe forwarding system ensures compatibility between different component versions:

```python
# In evaluation_engine.py
constraint_params = {
    "indent_pct": indent_pct_initial,
    "volume_pct": volume_pct_initial,
    "base_price": base_price,
    "slope_cap": slope_cap,
    # ... other parameters
}

# Filter to only accepted parameters
safe_params = filter_kwargs_for_function(enforce_schedule_shape_fixed, constraint_params)

# Call with safe parameters
result = enforce_schedule_shape_fixed(**safe_params)
```

#### Constraint Enforcement Flow
1. **Initial Generation**: Apply bands and slope caps
2. **Normalization**: Scale to sum=100 while preserving v0
3. **Iterative Refinement**: Re-apply constraints after normalization
4. **Final Validation**: Ensure all constraints are met

### Migration Notes
- No breaking changes for existing code
- New parameters are optional with sensible defaults
- Safe forwarding prevents errors from unknown parameters

## [Previous Versions]
(Previous changelog entries would go here)
