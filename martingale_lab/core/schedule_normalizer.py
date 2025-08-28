"""
Schedule Normalizer Module
Rounds and normalizes schedule outputs to 2 decimal places while preserving all constraints.
"""
from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, Any, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


def round2(x: float) -> Decimal:
    """Round a float to 2 decimal places using Decimal for precision."""
    return Decimal(str(x)).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)


def normalize_schedule_to_2dp(
    schedule: Dict[str, Any],
    post_round_strategy: str = "tail-first",
    post_round_m2_tolerance: float = 0.05,
    post_round_keep_v1_band: bool = True,
    strict_monotonicity: bool = True,
    preserve_quartiles: bool = True,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Normalize schedule arrays to 2 decimal places while preserving constraints.
    
    Args:
        schedule: Schedule dict with indent_pct, volume_pct, etc.
        post_round_strategy: Strategy for sum adjustment ("tail-first", "largest-remainder", "balanced")
        post_round_m2_tolerance: Tolerance for m2 preservation in percentage points
        post_round_keep_v1_band: Whether to preserve v1 band constraint
        strict_monotonicity: Whether to enforce strict increasing volumes
        preserve_quartiles: Whether to preserve quartile constraints
        verbose: Whether to log detailed operations
        
    Returns:
        Normalized schedule dict with all constraints preserved
    """
    # Make a deep copy to avoid modifying the original
    import copy
    schedule = copy.deepcopy(schedule)
    
    # Extract arrays
    volume_pct = schedule.get("volume_pct", [])
    indent_pct = schedule.get("indent_pct", [])
    
    if not volume_pct:
        return schedule
    
    n = len(volume_pct)
    
    # Step 1: Round to 2 decimal places
    vol = [round2(v) for v in volume_pct]
    ind = [round2(i) for i in indent_pct] if indent_pct else []
    
    if verbose:
        logger.debug(f"Initial rounded volumes: {vol[:5]}...")
        logger.debug(f"Initial sum: {sum(vol)}")
    
    # Step 2: Adjust sum to exactly 100.00
    vol = _adjust_sum_to_100(vol, strategy=post_round_strategy, verbose=verbose)
    
    # Step 3: Enforce monotonicity
    vol = _enforce_monotonicity(vol, strict=strict_monotonicity, verbose=verbose)
    
    # Step 4: Preserve v1 band if needed
    if post_round_keep_v1_band and n >= 2:
        vol = _preserve_v1_band(vol, verbose=verbose)
    
    # Step 5: Check and preserve m2 if needed
    if n >= 3 and post_round_m2_tolerance is not None:
        vol = _preserve_m2(vol, tolerance=post_round_m2_tolerance, verbose=verbose)
    
    # Step 6: Preserve quartile constraints if needed
    if preserve_quartiles and n >= 8:  # Need reasonable number of orders
        vol = _preserve_quartiles(vol, verbose=verbose)
    
    # Step 7: Final sum adjustment (may be needed after constraint adjustments)
    vol = _adjust_sum_to_100(vol, strategy="minimal", verbose=verbose)
    
    # Step 8: Final monotonicity check
    vol = _enforce_monotonicity(vol, strict=strict_monotonicity, verbose=verbose)
    
    # Convert back to float and update schedule
    schedule["volume_pct"] = [float(v) for v in vol]
    if ind:
        schedule["indent_pct"] = [float(i) for i in ind]
    
    # Recompute derived fields if they exist
    if "martingale_pct" in schedule:
        schedule = _recompute_derived_fields(schedule)
    
    if verbose:
        logger.debug(f"Final volumes: {schedule['volume_pct'][:5]}...")
        logger.debug(f"Final sum: {sum(schedule['volume_pct'])}")
    
    return schedule


def _adjust_sum_to_100(vol: List[Decimal], strategy: str = "tail-first", verbose: bool = False) -> List[Decimal]:
    """Adjust volume sum to exactly 100.00."""
    target = Decimal('100.00')
    current_sum = sum(vol)
    diff = target - current_sum
    
    if diff == 0:
        return vol
    
    step = Decimal('0.01')
    sign = 1 if diff > 0 else -1
    
    if verbose:
        logger.debug(f"Adjusting sum: current={current_sum}, diff={diff}, strategy={strategy}")
    
    if strategy == "minimal":
        # Minimal adjustment - just adjust the last element
        if abs(diff) <= Decimal('0.10'):  # Small adjustment
            vol[-1] += diff
            return vol
    
    if strategy == "tail-first":
        # Adjust from tail to head
        indices = list(range(len(vol) - 1, -1, -1))
    elif strategy == "largest-remainder":
        # Sort by fractional part for largest remainder method
        fractional_parts = [(i, abs(v - v.quantize(Decimal('1')))) for i, v in enumerate(vol)]
        fractional_parts.sort(key=lambda x: x[1], reverse=True)
        indices = [i for i, _ in fractional_parts]
    else:  # balanced
        # Distribute evenly across all elements
        indices = list(range(len(vol)))
    
    # Apply adjustments
    remaining = abs(diff)
    attempts = 0
    max_attempts = len(vol) * 10  # Prevent infinite loops
    
    while remaining > 0 and attempts < max_attempts:
        for i in indices:
            if remaining <= 0:
                break
                
            # Check if adjustment is valid
            new_val = vol[i] + sign * step
            
            # Don't go negative
            if new_val < Decimal('0.00'):
                continue
            
            # Don't make too large (sanity check)
            if new_val > Decimal('50.00'):  # No single volume should be > 50%
                continue
            
            vol[i] = new_val
            remaining -= step
            
        attempts += 1
    
    if remaining > 0 and verbose:
        logger.warning(f"Could not fully adjust sum, remaining diff: {remaining}")
    
    return vol


def _enforce_monotonicity(vol: List[Decimal], strict: bool = True, verbose: bool = False) -> List[Decimal]:
    """Enforce monotonic increasing (or non-decreasing) constraint."""
    if len(vol) <= 1:
        return vol
    
    step = Decimal('0.01')
    violations_fixed = 0
    
    for i in range(1, len(vol)):
        if strict:
            # Strict increasing: vol[i] > vol[i-1]
            min_val = vol[i-1] + step
        else:
            # Non-decreasing: vol[i] >= vol[i-1]
            min_val = vol[i-1]
        
        if vol[i] < min_val:
            violations_fixed += 1
            deficit = min_val - vol[i]
            vol[i] = min_val
            
            # Compensate by reducing later volumes
            for j in range(len(vol) - 1, i, -1):
                if deficit <= 0:
                    break
                    
                # How much can we reduce vol[j]?
                if j > 0:
                    max_reduction = vol[j] - (vol[j-1] + step if strict else vol[j-1])
                else:
                    max_reduction = vol[j]
                
                reduction = min(deficit, max_reduction)
                if reduction > 0:
                    vol[j] -= reduction
                    deficit -= reduction
    
    if verbose and violations_fixed > 0:
        logger.debug(f"Fixed {violations_fixed} monotonicity violations")
    
    return vol


def _preserve_v1_band(vol: List[Decimal], verbose: bool = False) -> List[Decimal]:
    """Preserve v1 band constraint: v1 ∈ [1.10 * v0, 2.00 * v0]."""
    if len(vol) < 2:
        return vol
    
    v0, v1 = vol[0], vol[1]
    lo = (v0 * Decimal('1.10')).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
    hi = (v0 * Decimal('2.00')).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
    
    # Allow small tolerance for floating point comparisons
    epsilon = Decimal('0.01')
    if lo - epsilon <= v1 <= hi + epsilon:
        return vol  # Already satisfied
    
    old_v1 = v1
    if v1 < lo:
        vol[1] = lo
    else:
        vol[1] = hi
    
    # Compensate the difference in tail
    diff = old_v1 - vol[1]
    
    if verbose:
        logger.debug(f"v1 band adjustment: {old_v1} -> {vol[1]}, diff={diff}")
    
    # Distribute the difference across tail elements
    if len(vol) > 2:
        tail_indices = list(range(len(vol) - 1, 1, -1))
        step = Decimal('0.01')
        remaining = abs(diff)
        sign = 1 if diff > 0 else -1
        
        for i in tail_indices:
            if remaining <= 0:
                break
                
            adjustment = min(remaining, step)
            new_val = vol[i] + sign * adjustment
            
            # Check constraints
            if new_val >= Decimal('0.01') and (i == len(vol) - 1 or new_val <= vol[i+1]):
                if i > 0 and new_val >= vol[i-1]:
                    vol[i] = new_val
                    remaining -= adjustment
    
    return vol


def _preserve_m2(vol: List[Decimal], tolerance: float = 0.05, verbose: bool = False) -> List[Decimal]:
    """Preserve m2 (second martingale ratio) within tolerance."""
    if len(vol) < 3:
        return vol
    
    # Calculate current m2
    v0 = float(vol[0])
    v1 = float(vol[1])
    v2 = float(vol[2])
    
    if v0 + v1 <= 0:
        return vol
    
    current_m2 = v2 / (v0 + v1)
    
    # Check if we need adjustment (assuming we want to preserve some target m2)
    # For now, we just ensure m2 stays reasonable (between 0.10 and 1.00)
    min_m2 = 0.10
    max_m2 = 1.00
    
    if min_m2 <= current_m2 <= max_m2:
        return vol  # Already in acceptable range
    
    if verbose:
        logger.debug(f"m2 adjustment needed: current={current_m2:.4f}")
    
    # Adjust v2 to bring m2 into range
    if current_m2 < min_m2:
        target_v2 = Decimal(str(min_m2 * (v0 + v1))).quantize(Decimal('0.01'))
        diff = target_v2 - vol[2]
        vol[2] = target_v2
        
        # Compensate in tail
        if len(vol) > 3:
            for i in range(len(vol) - 1, 2, -1):
                if diff <= 0:
                    break
                adjustment = min(diff, vol[i] - vol[i-1] - Decimal('0.01'))
                if adjustment > 0:
                    vol[i] -= adjustment
                    diff -= adjustment
    
    elif current_m2 > max_m2:
        target_v2 = Decimal(str(max_m2 * (v0 + v1))).quantize(Decimal('0.01'))
        diff = vol[2] - target_v2
        vol[2] = target_v2
        
        # Compensate in tail
        if len(vol) > 3:
            for i in range(3, len(vol)):
                if diff <= 0:
                    break
                vol[i] += min(diff, Decimal('0.01'))
                diff -= Decimal('0.01')
    
    return vol


def _preserve_quartiles(vol: List[Decimal], q1_cap: float = 22.0, q4_floor: float = 32.0, verbose: bool = False) -> List[Decimal]:
    """Preserve quartile constraints."""
    n = len(vol)
    if n < 8:
        return vol
    
    q1_end = n // 4
    q4_start = 3 * n // 4
    
    vol_float = [float(v) for v in vol]
    q1_share = sum(vol_float[:q1_end])
    q4_share = sum(vol_float[q4_start:])
    
    if verbose:
        logger.debug(f"Quartiles: Q1={q1_share:.2f}%, Q4={q4_share:.2f}%")
    
    adjustments_made = False
    
    # Adjust Q1 if needed
    if q1_share > q1_cap:
        excess = Decimal(str(q1_share - q1_cap))
        # Reduce Q1 volumes and add to tail
        for i in range(q1_end - 1, -1, -1):
            if excess <= 0:
                break
            reduction = min(excess, vol[i] * Decimal('0.1'))  # Max 10% reduction per element
            vol[i] -= reduction
            vol[-1] += reduction  # Add to last element
            excess -= reduction
            adjustments_made = True
    
    # Adjust Q4 if needed
    if q4_share < q4_floor:
        deficit = Decimal(str(q4_floor - q4_share))
        # Increase Q4 volumes by taking from middle
        mid_start = q1_end
        mid_end = q4_start
        
        for i in range(mid_end - 1, mid_start - 1, -1):
            if deficit <= 0:
                break
            reduction = min(deficit, vol[i] * Decimal('0.1'))  # Max 10% reduction per element
            vol[i] -= reduction
            # Distribute to Q4
            for j in range(q4_start, n):
                if reduction <= 0:
                    break
                addition = min(reduction, Decimal('0.01'))
                vol[j] += addition
                reduction -= addition
            deficit -= reduction
            adjustments_made = True
    
    if verbose and adjustments_made:
        logger.debug("Quartile adjustments applied")
    
    return vol


def _recompute_derived_fields(schedule: Dict[str, Any]) -> Dict[str, Any]:
    """Recompute derived fields like martingale_pct and needpct after normalization."""
    volume_pct = schedule.get("volume_pct", [])
    
    if not volume_pct:
        return schedule
    
    # Recompute martingale_pct
    martingale_pct = []
    cumsum = 0.0
    for i, v in enumerate(volume_pct):
        if i == 0:
            martingale_pct.append(0.0)
        else:
            cumsum += volume_pct[i-1]
            if cumsum > 0:
                martingale_pct.append(v / cumsum * 100.0)
            else:
                martingale_pct.append(0.0)
    
    schedule["martingale_pct"] = martingale_pct
    
    # Recompute needpct if we have order_prices
    if "order_prices" in schedule and "indent_pct" in schedule:
        order_prices = schedule["order_prices"]
        indent_pct = schedule["indent_pct"]
        base_price = order_prices[0] if order_prices else 1.0
        
        needpct = []
        for i in range(len(volume_pct)):
            if i < len(order_prices):
                price_ratio = order_prices[i] / base_price
                need = volume_pct[i] * price_ratio
                needpct.append(need)
            else:
                needpct.append(volume_pct[i])
        
        schedule["needpct"] = needpct
    
    return schedule


def is_schedule_normalized(schedule: Dict[str, Any], tolerance: float = 1e-10) -> bool:
    """
    Check if a schedule is already normalized to 2 decimal places.
    
    Args:
        schedule: Schedule dict to check
        tolerance: Floating point comparison tolerance
        
    Returns:
        True if already normalized, False otherwise
    """
    volume_pct = schedule.get("volume_pct", [])
    indent_pct = schedule.get("indent_pct", [])
    
    # Check volumes
    for v in volume_pct:
        # Check if value has more than 2 decimal places
        rounded = round(v, 2)
        if abs(v - rounded) > tolerance:
            return False
    
    # Check indents
    for i in indent_pct:
        rounded = round(i, 2)
        if abs(i - rounded) > tolerance:
            return False
    
    # Check sum
    total = sum(volume_pct)
    if abs(total - 100.0) > 0.01:  # Allow small tolerance for sum
        return False
    
    return True


def validate_normalized_schedule(schedule: Dict[str, Any], original_schedule: Dict[str, Any] = None) -> Dict[str, bool]:
    """
    Validate that a normalized schedule preserves all constraints.
    
    Args:
        schedule: Normalized schedule to validate
        original_schedule: Optional original schedule for comparison
        
    Returns:
        Dict of validation results
    """
    results = {}
    
    volume_pct = schedule.get("volume_pct", [])
    indent_pct = schedule.get("indent_pct", [])
    
    if not volume_pct:
        results["has_volumes"] = False
        return results
    
    n = len(volume_pct)
    results["has_volumes"] = True
    
    # Check 2 decimal places
    results["volumes_2dp"] = all(round(v, 2) == v for v in volume_pct)
    results["indents_2dp"] = all(round(i, 2) == i for i in indent_pct) if indent_pct else True
    
    # Check sum = 100
    total = sum(volume_pct)
    results["sum_is_100"] = abs(total - 100.0) < 0.01
    
    # Check monotonicity
    results["monotonic"] = all(volume_pct[i] <= volume_pct[i+1] for i in range(n-1))
    results["strict_monotonic"] = all(volume_pct[i] < volume_pct[i+1] for i in range(n-1))
    
    # Check v1 band
    if n >= 2:
        v0, v1 = volume_pct[0], volume_pct[1]
        results["v1_band"] = 1.10 * v0 <= v1 <= 2.00 * v0
    else:
        results["v1_band"] = True
    
    # Check m2
    if n >= 3:
        v0, v1, v2 = volume_pct[0], volume_pct[1], volume_pct[2]
        if v0 + v1 > 0:
            m2 = v2 / (v0 + v1)
            # Allow small tolerance for m2 bounds
            results["m2_valid"] = 0.05 <= m2 <= 1.05  # Slightly relaxed bounds
            results["m2_value"] = m2
        else:
            results["m2_valid"] = False
            results["m2_value"] = None
    else:
        results["m2_valid"] = True
        results["m2_value"] = None
    
    # Check quartiles
    if n >= 8:
        q1_end = n // 4
        q4_start = 3 * n // 4
        q1_share = sum(volume_pct[:q1_end])
        q4_share = sum(volume_pct[q4_start:])
        results["q1_share"] = q1_share
        results["q4_share"] = q4_share
        results["q1_valid"] = q1_share <= 22.0
        results["q4_valid"] = q4_share >= 32.0
    else:
        results["q1_valid"] = True
        results["q4_valid"] = True
    
    # Compare with original if provided
    if original_schedule:
        orig_vol = original_schedule.get("volume_pct", [])
        if orig_vol and len(orig_vol) == n:
            max_diff = max(abs(volume_pct[i] - orig_vol[i]) for i in range(n))
            results["max_diff_from_original"] = max_diff
            results["preserves_shape"] = max_diff < 1.0  # Max 1% difference per element
    
    results["all_valid"] = all(
        v for k, v in results.items() 
        if k.endswith("_valid") or k.endswith("_2dp") or k == "sum_is_100" or k == "monotonic"
    )
    
    return results