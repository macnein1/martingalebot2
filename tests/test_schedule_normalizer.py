"""
Unit tests for schedule normalization functionality.
"""
import pytest
import numpy as np
from decimal import Decimal
from martingale_lab.core.schedule_normalizer import (
    round2,
    normalize_schedule_to_2dp,
    is_schedule_normalized,
    validate_normalized_schedule,
    _adjust_sum_to_100,
    _enforce_monotonicity,
    _preserve_v1_band,
    _preserve_m2,
    _preserve_quartiles
)


class TestRound2:
    """Test the round2 function."""
    
    def test_round2_basic(self):
        """Test basic rounding to 2 decimal places."""
        assert round2(1.234) == Decimal('1.23')
        assert round2(1.235) == Decimal('1.24')  # ROUND_HALF_UP
        assert round2(1.999) == Decimal('2.00')
        assert round2(0.001) == Decimal('0.00')
        assert round2(99.999) == Decimal('100.00')
    
    def test_round2_negative(self):
        """Test rounding negative numbers."""
        assert round2(-1.234) == Decimal('-1.23')
        assert round2(-1.235) == Decimal('-1.24')
    
    def test_round2_already_rounded(self):
        """Test that already rounded numbers stay the same."""
        assert round2(1.00) == Decimal('1.00')
        assert round2(1.50) == Decimal('1.50')
        assert round2(99.99) == Decimal('99.99')


class TestAdjustSumTo100:
    """Test the _adjust_sum_to_100 function."""
    
    def test_already_100(self):
        """Test when sum is already 100."""
        vol = [Decimal('20.00'), Decimal('30.00'), Decimal('50.00')]
        result = _adjust_sum_to_100(vol)
        assert sum(result) == Decimal('100.00')
        assert result == vol  # Should be unchanged
    
    def test_sum_less_than_100(self):
        """Test when sum is less than 100."""
        vol = [Decimal('20.00'), Decimal('30.00'), Decimal('40.00')]
        result = _adjust_sum_to_100(vol, strategy="tail-first")
        assert sum(result) == Decimal('100.00')
        assert result[0] == Decimal('20.00')  # First unchanged
        assert result[1] == Decimal('30.00')  # Second unchanged
        assert result[2] == Decimal('50.00')  # Last adjusted
    
    def test_sum_more_than_100(self):
        """Test when sum is more than 100."""
        vol = [Decimal('30.00'), Decimal('40.00'), Decimal('40.00')]
        result = _adjust_sum_to_100(vol, strategy="tail-first")
        assert sum(result) == Decimal('100.00')
        assert result[2] < Decimal('40.00')  # Last should be reduced
    
    def test_minimal_strategy(self):
        """Test minimal adjustment strategy."""
        vol = [Decimal('20.00'), Decimal('30.00'), Decimal('49.95')]
        result = _adjust_sum_to_100(vol, strategy="minimal")
        assert sum(result) == Decimal('100.00')
        assert result[0] == Decimal('20.00')
        assert result[1] == Decimal('30.00')
        assert result[2] == Decimal('50.00')  # Only last adjusted


class TestEnforceMonotonicity:
    """Test the _enforce_monotonicity function."""
    
    def test_already_monotonic(self):
        """Test when already strictly monotonic."""
        vol = [Decimal('10.00'), Decimal('20.00'), Decimal('30.00'), Decimal('40.00')]
        result = _enforce_monotonicity(vol, strict=True)
        assert result == vol
    
    def test_fix_violations(self):
        """Test fixing monotonicity violations."""
        vol = [Decimal('20.00'), Decimal('15.00'), Decimal('30.00'), Decimal('40.00')]
        result = _enforce_monotonicity(vol, strict=True)
        assert all(result[i] < result[i+1] for i in range(len(result)-1))
        assert result[0] == Decimal('20.00')
        assert result[1] >= Decimal('20.01')  # Should be increased
    
    def test_non_decreasing(self):
        """Test non-decreasing (allows equal values)."""
        vol = [Decimal('10.00'), Decimal('10.00'), Decimal('20.00'), Decimal('30.00')]
        result = _enforce_monotonicity(vol, strict=False)
        assert all(result[i] <= result[i+1] for i in range(len(result)-1))


class TestPreserveV1Band:
    """Test the _preserve_v1_band function."""
    
    def test_v1_within_band(self):
        """Test when v1 is already within band."""
        vol = [Decimal('10.00'), Decimal('15.00'), Decimal('25.00'), Decimal('50.00')]
        result = _preserve_v1_band(vol)
        # v1 should be in [1.10 * v0, 2.00 * v0] = [11.00, 20.00]
        assert result == vol  # 15.00 is within band
    
    def test_v1_too_low(self):
        """Test when v1 is below minimum."""
        vol = [Decimal('10.00'), Decimal('10.50'), Decimal('25.00'), Decimal('54.50')]
        result = _preserve_v1_band(vol)
        assert result[1] >= Decimal('11.00')  # Minimum is 1.10 * 10.00
        assert sum(result) == Decimal('100.00')  # Sum preserved
    
    def test_v1_too_high(self):
        """Test when v1 is above maximum."""
        vol = [Decimal('10.00'), Decimal('25.00'), Decimal('30.00'), Decimal('35.00')]
        result = _preserve_v1_band(vol)
        assert result[1] <= Decimal('20.00')  # Maximum is 2.00 * 10.00
        assert sum(result) == Decimal('100.00')  # Sum preserved


class TestPreserveM2:
    """Test the _preserve_m2 function."""
    
    def test_m2_within_range(self):
        """Test when m2 is already within range."""
        # m2 = v2 / (v0 + v1)
        vol = [Decimal('10.00'), Decimal('20.00'), Decimal('15.00'), Decimal('55.00')]
        # m2 = 15 / 30 = 0.5, which is within [0.10, 1.00]
        result = _preserve_m2(vol)
        assert result == vol
    
    def test_m2_too_low(self):
        """Test when m2 is below minimum."""
        vol = [Decimal('20.00'), Decimal('30.00'), Decimal('2.00'), Decimal('48.00')]
        # m2 = 2 / 50 = 0.04, which is < 0.10
        result = _preserve_m2(vol)
        v0, v1, v2 = float(result[0]), float(result[1]), float(result[2])
        m2 = v2 / (v0 + v1)
        assert m2 >= 0.10 - 0.01  # Allow small tolerance
    
    def test_m2_too_high(self):
        """Test when m2 is above maximum."""
        vol = [Decimal('5.00'), Decimal('5.00'), Decimal('20.00'), Decimal('70.00')]
        # m2 = 20 / 10 = 2.0, which is > 1.00
        result = _preserve_m2(vol)
        v0, v1, v2 = float(result[0]), float(result[1]), float(result[2])
        m2 = v2 / (v0 + v1)
        assert m2 <= 1.00 + 0.01  # Allow small tolerance


class TestPreserveQuartiles:
    """Test the _preserve_quartiles function."""
    
    def test_quartiles_within_range(self):
        """Test when quartiles are already within range."""
        # 12 orders: Q1 = first 3, Q4 = last 3
        vol = [
            Decimal('5.00'), Decimal('6.00'), Decimal('7.00'),  # Q1 = 18%
            Decimal('8.00'), Decimal('9.00'), Decimal('10.00'),  # Q2
            Decimal('11.00'), Decimal('12.00'), Decimal('13.00'),  # Q3
            Decimal('14.00'), Decimal('15.00'), Decimal('16.00')  # Q4 = 45%
        ]
        result = _preserve_quartiles(vol)
        assert result == vol  # Q1 <= 22%, Q4 >= 32%
    
    def test_q1_too_high(self):
        """Test when Q1 share is too high."""
        # 12 orders: Q1 = first 3
        vol = [
            Decimal('10.00'), Decimal('10.00'), Decimal('10.00'),  # Q1 = 30%
            Decimal('7.00'), Decimal('7.00'), Decimal('7.00'),  # Q2
            Decimal('7.00'), Decimal('7.00'), Decimal('7.00'),  # Q3
            Decimal('10.00'), Decimal('10.00'), Decimal('10.00')  # Q4 = 30%
        ]
        result = _preserve_quartiles(vol, q1_cap=22.0)
        q1_share = float(sum(result[:3]))
        assert q1_share <= 22.0 + 1.0  # Allow small tolerance
    
    def test_q4_too_low(self):
        """Test when Q4 share is too low."""
        # 12 orders: Q4 = last 3
        vol = [
            Decimal('10.00'), Decimal('10.00'), Decimal('10.00'),  # Q1 = 30%
            Decimal('10.00'), Decimal('10.00'), Decimal('10.00'),  # Q2
            Decimal('10.00'), Decimal('10.00'), Decimal('10.00'),  # Q3
            Decimal('3.00'), Decimal('3.00'), Decimal('4.00')  # Q4 = 10%
        ]
        result = _preserve_quartiles(vol, q4_floor=32.0)
        q4_share = float(sum(result[-3:]))
        assert q4_share >= 32.0 - 1.0  # Allow small tolerance


class TestNormalizeSchedule:
    """Test the main normalize_schedule_to_2dp function."""
    
    def test_basic_normalization(self):
        """Test basic normalization of a schedule."""
        schedule = {
            "volume_pct": [10.12345, 20.67890, 30.11111, 39.08654],
            "indent_pct": [0.00, 5.12345, 10.67890, 15.99999]
        }
        
        result = normalize_schedule_to_2dp(schedule)
        
        # Check all values are 2 decimal places
        for v in result["volume_pct"]:
            assert round(v, 2) == v
        for i in result["indent_pct"]:
            assert round(i, 2) == i
        
        # Check sum is 100
        assert abs(sum(result["volume_pct"]) - 100.0) < 0.01
        
        # Check monotonicity
        vol = result["volume_pct"]
        assert all(vol[i] <= vol[i+1] for i in range(len(vol)-1))
    
    def test_preserve_v1_band(self):
        """Test that v1 band constraint is preserved."""
        schedule = {
            "volume_pct": [10.0, 8.0, 30.0, 52.0],  # v1 violates band
            "indent_pct": [0.0, 5.0, 10.0, 15.0]
        }
        
        result = normalize_schedule_to_2dp(schedule, post_round_keep_v1_band=True)
        
        v0, v1 = result["volume_pct"][0], result["volume_pct"][1]
        assert 1.10 * v0 <= v1 <= 2.00 * v0
    
    def test_idempotency(self):
        """Test that normalization is idempotent."""
        schedule = {
            "volume_pct": [10.12345, 20.67890, 30.11111, 39.08654],
            "indent_pct": [0.00, 5.12345, 10.67890, 15.99999]
        }
        
        result1 = normalize_schedule_to_2dp(schedule)
        result2 = normalize_schedule_to_2dp(result1)
        
        assert result1["volume_pct"] == result2["volume_pct"]
        assert result1["indent_pct"] == result2["indent_pct"]
    
    def test_with_derived_fields(self):
        """Test normalization with derived fields."""
        schedule = {
            "volume_pct": [10.12345, 20.67890, 30.11111, 39.08654],
            "indent_pct": [0.00, 5.12345, 10.67890, 15.99999],
            "martingale_pct": [0.0, 204.5, 145.6, 129.7],
            "needpct": [10.12345, 21.71285, 33.12222, 45.76345]
        }
        
        result = normalize_schedule_to_2dp(schedule)
        
        # Check that martingale_pct is recomputed
        assert "martingale_pct" in result
        assert len(result["martingale_pct"]) == len(result["volume_pct"])
        
        # Check first martingale is 0
        assert result["martingale_pct"][0] == 0.0


class TestIsScheduleNormalized:
    """Test the is_schedule_normalized function."""
    
    def test_normalized_schedule(self):
        """Test that a normalized schedule is detected as such."""
        schedule = {
            "volume_pct": [10.00, 20.00, 30.00, 40.00],
            "indent_pct": [0.00, 5.00, 10.00, 15.00]
        }
        assert is_schedule_normalized(schedule) == True
    
    def test_unnormalized_schedule(self):
        """Test that an unnormalized schedule is detected."""
        schedule = {
            "volume_pct": [10.12345, 20.00, 30.00, 40.00],
            "indent_pct": [0.00, 5.00, 10.00, 15.00]
        }
        assert is_schedule_normalized(schedule) == False
    
    def test_wrong_sum(self):
        """Test that wrong sum is detected."""
        schedule = {
            "volume_pct": [10.00, 20.00, 30.00, 41.00],  # Sum = 101
            "indent_pct": [0.00, 5.00, 10.00, 15.00]
        }
        assert is_schedule_normalized(schedule) == False


class TestValidateNormalizedSchedule:
    """Test the validate_normalized_schedule function."""
    
    def test_valid_schedule(self):
        """Test validation of a valid normalized schedule."""
        schedule = {
            "volume_pct": [5.00, 10.00, 20.00, 30.00, 35.00],
            "indent_pct": [0.00, 5.00, 10.00, 15.00, 20.00]
        }
        
        results = validate_normalized_schedule(schedule)
        
        assert results["has_volumes"] == True
        assert results["volumes_2dp"] == True
        assert results["indents_2dp"] == True
        assert results["sum_is_100"] == True
        assert results["monotonic"] == True
        assert results["v1_band"] == True
        assert results["m2_valid"] == True
        assert results["all_valid"] == True
    
    def test_invalid_v1_band(self):
        """Test detection of v1 band violation."""
        schedule = {
            "volume_pct": [10.00, 8.00, 30.00, 52.00],  # v1 < 1.10 * v0
            "indent_pct": [0.00, 5.00, 10.00, 15.00]
        }
        
        results = validate_normalized_schedule(schedule)
        
        assert results["v1_band"] == False
        assert results["all_valid"] == False
    
    def test_comparison_with_original(self):
        """Test comparison with original schedule."""
        original = {
            "volume_pct": [10.12345, 20.67890, 30.11111, 39.08654],
            "indent_pct": [0.00, 5.12345, 10.67890, 15.99999]
        }
        
        normalized = normalize_schedule_to_2dp(original)
        
        results = validate_normalized_schedule(normalized, original_schedule=original)
        
        assert "max_diff_from_original" in results
        assert "preserves_shape" in results
        assert results["preserves_shape"] == True  # Differences should be small


class TestRealWorldScenarios:
    """Test with real-world scenarios and edge cases."""
    
    def test_large_number_of_orders(self):
        """Test with many orders (e.g., 50)."""
        n = 50
        # Create a geometric progression
        base = 0.5
        growth = 1.08
        volumes = [base * (growth ** i) for i in range(n)]
        total = sum(volumes)
        volumes = [v / total * 100 for v in volumes]
        
        schedule = {
            "volume_pct": volumes,
            "indent_pct": [i * 0.5 for i in range(n)]
        }
        
        result = normalize_schedule_to_2dp(schedule)
        
        assert len(result["volume_pct"]) == n
        assert abs(sum(result["volume_pct"]) - 100.0) < 0.01
        assert all(result["volume_pct"][i] <= result["volume_pct"][i+1] for i in range(n-1))
    
    def test_extreme_values(self):
        """Test with extreme value distributions."""
        schedule = {
            "volume_pct": [0.01, 0.02, 0.03, 99.94],  # Very skewed
            "indent_pct": [0.00, 0.01, 0.02, 0.03]
        }
        
        result = normalize_schedule_to_2dp(schedule)
        
        assert abs(sum(result["volume_pct"]) - 100.0) < 0.01
        assert all(round(v, 2) == v for v in result["volume_pct"])
    
    def test_all_equal_volumes(self):
        """Test with all equal volumes."""
        n = 10
        schedule = {
            "volume_pct": [10.0] * n,
            "indent_pct": [i * 2.0 for i in range(n)]
        }
        
        result = normalize_schedule_to_2dp(schedule, strict_monotonicity=False)
        
        assert abs(sum(result["volume_pct"]) - 100.0) < 0.01
        # With strict=False, equal values are allowed
        assert all(result["volume_pct"][i] <= result["volume_pct"][i+1] for i in range(n-1))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])