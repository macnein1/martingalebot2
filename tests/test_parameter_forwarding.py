"""
Unit tests for parameter forwarding module.
"""
import pytest
from martingale_lab.core.parameter_forwarding import (
    get_function_parameters,
    filter_kwargs_for_function,
    safe_call,
    map_parameter_names
)


def sample_function(a: int, b: str, c: float = 1.0) -> str:
    """Sample function for testing."""
    return f"{a}-{b}-{c}"


def test_get_function_parameters():
    """Test parameter extraction from function signature."""
    params = get_function_parameters(sample_function)
    
    assert params == {'a', 'b', 'c'}
    assert 'self' not in params  # Should exclude special parameters


def test_filter_kwargs_basic():
    """Test basic kwargs filtering."""
    kwargs = {
        'a': 1,
        'b': 'test',
        'c': 2.0,
        'd': 'extra',  # Not in function signature
        'e': None
    }
    
    filtered = filter_kwargs_for_function(sample_function, kwargs)
    
    assert filtered == {'a': 1, 'b': 'test', 'c': 2.0}
    assert 'd' not in filtered
    assert 'e' not in filtered


def test_filter_kwargs_exclude_none():
    """Test None value exclusion."""
    kwargs = {
        'a': 1,
        'b': None,
        'c': 2.0
    }
    
    # With exclude_none=True (default)
    filtered = filter_kwargs_for_function(sample_function, kwargs, exclude_none=True)
    assert 'b' not in filtered
    
    # With exclude_none=False
    filtered = filter_kwargs_for_function(sample_function, kwargs, exclude_none=False)
    assert filtered['b'] is None


def test_safe_call():
    """Test safe function calling."""
    result = safe_call(sample_function, a=1, b='test', c=3.0, extra='ignored')
    assert result == '1-test-3.0'
    
    # Should work even with missing optional parameters
    result = safe_call(sample_function, a=2, b='hello', unknown='param')
    assert result == '2-hello-1.0'  # Uses default for c


def test_map_parameter_names():
    """Test parameter name mapping."""
    kwargs = {
        'w_sec': 1.0,
        'w_band': 2.0,
        'strict_inc_eps': 0.001,
        'other_param': 'value'
    }
    
    mapped = map_parameter_names(kwargs)
    
    assert mapped['w_second'] == 1.0
    assert mapped['w_gband'] == 2.0
    assert mapped['eps_inc'] == 0.001
    assert mapped['other_param'] == 'value'  # Unchanged


def test_custom_aliases():
    """Test custom parameter aliases."""
    kwargs = {'old_name': 'value', 'another': 123}
    aliases = {'old_name': 'new_name'}
    
    mapped = map_parameter_names(kwargs, aliases)
    
    assert mapped['new_name'] == 'value'
    assert mapped['another'] == 123
    assert 'old_name' not in mapped


class TestClassMethod:
    """Test with class methods."""
    
    def method(self, x: int, y: str = 'default') -> str:
        return f"{x}-{y}"
    
    def test_class_method_params(self):
        """Test parameter extraction excludes 'self'."""
        params = get_function_parameters(self.method)
        assert 'self' not in params
        assert params == {'x', 'y'}


def test_enforce_schedule_shape_fixed_compatibility():
    """Test that enforce_schedule_shape_fixed can be called safely."""
    from martingale_lab.core.constraints import enforce_schedule_shape_fixed
    
    # Prepare a large set of parameters (including unknown ones)
    all_params = {
        'indent_pct': [0.0, 0.5, 1.0],
        'volume_pct': [33.33, 33.33, 33.34],
        'base_price': 100.0,
        'first_volume_target': 0.01,
        'first_indent_target': 0.0,
        'k_front': 3,
        'front_cap': 5.0,
        'g_min': 1.01,
        'g_max': 1.20,
        'slope_cap': 0.25,
        'unknown_param': 'should be filtered',
        'future_param': 123
    }
    
    # Filter to safe parameters
    safe_params = filter_kwargs_for_function(enforce_schedule_shape_fixed, all_params)
    
    # Should not include unknown parameters
    assert 'unknown_param' not in safe_params
    assert 'future_param' not in safe_params
    
    # Should be able to call without error
    result = enforce_schedule_shape_fixed(**safe_params)
    assert len(result) == 7  # Returns 7-tuple
    assert result[0] is not None  # indent_pct
    assert result[1] is not None  # volume_pct


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
