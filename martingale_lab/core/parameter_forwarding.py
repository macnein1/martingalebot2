"""
Safe parameter forwarding utilities for signature-aware function calls.
Ensures compatibility between different component versions.
"""
from __future__ import annotations

import inspect
from typing import Any, Callable, Dict, Set


def get_function_parameters(func: Callable) -> Set[str]:
    """
    Extract parameter names from a function signature.
    
    Args:
        func: Function to inspect
        
    Returns:
        Set of parameter names accepted by the function
    """
    sig = inspect.signature(func)
    params = set()
    
    for name, param in sig.parameters.items():
        # Skip *args and **kwargs
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        params.add(name)
    
    return params


def filter_kwargs_for_function(func: Callable, kwargs: Dict[str, Any], 
                              exclude_none: bool = True) -> Dict[str, Any]:
    """
    Filter kwargs to only include parameters accepted by the target function.
    
    Args:
        func: Target function to call
        kwargs: All available keyword arguments
        exclude_none: If True, exclude None values from filtered kwargs
        
    Returns:
        Filtered kwargs that are safe to pass to the function
    """
    accepted_params = get_function_parameters(func)
    filtered = {}
    
    for key, value in kwargs.items():
        if key in accepted_params:
            if exclude_none and value is None:
                continue
            filtered[key] = value
    
    return filtered


def safe_call(func: Callable, **kwargs) -> Any:
    """
    Safely call a function with only the parameters it accepts.
    
    Args:
        func: Function to call
        **kwargs: All available keyword arguments
        
    Returns:
        Result of function call
    """
    filtered_kwargs = filter_kwargs_for_function(func, kwargs)
    return func(**filtered_kwargs)


# Parameter mapping for legacy compatibility
PARAMETER_ALIASES = {
    # Old name -> New name mappings
    "w_sec": "w_second",
    "w_band": "w_gband",
    "strict_inc_eps": "eps_inc",
    "first_volume": "first_volume_target",
    "first_indent": "first_indent_target",
    "g_pre_min": "g_min",
    "g_pre_max": "g_max",
}


def map_parameter_names(kwargs: Dict[str, Any], aliases: Dict[str, str] = None) -> Dict[str, Any]:
    """
    Map parameter names using aliases for backward compatibility.
    
    Args:
        kwargs: Original keyword arguments
        aliases: Dictionary of old_name -> new_name mappings
        
    Returns:
        Kwargs with mapped parameter names
    """
    if aliases is None:
        aliases = PARAMETER_ALIASES
    
    mapped = {}
    for key, value in kwargs.items():
        # Use new name if alias exists, otherwise keep original
        mapped_key = aliases.get(key, key)
        mapped[mapped_key] = value
    
    return mapped
