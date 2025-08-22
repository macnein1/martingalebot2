"""
Optimizer Adapters
Unified adapters that wrap different optimization strategies with a common interface
"""

from .auto_batch_adapter import AutoBatchAdapter
from .numba_adapter import NumbaAdapter

__all__ = [
    'AutoBatchAdapter',
    'NumbaAdapter', 
]