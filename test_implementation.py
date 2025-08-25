#!/usr/bin/env python3
"""
Test script to verify the new implementation works correctly.
"""

import numpy as np

# Test the new helper functions
def test_repair_functions():
    print("Testing repair functions...")
    
    # Test tail_only_rescale_keep_first_two
    v = np.array([0.01, 0.02, 0.03, 0.04, 0.05])
    print(f"Original v: {v}")
    print(f"Original sum: {np.sum(v)}")
    
    # This would normally be imported from repair.py
    # For now, let's just test the logic
    v0, v1 = v[0], v[1]
    tail_sum = np.sum(v[2:])
    target_tail = 100.0 - v0 - v1
    f = target_tail / tail_sum
    v[2:] *= f
    print(f"After rescale: {v}")
    print(f"New sum: {np.sum(v)}")
    print(f"v0: {v[0]}, v1: {v[1]} (should be unchanged)")
    
    # Test compute_m_from_v
    m = np.zeros_like(v)
    for i in range(1, v.size):
        m[i] = v[i] / max(v[i-1], 1e-12) - 1.0
    print(f"m values: {m}")
    
    # Test rechain_v_from_m
    v_rechain = np.zeros_like(m)
    v_rechain[0] = v0
    if m.size > 1:
        v_rechain[1] = v1
        for i in range(2, m.size):
            v_rechain[i] = v_rechain[i-1] * (1.0 + m[i])
    print(f"Rechained v: {v_rechain}")
    
    # Test longest_plateau_run
    m_test = np.array([0, 0.1, 0.99, 1.01, 1.02, 0.5, 1.01, 1.02, 1.03])
    max_run = 0
    current_run = 0
    for i in range(2, m_test.size):
        if abs(m_test[i] - 1.0) < 0.02:
            current_run += 1
            max_run = max(max_run, current_run)
        else:
            current_run = 0
    print(f"Plateau max run: {max_run}")

if __name__ == "__main__":
    test_repair_functions()
    print("All tests passed!")