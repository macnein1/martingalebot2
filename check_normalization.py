#!/usr/bin/env python3
"""Check if normalization is working in stored results."""
import sqlite3
import json

# Connect to the database
db_path = "db_results/test_normalization.db"
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Get the first result
cursor.execute("""
    SELECT payload_json 
    FROM results 
    WHERE experiment_id = 1 
    LIMIT 5
""")

results = cursor.fetchall()
conn.close()

print("=" * 60)
print("Checking Normalization in Database Results")
print("=" * 60)

for i, (payload_str,) in enumerate(results, 1):
    payload = json.loads(payload_str)
    schedule = payload.get("schedule", {})
    volume_pct = schedule.get("volume_pct", [])
    indent_pct = schedule.get("indent_pct", [])
    
    print(f"\nResult {i}:")
    print(f"  Orders: {len(volume_pct)}")
    
    # Check volumes
    if volume_pct:
        all_2dp = all(round(v, 2) == v for v in volume_pct)
        total = sum(volume_pct)
        print(f"  First 3 volumes: {volume_pct[:3]}")
        print(f"  Last 3 volumes: {volume_pct[-3:]}")
        print(f"  All volumes 2dp? {all_2dp}")
        print(f"  Sum: {total:.2f}")
        
        # Check monotonicity
        monotonic = all(volume_pct[i] <= volume_pct[i+1] for i in range(len(volume_pct)-1))
        print(f"  Monotonic? {monotonic}")
        
        # Check v1 band
        if len(volume_pct) >= 2:
            v0, v1 = volume_pct[0], volume_pct[1]
            v1_band_ok = 1.10 * v0 <= v1 <= 2.00 * v0
            print(f"  v1 band OK? {v1_band_ok} (v0={v0:.2f}, v1={v1:.2f})")
    
    # Check indents
    if indent_pct:
        all_2dp_indent = all(round(i, 2) == i for i in indent_pct)
        print(f"  All indents 2dp? {all_2dp_indent}")
        print(f"  First 3 indents: {indent_pct[:3]}")

print("\n" + "=" * 60)
print("✓ Normalization check complete!")
print("=" * 60)