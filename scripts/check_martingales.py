#!/usr/bin/env python3
"""Check martingale values and slope violations."""
import sqlite3
import json

db_path = "db_results/smoke_test.db"
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Get best result
query = """
WITH best AS (
  SELECT r.payload_json
  FROM results r JOIN (SELECT MAX(id) id FROM experiments) l ON r.experiment_id=l.id
  ORDER BY r.score ASC LIMIT 1
)
SELECT payload_json FROM best
"""

cursor.execute(query)
result = cursor.fetchone()
if result:
    payload = json.loads(result[0])
    martingales = [m/100.0 for m in payload['schedule']['martingale_pct']]
    
    print("Martingale values (as fractions):")
    for i, m in enumerate(martingales[:10]):
        print(f"  m[{i}] = {m:.4f}")
    
    print("\nMartingale differences:")
    for i in range(1, min(10, len(martingales))):
        diff = martingales[i] - martingales[i-1]
        violation = " VIOLATION!" if abs(diff) > 0.25 else ""
        print(f"  m[{i}] - m[{i-1}] = {diff:.4f}{violation}")

conn.close()
