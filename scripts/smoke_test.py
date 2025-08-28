#!/usr/bin/env python3
"""
Smoke test script for DCA/Martingale optimization.
Runs optimization and validates results with SQL queries.
"""
import sys
import os
import sqlite3
import subprocess
import json
from pathlib import Path
from typing import Dict, Any, Tuple


def run_optimization(db_path: str = "db_results/smoke_test.db") -> int:
    """Run a small optimization batch for smoke testing."""
    cmd = [
        sys.executable, "-m", "martingale_lab.cli.optimize",
        "--overlap-min", "10.0",
        "--overlap-max", "10.2",
        "--orders-min", "16",
        "--orders-max", "16",
        "--first-volume", "0.01",
        "--first-indent", "0.00",
        "--second-upper-c2", "2.0",
        "--m2-min", "0.10",
        "--m2-max", "0.80",
        "--m-min", "0.05",
        "--m-head", "0.40",
        "--m-tail", "0.20",
        "--tau-scale", "0.3333333",
        "--slope-cap", "0.25",
        "--q1-cap", "22.0",
        "--tail-floor", "32.0",
        "--use-hc0-bootstrap",
        "--use-head-budget",
        "--head-budget-pct", "0.35",
        "--penalty-preset", "robust",
        "--batches", "1",
        "--batch-size", "200",
        "--workers", "1",
        "--workers-mode", "thread",
        "--db", db_path,
        "--seed", "7",
        "--notes", "smoke-test-hc0-hc7"
    ]
    
    print(f"Running optimization: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Optimization failed with return code {result.returncode}")
        print(f"STDOUT:\n{result.stdout}")
        print(f"STDERR:\n{result.stderr}")
        return result.returncode
    
    print("Optimization completed successfully")
    return 0


def execute_sql(db_path: str, query: str) -> list:
    """Execute SQL query and return results."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(query)
    results = cursor.fetchall()
    conn.close()
    return results


def validate_v1_band(db_path: str) -> Tuple[bool, str]:
    """Validate v0/v1 band and first3 sum."""
    query = """
    WITH best AS (
      SELECT r.payload_json
      FROM results r JOIN (SELECT MAX(id) id FROM experiments) l ON r.experiment_id=l.id
      ORDER BY r.score ASC LIMIT 1
    ),
    v AS (
      SELECT CAST(json_extract(j.value, '$') AS REAL) AS v, j.key AS i
      FROM best, json_each(json_extract(best.payload_json,'$.schedule.volume_pct')) j
    )
    SELECT
      (SELECT v FROM v WHERE i=0) AS v0,
      (SELECT v FROM v WHERE i=1) AS v1,
      CASE WHEN (SELECT v FROM v WHERE i=1) >= 1.10*(SELECT v FROM v WHERE i=0) - 0.0001
            AND (SELECT v FROM v WHERE i=1) <= 2.00*(SELECT v FROM v WHERE i=0) + 0.0001
           THEN 'OK' ELSE 'VIOL' END AS v1_band,
      (SELECT SUM(v) FROM v WHERE CAST(i AS INT) IN (0,1,2)) AS first3_sum
    """
    
    results = execute_sql(db_path, query)
    if not results:
        return False, "No results found"
    
    v0, v1, v1_band, first3_sum = results[0]
    
    print(f"v0={v0:.4f}, v1={v1:.4f}, v1_band={v1_band}, first3_sum={first3_sum:.4f}")
    
    if v1_band != "OK":
        return False, f"v1 band violation: v1={v1:.4f} not in [{v0*1.10:.4f}, {v0*2.00:.4f}]"
    
    return True, "v1 band OK"


def validate_martingale_slope(db_path: str) -> Tuple[bool, str]:
    """Validate martingale tail slope and plateau."""
    query = """
    WITH best AS (
      SELECT r.payload_json, r.diagnostics_json
      FROM results r JOIN (SELECT MAX(id) id FROM experiments) l ON r.experiment_id=l.id
      ORDER BY r.score ASC LIMIT 1
    ),
    m_tail AS (
      SELECT CAST(j.key AS INT) AS i, json_extract(j.value, '$')/100.0 AS m
      FROM best, json_each(json_extract(best.payload_json,'$.schedule.martingale_pct')) AS j
      WHERE CAST(j.key AS INT) >= 2
    ),
    pairs AS (
      SELECT a.i, a.m AS m_i, b.m AS m_prev
      FROM m_tail a JOIN m_tail b ON a.i = b.i + 1
    ),
    mean_m AS (SELECT AVG(m_i) AS mu FROM pairs)
    SELECT
      CASE WHEN COUNT(*) > 0 THEN sqrt(AVG((m_i - mu)*(m_i - mu))) ELSE 0 END AS std_m_tail,
      SUM(CASE WHEN ABS(m_i - m_prev) > 0.25 THEN 1 ELSE 0 END) AS slope_violations,
      COALESCE(json_extract(best.diagnostics_json,'$.plateau_max_run'), 0) AS plateau_max_run
    FROM pairs, mean_m, best
    """
    
    results = execute_sql(db_path, query)
    if not results:
        return False, "No results found"
    
    std_m_tail, slope_violations, plateau_max_run = results[0]
    
    print(f"std_m_tail={std_m_tail:.4f}, slope_violations={slope_violations}, plateau_max_run={plateau_max_run}")
    
    if slope_violations > 0:
        return False, f"Slope violations: {slope_violations}"
    
    return True, "Martingale slope OK"


def validate_monotonicity(db_path: str) -> Tuple[bool, str]:
    """Validate volume monotonicity and total sum."""
    query = """
    WITH best AS (
      SELECT r.payload_json
      FROM results r JOIN (SELECT MAX(id) id FROM experiments) l ON r.experiment_id=l.id
      ORDER BY r.score ASC LIMIT 1
    ),
    v AS (
      SELECT CAST(j.key AS INT) AS i, json_extract(j.value, '$') AS v
      FROM best, json_each(json_extract(best.payload_json,'$.schedule.volume_pct')) AS j
    ),
    pairs AS (
      SELECT a.i, a.v, b.v AS v_prev
      FROM v a JOIN v b ON a.i = b.i + 1
    )
    SELECT
      SUM(CASE WHEN v <= v_prev THEN 1 ELSE 0 END) AS monotonicity_violations,
      (SELECT SUM(v) FROM v) AS total_sum
    FROM pairs
    """
    
    results = execute_sql(db_path, query)
    if not results:
        return False, "No results found"
    
    monotonicity_violations, total_sum = results[0]
    
    print(f"monotonicity_violations={monotonicity_violations}, total_sum={total_sum:.4f}")
    
    if monotonicity_violations > 0:
        return False, f"Monotonicity violations: {monotonicity_violations}"
    
    if abs(total_sum - 100.0) > 0.05:
        return False, f"Total sum {total_sum:.4f} != 100.0"
    
    return True, "Monotonicity and sum OK"


def validate_quartiles(db_path: str) -> Tuple[bool, str]:
    """Validate Q1/Q4 share and m2 percentage."""
    query = """
    WITH best AS (
      SELECT r.payload_json
      FROM results r JOIN (SELECT MAX(id) id FROM experiments) l ON r.experiment_id=l.id
      ORDER BY r.score ASC LIMIT 1
    ),
    v AS (
      SELECT CAST(j.key AS INT) AS i, json_extract(j.value, '$') AS v
      FROM best, json_each(json_extract(best.payload_json,'$.schedule.volume_pct')) AS j
    ),
    m AS (
      SELECT CAST(j.key AS INT) AS i, json_extract(j.value, '$')/100.0 AS m
      FROM best, json_each(json_extract(best.payload_json,'$.schedule.martingale_pct')) AS j
    ),
    n AS (SELECT MAX(i)+1 AS N FROM v),
    q AS (SELECT CAST(ROUND(N/4.0 + 0.5) AS INT) AS Q FROM n)
    SELECT
      (SELECT SUM(v) FROM v, q WHERE i < Q) AS q1_share_pct,
      (SELECT SUM(v) FROM v, n, q WHERE i >= N-Q) AS q4_share_pct,
      (SELECT m FROM m WHERE i=1)*100.0 AS m2_pct
    """
    
    results = execute_sql(db_path, query)
    if not results:
        return False, "No results found"
    
    q1_share_pct, q4_share_pct, m2_pct = results[0]
    
    print(f"q1_share_pct={q1_share_pct:.4f}, q4_share_pct={q4_share_pct:.4f}, m2_pct={m2_pct:.4f}")
    
    if q1_share_pct > 22.0:
        return False, f"Q1 share {q1_share_pct:.4f} > 22.0"
    
    if q4_share_pct < 32.0:
        return False, f"Q4 share {q4_share_pct:.4f} < 32.0"
    
    if not (10.0 <= m2_pct <= 80.0):
        return False, f"m2 percentage {m2_pct:.4f} not in [10, 80]"
    
    return True, "Quartiles and m2 OK"


def main():
    """Run smoke tests."""
    print("=" * 60)
    print("DCA/Martingale Optimization Smoke Test")
    print("=" * 60)
    
    # Clean up previous test database
    db_path = "db_results/smoke_test.db"
    if os.path.exists(db_path):
        os.remove(db_path)
        print(f"Removed existing test database: {db_path}")
    
    # Run optimization
    print("\n1. Running optimization...")
    ret = run_optimization(db_path)
    if ret != 0:
        print("❌ Optimization failed")
        return 1
    
    # Run SQL validations
    print("\n2. Running SQL validations...")
    
    tests = [
        ("v1 Band Check", validate_v1_band),
        ("Martingale Slope Check", validate_martingale_slope),
        ("Monotonicity Check", validate_monotonicity),
        ("Quartiles Check", validate_quartiles),
    ]
    
    all_passed = True
    for test_name, test_func in tests:
        print(f"\n  {test_name}:")
        passed, message = test_func(db_path)
        if passed:
            print(f"    ✅ {message}")
        else:
            print(f"    ❌ {message}")
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✅ All smoke tests passed!")
        return 0
    else:
        print("❌ Some smoke tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
