#!/bin/bash
#
# CI Smoke Test Script for Martingale Lab
# Runs optimization and validates results
#

set -e  # Exit on error

echo "=========================================="
echo "CI Smoke Test for Martingale Lab"
echo "=========================================="

# Clean up previous test database
rm -f db_results/ci_smoke.db

# Run optimization
echo "1. Running optimization..."
python3 -m martingale_lab.cli.optimize \
  --overlap-min 10.0 --overlap-max 10.2 \
  --orders-min 16 --orders-max 16 \
  --first-volume 0.01 --first-indent 0.00 \
  --second-upper-c2 2.0 \
  --m2-min 0.10 --m2-max 0.80 \
  --m-min 0.05 \
  --m-head 0.40 --m-tail 0.20 --tau-scale 0.3333333 \
  --slope-cap 0.25 \
  --q1-cap 22.0 --tail-floor 32.0 \
  --use-hc0-bootstrap \
  --use-head-budget --head-budget-pct 0.35 \
  --penalty-preset robust \
  --batches 1 --batch-size 100 \
  --workers 1 --workers-mode thread \
  --db db_results/ci_smoke.db \
  --seed 42 \
  --notes "ci-smoke-test"

if [ $? -ne 0 ]; then
  echo "❌ Optimization failed"
  exit 1
fi

echo "✅ Optimization completed"

# Run SQL validations
echo ""
echo "2. Running SQL validations..."

# Test 1: Volume monotonicity and sum
echo -n "  Testing volume monotonicity and sum... "
RESULT=$(sqlite3 db_results/ci_smoke.db "
WITH best AS (
  SELECT r.payload_json
  FROM results r JOIN (SELECT MAX(id) id FROM experiments) l ON r.experiment_id=l.id
  ORDER BY r.score ASC LIMIT 1
),
v AS (
  SELECT CAST(j.key AS INT) AS i, json_extract(j.value, '\$') AS v
  FROM best, json_each(json_extract(best.payload_json,'\$.schedule.volume_pct')) AS j
)
SELECT 
  SUM(v) AS total_sum,
  SUM(CASE WHEN i > 0 AND v < (SELECT v FROM v v2 WHERE v2.i = v.i-1) THEN 1 ELSE 0 END) AS violations
FROM v
")

TOTAL=$(echo "$RESULT" | cut -d'|' -f1)
VIOLATIONS=$(echo "$RESULT" | cut -d'|' -f2)

if [ "$VIOLATIONS" -eq 0 ] && (( $(echo "$TOTAL >= 99.95 && $TOTAL <= 100.05" | bc -l) )); then
  echo "✅ PASS (sum=$TOTAL, violations=$VIOLATIONS)"
else
  echo "❌ FAIL (sum=$TOTAL, violations=$VIOLATIONS)"
  exit 1
fi

# Test 2: First order volume
echo -n "  Testing first order volume constraint... "
V0=$(sqlite3 db_results/ci_smoke.db "
WITH best AS (
  SELECT r.payload_json
  FROM results r JOIN (SELECT MAX(id) id FROM experiments) l ON r.experiment_id=l.id
  ORDER BY r.score ASC LIMIT 1
)
SELECT json_extract(best.payload_json, '\$.schedule.volume_pct[0]')
FROM best
")

if (( $(echo "$V0 >= 0.009 && $V0 <= 0.011" | bc -l) )); then
  echo "✅ PASS (v0=$V0)"
else
  echo "❌ FAIL (v0=$V0, expected 0.01)"
  exit 1
fi

echo ""
echo "=========================================="
echo "✅ All CI smoke tests passed!"
echo "=========================================="
