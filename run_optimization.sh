#!/bin/bash

# Optimization script tuned for your strategy profile
# Your strategy: Q1=6.3%, Q4=53.8%, m[2]=0.118, Sharpe=1.3+

echo "Starting martingale optimization..."
echo "Target: Find strategies similar to your 1.3+ Sharpe profile"
echo "========================================="

# Clean previous results
rm -rf db_results/
mkdir -p db_results/

# Run optimization with your strategy parameters
python3 -m martingale_lab.cli.optimize \
  --overlap-min 10.0 --overlap-max 10.2 \
  --orders-min 20 --orders-max 20 \
  --wave-mode blocks --blocks 4 \
  --first-volume 1.0 --first-indent 0.00 \
  --m2-min 0.10 --m2-max 0.20 \
  --m-min 0.05 --m-max 0.30 \
  --front-cap 7.0 \
  --tail-cap 0.55 \
  --penalty-preset robust \
  --batches 3 \
  --batch-size 100 \
  --workers 2 --workers-mode thread \
  --prune-threshold 10.0 \
  --grace-batches 1 \
  --patience 2 \
  --log-level INFO \
  --log-eval-sample 0.1 \
  --log-every-batch 1 \
  --db db_results/optimization.db \
  --seed 42 \
  --notes "Exit-ease optimization targeting your strategy profile"

echo ""
echo "Optimization complete!"
echo ""

# Check if we have results
if [ -f "db_results/optimization.db" ]; then
    echo "Database created successfully"
    echo ""
    
    # Try to get result count
    python3 -c "
import sqlite3
conn = sqlite3.connect('db_results/optimization.db')
cursor = conn.cursor()
try:
    cursor.execute('SELECT COUNT(*) FROM results WHERE score < 1000000')
    count = cursor.fetchone()[0]
    print(f'Found {count} valid results')
    
    if count > 0:
        cursor.execute('''
            SELECT score, max_need, var_need, tail
            FROM results 
            WHERE score < 1000000
            ORDER BY score ASC 
            LIMIT 1
        ''')
        best = cursor.fetchone()
        print(f'Best score: {best[0]:.2f}')
        print(f'Max need: {best[1]:.2f}')
        print(f'Var need: {best[2]:.2f}')
        print(f'Tail: {best[3]:.2f}')
except Exception as e:
    print(f'Error reading results: {e}')
finally:
    conn.close()
"
else
    echo "No database found - optimization may have failed"
fi

echo ""
echo "Volume distribution of best strategy:"
echo "========================================="

sqlite3 -cmd ".headers on" -cmd ".mode column" db_results/optimization.db <<'SQL'
WITH best AS (
  SELECT payload_json
  FROM results 
  WHERE score < 1000000
  ORDER BY score ASC 
  LIMIT 1
),
volumes AS (
  SELECT 
    json_extract(value, '$') as vol,
    key as idx
  FROM best, json_each(json_extract(payload_json, '$.schedule.volume_pct'))
  WHERE CAST(key AS INT) < 10
)
SELECT 
  'Order ' || (CAST(idx AS INT) + 1) as order_num,
  printf('%.2f%%', vol) as volume
FROM volumes
ORDER BY CAST(idx AS INT);
SQL