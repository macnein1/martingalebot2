-- SQL Queries for Martingale Optimization Analysis
-- ================================================

-- 1. Best Strategy by Score
-- -------------------------
WITH best AS (
  SELECT 
    r.id,
    r.experiment_id,
    r.score,
    r.payload_json,
    r.diagnostics_json,
    e.notes
  FROM results r
  JOIN experiments e ON r.experiment_id = e.id
  WHERE r.score < 100000  -- Exclude inf scores
  ORDER BY r.score ASC
  LIMIT 1
)
SELECT 
  printf('%.2f', score) as score,
  json_extract(diagnostics_json, '$.q1_share') as q1_pct,
  json_extract(diagnostics_json, '$.q4_share') as q4_pct,
  printf('%.3f', json_extract(diagnostics_json, '$.m2')) as m2,
  printf('%.3f', json_extract(diagnostics_json, '$.ee_harmonic')) as ee_harmonic,
  printf('%.3f', json_extract(diagnostics_json, '$.ee_tail_weighted')) as ee_tail,
  notes
FROM best;

-- 2. Top 10 Strategies Comparison
-- --------------------------------
WITH top_strategies AS (
  SELECT 
    r.*,
    ROW_NUMBER() OVER (ORDER BY r.score ASC) as rank
  FROM results r
  WHERE r.score < 100000
)
SELECT 
  rank,
  printf('%.2f', score) as score,
  printf('%.1f', json_extract(diagnostics_json, '$.q1_share')) as q1,
  printf('%.1f', json_extract(diagnostics_json, '$.q4_share')) as q4,
  printf('%.3f', json_extract(diagnostics_json, '$.m2')) as m2,
  printf('%.3f', json_extract(diagnostics_json, '$.ee_tail_weighted')) as ee_tail
FROM top_strategies
WHERE rank <= 10;

-- 3. Strategy Distribution Analysis
-- ----------------------------------
WITH strategy_metrics AS (
  SELECT 
    json_extract(diagnostics_json, '$.q1_share') as q1,
    json_extract(diagnostics_json, '$.q4_share') as q4,
    json_extract(diagnostics_json, '$.m2') as m2,
    json_extract(diagnostics_json, '$.ee_tail_weighted') as ee_tail,
    score
  FROM results
  WHERE score < 100000
)
SELECT 
  'Q1 Distribution' as metric,
  printf('%.1f', MIN(q1)) as min_val,
  printf('%.1f', AVG(q1)) as avg_val,
  printf('%.1f', MAX(q1)) as max_val
FROM strategy_metrics
UNION ALL
SELECT 
  'Q4 Distribution',
  printf('%.1f', MIN(q4)),
  printf('%.1f', AVG(q4)),
  printf('%.1f', MAX(q4))
FROM strategy_metrics
UNION ALL
SELECT 
  'm[2] Distribution',
  printf('%.3f', MIN(m2)),
  printf('%.3f', AVG(m2)),
  printf('%.3f', MAX(m2))
FROM strategy_metrics
UNION ALL
SELECT 
  'Exit-Ease Tail',
  printf('%.3f', MIN(ee_tail)),
  printf('%.3f', AVG(ee_tail)),
  printf('%.3f', MAX(ee_tail))
FROM strategy_metrics;

-- 4. Volume Profile of Best Strategy
-- -----------------------------------
WITH best AS (
  SELECT payload_json
  FROM results
  WHERE score < 100000
  ORDER BY score ASC
  LIMIT 1
),
volumes AS (
  SELECT 
    CAST(key AS INT) + 1 as order_num,
    printf('%.2f', value) as volume_pct
  FROM best, 
       json_each(json_extract(payload_json, '$.schedule.volume_pct'))
  WHERE CAST(key AS INT) < 20
)
SELECT * FROM volumes
ORDER BY order_num;

-- 5. Martingale Ratios Analysis
-- ------------------------------
WITH best AS (
  SELECT payload_json
  FROM results
  WHERE score < 100000
  ORDER BY score ASC
  LIMIT 1
),
martingales AS (
  SELECT 
    CAST(key AS INT) as idx,
    value/100.0 as m_ratio
  FROM best,
       json_each(json_extract(payload_json, '$.schedule.martingale_pct'))
  WHERE CAST(key AS INT) >= 2 AND CAST(key AS INT) < 10
)
SELECT 
  idx as i,
  printf('%.3f', m_ratio) as m_value,
  CASE 
    WHEN m_ratio < 0.05 THEN '⚠️ Too Low'
    WHEN m_ratio > 0.30 THEN '⚠️ Too High'
    ELSE '✓ Good'
  END as status
FROM martingales
ORDER BY idx;

-- 6. Strategies Meeting Your Criteria
-- ------------------------------------
WITH filtered AS (
  SELECT 
    r.*,
    json_extract(r.diagnostics_json, '$.q1_share') as q1,
    json_extract(r.diagnostics_json, '$.q4_share') as q4,
    json_extract(r.diagnostics_json, '$.m2') as m2,
    json_extract(r.diagnostics_json, '$.ee_tail_weighted') as ee_tail
  FROM results r
  WHERE r.score < 100000
    AND json_extract(r.diagnostics_json, '$.q1_share') < 10  -- Low front-loading
    AND json_extract(r.diagnostics_json, '$.q4_share') > 50  -- Strong tail
    AND json_extract(r.diagnostics_json, '$.m2') BETWEEN 0.10 AND 0.20  -- Reasonable m2
    AND json_extract(r.diagnostics_json, '$.ee_tail_weighted') > 0.6  -- Good exit-ease
)
SELECT 
  COUNT(*) as matching_strategies,
  printf('%.2f', MIN(score)) as best_score,
  printf('%.2f', AVG(score)) as avg_score
FROM filtered;

-- 7. Parameter Sensitivity Analysis
-- ----------------------------------
WITH param_analysis AS (
  SELECT 
    json_extract(params_json, '$.overlap_pct') as overlap,
    json_extract(params_json, '$.num_orders') as n_orders,
    AVG(score) as avg_score,
    COUNT(*) as count
  FROM results
  WHERE score < 100000
  GROUP BY overlap, n_orders
)
SELECT 
  overlap,
  n_orders,
  printf('%.2f', avg_score) as avg_score,
  count
FROM param_analysis
ORDER BY avg_score ASC
LIMIT 10;

-- 8. Experiment Summary
-- ---------------------
SELECT 
  e.id as exp_id,
  e.created_at,
  COUNT(r.id) as total_results,
  COUNT(CASE WHEN r.score < 100000 THEN 1 END) as valid_results,
  printf('%.2f', MIN(CASE WHEN r.score < 100000 THEN r.score END)) as best_score,
  e.notes
FROM experiments e
LEFT JOIN results r ON e.id = r.experiment_id
GROUP BY e.id
ORDER BY e.created_at DESC;

-- 9. Check Monotonicity and Constraints
-- --------------------------------------
WITH best AS (
  SELECT payload_json, diagnostics_json
  FROM results
  WHERE score < 100000
  ORDER BY score ASC
  LIMIT 1
),
volumes AS (
  SELECT 
    CAST(key AS INT) as idx,
    value as vol
  FROM best,
       json_each(json_extract(payload_json, '$.schedule.volume_pct'))
),
violations AS (
  SELECT 
    COUNT(CASE WHEN v2.vol <= v1.vol THEN 1 END) as monotonicity_violations
  FROM volumes v1
  JOIN volumes v2 ON v2.idx = v1.idx + 1
)
SELECT 
  CASE 
    WHEN monotonicity_violations = 0 THEN '✅ Strict Monotonicity'
    ELSE '❌ ' || monotonicity_violations || ' violations'
  END as monotonicity_check,
  printf('%.4f', (SELECT SUM(vol) FROM volumes)) as total_sum,
  CASE 
    WHEN ABS((SELECT SUM(vol) FROM volumes) - 100.0) < 0.01 THEN '✅ Sum = 100%'
    ELSE '❌ Sum != 100%'
  END as sum_check
FROM violations;

-- 10. Performance Over Time
-- -------------------------
WITH time_series AS (
  SELECT 
    r.id,
    r.score,
    r.created_at,
    ROW_NUMBER() OVER (ORDER BY r.created_at) as eval_num,
    MIN(r.score) OVER (ORDER BY r.created_at ROWS UNBOUNDED PRECEDING) as best_so_far
  FROM results r
  WHERE r.score < 100000
)
SELECT 
  eval_num,
  printf('%.2f', score) as current_score,
  printf('%.2f', best_so_far) as best_score,
  CASE 
    WHEN score = best_so_far THEN '🎯 New Best!'
    ELSE ''
  END as milestone
FROM time_series
WHERE eval_num % 50 = 0 OR score = best_so_far
ORDER BY eval_num
LIMIT 20;
