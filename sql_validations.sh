#!/bin/bash
# SQL validation queries for HC1-HC7 implementation

echo "=== v0/v1/first3 Validation ==="
sqlite3 -cmd ".headers on" -cmd ".mode column" db_results/experiments.db <<'SQL'
WITH best AS (
  SELECT r.payload_json
  FROM results r JOIN (SELECT MAX(id) id FROM experiments) l ON r.experiment_id=l.id
  ORDER BY r.score ASC LIMIT 1
),
v AS (SELECT CAST(j.key AS INT) i, j.value v
      FROM best, json_each(json_extract(best.payload_json,'$.schedule.volume_pct')) j)
SELECT
  printf('%.5f',(SELECT v FROM v WHERE i=0)) AS v0,
  printf('%.5f',(SELECT v FROM v WHERE i=1)) AS v1,
  CASE WHEN (SELECT v FROM v WHERE i=1) BETWEEN 1.10*(SELECT v FROM v WHERE i=0)
                                             AND 2.00*(SELECT v FROM v WHERE i=0)
       THEN 'OK' ELSE 'VIOL' END AS v1_band,
  printf('%.5f',(SELECT SUM(v) FROM v WHERE i IN (0,1,2))) AS first3_sum;
SQL

echo ""
echo "=== m statistics and plateau/slope ==="
sqlite3 -cmd ".headers on" -cmd ".mode column" db_results/experiments.db <<'SQL'
WITH best AS (
  SELECT r.payload_json, r.diagnostics_json
  FROM results r JOIN (SELECT MAX(id) id FROM experiments) l ON r.experiment_id=l.id
  ORDER BY r.score ASC LIMIT 1
),
marts AS (
  SELECT CAST(j.key AS INT) i, j.value/100.0 AS m
  FROM best, json_each(json_extract(best.payload_json,'$.schedule.martingale_pct')) j
),
m_tail AS (SELECT i, m FROM marts WHERE i>=2),
pairs AS (SELECT a.i, a.m AS m_i, b.m AS m_prev
          FROM m_tail a JOIN m_tail b ON a.i=b.i+1)
SELECT
  printf('%.5f', (SELECT sqrt(AVG((m_i-(SELECT AVG(m_i) FROM pairs))*(m_i-(SELECT AVG(m_i) FROM pairs)))) FROM pairs)) AS std_m_tail,
  SUM(CASE WHEN ABS(m_i-m_prev) > 0.25 THEN 1 ELSE 0 END) AS slope_violations,
  (SELECT json_extract(diagnostics_json,'$.plateau_max_run') FROM best) AS plateau_max_run;
SQL

echo ""
echo "=== Strict monotonicity and sum validation ==="
sqlite3 -cmd ".headers on" -cmd ".mode column" db_results/experiments.db <<'SQL'
WITH best AS (
  SELECT r.payload_json
  FROM results r JOIN (SELECT MAX(id) id FROM experiments) l ON r.experiment_id=l.id
  ORDER BY r.score ASC LIMIT 1
),
v AS (SELECT CAST(j.key AS INT) i, j.value v
      FROM best, json_each(json_extract(best.payload_json,'$.schedule.volume_pct')) j),
pairs AS (SELECT a.i, a.v AS v_i, b.v AS v_prev FROM v a JOIN v b ON a.i=b.i+1)
SELECT
  SUM(CASE WHEN v_i <= v_prev THEN 1 ELSE 0 END) AS monotonicity_violations,
  printf('%.5f',(SELECT SUM(v) FROM v)) AS total_sum;
SQL

echo ""
echo "=== Q1/Q4 shares and m2 ==="
sqlite3 -cmd ".headers on" -cmd ".mode column" db_results/experiments.db <<'SQL'
WITH best AS (
  SELECT r.payload_json
  FROM results r JOIN (SELECT MAX(id) id FROM experiments) l ON r.experiment_id=l.id
  ORDER BY r.score ASC LIMIT 1
),
v AS (SELECT CAST(j.key AS INT) AS i, j.value AS v
      FROM best, json_each(json_extract(best.payload_json,'$.schedule.volume_pct')) j),
m AS (SELECT CAST(j.key AS INT) AS i, j.value/100.0 AS m
      FROM best, json_each(json_extract(best.payload_json,'$.schedule.martingale_pct')) j),
n AS (SELECT MAX(i)+1 AS N FROM v),
q AS (SELECT CAST(ceil(N/4.0) AS INT) AS Q FROM n)
SELECT
  printf('%.2f', (SELECT SUM(v) FROM v, q WHERE i < Q))      AS q1_share_pct,
  printf('%.2f', (SELECT SUM(v) FROM v, n, q WHERE i >= N-Q)) AS q4_share_pct,
  printf('%.2f', (SELECT m FROM m WHERE i=2)*100.0)           AS m2_pct;
SQL