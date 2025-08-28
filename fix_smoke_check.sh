#!/usr/bin/env bash
set -euo pipefail

echo "== step 0: env =="
python -V || true
which python || true

echo "== step 1: constraints imzası kontrol =="
python - <<'PY'
import inspect, sys
from importlib import import_module
C = import_module("martingale_lab.core.constraints")
sig = str(inspect.signature(C.enforce_schedule_shape_fixed))
print("enforce_schedule_shape_fixed:", sig)
needed = ["second_upper_c2", "m_head", "tau_scale", "q1_cap", "tail_floor", "use_hc0_bootstrap", "use_head_budget", "head_budget_pct"]
missing = [k for k in needed if k not in sig]
if missing:
    print("\n[FAIL] Bu parametreler imzada yok:", ", ".join(missing))
    print("Çözüm: doğru branch/tag'e geç, paketi editable kur ve tekrar dene:")
    print("  git fetch --all && git checkout <hc0-7 branch/tag> && git pull")
    print("  pip uninstall -y martingale_lab; pip install -e .")
    sys.exit(2)
else:
    print("[OK] Yeni API imzada görülüyor.")
PY

echo "== step 2: çağrı noktası (forward) kontrol =="
HITS=$(grep -RIn --line-number "enforce_schedule_shape_fixed" martingale_lab || true)
if [ -z "$HITS" ]; then
  echo "[FAIL] Kodda enforce_schedule_shape_fixed çağrısı bulunamadı."
  exit 2
fi
echo "$HITS" | sed -n '1,5p'
# basit içerik kontrolü
FORWARD_OK=1
for k in second_upper_c2 m_head m_tail tau_scale slope_cap q1_cap tail_floor use_hc0_bootstrap use_head_budget head_budget_pct m2_min m2_max m_min; do
  if ! grep -RIn "$k=" martingale_lab | grep -q "enforce_schedule_shape_fixed"; then
    echo "[WARN] Çağrıda '$k=' görünmüyor."
    FORWARD_OK=0
  fi
done
if [ $FORWARD_OK -eq 0 ]; then
  echo "[HINT] Yukarıdaki WARN alanlarını enforce_schedule_shape_fixed çağrısına forward et."
fi

echo "== step 3: temiz DB ve optimizasyon koşusu =="
rm -rf db_results/
python -m martingale_lab.cli.optimize \
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
  --batches 1 --batch-size 200 \
  --workers 1 --workers-mode process \
  --log-level INFO --log-eval-sample 1.0 --log-every-batch 1 \
  --db db_results/experiments.db \
  --seed 7 --notes "smoke-hc0-hc7"

echo "== step 4: sqlite kontrolleri =="
set +e

echo "-- v0/v1 band & first3 --"
sqlite3 -cmd ".headers on" -cmd ".mode column" db_results/experiments.db "
WITH best AS (
  SELECT r.payload_json
  FROM results r JOIN (SELECT MAX(id) id FROM experiments) l ON r.experiment_id=l.id
  ORDER BY r.score ASC LIMIT 1
),
v AS ( SELECT CAST(j.key AS INT) i, j.value v
       FROM best, json_each(best.payload_json,'$.schedule.volume_pct') j )
SELECT
  printf('%.5f',(SELECT v FROM v WHERE i=0)) AS v0,
  printf('%.5f',(SELECT v FROM v WHERE i=1)) AS v1,
  CASE WHEN (SELECT v FROM v WHERE i=1) BETWEEN 1.10*(SELECT v FROM v WHERE i=0)
                                             AND 2.00*(SELECT v FROM v WHERE i=0)
       THEN 'OK' ELSE 'VIOL' END AS v1_band,
  printf('%.5f',(SELECT SUM(v) FROM v WHERE i IN (0,1,2))) AS first3_sum;
"

echo "-- m-tail varyans / slope / plateau --"
sqlite3 -cmd ".headers on" -cmd ".mode column" db_results/experiments.db "
WITH best AS (
  SELECT r.payload_json, r.diagnostics_json
  FROM results r JOIN (SELECT MAX(id) id FROM experiments) l ON r.experiment_id=l.id
  ORDER BY r.score ASC LIMIT 1
),
m_tail AS (
  SELECT CAST(j.key AS INT) AS i, j.value/100.0 AS m
  FROM best, json_each(best.payload_json,'$.schedule.martingale_pct') AS j
  WHERE CAST(j.key AS INT) >= 2
),
pairs AS ( SELECT a.i, a.m AS m_i, b.m AS m_prev
           FROM m_tail a JOIN m_tail b ON a.i = b.i + 1 ),
mean_m AS (SELECT AVG(m_i) AS mu FROM pairs)
SELECT
  printf('%.5f', sqrt(AVG((m_i - mu)*(m_i - mu)))) AS std_m_tail,
  SUM(CASE WHEN ABS(m_i - m_prev) > 0.25 THEN 1 ELSE 0 END) AS slope_violations,
  json_extract(best.diagnostics_json,'$.plateau_max_run') AS plateau_max_run
FROM pairs, mean_m, best;
"

echo "-- monotonicity & toplam yüzde --"
sqlite3 -cmd ".headers on" -cmd ".mode column" db_results/experiments.db "
WITH best AS (
  SELECT r.payload_json
  FROM results r JOIN (SELECT MAX(id) id FROM experiments) l ON r.experiment_id=l.id
  ORDER BY r.score ASC LIMIT 1
),
v AS ( SELECT CAST(j.key AS INT) AS i, j.value AS v
       FROM best, json_each(best.payload_json,'$.schedule.volume_pct') AS j ),
pairs AS ( SELECT a.i, a.v AS v_i, b.v AS v_prev
           FROM v a JOIN v b ON a.i = b.i + 1 )
SELECT
  SUM(CASE WHEN v_i <= v_prev THEN 1 ELSE 0 END) AS monotonicity_violations,
  printf('%.6f',(SELECT SUM(v) FROM v)) AS total_sum
FROM pairs;
"

echo "-- Q1/Q4 payları ve m2 --"
sqlite3 -cmd ".headers on" -cmd ".mode column" db_results/experiments.db "
WITH best AS (
  SELECT r.payload_json
  FROM results r JOIN (SELECT MAX(id) id FROM experiments) l ON r.experiment_id=l.id
  ORDER BY r.score ASC LIMIT 1
),
v AS ( SELECT CAST(j.key AS INT) AS i, j.value AS v
       FROM best, json_each(best.payload_json,'$.schedule.volume_pct') AS j ),
m AS ( SELECT CAST(j.key AS INT) AS i, j.value/100.0 AS m
       FROM best, json_each(best.payload_json,'$.schedule.martingale_pct') AS j ),
n AS (SELECT MAX(i)+1 AS N FROM v),
q AS (SELECT CAST(ceil(N/4.0) AS INT) AS Q FROM n)
SELECT
  printf('%.2f', (SELECT SUM(v) FROM v, q WHERE i < Q))      AS q1_share_pct,
  printf('%.2f', (SELECT SUM(v) FROM v, n, q WHERE i >= N-Q)) AS q4_share_pct,
  printf('%.2f', (SELECT m FROM m WHERE i=2)*100.0)           AS m2_pct;
"

echo "== step 5: hızlı verdict =="
# küçük bir heuristik: v1 band OK, m2_pct 10-80, q1<=22, q4>=32, monotonicity=0, slope_violations=0
verdict_py=$(python - <<'PY'
import json, subprocess, re, sys

def q(sql):
    out = subprocess.check_output(["sqlite3", "db_results/experiments.db", sql], text=True).strip()
    return out

# v1 band & first3
vblock = q("""
WITH best AS ( SELECT r.payload_json FROM results r JOIN (SELECT MAX(id) id FROM experiments) l ON r.experiment_id=l.id
               ORDER BY r.score ASC LIMIT 1 ),
v AS ( SELECT CAST(j.key AS INT) i, j.value v
       FROM best, json_each(best.payload_json,'$.schedule.volume_pct') j )
SELECT (SELECT v FROM v WHERE i=0), (SELECT v FROM v WHERE i=1),
       CASE WHEN (SELECT v FROM v WHERE i=1) BETWEEN 1.10*(SELECT v FROM v WHERE i=0)
                                            AND 2.00*(SELECT v FROM v WHERE i=0) THEN 'OK' ELSE 'VIOL' END,
       (SELECT SUM(v) FROM v WHERE i IN (0,1,2));
""")
v0, v1, v1band, first3 = vblock.split('|')
v0=float(v0); v1=float(v1); first3=float(first3)

# tail stats
tblock = q("""
WITH best AS ( SELECT r.payload_json, r.diagnostics_json FROM results r JOIN (SELECT MAX(id) id FROM experiments) l ON r.experiment_id=l.id
               ORDER BY r.score ASC LIMIT 1 ),
m_tail AS ( SELECT CAST(j.key AS INT) AS i, j.value/100.0 AS m
            FROM best, json_each(best.payload_json,'$.schedule.martingale_pct') AS j WHERE CAST(j.key AS INT) >= 2 ),
pairs AS ( SELECT a.i, a.m AS m_i, b.m AS m_prev FROM m_tail a JOIN m_tail b ON a.i = b.i + 1 ),
mean_m AS (SELECT AVG(m_i) AS mu FROM pairs)
SELECT (SELECT printf('%.5f', sqrt(AVG((m_i - mu)*(m_i - mu)))) FROM pairs, mean_m),
       (SELECT SUM(CASE WHEN ABS(m_i - m_prev) > 0.25 THEN 1 ELSE 0 END) FROM pairs),
       (SELECT json_extract(best.diagnostics_json,'$.plateau_max_run') FROM best);
""")
std_m, slope_v, plateau = tblock.split('|')
std_m=float(std_m); slope_v=int(slope_v or 0)

# q1 q4 m2
qblock = q("""
WITH best AS ( SELECT r.payload_json FROM results r JOIN (SELECT MAX(id) id FROM experiments) l ON r.experiment_id=l.id
               ORDER BY r.score ASC LIMIT 1 ),
v AS ( SELECT CAST(j.key AS INT) AS i, j.value AS v
       FROM best, json_each(best.payload_json,'$.schedule.volume_pct') AS j ),
m AS ( SELECT CAST(j.key AS INT) AS i, j.value/100.0 AS m
       FROM best, json_each(best.payload_json,'$.schedule.martingale_pct') AS j ),
n AS (SELECT MAX(i)+1 AS N FROM v),
q AS (SELECT CAST(ceil(N/4.0) AS INT) AS Q FROM n)
SELECT (SELECT SUM(v) FROM v, q WHERE i < Q),
       (SELECT SUM(v) FROM v, n, q WHERE i >= N-Q),
       (SELECT m FROM m WHERE i=2);
""")
q1, q4, m2 = qblock.split('|')
q1=float(q1); q4=float(q4); m2=float(m2)

# monotonicity & total
mblock = q("""
WITH best AS ( SELECT r.payload_json FROM results r JOIN (SELECT MAX(id) id FROM experiments) l ON r.experiment_id=l.id
               ORDER BY r.score ASC LIMIT 1 ),
v AS ( SELECT CAST(j.key AS INT) AS i, j.value AS v
       FROM best, json_each(best.payload_json,'$.schedule.volume_pct') AS j ),
pairs AS ( SELECT a.i, a.v AS v_i, b.v AS v_prev FROM v a JOIN v b ON a.i = b.i + 1 )
SELECT (SELECT SUM(CASE WHEN v_i <= v_prev THEN 1 ELSE 0 END) FROM pairs),
       (SELECT SUM(v) FROM v);
""")
mono, tot = mblock.split('|')
mono=int(mono or 0); tot=float(tot)

ok = (v1band=="OK" and 0.10 <= m2 <= 0.80 and q1 <= 22.0 and q4 >= 32.0 and mono==0 and slope_v==0 and abs(tot-100.0) < 0.2)
print("VERDICT: " + ("OK ✅" if ok else "NEEDS FIX ❌"))
print(f"(v0={v0:.5f}, v1={v1:.5f}, v1_band={v1band}, m2={m2*100:.2f}%, q1={q1:.2f}, q4={q4:.2f}, mono={mono}, slope_viol={slope_v}, std_m_tail={std_m:.4f}, total={tot:.4f})")
PY
)
echo "$verdict_py"

