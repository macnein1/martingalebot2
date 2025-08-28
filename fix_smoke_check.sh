#!/usr/bin/env bash
set -euo pipefail

echo "== step 0: env =="
python3 -V || true
which python3 || true

echo "== step 1: constraints imzası kontrol =="
python3 - <<'PY'
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

echo "== step 3: temiz DB ve optimizasyon koşulu =="
rm -rf db_results/
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
  --batches 1 --batch-size 200 \
  --workers 1 --workers-mode process \
  --log-level INFO --log-eval-sample 1.0 --log-every-batch 1 \
  --db db_results/experiments.db \
  --seed 7 --notes "smoke-hc0-hc7"

echo "== step 4: sqlite kontrolleri =="
set +e
python3 - <<'PY'
import sqlite3, json, math, sys

try:
    con = sqlite3.connect("db_results/experiments.db")
    cur = con.cursor()
    row = cur.execute(
        """
        SELECT r.payload_json, r.diagnostics_json
        FROM results r JOIN (SELECT MAX(id) id FROM experiments) l ON r.experiment_id=l.id
        ORDER BY r.score ASC LIMIT 1
        """
    ).fetchone()
except Exception as e:
    print(f"[FAIL] DB open/query error: {e}")
    sys.exit(1)

if not row:
    print("[FAIL] En iyi kayıt bulunamadı.")
    sys.exit(1)

payload_json, diagnostics_json = row
payload = json.loads(payload_json)
diag = json.loads(diagnostics_json) if diagnostics_json else {}

sched = payload.get("schedule", {})
vol = list(sched.get("volume_pct", []))
mart = list(sched.get("martingale_pct", []))

# v0/v1 band & first3
v0 = float(vol[0]) if len(vol) > 0 else 0.0
v1 = float(vol[1]) if len(vol) > 1 else 0.0
v1_band_ok = (v1 >= 1.10*v0 and v1 <= 2.00*v0) if v0 > 0 else False
first3_sum = float(sum(vol[:3]))
print("-- v0/v1 band & first3 --")
print(f"v0={v0:.5f}")
print(f"v1={v1:.5f}")
print(f"v1_band={'OK' if v1_band_ok else 'VIOL'}")
print(f"first3_sum={first3_sum:.5f}")

# m-tail varyans / slope / plateau
m_tail = [(float(m)/100.0) for m in mart[2:]] if len(mart) > 2 else []
pairs = list(zip(m_tail[1:], m_tail[:-1])) if len(m_tail) > 1 else []
mu = (sum(m for m,_ in pairs)/len(pairs)) if pairs else 0.0
std_m = ((sum((m-mu)*(m-mu) for m,_ in pairs)/len(pairs))**0.5) if pairs else 0.0
slope_viol = sum(1 for m,prev in pairs if abs(m - prev) > 0.25)
plateau_max_run = diag.get("plateau_max_run")
print("-- m-tail varyans / slope / plateau --")
print(f"std_m_tail={std_m:.5f}")
print(f"slope_violations={slope_viol}")
print(f"plateau_max_run={plateau_max_run}")

# monotonicity & toplam yüzde
mono_viol = sum(1 for i in range(1, len(vol)) if vol[i] <= vol[i-1])
total_sum = float(sum(vol))
print("-- monotonicity & toplam yüzde --")
print(f"monotonicity_violations={mono_viol}")
print(f"total_sum={total_sum:.6f}")

# Q1/Q4 payları ve m2
N = len(vol)
Q = int(math.ceil(N/4.0)) if N > 0 else 0
q1_share = float(sum(vol[:Q])) if Q > 0 else 0.0
q4_share = float(sum(vol[N-Q:])) if Q > 0 else 0.0
m2 = (float(mart[2])/100.0) if len(mart) > 2 else 0.0
print("-- Q1/Q4 payları ve m2 --")
print(f"q1_share_pct={q1_share:.2f}")
print(f"q4_share_pct={q4_share:.2f}")
print(f"m2_pct={m2*100.0:.2f}")

# == step 5: hızlı verdict ==
ok = (v1_band_ok and 0.10 <= m2 <= 0.80 and q1_share <= 22.0 and q4_share >= 32.0 and mono_viol==0 and slope_viol==0 and abs(total_sum-100.0) < 0.2)
print("== step 5: hızlı verdict ==")
print("VERDICT: " + ("OK ✅" if ok else "NEEDS FIX ❌"))
print(f"(v0={v0:.5f}, v1={v1:.5f}, v1_band={'OK' if v1_band_ok else 'VIOL'}, m2={m2*100.0:.2f}%, q1={q1_share:.2f}, q4={q4_share:.2f}, mono={mono_viol}, slope_viol={slope_viol}, std_m_tail={std_m:.4f}, total={total_sum:.4f})")
PY

