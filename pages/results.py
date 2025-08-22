import streamlit as st
import sys
import os
import sqlite3
import json
import pandas as pd
import numpy as np

# Add the ui directory to the path so we can import utils
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from ui.utils.config import get_icon_html, setup_page_config

# Streamlit page configuration
st.set_page_config(
    page_title="Sonuçlar - Martingale Optimizer",
    page_icon="📊",
    layout="wide"
)

# Setup page config
setup_page_config()

st.markdown(f"""
<div class="main-header">
    {get_icon_html("chart")} Sonuçlar Sayfası
</div>
""", unsafe_allow_html=True)

DB_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'martingale_lab', 'orchestrator')  # placeholder not used


def _open_db(db_path: str = 'db_results/experiments.db'):
    return sqlite3.connect(db_path)


def _load_experiments(conn) -> pd.DataFrame:
    try:
        df = pd.read_sql_query("SELECT id, created_at, adapter, best_score, total_evals, elapsed_s FROM experiments WHERE deleted=0 ORDER BY created_at DESC", conn)
        return df
    except Exception as e:
        # Return empty DataFrame if table doesn't exist or other error
        return pd.DataFrame()


def _load_results(conn, experiment_id: int) -> pd.DataFrame:
    try:
        df = pd.read_sql_query("SELECT stable_id, score, params_json, schedule_json, risk_json, penalties_json, created_at FROM results WHERE experiment_id=? ORDER BY score ASC", conn, params=(experiment_id,))
        # Parse JSON columns
        def parse_json(col):
            try:
                return json.loads(col)
            except Exception:
                return {}
        if not df.empty:
            df['params'] = df['params_json'].apply(parse_json)
            df['schedule'] = df['schedule_json'].apply(parse_json)
            df['risk'] = df['risk_json'].apply(parse_json)
            df['penalties'] = df['penalties_json'].apply(lambda x: parse_json(x) if pd.notnull(x) else {})
        return df
    except Exception as e:
        # Return empty DataFrame if table doesn't exist or other error
        return pd.DataFrame()


def _format_bullets(schedule: dict) -> list:
    indent = schedule.get('indent_pct', [])
    volume = schedule.get('volume_pct', [])
    mart = schedule.get('martingale_pct', [])
    lines = []
    for i in range(len(indent)):
        if i == 0:
            lines.append(f"{i+1}. Emir: Indent %{indent[i]:.2f}  Volume %{volume[i]:.2f}  (no martingale, first order)")
        else:
            lines.append(f"{i+1}. Emir: Indent %{indent[i]:.2f}  Volume %{volume[i]:.2f}  (Martingale %{mart[i]:.2f})")
    return lines


def _sanity(schedule: dict, risk: dict) -> list:
    msgs = []
    m = len(schedule.get('indent_pct', []))
    if not (m and m == len(schedule.get('volume_pct', [])) == len(schedule.get('martingale_pct', [])) == len(schedule.get('needpct', []))):
        msgs.append('Len mismatch')
    if abs(sum(schedule.get('volume_pct', [])) - 100.0) > 1e-6:
        msgs.append('Sum(volume_pct) != 100')
    if m and schedule.get('martingale_pct', [0])[0] != 0:
        msgs.append('martingale_pct[0] != 0')
    if 'max_need' in risk and 'needpct' in schedule and schedule['needpct']:
        if abs(risk['max_need'] - max(schedule['needpct'])) > 1e-6:
            msgs.append('max_need mismatch')
    return msgs


# Default DB path - try to find it in common locations
default_db_path = "db_results/experiments.db"
possible_paths = [
    "db_results/experiments.db",
    "../db_results/experiments.db", 
    "../../db_results/experiments.db",
    "martingale_lab/orchestrator/db_results/experiments.db",
    "../martingale_lab/orchestrator/db_results/experiments.db"
]

# Try to find existing DB
found_db = None
for path in possible_paths:
    if os.path.exists(path):
        found_db = path
        break

if found_db:
    default_db_path = found_db

db_path = st.text_input("DB path", value=default_db_path)

if not os.path.exists(db_path):
    st.error(f"DB dosyası bulunamadı: {db_path}")
    st.info("Önce bir optimizasyon çalıştırın veya doğru DB yolunu girin.")
    st.stop()

try:
    conn = _open_db(db_path)
except Exception as e:
    st.error(f"DB açılamadı: {e}")
    st.stop()

with conn:
    exps = _load_experiments(conn)

if exps.empty:
    st.info("Kayıt bulunamadı. Önce bir optimizasyon çalıştırın.")
    st.stop()

# Experiments list
st.markdown("### Kayıtlar")
exp_options = [f"#{row.id} | {row.adapter} | best J={row.best_score:.4f} | evals={row.total_evals} | {row.created_at}" for _, row in exps.iterrows()]
sel = st.selectbox("Experiment seç", options=exp_options, index=0)
exp_id = int(sel.split('|')[0].strip()[1:])

with conn:
    results_df = _load_results(conn, exp_id)

if results_df.empty:
    st.info("Sonuç bulunamadı.")
    st.stop()

# Summary panel
st.markdown("### Özet")
best_row = results_df.iloc[0]
risk = best_row['risk']
st.json({
    "adapter": exps.loc[exps.id == exp_id, 'adapter'].values[0],
    "best_score": float(best_row['score']),
    "max_need": float(risk.get('max_need', 0.0)),
    "var_need": float(risk.get('var_need', 0.0)),
    "tail": float(risk.get('tail', 0.0)),
})

# Simple Pareto-esque scatter (max_need vs var_need, color=tail)
st.markdown("### Pareto Keşif (Basit Scatter)")
try:
    chart_df = pd.DataFrame({
        'max_need': results_df['risk'].apply(lambda r: r.get('max_need', np.nan)),
        'var_need': results_df['risk'].apply(lambda r: r.get('var_need', np.nan)),
        'tail': results_df['risk'].apply(lambda r: r.get('tail', np.nan)),
        'score': results_df['score'],
    })
    st.scatter_chart(chart_df, x='max_need', y='var_need', color='tail', size='score')
except Exception as e:
    st.warning(f"Grafik oluşturulamadı: {e}")

# Top-N table with small schedule preview
st.markdown("### Top-N Adaylar")
top_n = st.slider("Kaç aday?", 5, 50, 10)
tbl = []
for i in range(min(top_n, len(results_df))):
    row = results_df.iloc[i]
    sched = row['schedule']
    risk = row['risk']
    bullets = _format_bullets(sched)[:3]  # show first 3 lines
    sanity = _sanity(sched, risk)
    tbl.append({
        'rank': i + 1,
        'score': f"{row['score']:.4f}",
        'max_need': f"{risk.get('max_need', 0.0):.4f}",
        'var_need': f"{risk.get('var_need', 0.0):.4f}",
        'tail': f"{risk.get('tail', 0.0):.4f}",
        'bullets': '\n'.join(bullets),
        'sanity': 'OK' if not sanity else ', '.join(sanity)
    })
st.dataframe(pd.DataFrame(tbl), use_container_width=True)

# Export buttons for selected best row
st.markdown("### Dışa Aktarım")
col1, col2, col3 = st.columns(3)
best = best_row
sched = best['schedule']

# CSV export
with col1:
    if st.button("CSV İndir"):
        csv_rows = [
            {
                'order': i + 1,
                'indent_pct': sched['indent_pct'][i],
                'volume_pct': sched['volume_pct'][i],
                'martingale_pct': sched['martingale_pct'][i],
                'needpct': sched['needpct'][i],
                'order_price': sched['order_prices'][i],
                'price_step_pct': sched['price_step_pct'][i],
            }
            for i in range(len(sched.get('indent_pct', [])))
        ]
        csv = pd.DataFrame(csv_rows).to_csv(index=False).encode('utf-8')
        st.download_button("CSV", data=csv, file_name=f"result_{best['stable_id']}.csv", mime="text/csv")

# JSON export
with col2:
    if st.button("JSON İndir"):
        payload = {
            'stable_id': best['stable_id'],
            'score': float(best['score']),
            'params': best['params'],
            'schedule': sched,
            'risk': best['risk'],
            'penalties': best['penalties'],
        }
        js = json.dumps(payload, ensure_ascii=False, separators=(',', ':')).encode('utf-8')
        st.download_button("JSON", data=js, file_name=f"result_{best['stable_id']}.json", mime="application/json")

with col3:
    st.info("HTML/PNG export ileride eklenecek.")
