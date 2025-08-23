import os
import sys
import json
import sqlite3
from datetime import datetime
from typing import List, Dict, Any

import pandas as pd
import streamlit as st

# Ensure project root is importable
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ui.utils.config import setup_page_config, get_icon_html
from ui.utils.constants import DB_PATH
from ui.utils.logging_buffer import get_live_trace
from martingale_lab.storage.experiments_store import ExperimentsStore


def load_experiments(db_path: str) -> List[Dict[str, Any]]:
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT id, run_id, adapter, config_json, started_at, finished_at, 
                       status, best_score, eval_count, notes, created_at
                FROM experiments 
                WHERE deleted = 0 
                ORDER BY created_at DESC
                """
            )
            experiments = []
            for row in cursor.fetchall():
                experiments.append({
                    'id': row[0],
                    'run_id': row[1],
                    'adapter': row[2],
                    'config_json': row[3],
                    'started_at': row[4],
                    'finished_at': row[5],
                    'status': row[6],
                    'best_score': row[7],
                    'eval_count': row[8],
                    'notes': row[9],
                    'created_at': row[10]
                })
            return experiments
    except Exception as e:
        st.error(f"Failed to load experiments: {e}")
        return []


def load_top_results(db_path: str, experiment_id: int, limit: int = 100) -> List[Dict[str, Any]]:
    try:
        store = ExperimentsStore(db_path)
        return store.get_top_results(experiment_id, limit=limit)
    except Exception as e:
        st.error(f"Failed to load results: {e}")
        return []


def render_experiment_summary(experiment: Dict[str, Any]):
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Best Score", f"{experiment['best_score']:.4f}")
    with col2:
        st.metric("Evaluations", f"{experiment['eval_count']:,}")
    with col3:
        st.metric("Status", experiment['status'])
    with col4:
        try:
            config = json.loads(experiment['config_json'])
            overlap_range = f"{config.get('overlap_min', 0):.1f}-{config.get('overlap_max', 0):.1f}%"
            st.metric("Overlap Range", overlap_range)
        except Exception:
            st.metric("Config", "Parse Error")


def render_results_table(results: List[Dict[str, Any]], top_n: int = 20):
    if not results:
        st.warning("No results found")
        return
    rows = []
    for i, result in enumerate(results[:top_n]):
        payload = result.get('payload', {})
        diagnostics = payload.get('diagnostics', {})
        rows.append({
            'Rank': i + 1,
            'Score': f"{result['score']:.4f}",
            'Max Need': f"{payload.get('max_need', 0):.3f}",
            'Var Need': f"{payload.get('var_need', 0):.3f}",
            'Tail': f"{payload.get('tail', 0):.3f}",
            'WCI': f"{diagnostics.get('wci', 0):.2f}",
            'Sign Flips': diagnostics.get('sign_flips', 0),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def main():
    setup_page_config()
    st.title("📊 Results")

    with st.expander("Advanced"):
        st.markdown("Live logs (last 20 events):")
        try:
            logs = get_live_trace("mlab", last_n=20)
            for log in logs[-20:]:
                ts = log.get('ts', 0)
                try:
                    time_str = datetime.fromtimestamp(ts).strftime("%H:%M:%S")
                except Exception:
                    time_str = "--:--:--"
                st.write(f"[{time_str}] {log.get('event', '')}: {log.get('msg', '')}")
        except Exception as e:
            st.info(f"No live logs: {e}")

    db_path = st.text_input("DB Path", value=DB_PATH)
    if not os.path.exists(db_path):
        st.error(f"Database file not found: {db_path}")
        return

    experiments = load_experiments(db_path)
    if not experiments:
        st.info("No experiments found.")
        return

    st.subheader("Experiment Selection")
    options = [f"#{e['id']} | {e['adapter']} | Score {e['best_score']:.4f} | Evals {e['eval_count']:,}" for e in experiments]
    choice = st.selectbox("Select Experiment", options)
    exp_id = int(choice.split('|')[0].strip()[1:])
    exp = next((e for e in experiments if e['id'] == exp_id), None)
    if not exp:
        st.error("Selected experiment not found")
        return

    st.subheader("Summary")
    render_experiment_summary(exp)

    with st.expander("Advanced Controls"):
        top_n = st.slider("Top N Results", 5, 100, 20)
        if st.button("Refresh"):
            st.rerun()

    st.subheader("Top Results")
    results = load_top_results(db_path, exp_id, limit=top_n)
    render_results_table(results, top_n)


if __name__ == "__main__":
    main()
