import streamlit as st
import sys
import os
import sqlite3
import json
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime
import traceback

# Add the ui directory to the path so we can import utils
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from ui.utils.config import get_icon_html, setup_page_config
from ui.utils.constants import DB_PATH
from martingale_lab.storage.experiments_store import ExperimentsStore
from martingale_lab.utils.structured_logging import get_structured_logger, EventNames
from ui.utils.optimization_bridge import get_optimization_bridge
from ui.utils.logging_buffer import get_live_trace

# Initialize structured logger for results page
logger = get_structured_logger("mlab.results")

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


def load_experiments_data(db_path: str) -> List[Dict[str, Any]]:
    """Load experiments data with logging"""
    try:
        logger.info(
            EventNames.UI_RESULTS_LOAD,
            f"Loading experiments from {db_path}",
            db_path=db_path
        )
        
        store = ExperimentsStore(db_path)
        
        # Get experiments using raw SQL for now (can be enhanced later)
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, run_id, adapter, config_json, started_at, finished_at, 
                       status, best_score, eval_count, notes, created_at
                FROM experiments 
                WHERE deleted = 0 
                ORDER BY created_at DESC
            """)
            
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
        
        logger.info(
            EventNames.UI_RESULTS_LOAD,
            f"Loaded {len(experiments)} experiments",
            count=len(experiments)
        )
        
        return experiments
        
    except Exception as e:
        logger.error(
            EventNames.UI_RESULTS_LOAD,
            f"Failed to load experiments: {str(e)}",
            error=str(e),
            traceback=traceback.format_exc()
        )
        return []


def load_results_data(db_path: str, experiment_id: int, limit: int = 100) -> List[Dict[str, Any]]:
    """Load results data for an experiment with logging"""
    try:
        logger.info(
            EventNames.UI_RESULTS_LOAD,
            f"Loading results for experiment {experiment_id}",
            experiment_id=experiment_id,
            limit=limit
        )
        
        store = ExperimentsStore(db_path)
        results = store.get_top_results(experiment_id, limit=limit)
        
        logger.info(
            EventNames.UI_RESULTS_LOAD,
            f"Loaded {len(results)} results for experiment {experiment_id}",
            experiment_id=experiment_id,
            count=len(results)
        )
        
        return results
        
    except Exception as e:
        logger.error(
            EventNames.UI_RESULTS_LOAD,
            f"Failed to load results for experiment {experiment_id}: {str(e)}",
            experiment_id=experiment_id,
            error=str(e),
            traceback=traceback.format_exc()
        )
        return []


def create_needpct_sparkline(needpct: List[float], width: int = 20) -> str:
    """Create ASCII sparkline for NeedPct values"""
    if not needpct or len(needpct) < 2:
        return "─" * width
    
    min_val, max_val = min(needpct), max(needpct)
    if max_val - min_val < 1e-6:
        return "─" * width
    
    # Map to characters
    chars = "▁▂▃▄▅▆▇█"
    sparkline = ""
    
    for i, val in enumerate(needpct[:width]):
        normalized = (val - min_val) / (max_val - min_val)
        char_idx = min(len(chars) - 1, int(normalized * (len(chars) - 1)))
        sparkline += chars[char_idx]
    
    return sparkline


def create_sanity_badges(sanity: Dict[str, Any]) -> str:
    """Create sanity check badges"""
    badges = []
    
    if sanity.get('max_need_mismatch', False):
        badges.append("🔴 MaxNeed")
    if sanity.get('collapse_indents', False):
        badges.append("🟡 Collapse")
    if sanity.get('tail_overflow', False):
        badges.append("🟠 TailOvf")
    if sanity.get('error', False):
        badges.append("❌ Error")
    
    return " ".join(badges) if badges else "✅ OK"


def display_live_trace_panel():
    """Display live trace panel with recent logs"""
    st.sidebar.markdown("### 🔴 Live Trace")
    
    # Get optimization status
    bridge = get_optimization_bridge()
    status = bridge.get_status()
    
    if status['is_running']:
        st.sidebar.success("🟢 Optimization Running")
        st.sidebar.write(f"Run ID: `{status['run_id']}`")
    else:
        st.sidebar.info("⚪ No Active Optimization")
    
    # Event filter
    event_filter = st.sidebar.selectbox(
        "Filter Events",
        ["All", "ORCH", "EVAL", "DB", "UI"],
        index=0
    )
    
    filter_value = None if event_filter == "All" else event_filter
    
    # Get live logs
    try:
        logs = get_live_trace("mlab", event_filter=filter_value, last_n=20)
        
        if logs:
            st.sidebar.markdown("**Recent Events:**")
            for log in logs[-10:]:  # Show last 10
                event = log.get('event', 'UNKNOWN')
                msg = log.get('msg', '')
                ts = log.get('ts', 0)
                
                # Format timestamp
                try:
                    dt = datetime.fromtimestamp(ts)
                    time_str = dt.strftime("%H:%M:%S")
                except:
                    time_str = "??:??:??"
                
                # Color code by event type
                if event.startswith('ORCH'):
                    color = "🔵"
                elif event.startswith('EVAL'):
                    color = "🟢"
                elif event.startswith('DB'):
                    color = "🟡"
                elif event.startswith('UI'):
                    color = "🟣"
                else:
                    color = "⚪"
                
                st.sidebar.write(f"{color} `{time_str}` **{event}**")
                if msg and len(msg) < 50:
                    st.sidebar.write(f"   _{msg}_")
        else:
            st.sidebar.write("No recent events")
            
    except Exception as e:
        st.sidebar.error(f"Error loading live trace: {e}")


def display_experiment_summary(experiment: Dict[str, Any]):
    """Display experiment summary card"""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Best Score", f"{experiment['best_score']:.4f}")
    
    with col2:
        st.metric("Evaluations", f"{experiment['eval_count']:,}")
    
    with col3:
        status = experiment['status']
        status_color = {
            'COMPLETED': '🟢',
            'RUNNING': '🟡', 
            'FAILED': '🔴',
            'PENDING': '⚪'
        }.get(status, '❓')
        st.metric("Status", f"{status_color} {status}")
    
    with col4:
        try:
            config = json.loads(experiment['config_json'])
            overlap_range = f"{config.get('overlap_min', 0):.1f}-{config.get('overlap_max', 0):.1f}%"
            st.metric("Overlap Range", overlap_range)
        except:
            st.metric("Config", "Parse Error")


def display_results_table(results: List[Dict[str, Any]], top_n: int = 20):
    """Display results table with enhanced formatting"""
    if not results:
        st.warning("No results found")
        return
    
    # Prepare table data
    table_data = []
    
    for i, result in enumerate(results[:top_n]):
        payload = result.get('payload', {})
        schedule = payload.get('schedule', {})
        sanity = payload.get('sanity', {})
        diagnostics = payload.get('diagnostics', {})
        
        needpct = schedule.get('needpct', [])
        sparkline = create_needpct_sparkline(needpct)
        sanity_badges = create_sanity_badges(sanity)
        
        table_data.append({
            'Rank': i + 1,
            'Score': f"{result['score']:.4f}",
            'Max Need': f"{payload.get('max_need', 0):.3f}",
            'Var Need': f"{payload.get('var_need', 0):.3f}",
            'Tail': f"{payload.get('tail', 0):.3f}",
            'WCI': f"{diagnostics.get('wci', 0):.2f}",
            'Sign Flips': diagnostics.get('sign_flips', 0),
            'NeedPct ▼': sparkline,
            'Sanity': sanity_badges
        })
    
    # Display table
    df = pd.DataFrame(table_data)
    st.dataframe(df, use_container_width=True, hide_index=True)


def display_result_detail(result: Dict[str, Any], rank: int):
    """Display detailed view of a single result"""
    payload = result.get('payload', {})
    schedule = payload.get('schedule', {})
    sanity = payload.get('sanity', {})
    diagnostics = payload.get('diagnostics', {})
    penalties = payload.get('penalties', {})
    
    st.markdown(f"### 🎯 Rank #{rank} Detail")
    
    # Metrics row
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Score", f"{result['score']:.4f}")
    with col2:
        st.metric("Max Need", f"{payload.get('max_need', 0):.3f}%")
    with col3:
        st.metric("Var Need", f"{payload.get('var_need', 0):.3f}")
    with col4:
        st.metric("Tail", f"{payload.get('tail', 0):.3f}")
    with col5:
        st.metric("WCI", f"{diagnostics.get('wci', 0):.2f}")
    
    # Bullets format
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("**📋 Order Schedule:**")
        
        indent_pct = schedule.get('indent_pct', [])
        volume_pct = schedule.get('volume_pct', [])
        martingale_pct = schedule.get('martingale_pct', [])
        needpct = schedule.get('needpct', [])
        
        bullets = []
        for i in range(len(indent_pct)):
            need_str = f"{needpct[i]:.2f}%" if i < len(needpct) else "N/A"
            if i == 0:
                bullets.append(
                    f"**{i+1}.** Indent {indent_pct[i]:.2f}% • "
                    f"Volume {volume_pct[i]:.2f}% • "
                    f"Need {need_str} • "
                    f"(First order, no martingale)"
                )
            else:
                mart_str = f"{martingale_pct[i]:.1f}%" if i < len(martingale_pct) else "0%"
                bullets.append(
                    f"**{i+1}.** Indent {indent_pct[i]:.2f}% • "
                    f"Volume {volume_pct[i]:.2f}% • "
                    f"Need {need_str} • "
                    f"Martingale {mart_str}"
                )
        
        for bullet in bullets:
            st.markdown(bullet)
    
    with col2:
        st.markdown("**🔍 Diagnostics:**")
        st.write(f"• Gini: {diagnostics.get('gini', 0):.3f}")
        st.write(f"• Entropy: {diagnostics.get('entropy', 0):.3f}")
        st.write(f"• Sign Flips: {diagnostics.get('sign_flips', 0)}")
        
        st.markdown("**⚖️ Penalties:**")
        for key, value in penalties.items():
            if value > 1e-6:
                st.write(f"• {key}: {value:.4f}")
        
        st.markdown("**✅ Sanity:**")
        sanity_badges = create_sanity_badges(sanity)
        st.write(sanity_badges)


def export_data(result: Dict[str, Any], format_type: str):
    """Export result data in specified format with logging"""
    try:
        logger.info(
            EventNames.UI_EXPORT,
            f"Exporting data in {format_type} format",
            format_type=format_type
        )
        
        payload = result.get('payload', {})
        schedule = payload.get('schedule', {})
        
        if format_type == "CSV":
            # Create CSV data
            csv_rows = []
            indent_pct = schedule.get('indent_pct', [])
            volume_pct = schedule.get('volume_pct', [])
            martingale_pct = schedule.get('martingale_pct', [])
            needpct = schedule.get('needpct', [])
            order_prices = schedule.get('order_prices', [])
            price_step_pct = schedule.get('price_step_pct', [])
            
            for i in range(len(indent_pct)):
                csv_rows.append({
                    'order': i + 1,
                    'indent_pct': indent_pct[i] if i < len(indent_pct) else 0,
                    'volume_pct': volume_pct[i] if i < len(volume_pct) else 0,
                    'martingale_pct': martingale_pct[i] if i < len(martingale_pct) else 0,
                    'needpct': needpct[i] if i < len(needpct) else 0,
                    'order_price': order_prices[i+1] if i+1 < len(order_prices) else 0,
                    'price_step_pct': price_step_pct[i] if i < len(price_step_pct) else 0,
                })
            
            csv_data = pd.DataFrame(csv_rows).to_csv(index=False).encode('utf-8')
            return csv_data, f"result_{result['id']}.csv", "text/csv"
            
        elif format_type == "JSON":
            # Create JSON data
            json_data = json.dumps(payload, ensure_ascii=False, indent=2).encode('utf-8')
            return json_data, f"result_{result['id']}.json", "application/json"
            
    except Exception as e:
        logger.error(
            EventNames.UI_EXPORT,
            f"Export failed: {str(e)}",
            format_type=format_type,
            error=str(e)
        )
        st.error(f"Export failed: {e}")
        return None, None, None


# Main UI Logic
def main():
    # Display live trace panel in sidebar
    display_live_trace_panel()
    
    # Database path input
    db_path = st.text_input("DB Path", value=DB_PATH)
    
    if not os.path.exists(db_path):
        st.error(f"Database file not found: {db_path}")
        st.info("Run an optimization first or enter the correct DB path.")
        return
    
    # Load experiments
    experiments = load_experiments_data(db_path)
    
    if not experiments:
        st.info("No experiments found. Run an optimization first.")
        return
    
    # Experiment selection
    st.markdown("### 📋 Experiment Selection")
    
    exp_options = []
    for exp in experiments:
        option = (
            f"#{exp['id']} | {exp['adapter']} | "
            f"Score: {exp['best_score']:.4f} | "
            f"Evals: {exp['eval_count']:,} | "
            f"{exp['created_at']}"
        )
        exp_options.append(option)
    
    selected_option = st.selectbox("Select Experiment", options=exp_options, index=0)
    exp_id = int(selected_option.split('|')[0].strip()[1:])
    
    # Find selected experiment
    selected_exp = next((exp for exp in experiments if exp['id'] == exp_id), None)
    
    if not selected_exp:
        st.error("Selected experiment not found")
        return
    
    # Display experiment summary
    st.markdown("### 📊 Experiment Summary")
    display_experiment_summary(selected_exp)
    
    # Load and display results
    st.markdown("### 🏆 Results")
    
    # Results controls
    col1, col2 = st.columns([1, 3])
    
    with col1:
        top_n = st.slider("Top N Results", 5, 100, 20)
    
    with col2:
        if st.button("🔄 Refresh Data"):
            st.rerun()
    
    # Load results data
    results = load_results_data(db_path, exp_id, limit=top_n)
    
    if not results:
        st.warning("No results found for this experiment")
        return
    
    # Display results table
    display_results_table(results, top_n)
    
    # Detailed view
    st.markdown("### 🔍 Detailed View")
    
    detail_rank = st.selectbox(
        "Select rank for detailed view",
        options=list(range(1, min(len(results) + 1, top_n + 1))),
        index=0
    )
    
    if detail_rank <= len(results):
        selected_result = results[detail_rank - 1]
        display_result_detail(selected_result, detail_rank)
        
        # Export buttons
        st.markdown("### 📤 Export")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📄 Export CSV"):
                data, filename, mime = export_data(selected_result, "CSV")
                if data:
                    st.download_button(
                        "Download CSV",
                        data=data,
                        file_name=filename,
                        mime=mime
                    )
        
        with col2:
            if st.button("📋 Export JSON"):
                data, filename, mime = export_data(selected_result, "JSON")
                if data:
                    st.download_button(
                        "Download JSON",
                        data=data,
                        file_name=filename,
                        mime=mime
                    )
        
        with col3:
            st.info("HTML/PNG export coming soon")


if __name__ == "__main__":
    main()
