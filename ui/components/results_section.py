"""
Enhanced Results Section UI - Production Quality DCA/Martingale Results Display
Shows Top-N candidates with compact summary rows, detail cards, and advanced filtering.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Any, Optional
import json
import sqlite3

from ui.utils.config import (
    compute_exit_speed, compute_wave_score, build_bullets, 
    sanity_badges, create_needpct_sparkline
)
from martingale_lab.storage.experiments_store import ExperimentsStore
from ui.utils.logging_buffer import get_live_trace


def display_experiment_summary(experiment_id: int, store: ExperimentsStore):
    """Display experiment summary with refresh button."""
    st.subheader("📊 Experiment Summary")
    
    summary = store.get_experiment_summary(experiment_id)
    if not summary:
        st.error("Experiment not found!")
        return
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Best Score", f"{summary.get('best_score', 0):.3f}")
    with col2:
        total_evals = summary.get('total_evals', 0) or summary.get('eval_count', 0)
        st.metric("Total Evals", f"{total_evals:,}")
    with col3:
        st.metric("Status", summary.get('status', 'UNKNOWN'))
    with col4:
        config = json.loads(summary.get('config_json', '{}')) if summary.get('config_json') else {}
        overlap_range = f"{config.get('overlap_min', 0):.1f}-{config.get('overlap_max', 0):.1f}%"
        st.metric("Overlap Range", overlap_range)
    with col5:
        if st.button("🔄 Refresh", type="secondary"):
            st.rerun()


def display_filters():
    """Display filtering controls."""
    st.subheader("🔍 Filters")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        max_score = st.number_input("Max Score", min_value=0.0, value=100.0, step=1.0, help="Maximum score threshold")
    
    with col2:
        overlap_range = st.slider("Overlap % Range", 5.0, 50.0, (10.0, 30.0), step=0.5)
    
    with col3:
        orders_range = st.slider("Orders Range", 2, 30, (5, 15))
    
    with col4:
        wave_only = st.checkbox("Yalnızca Yelpaze+ (≥0.6)", help="Show only wave pattern scores ≥ 0.6")
    
    return {
        "max_score": max_score,
        "overlap_min": overlap_range[0],
        "overlap_max": overlap_range[1],
        "orders_min": orders_range[0],
        "orders_max": orders_range[1],
        "wave_only": wave_only
    }


def create_mini_chart(data: List[float], title: str, chart_type: str = "line", height: int = 120) -> go.Figure:
    """Create a mini chart for detail cards."""
    fig = go.Figure()
    
    if chart_type == "line":
        fig.add_trace(go.Scatter(
            y=data,
            mode='lines+markers',
            line=dict(color='#1f77b4', width=2),
            marker=dict(size=4)
        ))
    elif chart_type == "bar":
        fig.add_trace(go.Bar(
            y=data,
            marker_color='lightblue'
        ))
    
    fig.update_layout(
        title=title,
        height=height,
        margin=dict(l=10, r=10, t=30, b=10),
        showlegend=False,
        template="plotly_white"
    )
    
    return fig


def display_result_detail(result: Dict[str, Any]):
    """Display detailed view of a single result."""
    st.subheader(f"📋 Result Detail - Rank {result.get('rank', 'N/A')}")
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Score", f"{result.get('score', 0):.3f}")
    with col2:
        st.metric("Exit Speed", f"{result.get('exit_speed', 0):.3f}")
    with col3:
        st.metric("Wave Score", f"{result.get('wave_score', 0):.3f}")
    with col4:
        st.metric("Orders", result.get('orders', 0))
    
    # Bullets and charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Order Details")
        bullets = build_bullets(
            result.get('indent_pct', []),
            result.get('volume_pct', []),
            result.get('martingale_pct', []),
            result.get('needpct', [])
        )
        for bullet in bullets:
            st.markdown(f"• {bullet}")
    
    with col2:
        st.subheader("📈 Mini Charts")
        
        # NeedPct chart
        needpct = result.get('needpct', [])
        if needpct:
            fig_needpct = create_mini_chart(needpct, "NeedPct Progression")
            st.plotly_chart(fig_needpct, use_container_width=True)
        
        # Volume chart
        volume_pct = result.get('volume_pct', [])
        if volume_pct:
            fig_volume = create_mini_chart(volume_pct, "Volume Distribution", "bar")
            st.plotly_chart(fig_volume, use_container_width=True)
    
    # Additional metrics
    st.subheader("📊 Derived Metrics")
    col1, col2, col3, col4 = st.columns(4)
    
    diagnostics = result.get('diagnostics', {})
    with col1:
        st.metric("WCI", f"{diagnostics.get('wci', 0):.3f}")
        st.caption("Weight Center Index")
    
    with col2:
        st.metric("Sign Flips", diagnostics.get('sign_flips', 0))
        st.caption("Trend Changes")
    
    with col3:
        st.metric("Gini", f"{diagnostics.get('gini', 0):.3f}")
        st.caption("Volume Concentration")
    
    with col4:
        st.metric("Entropy", f"{diagnostics.get('entropy', 0):.3f}")
        st.caption("Volume Diversity")
    
    # Export options
    st.subheader("💾 Export")
    col1, col2 = st.columns(2)
    
    with col1:
        # CSV export for this result
        result_df = pd.DataFrame({
            "Order": range(1, len(result.get('volume_pct', [])) + 1),
            "Indent_Pct": result.get('indent_pct', []),
            "Volume_Pct": result.get('volume_pct', []),
            "Martingale_Pct": result.get('martingale_pct', []),
            "NeedPct": result.get('needpct', []),
        })
        csv_data = result_df.to_csv(index=False)
        st.download_button(
            "Download This Result CSV",
            csv_data,
            f"dca_result_rank_{result.get('rank', 'unknown')}.csv",
            "text/csv"
        )
    
    with col2:
        # JSON export for this result
        json_data = json.dumps(result.get('_raw', {}), indent=2, ensure_ascii=False)
        st.download_button(
            "Download This Result JSON",
            json_data,
            f"dca_result_rank_{result.get('rank', 'unknown')}.json",
            "application/json"
        )


def display_results_table(results: List[Dict[str, Any]], filters: Dict[str, Any]):
    """Display the enhanced results table."""
    if not results:
        st.warning("No results to display.")
        return
    
    # Apply filters
    filtered_results = []
    for result in results:
        if result.get('score', 0) > filters.get('max_score', 100.0):
            continue
        if not (filters.get('overlap_min', 0) <= result.get('overlap_pct', 0) <= filters.get('overlap_max', 100)):
            continue
        if not (filters.get('orders_min', 0) <= result.get('orders', 0) <= filters.get('orders_max', 100)):
            continue
        if filters.get('wave_only', False) and result.get('wave_score', 0) < 0.6:
            continue
        filtered_results.append(result)
    
    if not filtered_results:
        st.warning("No results match the current filters.")
        return
    
    st.subheader(f"🏆 Top {len(filtered_results)} Results")
    
    # Prepare table data
    table_data = []
    for i, result in enumerate(filtered_results):
        needpct = result.get('needpct', [])
        martingale_pct = result.get('martingale_pct', [])
        sanity = result.get('sanity', {})
        
        # Calculate derived metrics
        exit_speed = compute_exit_speed(needpct)
        wave_score = compute_wave_score(martingale_pct)
        need_spark = create_needpct_sparkline(needpct)
        sanity_badge_text = ", ".join(sanity_badges(sanity)) or "OK"
        
        diagnostics = result.get('diagnostics', {})
        
        row = {
            "Rank": i + 1,
            "Score": f"{result.get('score', 0):.3f}",
            "Max Need": f"{result.get('max_need', 0):.2f}%",
            "Var Need": f"{result.get('var_need', 0):.3f}",
            "Tail": f"{result.get('tail', 0):.3f}",
            "WCI": f"{diagnostics.get('wci', 0):.3f}",
            "Sign Flips": diagnostics.get('sign_flips', 0),
            "Overlap": f"{result.get('overlap_pct', 0):.1f}%",
            "M": result.get('orders', 0),
            "Exit Speed": f"{exit_speed:.3f}",
            "Wave Score": f"{wave_score:.3f}",
            "NeedPct": need_spark,
            "Sanity": sanity_badge_text,
            "_raw": result
        }
        table_data.append(row)
    
    # Display table
    df = pd.DataFrame(table_data)
    st.dataframe(df, use_container_width=True, hide_index=True)
    
    # Detail selection
    if filtered_results:
        selected_rank = st.selectbox(
            "Select result to view details:",
            range(len(filtered_results)),
            format_func=lambda x: f"Rank {x+1}: Score {filtered_results[x].get('score', 0):.3f}"
        )
        
        if selected_rank is not None:
            selected_result = filtered_results[selected_rank]
            selected_result['rank'] = selected_rank + 1
            display_result_detail(selected_result)
    
    # Export all results
    st.subheader("📥 Export All Results")
    col1, col2 = st.columns(2)
    
    with col1:
        # CSV export for all results
        export_df = pd.DataFrame(table_data)
        export_df = export_df.drop('_raw', axis=1)  # Remove raw data for CSV
        csv_data = export_df.to_csv(index=False)
        st.download_button(
            "Download All Results CSV",
            csv_data,
            "dca_all_results.csv",
            "text/csv"
        )
    
    with col2:
        # JSON export for all results
        json_data = json.dumps([r.get('_raw', {}) for r in filtered_results], indent=2, ensure_ascii=False)
        st.download_button(
            "Download All Results JSON",
            json_data,
            "dca_all_results.json",
            "application/json"
        )


def render_results_section(experiment_id: Optional[int] = None):
    """Main function to render the enhanced results section."""
    st.header("🎯 Enhanced DCA/Martingale Results - İşlemden En Hızlı Çıkış")
    
    # Live logs panel
    with st.expander("📊 Live Logs", expanded=False):
        logs = get_live_trace("mlab", last_n=200)
        if logs:
            for log in logs[-20:]:
                st.write(f"{log.get('event','')} | {log.get('msg','')}")
    
    store = ExperimentsStore()
    
    if experiment_id:
        # Display experiment summary
        display_experiment_summary(experiment_id, store)
        
        # Display filters
        filters = display_filters()
        
        # Get and display results
        results = store.get_results(experiment_id=experiment_id, limit=1000)
        
        if results:
            # Log refresh
            from martingale_lab.utils.structured_logging import get_structured_logger, EventNames
            logger = get_structured_logger("mlab.ui")
            logger.info(
                EventNames.RESULTS_REFRESH,
                f"Displaying {len(results)} results for experiment {experiment_id}",
                exp_id=experiment_id,
                result_count=len(results)
            )
            
            display_results_table(results, filters)
        else:
            st.warning("No results found for this experiment.")
            
    else:
        st.info("Please select an experiment to view results.")
        
        # Show available experiments
        with sqlite3.connect(store.db_path) as conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT id, created_at, adapter, best_score, total_evals FROM experiments WHERE deleted = 0 ORDER BY created_at DESC LIMIT 10"
            )
            experiments = cur.fetchall()
        
        if experiments:
            st.subheader("Recent Experiments")
            exp_df = pd.DataFrame(experiments, columns=["ID", "Created", "Adapter", "Best Score", "Total Evals"])
            st.dataframe(exp_df, hide_index=True)
        else:
            st.warning("No experiments found. Run an optimization first.")
