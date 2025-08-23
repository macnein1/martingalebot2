"""
Results Section UI - DCA/Martingale "İşlemden En Hızlı Çıkış" Results Display
Shows Top-N candidates with bullets format, NeedPct sparklines, sanity badges, and filtering.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Any, Optional
import json
import sqlite3


def create_needpct_sparkline(needpct: List[float], width: int = 100, height: int = 30) -> str:
    """Create a simple ASCII sparkline for NeedPct values."""
    if not needpct or len(needpct) == 0:
        return "─" * 10
    
    min_val = min(needpct)
    max_val = max(needpct)
    if max_val == min_val:
        return "─" * len(needpct)
    
    # Normalize to 0-7 range for spark characters
    spark_chars = "▁▂▃▄▅▆▇█"
    normalized = [(val - min_val) / (max_val - min_val) * 7 for val in needpct]
    sparkline = "".join([spark_chars[min(7, int(val))] for val in normalized])
    
    return sparkline


def format_bullets(schedule: Dict[str, Any]) -> List[str]:
    """
    Format bullets exactly as specified:
    1. Emir: Indent %x Volume %y (no martingale) — NeedPct %n
    2. Emir: Indent %x Volume %y (Martingale %m) — NeedPct %n
    """
    indent_pct = schedule.get("indent_pct", [])
    volume_pct = schedule.get("volume_pct", [])
    martingale_pct = schedule.get("martingale_pct", [])
    needpct = schedule.get("needpct", [])
    
    bullets = []
    n = len(volume_pct)
    
    for i in range(n):
        indent = indent_pct[i] if i < len(indent_pct) else 0.0
        volume = volume_pct[i] if i < len(volume_pct) else 0.0
        martingale = martingale_pct[i] if i < len(martingale_pct) else 0.0
        need = needpct[i] if i < len(needpct) else 0.0
        
        if i == 0:
            bullet = f"{i+1}. Emir: Indent %{indent:.2f}  Volume %{volume:.2f}  (no martingale) — NeedPct %{need:.2f}"
        else:
            bullet = f"{i+1}. Emir: Indent %{indent:.2f}  Volume %{volume:.2f}  (Martingale %{martingale:.2f}) — NeedPct %{need:.2f}"
        
        bullets.append(bullet)
    
    return bullets


def create_sanity_badges(sanity: Dict[str, bool]) -> str:
    """Create compact badges for sanity check flags."""
    badges = []
    if sanity.get("collapse_indents", False):
        badges.append("indent↧")
    if sanity.get("tail_overflow", False):
        badges.append("tail↑")
    if sanity.get("max_need_mismatch", False):
        badges.append("max≠")
    return " ".join(badges) if badges else "OK"


def create_needpct_chart(needpct: List[float], title: str = "NeedPct Progression") -> go.Figure:
    """Create detailed NeedPct chart for expanded view."""
    if not needpct:
        fig = go.Figure()
        fig.add_annotation(text="No data available", xref="paper", yref="paper", x=0.5, y=0.5)
        return fig
    
    fig = go.Figure()
    
    # Main line
    fig.add_trace(go.Scatter(
        x=list(range(1, len(needpct) + 1)),
        y=needpct,
        mode='lines+markers',
        name='NeedPct',
        line=dict(color='#1f77b4', width=3),
        marker=dict(size=8)
    ))
    
    # Add zero line
    fig.add_hline(y=0, line_dash="dash", line_color="gray", annotation_text="Break-even")
    
    fig.update_layout(
        title=title,
        xaxis_title="Order Number",
        yaxis_title="NeedPct (%)",
        height=300,
        showlegend=False,
        template="plotly_white"
    )
    
    return fig


def create_volume_distribution_chart(schedule: Dict[str, Any]) -> go.Figure:
    """Create volume distribution chart with indent and martingale info."""
    volume_pct = schedule.get("volume_pct", [])
    indent_pct = schedule.get("indent_pct", [])
    martingale_pct = schedule.get("martingale_pct", [])
    
    if not volume_pct:
        fig = go.Figure()
        fig.add_annotation(text="No data available", xref="paper", yref="paper", x=0.5, y=0.5)
        return fig
    
    fig = go.Figure()
    
    # Volume bars
    fig.add_trace(go.Bar(
        x=list(range(1, len(volume_pct) + 1)),
        y=volume_pct,
        name='Volume %',
        marker_color='lightblue',
        text=[f"{v:.1f}%" for v in volume_pct],
        textposition='auto'
    ))
    
    # Indent line (secondary y-axis)
    if indent_pct:
        fig.add_trace(go.Scatter(
            x=list(range(1, len(indent_pct) + 1)),
            y=indent_pct,
            mode='lines+markers',
            name='Indent %',
            yaxis='y2',
            line=dict(color='red', dash='dot'),
            marker=dict(size=6)
        ))
    
    # Martingale line (secondary y-axis)
    if martingale_pct:
        fig.add_trace(go.Scatter(
            x=list(range(1, len(martingale_pct) + 1)),
            y=martingale_pct,
            mode='lines+markers',
            name='Martingale %',
            yaxis='y2',
            line=dict(color='green', dash='dash'),
            marker=dict(size=6)
        ))
    
    fig.update_layout(
        title="Volume Distribution with Indent & Martingale",
        xaxis_title="Order Number",
        yaxis=dict(title="Volume %", side="left"),
        yaxis2=dict(title="Indent/Martingale %", side="right", overlaying="y"),
        height=400,
        template="plotly_white"
    )
    
    return fig


def display_results_filters():
    """Display filtering controls."""
    st.subheader("🔍 Filters")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        max_score = st.number_input("Max Score (J)", min_value=0.0, value=100.0, step=1.0)
        overlap_range = st.slider("Overlap % Range", 5.0, 50.0, (10.0, 30.0), step=0.5)
    
    with col2:
        max_need_range = st.slider("Max Need % Range", 0.0, 50.0, (0.0, 20.0), step=0.5)
        orders_range = st.slider("Orders Range", 2, 30, (5, 15))
    
    with col3:
        wave_pattern_only = st.checkbox("Wave Pattern Only")
        pareto_only = st.checkbox("Pareto Optimal Only")
    
    return {
        "max_score": max_score,
        "overlap_min": overlap_range[0],
        "overlap_max": overlap_range[1],
        "min_max_need": max_need_range[0],
        "max_max_need": max_need_range[1],
        "orders_min": orders_range[0],
        "orders_max": orders_range[1],
        "wave_pattern_only": wave_pattern_only,
        "pareto_only": pareto_only
    }


def display_summary_card(summary: Dict[str, Any]):
    """Display experiment summary card."""
    st.subheader("📊 Experiment Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Best Score", f"{summary.get('best_score', 0):.3f}")
        st.metric("Total Results", f"{summary.get('statistics', {}).get('total_results', 0):,}")
    
    with col2:
        stats = summary.get('statistics', {})
        st.metric("Min Max Need", f"{stats.get('min_max_need', 0):.2f}%")
        st.metric("Avg Max Need", f"{stats.get('avg_max_need', 0):.2f}%")
    
    with col3:
        st.metric("Avg Var Need", f"{stats.get('avg_var_need', 0):.3f}")
        st.metric("Avg Tail", f"{stats.get('avg_tail', 0):.3f}")
    
    with col4:
        config = summary.get('config', {})
        st.metric("Orders Range", f"{config.get('orders_min', 0)}-{config.get('orders_max', 0)}")
        st.metric("Overlap Range", f"{config.get('overlap_min', 0):.1f}%-{config.get('overlap_max', 0):.1f}%")


def display_results_table(results: List[Dict[str, Any]], limit: int = 50):
    """Display the main results table with expandable rows."""
    if not results:
        st.warning("No results to display.")
        return
    
    st.subheader(f"🏆 Top {min(len(results), limit)} Results")
    
    # Determine maximum number of exit requirements across results
    max_m = 0
    for res in results[:limit]:
        payload = res.get("payload", {})
        schedule = payload.get("schedule", {})
        needpct = schedule.get("needpct", [])
        if isinstance(needpct, list):
            max_m = max(max_m, len(needpct))
    
    # Prepare table data
    table_rows: List[Dict[str, Any]] = []
    for i, result in enumerate(results[:limit]):
        payload = result.get("payload", {})
        schedule = payload.get("schedule", {})
        sanity = payload.get("sanity", {}) or result.get("sanity", {})
        diagnostics = payload.get("diagnostics", {}) or result.get("diagnostics", {})
        needpct = schedule.get("needpct", []) or []
        
        row: Dict[str, Any] = {
            "Rank": i + 1,
            "Score (J)": f"{result.get('score', 0):.3f}",
            "Exit Mean": f"{(float(np.mean(needpct)) if len(needpct) else 0.0):.2f}",
            "Exit Max": f"{(float(np.max(needpct)) if len(needpct) else 0.0):.2f}",
            "Badges": create_sanity_badges(sanity),
        }
        # Add exit_req_1..M columns
        for k in range(max_m):
            col_name = f"exit_req_{k+1}"
            row[col_name] = f"{needpct[k]:.2f}" if k < len(needpct) else ""
        
        table_rows.append(row)
    
    df = pd.DataFrame(table_rows)
    st.dataframe(df, use_container_width=True, hide_index=True)
    
    # Expandable details sections
    st.subheader("📋 Detailed View")
    
    selected_index = st.selectbox(
        "Select result to view details:",
        range(min(len(results), limit)),
        format_func=lambda x: f"Rank {x+1}: Score {results[x].get('score', 0):.3f}"
    )
    
    if selected_index is not None and selected_index < len(results):
        result = results[selected_index]
        display_result_detail(result)


def display_result_detail(result: Dict[str, Any]):
    """Display detailed view of a single result."""
    payload = result.get("payload", {})
    schedule = payload.get("schedule", {})
    sanity = payload.get("sanity", {}) or result.get("sanity", {})
    diagnostics = payload.get("diagnostics", {}) or result.get("diagnostics", {})
    penalties = payload.get("penalties", {}) or result.get("penalties", {})
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Score (J)", f"{result.get('score', 0):.3f}")
    with col2:
        st.metric("Exit Mean", f"{(float(np.mean(schedule.get('needpct', [])) if schedule.get('needpct') else 0.0)):.2f}")
    with col3:
        st.metric("Exit Max", f"{(float(np.max(schedule.get('needpct', [])) if schedule.get('needpct') else 0.0)):.2f}")
    with col4:
        st.metric("Badges", create_sanity_badges(sanity))
    
    # Bullets format
    st.subheader("🎯 Order Bullets")
    bullets = format_bullets(schedule)
    for bullet in bullets:
        st.text(bullet)
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        needpct = schedule.get("needpct", [])
        fig_needpct = create_needpct_chart(needpct)
        st.plotly_chart(fig_needpct, use_container_width=True)
    
    with col2:
        fig_volume = create_volume_distribution_chart(schedule)
        st.plotly_chart(fig_volume, use_container_width=True)
    
    # Additional metrics
    st.subheader("📈 Diagnostics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("WCI", f"{diagnostics.get('wci', 0):.3f}")
        st.caption("Weight Center Index (0=early, 1=late)")
    
    with col2:
        st.metric("Sign Flips", diagnostics.get('sign_flips', 0))
        st.caption("NeedPct trend changes")
    
    with col3:
        st.metric("Gini", f"{diagnostics.get('gini', 0):.3f}")
        st.caption("Volume concentration")
    
    with col4:
        st.metric("Entropy", f"{diagnostics.get('entropy', 0):.3f}")
        st.caption("Volume diversity")
    
    # Download options
    st.subheader("💾 Download")
    col1, col2 = st.columns(2)
    
    with col1:
        # CSV download for schedule + needpct
        schedule_df = pd.DataFrame({
            "Order": range(1, len(schedule.get("volume_pct", [])) + 1),
            "Indent_Pct": schedule.get("indent_pct", []),
            "Volume_Pct": schedule.get("volume_pct", []),
            "Martingale_Pct": schedule.get("martingale_pct", []),
            "NeedPct": schedule.get("needpct", []),
        })
        csv_data = schedule_df.to_csv(index=False)
        st.download_button(
            "Download Schedule CSV",
            csv_data,
            f"dca_schedule_rank_{result.get('rank', 'unknown')}.csv",
            "text/csv"
        )
    
    with col2:
        # JSON download for payload
        json_data = json.dumps(payload, indent=2, ensure_ascii=False)
        st.download_button(
            "Download Payload JSON",
            json_data,
            f"dca_payload_rank_{result.get('rank', 'unknown')}.json",
            "application/json"
        )


def render_results_section(experiment_id: Optional[int] = None):
    """Main function to render the complete results section."""
    st.header("🎯 DCA/Martingale Results - İşlemden En Hızlı Çıkış")
    
    # Import here to avoid circular imports
    from martingale_lab.storage.experiments_store import ExperimentsStore
    
    store = ExperimentsStore()
    
    if experiment_id:
        # Get experiment summary
        summary = store.get_experiment_summary(experiment_id)
        if summary:
            display_summary_card(summary)
        
        # Display filters
        filters = display_results_filters()
        
        # Get and display results
        results = store.get_top_results(experiment_id=experiment_id, limit=100, filters=filters)
        display_results_table(results)
        
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
