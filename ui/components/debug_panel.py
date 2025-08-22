"""
Debug panel component for detailed candidate inspection and system diagnostics.
Provides comprehensive debugging interface for optimization runs.
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from typing import Dict, Any, List, Optional
from datetime import datetime

from ..utils.optimization_bridge import OptimizationBridge
from ..utils.performance_monitor import PerformanceMonitor


class DebugPanel:
    """Debug panel for detailed system inspection."""
    
    def __init__(self, bridge: OptimizationBridge, perf_monitor: PerformanceMonitor):
        self.bridge = bridge
        self.perf_monitor = perf_monitor
    
    def render(self):
        """Render the complete debug panel."""
        st.header("🔍 Debug & Diagnostics")
        
        # Debug tabs
        tab_candidate, tab_logs, tab_system, tab_database = st.tabs([
            "📋 Candidate Details", "📝 Logs", "⚙️ System", "🗄️ Database"
        ])
        
        with tab_candidate:
            self._render_candidate_debug()
        
        with tab_logs:
            self._render_logs_panel()
        
        with tab_system:
            self._render_system_diagnostics()
        
        with tab_database:
            self._render_database_inspector()
    
    def _render_candidate_debug(self):
        """Render candidate debugging interface."""
        st.subheader("Candidate Inspector")
        
        # Get available runs
        runs = self.bridge.get_available_runs()
        if not runs:
            st.info("No optimization runs available. Start a run to see candidates.")
            return
        
        # Run selection
        run_options = [f"{run['id'][:8]}... ({run['status']}) - {run['started_at'][:10]}" 
                      for run in runs]
        selected_run_idx = st.selectbox("Select Run:", range(len(run_options)), 
                                       format_func=lambda x: run_options[x])
        
        selected_run = runs[selected_run_idx]
        run_id = selected_run['id']
        
        # Get candidates for selected run
        candidates = self.bridge.get_run_candidates(run_id, limit=50)
        
        if not candidates:
            st.info("No candidates found for this run.")
            return
        
        # Candidate selection
        candidate_options = [f"{c['id']} (Score: {c.get('J', 'N/A')})" for c in candidates]
        selected_candidate_idx = st.selectbox("Select Candidate:", range(len(candidate_options)),
                                             format_func=lambda x: candidate_options[x])
        
        candidate = candidates[selected_candidate_idx]
        
        # Display candidate details
        self._display_candidate_details(candidate)
    
    def _display_candidate_details(self, candidate: Dict[str, Any]):
        """Display detailed candidate information."""
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Basic Info")
            st.text(f"ID: {candidate['id']}")
            st.text(f"Overlap: {candidate.get('overlap', 'N/A')}%")
            st.text(f"Orders: {candidate.get('orders', 'N/A')}")
            st.text(f"Score (J): {candidate.get('J', 'N/A')}")
            st.text(f"Evaluation Time: {candidate.get('evaluation_time_ms', 'N/A')} ms")
            st.text(f"Fallback Used: {'Yes' if candidate.get('fallback_used') else 'No'}")
        
        with col2:
            st.subheader("📈 Metrics")
            st.text(f"Max Need: {candidate.get('max_need', 'N/A')}")
            st.text(f"Var Need: {candidate.get('var_need', 'N/A')}")
            st.text(f"Tail Risk: {candidate.get('tail', 'N/A')}")
            st.text(f"Gini Coefficient: {candidate.get('gini', 'N/A')}")
            st.text(f"Entropy: {candidate.get('entropy', 'N/A')}")
        
        # Parse and display schedule if available
        if candidate.get('schedule_json'):
            try:
                schedule_data = json.loads(candidate['schedule_json'])
                self._display_schedule_visualization(schedule_data)
            except json.JSONDecodeError:
                st.warning("Could not parse schedule data")
        
        # Parse and display penalties if available
        if candidate.get('penalties_json'):
            try:
                penalties = json.loads(candidate['penalties_json'])
                self._display_penalty_breakdown(penalties)
            except json.JSONDecodeError:
                st.warning("Could not parse penalty data")
        
        # Display parameters
        if candidate.get('params_json'):
            try:
                params = json.loads(candidate['params_json'])
                st.subheader("⚙️ Parameters")
                st.json(params)
            except json.JSONDecodeError:
                st.warning("Could not parse parameter data")
    
    def _display_schedule_visualization(self, schedule_data: Dict[str, Any]):
        """Display schedule visualization."""
        st.subheader("📋 Order Schedule")
        
        # Create mock data if real schedule structure is different
        if 'orders' in schedule_data and 'volumes' in schedule_data:
            orders = schedule_data['orders']
            volumes = schedule_data['volumes']
            indents = schedule_data.get('indents', list(range(len(orders))))
            
            # Create DataFrame for display
            df = pd.DataFrame({
                'Order': range(1, len(orders) + 1),
                'Indent %': indents,
                'Volume %': [v * 100 for v in volumes],
                'Order Size': orders
            })
            
            # Display as table
            st.dataframe(df, use_container_width=True)
            
            # Visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                # Volume distribution
                fig_vol = px.bar(df, x='Order', y='Volume %', 
                               title='Volume Distribution by Order')
                fig_vol.update_layout(height=300)
                st.plotly_chart(fig_vol, use_container_width=True)
            
            with col2:
                # Indent progression
                fig_indent = px.line(df, x='Order', y='Indent %', 
                                   title='Indent Progression', markers=True)
                fig_indent.update_layout(height=300)
                st.plotly_chart(fig_indent, use_container_width=True)
        else:
            st.json(schedule_data)
    
    def _display_penalty_breakdown(self, penalties: Dict[str, Any]):
        """Display penalty breakdown."""
        st.subheader("⚖️ Penalty Breakdown")
        
        if isinstance(penalties, dict) and penalties:
            # Create penalty visualization
            penalty_names = list(penalties.keys())
            penalty_values = list(penalties.values())
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Penalty table
                penalty_df = pd.DataFrame({
                    'Penalty Type': penalty_names,
                    'Value': penalty_values
                })
                st.dataframe(penalty_df, use_container_width=True)
            
            with col2:
                # Penalty pie chart
                fig_penalty = px.pie(penalty_df, values='Value', names='Penalty Type',
                                   title='Penalty Contribution')
                fig_penalty.update_layout(height=300)
                st.plotly_chart(fig_penalty, use_container_width=True)
        else:
            st.json(penalties)
    
    def _render_logs_panel(self):
        """Render logs inspection panel."""
        st.subheader("📝 System Logs")
        
        # Log filtering options
        col1, col2, col3 = st.columns(3)
        
        with col1:
            log_level = st.selectbox("Log Level:", ["ALL", "ERROR", "WARNING", "INFO"])
        
        with col2:
            event_filter = st.text_input("Event Filter:", placeholder="e.g., candidate_error")
        
        with col3:
            max_lines = st.number_input("Max Lines:", min_value=10, max_value=1000, value=100)
        
        # Get and display logs
        try:
            logs = self.bridge.get_recent_logs(max_lines=max_lines, 
                                             event_filter=event_filter if event_filter else None)
            
            if logs:
                # Convert to DataFrame for better display
                log_df = pd.DataFrame(logs)
                
                # Apply level filter
                if log_level != "ALL":
                    log_df = log_df[log_df.get('level', 'INFO') == log_level]
                
                # Display log entries
                for _, log_entry in log_df.iterrows():
                    with st.expander(f"{log_entry.get('timestamp', 'Unknown')} - {log_entry.get('event', 'Unknown')}"):
                        st.json(log_entry.to_dict())
            else:
                st.info("No logs available")
        
        except Exception as e:
            st.error(f"Error loading logs: {str(e)}")
    
    def _render_system_diagnostics(self):
        """Render system diagnostics panel."""
        st.subheader("⚙️ System Diagnostics")
        
        # Performance metrics
        try:
            metrics = self.perf_monitor.get_current_metrics()
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Memory Usage", f"{metrics.get('memory_usage_mb', 0):.1f} MB")
                st.metric("CPU Usage", f"{metrics.get('cpu_usage_pct', 0):.1f}%")
            
            with col2:
                st.metric("Eval Rate", f"{metrics.get('evaluations_per_second', 0):.1f}/s")
                st.metric("Accept Rate", f"{metrics.get('accept_ratio', 0)*100:.1f}%")
            
            with col3:
                st.metric("NaN Count", metrics.get('nan_count', 0))
                st.metric("Fallback Count", metrics.get('fallback_count', 0))
            
            # Sparkline charts
            sparkline_data = self.perf_monitor.get_sparkline_data()
            
            if sparkline_data and any(sparkline_data.values()):
                st.subheader("📈 Performance Trends")
                
                # Create subplot for sparklines
                fig = make_subplots(
                    rows=2, cols=2,
                    subplot_titles=['Evaluation Rate', 'Accept Rate', 'Memory Usage', 'Reserved'],
                    specs=[[{"secondary_y": False}, {"secondary_y": False}],
                           [{"secondary_y": False}, {"secondary_y": False}]]
                )
                
                # Add sparklines
                if sparkline_data.get('eval_rate'):
                    fig.add_trace(
                        go.Scatter(y=sparkline_data['eval_rate'], mode='lines', name='Eval Rate'),
                        row=1, col=1
                    )
                
                if sparkline_data.get('accept_rate'):
                    fig.add_trace(
                        go.Scatter(y=sparkline_data['accept_rate'], mode='lines', name='Accept Rate'),
                        row=1, col=2
                    )
                
                if sparkline_data.get('memory_usage'):
                    fig.add_trace(
                        go.Scatter(y=sparkline_data['memory_usage'], mode='lines', name='Memory'),
                        row=2, col=1
                    )
                
                fig.update_layout(height=400, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
        
        except Exception as e:
            st.error(f"Error loading system metrics: {str(e)}")
        
        # Error summary
        try:
            error_summary = self.bridge.get_error_summary()
            if error_summary:
                st.subheader("🚨 Error Summary")
                
                error_df = pd.DataFrame([
                    {"Error Type": k, "Count": v} 
                    for k, v in error_summary.items()
                ])
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.dataframe(error_df, use_container_width=True)
                
                with col2:
                    if not error_df.empty:
                        fig_errors = px.bar(error_df, x='Error Type', y='Count',
                                          title='Error Distribution')
                        st.plotly_chart(fig_errors, use_container_width=True)
        
        except Exception as e:
            st.warning(f"Could not load error summary: {str(e)}")
    
    def _render_database_inspector(self):
        """Render database inspection panel."""
        st.subheader("🗄️ Database Inspector")
        
        try:
            # Database statistics
            db_stats = self.bridge.get_database_stats()
            
            if db_stats:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Total Runs", db_stats.get('total_runs', 0))
                    st.metric("Active Runs", db_stats.get('active_runs', 0))
                
                with col2:
                    st.metric("Total Candidates", db_stats.get('total_candidates', 0))
                    st.metric("Total Batches", db_stats.get('total_batches', 0))
                
                with col3:
                    st.metric("DB Size", f"{db_stats.get('db_size_mb', 0):.1f} MB")
                    st.metric("Log Entries", db_stats.get('log_entries', 0))
            
            # Recent activity
            st.subheader("📊 Recent Activity")
            
            recent_candidates = self.bridge.get_recent_candidates(limit=20)
            if recent_candidates:
                recent_df = pd.DataFrame(recent_candidates)
                
                # Select relevant columns for display
                display_columns = ['id', 'J', 'overlap', 'orders', 'evaluation_time_ms', 'fallback_used']
                available_columns = [col for col in display_columns if col in recent_df.columns]
                
                if available_columns:
                    st.dataframe(recent_df[available_columns], use_container_width=True)
                else:
                    st.dataframe(recent_df, use_container_width=True)
            else:
                st.info("No recent candidates found")
        
        except Exception as e:
            st.error(f"Error loading database information: {str(e)}")
        
        # Database maintenance
        st.subheader("🔧 Database Maintenance")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Clean Old Runs"):
                try:
                    cleaned_count = self.bridge.cleanup_old_runs(keep_days=30)
                    st.success(f"Cleaned {cleaned_count} old runs")
                except Exception as e:
                    st.error(f"Cleanup failed: {str(e)}")
        
        with col2:
            if st.button("Vacuum Database"):
                try:
                    self.bridge.vacuum_database()
                    st.success("Database vacuumed successfully")
                except Exception as e:
                    st.error(f"Vacuum failed: {str(e)}")


class CandidateDetailModal:
    """Modal dialog for detailed candidate inspection."""
    
    @staticmethod
    def show(candidate: Dict[str, Any]):
        """Show candidate detail modal."""
        st.subheader(f"🔍 Candidate Details: {candidate['id']}")
        
        # Comprehensive candidate information
        tabs = st.tabs(["📊 Overview", "📋 Schedule", "⚖️ Penalties", "🔧 Raw Data"])
        
        with tabs[0]:
            CandidateDetailModal._render_overview(candidate)
        
        with tabs[1]:
            CandidateDetailModal._render_schedule_detail(candidate)
        
        with tabs[2]:
            CandidateDetailModal._render_penalty_detail(candidate)
        
        with tabs[3]:
            CandidateDetailModal._render_raw_data(candidate)
    
    @staticmethod
    def _render_overview(candidate: Dict[str, Any]):
        """Render candidate overview."""
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("🎯 Objectives")
            st.metric("Final Score (J)", f"{candidate.get('J', 'N/A')}")
            st.metric("Max Need %", f"{candidate.get('max_need', 'N/A')}")
            st.metric("Need Variance", f"{candidate.get('var_need', 'N/A')}")
            st.metric("Tail Risk", f"{candidate.get('tail', 'N/A')}")
        
        with col2:
            st.subheader("📐 Configuration")
            st.metric("Overlap %", f"{candidate.get('overlap', 'N/A')}")
            st.metric("Order Count", f"{candidate.get('orders', 'N/A')}")
            st.metric("Gini Coefficient", f"{candidate.get('gini', 'N/A'):.4f}" if candidate.get('gini') else 'N/A')
            st.metric("Entropy", f"{candidate.get('entropy', 'N/A'):.4f}" if candidate.get('entropy') else 'N/A')
        
        with col3:
            st.subheader("⚡ Performance")
            st.metric("Evaluation Time", f"{candidate.get('evaluation_time_ms', 'N/A')} ms")
            st.metric("Fallback Used", "Yes" if candidate.get('fallback_used') else "No")
            
            # Performance indicator
            eval_time = candidate.get('evaluation_time_ms', 0)
            if eval_time > 0:
                if eval_time < 100:
                    st.success("Fast evaluation")
                elif eval_time < 1000:
                    st.info("Normal evaluation")
                else:
                    st.warning("Slow evaluation")
    
    @staticmethod
    def _render_schedule_detail(candidate: Dict[str, Any]):
        """Render detailed schedule information."""
        if not candidate.get('schedule_json'):
            st.info("No schedule data available")
            return
        
        try:
            schedule_data = json.loads(candidate['schedule_json'])
            
            # Create detailed schedule table
            if 'orders' in schedule_data and 'volumes' in schedule_data:
                orders = schedule_data['orders']
                volumes = schedule_data['volumes']
                indents = schedule_data.get('indents', list(range(len(orders))))
                
                # Calculate additional metrics
                martingale_pcts = []
                for i in range(len(orders)):
                    if i == 0:
                        martingale_pcts.append(0.0)  # First order has no martingale
                    else:
                        # Calculate martingale percentage (simplified)
                        martingale_pct = ((volumes[i] / volumes[i-1]) - 1) * 100 if volumes[i-1] > 0 else 0
                        martingale_pcts.append(max(0, martingale_pct))
                
                # Create comprehensive DataFrame
                df = pd.DataFrame({
                    'Order #': range(1, len(orders) + 1),
                    'Indent %': [f"{indent:.2f}" for indent in indents],
                    'Volume %': [f"{vol*100:.2f}" for vol in volumes],
                    'Order Size': orders,
                    'Martingale %': [f"{mp:.1f}" if mp > 0 else "N/A" for mp in martingale_pcts]
                })
                
                st.dataframe(df, use_container_width=True)
                
                # Bullet-point format as requested
                st.subheader("📋 Order Breakdown")
                for i, (order_num, indent, volume, order_size, martingale) in enumerate(zip(
                    range(1, len(orders) + 1), indents, volumes, orders, martingale_pcts
                )):
                    if i == 0:
                        st.markdown(f"**{order_num}. Order:** Indent %{indent:.2f} Volume %{volume*100:.2f} (no martingale, first order)")
                    else:
                        st.markdown(f"**{order_num}. Order:** Indent %{indent:.2f} Volume %{volume*100:.2f} (Martingale %{martingale:.1f})")
            else:
                st.json(schedule_data)
                
        except json.JSONDecodeError:
            st.error("Could not parse schedule data")
    
    @staticmethod
    def _render_penalty_detail(candidate: Dict[str, Any]):
        """Render detailed penalty breakdown."""
        if not candidate.get('penalties_json'):
            st.info("No penalty data available")
            return
        
        try:
            penalties = json.loads(candidate['penalties_json'])
            
            if isinstance(penalties, dict) and penalties:
                # Detailed penalty analysis
                total_penalty = sum(penalties.values()) if all(isinstance(v, (int, float)) for v in penalties.values()) else 0
                
                st.metric("Total Penalty", f"{total_penalty:.6f}")
                
                # Penalty breakdown with explanations
                penalty_explanations = {
                    'gini': 'Measures volume concentration (0 = equal distribution, 1 = all in one order)',
                    'entropy': 'Measures diversity penalty (higher entropy = more diverse)',
                    'monotone': 'Penalty for non-monotonic indent progression',
                    'step_smooth': 'Penalty for abrupt NeedPct changes between orders',
                    'tail_cap': 'Penalty if last order volume exceeds threshold',
                    'extreme_vol': 'Penalty for volumes outside reasonable bounds',
                    'need_variance': 'Penalty for high variance in NeedPct values'
                }
                
                for penalty_name, penalty_value in penalties.items():
                    with st.expander(f"{penalty_name}: {penalty_value:.6f}"):
                        explanation = penalty_explanations.get(penalty_name, "No explanation available")
                        st.write(explanation)
                        
                        if penalty_value > 0.001:  # Significant penalty
                            st.warning("⚠️ This penalty is active and may indicate a constraint violation")
                        else:
                            st.success("✅ This penalty is minimal")
            else:
                st.json(penalties)
                
        except json.JSONDecodeError:
            st.error("Could not parse penalty data")
    
    @staticmethod
    def _render_raw_data(candidate: Dict[str, Any]):
        """Render raw candidate data."""
        st.subheader("🔧 Raw Data")
        
        # Display all candidate data
        st.json(candidate)
        
        # Export options
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📋 Copy to Clipboard"):
                st.code(json.dumps(candidate, indent=2))
        
        with col2:
            # Download as JSON
            json_str = json.dumps(candidate, indent=2)
            st.download_button(
                label="💾 Download JSON",
                data=json_str,
                file_name=f"candidate_{candidate['id']}.json",
                mime="application/json"
            )