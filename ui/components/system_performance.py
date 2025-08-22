import streamlit as st
import psutil
from ui.utils.config import get_icon_html
from ui.utils.performance_monitor import PerformanceMonitor, display_performance_metrics

def get_system_metrics():
    """Get current system performance metrics"""
    try:
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        cpu_count = psutil.cpu_count()
        available_ram_gb = memory.available / (1024**3)
        
        return {
            'cpu_usage': cpu_percent,
            'ram_usage': memory.percent,
            'cpu_count': cpu_count,
            'available_ram': available_ram_gb
        }
    except Exception as e:
        st.error(f"Sistem metrikleri alınamadı: {e}")
        return {
            'cpu_usage': 0.0,
            'ram_usage': 0.0,
            'cpu_count': 0,
            'available_ram': 0.0
        }

def render_system_performance():
    """Render the system performance section"""
    st.markdown(f"""
    <div class="section-header">
        {get_icon_html("chart")} Sistem Performansı
    </div>
    """, unsafe_allow_html=True)
    
    # Use the new performance monitor
    metrics, summary = display_performance_metrics()
    
    # Additional optimization metrics
    if st.session_state.get('optimization_running', False):
        st.markdown("### Optimizasyon Durumu")
        
        # Real-time optimization stats
        col1, col2, col3 = st.columns(3)
        
        with col1:
            current_score = st.session_state.get('current_score', 0)
            st.metric("Mevcut Skor", f"{current_score:.4f}")
        
        with col2:
            best_score = st.session_state.get('best_score', 0)
            st.metric("En İyi Skor", f"{best_score:.4f}")
        
        with col3:
            progress = st.session_state.get('optimization_progress', 0)
            st.metric("İlerleme", f"{progress:.1f}%")
    
    # Performance recommendations
    status = summary.get('status', 'Good').lower()
    if status == 'critical':
        st.error("⚠️ Sistem performansı kritik seviyede. Optimizasyon yavaş çalışabilir.")
    elif status == 'warning':
        st.warning("⚡ Sistem yükü yüksek. Optimizasyon performansı etkilenebilir.")
    else:
        st.success("✅ Sistem performansı optimal düzeyde.")
    
    return metrics
